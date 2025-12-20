#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <libgen.h>
#include <linux/limits.h>
#include "lzo_defaults.h"
#include "timing.h"
#include "lzo_gpu_utils.h"

/* Helper for multi-threaded pread into a destination buffer.
 * Each thread handles a subrange (off,len) of the destination.
 */
typedef struct {
    int fd;
    void *dest; /* base pointer */
    off_t off;  /* offset into dest */
    size_t len; /* length to read */
    int err;    /* errno on failure, 0 on success */
} mt_io_arg_t;

/* forward declaration of now_ns used below by the reaper */
static inline uint64_t now_ns(void);


static void* mt_pread_worker(void *v) {
    mt_io_arg_t *a = (mt_io_arg_t*)v;
    /* validate args first */
    if (!a || a->len == 0) { if (a) a->err = 0; return NULL; }
    if ((uint64_t)a->off + (uint64_t)a->len < (uint64_t)a->off) { /* overflow */
        a->err = EINVAL; return NULL; }
    size_t left = a->len;
    off_t pos = a->off;
    char *p = (char*)a->dest + pos;
    while (left > 0) {
        ssize_t r;
        do {
            r = pread(a->fd, p, left, pos);
        } while (r < 0 && errno == EINTR);
        if (r < 0) { a->err = errno; return NULL; }
        if (r == 0) { /* short read */ a->err = EIO; return NULL; }
        left -= r; p += r; pos += r;
    }
    a->err = 0;
    return NULL;
}

/*
* 压缩文件格式：
uint16  magic     = 0x4C5A   // 'L''Z'
uint32  orig_size               (≤4 GiB)
uint32  blk_size
uint32  nblk
uint32  len[nblk]               // 每块压缩长度
-----   nblk 个压缩块数据
*/
#define MAGIC  0x4C5A   /* 'L''Z' */
#define D_BITS          11
//#define BLK_SIZE        (32 * 1024)
/* Compression ratio tracking */
#define ENABLE_COMPRESSION_RATIO_TRACKING 1

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline uint64_t now_ns(void)
{
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    /*   counter / freq = 秒
     * → counter * 1e9 / freq = 纳秒
     */
    return (uint64_t)counter.QuadPart * (uint64_t)1000000000ULL /
        (uint64_t)freq.QuadPart;
}

#else
#include <time.h>

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

static inline void print_ns(const char* tag, uint64_t ns) {
    unsigned long us = (unsigned long)(ns / 1000ULL);
    /* print_us_tag is provided by timing.h */
    print_us_tag(stdout, tag, us);
}

/* CLI override variables (set by parsing -B/--block-size and --local) */
static size_t g_cli_fixed_block_bytes = 0; /* 0 = not specified */
static size_t g_cli_local_size = 0;       /* 0 = not specified */

#define CHECK(expr)  do{ cl_int _e=(expr);                       \
        if(_e!=CL_SUCCESS){                                      \
            fprintf(stderr,"OpenCL error %d at %s:%d\n",         \
                    _e,__FILE__,__LINE__); exit(1);} }while(0)

static inline size_t lzo_worst(size_t n) {
    return n + n / 16 + 64 + 3;
}

static cl_context  ctx;
static cl_command_queue q;
static cl_device_id dev;
static int debug = 0;  /* 全局debug标志 */

/* 优化: 全局缓存以避免重复编译和创建内核 */
#define MAX_CACHED_PROGRAMS 16
static struct {
    char name[128];
    cl_program prog;
    cl_kernel krn_compress;
    cl_kernel krn_decompress;
} prog_cache[MAX_CACHED_PROGRAMS];
static int prog_cache_count = 0;

/* 优化: 持久化缓冲区缓存以避免重复创建和释放 */
static struct {
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    size_t in_size;
    size_t out_size;
    size_t len_size;
} buffer_cache = {0};

static cl_mem get_or_create_buffer(cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags) {
    if (*cached_size < required_size) {
        if (*cached_buf) clReleaseMemObject(*cached_buf);
        cl_int err;
        /* Use Pinned Host mapping (CL_MEM_ALLOC_HOST_PTR) unconditionally; this
         * restores the original zero-copy semantics which improve upload/download
         * throughput and stability for many workloads. Keeping this flag ensures
         * the mapped pointer returned by clEnqueueMapBuffer is a host-accessible
         * pinned buffer optimized for DMA interactions with the device.
         */
        cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags, required_size, NULL, &err);
        CHECK(err);
        *cached_size = required_size;
    }
    return *cached_buf;
}

static void ocl_init(void)
{
    cl_platform_id pf;
    /* allow selecting CPU device for testing CPU-style kernels via env
     * variable LZO_OPENCL_DEVICE=CPU; default remains GPU. */
    const char* prefer = getenv("LZO_OPENCL_DEVICE");
    cl_device_type dtype = CL_DEVICE_TYPE_GPU;
    if (prefer && strcmp(prefer, "CPU") == 0) dtype = CL_DEVICE_TYPE_CPU;
    CHECK(clGetPlatformIDs(1, &pf, NULL));
    CHECK(clGetDeviceIDs(pf, dtype, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    q = clCreateCommandQueueWithProperties(ctx, dev, props, NULL);
}

void print_buildlog(cl_program program, cl_device_id device) {
    char* buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-1);
    }
    buff_erro = malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr, "Build log: \n%s\n", buff_erro); //Be careful with fprint
    free(buff_erro);
    fprintf(stderr, "clBuildProgram failed\n");
}

/* forward decl so main can call it */
/* static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path); */

/* 优化: 缓存查找和管理函数 */
static int find_cached_program(const char* name) {
    for (int i = 0; i < prog_cache_count; i++) {
        if (strcmp(prog_cache[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void cache_program(const char* name, cl_program prog, cl_kernel krn_c, cl_kernel krn_d) {
    if (prog_cache_count >= MAX_CACHED_PROGRAMS) {
        fprintf(stderr, "warning: program cache full, not caching %s\n", name);
        return;
    }
    strncpy(prog_cache[prog_cache_count].name, name, sizeof(prog_cache[0].name) - 1);
    prog_cache[prog_cache_count].prog = prog;
    prog_cache[prog_cache_count].krn_compress = krn_c;
    prog_cache[prog_cache_count].krn_decompress = krn_d;
    prog_cache_count++;
}

/* Helper: load program from source file with D_BITS macro */
static cl_program load_prog_with_dbits(const char* alg_name, int bits)
{
    cl_int err;
    cl_program prog = NULL;

    /* Try to load precompiled binary first: <alg>_<bits>.clbin or <alg>.clbin */
    {
        char bin_name[64];
        if (bits > 0)
            snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", alg_name, bits);
        else
            snprintf(bin_name, sizeof(bin_name), "%s.clbin", alg_name);

        char resolved_bin[PATH_MAX];

        if (lzo_find_file_path(bin_name, resolved_bin, sizeof(resolved_bin)) == 0) {
            FILE* fb = fopen(resolved_bin, "rb");
            if (fb) {
                fseek(fb, 0, SEEK_END);
                long bsz = ftell(fb);
                fseek(fb, 0, SEEK_SET);
                unsigned char* bin = malloc(bsz);
                if (fread(bin, 1, bsz, fb) == (size_t)bsz) {
                    cl_int binary_status;
                    prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
                        (const unsigned char**)&bin, &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                        err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
                        if (err == CL_SUCCESS) {
                            if (debug) fprintf(stderr, "DBG: loaded precompiled kernel %s\n", bin_name);
                            free(bin);
                            fclose(fb);
                            return prog;
                        }
                    }
                }
                free(bin);
                fclose(fb);
                if (prog) { clReleaseProgram(prog); prog = NULL; }
            }
        }
    }

    /* Fallback to source compilation */
    char cl_src_path[64];
    snprintf(cl_src_path, sizeof(cl_src_path), "%s.cl", alg_name);

    size_t src_len = 0; char* src = NULL;
    /* Find source via executable dir or fallback paths */
    char resolved_src[PATH_MAX];
    if (lzo_find_file_path(cl_src_path, resolved_src, sizeof(resolved_src)) == 0) {
        src = lzo_read_file(resolved_src, &src_len);
    } else {
        fprintf(stderr, "source file %s not found\n", cl_src_path);
        exit(1);
    }

    if (!src) { fprintf(stderr, "source %s not read\n", cl_src_path); exit(1); }

    prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed (err=%d)\n", err); free(src); exit(1); }

    char build_opts[128];
    snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -I./lzo_gpu -I.. -D D_BITS=%d", bits);

    err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
        fprintf(stderr, "Build log (from source):\n%s\n", log); free(log); free(src); exit(1);
    }
    free(src);
    return prog;
}

static inline void show_help(char *prog_name)
{
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s [--debug|-v] [--verify|-c] [-a alg] [-L level] [-o out.lzo] input_file\n", prog_name);
        fprintf(stderr, "     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        fprintf(stderr, "     - --verify/-c (compress mode): do in-memory roundtrip check (no arg).\n");
        fprintf(stderr, "     - -a|--alg ALG        : algorithm (lzo1x, lzo1y) (default: lzo1x)\n");
        fprintf(stderr, "     - -L|-l|--level LEVEL : dictionary size in bits (10-14) (default: 12)\n");
        fprintf(stderr, "     -B N|--block-size N   Fix block size (accepts units: B/KB/MB) \n");
        fprintf(stderr, "     --local N             Set local work-group size for kernels (1,8,64)\n");
        fprintf(stderr, "  %s -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo\n", prog_name);
        fprintf(stderr, "     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        fprintf(stderr, "     - --verify/-c ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  Compress with lzo1y level 12: %s -a lzo1y -L 12 input.dat -o out.lzo\n", prog_name);
        fprintf(stderr, "  Decompress and verify:        %s -d --verify input.dat out.lzo -o out.dec\n", prog_name);
        fprintf(stderr, "  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", prog_name);
        fprintf(stderr, "  %s -h|--help                                 # show this help\n", prog_name);
        fprintf(stderr, "\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
        fprintf(stderr, "    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload).\n");
        fprintf(stderr, "    LZO_MT_IO=0|1            Enable multi-threaded pread reads / parallel uploads.\n");
        fprintf(stderr, "    LZO_MT_IO_THREADS=N      Threads for MT I/O (1-32; common values: 4/8; default: %d ).\n", LZO_DEFAULT_MT_IO_THREADS);
        fprintf(stderr, "    LZO_DEBUG=1              Enable debug prints and timing traces.\n");
        fprintf(stdout, "    LZO_COALESCE_OUTPUT=0|1    Enable output coalescing by default (default: %d)\n", LZO_DEFAULT_COALESCE_OUTPUT);
        fprintf(stdout, "    LZO_COALESCE_CHUNK_MB=N    Chunk size (MB) for chunked coalesce fallback (default: %d)\n", LZO_DEFAULT_COALESCE_CHUNK_MB);
        fprintf(stdout, "    LZO_COALESCE_MAX_MB=N      Maximum MB allowed for full contiguous coalesce (default: %d)\n", LZO_DEFAULT_COALESCE_MAX_MB);
        fprintf(stdout, "    LZO_STDIO_BUF_MB=N         Default stdio buffer size in MB for fwrite() (default: %d)\n", LZO_DEFAULT_STDIO_BUF_MB);
}

int main(int argc, char** argv)
{
    uint64_t t_start_total = now_ns();
    if (argc < 2) {
        /* Print the detailed help when no args are provided to keep output
         * consistent with --help behaviour (avoid the short, outdated usage)
         */
        argc = 1;
        show_help(argv[0]);
        return 0;
    }

    /* simple CLI parsing: support optional --debug/-v flag and -d decompress mode */
    int verify_flag = 0; /* only when set, do roundtrip/verify prints */
    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *orig_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int comp_level = 12;
    const char *alg_name = "lzo1x";

    /* pass 1: only detect mode (-d) and help, to know how to parse verify */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "-d") == 0) { decompress_mode = 1; }
    }

    /* pass 2: parse options and positionals with knowledge of mode */
    const char* verify_path = NULL; /* only for -d mode */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--debug") == 0 || strcmp(arg, "-v") == 0) { debug = 1; continue; }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "-l") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            comp_level = atoi(argv[++i]);
            continue;
        }
        if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            alg_name = argv[++i];
            if (strcmp(alg_name, "1x") == 0) alg_name = "lzo1x";
            else if (strcmp(alg_name, "1y") == 0) alg_name = "lzo1y";
            else { fprintf(stderr, "错误: 未知算法 '%s'. 支持: 1x, 1y\n", alg_name); return 1; }
            continue;
        }
        if (strcmp(arg, "-c") == 0 || strcmp(arg, "--verify") == 0) {
            if (decompress_mode) {
                if (i + 1 >= argc || argv[i+1][0] == '-') { fprintf(stderr, "--verify requires a reference file in -d mode\n"); return 1; }
                verify_path = argv[++i];
            } else {
                verify_flag = 1; /* compress-mode in-memory roundtrip */
            }
            continue;
        }
        if (strcmp(arg, "-d") == 0) { /* already noted */ continue; }
        if (strncmp(arg, "-B=", 3) == 0) { /* short shorthand -B=value */
            size_t b = lzo_parse_block_size(arg + 3);
            if (b > 0) g_cli_fixed_block_bytes = b;
            continue;
        }
        if (strcmp(arg, "-B") == 0) { if (i + 1 < argc) { size_t b = lzo_parse_block_size(argv[++i]); if (b > 0) g_cli_fixed_block_bytes = b; } continue; }
        if (strncmp(arg, "--block-size=", 13) == 0) { size_t b = lzo_parse_block_size(arg + 13); if (b > 0) g_cli_fixed_block_bytes = b; continue; }
        if (strcmp(arg, "--block-size") == 0) { if (i + 1 < argc) { size_t b = lzo_parse_block_size(argv[++i]); if (b > 0) g_cli_fixed_block_bytes = b; } continue; }
        if (strncmp(arg, "--local=", 8) == 0) { g_cli_local_size = (size_t)atoi(arg + 8); if (g_cli_local_size == 0) g_cli_local_size = 0; continue; }
        if (strcmp(arg, "--local") == 0) { if (i + 1 < argc) { g_cli_local_size = (size_t)atoi(argv[++i]); if (g_cli_local_size == 0) g_cli_local_size = 0; } continue; }
        /* positional */
        if (arg[0] != '-') {
            if (decompress_mode) {
                if (!lz_path) { lz_path = arg; continue; }
                /* ignore extra positionals; verify file should come via --verify */
            } else {
                if (!in_path) { in_path = arg; continue; }
            }
        }
    }

    /* After parsing CLI, establish effective values for overridden options:
     * Precedence: CLI option (opt_*) -> environment variable -> defaults from lzo_defaults.h
     */
    const char* sopt = NULL;
    size_t stdio_buf_mb_local = (size_t)LZO_DEFAULT_STDIO_BUF_MB;
    if ((sopt = getenv("LZO_STDIO_BUF_MB"))) {
        int v = atoi(sopt); if (v > 0) stdio_buf_mb_local = (size_t)v;
    }

    int coalesce_local = LZO_DEFAULT_COALESCE_OUTPUT;
    if ((sopt = getenv("LZO_COALESCE_OUTPUT"))) {
        coalesce_local = atoi(sopt) ? 1 : 0;
    }

    size_t coalesce_max_mb_local = (size_t)LZO_DEFAULT_COALESCE_MAX_MB;
    if ((sopt = getenv("LZO_COALESCE_MAX_MB"))) {
        int v = atoi(sopt); if (v > 0) coalesce_max_mb_local = (size_t)v;
    }

    size_t coalesce_chunk_mb_local = (size_t)LZO_DEFAULT_COALESCE_CHUNK_MB;
    if ((sopt = getenv("LZO_COALESCE_CHUNK_MB"))) {
        int v = atoi(sopt); if (v > 0) coalesce_chunk_mb_local = (size_t)v;
    }

    /* Decompress mode */
    if (decompress_mode) {
        if (!lz_path) { fprintf(stderr, "no input .lzo specified (after -d)\n"); return 1; }
    uint64_t t_io_in = now_ns();
    size_t lz_sz; unsigned char* lz_buf = (unsigned char*)lzo_read_file(lz_path, &lz_sz);
    unsigned char* ref = NULL;
    size_t ref_sz = 0;
        unsigned char* p = lz_buf;
        uint16_t magic = *(uint16_t*)p; p += 2;
        if (magic != MAGIC) { fprintf(stderr, "bad magic\n"); return 1; }
        uint32_t orig_sz = *(uint32_t*)p; p += 4;
        uint32_t blk_sz = *(uint32_t*)p; p += 4;
        uint32_t nblk = *(uint32_t*)p; p += 4;
        uint32_t alg_id = *(uint32_t*)p; p += 4;
        if (alg_id == 0) alg_name = "lzo1x";
        else if (alg_id == 1) alg_name = "lzo1y";
        else { fprintf(stderr, "unknown alg_id %u\n", alg_id); return 1; }
        uint32_t* len_arr = (uint32_t*)p; p += 4 * nblk;
        size_t comp_sz = lz_sz - (p - lz_buf);

        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (uint32_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

    uint64_t t_io_after = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
    uint64_t t_kernel_load_start = 0, t_kernel_load_end = 0;
        /* Determine decompression kernel based on algorithm */
        char decomp_base[64];
        snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);

        /* 优化: 检查缓存以避免重复编译和创建内核 */
        cl_program prog_d = NULL;
        cl_kernel krn_d = NULL;
        t_kernel_load_start = now_ns();
        int cache_idx_d = find_cached_program(decomp_base);

        if (cache_idx_d >= 0) {
            /* 使用缓存的程序和内核 */
            prog_d = prog_cache[cache_idx_d].prog;
            krn_d = prog_cache[cache_idx_d].krn_decompress;
            if (debug) fprintf(stderr, "DBG: using cached decompress program/kernel for %s\n", decomp_base);
        } else {
            /* 首次加载: 编译并缓存 */
            if (debug) fprintf(stderr, "DBG: loading and caching decompress program %s\n", decomp_base);

            /* Use load_prog_with_dbits (bits=0, macro ignored by decomp usually) */
            prog_d = load_prog_with_dbits(decomp_base, 0);

            cl_int err;
            char kernel_name[64];
            snprintf(kernel_name, sizeof(kernel_name), "%s_block_decompress", alg_name);
            krn_d = clCreateKernel(prog_d, kernel_name, &err);
            CHECK(err);

            /* 缓存程序和内核供后续使用 */
            cache_program(decomp_base, prog_d, NULL, krn_d);
        }
        t_kernel_load_end = now_ns();


        cl_int err;

        /* 优化: 使用 pinned memory (CL_MEM_ALLOC_HOST_PTR) 以支持零拷贝 */
        uint64_t t_buffer_start = now_ns();
        /* Restore pinned-memory usage for direct buffers too */
        cl_mem_flags f_comp = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        cl_mem d_comp = clCreateBuffer(ctx, f_comp, comp_sz, NULL, &err);
    CHECK(err);
    /* 上传压缩数据 */
    void* mapped_comp = clEnqueueMapBuffer(q, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_comp, p, comp_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_comp, mapped_comp, 0, NULL, NULL));

    cl_mem_flags f_off = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
    cl_mem d_off = clCreateBuffer(ctx, f_off, (nblk + 1) * sizeof(cl_uint), NULL, &err);
    CHECK(err);
    void* mapped_off = clEnqueueMapBuffer(q, d_off, CL_TRUE, CL_MAP_WRITE, 0, (nblk + 1) * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_off, off_arr, (nblk + 1) * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(q, d_off, mapped_off, 0, NULL, NULL));

    cl_mem_flags f_out = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
    cl_mem d_out2 = clCreateBuffer(ctx, f_out, orig_sz, NULL, &err);
    CHECK(err);
    /* decompressor expects an out_lens buffer as arg 3 */
    cl_mem d_out_lens = clCreateBuffer(ctx, f_out, nblk * sizeof(cl_uint), NULL, &err);
    CHECK(err);
    uint64_t t_buffer_end = now_ns();

    uint64_t t_upload_start = now_ns();
    CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
    CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));
    uint64_t t_upload_end = now_ns();

    /* 解压优化: local_size=1 最优 (每个work-item独立处理一个块)
     * 测试结果: local_size=1 比 local_size=8 快 ~180% (3050 vs 1091 MB/s)
     * 原因: 解压是独立任务，无需work-group协作，local_size=1避免同步开销 */
    /* Prefer CLI override if provided; else fallback to env var if set */
    size_t lsz = (g_cli_local_size > 0) ? g_cli_local_size : 1;  /* default 1 */
    if (g_cli_local_size > 0) {
        if (debug) fprintf(stderr, "DBG: using local_size=%zu from CLI (--local)\n", lsz);
    }
    size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;  /* round up to multiple of lsz */
    cl_event evt_decomp;
    uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt_decomp));
    uint64_t t_kernel_end = now_ns();

    /* 异步下载：启动非阻塞读取，然后立即返回（不等待） */
    uint64_t t_read_start = now_ns();
    cl_event evt_read;
    unsigned char* out2 = malloc(orig_sz);
    /* 先启动异步下载（不阻塞主线程） */
    CHECK(clEnqueueReadBuffer(q, d_out2, CL_FALSE, 0, orig_sz, out2, 1, &evt_decomp, &evt_read));

    /* 等待kernel和读取完成 */
    clWaitForEvents(1, &evt_decomp);
    uint64_t t_exec_end = now_ns();
    clWaitForEvents(1, &evt_read);
    uint64_t t_read_end = now_ns();
    clReleaseEvent(evt_read);

    /* perform verify first if requested; on failure do not write and exit non-zero */
    if (verify_path) {
        /* no debug: keep verify path clean */
        size_t ref_sz; unsigned char* ref = (unsigned char*)lzo_read_file(verify_path, &ref_sz);
        if (ref_sz != orig_sz || memcmp(ref, out2, orig_sz) != 0) {
            fprintf(stderr, "decompress verify FAILED!\n");
            for (size_t i = 0; i < orig_sz; ++i) {
                if (ref[i] != out2[i]) { fprintf(stderr, "first_mismatch_offset=%zu (ref=0x%02x out=0x%02x)\n", i, ref[i], out2[i]); break; }
            }
            free(ref);
            /* cleanup and exit */
            clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
            clReleaseKernel(krn_d); clReleaseProgram(prog_d);
            clReleaseCommandQueue(q); clReleaseContext(ctx);
            free(lz_buf); free(off_arr); free(out2);
            return 1;
        }
        free(ref);
        if (!suppress_non_data) puts("verify OK");
    }

    /* decide whether to write output:
       - If user requested --verify (decompress mode) and did not explicitly pass -o,
         do NOT write the decompressed file (only perform in-memory verification).
       - If user explicitly passed -o, honor it and write output as requested. */
    uint64_t t_write_start = now_ns();
    if (verify_path && !output_explicit) {
        /* skip writing decompressed output when verify requested without -o */
        if (!suppress_non_data) puts("verify mode: not writing decompressed output (no -o given)");
    } else {
        /* compute default output path if not explicitly provided */
        if (!output_path) {
            const char* in = lz_path;
            size_t L = strlen(in);
            int ends_lzo = (L >= 4 && strcmp(in + L - 4, ".lzo") == 0);
            size_t outL = ends_lzo ? (L - 4) : (L + 4);
            char* def = (char*)malloc(outL + 1);
            if (ends_lzo) { memcpy(def, in, L - 4); def[L - 4] = '\0'; }
            else { memcpy(def, in, L); memcpy(def + L, ".raw", 4); def[L + 4] = '\0'; }
            output_path = def;
        }

        if (output_path) {
            if (strcmp(output_path, "-") == 0) {
                /* write raw data to stdout (only data) */
                if (fwrite(out2, 1, orig_sz, stdout) != orig_sz) { perror("stdout write"); }
                fflush(stdout);
            } else {
                if (verify_path && strcmp(output_path, verify_path) == 0) {
                    fprintf(stderr, "refusing to write output to the same path as --verify reference: %s\n", output_path);
                    return 1;
                }
                FILE* fo = fopen(output_path, "wb");
                if (!fo) { perror(output_path); return 1; }

                /* 对大文件使用直接 write() 以避免 stdio 缓冲开销 */
                int fd = fileno(fo);
                if (fd >= 0 && orig_sz > 100 * 1024 * 1024) {
                    /* 大文件: 使用系统 write() 直接写入 (不经过 stdio) */
                    size_t write_chunk = 256 * 1024 * 1024;  /* 256MB chunks for fast sequential write */
                    for (size_t offset = 0; offset < orig_sz; offset += write_chunk) {
                        size_t to_write = (orig_sz - offset < write_chunk) ? (orig_sz - offset) : write_chunk;
                        ssize_t written = write(fd, out2 + offset, to_write);
                        if ((size_t)written != to_write) {
                            perror("write"); fclose(fo); return 1;
                        }
                    }
                } else {
                    /* 小文件或无法获取 fd: 使用 stdio */
                    char *vbuf = NULL;
                    size_t vsize = (size_t)LZO_DEFAULT_STDIO_BUF_MB * 1024 * 1024;
                    if (orig_sz > 0) {
                        if (vsize > orig_sz) vsize = orig_sz;
                        vbuf = (char*)malloc(vsize);
                        if (vbuf && setvbuf(fo, vbuf, _IOFBF, (int)vsize) != 0) {
                            free(vbuf); vbuf = NULL;
                        }
                    }
                    size_t written = fwrite(out2, 1, orig_sz, fo);
                    fflush(fo);  /* 确保数据写入 */
                    if (written != orig_sz) {
                        perror("fwrite"); if (vbuf) free(vbuf); fclose(fo); return 1;
                    }
                    if (vbuf) free(vbuf);
                }

                fclose(fo);
                if (!suppress_non_data) printf("wrote %s\n", output_path);
            }
        }
    }
    uint64_t t_write_end = now_ns();

        uint64_t t_total_end = now_ns();
        double ms_file_read = (t_io_after - t_io_in)/1e6;
        double ms_ocl_init = (t_ocl_init - t_io_after)/1e6;
        double ms_kernel_load = (t_kernel_load_end - t_kernel_load_start)/1e6;
        double ms_buffer = (t_buffer_end - t_buffer_start)/1e6;
        double ms_upload = (t_upload_end - t_upload_start)/1e6;
        double ms_kernel = (t_exec_end - t_exec_start)/1e6;  /* kernel执行时间 */
        double ms_download = (t_read_end - t_exec_end)/1e6;  /* kernel完成后的数据传输时间 */
        double ms_write = (t_write_end - t_write_start)/1e6;
        double ms_total = (t_total_end - t_start_total)/1e6;
        double ratio = lz_sz > 0 ? (double)orig_sz / (double)lz_sz : 0.0;
        double thrpt = ms_kernel > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;

#if ENABLE_COMPRESSION_RATIO_TRACKING
        /* 输出解压缩统计信息 */
        printf("\n=== Decompression Statistics ===\n");
        printf("Compressed size    : %zu bytes (%.2f MB)\n", (size_t)lz_sz, lz_sz / (1024.0 * 1024.0));
        printf("Output size        : %zu bytes (%.2f MB)\n", (size_t)orig_sz, orig_sz / (1024.0 * 1024.0));
        printf("Block size (blocks):%u bytes/%u KB (%u)\n", blk_sz, blk_sz / 1024, nblk);
        printf("Kernel           : %s\n",
               decomp_base);
        printf("Work groups      : global=%zu, local=%zu\n", gsz, lsz);
        printf("Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n",
               ((double)orig_sz / (1024.0*1024.0)) / (ms_total/1000.0),
               thrpt);
        printf("==============================\n\n");
#endif

        /* 打印详细的时间分解 */
        printf("\n=== Time Breakdown (Decompression) ===\n");
        print_ns("1. File Read", t_io_after - t_io_in);
        print_ns("2. OCL Init", t_ocl_init - t_io_after);
        print_ns("3. Kernel Load", t_kernel_load_end - t_kernel_load_start);
        print_ns("4. Buffer Alloc", t_buffer_end - t_buffer_start);
        print_ns("5. Data Upload", t_upload_end - t_upload_start);
        print_ns("6. Kernel Exec", t_exec_end - t_exec_start);
        print_ns("7. Data Download", t_read_end - t_exec_end);
        print_ns("8. File Write", t_write_end - t_write_start);
        print_ns("TOTAL", t_total_end - t_start_total);
        printf("\n");

         /* 计算占比 */
         printf("=== Percentage Breakdown ===\n");
         /* protect against zero total */
         double denom = (ms_total > 0.0) ? ms_total : 1.0;
         int zero_total = (ms_total <= 0.0);

         printf("Kernel Exec     : %6.2f%%\n", zero_total ? 0.0 : 100.0 * ms_kernel / denom);
         printf("Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
             zero_total ? 0.0 : 100.0 * (ms_upload + ms_download) / denom,
             zero_total ? 0.0 : 100.0 * ms_upload / denom,
             zero_total ? 0.0 : 100.0 * ms_download / denom);
         printf("File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
             zero_total ? 0.0 : 100.0 * (ms_file_read + ms_write) / denom,
             zero_total ? 0.0 : 100.0 * ms_file_read / denom,
             zero_total ? 0.0 : 100.0 * ms_write / denom);
         printf("Buffer Alloc    : %6.2f%%\n",
             zero_total ? 0.0 : 100.0 * ms_buffer / denom);
           printf("OCL Setup       : %6.2f%%\n",
               zero_total ? 0.0 : 100.0 * (ms_ocl_init + ms_kernel_load) / denom);
        printf("\n");

    clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        free(lz_buf); free(off_arr); free(out2);
        if (orig_path) free(ref);
        return 0;
    }

    /* Compress path (simple, fast) */
    if (!in_path) { fprintf(stderr, "no input file specified for compression\n"); return 1; }
    uint64_t t_compress_start = now_ns();

    /* Phase 8.3: 改用Daemon方式 - 直接fread到Pinned Memory，消除中间buffer */
    /* 先获取文件大小 */
    FILE* f_in = fopen(in_path, "rb");
    if (!f_in) {
        perror(in_path);
        return 1;
    }
    fseek(f_in, 0, SEEK_END);
    size_t in_sz = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    uint64_t t_after_fopen = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
    /* select compression kernel variant based on comp_level */
    /* Use standalone kernel with dynamic D_BITS */

    /* 优化: 检查缓存以避免重复编译和创建内核 */
    uint64_t t_kernel_load_start = now_ns();
    cl_int err;
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;

    char cache_key[64];
    snprintf(cache_key, sizeof(cache_key), "%s_%d", alg_name, comp_level);
    int cache_idx = find_cached_program(cache_key);

    if (cache_idx >= 0) {
        /* 使用缓存的程序和内核 */
        prog_c = prog_cache[cache_idx].prog;
        krn_c = prog_cache[cache_idx].krn_compress;
        if (debug) fprintf(stderr, "DBG: using cached program/kernel for %s\n", cache_key);
    } else {
        /* 首次加载: 编译并缓存 */
        if (debug) fprintf(stderr, "DBG: loading and caching program %s\n", cache_key);

        prog_c = load_prog_with_dbits(alg_name, comp_level);

        /* select kernel function name according to the kernel_base we loaded */
        char krn_name[64];
        snprintf(krn_name, sizeof(krn_name), "%s_block_compress", alg_name);
        krn_c = clCreateKernel(prog_c, krn_name, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); exit(1);
        }

        /* 缓存程序和内核供后续使用 */
        cache_program(cache_key, prog_c, krn_c, NULL);
    }
    uint64_t t_kernel_load_end = now_ns();

    /* Phase 8.3: File Read — 支持两种复制模式 (zero-copy / standard-copy)
     * 默认: zero-copy (map + fread -> device-accessible pinned memory)
     * 若环境变量 LZO_STANDARD_COPY=1 则使用 standard copy:
     *   fread -> host buffer (posix_memalign) ; clEnqueueWriteBuffer -> device
     * 这可帮助在不同平台（iGPU vs dGPU）对比两种传输路径的性能。
     */
    int standard_copy = 0;
    const char* std_s = getenv("LZO_STANDARD_COPY");
    if (std_s && atoi(std_s) == 1) standard_copy = 1;

    /* multi-threaded I/O control */
    int mt_io = 0;
    int mt_threads = 0;
    const char* mt_s = getenv("LZO_MT_IO");
    if (mt_s && atoi(mt_s) == 1) {
        mt_io = 1;
        const char* mt_threads_s = getenv("LZO_MT_IO_THREADS");
        mt_threads = mt_threads_s ? atoi(mt_threads_s) : 4;
        if (mt_threads < 1) mt_threads = 1;
        if (mt_threads > 32) mt_threads = 32;
    }

    uint64_t t_io_read_start = now_ns();
    cl_mem d_in = NULL;
    unsigned long upload_us = 0;
    unsigned long buffer_in_us = 0;
    uint64_t t_io_read_done = 0;
    void* mapped_in = NULL;
    unsigned char* host_in = NULL; /* used only in standard-copy */
    const unsigned char* data_for_blocking = NULL;

    if (standard_copy) {
        if (debug) fprintf(stderr, "DBG: using STANDARD copy path (LZO_STANDARD_COPY=1)\n");
        /* read into host buffer first */
        uint64_t t_file_read_start = now_ns();
        /* reuse outer host_in variable (do not shadow) */
        host_in = NULL;
        /* Align host buffer for DMA/pinned-friendly boundaries */
        if (posix_memalign((void**)&host_in, ALIGN_BYTES, in_sz) != 0) {
            host_in = malloc(in_sz);
            if (!host_in) { perror("malloc"); fclose(f_in); return 1; }
        }
        /* multi-threaded read into host_in (pread) if enabled */
        if (mt_io && mt_threads > 1) {
            int fd = fileno(f_in);
            /* only spawn as many workers as needed (skip zero-length tasks) */
            pthread_t *tids = calloc(mt_threads, sizeof(pthread_t));
            mt_io_arg_t *args = calloc(mt_threads, sizeof(mt_io_arg_t));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t cur = 0;
            int failed = 0;
            int created_count = 0;
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                /* skip creating a thread with no work */
                if (len == 0) { args[i].len = 0; args[i].err = 0; continue; }
                args[i].fd = fd;
                args[i].dest = host_in;
                args[i].off = cur;
                args[i].len = len;
                args[i].err = 0;
                    int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                    if (rc != 0) {
                        /* failed to create thread — mark error and break to fallback */
                        args[i].err = rc;
                        failed = 1; break;
                    }
                    created_count++;
                cur += len;
            }
            for (int i = 0; i < mt_threads; ++i) {
                if (args[i].len == 0) continue; /* no thread created for this slot */
                pthread_join(tids[i], NULL);
                if (args[i].err) failed = 1;
            }
            free(tids); free(args);
            if (failed) {
                if (fseek(f_in, 0, SEEK_SET) != 0) { perror("fseek"); free(host_in); fclose(f_in); return 1; }
                if (fread(host_in, 1, in_sz, f_in) != in_sz) { perror("fread"); free(host_in); fclose(f_in); return 1; }
            }
        } else {
            if (fread(host_in, 1, in_sz, f_in) != in_sz) { perror("fread"); free(host_in); fclose(f_in); return 1; }
        }
        fclose(f_in);
        uint64_t t_file_read_end = now_ns();
        t_io_read_done = t_file_read_end;
        /* keep host_in for blocking calc and upload later */
        data_for_blocking = (const unsigned char*)host_in;
    } else {
        if (debug) fprintf(stderr, "DBG: using ZERO-COPY path (default)\n");
        /* zero-copy: create pinned device buffer then map and fread into it */
        if (debug) fprintf(stderr, "DBG: getting cached d_in size=%zu (pinned)\n", in_sz);
        uint64_t t_buf_in_start = now_ns();
        d_in = get_or_create_buffer(&buffer_cache.d_in, &buffer_cache.in_size,
                                    in_sz, CL_MEM_READ_ONLY);
        uint64_t t_buf_in_end = now_ns();
        /* mapping counted as part of File Read in zero-copy path */
        mapped_in = clEnqueueMapBuffer(q, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz,
                             0, NULL, NULL, &err);
        CHECK(err);

        /* Support multi-threaded pread into mapped device-accessible memory */
        if (mt_io && mt_threads > 1) {
            int fd = fileno(f_in);
            pthread_t *tids = calloc(mt_threads, sizeof(pthread_t));
            mt_io_arg_t *args = calloc(mt_threads, sizeof(mt_io_arg_t));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t cur = 0;
            int failed = 0;
            int created_count2 = 0;
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                /* skip zero-length chunks to avoid creating useless threads */
                if (len == 0) { args[i].len = 0; args[i].err = 0; continue; }
                args[i].fd = fd;
                args[i].dest = mapped_in;
                args[i].off = cur;
                args[i].len = len;
                args[i].err = 0;
                int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                if (rc != 0) {
                    args[i].err = rc; /* mark failure for fallback */
                    failed = 1;
                    break;
                }
                created_count2++;
                cur += len;
            }
            for (int i = 0; i < mt_threads; ++i) {
                if (args[i].len == 0) continue;
                pthread_join(tids[i], NULL);
                if (args[i].err) failed = 1;
            }
            free(tids); free(args);
            if (failed) {
                if (fseek(f_in, 0, SEEK_SET) != 0) { perror("fseek"); clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL); fclose(f_in); return 1; }
                if (fread(mapped_in, 1, in_sz, f_in) != in_sz) { perror("fread"); clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL); fclose(f_in); return 1; }
            }
        } else {
            if (fread(mapped_in, 1, in_sz, f_in) != in_sz) {
                perror("fread");
                clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL);
                fclose(f_in);
                return 1;
            }
        }
        fclose(f_in);  /* 文件读取完成，关闭 */
        uint64_t t_map_fread_end = now_ns();
        /* in zero-copy, buffer alloc was effectively part of file read */
        t_io_read_done = t_map_fread_end;
        buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000; /* small */
        data_for_blocking = (const unsigned char*)mapped_in;
        upload_us = 0;

        /* leave mapped_in accessible for later use (we will unmap after blocking calc) */
        /* store mapped_in to a temporary symbol name used below */
        /* make sure mapped_in is available in the following block; reuse variable name */
        /* We'll keep mapped_in but no-op in other path */
        /* map pointer is kept in the local scope below where used for blocking */
    }

    /* Phase 7.2: 使用自适应块大小 (基于数据熵)
     * 此时数据已在mapped_in中，可用于熵计算
     */
    uint64_t t_blocking_start = now_ns();
    /* timestamps for allocation/upload phases — declared early so the
     * standard-copy branch can write into them; defaults assigned below
     * after blocking is recorded.
     */
    uint64_t t_buffer_alloc_start = 0, t_buffer_alloc_end = 0;
    uint64_t t_upload_start = 0, t_upload_end = 0;
    /* async upload support removed: always perform synchronous uploads */
    size_t blk = 0, nblk = 0;
    lzo_choose_blocking_adaptive(data_for_blocking, in_sz, dev, g_cli_fixed_block_bytes, &blk, &nblk, debug);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    if (debug) {
        fprintf(stderr, "DBG: choose_blocking -> in_sz=%zu blk=%zu nblk=%zu worst_blk=%zu out_cap=%zu\n",
                in_sz, blk, nblk, worst_blk, out_cap);
    }

    /* Blocking calc finished: record end now so we don't include subsequent
     * buffer allocation or upload timings in the "Blocking Calc" measurement.
     */
    uint64_t t_blocking_end = now_ns();

    /* Unmap输入缓冲区 (数据已就绪且熵计算完成) */
    if (!standard_copy) {
        CHECK(clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL));
    }
    else {

        /* Standard copy: now upload host_in into d_in (device buffer)
         * Record allocations and upload using the outer timestamps so the
         * final printed breakdown includes these times separately from
         * the blocking calculation.
         */
        t_buffer_alloc_start = now_ns();
        d_in = get_or_create_buffer(&buffer_cache.d_in, &buffer_cache.in_size,
                                    in_sz, CL_MEM_READ_ONLY);
        t_buffer_alloc_end = now_ns();

        if (mt_io && mt_threads > 1) {
            /* parallel non-blocking writes + wait */
            cl_event *events = calloc(mt_threads, sizeof(cl_event));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t off = 0;
            t_upload_start = now_ns();
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                err = clEnqueueWriteBuffer(q, d_in, CL_FALSE, off, len, (char*)host_in + off, 0, NULL, &events[i]);
                CHECK(err);
                off += len;
            }
            /* synchronous completion: wait for all writes, then release events and free host buffer */
            CHECK(clWaitForEvents(mt_threads, events));
            t_upload_end = now_ns();
            for (int i = 0; i < mt_threads; ++i) clReleaseEvent(events[i]);
            free(events);
            free(host_in); host_in = NULL;
        } else {
            t_upload_start = now_ns();
            err = clEnqueueWriteBuffer(q, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
            CHECK(err);
            t_upload_end = now_ns();
            free(host_in); host_in = NULL;
        }
    }
    /* NOTE: t_blocking_end already recorded above; any upload / buffer alloc
     * timings are measured separately below and will not be attributed to
     * the blocking calculation.
     */

    /* Data Upload / Buffer alloc timestamps: default to zero-size intervals
     * for zero-copy path (overwrite if standard-copy branch set them).
     */
    if (t_upload_start == 0) t_upload_start = t_blocking_end;
    if (t_upload_end == 0) t_upload_end = t_blocking_end;

    /* Buffer Alloc (in) default to zero (already included in File Read for
     * zero-copy), will be overwritten by standard-copy branch above.
     */
    if (t_buffer_alloc_start == 0) t_buffer_alloc_start = t_io_read_start;
    if (t_buffer_alloc_end == 0) t_buffer_alloc_end = t_io_read_start;

    /* 创建输出缓冲区 (Pinned Memory) */
    uint64_t t_out_buffer_start = now_ns();
    if (debug) fprintf(stderr, "DBG: getting cached d_out size=%zu (pinned)\n", out_cap);
    cl_mem d_out = get_or_create_buffer(&buffer_cache.d_out, &buffer_cache.out_size,
                                        out_cap, CL_MEM_WRITE_ONLY);
    uint64_t t_out_buffer_end = now_ns();


    /* 优化: 使用缓冲区缓存 */
    uint64_t t_len_buffer_start = now_ns();
    size_t len_bytes = nblk * sizeof(cl_uint);
    if (debug) fprintf(stderr, "DBG: getting cached d_len size=%zu\n", len_bytes);
    cl_mem d_len = get_or_create_buffer(&buffer_cache.d_len, &buffer_cache.len_size,
                                        len_bytes, CL_MEM_READ_WRITE);  /* 优化:移除ALLOC_HOST_PTR */
    uint64_t t_len_buffer_end = now_ns();

    /* Set kernel args each run */
    uint64_t t_setup_args_start = now_ns();
    CHECK(clSetKernelArg(krn_c, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(krn_c, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(krn_c, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(krn_c, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(krn_c, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(krn_c, 5, sizeof(cl_uint), &worst_blk));
    uint64_t t_setup_args_end = now_ns();

    /* 压缩: 必须使用local_size=1 (每个work-item需2KB字典，local_size>1会内存溢出)
     * 测试显示local_size=8时性能暴跌94%，因为字典溢出到全局内存 */
    size_t lsz = 1;  /* 压缩必须为1，不可修改 */
    if (g_cli_local_size > 0 && g_cli_local_size != 1) {
        fprintf(stderr, "WARN: CLI --local=%zu requested for compression but compression kernels require local=1; ignoring and forcing to 1.\n", g_cli_local_size);
    }
    /* Ignore LZO_LOCAL_SIZE for compression; always use local_size=1 */
    lsz = 1; /* compression always uses local_size=1 */
    size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;  /* round up to multiple of lsz */
    cl_event evt_compute;
    uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_c, 1, NULL, &gsz, &lsz, 0, NULL, &evt_compute));
    clWaitForEvents(1, &evt_compute);
    uint64_t t_exec_end = now_ns();

    /* 优化: 使用map读取长度数组(零拷贝) */
    uint64_t t_download_start = now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    uint64_t t_len_read_start = now_ns();
    void* mapped_len = clEnqueueMapBuffer(q, d_len, CL_TRUE, CL_MAP_READ, 0, len_bytes,
                                          0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, len_bytes);
    CHECK(clEnqueueUnmapMemObject(q, d_len, mapped_len, 0, NULL, NULL));
    uint64_t t_len_read_end = now_ns();

    if (debug) {
        fprintf(stderr, "Per-block compressed lengths (nblk=%zu):\n", nblk);
        for (size_t i = 0; i < nblk; ++i) {
            fprintf(stderr, "  block %4zu : %u\n", i, len_arr[i]);
        }
    }

    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += len_arr[i];
    unsigned char* out_buf = NULL;
    size_t host_off = 0;
    /* 优化: 使用map读取输出缓冲区(零拷贝) */
    uint64_t t_bulk_read_start = now_ns();
    if (debug) fprintf(stderr, "DBG: about to map d_out size=%zu\n", out_cap);
    unsigned char* dev_out = (unsigned char*)clEnqueueMapBuffer(q, d_out, CL_TRUE, CL_MAP_READ,
                                                                  0, out_cap, 0, NULL, NULL, &err);
    CHECK(err);
    if (debug) fprintf(stderr, "DBG: map completed\n");
    uint64_t t_bulk_read_end = now_ns();
    /* debug: dump first 32 bytes of first block to help diagnose visibility */
    if (debug) {
        fprintf(stderr, "dev_out[0..31]:");
        for (size_t i = 0; i < 32 && i < out_cap; ++i) fprintf(stderr, " %02x", dev_out[i]);
        fprintf(stderr, "\n");
    }

    /* If kernel didn't populate `out_len` (all zeros), try reconstructing per-block
     * lengths from the device output buffer: we expect the kernel to write a little-endian
     * 32-bit length at the start of each block region as a robust fallback. */
    if (out_sz == 0) {
        /* Attempt to recover per-block lengths from device output buffer.
         * Guard against interpreting arbitrary bytes as huge lengths which
         * can lead to oversized allocations and crashes. Accept a length
         * only if it is non-zero and reasonably bounded by `worst_blk` and
         * `out_cap`. Accumulate into a temporary size and check overflow. */
        size_t tmp_out_sz = 0;
        for (size_t i = 0; i < nblk; ++i) {
            size_t dev_off = i * worst_blk;
            if (dev_off + 4 <= out_cap) {
                uint32_t v = (uint32_t)dev_out[dev_off + 0]
                           | ((uint32_t)dev_out[dev_off + 1] << 8)
                           | ((uint32_t)dev_out[dev_off + 2] << 16)
                           | ((uint32_t)dev_out[dev_off + 3] << 24);
                /* sanity checks */
                if (v == 0 || v > worst_blk || v > out_cap) {
                    len_arr[i] = 0;
                } else {
                    /* check overflow before adding */
                    if (tmp_out_sz + (size_t)v < tmp_out_sz) {
                        len_arr[i] = 0;
                    } else {
                        len_arr[i] = v;
                        tmp_out_sz += (size_t)v;
                    }
                }
            } else {
                len_arr[i] = 0;
            }
        }
        out_sz = tmp_out_sz;
        if (out_sz == 0) {
            fprintf(stderr, "ERR: failed to recover per-block lengths from device output; aborting\n");
            free(dev_out);
            free(len_arr);
            /* cleanup and exit with error */
            clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
            clReleaseKernel(krn_c); clReleaseProgram(prog_c);
            clReleaseCommandQueue(q); clReleaseContext(ctx);
            /* Phase 8.3: in_buf不再使用(直接fread到mapped) */
            return 1;
        }
    }

    if (out_sz > out_cap) {
        fprintf(stderr, "ERR: computed total output size (%zu) exceeds device capacity (%zu); aborting\n", out_sz, out_cap);
        free(dev_out);
        free(len_arr);
        clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
        clReleaseKernel(krn_c); clReleaseProgram(prog_c);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        return 1;
    }
    /* Zero-Copy 优化 - 移除中间的 memcpy 打包，直接使用 mapped memory */
    /* 不再分配 out_buf，保持 dev_out mapped，稍后直接 scatter-gather 写入 */
    out_buf = NULL;  /* 标记为未使用 */
    /* 优化: 保持 dev_out mapped，等写入文件后再 unmap */
    uint64_t t_download_end = now_ns();

    /* decide output path if not specified: default to input_file.lzo */
    uint64_t t_write_start = now_ns();
    if (!output_path) {
        size_t L = strlen(in_path); char* def = (char*)malloc(L + 4 + 1);
        memcpy(def, in_path, L); memcpy(def + L, ".lzo", 4); def[L + 4] = '\0';
        output_path = def;
    }
    /* write LZO container: magic, orig_size, blk_size, nblk, len[nblk], then data */
    FILE* fo = fopen(output_path, "wb");
    if (!fo) { perror(output_path); return 1; }
    /* 使用 stdio 缓冲区减少频繁 fwrite 对写入时延造成的波动.
     * 标准默认: stdio buffer 为 LZO_DEFAULT_STDIO_BUF_MB=4MB (配置可通过 CLI 覆盖),
     * 请勿通过环境变量配置此值;如果仍希望微调,可通过命令行参数进行覆盖(高级特性)。 */
    char *vbuf = NULL;
    size_t vsize = (size_t)LZO_DEFAULT_STDIO_BUF_MB * 1024 * 1024;
    if (vsize > out_sz && out_sz > 0) vsize = out_sz;
    if (vsize > 0) {
        vbuf = (char*)malloc(vsize);
        if (vbuf) {
            if (setvbuf(fo, vbuf, _IOFBF, (int)vsize) != 0) { free(vbuf); vbuf = NULL; }
        }
    }
    uint16_t magic = MAGIC;
    uint32_t orig_sz32 = (uint32_t)in_sz;
    uint32_t blk32 = (uint32_t)blk;
    uint32_t nblk32 = (uint32_t)nblk;
    uint32_t alg_id = 0;
    if (strcmp(alg_name, "lzo1y") == 0) alg_id = 1;

    if (fwrite(&magic, sizeof(magic), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&orig_sz32, sizeof(orig_sz32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&blk32, sizeof(blk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&nblk32, sizeof(nblk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&alg_id, sizeof(alg_id), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(len_arr, sizeof(uint32_t), nblk, fo) != nblk) { perror("fwrite"); fclose(fo); return 1; }
    /* Zero-Copy 写入 - 直接从 mapped memory (dev_out) scatter-gather 到文件
     * 优化: 当总压缩大小较小时，合并拷贝到单个连续缓冲区并一次性写入以减少syscall数量并降低写入时延方差。
     * 控制:
     *  - 合并写入（coalesced write）已默认启用以减少 syscall 并降低写入时延方差
     *  - 为避免极大内存分配，自动在输出大小 > LZO_DEFAULT_COALESCE_MAX_MB 时禁用一次性合并写入
     *  - Enable per-block and coalesce write timing when debug diagnostics are enabled (e.g., LZO_DEBUG=1)
     */
    /* Default: coalesce output is enabled; avoid attempting full coalesce if
     * out_sz is larger than a default threshold LZO_DEFAULT_COALESCE_MAX_MB.
     * The chunking size for a fallback is LZO_DEFAULT_COALESCE_CHUNK_MB.
     * These are library/app defaults (for reduced env var list) and can be
     * overridden by command-line flags for advanced runs.
     */
    int coalesce = LZO_DEFAULT_COALESCE_OUTPUT; /* placeholder: add CLI/env override support */
    size_t coalesce_max_mb = (size_t)LZO_DEFAULT_COALESCE_MAX_MB; /* placeholder: add CLI/env override support */

    int profile_writes = debug; /* profile write diagnostics gated by debug flag */

    if (coalesce && out_sz > 0) {
        /* attempt a single contiguous copy + single fwrite */
        unsigned char *contig = (unsigned char*)malloc(out_sz);
        if (contig) {
            uint64_t t_copy_start = now_ns();
            size_t dest = 0;
            for (size_t i = 0; i < nblk; ++i) {
                if (len_arr[i] > 0) {
                    size_t dev_off = i * worst_blk;
                    memcpy(contig + dest, dev_out + dev_off, len_arr[i]);
                    dest += len_arr[i];
                }
            }
            uint64_t t_copy_end = now_ns();

            uint64_t t_write_blk_start = now_ns();
            if (fwrite(contig, 1, out_sz, fo) != out_sz) {
                perror("fwrite contiguous");
                free(contig);
                fclose(fo);
                if (vbuf) { free(vbuf); vbuf = NULL; }
                CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                free(len_arr);
                return 1;
            }
            uint64_t t_write_blk_end = now_ns();

            if (profile_writes) {
                printf("COALESCE_COPY: %.3f ms\n", (t_copy_end - t_copy_start)/1e6);
                printf("COALESCE_WRITE: %.3f ms\n", (t_write_blk_end - t_write_blk_start)/1e6);
            }

            free(contig);
        } else {
            /* allocation failed - try chunked coalesce before falling back to per-block writes */
            size_t chunk_mb = (size_t)LZO_DEFAULT_COALESCE_CHUNK_MB; /* placeholder: add CLI/env override support */
            size_t chunk_size = chunk_mb * 1024 * 1024;
            if (chunk_size == 0) chunk_size = 16 * 1024 * 1024;
            unsigned char *chunk = (unsigned char*)malloc(chunk_size);
            if (chunk) {
                size_t used = 0;
                for (size_t i = 0; i < nblk; ++i) {
                    if (len_arr[i] == 0) continue;
                    size_t dev_off = i * worst_blk;
                    if (len_arr[i] > chunk_size && used == 0) {
                        uint64_t t1 = now_ns();
                        if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                            perror("fwrite block large");
                            free(chunk);
                            fclose(fo);
                            if (vbuf) { free(vbuf); vbuf = NULL; }
                            CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                            free(len_arr);
                            return 1;
                        }
                        uint64_t t2 = now_ns();
                        if (profile_writes) printf("BLOCK_WRITE %zu len=%u : %.3f ms\n", i, len_arr[i], (t2 - t1)/1e6);
                        continue;
                    }
                    if (used + len_arr[i] > chunk_size) {
                        uint64_t t_write_chunk_start = now_ns();
                        if (fwrite(chunk, 1, used, fo) != used) {
                            perror("fwrite chunk");
                            free(chunk);
                            fclose(fo);
                            if (vbuf) { free(vbuf); vbuf = NULL; }
                            CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                            free(len_arr);
                            return 1;
                        }
                        uint64_t t_write_chunk_end = now_ns();
                        if (profile_writes) printf("CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_write_chunk_end - t_write_chunk_start)/1e6, used);
                        used = 0;
                    }
                    memcpy(chunk + used, dev_out + dev_off, len_arr[i]);
                    used += len_arr[i];
                }
                if (used > 0) {
                    uint64_t t_write_chunk_start = now_ns();
                    if (fwrite(chunk, 1, used, fo) != used) {
                        perror("fwrite chunk final");
                        free(chunk);
                        fclose(fo);
                        if (vbuf) { free(vbuf); vbuf = NULL; }
                        CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                        free(len_arr);
                        return 1;
                    }
                    uint64_t t_write_chunk_end = now_ns();
                    if (profile_writes) printf("CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_write_chunk_end - t_write_chunk_start)/1e6, used);
                }
                free(chunk);
            } else {
                fprintf(stderr, "warning: coalesce allocation and chunk alloc both failed (%zu bytes), falling back to per-block writes\n", out_sz);
                for (size_t i = 0; i < nblk; ++i) {
                    if (len_arr[i] > 0) {
                        size_t dev_off = i * worst_blk;
                        if (profile_writes) {
                            uint64_t t1 = now_ns();
                            if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                                perror("fwrite block");
                                fclose(fo);
                                if (vbuf) { free(vbuf); vbuf = NULL; }
                                CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                                free(len_arr);
                                return 1;
                            }
                            uint64_t t2 = now_ns();
                            printf("BLOCK_WRITE %zu len=%u : %.3f ms\n", i, len_arr[i], (t2 - t1)/1e6);
                        } else {
                            if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                                perror("fwrite block");
                                fclose(fo);
                                if (vbuf) { free(vbuf); vbuf = NULL; }
                                CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                                free(len_arr);
                                return 1;
                            }
                        }
                    }
                }
            }
        }
    } else {
        /* scatter-gather write (original behavior), optionally profile per-block writes */
        for (size_t i = 0; i < nblk; ++i) {
            if (len_arr[i] > 0) {
                size_t dev_off = i * worst_blk;
                if (profile_writes) {
                    uint64_t t1 = now_ns();
                    if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                        perror("fwrite block");
                        fclose(fo);
                        if (vbuf) { free(vbuf); vbuf = NULL; }
                        CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                        free(len_arr);
                        return 1;
                    }
                    uint64_t t2 = now_ns();
                    printf("BLOCK_WRITE %zu len=%u : %.3f ms\n", i, len_arr[i], (t2 - t1)/1e6);
                } else {
                    if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                        perror("fwrite block");
                        fclose(fo);
                        if (vbuf) { free(vbuf); vbuf = NULL; }
                        CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                        free(len_arr);
                        return 1;
                    }
                }
            }
        }
    }
    fclose(fo);
    if (vbuf) { free(vbuf); vbuf = NULL; }
    /* 写入完成后才 unmap */
    CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
    printf("wrote %s\n", output_path);

    uint64_t t_after_write = now_ns();



    /* 计算各阶段耗时 */
    double ms_file_read = (t_io_read_done - t_io_read_start)/1e6;
    double ms_ocl_init = (t_ocl_init - t_after_fopen)/1e6;
    double ms_kernel_load = (t_kernel_load_end - t_kernel_load_start)/1e6;
    double ms_blocking = (t_blocking_end - t_blocking_start)/1e6;
    double ms_buffer_alloc_in = (t_buffer_alloc_end - t_buffer_alloc_start)/1e6;
    double ms_upload = (t_upload_end - t_upload_start)/1e6;
    double ms_buffer_alloc_out = (t_out_buffer_end - t_out_buffer_start)/1e6;
    double ms_buffer_alloc_len = (t_len_buffer_end - t_len_buffer_start)/1e6;
    double ms_setup_args = (t_setup_args_end - t_setup_args_start)/1e6;
    double ms_kernel = (t_exec_end - t_exec_start)/1e6;
    double ms_len_read = (t_len_read_end - t_len_read_start)/1e6;
    double ms_bulk_read = (t_bulk_read_end - t_bulk_read_start)/1e6;
    double ms_download_total = (t_download_end - t_download_start)/1e6;
    double ms_file_write = (t_after_write - t_write_start)/1e6;
    /* Use compress start as the canonical total start so OCL init/kernel
     * loads (performed before file read start) are included in TOTAL.
     */
    double ms_total = (t_after_write - t_compress_start)/1e6;
    double ms_buffer_alloc_total = ms_buffer_alloc_in + ms_buffer_alloc_out + ms_buffer_alloc_len;

    double ratio = out_sz > 0 ? (double)in_sz / (double)out_sz : 0.0;
    double thrpt = ms_kernel > 0 ? ((double)in_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;

#if ENABLE_COMPRESSION_RATIO_TRACKING
    /* 输出压缩统计信息 */
    printf("\n=== Compression Statistics ===\n");
    printf("Input size         : %zu bytes (%.2f MB)\n", in_sz, in_sz / (1024.0 * 1024.0));
    printf("Compressed size    : %zu bytes (%.2f MB)\n", out_sz, out_sz / (1024.0 * 1024.0));
    printf("Compression ratio  : %.2f:1 (%.2f%% of original)\n", ratio, 100.0 / ratio);
    printf("Block size         : %zu bytes/%zu KB\n", blk, blk / 1024);
    printf("Kernel             : %s (bits=%d)\n", alg_name, comp_level);
    printf("Work groups        : global=%zu, local=%zu\n", gsz, lsz);
    printf("Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n",
           ((double)in_sz / (1024.0*1024.0)) / (ms_total/1000.0), thrpt);
    printf("==============================\n\n");
#endif

    /* 打印详细的时间分解 */
    printf("\n=== Time Breakdown (Compression) ===\n");
    print_ns("1. File Read", t_io_read_done - t_io_read_start);
    print_ns("2. OCL Init", t_ocl_init - t_after_fopen);
    print_ns("3. Kernel Load", t_kernel_load_end - t_kernel_load_start);
    print_ns("4. Blocking Calc", t_blocking_end - t_blocking_start);
    print_ns("5. Buffer Alloc (in)", t_buffer_alloc_end - t_buffer_alloc_start);
    print_ns("6. Data Upload", t_upload_end - t_upload_start);
    print_ns("7. Buffer Alloc (out)", t_out_buffer_end - t_out_buffer_start);
    print_ns("8. Buffer Alloc (len)", t_len_buffer_end - t_len_buffer_start);
    print_ns("9. Setup Args", t_setup_args_end - t_setup_args_start);
    print_ns("10. Kernel Exec", t_exec_end - t_exec_start);
    print_ns("11. Download (len)", t_len_read_end - t_len_read_start);
    print_ns("12. Download (bulk)", t_bulk_read_end - t_bulk_read_start);
    print_ns("13. Download Total", t_download_end - t_download_start);
    print_ns("14. File Write", t_after_write - t_write_start);
    print_ns("TOTAL", t_after_write - t_compress_start);
    printf("\n");

        /* 计算占比 */
        printf("=== Percentage Breakdown ===\n");
        /* protect against zero total */
        double denom2 = (ms_total > 0.0) ? ms_total : 1.0;
        int zero_total2 = (ms_total <= 0.0);
        printf("Kernel Exec     : %6.2f%%\n", zero_total2 ? 0.0 : 100.0 * ms_kernel / denom2);
        printf("Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
               zero_total2 ? 0.0 : 100.0 * (ms_upload + ms_download_total) / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_upload / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_download_total / denom2);
        printf("File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
               zero_total2 ? 0.0 : 100.0 * (ms_file_read + ms_file_write) / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_file_read / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_file_write / denom2);
        printf("Buffer Alloc    : %6.2f%% (in=%.2f%% + out=%.2f%% + len=%.2f%%)\n",
               zero_total2 ? 0.0 : 100.0 * ms_buffer_alloc_total / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_buffer_alloc_in / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_buffer_alloc_out / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_buffer_alloc_len / denom2);
        printf("OCL Setup       : %6.2f%% (init=%.2f%% + kernel_load=%.2f%%)\n",
               zero_total2 ? 0.0 : 100.0 * (ms_ocl_init + ms_kernel_load) / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_ocl_init / denom2,
               zero_total2 ? 0.0 : 100.0 * ms_kernel_load / denom2);
        printf("Kernel Args     : %6.2f%%\n",
               zero_total2 ? 0.0 : 100.0 * ms_setup_args / denom2);
        printf("Other           : %6.2f%%\n",
               zero_total2 ? 0.0 : 100.0 * ms_blocking / denom2);
    printf("\n");

    /* optional roundtrip verification only when --verify set */
    if (verify_flag) {
        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (size_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

        /* Determine decompression kernel based on algorithm */
        char decomp_base2[64];
        snprintf(decomp_base2, sizeof(decomp_base2), "%s_decomp", alg_name);

        /* Load program */
        cl_program prog_d = load_prog_with_dbits(decomp_base2, 0);

        if (!prog_d) {
            fprintf(stderr, "error: unable to load decompressor for verify\n");
            return 1;
        }

        cl_kernel krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err);
        CHECK(err);
        /* Try to create device buffer using CL_MEM_COPY_HOST_PTR first; if driver rejects host pointer
         * (returns CL_INVALID_HOST_PTR), fall back to explicit clEnqueueWriteBuffer to upload.
         */
        cl_mem d_comp;
        unsigned char *packed = NULL;
        /* If we removed the contiguous host compressed buffer earlier (zero-copy path),
         * we must assemble a contiguous host buffer for verify: the device verify kernel
         * expects contiguous compressed bytes with offsets matching `off_arr`.
         */
        if (out_buf == NULL && out_sz > 0 && dev_out != NULL) {
            /* allocate and pack compressed blocks from the dev_out mapped buffer */
            packed = malloc(out_sz);
            if (!packed) {
                fprintf(stderr, "error: unable to allocate %zu bytes for verify packed buffer\n", out_sz);
                free(off_arr);
                clReleaseKernel(krn_d); clReleaseProgram(prog_d);
                return 1;
            }
            for (size_t bi = 0; bi < (size_t)nblk; ++bi) {
                size_t src_off = (size_t)bi * worst_blk;
                uint32_t l = len_arr[bi];
                if (l > 0 && src_off + l <= out_cap) {
                    memcpy(packed + off_arr[bi], dev_out + src_off, l);
                } else if (l > 0) {
                    fprintf(stderr, "error: dev_out bounds exceeded while packing for verify\n");
                    free(packed);
                    free(off_arr);
                    clReleaseKernel(krn_d); clReleaseProgram(prog_d);
                    return 1;
                }
            }
            /* Leave original out_buf untouched; use packed for upload instead. */
        }

        d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_sz, (packed ? packed : out_buf), &err);
        if (err == CL_INVALID_HOST_PTR) {
            /* fallback: create buffer without host pointer and upload via enqueue write */
            d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY, out_sz, NULL, &err);
            CHECK(err);
            CHECK(clEnqueueWriteBuffer(q, d_comp, CL_TRUE, 0, out_sz, (packed ? packed : out_buf), 0, NULL, NULL));
        } else {
            CHECK(err);
        }
        cl_mem d_off;
        d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (nblk + 1) * sizeof(cl_uint), off_arr, &err);
        if (err == CL_INVALID_HOST_PTR) {
            d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (nblk + 1) * sizeof(cl_uint), NULL, &err);
            CHECK(err);
            CHECK(clEnqueueWriteBuffer(q, d_off, CL_TRUE, 0, (nblk + 1) * sizeof(cl_uint), off_arr, 0, NULL, NULL));
        } else {
            CHECK(err);
        }
        cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, in_sz, NULL, &err); CHECK(err);
        cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

        CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
        CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
        CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
        CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
        CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk));
        CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &in_sz));
        CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));

        cl_event evt_verify;
        CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt_verify));
        clWaitForEvents(1, &evt_verify);
        unsigned char* out2 = malloc(in_sz);
        CHECK(clEnqueueReadBuffer(q, d_out2, CL_TRUE, 0, in_sz, out2, 0, NULL, NULL));

        /* 新读取原始文件用于验证 */
        size_t verify_sz;
        unsigned char* verify_buf = (unsigned char*)lzo_read_file(in_path, &verify_sz);
        if (verify_sz != in_sz) {
            fprintf(stderr, "verify size mismatch: %zu != %zu\n", verify_sz, in_sz);
        } else if (memcmp(verify_buf, out2, in_sz) == 0) {
            printf("verify OK\n");
        } else {
            printf("verify FAILED\n");
            for (size_t i=0; i<in_sz; i++) {
                if (verify_buf[i] != out2[i]) {
                    printf("first mismatch at %zu (0x%02x != 0x%02x)\n", i, verify_buf[i], out2[i]);
                    break;
                }
            }
        }
        free(verify_buf);

        clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        free(off_arr); free(out2);
        if (packed) free(packed);
    }

    /* cleanup */
    clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
    clReleaseKernel(krn_c);
    clReleaseProgram(prog_c);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(len_arr);
    return 0;
}
