#define _POSIX_C_SOURCE 199309L
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
#define OCC_FACTOR        128           /* 优化: 从12提升到128，大幅增加并行度 */
#define ALIGN_BYTES       16384         /* 优化: 从64KB降到16KB，减小对齐浪费 */
#define MIN_BLOCK_SIZE    (64 * 1024)   /* 最小块大小: 64KB (从512KB大幅降低) */
#define MAX_BLOCK_SIZE    (256 * 1024)  /* 最大块大小: 256KB (从2MB降低) */

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
    printf("%-22s : %8.3f ms\n", tag, ns / 1e6);
}

static void choose_blocking(size_t in_sz, cl_device_id dev,
    size_t* blk_sz_out, size_t* nblk_out)
{
    /* 1. 取 GPU 计算单元数 */
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 2. ⽬标块数：CU × OCC_FACTOR，但不能多于字节数 */
    size_t tgt_blk = (size_t)cu * OCC_FACTOR;
    /* Allow forcing a smaller target block count for testing via
     * LZO_FORCE_NBLK environment variable (helps run CPU-style kernel
     * on GPUs by reducing concurrent work-items). */
    const char* env_nblk = getenv("LZO_FORCE_NBLK");
    if (env_nblk) {
        int v = atoi(env_nblk);
        if (v > 0) tgt_blk = (size_t)v;
    }
    if (tgt_blk > in_sz) tgt_blk = in_sz;        /* 每块≥1 B */

    /* 3. 初步块⼤⼩ = ceil(in_sz / tgt_blk) */
    size_t blk = (in_sz + tgt_blk - 1) / tgt_blk;

    /* 4. 向 ALIGN_BYTES 对齐，且不得为 0 */
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    /* 4.5 优化: 限制块大小在合理范围内 */
    if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
    if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;

    /* 5. 重新得到块数；如仍 < CU，则再细分以 nblk = CU 为下限 */
    size_t nblk = (in_sz + blk - 1) / blk;
    if (nblk < cu) {
        nblk = cu;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        /* 再次检查范围 */
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    }

    /* 6. 尾块太⼩（< blk/4）时，把数据平均回各块 */
    size_t tail = in_sz - blk * (nblk - 1);
    if (nblk > 1 && tail < blk / 4) {
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        /* 再次检查范围 */
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    }

    *blk_sz_out = blk;
    *nblk_out = (in_sz + blk - 1) / blk;
}

#define CHECK(expr)  do{ cl_int _e=(expr);                       \
        if(_e!=CL_SUCCESS){                                      \
            fprintf(stderr,"OpenCL error %d at %s:%d\n",         \
                    _e,__FILE__,__LINE__); exit(1);} }while(0)

static inline size_t lzo_worst(size_t n) {
    return n + n / 16 + 64 + 3;
}

/* read mem-images/kernel-source */
static char* read_file(const char* path, size_t* sz_out)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror(path); exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* buf = malloc(sz + 1);
    fread(buf, 1, sz, fp);
    fclose(fp);

    if (sz_out)
        *sz_out = (size_t)sz;
    buf[sz] = '\0';
    return buf;
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

/* 优化: 内核参数缓存以避免重复设置 */
static struct {
    cl_kernel kernel;
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    cl_uint in_sz;
    cl_uint blk;
    cl_uint worst_blk;
} kernel_args_cache = {0};

static void set_kernel_args_cached(cl_kernel krn, cl_mem d_in, cl_mem d_out,
                                   cl_mem d_len, cl_uint in_sz, cl_uint blk,
                                   cl_uint worst_blk) {
    /* 仅在参数变化时才设置 */
    if (kernel_args_cache.kernel != krn ||
        kernel_args_cache.d_in != d_in ||
        kernel_args_cache.d_out != d_out ||
        kernel_args_cache.d_len != d_len ||
        kernel_args_cache.in_sz != in_sz ||
        kernel_args_cache.blk != blk ||
        kernel_args_cache.worst_blk != worst_blk) {

        if (debug) fprintf(stderr, "DBG: Setting kernel args (cache miss)\n");

        CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &d_in));
        CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &d_out));
        CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &d_len));
        CHECK(clSetKernelArg(krn, 3, sizeof(cl_uint), &in_sz));
        CHECK(clSetKernelArg(krn, 4, sizeof(cl_uint), &blk));
        CHECK(clSetKernelArg(krn, 5, sizeof(cl_uint), &worst_blk));

        /* 更新缓存 */
        kernel_args_cache.kernel = krn;
        kernel_args_cache.d_in = d_in;
        kernel_args_cache.d_out = d_out;
        kernel_args_cache.d_len = d_len;
        kernel_args_cache.in_sz = in_sz;
        kernel_args_cache.blk = blk;
        kernel_args_cache.worst_blk = worst_blk;
    } else {
        if (debug) fprintf(stderr, "DBG: Kernel args cached (skip setting)\n");
    }
}

static cl_mem get_or_create_buffer(cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags) {
    if (*cached_size < required_size) {
        if (*cached_buf) clReleaseMemObject(*cached_buf);
        cl_int err;
        *cached_buf = clCreateBuffer(ctx, flags, required_size, NULL, &err);
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
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path);

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

/* Helper: load program from <base>.bin or from source file */
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path)
{
    char use_base[256]; strncpy(use_base, base, sizeof(use_base)-1); use_base[sizeof(use_base)-1]='\0';
    char use_cl_src[256]; strncpy(use_cl_src, cl_src_path, sizeof(use_cl_src)-1); use_cl_src[sizeof(use_cl_src)-1]='\0';

    char bin_path[512]; snprintf(bin_path, sizeof(bin_path), "%s.bin", use_base);
    /* also prepare an alternate path inside the lzo_gpu subdir to be robust
     * against differing working directories when running via tools/runner */
    char bin_path_alt[512]; snprintf(bin_path_alt, sizeof(bin_path_alt), "lzo_gpu/%s.bin", use_base);
    /* Attempt to use a precompiled binary first (robust fallback to source
     * compilation is performed below if binary is incompatible). */
    FILE* fb = fopen(bin_path, "rb");
    if (!fb) {
        /* try lzo_gpu/ subdir */
        fb = fopen(bin_path_alt, "rb");
    }
    cl_int err; cl_program prog = NULL;
    if (fb) {
        /* attempt to use precompiled binary; on any failure fall back to source */
        fseek(fb, 0, SEEK_END); long bsz = ftell(fb); fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin,1,bsz,fb) != (size_t)bsz) { perror("fread"); fclose(fb); free(bin); exit(1); }
        fclose(fb);
        cl_int binary_status;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
            (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
            fprintf(stderr, "warning: precompiled binary %s.bin incompatible, falling back to source (clCreateProgramWithBinary err=%d bin_status=%d)\n", base, err, binary_status);
            if (prog) { clReleaseProgram(prog); prog = NULL; }
        } else {
            err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
                fprintf(stderr, "Build log (from binary):\n%s\n", log); free(log);
                fprintf(stderr, "warning: build from binary failed for %s.bin (err=%d), falling back to source\n", base, err);
                clReleaseProgram(prog); prog = NULL;
            }
        }
    }

    if (!prog) {
        /* compile from source as fallback - use direct .cl file without frontend combinations */
        size_t src_len = 0; char* src = NULL;
        /* try source in current dir, otherwise try lzo_gpu/ subdir */
        FILE* fchk = fopen(use_cl_src, "rb");
        if (!fchk) {
            char use_cl_src_alt[512]; snprintf(use_cl_src_alt, sizeof(use_cl_src_alt), "lzo_gpu/%s", use_cl_src);
            fchk = fopen(use_cl_src_alt, "rb");
            if (fchk) {
                fclose(fchk);
                src = read_file(use_cl_src_alt, &src_len);
            }
        } else {
            fclose(fchk);
            src = read_file(use_cl_src, &src_len);
        }

        if (!src) {
            fprintf(stderr, "source file %s not found (frontend combinations removed)\n", use_cl_src);
            exit(1);
        }

        /* create program from assembled source */
        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed (err=%d)\n", err); free(src); exit(1); }
        /* Add include paths for OpenCL compiler to resolve #include directives
         * Try both current directory and lzo_gpu/ subdirectory */
        const char* build_opts = "-I. -I./lzo_gpu -I..";
        err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
            fprintf(stderr, "Build log (from source):\n%s\n", log); free(log); free(src); exit(1);
        }
        free(src);
    }
    return prog;
}

int main(int argc, char** argv)
{
    uint64_t t_start_total = now_ns();
    if (argc < 2) { fprintf(stderr, "usage: %s [--debug|-v] input_file (or -d [--debug|-v] lzfile orig_file)\n", argv[0]); return 1; }

    /* simple CLI parsing: support optional --debug/-v flag and -d decompress mode */
    /* debug is now global */
    int verify_flag = 0; /* only when set, do roundtrip/verify prints */
    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *orig_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int show_help = 0;
    const char *comp_level = "1"; /* compression level: "1", "1k", "1l", "1o" */

    /* pass 1: only detect mode (-d) and help, to know how to parse verify */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { show_help = 1; }
        if (strcmp(argv[i], "-d") == 0) { decompress_mode = 1; }
    }

    /* pass 2: parse options and positionals with knowledge of mode */
    const char* verify_path = NULL; /* only for -d mode */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--debug") == 0 || strcmp(arg, "-v") == 0) { debug = 1; continue; }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) { /* already noted */ continue; }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        /* --strategy option removed: strategy dimension is no longer supported */
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            comp_level = argv[++i];
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

    if (show_help) {
        printf("Usage:\n");
        printf("  %s [--debug|-v] [--verify|-c] [-L level] [-o out.lzo] input_file\n", argv[0]);
        printf("     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        printf("     - --verify/-c (compress mode): do in-memory roundtrip check (no arg).\n");
        printf("     - -L|--level LEVEL : compression level to select kernel variant (default: 1)\n");
        printf("         supported LEVEL values:\n");
        printf("            1   : default LZO1X-1 compressor (kernel: lzo1x_1)\n");
        printf("            1k  : LZO1X-1K variant (kernel: lzo1x_1k) - optimized for kernel K behavior\n");
        printf("            1l  : LZO1X-1L variant (kernel: lzo1x_1l) - alternative lookup/heuristics\n");
        printf("            1o  : LZO1X-1O variant (kernel: lzo1x_1o) - other tuning/optimizations\n");
        printf("\n");
        printf("  %s -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo\n", argv[0]);
        printf("     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        printf("     - --verify/-c ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n");
        printf("\n");
        printf("Examples:\n");
        printf("  Compress with default level: %s input.dat -o out.lzo\n", argv[0]);
        printf("  Compress with level 1k:      %s -L 1k input.dat -o out.lzo\n", argv[0]);
        printf("  Decompress and verify:      %s -d --verify input.dat out.lzo -o out.dec\n", argv[0]);
        printf("  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", argv[0]);
        printf("  %s -h|--help                                 # show this help\n", argv[0]);
        return 0;
    }

    /* Decompress mode */
    if (decompress_mode) {
        if (!lz_path) { fprintf(stderr, "no input .lzo specified (after -d)\n"); return 1; }
    uint64_t t_io_in = now_ns();
    size_t lz_sz; unsigned char* lz_buf = read_file(lz_path, &lz_sz);
    unsigned char* ref = NULL;
    size_t ref_sz = 0;
        unsigned char* p = lz_buf;
        uint16_t magic = *(uint16_t*)p; p += 2;
        if (magic != MAGIC) { fprintf(stderr, "bad magic\n"); return 1; }
        uint32_t orig_sz = *(uint32_t*)p; p += 4;
        uint32_t blk_sz = *(uint32_t*)p; p += 4;
        uint32_t nblk = *(uint32_t*)p; p += 4;
        uint32_t* len_arr = (uint32_t*)p; p += 4 * nblk;
        size_t comp_sz = lz_sz - (p - lz_buf);

        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (uint32_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

    uint64_t t_io_after = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
        /* allow selecting a vectorized decompressor via env var LZO_DECOMP_VEC=1
         * If not set, auto-detect from device capabilities: require
         * CL_DEVICE_MEM_BASE_ADDR_ALIGN >= 128 (bits => 16 bytes) and
         * CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR >= 16 to enable vec path.
         */
        /* Use explicit environment variable to enable vectorized decompressor.
         * Previously we auto-detected device capabilities and enabled vec by
         * default. Change behavior: default to the scalar decompressor unless
         * `LZO_DECOMP_VEC=1` is set. This avoids surprising runtime selection
         * and makes experimental runs deterministic. */
        const char* devec_env = getenv("LZO_DECOMP_VEC");
        int devec_flag = 0;
        if (devec_env) {
            devec_flag = (strcmp(devec_env, "1") == 0);
        } else {
            /* no env override: default to scalar (devec_flag == 0)
             * (do not auto-detect device vector width anymore)
             */
            devec_flag = 0;
        }
        const char* decomp_base = devec_flag ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src  = devec_flag ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";
        /* Emit stable, parseable identifiers for aggregation tools */
        if (!suppress_non_data) {
            printf("KERNEL=%s\n", decomp_base);
        }

        /* 优化: 检查缓存以避免重复编译和创建内核 */
        cl_program prog_d = NULL;
        cl_kernel krn_d = NULL;
        int cache_idx_d = find_cached_program(decomp_base);

        if (cache_idx_d >= 0) {
            /* 使用缓存的程序和内核 */
            prog_d = prog_cache[cache_idx_d].prog;
            krn_d = prog_cache[cache_idx_d].krn_decompress;
            if (debug) fprintf(stderr, "DBG: using cached decompress program/kernel for %s\n", decomp_base);
        } else {
            /* 首次加载: 编译并缓存 */
            if (debug) fprintf(stderr, "DBG: loading and caching decompress program %s\n", decomp_base);
            prog_d = load_prog_from_bin_or_src(decomp_base, decomp_src);
            cl_int err;
            krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err);
            CHECK(err);

            /* 缓存程序和内核供后续使用 */
            cache_program(decomp_base, prog_d, NULL, krn_d);
        }
        cl_int err;

    /* 优化: 使用pinned memory创建所有缓冲区 */
    cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, comp_sz, NULL, &err);
    CHECK(err);
    /* 使用map上传压缩数据 */
    void* mapped_comp = clEnqueueMapBuffer(q, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_comp, p, comp_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_comp, mapped_comp, 0, NULL, NULL));

    cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (nblk + 1) * sizeof(cl_uint), NULL, &err);
    CHECK(err);
    void* mapped_off = clEnqueueMapBuffer(q, d_off, CL_TRUE, CL_MAP_WRITE, 0, (nblk + 1) * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_off, off_arr, (nblk + 1) * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(q, d_off, mapped_off, 0, NULL, NULL));

    cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, orig_sz, NULL, &err);
    CHECK(err);
    /* decompressor expects an out_lens buffer as arg 3 */
    cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, nblk * sizeof(cl_uint), NULL, &err);
    CHECK(err);

    CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
    CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));

    size_t gsz = nblk, lsz = 1;
    cl_event evt_decomp;
    uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt_decomp));
    clWaitForEvents(1, &evt_decomp);
    uint64_t t_exec_end = now_ns();

    /* 优化: 使用map读取解压数据(零拷贝) */
    unsigned char* out2 = malloc(orig_sz);
    uint64_t t_read_start = now_ns();
    void* mapped_out2 = clEnqueueMapBuffer(q, d_out2, CL_TRUE, CL_MAP_READ, 0, orig_sz,
                                           0, NULL, NULL, &err);
    CHECK(err);
    memcpy(out2, mapped_out2, orig_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_out2, mapped_out2, 0, NULL, NULL));
    uint64_t t_read_end = now_ns();

    /* perform verify first if requested; on failure do not write and exit non-zero */
    if (verify_path) {
        size_t ref_sz; unsigned char* ref = (unsigned char*)read_file(verify_path, &ref_sz);
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
                if (fwrite(out2, 1, orig_sz, fo) != orig_sz) { perror("fwrite"); fclose(fo); return 1; }
                fclose(fo);
                if (!suppress_non_data) printf("wrote %s\n", output_path);
            }
        }
    }

        uint64_t t_total_end = now_ns();
        double ms_total = (t_total_end - t_start_total)/1e6;
        double ms_io = (t_io_after - t_io_in)/1e6;
        double ms_kernel = (t_exec_end - t_exec_start)/1e6;
        double ms_read = (t_read_end - t_read_start)/1e6;
        double ratio = lz_sz > 0 ? (double)orig_sz / (double)lz_sz : 0.0;
        double thrpt = ms_kernel > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;
        printf("[DECOMP] orig=%zu comp=%zu blocks=%u blk_size=%u ratio=%.3f kernel=%.3f ms io=%.3f ms read=%.3f ms total=%.3f ms thrpt=%.2f MB/s\n",
                (size_t)orig_sz, (size_t)lz_sz, nblk, blk_sz, ratio, ms_kernel, ms_io, ms_read, ms_total, thrpt);

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
    uint64_t t_io_in = now_ns();
    size_t in_sz; unsigned char* in_buf = (unsigned char*)read_file(in_path, &in_sz);

    uint64_t t_io_read_done = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
    /* select compression kernel variant based on comp_level */
    char kernel_base[64]; char cl_src[128];
    if (strcmp(comp_level, "1") == 0 || strcmp(comp_level, "1x") == 0) {
        strcpy(kernel_base, "lzo1x_1");
    } else if (strcmp(comp_level, "1k") == 0) {
        strcpy(kernel_base, "lzo1x_1k");
    } else if (strcmp(comp_level, "1l") == 0) {
        strcpy(kernel_base, "lzo1x_1l");
    } else if (strcmp(comp_level, "1o") == 0) {
        strcpy(kernel_base, "lzo1x_1o");
    } else {
        fprintf(stderr, "unknown compression level: %s\n", comp_level);
        return 1;
    }

    /* Use standalone kernel (no frontend combinations) */
    snprintf(cl_src, sizeof(cl_src), "%s.cl", kernel_base);

    /* Emit stable, parseable identifiers for aggregation tools */
    if (!suppress_non_data) {
        printf("KERNEL=%s\n", kernel_base);
    }

    /* 优化: 检查缓存以避免重复编译和创建内核 */
    uint64_t t_kernel_load_start = now_ns();
    cl_int err;
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    int cache_idx = find_cached_program(kernel_base);

    if (cache_idx >= 0) {
        /* 使用缓存的程序和内核 */
        prog_c = prog_cache[cache_idx].prog;
        krn_c = prog_cache[cache_idx].krn_compress;
        if (debug) fprintf(stderr, "DBG: using cached program/kernel for %s\n", kernel_base);
    } else {
        /* 首次加载: 编译并缓存 */
        if (debug) fprintf(stderr, "DBG: loading and caching program %s\n", kernel_base);
        prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);

        /* select kernel function name according to the kernel_base we loaded
         * Use canonical exported symbol 'lzo1x_block_compress' from frontends.
         */
        char krn_name[64];
        strcpy(krn_name, "lzo1x_block_compress");
        krn_c = clCreateKernel(prog_c, krn_name, &err);
        if (err != CL_SUCCESS) {
            /* Simplified fallback: report available kernels and force a single
             * source rebuild retry. Avoid multiple name-specific fallbacks — precompile
             * step should produce binaries matching canonical exported symbol.
             */
            if (err == CL_INVALID_KERNEL_NAME) {
                size_t kn_sz = 0;
                clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kn_sz);
                if (kn_sz > 0) {
                    char* kn = malloc(kn_sz + 1);
                    clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, kn_sz, kn, NULL);
                    kn[kn_sz] = '\0';
                    fprintf(stderr, "kernel '%s' not found; available kernels: %s\n", krn_name, kn);
                    free(kn);
                } else {
                    fprintf(stderr, "kernel '%s' not found and program reports no kernel names\n", krn_name);
                }
                /* Force a source-built program and retry once */
                clReleaseProgram(prog_c);
                prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);
                krn_c = clCreateKernel(prog_c, krn_name, &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel after source rebuild failed (err=%d)\n", err); exit(1); }
            } else {
                fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); exit(1);
            }
        }

        /* 缓存程序和内核供后续使用 */
        cache_program(kernel_base, prog_c, krn_c, NULL);
    }
    uint64_t t_kernel_load_end = now_ns();

    /* choose blocking dynamically (uses GPU CU count and ALIGN_BYTES) */
    uint64_t t_blocking_start = now_ns();
    size_t blk = 0, nblk = 0;
    choose_blocking(in_sz, dev, &blk, &nblk);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    if (debug) {
        fprintf(stderr, "DBG: choose_blocking -> in_sz=%zu blk=%zu nblk=%zu worst_blk=%zu out_cap=%zu\n",
                in_sz, blk, nblk, worst_blk, out_cap);
    }
    uint64_t t_blocking_end = now_ns();

    /* 优化: 使用缓冲区缓存避免重复创建 */
    uint64_t t_buffer_alloc_start = now_ns();
    if (debug) fprintf(stderr, "DBG: getting cached d_in size=%zu\n", in_sz);
    cl_mem d_in = get_or_create_buffer(&buffer_cache.d_in, &buffer_cache.in_size,
                                       in_sz, CL_MEM_READ_ONLY);  /* 优化:移除ALLOC_HOST_PTR */
    uint64_t t_buffer_alloc_end = now_ns();

    /* 使用map上传(保持同步以确保数据完整性) */
    uint64_t t_upload_start = now_ns();
    void* mapped_in = clEnqueueMapBuffer(q, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz,
                                         0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_in, in_buf, in_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL));
    uint64_t t_upload_end = now_ns();

    /* 创建输出缓冲区 */
    uint64_t t_out_buffer_start = now_ns();
    if (debug) fprintf(stderr, "DBG: getting cached d_out size=%zu\n", out_cap);
    cl_mem d_out = get_or_create_buffer(&buffer_cache.d_out, &buffer_cache.out_size,
                                        out_cap, CL_MEM_WRITE_ONLY);  /* 优化:移除ALLOC_HOST_PTR */
    uint64_t t_out_buffer_end = now_ns();

    /* map_mode: 0=default CL_MEM_WRITE_ONLY + clEnqueueReadBuffer
     * 1=ALLOC_HOST_PTR + clEnqueueMapBuffer
     * 2=USE_HOST_PTR with host pointer (posix_memalign)
     */
    int map_mode = 0;
    /* Default to explicit reads (map_mode=0) for all kernels. Individual
     * experiments can still force mapping via LZO_FORCE_MAP=1. */
    /* Allow overriding map_mode via environment for testing/comparison.
     * Set environment variable LZO_FORCE_MAP=0 or =1 to force explicit read
     * or alloc-host+map respectively. Helpful to produce two test binaries
     * that exercise map==0 vs map==1 deterministically. */
    const char* force_map_s = getenv("LZO_FORCE_MAP");
    if (force_map_s) {
        int fm = atoi(force_map_s);
        if (fm == 0 || fm == 1) {
            if (debug) fprintf(stderr, "DBG: forcing map_mode from env LZO_FORCE_MAP=%d\n", fm);
            map_mode = fm;
        } else {
            if (debug) fprintf(stderr, "DBG: LZO_FORCE_MAP='%s' ignored (not 0 or 1)\n", force_map_s);
        }
    }
    /* The 'usehost' strategy has been removed; only map_mode 0/1 are used. */

    /* 优化: 使用缓冲区缓存 */
    uint64_t t_len_buffer_start = now_ns();
    size_t len_bytes = nblk * sizeof(cl_uint);
    if (debug) fprintf(stderr, "DBG: getting cached d_len size=%zu\n", len_bytes);
    cl_mem d_len = get_or_create_buffer(&buffer_cache.d_len, &buffer_cache.len_size,
                                        len_bytes, CL_MEM_READ_WRITE);  /* 优化:移除ALLOC_HOST_PTR */
    uint64_t t_len_buffer_end = now_ns();

    /* 优化: 使用参数缓存避免重复设置 */
    uint64_t t_setup_args_start = now_ns();
    set_kernel_args_cached(krn_c, d_in, d_out, d_len, in_sz, blk, worst_blk);
    uint64_t t_setup_args_end = now_ns();

    size_t gsz = nblk, lsz = 1;
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
            free(in_buf);
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
        free(in_buf);
        return 1;
    }
    out_buf = malloc(out_sz);
    if (!out_buf) {
        fprintf(stderr, "ERR: malloc(%zu) failed\n", out_sz);
        free(dev_out);
        free(len_arr);
        clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
        clReleaseKernel(krn_c); clReleaseProgram(prog_c);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        free(in_buf);
        return 1;
    }
    for (size_t i = 0; i < nblk; ++i) {
        size_t dev_off = i * worst_blk;
        if (len_arr[i] > 0) {
            memcpy(out_buf + host_off, dev_out + dev_off, len_arr[i]);
            host_off += len_arr[i];
        }
    }
    /* 优化: unmap而非free */
    CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
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
    uint16_t magic = MAGIC;
    uint32_t orig_sz32 = (uint32_t)in_sz;
    uint32_t blk32 = (uint32_t)blk;
    uint32_t nblk32 = (uint32_t)nblk;
    if (fwrite(&magic, sizeof(magic), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&orig_sz32, sizeof(orig_sz32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&blk32, sizeof(blk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&nblk32, sizeof(nblk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(len_arr, sizeof(uint32_t), nblk, fo) != nblk) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(out_buf, 1, out_sz, fo) != out_sz) { perror("fwrite"); fclose(fo); return 1; }
    fclose(fo);
    printf("wrote %s\n", output_path);

    uint64_t t_after_write = now_ns();

    /* 计算各阶段耗时 */
    double ms_file_read = (t_io_read_done - t_io_in)/1e6;
    double ms_ocl_init = (t_ocl_init - t_io_read_done)/1e6;
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
    double ms_total = (t_after_write - t_compress_start)/1e6;
    double ms_buffer_alloc_total = ms_buffer_alloc_in + ms_buffer_alloc_out + ms_buffer_alloc_len;

    double ratio = out_sz > 0 ? (double)in_sz / (double)out_sz : 0.0;
    double thrpt = ms_kernel > 0 ? ((double)in_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;

    printf("[COMP ] orig=%zu comp=%zu blocks=%zu blk_size=%zu ratio=%.3f kernel=%.3f ms total=%.3f ms thrpt=%.2f MB/s\n",
           in_sz, out_sz, nblk, blk, ratio, ms_kernel, ms_total, thrpt);

    /* 打印详细的时间分解 */
    printf("\n=== Time Breakdown (Compression) ===\n");
    print_ns("1. File Read", t_io_read_done - t_io_in);
    print_ns("2. OCL Init", t_ocl_init - t_io_read_done);
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
    printf("Kernel Exec     : %6.2f%%\n", 100.0 * ms_kernel / ms_total);
    printf("Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
           100.0 * (ms_upload + ms_download_total) / ms_total,
           100.0 * ms_upload / ms_total,
           100.0 * ms_download_total / ms_total);
    printf("File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
           100.0 * (ms_file_read + ms_file_write) / ms_total,
           100.0 * ms_file_read / ms_total,
           100.0 * ms_file_write / ms_total);
    printf("Buffer Alloc    : %6.2f%% (in=%.2f%% + out=%.2f%% + len=%.2f%%)\n",
           100.0 * ms_buffer_alloc_total / ms_total,
           100.0 * ms_buffer_alloc_in / ms_total,
           100.0 * ms_buffer_alloc_out / ms_total,
           100.0 * ms_buffer_alloc_len / ms_total);
    printf("OCL Setup       : %6.2f%% (init=%.2f%% + kernel_load=%.2f%%)\n",
           100.0 * (ms_ocl_init + ms_kernel_load) / ms_total,
           100.0 * ms_ocl_init / ms_total,
           100.0 * ms_kernel_load / ms_total);
    printf("Kernel Args     : %6.2f%%\n",
           100.0 * ms_setup_args / ms_total);
    printf("Other           : %6.2f%%\n",
           100.0 * ms_blocking / ms_total);
    printf("\n");

    /* optional roundtrip verification only when --verify set */
    if (verify_flag) {
        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (size_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

        /* choose vectorized decompress kernel if requested for verify */
        const char* devec2 = getenv("LZO_DECOMP_VEC");
        int devec_flag2 = 0;
        if (devec2) {
            devec_flag2 = (strcmp(devec2, "1") == 0);
        } else {
            cl_uint mem_align_bits2 = 0, pref_char2 = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align_bits2), &mem_align_bits2, NULL);
            clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(pref_char2), &pref_char2, NULL);
            if (mem_align_bits2 >= 128 && pref_char2 >= 16) devec_flag2 = 1;
        }
        const char* decomp_base2 = devec_flag2 ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src2  = devec_flag2 ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";
        cl_program prog_d = load_prog_from_bin_or_src(decomp_base2, decomp_src2);
        cl_kernel krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err); CHECK(err);
        cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_sz, out_buf, &err); CHECK(err);
        cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (nblk + 1) * sizeof(cl_uint), off_arr, &err); CHECK(err);
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

        if (memcmp(in_buf, out2, in_sz) == 0) printf("verify OK\n"); else { printf("verify FAILED\n"); for (size_t i=0;i<in_sz;i++){ if (in_buf[i]!=out2[i]){ printf("first mismatch at %zu (0x%02x != 0x%02x)\n", i, in_buf[i], out2[i]); break;} } }

        clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        free(off_arr); free(out2);
    }

    /* cleanup */
    clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
    clReleaseKernel(krn_c);
    clReleaseProgram(prog_c);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(in_buf); free(len_arr); free(out_buf);
    return 0;
}
