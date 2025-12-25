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
#include "lzo_gpu_core.h"

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
static int g_cli_fixed_block_exact = 0; /* 1 = user specified bytes (B suffix) -> respect exact */
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
static int debug = 0;  /* use --debug to enable kernel instrumentation */
static int kernel_opt = 0; /* enable optimized debug kernel variant when set via LZO_KERNEL_OPT (env) */

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


/* Helper: load program from source file with D_BITS macro */
static cl_program load_prog_with_dbits(const char* alg_name, int bits)
{
    char build_log[8192] = {0};
    cl_program p = lzo_load_program_with_dbits(ctx, dev, alg_name, bits, build_log, sizeof(build_log));
    if (!p && debug) {
        fprintf(stderr, "DBG: failed to load/compile kernel %s (D_BITS=%d): %s\n", alg_name, bits, build_log);
    }
    return p;
}

static inline void show_help(char *prog_name)
{
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s [options] [-c] [-a alg] [-L level] [-o out.lzo] input_file\n", prog_name);
        fprintf(stderr, "     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --debug              Enable instrumented kernel variant (writes per-block debug stats)\n");
        fprintf(stderr, "  -c                   Compress mode (default, explicit flag)\n");
        fprintf(stderr, "  --verify             (compress mode): do in-memory roundtrip check (no arg)\n");
        fprintf(stderr, "  -a, --alg ALG        Algorithm (lzo1x, lzo1y) (default: lzo1x)\n");
        fprintf(stderr, "  -L, -l, --level LEVEL Dictionary size in bits (10-14) (default: 12)\n");
        fprintf(stderr, "  -B N, --block-size N  Fix block size (accepts units: B/KB/MB)\n");
        fprintf(stderr, "  --local N             Set local work-group size for kernels (1,8,64)\n");
        fprintf(stderr, "\nDecompression:\n");
        fprintf(stderr, "  %s -d [--verify ORIG] [-o out_file] input.lzo\n", prog_name);
        fprintf(stderr, "     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        fprintf(stderr, "     - --verify ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  Compress with lzo1y level 12: %s -a lzo1y -L 12 input.dat -o out.lzo\n", prog_name);
        fprintf(stderr, "  Decompress and verify:        %s -d --verify input.dat out.lzo -o out.dec\n", prog_name);
        fprintf(stderr, "  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", prog_name);
        fprintf(stderr, "  %s -h|--help                                 # show this help\n", prog_name);
        fprintf(stderr, "\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
        fprintf(stderr, "    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload). Applies to both compression and decompression.\n");
        fprintf(stderr, "    LZO_MT_IO=0|1            Enable multi-threaded pread reads / parallel uploads.\n");
        fprintf(stderr, "    LZO_MT_IO_THREADS=N      Threads for MT I/O (1-32; common values: 4/8; default: %d ).\n", LZO_DEFAULT_MT_IO_THREADS);
        fprintf(stderr, "    LZO_KERNEL_OPT=1         Enable the optimized instrumented kernel (unrolled vector loop) for testing.\n");
        fprintf(stdout, "    LZO_COALESCE_OUTPUT=0|1    Enable output coalescing by default (default: %d)\n", LZO_DEFAULT_COALESCE_OUTPUT);
        fprintf(stdout, "    LZO_COALESCE_CHUNK_MB=N    Chunk size (MB) for chunked coalesce fallback (default: %d)\n", LZO_DEFAULT_COALESCE_CHUNK_MB);
        fprintf(stdout, "    LZO_COALESCE_MAX_MB=N      Maximum MB allowed for full contiguous coalesce (default: %d)\n", LZO_DEFAULT_COALESCE_MAX_MB);
        fprintf(stdout, "    LZO_STDIO_BUF_MB=N         Default stdio buffer size in MB for fwrite() (default: %d)\n", LZO_DEFAULT_STDIO_BUF_MB);
}

/* Prototypes for extracted helpers to keep main concise */
static int do_decompress_mode(const char* lz_path, const char* output_path, const char* verify_path, int output_explicit, int suppress_non_data);
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int verify_flag, int suppress_non_data, const char* alg_name, int comp_level);

/* Implementations: wrappers that use the shared core backend (lzo_gpu_core.c)
 * The helpers create short-lived OpenCL contexts, load the appropriate kernels
 * and call lzo_compress_core / lzo_decompress_core to perform the heavy lifting.
 */
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int verify_flag, int suppress_non_data, const char* alg_name, int comp_level)
{
    if (!in_path) {
        fprintf(stderr, "error: missing input\n");
        return 1;
    }

    /* Adaptive compression level: if user didn't specify level (-1), choose based on file size and entropy */
    int adaptive_level_used = 0;
    if (comp_level < 0) {
        /* Get file size for adaptive selection */
        FILE* f_check = fopen(in_path, "rb");
        if (!f_check) {
            fprintf(stderr, "error: cannot open input file for size check: %s\n", in_path);
            return 1;
        }
        fseek(f_check, 0, SEEK_END);
        size_t file_size = ftell(f_check);
        fseek(f_check, 0, SEEK_SET);

        /* For entropy calculation, sample first 1MB (or entire file if smaller) */
        size_t sample_size = (file_size < 1024*1024) ? file_size : 1024*1024;
        unsigned char* sample = malloc(sample_size);
        double entropy = 0.0;
        if (sample && fread(sample, 1, sample_size, f_check) == sample_size) {
            entropy = lzo_calc_entropy(sample, sample_size);
        }
        free(sample);
        fclose(f_check);

        /* Call adaptive level selection */
        comp_level = lzo_adaptive_compression_level(file_size, entropy, debug);
        adaptive_level_used = 1;
        if (!suppress_non_data) {
            fprintf(stderr, "[ADAPTIVE] Selected compression level %d (file_size=%zu entropy=%.4f)\n",
                    comp_level, file_size, entropy);
        }
    }

    /* initialize OCL context/queue/device */
    ocl_init(); /* sets global ctx, q, dev */

    /* load compression kernel */
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    int kernel_has_dbg = 0;
    char build_log[8192] = {0};
    int kernel_debug_flag = 1; /* allow debug wrapper when --debug is used */
    if (lzo_load_comp_kernel(ctx, dev, alg_name, comp_level, kernel_debug_flag, kernel_opt, debug, &prog_c, &krn_c, &kernel_has_dbg, build_log, sizeof(build_log)) != 0) {
        if (build_log[0]) fprintf(stderr, "error: failed to load kernel: %s\n", build_log);
        else fprintf(stderr, "error: failed to load kernel for %s bits=%d\n", alg_name, comp_level);
        return 1;
    }

    /* runtime options read from env (standalone semantics) */
    int standard_copy = (getenv("LZO_STANDARD_COPY") && atoi(getenv("LZO_STANDARD_COPY")) == 1) ? 1 : 0;
    int mt_io = (getenv("LZO_MT_IO") && atoi(getenv("LZO_MT_IO")) == 1) ? 1 : 0;
    int mt_threads = LZO_DEFAULT_MT_IO_THREADS;
    const char *s = getenv("LZO_MT_IO_THREADS"); if (s && atoi(s) > 0) mt_threads = atoi(s);
    int fixed_block_kb = (g_cli_fixed_block_bytes > 0) ? (int)(g_cli_fixed_block_bytes / 1024) : 0;

    int coalesce_output = LZO_DEFAULT_COALESCE_OUTPUT;
    const char *t = getenv("LZO_COALESCE_OUTPUT"); if (t) coalesce_output = atoi(t) ? 1 : 0;
    int coalesce_chunk_mb = LZO_DEFAULT_COALESCE_CHUNK_MB; t = getenv("LZO_COALESCE_CHUNK_MB"); if (t && atoi(t) > 0) coalesce_chunk_mb = atoi(t);
    int coalesce_max_mb = LZO_DEFAULT_COALESCE_MAX_MB; t = getenv("LZO_COALESCE_MAX_MB"); if (t && atoi(t) > 0) coalesce_max_mb = atoi(t);
    int stdio_buf_mb = LZO_DEFAULT_STDIO_BUF_MB; t = getenv("LZO_STDIO_BUF_MB"); if (t && atoi(t) > 0) stdio_buf_mb = atoi(t);

    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;

    unsigned long time_us = 0;
    size_t output_size = 0;
    timing_t t_out = {0};

    /* Create parameter object */
    lzo_compress_params_t params = {
        .level = comp_level,
        .alg_id = alg_id,
        .standard_copy = standard_copy,
        .mt_io = mt_io,
        .mt_threads = mt_threads,
        .fixed_block_kb = fixed_block_kb,
        .coalesce_output = coalesce_output,
        .coalesce_chunk_mb = coalesce_chunk_mb,
        .coalesce_max_mb = coalesce_max_mb,
        .stdio_buf_mb = stdio_buf_mb,
        .local_size_param = (int)g_cli_local_size,
        .debug = debug
    };

    int ret = lzo_compress_core(ctx, q, dev, krn_c, in_path, output_path ? output_path : "",
                                &params, &time_us, &output_size, &t_out);

    /* cleanup compiled program and kernel */
    if (krn_c) clReleaseKernel(krn_c);
    if (prog_c) clReleaseProgram(prog_c);
    /* release short-lived context/queue */
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }

    if (ret != 0) return ret;

    /* Optional verify: decompress to temp file and compare with input */
    if (verify_flag) {
        if (!output_path || !output_path[0]) {
            fprintf(stderr, "verify: requires -o option to specify compressed output\n");
            return 1;
        }

        /* Create temp file for decompressed output */
        char temp_dec[512];
        snprintf(temp_dec, sizeof(temp_dec), "/tmp/lzo_verify_dec_%d.raw", getpid());

        /* Initialize OCL for decompression */
        ocl_init();

        /* Load decompression kernel */
        FILE* f = fopen(output_path, "rb");
        if (!f) {
            fprintf(stderr, "verify: cannot open compressed file %s\n", output_path);
            if (q) { clReleaseCommandQueue(q); q = NULL; }
            if (ctx) { clReleaseContext(ctx); ctx = NULL; }
            return 1;
        }
        uint16_t magic; fread(&magic, 2, 1, f);
        uint32_t tmp_u32; fread(&tmp_u32, 4, 1, f);  /* orig_sz */
        fread(&tmp_u32, 4, 1, f);  /* blk_sz */
        fread(&tmp_u32, 4, 1, f);  /* nblk */
        uint32_t hdr_alg_id; fread(&hdr_alg_id, 4, 1, f);
        fclose(f);

        const char* decomp_alg = (hdr_alg_id == 1) ? "lzo1y" : "lzo1x";
        char decomp_base[64]; snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", decomp_alg);

        cl_program prog_d = load_prog_with_dbits(decomp_base, 0);
        if (!prog_d) {
            fprintf(stderr, "verify: unable to load decompressor\n");
            if (q) { clReleaseCommandQueue(q); q = NULL; }
            if (ctx) { clReleaseContext(ctx); ctx = NULL; }
            return 1;
        }

        char krn_name[64]; snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", decomp_alg);
        cl_int err_krn; cl_kernel krn_d = clCreateKernel(prog_d, krn_name, &err_krn);
        if (err_krn != CL_SUCCESS) {
            fprintf(stderr, "verify: clCreateKernel failed\n");
            clReleaseProgram(prog_d);
            if (q) { clReleaseCommandQueue(q); q = NULL; }
            if (ctx) { clReleaseContext(ctx); ctx = NULL; }
            return 1;
        }

        /* Decompress to temp file */
        int standard_copy = (getenv("LZO_STANDARD_COPY") && atoi(getenv("LZO_STANDARD_COPY")) == 1) ? 1 : 0;
        unsigned long verify_time = 0;
        size_t verify_size = 0;
        timing_t verify_t = {0};
        int verify_rc = lzo_decompress_core(ctx, q, dev, krn_d, output_path, temp_dec,
                                            standard_copy, (int)g_cli_local_size, 0,
                                            &verify_time, &verify_size, &verify_t);

        clReleaseKernel(krn_d);
        clReleaseProgram(prog_d);
        if (q) { clReleaseCommandQueue(q); q = NULL; }
        if (ctx) { clReleaseContext(ctx); ctx = NULL; }

        if (verify_rc != 0) {
            fprintf(stderr, "verify: decompression failed\n");
            unlink(temp_dec);
            return 1;
        }

        /* Compare files */
        size_t s1 = 0, s2 = 0;
        unsigned char *b1 = (unsigned char*)lzo_read_file(in_path, &s1);
        unsigned char *b2 = (unsigned char*)lzo_read_file(temp_dec, &s2);

        int verify_ok = (b1 && b2 && s1 == s2 && memcmp(b1, b2, s1) == 0);

        if (b1) free(b1);
        if (b2) free(b2);
        unlink(temp_dec);

        if (verify_ok) {
            printf("verify OK\n");
        } else {
            fprintf(stderr, "verify FAILED (orig=%zu decomp=%zu)\n", s1, s2);
            return 1;
        }
    }

    return 0;
}

static int do_decompress_mode(const char* lz_path, const char* output_path, const char* verify_path, int output_explicit, int suppress_non_data)
{
    if (!lz_path) { fprintf(stderr, "error: missing input .lzo\n"); return 1; }

    /* Read zero-copy configuration (default is zero-copy, same as compression) */
    int standard_copy = (getenv("LZO_STANDARD_COPY") && atoi(getenv("LZO_STANDARD_COPY")) == 1) ? 1 : 0;

    /* Peek header to detect algorithm */
    FILE* f = fopen(lz_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    uint16_t magic; if (fread(&magic, sizeof(magic), 1, f) != 1) { perror("fread magic"); fclose(f); return 1; }
    if (magic != 0x4C5A) { fprintf(stderr, "[DECOMP] wrong file magic\n"); fclose(f); return 1; }
    uint32_t orig_sz, blk_sz, nblk, alg_id; if (fread(&orig_sz, sizeof(orig_sz), 1, f) != 1 || fread(&blk_sz, sizeof(blk_sz), 1, f) != 1 || fread(&nblk, sizeof(nblk), 1, f) != 1 || fread(&alg_id, sizeof(alg_id), 1, f) != 1) { perror("fread header"); fclose(f); return 1; }
    fclose(f);

    const char* alg_name = (alg_id == 1) ? "lzo1y" : "lzo1x";
    char decomp_base[64]; snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);

    ocl_init(); /* create ctx/q/dev */

    cl_program prog_d = load_prog_with_dbits(decomp_base, 0);
    if (!prog_d) { fprintf(stderr, "error: unable to load decompressor for %s\n", decomp_base); return 1; }
    char krn_name[64]; snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", alg_name);
    cl_int err; cl_kernel krn_d = clCreateKernel(prog_d, krn_name, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); clReleaseProgram(prog_d); return 1; }

    unsigned long time_us = 0; size_t output_size = 0; timing_t t_out = {0};
    int rc = lzo_decompress_core(ctx, q, dev, krn_d, lz_path, output_path ? output_path : "", standard_copy, (int)g_cli_local_size, debug, &time_us, &output_size, &t_out);

    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }

    if (rc != 0) return rc;

    /* Verify: compare decompressed output with reference file */
    if (verify_path && verify_path[0] != '\0') {
        /* Read reference file */
        size_t s1=0;
        unsigned char *b1 = (unsigned char*)lzo_read_file(verify_path, &s1);
        if (!b1) {
            fprintf(stderr, "verify: failed to read reference file %s\n", verify_path);
            return 1;
        }

        /* Determine actual output path (may need temp file if output_path is empty) */
        const char *actual_output = output_path;
        char temp_output[512];
        int need_cleanup = 0;

        if (!output_path || output_path[0] == '\0' || !output_explicit) {
            /* No output file specified, need to decompress to temp file for verification */
            snprintf(temp_output, sizeof(temp_output), "/tmp/lzo_verify_decomp_%d.raw", getpid());
            actual_output = temp_output;
            need_cleanup = 1;

            /* Re-initialize and decompress to temp file */
            ocl_init();
            cl_program prog_d2 = load_prog_with_dbits(decomp_base, 0);
            if (!prog_d2) {
                fprintf(stderr, "verify: unable to re-load decompressor\n");
                free(b1);
                return 1;
            }
            cl_kernel krn_d2 = clCreateKernel(prog_d2, krn_name, &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "verify: clCreateKernel failed\n");
                clReleaseProgram(prog_d2);
                free(b1);
                return 1;
            }

            unsigned long verify_time = 0;
            size_t verify_size = 0;
            timing_t verify_t = {0};
            int verify_rc = lzo_decompress_core(ctx, q, dev, krn_d2, lz_path, actual_output,
                                                standard_copy, (int)g_cli_local_size, 0,
                                                &verify_time, &verify_size, &verify_t);

            clReleaseKernel(krn_d2);
            clReleaseProgram(prog_d2);
            if (q) { clReleaseCommandQueue(q); q = NULL; }
            if (ctx) { clReleaseContext(ctx); ctx = NULL; }

            if (verify_rc != 0) {
                fprintf(stderr, "verify: decompression to temp file failed\n");
                free(b1);
                if (need_cleanup) unlink(actual_output);
                return 1;
            }
        }

        /* Read decompressed output */
        size_t s2=0;
        unsigned char *b2 = (unsigned char*)lzo_read_file(actual_output, &s2);
        if (!b2) {
            fprintf(stderr, "verify: failed to read decompressed output %s\n", actual_output);
            free(b1);
            if (need_cleanup) unlink(actual_output);
            return 1;
        }

        /* Compare */
        int verify_ok = (s1 == s2) && (memcmp(b1, b2, s1) == 0);
        free(b1);
        free(b2);

        if (need_cleanup) unlink(actual_output);

        if (verify_ok) {
            printf("verify OK\n");
        } else {
            fprintf(stderr, "verify FAILED (size: ref=%zu decomp=%zu)\n", s1, s2);
            return 1;
        }
    }

    return 0;
}

/* Implementations will call into lzo_gpu_core.c which provides lzo_compress_core/lzo_decompress_core */
#include "lzo_gpu_core.h"


int main(int argc, char** argv)
{
    if (argc < 2) {
        /* Print the detailed help when no args are provided to keep output
         * consistent with --help behaviour (avoid the short, outdated usage)
         */
        argc = 1;
        show_help(argv[0]);
        return 0;
    }

    int verify_flag = 0; /* only when set, do roundtrip/verify prints */
    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    // const char *orig_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int comp_level = -1; /* -1 means adaptive (not explicitly specified by user) */
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
        if (strcmp(arg, "--debug") == 0) { debug = 1; continue; }

        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing argument for %s\n", arg);
                return 1;
            }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "-l") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing argument for %s\n", arg);
                return 1;
            }
            comp_level = atoi(argv[++i]);
            if (comp_level < 10 || comp_level > 14) {
                fprintf(stderr, "error: dictionary size must be between 10 and 14 bits (got %d)\n", comp_level);
                return 1;
            }
            continue;
        }
        if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing argument for %s\n", arg);
                return 1;
            }
            alg_name = argv[++i];
            if (strcmp(alg_name, "1x") == 0 || strcmp(alg_name, "lzo1x") == 0)
                alg_name = "lzo1x";
            else if (strcmp(alg_name, "1y") == 0 || strcmp(alg_name, "lzo1y") == 0)
                alg_name = "lzo1y";
            else {
                fprintf(stderr, "错误: 未知算法 '%s'. 支持: lzo1x, lzo1y (或 1x/1y)\n", alg_name);
                return 1;
            }
            continue;
        }
        if (strcmp(arg, "-c") == 0) {
            /* -c means compress mode (default, explicit flag for clarity) */
            decompress_mode = 0;
            continue;
        }
        if (strcmp(arg, "--verify") == 0) {
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
            const char* s = arg + 3;
            g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s);
            size_t b = lzo_parse_block_size(s);
            if (b > 0) g_cli_fixed_block_bytes = b;
            continue;
        }
        if (strcmp(arg, "-B") == 0) { if (i + 1 < argc) { const char* s = argv[++i]; g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s); size_t b = lzo_parse_block_size(s); if (b > 0) g_cli_fixed_block_bytes = b; } continue; }
        if (strncmp(arg, "--block-size=", 13) == 0) { const char* s = arg + 13; g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s); size_t b = lzo_parse_block_size(s); if (b > 0) g_cli_fixed_block_bytes = b; continue; }
        if (strcmp(arg, "--block-size") == 0) { if (i + 1 < argc) { const char* s = argv[++i]; g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s); size_t b = lzo_parse_block_size(s); if (b > 0) g_cli_fixed_block_bytes = b; } continue; }
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

    /* kernel_opt is controlled via environment variable only (testing only) */
    {
        const char* env_kopt = getenv("LZO_KERNEL_OPT");
        if (env_kopt && atoi(env_kopt) != 0) {
            kernel_opt = 1;
            if (debug) fprintf(stderr, "DBG: kernel_opt enabled via LZO_KERNEL_OPT\n");
        }
    }

    /* Decompress mode */
    if (decompress_mode) {
        return do_decompress_mode(lz_path, output_path, verify_path, output_explicit, suppress_non_data);
    }

    /* Compress path (simple, fast) */
    return do_compress_mode(in_path, output_path, output_explicit, verify_flag, suppress_non_data, alg_name, comp_level);
}

