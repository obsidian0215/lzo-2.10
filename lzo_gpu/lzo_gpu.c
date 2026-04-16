#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#endif
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <io.h>
#define access _access
#define F_OK 0
#define strcasecmp _stricmp
#else
#include <strings.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <libgen.h>
#include <linux/limits.h>
#include <signal.h>
#endif
#include "lzo_defaults.h"
#include "timing.h"
#include "lzo_gpu_utils.h"
#include "lzo_gpu_core.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define now_ns lzo_now_ns

#if defined(_WIN32) || defined(_WIN64)
static int run_lzo_daemon(int argc, char** argv) {
    (void)argc; (void)argv;
    fprintf(stderr, "Daemon mode is not supported in Windows builds. Use standalone or bench mode.\n");
    return 1;
}
static int run_lzo_client(int argc, char** argv) {
    (void)argc; (void)argv;
    fprintf(stderr, "--use-daemon is not supported in Windows builds. Use standalone mode.\n");
    return 1;
}
#else
int run_lzo_daemon(int argc, char** argv);
int run_lzo_client(int argc, char** argv);
#endif

/* Helper for multi-threaded pread into a destination buffer.
* 压缩文件格式：
uint16  magic     = 0x4C5A   // 'L''Z'
uint32  orig_size               (≤4 GiB)
uint32  blk_size
uint32  nblk
uint32  len[nblk]               // 每块压缩长度
-----   nblk 个压缩块数据
*/
#define MAGIC  0x4C5A   /* 'L''Z' */

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

static int lzo_device_host_unified_memory(cl_device_id device)
{
    cl_bool unified = CL_FALSE;
    if (!device) return 1;
    if (clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL) != CL_SUCCESS) {
        return 1;
    }
    return unified == CL_TRUE;
}

static int lzo_resolve_standard_copy(cl_device_id device)
{
    const char* env = getenv("LZO_STANDARD_COPY");
    if (env && *env) return atoi(env) == 1;
    return lzo_device_host_unified_memory(device) ? 0 : 1;
}

static cl_device_type preferred_opencl_device_type(void)
{
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

static int lzo_ocl_debug_enabled(void)
{
    return 0;
}

static int lzo_queue_profiling_enabled(void)
{
    return 1;
}

static cl_int lzo_try_get_device(cl_platform_id *platforms, cl_uint num_platforms, cl_device_type dtype, cl_device_id *out_dev, cl_platform_id *out_pf)
{
    for (cl_uint pi = 0; pi < num_platforms; pi++) {
        cl_device_id tmp_dev = NULL;
        cl_int r = clGetDeviceIDs(platforms[pi], dtype, 1, &tmp_dev, NULL);
        if (r == CL_SUCCESS && tmp_dev != NULL) {
            *out_dev = tmp_dev;
            *out_pf = platforms[pi];
            return CL_SUCCESS;
        }
    }
    return CL_DEVICE_NOT_FOUND;
}

static void ocl_init(void)
{
    uint64_t t1 = now_ns();
    cl_int err;
    cl_device_type pref_type = preferred_opencl_device_type();

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        ctx = NULL;
        q = NULL;
        return;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) {
        fprintf(stderr, "OpenCL init failed: malloc platforms\n");
        ctx = NULL;
        q = NULL;
        return;
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        free(platforms);
        ctx = NULL;
        q = NULL;
        return;
    }

    dev = NULL;
    cl_platform_id selected_pf = NULL;

    cl_int r = CL_DEVICE_NOT_FOUND;
    if (pref_type == CL_DEVICE_TYPE_GPU) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
    } else if (pref_type == CL_DEVICE_TYPE_CPU) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
    } else if (pref_type == CL_DEVICE_TYPE_DEFAULT) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
    } else { /* CL_DEVICE_TYPE_ALL */
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
    }

    free(platforms);

    if (dev == NULL) {
        fprintf(stderr, "OpenCL init failed: clGetDeviceIDs failed for all type/plat combos\n");
        ctx = NULL;
        q = NULL;
        return;
    }

    if (dev == NULL) {
        fprintf(stderr, "OpenCL init failed: clGetDeviceIDs failed for all type/plat combos\n");
        ctx = NULL;
        q = NULL;
        return;
    }

    if (lzo_ocl_debug_enabled()) {
        char pfname[256] = {0};
        char devname[256] = {0};
        cl_device_type devtype = 0;
        clGetPlatformInfo(selected_pf, CL_PLATFORM_NAME, sizeof(pfname), pfname, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devname), devname, NULL);
        clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
        fprintf(stderr, "[OpenCL DEBUG] Selected platform=%s, device=%s (type=%s)\n",
                pfname,
                devname,
                (devtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                (devtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
                (devtype & CL_DEVICE_TYPE_DEFAULT) ? "DEFAULT" : "UNKNOWN");
    }

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || ctx == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateContext err=%d\n", err);
        ctx = NULL;
        q = NULL;
        return;
    }
    cl_queue_properties profiling_flag = lzo_queue_profiling_enabled() ? CL_QUEUE_PROFILING_ENABLE : 0;
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, profiling_flag, 0
    };
    q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
    if (err != CL_SUCCESS || q == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateCommandQueueWithProperties err=%d\n", err);
        if (ctx) clReleaseContext(ctx);
        ctx = NULL;
        q = NULL;
        return;
    }
    uint64_t t2 = now_ns();
    g_ocl_init_us = (unsigned long)((t2 - t1) / 1000);
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
static inline void show_help(char *prog_name)
{
    fprintf(stderr, "Unified LZO GPU Tool\n");
    fprintf(stderr, "Usage Modes:\n");
    fprintf(stderr, "  1. Standalone:   %s [options] <input_file|->\n", prog_name);
    fprintf(stderr, "  2. Run Daemon:   %s --daemon [options]\n", prog_name);
    fprintf(stderr, "  3. Use Daemon:   %s --use-daemon [options] <input_file>\n", prog_name);
    fprintf(stderr, "  4. Stop Daemon:  %s --stop-daemon\n", prog_name);

    fprintf(stderr, "\nBasic Options:\n");
    fprintf(stderr, "  -c                   Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress     Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE    Output file (use '-' for stdout)\n");
    fprintf(stderr, "  -a, --alg ALG        Algorithm (lzo1x, lzo1y) (default: lzo1x)\n");
    fprintf(stderr, "  -L, --level LEVEL    Compression level: 11-15=D_BITS (default: 14), 99=enhanced greedy, 999=optimal (SWD)\n");
    fprintf(stderr, "  -B, --block-size N   Block size (B/KB/MB) (default: 64KB)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
    fprintf(stderr, "  --local N            Local work-group size (default: 1)\n");
    fprintf(stderr, "  --bench [N]          Stable benchmark (compress+decompress+verify), optional N seconds (default: 3)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming:\n");
    fprintf(stderr, "  input '-'            Read input from stdin (standalone mode only)\n");
    fprintf(stderr, "  output '-'           Write output to stdout (standalone mode only)\n");
    fprintf(stderr, "Detailed Help:\n");
    fprintf(stderr, "  %s --daemon -h           # Daemon-specific settings\n", prog_name);
    fprintf(stderr, "  %s --use-daemon -h       # Client-specific settings\n\n", prog_name);

    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  Compress:         %s input.dat -o out.lzo\n", prog_name);
    fprintf(stderr, "  Decompress:       %s -d out.lzo -o out.dec\n", prog_name);
    fprintf(stderr, "  Using Daemon:     %s --use-daemon -a lzo1y bigfile.bin\n", prog_name);
    fprintf(stderr, "  %s -h|--help                                 # show this help\n", prog_name);
    fprintf(stderr, "\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
    fprintf(stderr, "    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload). Applies to both compression and decompression.\n");
    fprintf(stderr, "    LZO_GPU_ENABLE_COMPACTION=0|1  Enable device compaction/pack path (default: 0).\n");
    fprintf(stderr, "    LZO_GPU_FORCE_COMPACTION=0|1   Force off/on device compaction, overrides adaptive gate.\n");
    fprintf(stderr, "    LZO_PIPELINE_ENABLE=0|1        Enable chunked pipeline compression path (default: 0).\n");
    fprintf(stderr, "    LZO_PIPELINE_OVERLAP_ENABLE=0|1 Enable upload/compute overlap on standard-copy pipeline path (default: 0).\n");
    fprintf(stderr, "    LZO_PIPELINE_THRESHOLD_MB=N    Min input size to enable pipeline when on (default: 64).\n");
    fprintf(stderr, "    LZO_PIPELINE_CHUNK_BLOCKS=N    Pipeline chunk blocks (default: 512).\n");
    fprintf(stderr, "    LZO_PIPELINE_ENTROPY_ENABLE=0|1  Enable entropy gate for pipeline.\n");
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_double(const double *vals, size_t n) {
    if (!vals || n == 0) return 0.0;
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return 0.0;
    memcpy(tmp, vals, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_asc);
    double out = (n % 2 == 0) ? (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0 : tmp[n / 2];
    free(tmp);
    return out;
}

static double event_elapsed_us(cl_event ev) {
    cl_ulong st = 0, en = 0;
    if (!ev) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return 0.0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return 0.0;
    if (en <= st) return 0.0;
    return (double)(en - st) / 1000.0;
}

static double elapsed_sec(const struct timespec *start, const struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int lzo_debug_counters_enabled_cli(void) {
    return 0;
}

static int path_is_dash(const char* path) {
    return path && strcmp(path, "-") == 0;
}

static int create_temp_path(char* path_buf, size_t path_buf_size, const char* templ) {
    if (!path_buf || path_buf_size == 0 || !templ) return -1;
#if defined(_WIN32) || defined(_WIN64)
    (void)path_buf_size;
    (void)templ;
    return -1;
#else
    {
        int fd;
        size_t n = strlen(templ);
        if (n + 1 > path_buf_size) return -1;
        memcpy(path_buf, templ, n + 1);
        fd = mkstemp(path_buf);
        if (fd < 0) return -1;
        close(fd);
        unlink(path_buf);
        return 0;
    }
#endif
}

static int copy_stream_to_path(FILE* in, const char* path) {
    FILE* out;
    unsigned char buf[1 << 20];
    size_t nread;

    if (!in || !path) return -1;
    out = fopen(path, "wb");
    if (!out) return -1;

    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) {
            fclose(out);
            return -1;
        }
    }
    if (ferror(in)) {
        fclose(out);
        return -1;
    }
    if (fclose(out) != 0) return -1;
    return 0;
}

static int copy_path_to_stream(const char* path, FILE* out) {
    FILE* in;
    unsigned char buf[1 << 20];
    size_t nread;

    if (!path || !out) return -1;
    in = fopen(path, "rb");
    if (!in) return -1;

    while ((nread = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, nread, out) != nread) {
            fclose(in);
            return -1;
        }
    }
    if (ferror(in)) {
        fclose(in);
        return -1;
    }

    if (fclose(in) != 0) return -1;
    if (fflush(out) != 0) return -1;
    return 0;
}


/* Prototypes for extracted helpers to keep main concise */
static int do_decompress_mode(const char* lz_path, const char* output_path, int output_explicit, int suppress_non_data);
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int suppress_non_data, const char* alg_name, int comp_level);

/* Implementations: wrappers that use the shared core backend (lzo_gpu_core.c)
 * The helpers create short-lived OpenCL contexts, load the appropriate kernels
 * and call lzo_compress_core / lzo_decompress_core to perform the heavy lifting.
 */
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int suppress_non_data, const char* alg_name, int comp_level)
{
    (void)output_explicit;
    if (!in_path) {
        fprintf(stderr, "error: missing input\n");
        return 1;
    }

    if (comp_level < 0) comp_level = LZO_DEFAULT_COMP_LEVEL;

    uint64_t t_total_start = now_ns();

    ocl_init();
    if (!ctx || !q) {
        fprintf(stderr, "error: failed to initialize OpenCL runtime\n");
        return 1;
    }

    /* load compression kernel */
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    cl_kernel pack_krn = NULL;
    cl_int err = CL_SUCCESS;
    char build_log[8192] = {0};
    int kernel_has_dbg = 0;
    int debug_counters = lzo_debug_counters_enabled_cli();
    uint64_t tk1 = now_ns();
    if (lzo_load_comp_kernel(ctx, dev, alg_name, comp_level, debug_counters, &prog_c, &krn_c, &kernel_has_dbg, build_log, sizeof(build_log)) != 0) {
        if (build_log[0]) fprintf(stderr, "error: failed to load kernel for %s bits=%d: %s\n", alg_name, comp_level, build_log);
        else fprintf(stderr, "error: failed to load kernel for %s bits=%d\n", alg_name, comp_level);
        return 1;
    }
    uint64_t tk2 = now_ns();
    g_kernel_load_us = (unsigned long)((tk2 - tk1) / 1000);

    err = CL_SUCCESS;
    pack_krn = clCreateKernel(prog_c, "lzo_pack_compressed_blocks", &err);
    if (err != CL_SUCCESS || !pack_krn) pack_krn = NULL;

    int standard_copy = lzo_resolve_standard_copy(dev);
    size_t block_size = g_cli_fixed_block_bytes;
    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;

    lzo_gpu_workspace_t ws;
    lzo_gpu_workspace_init(&ws);

    /* Create parameter object */
    lzo_compress_params_t params = {
        .level = comp_level,
        .alg_id = alg_id,
        .standard_copy = standard_copy,
        .block_size = block_size,
        .local_size_param = (int)g_cli_local_size,
        .debug = debug_counters
    };

    unsigned long time_us = 0;
    size_t output_size = 0;
    timing_t t_out = {0};

    int ret = lzo_compress_core(ctx, q, dev, krn_c, pack_krn, in_path, output_path, &params, 0, &ws, &time_us, &output_size, &t_out);
    uint64_t t_total_end = now_ns();
    unsigned long overall_us = (unsigned long)((t_total_end - t_total_start) / 1000ULL);

    if (ret == 0) {
        FILE* msg = suppress_non_data ? stderr : stdout;
        if (g_verbose && !suppress_non_data) {
            response_t r = {0};
            r.status = 0;
            r.time_us = overall_us;
            r.out_size = output_size;
            r.timing = t_out;
            lzo_print_response_stats(&r, in_path, 'C', alg_id);
        } else {
            double ratio = (double)t_out.in_size / (t_out.out_size > 0 ? t_out.out_size : 1);
            fprintf(msg,
                    "%s : %zu -> %zu (%.2f:1) in %.2f ms\n",
                    in_path,
                    (size_t)t_out.in_size,
                    (size_t)t_out.out_size,
                    ratio,
                    overall_us / 1000.0);
        }
    }
    if (pack_krn) clReleaseKernel(pack_krn);
    if (krn_c) clReleaseKernel(krn_c);
    if (prog_c) clReleaseProgram(prog_c);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return ret;
}

static int do_decompress_mode(const char* lz_path, const char* output_path, int output_explicit, int suppress_non_data)
{
    (void)output_explicit;
    if (!lz_path) { fprintf(stderr, "error: missing input .lzo\n"); return 1; }
    int standard_copy = 0;

    FILE* f = fopen(lz_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    uint16_t magic; fread(&magic, 2, 1, f);
    if (magic != 0x4C5A) { fprintf(stderr, "error: magic mismatch\n"); fclose(f); return 1; }
    uint32_t a[4]; fread(a, 4, 4, f); // orig_sz, blk_sz, nblk, alg_id
    fclose(f);

    const char* alg_name = (a[3] == 1) ? "lzo1y" : "lzo1x";
    int debug_counters = lzo_debug_counters_enabled_cli();
    char decomp_base[64];
    if (debug_counters)
        snprintf(decomp_base, sizeof(decomp_base), "%s_decomp_debug", alg_name);
    else
        snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);

    uint64_t t_total_start = now_ns();

    ocl_init();
    if (!ctx || !q) {
        fprintf(stderr, "error: failed to initialize OpenCL runtime\n");
        return 1;
    }
    standard_copy = lzo_resolve_standard_copy(dev);
    cl_program prog_d = NULL;
    cl_int err;
    char build_log[8192] = {0};
    const int decomp_bits = LZO_DEFAULT_COMP_LEVEL;

    uint64_t tk1 = now_ns();
    /* Fallback to base decompressor */
    if (!prog_d) {
        prog_d = lzo_load_program_with_dbits(ctx, dev, decomp_base, decomp_bits, build_log, sizeof(build_log));
        if (!prog_d) {
            if (build_log[0]) {
                fprintf(stderr, "error: unable to load decompressor for %s (D_BITS=%d): %s\n", decomp_base, decomp_bits, build_log);
            } else {
                fprintf(stderr, "error: unable to load decompressor for %s (D_BITS=%d)\n", decomp_base, decomp_bits);
            }
            return 1;
        }
    }

    char krn_name[64];
    cl_kernel krn_d = NULL;

    snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", alg_name);
    krn_d = clCreateKernel(prog_d, krn_name, &err);
    if (err != CL_SUCCESS || !krn_d) {
        fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err);
        clReleaseProgram(prog_d);
        return 1;
    }

    uint64_t tk2 = now_ns();
    g_kernel_load_us = (unsigned long)((tk2 - tk1) / 1000);

    lzo_gpu_workspace_t ws;
    lzo_gpu_workspace_init(&ws);
    unsigned long time_us = 0; size_t output_size = 0; timing_t t_out = {0};

    int rc = lzo_decompress_core(ctx, q, dev, krn_d, lz_path, output_path, &ws, standard_copy, (int)g_cli_local_size, debug_counters, &time_us, &output_size, &t_out);
    uint64_t t_total_end = now_ns();
    unsigned long overall_us = (unsigned long)((t_total_end - t_total_start) / 1000ULL);
    if (rc == 0) {
        FILE* msg = suppress_non_data ? stderr : stdout;
        if (g_verbose && !suppress_non_data) {
            response_t r = {0};
            r.status = 0;
            r.time_us = overall_us;
            r.out_size = output_size;
            r.timing = t_out;
            lzo_print_response_stats(&r, lz_path, 'D', (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0);
        } else {
            fprintf(msg,
                    "%s : %zu -> %zu in %.2f ms\n",
                    lz_path,
                    (size_t)t_out.in_size,
                    (size_t)t_out.out_size,
                    overall_us / 1000.0);
        }
    }
    lzo_gpu_workspace_free(&ws);

    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return rc;
}

static int run_lzo_bench(const char *in_path,
                         const char *alg_name,
                         int comp_level,
                         double bench_seconds) {
    struct stat st;
    if (!in_path || stat(in_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "bench error: invalid input file\n");
        return 1;
    }
    if (bench_seconds <= 0.0) bench_seconds = 3.0;
    if (comp_level < 0) comp_level = LZO_DEFAULT_COMP_LEVEL;

    size_t in_size_ref = 0;
    unsigned char* input_ref = (unsigned char*)lzo_read_file(in_path, &in_size_ref);
    if (!input_ref || in_size_ref == 0) {
        free(input_ref);
        fprintf(stderr, "bench error: failed to read input\n");
        return 1;
    }

    int standard_copy = 0;
    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;
    int debug_counters = lzo_debug_counters_enabled_cli();

    ocl_init();
    if (!ctx || !q) {
        fprintf(stderr, "bench error: failed to initialize OpenCL runtime\n");
        free(input_ref);
        return 1;
    }
    standard_copy = lzo_resolve_standard_copy(dev);

    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    cl_kernel pack_krn = NULL;
    cl_int err = CL_SUCCESS;
    int kernel_has_dbg = 0;
    char build_log[8192] = {0};
    if (lzo_load_comp_kernel(ctx, dev, alg_name, comp_level, debug_counters,
                             &prog_c, &krn_c, &kernel_has_dbg,
                             build_log, sizeof(build_log)) != 0 || !prog_c || !krn_c) {
        fprintf(stderr, "bench error: failed to load compression kernel\n");
        if (prog_c) clReleaseProgram(prog_c);
        if (q) { clReleaseCommandQueue(q); q = NULL; }
        if (ctx) { clReleaseContext(ctx); ctx = NULL; }
        free(input_ref);
        return 1;
    }

    err = CL_SUCCESS;
    pack_krn = clCreateKernel(prog_c, "lzo_pack_compressed_blocks", &err);
    if (err != CL_SUCCESS || !pack_krn) pack_krn = NULL;

    char decomp_base[64];
    /* For decompression path, use decomp-only marker to skip unnecessary comp-kernel validation. */
    if (debug_counters && comp_level != 999 && comp_level != 99)
        snprintf(decomp_base, sizeof(decomp_base), "%s_decomp_debug", alg_name);
    else
        snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);
    int decomp_bits = (comp_level == 999 || comp_level == 99) ? LZO_DEFAULT_COMP_LEVEL : comp_level;
    cl_program prog_d = lzo_load_program_with_dbits(ctx, dev, decomp_base, decomp_bits, build_log, sizeof(build_log));
    if (!prog_d) {
        if (build_log[0]) {
            fprintf(stderr, "bench error: failed to load decompressor program (D_BITS=%d): %s\n", comp_level, build_log);
        } else {
            fprintf(stderr, "bench error: failed to load decompressor program (D_BITS=%d)\n", comp_level);
        }
        clReleaseKernel(krn_c);
        clReleaseProgram(prog_c);
        if (q) { clReleaseCommandQueue(q); q = NULL; }
        if (ctx) { clReleaseContext(ctx); ctx = NULL; }
        free(input_ref);
        return 1;
    }

    char krn_name[64];
    cl_kernel krn_d = NULL;

    snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", alg_name);
    krn_d = clCreateKernel(prog_d, krn_name, &err);
    if (err != CL_SUCCESS || !krn_d) {
        fprintf(stderr, "bench error: failed to create decompressor kernel (%s, err=%d)\n", krn_name, err);
        clReleaseProgram(prog_d);
        clReleaseKernel(krn_c);
        clReleaseProgram(prog_c);
        if (q) { clReleaseCommandQueue(q); q = NULL; }
        if (ctx) { clReleaseContext(ctx); ctx = NULL; }
        free(input_ref);
        return 1;
    }

    cl_uint krn_num_args = 0;
    int kernel_has_dbg_dec = 0;
    if (clGetKernelInfo(krn_d, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg_dec = (krn_num_args >= 10U);
    }

    lzo_gpu_workspace_t ws;
    lzo_gpu_workspace_init(&ws);

    lzo_compress_params_t params = {
        .level = comp_level,
        .alg_id = alg_id,
        .standard_copy = standard_copy,
        .block_size = g_cli_fixed_block_bytes,
        .local_size_param = (int)g_cli_local_size,
        .debug = debug_counters
    };
    size_t bench_cached_blk = 0;
    size_t bench_cached_nblk = 0;

    size_t cap = 16, n = 0;
    const size_t bench_drop_iterations = 1;
    size_t total_successful_iterations = 0;
    double *comp_tp = (double *)malloc(cap * sizeof(double));
    double *dec_tp = (double *)malloc(cap * sizeof(double));
    double *ratio_pct = (double *)malloc(cap * sizeof(double));
    int verify_ok = 1;
    if (!comp_tp || !dec_tp || !ratio_pct) {
        verify_ok = 0;
    }

    if (verify_ok) {
        cl_int prep_err = CL_SUCCESS;
        size_t blk_bytes = (params.block_size > 0) ? params.block_size : 0;
        lzo_choose_blocking_adaptive(input_ref,
                                     in_size_ref,
                                     dev,
                                     blk_bytes,
                                     0,
                                     &bench_cached_blk,
                                     &bench_cached_nblk,
                                     params.debug);
        if (bench_cached_blk == 0 || bench_cached_nblk == 0) {
            verify_ok = 0;
        }
        if (verify_ok) {
            if (ws.d_in) {
                clReleaseMemObject(ws.d_in);
                ws.d_in = NULL;
                ws.in_size = 0;
            }
            ws.d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_size_ref, NULL, &prep_err);
            if (prep_err != CL_SUCCESS || !ws.d_in) {
                verify_ok = 0;
            }
        }
        if (verify_ok) {
            prep_err = clEnqueueWriteBuffer(q,
                                            ws.d_in,
                                            CL_TRUE,
                                            0,
                                            in_size_ref,
                                            input_ref,
                                            0,
                                            NULL,
                                            NULL);
            if (prep_err != CL_SUCCESS) {
                verify_ok = 0;
            }
        }
        if (verify_ok) {
            ws.in_size = in_size_ref;
            ws.comp_cached_input_size = in_size_ref;
            ws.comp_cached_blk_size = bench_cached_blk;
            ws.comp_cached_nblk = bench_cached_nblk;
        } else {
            fprintf(stderr, "bench error: failed to preload input buffer\n");
        }
    }

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    unsigned long long dbg_dec_tokens_total = 0;
    unsigned long long dbg_dec_literals_total = 0;
    unsigned long long dbg_dec_matches_total = 0;
    unsigned long long dbg_dec_small_offsets_total = 0;
    unsigned long long dbg_dec_output_errors_total = 0;

    /* --- Host-side optimization: pre-allocate reusable resources --- */
    /* Decompression CL buffers (reused across iterations, grown if needed) */
    cl_mem bench_d_off = NULL, bench_d_comp_lens = NULL, bench_d_out = NULL, bench_d_out_lens = NULL;
    size_t bench_d_off_cap = 0, bench_d_comp_lens_cap = 0, bench_d_out_cap = 0, bench_d_out_lens_cap = 0;
    /* Reusable host arrays (grown if needed) */
    cl_uint *bench_h_lens = NULL, *bench_h_off = NULL, *bench_h_out_lens = NULL;
    size_t bench_h_lens_cap = 0, bench_h_off_cap = 0, bench_h_out_lens_cap = 0;
    cl_uint *bench_prev_lens = NULL, *bench_prev_off = NULL;
    size_t bench_prev_lens_cap = 0, bench_prev_off_cap = 0;
    int bench_prev_meta_valid = 0;
    unsigned char* bench_verify_out = NULL;
    size_t bench_verify_out_cap = 0;
    /* Decompression kernel args that stay constant across iterations */
    int bench_dec_kernel_set = 0;
    /* Cached decompression dispatch sizes */
    size_t bench_dec_global = 0, bench_dec_local = 0;
    /* Track decompression parameters to reset kernel args when data shape changes. */
    size_t bench_dec_prev_nblk = 0, bench_dec_prev_blk = 0, bench_dec_prev_in_size = 0;

    while (verify_ok) {
        timing_t tc = {0};
        unsigned long time_us = 0;
        size_t out_size = 0;
        double dec_kernel_us = 0.0;

        cl_uint *h_dbg_dec = NULL;
        cl_mem d_dbg_dec = NULL;

        int skip_input_upload = 1;
        int rc = lzo_compress_core(ctx, q, dev, krn_c, pack_krn, in_path, NULL,
                       &params, skip_input_upload, &ws, &time_us, &out_size, &tc);
        if (rc != 0) {
            verify_ok = 0;
            break;
        }

        size_t nblk = (size_t)tc.nblk;
        size_t blk = (size_t)tc.blk_size_bytes;
        size_t worst_blk = lzo_worst(blk);
        size_t comp_total = (size_t)tc.out_size;
        size_t in_size = (size_t)tc.in_size;
        if (nblk == 0 || blk == 0 || comp_total == 0 || in_size == 0 || in_size != in_size_ref || in_size > 0xFFFFFFFFu) {
            verify_ok = 0;
            break;
        }

        /* Grow host arrays only when needed */
        if (nblk * sizeof(cl_uint) > bench_h_lens_cap) {
            free(bench_h_lens);
            bench_h_lens = (cl_uint*)malloc(nblk * sizeof(cl_uint));
            bench_h_lens_cap = bench_h_lens ? nblk * sizeof(cl_uint) : 0;
        }
        if (nblk * sizeof(cl_uint) > bench_h_off_cap) {
            free(bench_h_off);
            bench_h_off = (cl_uint*)malloc(nblk * sizeof(cl_uint));
            bench_h_off_cap = bench_h_off ? nblk * sizeof(cl_uint) : 0;
        }
        if (nblk * sizeof(cl_uint) > bench_h_out_lens_cap) {
            free(bench_h_out_lens);
            bench_h_out_lens = (cl_uint*)malloc(nblk * sizeof(cl_uint));
            bench_h_out_lens_cap = bench_h_out_lens ? nblk * sizeof(cl_uint) : 0;
        }
        if (!bench_h_lens || !bench_h_off || !bench_h_out_lens) {
            verify_ok = 0;
            goto iter_cleanup;
        }

        err = clEnqueueReadBuffer(q,
                                  ws.d_len,
                                  CL_TRUE,
                                  0,
                                  nblk * sizeof(cl_uint),
                                  bench_h_lens,
                                  0,
                                  NULL,
                                  NULL);
        if (err != CL_SUCCESS) {
            verify_ok = 0;
            goto iter_cleanup;
        }

        for (size_t i = 0; i < nblk; ++i) {
            if ((size_t)bench_h_lens[i] > worst_blk) {
                verify_ok = 0;
                break;
            }
            bench_h_off[i] = (cl_uint)(i * worst_blk);
        }
        if (!verify_ok) {
            goto iter_cleanup;
        }

        int dec_meta_buffers_recreated = 0;

        /* Grow decompression CL buffers only when capacity is insufficient */
        if (nblk * sizeof(cl_uint) > bench_d_off_cap) {
            if (bench_d_off) clReleaseMemObject(bench_d_off);
            bench_d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nblk * sizeof(cl_uint), NULL, &err);
            if (err != CL_SUCCESS || !bench_d_off) { bench_d_off = NULL; bench_d_off_cap = 0; verify_ok = 0; goto iter_cleanup; }
            bench_d_off_cap = nblk * sizeof(cl_uint);
            bench_dec_kernel_set = 0;
            dec_meta_buffers_recreated = 1;
        }
        if (nblk * sizeof(cl_uint) > bench_d_comp_lens_cap) {
            if (bench_d_comp_lens) clReleaseMemObject(bench_d_comp_lens);
            bench_d_comp_lens = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nblk * sizeof(cl_uint), NULL, &err);
            if (err != CL_SUCCESS || !bench_d_comp_lens) { bench_d_comp_lens = NULL; bench_d_comp_lens_cap = 0; verify_ok = 0; goto iter_cleanup; }
            bench_d_comp_lens_cap = nblk * sizeof(cl_uint);
            bench_dec_kernel_set = 0;
            dec_meta_buffers_recreated = 1;
        }
        if (in_size > bench_d_out_cap) {
            if (bench_d_out) clReleaseMemObject(bench_d_out);
            bench_d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, in_size, NULL, &err);
            if (err != CL_SUCCESS || !bench_d_out) { bench_d_out = NULL; bench_d_out_cap = 0; verify_ok = 0; goto iter_cleanup; }
            bench_d_out_cap = in_size;
            bench_dec_kernel_set = 0;
        }
        if (nblk * sizeof(cl_uint) > bench_d_out_lens_cap) {
            if (bench_d_out_lens) clReleaseMemObject(bench_d_out_lens);
            bench_d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err);
            if (err != CL_SUCCESS || !bench_d_out_lens) { bench_d_out_lens = NULL; bench_d_out_lens_cap = 0; verify_ok = 0; goto iter_cleanup; }
            bench_d_out_lens_cap = nblk * sizeof(cl_uint);
            bench_dec_kernel_set = 0;
        }

        size_t meta_bytes = nblk * sizeof(cl_uint);
        int dec_meta_changed = 1;
        if (!dec_meta_buffers_recreated &&
            bench_prev_meta_valid &&
            bench_prev_lens &&
            bench_prev_off &&
            bench_prev_lens_cap >= meta_bytes &&
            bench_prev_off_cap >= meta_bytes &&
            bench_dec_prev_nblk == nblk &&
            bench_dec_prev_blk == blk &&
            bench_dec_prev_in_size == in_size &&
            memcmp(bench_prev_lens, bench_h_lens, meta_bytes) == 0 &&
            memcmp(bench_prev_off, bench_h_off, meta_bytes) == 0) {
            dec_meta_changed = 0;
        }

        if (dec_meta_changed) {
            if (bench_prev_lens_cap < meta_bytes) {
                cl_uint *nlens = (cl_uint*)realloc(bench_prev_lens, meta_bytes);
                if (!nlens) {
                    verify_ok = 0;
                    goto iter_cleanup;
                }
                bench_prev_lens = nlens;
                bench_prev_lens_cap = meta_bytes;
            }
            if (bench_prev_off_cap < meta_bytes) {
                cl_uint *noff = (cl_uint*)realloc(bench_prev_off, meta_bytes);
                if (!noff) {
                    verify_ok = 0;
                    goto iter_cleanup;
                }
                bench_prev_off = noff;
                bench_prev_off_cap = meta_bytes;
            }
            memcpy(bench_prev_lens, bench_h_lens, meta_bytes);
            memcpy(bench_prev_off, bench_h_off, meta_bytes);
            bench_prev_meta_valid = 1;
        }

        cl_event write_events[2] = { NULL, NULL };
        cl_uint write_event_count = 0;
        if (dec_meta_changed) {
            cl_int err_w0 = clEnqueueWriteBuffer(q, bench_d_off, CL_FALSE, 0, meta_bytes, bench_h_off, 0, NULL, &write_events[0]);
            cl_int err_w1 = clEnqueueWriteBuffer(q, bench_d_comp_lens, CL_FALSE, 0, meta_bytes, bench_h_lens, 0, NULL, &write_events[1]);
            if (err_w0 != CL_SUCCESS || err_w1 != CL_SUCCESS) {
                if (write_events[0]) clReleaseEvent(write_events[0]);
                if (write_events[1]) clReleaseEvent(write_events[1]);
                verify_ok = 0;
                goto iter_cleanup;
            }
            write_event_count = 2;
        }

        /* Force kernel arg reset when compression parameters changed between iterations. */
        if (bench_dec_kernel_set && (bench_dec_prev_nblk != nblk || bench_dec_prev_blk != blk || bench_dec_prev_in_size != in_size)) {
            bench_dec_kernel_set = 0;
        }

        /* Set kernel args only when buffers changed (first iter or resize), or parameters changed. */
        if (!bench_dec_kernel_set) {
            cl_uint blk_sz = (cl_uint)blk;
            cl_uint orig_sz = (cl_uint)in_size;
            cl_uint nblk_cl = (cl_uint)nblk;
            CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &ws.d_out));
            CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &bench_d_off));
            CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &bench_d_comp_lens));
            CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &bench_d_out));
            CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_mem), &bench_d_out_lens));
            CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &blk_sz));
            CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &orig_sz));
            CHECK(clSetKernelArg(krn_d, 7, sizeof(cl_uint), &nblk_cl));

            int dbg_dec_enabled = (debug_counters && kernel_has_dbg_dec);
            if (dbg_dec_enabled) {
                size_t dbg_bytes = nblk * 5 * sizeof(cl_uint);
                d_dbg_dec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dbg_bytes, NULL, &err);
                if (err != CL_SUCCESS || !d_dbg_dec) {
                    dbg_dec_enabled = 0;
                } else {
                    h_dbg_dec = (cl_uint*)calloc(nblk * 5, sizeof(cl_uint));
                    if (!h_dbg_dec) {
                        dbg_dec_enabled = 0;
                    } else {
                        err = clEnqueueWriteBuffer(q, d_dbg_dec, CL_TRUE, 0, dbg_bytes, h_dbg_dec, 0, NULL, NULL);
                        if (err != CL_SUCCESS) dbg_dec_enabled = 0;
                    }
                }
                if (!dbg_dec_enabled) {
                    if (d_dbg_dec) { clReleaseMemObject(d_dbg_dec); d_dbg_dec = NULL; }
                    free(h_dbg_dec); h_dbg_dec = NULL;
                }
            }
            if (kernel_has_dbg_dec) {
                cl_mem dbg_arg = dbg_dec_enabled ? d_dbg_dec : bench_d_out_lens;
                cl_uint dbg_flag = dbg_dec_enabled ? 1U : 0U;
                CHECK(clSetKernelArg(krn_d, 8, sizeof(cl_mem), &dbg_arg));
                CHECK(clSetKernelArg(krn_d, 9, sizeof(cl_uint), &dbg_flag));
            }

            /* Compute dispatch sizes once */
            bench_dec_local = (g_cli_local_size > 0) ? g_cli_local_size : 1;
            size_t target_items = (size_t)nblk;
            if (target_items == 0) target_items = 1;
            if (bench_dec_local > target_items) bench_dec_local = 1;
            bench_dec_global = ((target_items + bench_dec_local - 1) / bench_dec_local) * bench_dec_local;
            if (bench_dec_global == 0) bench_dec_global = 1;

            bench_dec_prev_nblk = nblk;
            bench_dec_prev_blk = blk;
            bench_dec_prev_in_size = in_size;
            bench_dec_kernel_set = 1;
        }

        cl_event dec_kernel_evt = NULL;
        err = clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &bench_dec_global, &bench_dec_local,
                         write_event_count,
                         (write_event_count > 0) ? write_events : NULL,
                         &dec_kernel_evt);
        double dec_upload_us = 0.0;
        for (int wi = 0; wi < 2; ++wi) {
            if (write_events[wi]) {
                clWaitForEvents(1, &write_events[wi]);
                dec_upload_us += event_elapsed_us(write_events[wi]);
            }
        }
        if (write_events[0]) clReleaseEvent(write_events[0]);
        if (write_events[1]) clReleaseEvent(write_events[1]);
        double dec_kernel_profile_us = 0.0;
        if (err == CL_SUCCESS && dec_kernel_evt) {
            clWaitForEvents(1, &dec_kernel_evt);
            dec_kernel_profile_us = event_elapsed_us(dec_kernel_evt);
            clReleaseEvent(dec_kernel_evt);
            dec_kernel_evt = NULL;
        }
        dec_kernel_us = dec_kernel_profile_us;
        if (err != CL_SUCCESS) {
            verify_ok = 0;
            goto iter_cleanup;
        }

        if (d_dbg_dec && h_dbg_dec) {
            size_t dbg_bytes = nblk * 5 * sizeof(cl_uint);
            err = clEnqueueReadBuffer(q, d_dbg_dec, CL_TRUE, 0, dbg_bytes, h_dbg_dec, 0, NULL, NULL);
            if (err == CL_SUCCESS) {
                for (size_t i = 0; i < nblk; ++i) {
                    size_t base = i * 5;
                    dbg_dec_tokens_total += h_dbg_dec[base + 0];
                    dbg_dec_literals_total += h_dbg_dec[base + 1];
                    dbg_dec_matches_total += h_dbg_dec[base + 2];
                    dbg_dec_small_offsets_total += h_dbg_dec[base + 3];
                    dbg_dec_output_errors_total += h_dbg_dec[base + 4];
                }
            }
        }

        cl_event read_lens_evt = NULL;
        err = clEnqueueReadBuffer(q, bench_d_out_lens, CL_FALSE, 0, nblk * sizeof(cl_uint), bench_h_out_lens, 0, NULL, &read_lens_evt);
        if (err != CL_SUCCESS) {
            verify_ok = 0;
            goto iter_cleanup;
        }
        clWaitForEvents(1, &read_lens_evt);
        clReleaseEvent(read_lens_evt);
        size_t out_total = 0;
        for (size_t i = 0; i < nblk; ++i) {
            if (bench_h_out_lens[i] == 0xFFFFFFFFu) {
                verify_ok = 0;
                break;
            }
            out_total += (size_t)bench_h_out_lens[i];
        }
        if (!verify_ok || out_total != in_size) {
            fprintf(stderr, "[BENCH] level=%d nblk=%zu in_size=%zu out_total=%zu\n", comp_level, nblk, in_size, out_total);
            verify_ok = 0;
            goto iter_cleanup;
        }

        if (in_size > bench_verify_out_cap) {
            unsigned char* nbuf = (unsigned char*)realloc(bench_verify_out, in_size);
            if (!nbuf) {
                verify_ok = 0;
                goto iter_cleanup;
            }
            bench_verify_out = nbuf;
            bench_verify_out_cap = in_size;
        }

        err = clEnqueueReadBuffer(q,
                                  bench_d_out,
                                  CL_TRUE,
                                  0,
                                  in_size,
                                  bench_verify_out,
                                  0,
                                  NULL,
                                  NULL);
        if (err != CL_SUCCESS) {
            verify_ok = 0;
            goto iter_cleanup;
        }
        if (memcmp(bench_verify_out, input_ref, in_size) != 0) {
            fprintf(stderr, "[BENCH] content mismatch level=%d nblk=%zu in_size=%zu\n", comp_level, nblk, in_size);
            size_t mpos = 0;
            unsigned char* mdata = bench_verify_out;
            for (size_t i = 0; i < in_size; ++i) {
                if (mdata[i] != input_ref[i]) {
                    mpos = i;
                    break;
                }
            }
            fprintf(stderr, "[BENCH] first mismatch at offset %zu: got %02x expected %02x\n", mpos, mdata[mpos], input_ref[mpos]);
            verify_ok = 0;
        }
        if (!verify_ok) {
            goto iter_cleanup;
        }

        total_successful_iterations += 1;
        if (total_successful_iterations <= bench_drop_iterations) {
            goto iter_cleanup;
        }

        if (n == cap) {
            size_t new_cap = cap * 2;
            double *nc = (double *)realloc(comp_tp, new_cap * sizeof(double));
            if (!nc) { verify_ok = 0; break; }
            comp_tp = nc;

            double *nd = (double *)realloc(dec_tp, new_cap * sizeof(double));
            if (!nd) { verify_ok = 0; break; }
            dec_tp = nd;

            double *nr = (double *)realloc(ratio_pct, new_cap * sizeof(double));
            if (!nr) { verify_ok = 0; break; }
            ratio_pct = nr;
            cap = new_cap;
        }

        double in_mb = (double)tc.in_size / (1024.0 * 1024.0);
        comp_tp[n] = (tc.kernel_exec_us > 0) ? (in_mb * 1000000.0 / (double)tc.kernel_exec_us) : 0.0;
        dec_tp[n] = (dec_kernel_us > 0.0) ? (in_mb * 1000000.0 / dec_kernel_us) : 0.0;
        ratio_pct[n] = (tc.in_size > 0) ? (100.0 * (double)tc.out_size / (double)tc.in_size) : 0.0;
        n++;

        iter_cleanup:
            if (d_dbg_dec) clReleaseMemObject(d_dbg_dec);
            free(h_dbg_dec);
            if (!verify_ok) break;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    /* Free bench-loop persistent resources */
    if (bench_d_off) clReleaseMemObject(bench_d_off);
    if (bench_d_comp_lens) clReleaseMemObject(bench_d_comp_lens);
    if (bench_d_out) clReleaseMemObject(bench_d_out);
    if (bench_d_out_lens) clReleaseMemObject(bench_d_out_lens);
    free(bench_h_lens);
    free(bench_h_off);
    free(bench_h_out_lens);
    free(bench_prev_lens);
    free(bench_prev_off);
    free(bench_verify_out);

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s ratio=%.2f%%\n",
            median_double(comp_tp, n), median_double(ratio_pct, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s verify=%s\n",
            median_double(dec_tp, n), verify_ok ? "OK" : "FAIL");
        if (debug_counters) {
            printf("[LZO-DBG][DECOMP] tokens=%llu literal_bytes=%llu match_bytes=%llu small_offsets=%llu output_errors=%llu\n",
                dbg_dec_tokens_total,
                dbg_dec_literals_total,
                dbg_dec_matches_total,
                dbg_dec_small_offsets_total,
                dbg_dec_output_errors_total);
        }
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_tp);
    free(dec_tp);
    free(ratio_pct);
    free(input_ref);
    lzo_gpu_workspace_free(&ws);
    if (pack_krn) clReleaseKernel(pack_krn);
    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (krn_c) clReleaseKernel(krn_c);
    if (prog_c) clReleaseProgram(prog_c);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return verify_ok ? 0 : 1;
}


/* Implementations will call into lzo_gpu_core.c which provides lzo_compress_core/lzo_decompress_core */
#include "lzo_gpu_core.h"


int run_lzo_standalone(int argc, char** argv)
{
    if (argc < 1) { // Changed from 2 to 1 because we might pass 0/1 args if flags stripped
        show_help(argv[0]);
        return 0;
    }

    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int bench_mode = 0;
    double bench_seconds = 3.0;
    int comp_level = -1; /* -1 means adaptive (not explicitly specified by user) */
    const char *alg_name = "lzo1x";
    const char *effective_in_path = NULL;
    const char *effective_lz_path = NULL;
    const char *effective_output_path = NULL;
    char temp_input_path[PATH_MAX] = {0};
    char temp_output_path[PATH_MAX] = {0};
    int input_from_stdin = 0;
    int output_to_stdout = 0;
    int have_temp_input = 0;
    int have_temp_output = 0;
    int rc = 1;

    /* parse options and positionals */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            show_help(argv[0]);
            return 0;
        }
        if (strcmp(arg, "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bench_seconds = atof(argv[++i]);
            }
            continue;
        }
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            g_verbose = 1;
            continue;
        }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
                return 1;
            }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "-l") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
                return 1;
            }
            const char* level_arg = argv[++i];
            comp_level = atoi(level_arg);
            if (comp_level != 999 && comp_level != 99 && (comp_level < 11 || comp_level > 15)) {
                fprintf(stderr, "error: dictionary size must be between 11 and 15 bits (got %d)\n", comp_level);
                return 1;
            }
            continue;
        }
        if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
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
            decompress_mode = 0;
            continue;
        }
        if (strcmp(arg, "-d") == 0 || strcmp(arg, "--decompress") == 0) {
            decompress_mode = 1;
            continue;
        }
        if (strcmp(arg, "-B") == 0 || strcmp(arg, "--block-size") == 0) {
            if (i + 1 < argc) {
                const char* s = argv[++i];
                g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s);
                size_t b = lzo_parse_block_size(s);
                if (b > 0) g_cli_fixed_block_bytes = b;
            } else {
                fprintf(stderr, "Error: -B requires an argument\n");
                return 1;
            }
            continue;
        }
        if (strncmp(arg, "--block-size=", 13) == 0) {
            const char* s = arg + 13;
            g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s);
            size_t b = lzo_parse_block_size(s);
            if (b > 0) g_cli_fixed_block_bytes = b;
            continue;
        }
        if (strncmp(arg, "--local=", 8) == 0) {
            g_cli_local_size = (size_t)atoi(arg + 8);
            continue;
        }
        if (strcmp(arg, "--local") == 0) {
            if (i + 1 < argc) g_cli_local_size = (size_t)atoi(argv[++i]);
            else { fprintf(stderr, "Error: --local requires an argument\n"); return 1; }
            continue;
        }
        /* positional or error */
        if (arg[0] == '-' && strcmp(arg, "-") != 0) {
            fprintf(stderr, "Error: Unknown option %s\n", arg);
            fprintf(stderr, "Tips: Use -h or --help for help. Standard positional usage: %s <input> [output]\n", argv[0]);
            return 1;
        } else {
            if (!in_path && !lz_path) {
                if (decompress_mode) lz_path = arg; else in_path = arg;
            } else if (!output_explicit) {
                output_path = arg;
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                return 1;
            }
        }
    }

    if (bench_mode && decompress_mode) {
        fprintf(stderr, "Error: --bench only supports compress mode input (it runs compress+decompress internally)\n");
        return 1;
    }

    input_from_stdin = decompress_mode ? path_is_dash(lz_path) : path_is_dash(in_path);

    /* Set default output names if not specified */
    char default_output[512];
    if (output_explicit == 0 || output_path == NULL) {
        if (decompress_mode) {
            if (lz_path) {
                if (path_is_dash(lz_path)) {
                    snprintf(default_output, sizeof(default_output), "-");
                } else {
                    size_t ilen = strlen(lz_path);
                    const char *suf = ".lzo";
                    size_t suf_len = strlen(suf);
                    if (ilen > suf_len && strcmp(lz_path + ilen - suf_len, suf) == 0) {
                        /* strip suffix */
                        size_t n = (ilen - suf_len < sizeof(default_output) - 1) ? ilen - suf_len : sizeof(default_output) - 1;
                        memcpy(default_output, lz_path, n);
                        default_output[n] = '\0';
                    } else {
                        /* append .dec */
                        snprintf(default_output, sizeof(default_output), "%s.dec", lz_path);
                    }
                }
                output_path = default_output;
            }
        } else {
            if (in_path) {
                if (path_is_dash(in_path)) {
                    snprintf(default_output, sizeof(default_output), "-");
                } else {
                    snprintf(default_output, sizeof(default_output), "%s.lzo", in_path);
                }
                output_path = default_output;
            }
        }
    }

    output_to_stdout = path_is_dash(output_path);
    if (output_to_stdout) suppress_non_data = 1;

    if (bench_mode) {
        if (input_from_stdin) {
            fprintf(stderr, "Error: --bench does not support stdin input ('-')\n");
            return 1;
        }
        return run_lzo_bench(in_path, alg_name, comp_level, bench_seconds);
    }

    effective_in_path = in_path;
    effective_lz_path = lz_path;
    effective_output_path = output_path;

    if (input_from_stdin) {
        if (create_temp_path(temp_input_path, sizeof(temp_input_path), "/tmp/lzo_gpu_stdin_XXXXXX") != 0) {
            fprintf(stderr, "Error: failed to create temporary input path for stdin stream\n");
            return 1;
        }
        if (copy_stream_to_path(stdin, temp_input_path) != 0) {
            fprintf(stderr, "Error: failed to capture stdin into temporary input file\n");
            remove(temp_input_path);
            return 1;
        }
        have_temp_input = 1;
        if (decompress_mode) effective_lz_path = temp_input_path;
        else effective_in_path = temp_input_path;
    }

    if (output_to_stdout) {
        if (create_temp_path(temp_output_path, sizeof(temp_output_path), "/tmp/lzo_gpu_stdout_XXXXXX") != 0) {
            fprintf(stderr, "Error: failed to create temporary output path for stdout stream\n");
            if (have_temp_input) remove(temp_input_path);
            return 1;
        }
        have_temp_output = 1;
        effective_output_path = temp_output_path;
    }

    /* Decompress mode */
    if (decompress_mode) {
        rc = do_decompress_mode(effective_lz_path, effective_output_path, output_explicit, suppress_non_data);
    } else {
        /* Compress path (simple, fast) */
        rc = do_compress_mode(effective_in_path, effective_output_path, output_explicit, suppress_non_data, alg_name, comp_level);
    }

    if (rc == 0 && output_to_stdout) {
        if (copy_path_to_stream(effective_output_path, stdout) != 0) {
            fprintf(stderr, "Error: failed to emit streamed output to stdout\n");
            rc = 1;
        }
    }

    if (have_temp_input) remove(temp_input_path);
    if (have_temp_output) remove(temp_output_path);
    return rc;
}


/* --- Unified Tool Main Entry --- */
static int stop_daemon(void) {
#if defined(_WIN32) || defined(_WIN64)
    fprintf(stderr, "--stop-daemon is not supported in Windows builds.\n");
    return 1;
#else
    const char* pid_path = lzo_daemon_pidfile_path();
    const char* sock_path = lzo_daemon_socket_path();
    FILE* f = fopen(pid_path, "r");
    if (!f) {
        printf("守护进程似乎没有运行 (未找到 PID 文件: %s)\n", pid_path);
        /* Check if socket exists anyway */
        if (access(sock_path, F_OK) == 0) unlink(sock_path);
        return 0;
    }
    pid_t pid;
    if (fscanf(f, "%d", &pid) != 1) {
        fclose(f);
        printf("无法读取 PID 文件内容\n");
        return 1;
    }
    fclose(f);

    printf("正在停止守护进程 (PID: %d)...\n", pid);
    if (kill(pid, SIGTERM) == 0) {
        /* Wait up to 5 seconds for it to exit */
        for (int i = 0; i < 50; i++) {
            if (kill(pid, 0) != 0) break;
            usleep(100000);
        }
        if (kill(pid, 0) == 0) {
            printf("警告: 守护进程未能在 5 秒内退出，正在强制终止...\n");
            kill(pid, SIGKILL);
        }
    } else if (errno == ESRCH) {
        printf("进程 %d 已经退出\n", pid);
    } else {
        perror("停止守护进程失败");
    }

    unlink(pid_path);
    unlink(sock_path);
    printf("守护进程清理完成\n");
    return 0;
#endif
}

int main(int argc, char** argv) {
    /* Handle global options like --socket/--pid first for all sub-commands */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--socket") == 0) {
            if (i + 1 < argc) lzo_set_daemon_socket_path(argv[i+1]);
        } else if (strcmp(argv[i], "--pid") == 0) {
            if (i + 1 < argc) lzo_set_daemon_pidfile_path(argv[i+1]);
        }
    }

    if (argc >= 2) {
        if (strcmp(argv[1], "--daemon") == 0) {
            return run_lzo_daemon(argc - 1, argv + 1);
        }
        if (strcmp(argv[1], "--stop-daemon") == 0) {
            return stop_daemon();
        }
        if (strcmp(argv[1], "--use-daemon") == 0) {
            /* If --use-daemon is the first arg, we shift args for the client */
            return run_lzo_client(argc - 1, argv + 1);
        }
    }
    /* Default: Standalone mode */
    return run_lzo_standalone(argc, argv);
}
