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
#include <pthread.h>
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
#include "lzo_hybrid_utils.h"
#include "lzo_hybrid_core.h"
#include "lzo_env_config.h"

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
static int g_cli_debug_kernel = 0;

enum {
    BENCH_LZO_DBG_DEC_TOKENS = 0,
    BENCH_LZO_DBG_DEC_LITERAL_BYTES,
    BENCH_LZO_DBG_DEC_MATCH_BYTES,
    BENCH_LZO_DBG_DEC_SMALL_OFFSETS,
    BENCH_LZO_DBG_DEC_OUTPUT_ERROR,
    BENCH_LZO_DBG_DEC_LITERAL_OPS,
    BENCH_LZO_DBG_DEC_MATCH_OPS,
    BENCH_LZO_DBG_DEC_OVERLAP_MATCHES,
    BENCH_LZO_DBG_DEC_M2_MATCHES,
    BENCH_LZO_DBG_DEC_M3_MATCHES,
    BENCH_LZO_DBG_DEC_M4_MATCHES,
    BENCH_LZO_DBG_DEC_FIRST_LITERAL_RUN_BYTES,
    BENCH_LZO_DBG_DEC_FIRST_LITERAL_RUN_OPS,
    BENCH_LZO_DBG_DEC_POST_MATCH_LITERAL_BYTES,
    BENCH_LZO_DBG_DEC_POST_MATCH_LITERAL_OPS,
    BENCH_LZO_DBG_DEC_EOF_MARKERS,
    BENCH_LZO_DBG_DEC_N
};

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
    const lzo_env_config_t* cfg = lzo_env_config();
    if (cfg->standard_copy_set) return cfg->standard_copy ? 1 : 0;
    return lzo_device_host_unified_memory(device) ? 0 : 1;
}

static void lzo_print_bench_env(cl_device_id device, int standard_copy,
                                const char* alg_name, int comp_level,
                                int debug_counters, double bench_seconds) {
    char dev_name[256] = {0};
    char vendor[256] = {0};
    char driver[256] = {0};
    char version[256] = {0};
    char opencl_c[256] = {0};
    cl_uint cu = 0;
    cl_ulong global_mem = 0;
    cl_ulong max_alloc = 0;
    cl_bool unified = CL_FALSE;
    size_t max_wg = 0;
    cl_command_queue_properties queue_props = 0;

    if (!device) return;
    (void)clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    (void)clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_c), opencl_c, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
    (void)clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    if (q) (void)clGetCommandQueueInfo(q, CL_QUEUE_PROPERTIES, sizeof(queue_props), &queue_props, NULL);

    printf("[LZO-BENCH-ENV] device=\"%s\" vendor=\"%s\" driver=\"%s\" device_version=\"%s\" opencl_c=\"%s\"\n",
           dev_name[0] ? dev_name : "unknown",
           vendor[0] ? vendor : "unknown",
           driver[0] ? driver : "unknown",
           version[0] ? version : "unknown",
           opencl_c[0] ? opencl_c : "unknown");
    printf("[LZO-BENCH-DEVICE] cu=%u global_mem=%llu max_alloc=%llu unified_memory=%d max_wg=%zu queue_profiling=%d\n",
           (unsigned)cu,
           (unsigned long long)global_mem,
           (unsigned long long)max_alloc,
           unified == CL_TRUE ? 1 : 0,
           max_wg,
           (queue_props & CL_QUEUE_PROFILING_ENABLE) ? 1 : 0);
    printf("[LZO-BENCH-CONFIG] alg=%s level=%d block=%zu local=%zu standard_copy=%d debug_kernel=%d bench_seconds=%.3f\n",
           alg_name ? alg_name : "unknown",
           comp_level,
           g_cli_fixed_block_bytes,
           g_cli_local_size ? g_cli_local_size : 1,
           standard_copy,
           debug_counters,
           bench_seconds);
}

static cl_device_type preferred_opencl_device_type(void)
{
    const char* pref = lzo_env_config()->opencl_device_type;
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

static int lzo_strict_opencl_device_enabled(void)
{
    return lzo_env_config()->opencl_strict_device ? 1 : 0;
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
    int strict_device = lzo_strict_opencl_device_enabled();
    if (pref_type == CL_DEVICE_TYPE_GPU) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        if (!strict_device) {
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        }
    } else if (pref_type == CL_DEVICE_TYPE_CPU) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
        if (!strict_device) {
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
        }
    } else if (pref_type == CL_DEVICE_TYPE_DEFAULT) {
        r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, &dev, &selected_pf);
        if (!strict_device) {
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_GPU, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_CPU, &dev, &selected_pf);
            if (r != CL_SUCCESS) r = lzo_try_get_device(platforms, num_platforms, CL_DEVICE_TYPE_ALL, &dev, &selected_pf);
        }
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
    fprintf(stderr, "  -L, --level LEVEL    Compression level: 11-15=D_BITS (default: 14)\n");
    fprintf(stderr, "  -B, --block-size N   Block size (B/KB/MB) (default: 64KB)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
    fprintf(stderr, "  --local N            Local work-group size (default: 1)\n");
    fprintf(stderr, "  --bench [N]          Stable benchmark (compress+decompress+verify), optional N seconds (default: 3)\n");
    fprintf(stderr, "  --debug-kernel       Use debug-enabled kernels and print diagnostic counters\n");
    fprintf(stderr, "  --cpu-threads N      OpenCL CPU slots/worker items for CPU or mixed paths\n");
    fprintf(stderr, "  --gpu-ratio F        GPU block fraction 0.0-1.0; CPU fraction is 1-F\n");
    fprintf(stderr, "  --adaptive           Enable adaptive split mode (currently starts from 50/50 split)\n");
    fprintf(stderr, "  --gpu-level LEVEL    GPU compression D_BITS for hybrid mode (default: --level)\n");
    fprintf(stderr, "  --cpu-level LEVEL    CPU compression D_BITS for hybrid mode (default: --level)\n");
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
    fprintf(stderr, "\nEnvironment variables:\n");
    fprintf(stderr, "    LZO_OPENCL_DEVICE_TYPE=GPU|CPU|DEFAULT|ALL  Device preference (default: GPU).\n");
    fprintf(stderr, "    LZO_OPENCL_STRICT_DEVICE=0|1                Require the requested device type.\n");
    fprintf(stderr, "    LZO_STANDARD_COPY=0|1                       0=map/zero-copy when possible, 1=explicit copy.\n");
    fprintf(stderr, "    LZO_BENCH_ROUNDS=N                         Override --bench round count.\n");
    fprintf(stderr, "    LZO_BLOCK_MIN_KB=N / LZO_BLOCK_MAX_KB=N     Adaptive block bounds when -B is omitted.\n");
    fprintf(stderr, "    LZO_BLOCK_TARGET_SLOTS=N                   Adaptive block target block count.\n");
    fprintf(stderr, "    LZO_BLOCK_SLOTS_PER_CU=N                   Adaptive block target as CU multiple.\n");
    fprintf(stderr, "    LZO_GPU_COMP_SLOTS=N                       GPU compression global/dictionary slots cap.\n");
    fprintf(stderr, "    LZO_GPU_DECOMP_CHUNKED=auto|on|off         Chunked decompression readback policy.\n");
    fprintf(stderr, "    LZO_GPU_DECOMP_CHUNK_KB=N                  Chunked decompression readback size.\n");
    fprintf(stderr, "    LZO_GPU_DEBUG=0|1                          Debug kernel/counters path.\n");
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
    const char* env;
    if (g_cli_debug_kernel) return 1;
    env = getenv("LZO_GPU_DEBUG");
    if (!env || !*env) return 0;
    return (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 ||
            strcasecmp(env, "yes") == 0 || strcasecmp(env, "on") == 0);
}

static int lzo_format_program_name(char* out, size_t out_sz, const char* base_name, int debug_enabled) {
    static const char debug_suffix[] = "_debug";
    size_t base_len;
    size_t suffix_len = sizeof(debug_suffix) - 1;

    if (!out || out_sz == 0) return -1;
    if (!base_name) {
        out[0] = '\0';
        return -1;
    }
    base_len = strlen(base_name);
    if (base_len >= out_sz) {
        out[0] = '\0';
        return -1;
    }
    memcpy(out, base_name, base_len + 1);
    if (!debug_enabled) return 0;
    if (base_len + suffix_len >= out_sz) {
        out[0] = '\0';
        return -1;
    }
    memcpy(out + base_len, debug_suffix, suffix_len + 1);
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

typedef struct {
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel kernel;
    cl_kernel full_kernel;
    const char* label;
    void* cache_slot;
    int owns_context;
    int owns_queue;
    int owns_program;
} hybrid_ocl_t;

typedef struct {
    int valid;
    cl_device_type dtype;
    int bits;
    char label[16];
    char alg[16];
    char suffix[32];
    hybrid_ocl_t h;
    cl_mem bufs[4];
    size_t buf_sizes[4];
} daemon_hybrid_cache_t;

static int g_daemon_hybrid_cache_enabled = 0;
static pthread_mutex_t g_daemon_hybrid_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static daemon_hybrid_cache_t g_daemon_hybrid_cache[8];

static void
#if defined(__GNUC__)
__attribute__((unused))
#endif
lzo_hybrid_set_daemon_cache_enabled(int enabled)
{
    g_daemon_hybrid_cache_enabled = enabled ? 1 : 0;
}

static cl_device_type lzo_hybrid_label_dtype(const char* label)
{
    return (label && strcmp(label, "CPU") == 0) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;
}

static daemon_hybrid_cache_t* lzo_hybrid_find_cache(cl_device_type dtype, const char* label, const char* alg_name, int bits, const char* suffix)
{
    size_t i;
    for (i = 0; i < sizeof(g_daemon_hybrid_cache) / sizeof(g_daemon_hybrid_cache[0]); ++i) {
        daemon_hybrid_cache_t* c = &g_daemon_hybrid_cache[i];
        if (!c->valid) continue;
        if (c->dtype == dtype && c->bits == bits &&
            strcmp(c->label, label ? label : "") == 0 &&
            strcmp(c->alg, alg_name ? alg_name : "") == 0 &&
            strcmp(c->suffix, suffix ? suffix : "") == 0) {
            return c;
        }
    }
    return NULL;
}

static daemon_hybrid_cache_t* lzo_hybrid_alloc_cache_slot(void)
{
    size_t i;
    for (i = 0; i < sizeof(g_daemon_hybrid_cache) / sizeof(g_daemon_hybrid_cache[0]); ++i) {
        if (!g_daemon_hybrid_cache[i].valid) return &g_daemon_hybrid_cache[i];
    }
    return &g_daemon_hybrid_cache[0];
}

static cl_mem lzo_hybrid_cached_buffer(hybrid_ocl_t* h,
                                       int index,
                                       size_t size,
                                       cl_mem_flags flags,
                                       cl_int* err_out)
{
    daemon_hybrid_cache_t* cache = NULL;
    cl_int err = CL_SUCCESS;

    if (err_out) *err_out = CL_SUCCESS;
    if (size == 0) size = 1;
    if (!g_daemon_hybrid_cache_enabled || !h || !h->cache_slot || index < 0 || index >= 4) {
        cl_mem buf = clCreateBuffer(h ? h->ctx : NULL, flags, size, NULL, &err);
        if (err_out) *err_out = err;
        return buf;
    }

    cache = (daemon_hybrid_cache_t*)h->cache_slot;
    if (cache->bufs[index] && cache->buf_sizes[index] < size) {
        clReleaseMemObject(cache->bufs[index]);
        cache->bufs[index] = NULL;
        cache->buf_sizes[index] = 0;
    }
    if (!cache->bufs[index]) {
        cache->bufs[index] = clCreateBuffer(h->ctx, flags, size, NULL, &err);
        if (err != CL_SUCCESS) {
            if (err_out) *err_out = err;
            return NULL;
        }
        cache->buf_sizes[index] = size;
    }
    if (err_out) *err_out = CL_SUCCESS;
    return cache->bufs[index];
}

static size_t g_accel_cpu_slots_override = 0;
static size_t g_cli_cpu_threads = 0;
static int g_cli_cpu_threads_set = 0;
static size_t g_cli_cpu_share_pct = 0;
static int g_cli_gpu_ratio_set = 0;
static int g_cli_adaptive_enabled = 0;
static int g_cli_gpu_level = -1;
static int g_cli_cpu_level = -1;

static void lzo_cli_set_cpu_threads(long threads)
{
    if (threads < 0) threads = 0;
    if (threads > 65535) threads = 65535;
    g_cli_cpu_threads = (size_t)threads;
    g_cli_cpu_threads_set = 1;
}

static void lzo_cli_set_gpu_ratio(double gpu_ratio)
{
    long cpu_share;
    if (gpu_ratio < 0.0) gpu_ratio = 0.0;
    if (gpu_ratio > 1.0) gpu_ratio = 1.0;
    cpu_share = (long)((1.0 - gpu_ratio) * 100.0 + 0.5);
    if (cpu_share < 0) cpu_share = 0;
    if (cpu_share > 100) cpu_share = 100;
    g_cli_cpu_share_pct = (size_t)cpu_share;
    g_cli_gpu_ratio_set = 1;
}

static void lzo_cli_enable_adaptive(void)
{
    g_cli_adaptive_enabled = 1;
    if (!g_cli_gpu_ratio_set) {
        lzo_cli_set_gpu_ratio(0.5);
    }
}

static int lzo_cli_parse_level_value(const char* text, const char* opt)
{
    int level = atoi(text);
    if (level < 11 || level > 20) {
        fprintf(stderr, "error: %s must be between 11 and 20 bits (got %d)\n", opt, level);
        return -1;
    }
    return level;
}

static size_t lzo_auto_gpu_comp_slots(cl_uint cus, size_t block_count, size_t computed_slots)
{
    size_t out = computed_slots;
    if (lzo_env_config()->gpu_comp_slots > 0) {
        if (block_count > 0 && out > block_count) out = block_count;
        if (out == 0) out = 1;
        return out;
    }
    if (cus == 0) cus = 1;
    size_t cap = (size_t)cus * (size_t)LZO_GPU_COMP_AUTO_SLOTS_PER_CU;
    if (cap > 0 && out > cap) out = cap;
    if (block_count > 0 && out > block_count) out = block_count;
    if (out == 0) out = 1;
    return out;
}

static size_t lzo_accel_slot_limit_for_device(cl_device_id dev, size_t block_count, int level, int is_cpu)
{
    cl_uint cus = 0;
    cl_ulong global_mem = 0;
    cl_ulong max_alloc = 0;
    size_t dict_per_block = (1ULL << level) * sizeof(uint32_t);
    size_t target_items = block_count ? block_count : 1;
    size_t occ_cap;
    size_t mem_cap = SIZE_MAX;
    long slots_env = is_cpu ? (long)g_accel_cpu_slots_override : lzo_env_config()->gpu_comp_slots;
    long per_cu_env = 0;

    if (dev) {
        (void)clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
        (void)clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
        (void)clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
    }
    if (cus == 0) cus = 1;

    int explicit_per_cu = per_cu_env > 0;
    if (per_cu_env <= 0) per_cu_env = is_cpu ? 1 : LZO_OCC_FACTOR_DEFAULT;
    occ_cap = (size_t)cus * (size_t)per_cu_env;
    if (slots_env > 0) occ_cap = (size_t)slots_env;
    if (!is_cpu && !explicit_per_cu && slots_env <= 0 && occ_cap < 1024U) occ_cap = 1024U;
    if (occ_cap == 0) occ_cap = 1;

    if (dict_per_block > 0) {
        size_t cap_by_global = SIZE_MAX;
        size_t cap_by_alloc = SIZE_MAX;
        if (global_mem > 0) cap_by_global = (size_t)(global_mem / 6ULL) / dict_per_block;
        if (max_alloc > 0) cap_by_alloc = (size_t)(max_alloc * 9ULL / 10ULL) / dict_per_block;
        mem_cap = (cap_by_global < cap_by_alloc) ? cap_by_global : cap_by_alloc;
    }
    if (mem_cap == 0) mem_cap = 1;

    if (target_items > occ_cap) target_items = occ_cap;
    if (target_items > mem_cap) target_items = mem_cap;
    if (target_items == 0) target_items = 1;
    if (!is_cpu) target_items = lzo_auto_gpu_comp_slots(cus, block_count, target_items);
    return target_items;
}

static int lzo_hybrid_enabled(void) {
    return g_cli_adaptive_enabled || (g_cli_gpu_ratio_set && g_cli_cpu_share_pct > 0);
}

static int lzo_accel_bench_requested(void) {
    return g_cli_gpu_ratio_set || g_cli_cpu_threads_set || g_cli_adaptive_enabled;
}

static size_t lzo_hybrid_cpu_share_pct(void) {
    return g_cli_cpu_share_pct;
}

static size_t lzo_hybrid_cpu_slots_for_device(cl_device_id cpu_dev, size_t block_count) {
    cl_uint cu = 1;
    long user = g_accel_cpu_slots_override > 0 ? (long)g_accel_cpu_slots_override : (g_cli_cpu_threads_set ? (long)g_cli_cpu_threads : 0);
    if (cpu_dev) (void)clGetDeviceInfo(cpu_dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    size_t slots = (user > 0) ? (size_t)user : lzo_accel_slot_limit_for_device(cpu_dev, block_count, LZO_DEFAULT_COMP_LEVEL, 1);
    if (slots == 0) slots = 1;
    if (block_count > 0 && slots > block_count) slots = block_count;
    if (slots == 0) slots = 1;
    return slots;
}

static size_t lzo_hybrid_gpu_items_for_device(cl_device_id gpu_dev, size_t block_count, int level) {
    return lzo_accel_slot_limit_for_device(gpu_dev, block_count, level, 0);
}

static int lzo_hybrid_init_device(hybrid_ocl_t* h, cl_device_type dtype, const char* label) {
    if (g_daemon_hybrid_cache_enabled) {
        daemon_hybrid_cache_t* cache;
        pthread_mutex_lock(&g_daemon_hybrid_cache_lock);
        for (cache = lzo_hybrid_find_cache(dtype, label, NULL, 0, NULL); cache; cache = NULL) {
            *h = cache->h;
            h->cache_slot = cache;
            h->owns_context = 0;
            h->owns_queue = 0;
            h->owns_program = 0;
            pthread_mutex_unlock(&g_daemon_hybrid_cache_lock);
            return 0;
        }
        pthread_mutex_unlock(&g_daemon_hybrid_cache_lock);
    }
    cl_int err;
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;
    cl_platform_id pf = NULL;
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };

    memset(h, 0, sizeof(*h));
    h->label = label;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return -1;
    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) return -1;
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return -1;
    }
    err = lzo_try_get_device(platforms, num_platforms, dtype, &h->dev, &pf);
    free(platforms);
    if (err != CL_SUCCESS || !h->dev) return -1;

    h->ctx = clCreateContext(NULL, 1, &h->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !h->ctx) return -1;
    h->q = clCreateCommandQueueWithProperties(h->ctx, h->dev, props, &err);
    if (err != CL_SUCCESS || !h->q) return -1;
    h->owns_context = 1;
    h->owns_queue = 1;
    if (g_daemon_hybrid_cache_enabled) {
        daemon_hybrid_cache_t* cache = lzo_hybrid_alloc_cache_slot();
        if (cache) {
            memset(cache, 0, sizeof(*cache));
            cache->valid = 1;
            cache->dtype = dtype;
            cache->bits = 0;
            strncpy(cache->label, label ? label : "", sizeof(cache->label) - 1);
            cache->h = *h;
            cache->h.cache_slot = cache;
            cache->h.owns_context = 1;
            cache->h.owns_queue = 1;
            cache->h.owns_program = 0;
            h->cache_slot = cache;
        }
    }
    return 0;
}

static void lzo_hybrid_release(hybrid_ocl_t* h) {
    if (!h) return;
    if (g_daemon_hybrid_cache_enabled &&
        !h->owns_context && !h->owns_queue && !h->owns_program) {
        memset(h, 0, sizeof(*h));
        return;
    }
    if (h->full_kernel) clReleaseKernel(h->full_kernel);
    if (h->kernel) clReleaseKernel(h->kernel);
    if (h->prog && h->owns_program) clReleaseProgram(h->prog);
    if (h->q && h->owns_queue) clReleaseCommandQueue(h->q);
    if (h->ctx && h->owns_context) clReleaseContext(h->ctx);
    memset(h, 0, sizeof(*h));
}

static int lzo_hybrid_load_kernel(hybrid_ocl_t* h, const char* alg_name, int bits, const char* suffix) {
    if (g_daemon_hybrid_cache_enabled) {
        daemon_hybrid_cache_t* cache;
        pthread_mutex_lock(&g_daemon_hybrid_cache_lock);
        cache = lzo_hybrid_find_cache(lzo_hybrid_label_dtype(h->label), h->label, alg_name, bits, suffix);
        if (cache && cache->h.prog && cache->h.kernel) {
            *h = cache->h;
            h->cache_slot = cache;
            h->owns_context = 0;
            h->owns_queue = 0;
            h->owns_program = 0;
            pthread_mutex_unlock(&g_daemon_hybrid_cache_lock);
            return 0;
        }
        pthread_mutex_unlock(&g_daemon_hybrid_cache_lock);
    }
    char build_log[8192] = {0};
    char krn_name[128];
    char full_krn_name[128];
    cl_int err;

    h->prog = lzo_load_program_with_dbits(h->ctx, h->dev, alg_name, bits, build_log, sizeof(build_log));
    if (!h->prog) {
        fprintf(stderr, "[HYBRID] %s failed to load %s D_BITS=%d: %s\n",
                h->label, alg_name, bits, build_log[0] ? build_log : "no build log");
        return -1;
    }
    h->owns_program = 1;
    snprintf(krn_name, sizeof(krn_name), "%s_%s", alg_name, suffix);
    h->kernel = clCreateKernel(h->prog, krn_name, &err);
    if (err != CL_SUCCESS || !h->kernel) {
        fprintf(stderr, "[HYBRID] %s clCreateKernel failed for %s err=%d\n", h->label, krn_name, err);
        return -1;
    }
    h->full_kernel = NULL;
    {
        size_t suffix_len = strlen(suffix);
        const char* range_tail = "_range";
        size_t range_tail_len = strlen(range_tail);
        if (suffix_len > range_tail_len &&
            strcmp(suffix + suffix_len - range_tail_len, range_tail) == 0) {
            char full_suffix[96];
            size_t full_len = suffix_len - range_tail_len;
            if (full_len >= sizeof(full_suffix)) full_len = sizeof(full_suffix) - 1;
            memcpy(full_suffix, suffix, full_len);
            full_suffix[full_len] = '\0';
            snprintf(full_krn_name, sizeof(full_krn_name), "%s_%s", alg_name, full_suffix);
            h->full_kernel = clCreateKernel(h->prog, full_krn_name, &err);
            if (err != CL_SUCCESS || !h->full_kernel) {
                fprintf(stderr, "[HYBRID] %s clCreateKernel failed for full-range fast path %s err=%d\n",
                        h->label, full_krn_name, err);
                return -1;
            }
        }
    }
    if (g_daemon_hybrid_cache_enabled) {
        daemon_hybrid_cache_t* cache = lzo_hybrid_alloc_cache_slot();
        if (cache) {
            memset(cache, 0, sizeof(*cache));
            cache->valid = 1;
            cache->dtype = lzo_hybrid_label_dtype(h->label);
            cache->bits = bits;
            strncpy(cache->label, h->label ? h->label : "", sizeof(cache->label) - 1);
            strncpy(cache->alg, alg_name ? alg_name : "", sizeof(cache->alg) - 1);
            strncpy(cache->suffix, suffix ? suffix : "", sizeof(cache->suffix) - 1);
            cache->h = *h;
            cache->h.cache_slot = cache;
            cache->h.owns_context = 1;
            cache->h.owns_queue = 1;
            cache->h.owns_program = 1;
            h->cache_slot = cache;
            h->owns_context = 0;
            h->owns_queue = 0;
            h->owns_program = 0;
        }
    }
    return 0;
}

static int lzo_hybrid_alias_kernel(hybrid_ocl_t* h, const hybrid_ocl_t* owner, const char* label, const char* alg_name, const char* suffix) {
    char krn_name[128];
    char full_krn_name[128];
    cl_int err;

    memset(h, 0, sizeof(*h));
    h->dev = owner->dev;
    h->ctx = owner->ctx;
    h->q = owner->q;
    h->prog = owner->prog;
    h->label = label;
    h->owns_context = 0;
    h->owns_queue = 0;
    h->owns_program = 0;

    snprintf(krn_name, sizeof(krn_name), "%s_%s", alg_name, suffix);
    h->kernel = clCreateKernel(h->prog, krn_name, &err);
    if (err != CL_SUCCESS || !h->kernel) {
        fprintf(stderr, "[HYBRID] %s clCreateKernel failed for shared-context kernel %s err=%d\n", h->label, krn_name, err);
        return -1;
    }
    {
        size_t suffix_len = strlen(suffix);
        const char* range_tail = "_range";
        size_t range_tail_len = strlen(range_tail);
        if (suffix_len > range_tail_len &&
            strcmp(suffix + suffix_len - range_tail_len, range_tail) == 0) {
            char full_suffix[96];
            size_t full_len = suffix_len - range_tail_len;
            if (full_len >= sizeof(full_suffix)) full_len = sizeof(full_suffix) - 1;
            memcpy(full_suffix, suffix, full_len);
            full_suffix[full_len] = '\0';
            snprintf(full_krn_name, sizeof(full_krn_name), "%s_%s", alg_name, full_suffix);
            h->full_kernel = clCreateKernel(h->prog, full_krn_name, &err);
            if (err != CL_SUCCESS || !h->full_kernel) {
                fprintf(stderr, "[HYBRID] %s clCreateKernel failed for shared-context full kernel %s err=%d\n",
                        h->label, full_krn_name, err);
                return -1;
            }
        }
    }
    return 0;
}

static int lzo_write_sparse_payload(FILE* f,
                                    const unsigned char* sparse,
                                    size_t blocks,
                                    size_t worst_blk,
                                    const unsigned int* lens) {
    size_t pack_kb = 1024;
    unsigned char* pack = NULL;
    size_t pack_cap = 0;
    size_t fill = 0;

    if (!f || !sparse || !lens || worst_blk == 0) return -1;
    pack_cap = pack_kb * 1024;
    pack = (unsigned char*)malloc(pack_cap);
    if (!pack) {
        for (size_t i = 0; i < blocks; ++i) {
            size_t clen = (size_t)lens[i];
            if (clen > 0 && fwrite(sparse + i * worst_blk, 1, clen, f) != clen) return -1;
        }
        return 0;
    }

    for (size_t i = 0; i < blocks; ++i) {
        size_t clen = (size_t)lens[i];
        const unsigned char* src = sparse + i * worst_blk;
        if (clen == 0) continue;
        if (clen > pack_cap) {
            if (fill > 0) {
                if (fwrite(pack, 1, fill, f) != fill) {
                    free(pack);
                    return -1;
                }
                fill = 0;
            }
            if (fwrite(src, 1, clen, f) != clen) {
                free(pack);
                return -1;
            }
            continue;
        }
        if (fill + clen > pack_cap) {
            if (fwrite(pack, 1, fill, f) != fill) {
                free(pack);
                return -1;
            }
            fill = 0;
        }
        memcpy(pack + fill, src, clen);
        fill += clen;
    }
    if (fill > 0 && fwrite(pack, 1, fill, f) != fill) {
        free(pack);
        return -1;
    }
    free(pack);
    return 0;
}

static int lzo_write_sparse_split_container(const char* path,
                                            size_t orig_size,
                                            size_t blk_size,
                                            size_t nblk,
                                            const unsigned int* lens,
                                            const unsigned char* gpu_sparse,
                                            size_t gpu_blocks,
                                            const unsigned char* cpu_sparse,
                                            size_t cpu_blocks,
                                            size_t worst_blk,
                                            int alg_id) {
    FILE* f = NULL;
    char* vbuf = NULL;
    uint16_t magic = 0x4C5A;
    unsigned int header[4];
    if (!path || !lens || nblk == 0 || gpu_blocks + cpu_blocks != nblk) return -1;
    f = fopen(path, "wb");
    if (!f) return -1;
    vbuf = (char*)malloc(2 * 1024 * 1024);
    if (vbuf) setvbuf(f, vbuf, _IOFBF, 2 * 1024 * 1024);
    header[0] = (unsigned int)orig_size;
    header[1] = (unsigned int)blk_size;
    header[2] = (unsigned int)nblk;
    header[3] = (unsigned int)alg_id;
    if (fwrite(&magic, 1, 2, f) != 2) goto err;
    if (fwrite(header, 4, 4, f) != 4) goto err;
    if (fwrite(lens, 4, nblk, f) != nblk) goto err;
    if (gpu_blocks > 0 && lzo_write_sparse_payload(f, gpu_sparse, gpu_blocks, worst_blk, lens) != 0) goto err;
    if (cpu_blocks > 0 && lzo_write_sparse_payload(f, cpu_sparse, cpu_blocks, worst_blk, lens + gpu_blocks) != 0) goto err;
    fclose(f);
    free(vbuf);
    return 0;
err:
    if (f) fclose(f);
    free(vbuf);
    return -1;
}

static int lzo_hybrid_set_comp_args(cl_kernel kernel,
                                    cl_mem d_in,
                                    cl_mem d_out,
                                    cl_mem d_len,
                                    cl_uint in_sz,
                                    cl_uint blk,
                                    cl_uint worst,
                                    cl_mem d_dict,
                                    cl_uint pool_size,
                                    cl_uint epoch_base,
                                    cl_uint block_start,
                                    cl_uint block_count) {
    cl_uint num_args = 0;
    cl_int err;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));
    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &epoch_base));
    CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uint), &block_start));
    CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uint), &block_count));
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (err == CL_SUCCESS && num_args >= 13U) {
        cl_mem dbg = d_len;
        cl_uint dbg_flag = 0;
        CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), &dbg));
        CHECK(clSetKernelArg(kernel, 12, sizeof(cl_uint), &dbg_flag));
    }
    return 0;
}

static int lzo_hybrid_set_comp_full_args(cl_kernel kernel,
                                         cl_mem d_in,
                                         cl_mem d_out,
                                         cl_mem d_len,
                                         cl_uint in_sz,
                                         cl_uint blk,
                                         cl_uint worst,
                                         cl_mem d_dict,
                                         cl_uint pool_size,
                                         cl_uint epoch_base) {
    cl_uint num_args = 0;
    cl_int err;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));
    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &epoch_base));
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (err == CL_SUCCESS && num_args >= 11U) {
        cl_mem dbg = d_len;
        cl_uint dbg_flag = 0;
        CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &dbg));
        CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uint), &dbg_flag));
    }
    return 0;
}

static int lzo_hybrid_set_dec_args(cl_kernel kernel,
                                   cl_mem d_comp,
                                   cl_mem d_off,
                                   cl_mem d_len,
                                   cl_mem d_out,
                                   cl_uint blk,
                                   cl_uint orig_size,
                                   cl_uint block_start,
                                   cl_uint block_count) {
    cl_uint num_args = 0;
    cl_mem out_lens = (cl_mem)NULL;
    cl_int err;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_lens));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &orig_size));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &block_start));
    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &block_count));
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (err == CL_SUCCESS && num_args >= 11U) {
        cl_mem dbg = out_lens;
        cl_uint dbg_flag = 0;
        CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &dbg));
        CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uint), &dbg_flag));
    }
    return 0;
}

static int lzo_hybrid_set_dec_full_args(cl_kernel kernel,
                                        cl_mem d_comp,
                                        cl_mem d_off,
                                        cl_mem d_len,
                                        cl_mem d_out,
                                        cl_uint blk,
                                        cl_uint orig_size,
                                        cl_uint nblk) {
    cl_uint num_args = 0;
    cl_mem out_lens = (cl_mem)NULL;
    cl_int err;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_lens));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &orig_size));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &nblk));
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (err == CL_SUCCESS && num_args >= 10U) {
        cl_mem dbg = out_lens;
        cl_uint dbg_flag = 0;
        CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &dbg));
        CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uint), &dbg_flag));
    }
    return 0;
}

static int do_hybrid_compress_mode(const char* in_path,
                                   const char* output_path,
                                   int suppress_non_data,
                                   const char* alg_name,
                                   int comp_level) {
    hybrid_ocl_t gpu, cpu;
    cl_int err;
    uint64_t t_total_start = now_ns();
    uint64_t t_init_start, t_init_end, t_kernel_start, t_kernel_end, t_download_start, t_download_end, t_write_start, t_write_end;
    size_t in_sz = 0, blk = 0, nblk = 0, worst_blk = 0;
    size_t cpu_pct = lzo_hybrid_cpu_share_pct();
    size_t cpu_blocks = 0, gpu_blocks = 0;
    size_t gpu_items = 1, gpu_dict_slots = 1, cpu_slots = 1;
    size_t gpu_local = 1, cpu_local = 1, gpu_global = 1, cpu_global = 1;
    size_t gpu_in_off = 0, cpu_in_off = 0, gpu_in_sz = 0, cpu_in_sz = 0;
    unsigned char* input = NULL;
    unsigned int* lens = NULL;
    unsigned char* gpu_sparse = NULL;
    unsigned char* cpu_sparse = NULL;
    size_t comp_total = 0;
    cl_mem gpu_in = NULL, gpu_out = NULL, gpu_len = NULL, gpu_dict = NULL;
    cl_mem cpu_in = NULL, cpu_out = NULL, cpu_len = NULL, cpu_dict = NULL;
    cl_event ev_gpu = NULL, ev_cpu = NULL;
    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;
    FILE* msg = suppress_non_data ? stderr : stdout;
    int rc = 1;

    memset(&gpu, 0, sizeof(gpu));
    memset(&cpu, 0, sizeof(cpu));
    if (comp_level < 0) comp_level = LZO_DEFAULT_COMP_LEVEL;
    int gpu_level = (g_cli_gpu_level >= 0) ? g_cli_gpu_level : comp_level;
    int cpu_level = (g_cli_cpu_level >= 0) ? g_cli_cpu_level : comp_level;
    input = (unsigned char*)lzo_read_file(in_path, &in_sz);
    if (!input || in_sz == 0) {
        fprintf(stderr, "[HYBRID] failed to read input\n");
        goto cleanup;
    }

    t_init_start = now_ns();
    if ((cpu_pct < 100 && lzo_hybrid_init_device(&gpu, CL_DEVICE_TYPE_GPU, "GPU") != 0) ||
        (cpu_pct > 0 && lzo_hybrid_init_device(&cpu, CL_DEVICE_TYPE_CPU, "CPU") != 0)) {
        fprintf(stderr, "[HYBRID] failed to initialize required GPU/CPU OpenCL devices\n");
        goto cleanup;
    }
    if ((cpu_pct < 100 && lzo_hybrid_load_kernel(&gpu, alg_name, gpu_level, "block_compress_range") != 0) ||
        (cpu_pct > 0 && lzo_hybrid_load_kernel(&cpu, alg_name, cpu_level, "block_compress_range") != 0)) {
        goto cleanup;
    }
    t_init_end = now_ns();

    lzo_choose_blocking_adaptive(input, in_sz, (cpu_pct < 100) ? gpu.dev : cpu.dev, g_cli_fixed_block_bytes, g_cli_fixed_block_exact, &blk, &nblk, 0);
    if (blk == 0 || nblk == 0) goto cleanup;
    worst_blk = lzo_worst(blk);
    cpu_blocks = (nblk * cpu_pct + 99) / 100;
    if (cpu_pct < 100 && cpu_blocks >= nblk && nblk > 1) cpu_blocks = nblk - 1;
    if (cpu_pct > 0 && cpu_blocks == 0) cpu_blocks = 1;
    gpu_blocks = nblk - cpu_blocks;
    gpu_in_off = 0;
    cpu_in_off = gpu_blocks * blk;
    gpu_in_sz = gpu_blocks * blk;
    if (gpu_in_sz > in_sz) gpu_in_sz = in_sz;
    cpu_in_sz = (cpu_in_off < in_sz) ? (in_sz - cpu_in_off) : 0;

    gpu_dict_slots = gpu_blocks ? lzo_hybrid_gpu_items_for_device(gpu.dev, gpu_blocks, gpu_level) : 0;
    gpu_items = gpu_dict_slots ? gpu_dict_slots : 1;
    gpu_local = (g_cli_local_size > 0) ? g_cli_local_size : 1;
    if (gpu_local > gpu_items) gpu_local = gpu_items;
    if (gpu_local == 0) gpu_local = 1;
    gpu_global = ((gpu_items + gpu_local - 1) / gpu_local) * gpu_local;
    cpu_slots = cpu_blocks ? lzo_hybrid_cpu_slots_for_device(cpu.dev, cpu_blocks) : 0;
    cpu_global = cpu_slots;

    lens = (unsigned int*)calloc(nblk, sizeof(unsigned int));
    if (!lens) goto cleanup;

    size_t gpu_dict_bytes = gpu_dict_slots * (1ULL << gpu_level) * sizeof(uint32_t);
    size_t cpu_dict_bytes = cpu_global * (1ULL << cpu_level) * sizeof(uint32_t);
    if (gpu_blocks > 0) {
        gpu_in = lzo_hybrid_cached_buffer(&gpu, 0, gpu_in_sz ? gpu_in_sz : 1, CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_out = lzo_hybrid_cached_buffer(&gpu, 1, gpu_blocks * worst_blk, CL_MEM_WRITE_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_len = lzo_hybrid_cached_buffer(&gpu, 2, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_WRITE, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_dict = lzo_hybrid_cached_buffer(&gpu, 3, gpu_dict_bytes + sizeof(cl_uint), CL_MEM_READ_WRITE, &err); if (err != CL_SUCCESS) goto cleanup;
        CHECK(clEnqueueWriteBuffer(gpu.q, gpu_in, CL_FALSE, 0, gpu_in_sz, input + gpu_in_off, 0, NULL, NULL));
        CHECK(clEnqueueFillBuffer(gpu.q, gpu_dict, &(cl_uint){0}, sizeof(cl_uint), 0, gpu_dict_bytes + sizeof(cl_uint), 0, NULL, NULL));
        lzo_hybrid_set_comp_args(gpu.kernel, gpu_in, gpu_out, gpu_len, (cl_uint)in_sz, (cl_uint)blk, (cl_uint)worst_blk,
                                 gpu_dict, (cl_uint)gpu_dict_slots, 1U, 0U, (cl_uint)gpu_blocks);
    }
    if (cpu_blocks > 0) {
        cpu_in = lzo_hybrid_cached_buffer(&cpu, 0, cpu_in_sz ? cpu_in_sz : 1, CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_out = lzo_hybrid_cached_buffer(&cpu, 1, cpu_blocks * worst_blk, CL_MEM_WRITE_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_len = lzo_hybrid_cached_buffer(&cpu, 2, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_WRITE, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_dict = lzo_hybrid_cached_buffer(&cpu, 3, cpu_dict_bytes + sizeof(cl_uint), CL_MEM_READ_WRITE, &err); if (err != CL_SUCCESS) goto cleanup;
        CHECK(clEnqueueWriteBuffer(cpu.q, cpu_in, CL_FALSE, 0, cpu_in_sz, input + cpu_in_off, 0, NULL, NULL));
        CHECK(clEnqueueFillBuffer(cpu.q, cpu_dict, &(cl_uint){0}, sizeof(cl_uint), 0, cpu_dict_bytes + sizeof(cl_uint), 0, NULL, NULL));
        lzo_hybrid_set_comp_args(cpu.kernel, cpu_in, cpu_out, cpu_len, (cl_uint)in_sz, (cl_uint)blk, (cl_uint)worst_blk,
                                 cpu_dict, (cl_uint)cpu_global, 1U, (cl_uint)gpu_blocks, (cl_uint)cpu_blocks);
    }

    t_kernel_start = now_ns();
    if (gpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(gpu.q, gpu.kernel, 1, NULL, &gpu_global, &gpu_local, 0, NULL, &ev_gpu));
    if (cpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(cpu.q, cpu.kernel, 1, NULL, &cpu_global, &cpu_local, 0, NULL, &ev_cpu));
    if (ev_gpu) clWaitForEvents(1, &ev_gpu);
    if (ev_cpu) clWaitForEvents(1, &ev_cpu);
    t_kernel_end = now_ns();

    t_download_start = now_ns();
    if (gpu_blocks > 0) CHECK(clEnqueueReadBuffer(gpu.q, gpu_len, CL_TRUE, 0, gpu_blocks * sizeof(cl_uint), lens, 0, NULL, NULL));
    if (cpu_blocks > 0) CHECK(clEnqueueReadBuffer(cpu.q, cpu_len, CL_TRUE, 0, cpu_blocks * sizeof(cl_uint), lens + gpu_blocks, 0, NULL, NULL));
    for (size_t i = 0; i < nblk; ++i) comp_total += lens[i];
    if (gpu_blocks > 0) {
        gpu_sparse = (unsigned char*)malloc(gpu_blocks * worst_blk);
        if (!gpu_sparse) goto cleanup;
        CHECK(clEnqueueReadBuffer(gpu.q, gpu_out, CL_TRUE, 0, gpu_blocks * worst_blk, gpu_sparse, 0, NULL, NULL));
    }
    if (cpu_blocks > 0) {
        cpu_sparse = (unsigned char*)malloc(cpu_blocks * worst_blk);
        if (!cpu_sparse) goto cleanup;
        CHECK(clEnqueueReadBuffer(cpu.q, cpu_out, CL_TRUE, 0, cpu_blocks * worst_blk, cpu_sparse, 0, NULL, NULL));
    }
    t_download_end = now_ns();

    t_write_start = now_ns();
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        if (lzo_write_sparse_split_container(output_path, in_sz, blk, nblk, lens,
                                             gpu_sparse, gpu_blocks,
                                             cpu_sparse, cpu_blocks,
                                             worst_blk, alg_id) != 0) {
            fprintf(stderr, "[HYBRID] failed to write compressed output\n");
            goto cleanup;
        }
    }
    t_write_end = now_ns();

    {
        double gpu_us = event_elapsed_us(ev_gpu);
        double cpu_us = event_elapsed_us(ev_cpu);
        unsigned long overall_us = (unsigned long)((now_ns() - t_total_start) / 1000ULL);
        fprintf(msg,
                "[HYBRID][C] %s : %zu -> %zu (%.2f:1) in %.2f ms blocks=%zu gpu=%zu cpu=%zu cpu_slots=%zu gpu_level=%d cpu_level=%d span=%.2f ms gpu_kernel=%.2f ms cpu_kernel=%.2f ms init_load=%.2f ms download_merge=%.2f ms write=%.2f ms\n",
                in_path, in_sz, comp_total,
                (double)in_sz / (double)(comp_total ? comp_total : 1),
                overall_us / 1000.0, nblk, gpu_blocks, cpu_blocks, cpu_slots, gpu_level, cpu_level,
                (double)(t_kernel_end - t_kernel_start) / 1000000.0,
                gpu_us / 1000.0, cpu_us / 1000.0,
                (double)(t_init_end - t_init_start) / 1000000.0,
                (double)(t_download_end - t_download_start) / 1000000.0,
                (double)(t_write_end - t_write_start) / 1000000.0);
    }
    rc = 0;

cleanup:
    if (ev_gpu) clReleaseEvent(ev_gpu);
    if (ev_cpu) clReleaseEvent(ev_cpu);
    if (!g_daemon_hybrid_cache_enabled) {
        if (gpu_in) clReleaseMemObject(gpu_in);
        if (gpu_out) clReleaseMemObject(gpu_out);
        if (gpu_len) clReleaseMemObject(gpu_len);
        if (gpu_dict) clReleaseMemObject(gpu_dict);
        if (cpu_in) clReleaseMemObject(cpu_in);
        if (cpu_out) clReleaseMemObject(cpu_out);
        if (cpu_len) clReleaseMemObject(cpu_len);
        if (cpu_dict) clReleaseMemObject(cpu_dict);
    }
    lzo_hybrid_release(&gpu);
    lzo_hybrid_release(&cpu);
    free(input);
    free(lens);
    free(gpu_sparse);
    free(cpu_sparse);
    return rc;
}

static int do_hybrid_decompress_mode(const char* lz_path,
                                     const char* output_path,
                                     int suppress_non_data) {
    hybrid_ocl_t gpu, cpu;
    FILE* f = NULL;
    cl_int err;
    uint64_t t_total_start = now_ns();
    uint64_t t_init_start, t_init_end, t_kernel_start, t_kernel_end, t_download_start, t_download_end, t_write_start, t_write_end;
    uint16_t magic = 0;
    uint32_t header[4] = {0};
    uint32_t orig_sz = 0, blk_sz = 0, nblk32 = 0, alg_id = 0;
    size_t nblk = 0, cpu_pct = lzo_hybrid_cpu_share_pct(), cpu_blocks = 0, gpu_blocks = 0;
    size_t comp_sz = 0, gpu_comp_sz = 0, cpu_comp_sz = 0, gpu_out_sz = 0, cpu_out_sz = 0;
    uint32_t* lens = NULL;
    uint32_t* gpu_off = NULL;
    uint32_t* cpu_off = NULL;
    unsigned char* comp = NULL;
    unsigned char* gpu_out_h = NULL;
    unsigned char* cpu_out_h = NULL;
    cl_mem gpu_comp = NULL, gpu_off_d = NULL, gpu_len_d = NULL, gpu_out = NULL;
    cl_mem cpu_comp = NULL, cpu_off_d = NULL, cpu_len_d = NULL, cpu_out = NULL;
    cl_event ev_gpu = NULL, ev_cpu = NULL;
    const char* alg_name = NULL;
    size_t gpu_local = (g_cli_local_size > 0) ? g_cli_local_size : 1;
    size_t cpu_local = 1, gpu_global = 1, cpu_global = 1;
    FILE* msg = suppress_non_data ? stderr : stdout;
    int rc = 1;

    memset(&gpu, 0, sizeof(gpu));
    memset(&cpu, 0, sizeof(cpu));
    f = fopen(lz_path, "rb");
    if (!f) goto cleanup;
    if (fread(&magic, 1, 2, f) != 2 || magic != MAGIC) goto cleanup;
    if (fread(header, 4, 4, f) != 4) goto cleanup;
    orig_sz = header[0]; blk_sz = header[1]; nblk32 = header[2]; alg_id = header[3];
    nblk = nblk32;
    if (nblk == 0 || blk_sz == 0) goto cleanup;
    alg_name = (alg_id == 1) ? "lzo1y" : "lzo1x";

    lens = (uint32_t*)malloc(nblk * sizeof(uint32_t));
    if (!lens) goto cleanup;
    if (fread(lens, sizeof(uint32_t), nblk, f) != nblk) goto cleanup;
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, pos, SEEK_SET);
    if (end < pos) goto cleanup;
    comp_sz = (size_t)(end - pos);
    comp = (unsigned char*)malloc(comp_sz ? comp_sz : 1);
    if (!comp) goto cleanup;
    if (comp_sz > 0 && fread(comp, 1, comp_sz, f) != comp_sz) goto cleanup;
    fclose(f);
    f = NULL;

    cpu_blocks = (nblk * cpu_pct + 99) / 100;
    if (cpu_pct < 100 && cpu_blocks >= nblk && nblk > 1) cpu_blocks = nblk - 1;
    if (cpu_pct > 0 && cpu_blocks == 0) cpu_blocks = 1;
    gpu_blocks = nblk - cpu_blocks;

    if (gpu_blocks > 0) gpu_off = (uint32_t*)calloc(gpu_blocks, sizeof(uint32_t));
    if (cpu_blocks > 0) cpu_off = (uint32_t*)calloc(cpu_blocks, sizeof(uint32_t));
    if ((gpu_blocks > 0 && !gpu_off) || (cpu_blocks > 0 && !cpu_off)) goto cleanup;
    for (size_t i = 0; i < gpu_blocks; ++i) {
        gpu_off[i] = (uint32_t)gpu_comp_sz;
        gpu_comp_sz += lens[i];
    }
    for (size_t i = 0; i < cpu_blocks; ++i) {
        cpu_off[i] = (uint32_t)cpu_comp_sz;
        cpu_comp_sz += lens[gpu_blocks + i];
    }
    gpu_out_sz = gpu_blocks * (size_t)blk_sz;
    cpu_out_sz = (size_t)orig_sz - gpu_out_sz;
    if (gpu_blocks > 0) {
        gpu_global = gpu_blocks;
        if (gpu_local > gpu_global) gpu_local = gpu_global;
        if (gpu_local == 0) gpu_local = 1;
        gpu_global = ((gpu_global + gpu_local - 1) / gpu_local) * gpu_local;
    }
    if (cpu_blocks > 0) {
        cpu_global = lzo_hybrid_cpu_slots_for_device(NULL, cpu_blocks);
        if (cpu_global > cpu_blocks) cpu_global = cpu_blocks;
        if (cpu_global == 0) cpu_global = 1;
    }

    t_init_start = now_ns();
    if ((gpu_blocks > 0 && lzo_hybrid_init_device(&gpu, CL_DEVICE_TYPE_GPU, "GPU") != 0) ||
        (cpu_blocks > 0 && lzo_hybrid_init_device(&cpu, CL_DEVICE_TYPE_CPU, "CPU") != 0)) {
        fprintf(stderr, "[HYBRID] failed to initialize required GPU/CPU OpenCL devices\n");
        goto cleanup;
    }
    if (cpu_blocks > 0) cpu_global = lzo_hybrid_cpu_slots_for_device(cpu.dev, cpu_blocks);
    if ((gpu_blocks > 0 && lzo_hybrid_load_kernel(&gpu, alg_name, LZO_DEFAULT_COMP_LEVEL, "block_decompress_range") != 0) ||
        (cpu_blocks > 0 && lzo_hybrid_load_kernel(&cpu, alg_name, LZO_DEFAULT_COMP_LEVEL, "block_decompress_range") != 0)) {
        goto cleanup;
    }
    t_init_end = now_ns();

    if (gpu_blocks > 0) {
        gpu_comp = lzo_hybrid_cached_buffer(&gpu, 0, gpu_comp_sz ? gpu_comp_sz : 1, CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_off_d = lzo_hybrid_cached_buffer(&gpu, 1, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_len_d = lzo_hybrid_cached_buffer(&gpu, 2, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        gpu_out = lzo_hybrid_cached_buffer(&gpu, 3, gpu_out_sz ? gpu_out_sz : 1, CL_MEM_WRITE_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        CHECK(clEnqueueWriteBuffer(gpu.q, gpu_comp, CL_FALSE, 0, gpu_comp_sz, comp, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(gpu.q, gpu_off_d, CL_FALSE, 0, gpu_blocks * sizeof(cl_uint), gpu_off, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(gpu.q, gpu_len_d, CL_FALSE, 0, gpu_blocks * sizeof(cl_uint), lens, 0, NULL, NULL));
        lzo_hybrid_set_dec_args(gpu.kernel, gpu_comp, gpu_off_d, gpu_len_d, gpu_out,
                                blk_sz, orig_sz, 0U, (cl_uint)gpu_blocks);
    }
    if (cpu_blocks > 0) {
        cpu_comp = lzo_hybrid_cached_buffer(&cpu, 0, cpu_comp_sz ? cpu_comp_sz : 1, CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_off_d = lzo_hybrid_cached_buffer(&cpu, 1, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_len_d = lzo_hybrid_cached_buffer(&cpu, 2, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        cpu_out = lzo_hybrid_cached_buffer(&cpu, 3, cpu_out_sz ? cpu_out_sz : 1, CL_MEM_WRITE_ONLY, &err); if (err != CL_SUCCESS) goto cleanup;
        CHECK(clEnqueueWriteBuffer(cpu.q, cpu_comp, CL_FALSE, 0, cpu_comp_sz, comp + gpu_comp_sz, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(cpu.q, cpu_off_d, CL_FALSE, 0, cpu_blocks * sizeof(cl_uint), cpu_off, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(cpu.q, cpu_len_d, CL_FALSE, 0, cpu_blocks * sizeof(cl_uint), lens + gpu_blocks, 0, NULL, NULL));
        lzo_hybrid_set_dec_args(cpu.kernel, cpu_comp, cpu_off_d, cpu_len_d, cpu_out,
                                blk_sz, orig_sz, (cl_uint)gpu_blocks, (cl_uint)cpu_blocks);
    }

    t_kernel_start = now_ns();
    if (gpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(gpu.q, gpu.kernel, 1, NULL, &gpu_global, &gpu_local, 0, NULL, &ev_gpu));
    if (cpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(cpu.q, cpu.kernel, 1, NULL, &cpu_global, &cpu_local, 0, NULL, &ev_cpu));
    if (ev_gpu) clWaitForEvents(1, &ev_gpu);
    if (ev_cpu) clWaitForEvents(1, &ev_cpu);
    t_kernel_end = now_ns();

    t_download_start = now_ns();
    if (gpu_blocks > 0) {
        gpu_out_h = (unsigned char*)malloc(gpu_out_sz ? gpu_out_sz : 1);
        if (!gpu_out_h) goto cleanup;
        CHECK(clEnqueueReadBuffer(gpu.q, gpu_out, CL_TRUE, 0, gpu_out_sz, gpu_out_h, 0, NULL, NULL));
    }
    if (cpu_blocks > 0) {
        cpu_out_h = (unsigned char*)malloc(cpu_out_sz ? cpu_out_sz : 1);
        if (!cpu_out_h) goto cleanup;
        CHECK(clEnqueueReadBuffer(cpu.q, cpu_out, CL_TRUE, 0, cpu_out_sz, cpu_out_h, 0, NULL, NULL));
    }
    t_download_end = now_ns();

    t_write_start = now_ns();
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        FILE* out = fopen(output_path, "wb");
        if (!out) goto cleanup;
        if (gpu_out_sz > 0 && fwrite(gpu_out_h, 1, gpu_out_sz, out) != gpu_out_sz) { fclose(out); goto cleanup; }
        if (cpu_out_sz > 0 && fwrite(cpu_out_h, 1, cpu_out_sz, out) != cpu_out_sz) { fclose(out); goto cleanup; }
        fclose(out);
    }
    t_write_end = now_ns();

    {
        double gpu_us = event_elapsed_us(ev_gpu);
        double cpu_us = event_elapsed_us(ev_cpu);
        unsigned long overall_us = (unsigned long)((now_ns() - t_total_start) / 1000ULL);
        fprintf(msg,
                "[HYBRID][D] %s : %zu -> %u in %.2f ms blocks=%zu gpu=%zu cpu=%zu cpu_slots=%zu span=%.2f ms gpu_kernel=%.2f ms cpu_kernel=%.2f ms init_load=%.2f ms download=%.2f ms write=%.2f ms\n",
                lz_path, comp_sz, orig_sz,
                overall_us / 1000.0, nblk, gpu_blocks, cpu_blocks, cpu_global,
                (double)(t_kernel_end - t_kernel_start) / 1000000.0,
                gpu_us / 1000.0, cpu_us / 1000.0,
                (double)(t_init_end - t_init_start) / 1000000.0,
                (double)(t_download_end - t_download_start) / 1000000.0,
                (double)(t_write_end - t_write_start) / 1000000.0);
    }
    rc = 0;

cleanup:
    if (f) fclose(f);
    if (ev_gpu) clReleaseEvent(ev_gpu);
    if (ev_cpu) clReleaseEvent(ev_cpu);
    if (!g_daemon_hybrid_cache_enabled) {
        if (gpu_comp) clReleaseMemObject(gpu_comp);
        if (gpu_off_d) clReleaseMemObject(gpu_off_d);
        if (gpu_len_d) clReleaseMemObject(gpu_len_d);
        if (gpu_out) clReleaseMemObject(gpu_out);
        if (cpu_comp) clReleaseMemObject(cpu_comp);
        if (cpu_off_d) clReleaseMemObject(cpu_off_d);
        if (cpu_len_d) clReleaseMemObject(cpu_len_d);
        if (cpu_out) clReleaseMemObject(cpu_out);
    }
    lzo_hybrid_release(&gpu);
    lzo_hybrid_release(&cpu);
    free(lens);
    free(gpu_off);
    free(cpu_off);
    free(comp);
    free(gpu_out_h);
    free(cpu_out_h);
    return rc;
}

#if !defined(_WIN32)
static pthread_mutex_t g_daemon_hybrid_call_lock = PTHREAD_MUTEX_INITIALIZER;

int lzo_hybrid_daemon_hybrid_file_request(char operation,
                                          const char* input_path,
                                          const char* output_path,
                                          int alg,
                                          int level,
                                          int block_size,
                                          uint32_t local_size,
                                          uint32_t cpu_share_pct,
                                          uint32_t cpu_threads,
                                          unsigned long* elapsed_us)
{
    int rc;
    uint64_t t0;
    size_t old_cpu_threads = g_cli_cpu_threads;
    int old_cpu_threads_set = g_cli_cpu_threads_set;
    size_t old_cpu_share_pct = g_cli_cpu_share_pct;
    int old_gpu_ratio_set = g_cli_gpu_ratio_set;
    int old_adaptive_enabled = g_cli_adaptive_enabled;
    size_t old_fixed_block_bytes = g_cli_fixed_block_bytes;
    int old_fixed_block_exact = g_cli_fixed_block_exact;
    size_t old_local_size = g_cli_local_size;
    const char* alg_name = (alg == 1) ? "lzo1y" : "lzo1x";

    if (cpu_share_pct > 100U) cpu_share_pct = 100U;
    if (level < 0) level = LZO_DEFAULT_COMP_LEVEL;

    pthread_mutex_lock(&g_daemon_hybrid_call_lock);
    lzo_hybrid_set_daemon_cache_enabled(1);
    g_cli_cpu_share_pct = cpu_share_pct;
    g_cli_gpu_ratio_set = 1;
    g_cli_adaptive_enabled = 0;
    g_cli_cpu_threads = cpu_threads;
    g_cli_cpu_threads_set = cpu_threads > 0 ? 1 : 0;
    g_cli_fixed_block_bytes = block_size > 0 ? (size_t)block_size : 0;
    g_cli_fixed_block_exact = block_size > 0 ? 1 : 0;
    g_cli_local_size = local_size;

    t0 = now_ns();
    if (operation == 'C') {
        rc = do_hybrid_compress_mode(input_path, output_path, 1, alg_name, level);
    } else {
        rc = do_hybrid_decompress_mode(input_path, output_path, 1);
    }
    if (elapsed_us) *elapsed_us = (unsigned long)((now_ns() - t0) / 1000ULL);

    g_cli_cpu_threads = old_cpu_threads;
    g_cli_cpu_threads_set = old_cpu_threads_set;
    g_cli_cpu_share_pct = old_cpu_share_pct;
    g_cli_gpu_ratio_set = old_gpu_ratio_set;
    g_cli_adaptive_enabled = old_adaptive_enabled;
    g_cli_fixed_block_bytes = old_fixed_block_bytes;
    g_cli_fixed_block_exact = old_fixed_block_exact;
    g_cli_local_size = old_local_size;
    lzo_hybrid_set_daemon_cache_enabled(0);
    pthread_mutex_unlock(&g_daemon_hybrid_call_lock);
    return rc;
}
#endif


/* Prototypes for extracted helpers to keep main concise */
static int do_decompress_mode(const char* lz_path, const char* output_path, int output_explicit, int suppress_non_data);
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int suppress_non_data, const char* alg_name, int comp_level);
static int run_lzo_accel_bench(const char *in_path, const char *alg_name, int comp_level, double bench_seconds);

/* Implementations: wrappers that use the shared core backend (lzo_hybrid_core.c)
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
    if (lzo_hybrid_enabled()) {
        return do_hybrid_compress_mode(in_path, output_path, suppress_non_data, alg_name, comp_level);
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

    int standard_copy = lzo_resolve_standard_copy(dev);
    size_t block_size = g_cli_fixed_block_bytes;
    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;

    lzo_hybrid_workspace_t ws;
    lzo_hybrid_workspace_init(&ws);

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

    int ret = lzo_compress_core(ctx, q, dev, krn_c, in_path, output_path, &params, 0, &ws, &time_us, &output_size, &t_out);
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
    if (lzo_hybrid_enabled()) {
        return do_hybrid_decompress_mode(lz_path, output_path, suppress_non_data);
    }
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
    char decomp_prog_name[64];

    uint64_t tk1 = now_ns();
    /* Fallback to base decompressor */
    if (!prog_d) {
        if (lzo_format_program_name(decomp_prog_name, sizeof(decomp_prog_name), decomp_base, debug_counters) != 0) {
            fprintf(stderr, "error: decompressor program name too long for %s\n", decomp_base);
            return 1;
        }
        prog_d = lzo_load_program_with_dbits(ctx, dev, decomp_prog_name, decomp_bits, build_log, sizeof(build_log));
        if (!prog_d) {
            if (build_log[0]) {
                fprintf(stderr, "error: unable to load decompressor for %s (D_BITS=%d): %s\n", decomp_prog_name, decomp_bits, build_log);
            } else {
                fprintf(stderr, "error: unable to load decompressor for %s (D_BITS=%d)\n", decomp_prog_name, decomp_bits);
            }
            return 1;
        }
    }

    char krn_name[64];
    cl_kernel krn_d = NULL;

    snprintf(krn_name, sizeof(krn_name), "%s_block_decompress_range", alg_name);
    krn_d = clCreateKernel(prog_d, krn_name, &err);
    if (err != CL_SUCCESS || !krn_d) {
        fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err);
        clReleaseProgram(prog_d);
        return 1;
    }

    uint64_t tk2 = now_ns();
    g_kernel_load_us = (unsigned long)((tk2 - tk1) / 1000);

    lzo_hybrid_workspace_t ws;
    lzo_hybrid_workspace_init(&ws);
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
    lzo_hybrid_workspace_free(&ws);

    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return rc;
}

typedef struct {
    double comp_kernel_event_ms;
    double dec_kernel_event_ms;
    double payload_ratio_pct;
    double container_ratio_pct;
    size_t payload_size;
    size_t container_size;
    size_t comp_gpu_global;
    size_t comp_gpu_dict_slots;
    size_t comp_cpu_global;
    size_t dec_gpu_global;
    size_t dec_cpu_global;
    int ok;
} accel_bench_sample_t;

typedef struct {
    cl_mem in;
    cl_mem out;
    cl_mem len;
    cl_mem dict;
    cl_mem comp_d;
    cl_mem off_d;
    cl_mem len_d;
    cl_mem dec_d;
    size_t in_cap;
    size_t out_cap;
    size_t len_cap;
    size_t dict_cap;
    size_t comp_cap;
    size_t off_cap;
    size_t len_d_cap;
    size_t dec_cap;
    unsigned char* sparse;
    unsigned char* dec_out;
    uint32_t* off;
    size_t sparse_cap;
    size_t dec_out_cap;
    size_t off_host_cap;
    uint32_t epoch_base;
} accel_device_ws_t;

typedef struct {
    accel_device_ws_t gpu;
    accel_device_ws_t cpu;
    unsigned int* lens;
    unsigned char* compact;
    size_t lens_cap;
    size_t compact_cap;
} accel_run_ws_t;

static void accel_release_device_ws(accel_device_ws_t* ws) {
    if (!ws) return;
    if (ws->in) clReleaseMemObject(ws->in);
    if (ws->out) clReleaseMemObject(ws->out);
    if (ws->len) clReleaseMemObject(ws->len);
    if (ws->dict) clReleaseMemObject(ws->dict);
    if (ws->comp_d) clReleaseMemObject(ws->comp_d);
    if (ws->off_d) clReleaseMemObject(ws->off_d);
    if (ws->len_d) clReleaseMemObject(ws->len_d);
    if (ws->dec_d) clReleaseMemObject(ws->dec_d);
    free(ws->sparse);
    free(ws->dec_out);
    free(ws->off);
    memset(ws, 0, sizeof(*ws));
}

static void accel_release_run_ws(accel_run_ws_t* ws) {
    if (!ws) return;
    accel_release_device_ws(&ws->gpu);
    accel_release_device_ws(&ws->cpu);
    free(ws->lens);
    free(ws->compact);
    memset(ws, 0, sizeof(*ws));
}

static int accel_ensure_cl_buffer(cl_context ctx, cl_mem* mem, size_t* cap, size_t need, cl_mem_flags flags) {
    cl_int err;
    if (need == 0) need = 1;
    if (*mem && *cap >= need) return 0;
    if (*mem) {
        clReleaseMemObject(*mem);
        *mem = NULL;
        *cap = 0;
    }
    *mem = clCreateBuffer(ctx, flags, need, NULL, &err);
    if (err != CL_SUCCESS || !*mem) return -1;
    *cap = need;
    return 1;
}

static int accel_ensure_host(void** ptr, size_t* cap, size_t need) {
    void* p;
    if (need == 0) need = 1;
    if (*ptr && *cap >= need) return 0;
    p = realloc(*ptr, need);
    if (!p) return -1;
    *ptr = p;
    *cap = need;
    return 0;
}

static size_t parse_accel_cpu_shares(size_t* shares, size_t max_shares) {
    size_t n = 0;
    if (max_shares == 0) return 0;
    shares[n++] = (g_cli_gpu_ratio_set || g_cli_adaptive_enabled) ? g_cli_cpu_share_pct : 0;
    return n;
}

static size_t parse_accel_cpu_slots_list(size_t* slots, size_t max_slots) {
    if (max_slots == 0) return 0;
    slots[0] = (g_cli_cpu_threads_set && g_cli_cpu_threads > 0) ? g_cli_cpu_threads : 4;
    return 1;
}

static int accel_init_kernel_pair(hybrid_ocl_t* comp, hybrid_ocl_t* dec, cl_device_type dtype, const char* label, const char* alg_name, int comp_level) {
    int dec_level = LZO_DEFAULT_COMP_LEVEL;
    if (lzo_hybrid_init_device(comp, dtype, label) != 0) return -1;
    if (lzo_hybrid_load_kernel(comp, alg_name, comp_level, "block_compress_range") != 0) return -1;
    if (dtype == CL_DEVICE_TYPE_GPU && comp_level == dec_level) {
        if (lzo_hybrid_alias_kernel(dec, comp, label, alg_name, "block_decompress_range") != 0) return -1;
    } else {
        if (lzo_hybrid_init_device(dec, dtype, label) != 0) return -1;
        if (lzo_hybrid_load_kernel(dec, alg_name, dec_level, "block_decompress_range") != 0) return -1;
    }
    return 0;
}

static int run_accel_one_ratio(const unsigned char* input,
                               size_t in_sz,
                               size_t blk,
                               size_t nblk,
                               const char* alg_name,
                               int gpu_level,
                               int cpu_level,
                               size_t cpu_pct,
                               hybrid_ocl_t* gpu_c,
                               hybrid_ocl_t* cpu_c,
                               hybrid_ocl_t* gpu_d,
                               hybrid_ocl_t* cpu_d,
                               accel_run_ws_t* ws,
                               accel_bench_sample_t* out) {
    size_t worst_blk = lzo_worst(blk);
    size_t cpu_blocks = (nblk * cpu_pct + 99) / 100;
    size_t gpu_blocks = nblk - cpu_blocks;
    size_t gpu_items = 0, gpu_dict_slots = 0, cpu_slots = 0;
    size_t gpu_local = (g_cli_local_size > 0) ? g_cli_local_size : 1;
    size_t cpu_local = 1, gpu_global = 0, cpu_global = 0;
    size_t dec_gpu_global = 0, dec_cpu_global = 0;
    size_t gpu_dict_bytes = 0, cpu_dict_bytes = 0;
    size_t gpu_in_off = 0, cpu_in_off = 0, gpu_in_sz = 0, cpu_in_sz = 0;
    unsigned int* lens = NULL;
    unsigned int* gpu_lens = NULL;
    unsigned int* cpu_lens = NULL;
    uint32_t* gpu_off = NULL;
    uint32_t* cpu_off = NULL;
    unsigned char* cpu_sparse = NULL;
    unsigned char* compact = NULL;
    unsigned char* gpu_dec_out = NULL;
    unsigned char* cpu_dec_out = NULL;
    size_t comp_total = 0, compact_off = 0;
    size_t gpu_comp_total = 0, cpu_comp_total = 0;
    size_t gpu_out_sz = 0;
    size_t cpu_out_sz = 0;
    cl_mem gpu_in = NULL, gpu_out = NULL, gpu_len = NULL, gpu_dict = NULL;
    cl_mem cpu_in = NULL, cpu_out = NULL, cpu_len = NULL, cpu_dict = NULL;
    cl_mem gpu_comp_d = NULL, gpu_off_d = NULL, gpu_len_d = NULL, gpu_dec_d = NULL;
    cl_mem cpu_comp_d = NULL, cpu_off_d = NULL, cpu_len_d = NULL, cpu_dec_d = NULL;
    cl_event ev_gpu_c = NULL, ev_cpu_c = NULL, ev_gpu_d = NULL, ev_cpu_d = NULL;
    uint32_t gpu_epoch = 1U, cpu_epoch = 1U;
    int gpu_dict_reset = 0, cpu_dict_reset = 0;
    int rc = -1;

    memset(out, 0, sizeof(*out));
    if (nblk == 0 || blk == 0 || !ws) return -1;
    if (cpu_blocks > nblk) cpu_blocks = nblk;
    gpu_blocks = nblk - cpu_blocks;
    gpu_out_sz = gpu_blocks * blk;
    if (gpu_out_sz > in_sz) gpu_out_sz = in_sz;
    cpu_out_sz = in_sz - gpu_out_sz;
    gpu_in_off = 0;
    cpu_in_off = gpu_blocks * blk;
    gpu_in_sz = gpu_blocks * blk;
    if (gpu_in_sz > in_sz) gpu_in_sz = in_sz;
    cpu_in_sz = (cpu_in_off < in_sz) ? (in_sz - cpu_in_off) : 0;

    if (accel_ensure_host((void**)&ws->lens, &ws->lens_cap, nblk * sizeof(unsigned int)) != 0) goto cleanup;
    lens = ws->lens;
    memset(lens, 0, nblk * sizeof(unsigned int));

    if (gpu_blocks > 0) {
        gpu_dict_slots = lzo_hybrid_gpu_items_for_device(gpu_c->dev, gpu_blocks, gpu_level);
        gpu_items = gpu_dict_slots ? gpu_dict_slots : 1;
        if (gpu_local > gpu_items) gpu_local = gpu_items;
        if (gpu_local == 0) gpu_local = 1;
        gpu_global = ((gpu_items + gpu_local - 1) / gpu_local) * gpu_local;
        gpu_lens = lens;
        if (accel_ensure_cl_buffer(gpu_c->ctx, &ws->gpu.in, &ws->gpu.in_cap, gpu_in_sz ? gpu_in_sz : 1, CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(gpu_c->ctx, &ws->gpu.out, &ws->gpu.out_cap, gpu_blocks * worst_blk, CL_MEM_WRITE_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(gpu_c->ctx, &ws->gpu.len, &ws->gpu.len_cap, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_WRITE) < 0) goto cleanup;
        {
            gpu_dict_bytes = gpu_dict_slots * (1ULL << gpu_level) * sizeof(uint32_t);
            int grow = accel_ensure_cl_buffer(gpu_c->ctx, &ws->gpu.dict, &ws->gpu.dict_cap, gpu_dict_bytes + sizeof(cl_uint), CL_MEM_READ_WRITE);
            if (grow < 0) goto cleanup;
            gpu_dict_reset = (grow > 0);
        }
        gpu_in = ws->gpu.in; gpu_out = ws->gpu.out; gpu_len = ws->gpu.len; gpu_dict = ws->gpu.dict;
        if (ws->gpu.epoch_base == 0 || ws->gpu.epoch_base + (uint32_t)gpu_blocks + 2U > 4095U) {
            gpu_dict_reset = 1;
        }
    }
    if (cpu_blocks > 0) {
        cpu_slots = lzo_hybrid_cpu_slots_for_device(cpu_c->dev, cpu_blocks);
        cpu_global = cpu_slots;
        cpu_lens = lens + gpu_blocks;
        if (accel_ensure_host((void**)&ws->cpu.sparse, &ws->cpu.sparse_cap, cpu_blocks * worst_blk) != 0) goto cleanup;
        cpu_sparse = ws->cpu.sparse;
        if (accel_ensure_cl_buffer(cpu_c->ctx, &ws->cpu.in, &ws->cpu.in_cap, cpu_in_sz ? cpu_in_sz : 1, CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(cpu_c->ctx, &ws->cpu.out, &ws->cpu.out_cap, cpu_blocks * worst_blk, CL_MEM_WRITE_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(cpu_c->ctx, &ws->cpu.len, &ws->cpu.len_cap, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_WRITE) < 0) goto cleanup;
        {
            cpu_dict_bytes = cpu_global * (1ULL << cpu_level) * sizeof(uint32_t);
            int grow = accel_ensure_cl_buffer(cpu_c->ctx, &ws->cpu.dict, &ws->cpu.dict_cap, cpu_dict_bytes + sizeof(cl_uint), CL_MEM_READ_WRITE);
            if (grow < 0) goto cleanup;
            cpu_dict_reset = (grow > 0);
        }
        cpu_in = ws->cpu.in; cpu_out = ws->cpu.out; cpu_len = ws->cpu.len; cpu_dict = ws->cpu.dict;
        if (ws->cpu.epoch_base == 0 || ws->cpu.epoch_base + (uint32_t)cpu_blocks + 2U > 4095U) {
            cpu_dict_reset = 1;
        }
    }

    if (gpu_blocks > 0) {
        CHECK(clEnqueueWriteBuffer(gpu_c->q, gpu_in, CL_FALSE, 0, gpu_in_sz, input + gpu_in_off, 0, NULL, NULL));
        if (gpu_dict_reset) {
            CHECK(clEnqueueFillBuffer(gpu_c->q, gpu_dict, &(cl_uint){0}, sizeof(cl_uint), 0, gpu_dict_bytes + sizeof(cl_uint), 0, NULL, NULL));
            ws->gpu.epoch_base = 1U;
        } else {
            cl_uint zero = 0;
            CHECK(clEnqueueWriteBuffer(gpu_c->q, gpu_dict, CL_FALSE, gpu_dict_bytes, sizeof(zero), &zero, 0, NULL, NULL));
        }
        gpu_epoch = ws->gpu.epoch_base ? ws->gpu.epoch_base : 1U;
        ws->gpu.epoch_base = gpu_epoch + (uint32_t)gpu_blocks + 1U;
    }
    if (cpu_blocks > 0) {
        CHECK(clEnqueueWriteBuffer(cpu_c->q, cpu_in, CL_FALSE, 0, cpu_in_sz, input + cpu_in_off, 0, NULL, NULL));
        if (cpu_dict_reset) {
            CHECK(clEnqueueFillBuffer(cpu_c->q, cpu_dict, &(cl_uint){0}, sizeof(cl_uint), 0, cpu_dict_bytes + sizeof(cl_uint), 0, NULL, NULL));
            ws->cpu.epoch_base = 1U;
        } else {
            cl_uint zero = 0;
            CHECK(clEnqueueWriteBuffer(cpu_c->q, cpu_dict, CL_FALSE, cpu_dict_bytes, sizeof(zero), &zero, 0, NULL, NULL));
        }
        cpu_epoch = ws->cpu.epoch_base ? ws->cpu.epoch_base : 1U;
        ws->cpu.epoch_base = cpu_epoch + (uint32_t)cpu_blocks + 1U;
    }
    if (gpu_blocks > 0) clFinish(gpu_c->q);
    if (cpu_blocks > 0) clFinish(cpu_c->q);

    if (gpu_blocks > 0) {
        if (cpu_blocks == 0 && gpu_c->full_kernel) {
            lzo_hybrid_set_comp_full_args(gpu_c->full_kernel, gpu_in, gpu_out, gpu_len, (cl_uint)in_sz, (cl_uint)blk, (cl_uint)worst_blk,
                                          gpu_dict, (cl_uint)gpu_dict_slots, gpu_epoch);
        } else {
            lzo_hybrid_set_comp_args(gpu_c->kernel, gpu_in, gpu_out, gpu_len, (cl_uint)in_sz, (cl_uint)blk, (cl_uint)worst_blk,
                                     gpu_dict, (cl_uint)gpu_dict_slots, gpu_epoch, 0U, (cl_uint)gpu_blocks);
        }
    }
    if (cpu_blocks > 0) {
        lzo_hybrid_set_comp_args(cpu_c->kernel, cpu_in, cpu_out, cpu_len, (cl_uint)in_sz, (cl_uint)blk, (cl_uint)worst_blk,
                                 cpu_dict, (cl_uint)cpu_global, cpu_epoch, (cl_uint)gpu_blocks, (cl_uint)cpu_blocks);
    }
    if (gpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(gpu_c->q, (cpu_blocks == 0 && gpu_c->full_kernel) ? gpu_c->full_kernel : gpu_c->kernel, 1, NULL, &gpu_global, &gpu_local, 0, NULL, &ev_gpu_c));
    if (cpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(cpu_c->q, cpu_c->kernel, 1, NULL, &cpu_global, &cpu_local, 0, NULL, &ev_cpu_c));
    if (ev_gpu_c) clWaitForEvents(1, &ev_gpu_c);
    if (ev_cpu_c) clWaitForEvents(1, &ev_cpu_c);
    {
        double gpu_ms = event_elapsed_us(ev_gpu_c) / 1000.0;
        double cpu_ms = event_elapsed_us(ev_cpu_c) / 1000.0;
        out->comp_kernel_event_ms = (gpu_ms > cpu_ms) ? gpu_ms : cpu_ms;
    }

    if (gpu_blocks > 0) {
        CHECK(clEnqueueReadBuffer(gpu_c->q, gpu_len, CL_TRUE, 0, gpu_blocks * sizeof(cl_uint), gpu_lens, 0, NULL, NULL));
    }
    if (cpu_blocks > 0) {
        CHECK(clEnqueueReadBuffer(cpu_c->q, cpu_len, CL_TRUE, 0, cpu_blocks * sizeof(cl_uint), cpu_lens, 0, NULL, NULL));
    }

    if (gpu_blocks && accel_ensure_host((void**)&ws->gpu.off, &ws->gpu.off_host_cap, gpu_blocks * sizeof(uint32_t)) != 0) goto cleanup;
    if (cpu_blocks && accel_ensure_host((void**)&ws->cpu.off, &ws->cpu.off_host_cap, cpu_blocks * sizeof(uint32_t)) != 0) goto cleanup;
    gpu_off = gpu_blocks ? ws->gpu.off : NULL;
    cpu_off = cpu_blocks ? ws->cpu.off : NULL;
    if ((gpu_blocks && !gpu_off) || (cpu_blocks && !cpu_off)) goto cleanup;
    for (size_t i = 0; i < gpu_blocks; ++i) {
        gpu_off[i] = (uint32_t)gpu_comp_total;
        gpu_comp_total += lens[i];
    }
    for (size_t i = 0; i < cpu_blocks; ++i) {
        cpu_off[i] = (uint32_t)cpu_comp_total;
        cpu_comp_total += lens[gpu_blocks + i];
    }
    comp_total = gpu_comp_total + cpu_comp_total;

    if (accel_ensure_host((void**)&ws->compact, &ws->compact_cap, comp_total ? comp_total : 1) != 0) goto cleanup;
    compact = ws->compact;
    if (gpu_blocks > 0) {
        if (accel_ensure_host((void**)&ws->gpu.sparse, &ws->gpu.sparse_cap, gpu_blocks * worst_blk) != 0) goto cleanup;
        CHECK(clEnqueueReadBuffer(gpu_c->q, gpu_out, CL_TRUE, 0, gpu_blocks * worst_blk, ws->gpu.sparse, 0, NULL, NULL));
        for (size_t i = 0; i < gpu_blocks; ++i) {
            memcpy(compact + gpu_off[i], ws->gpu.sparse + i * worst_blk, lens[i]);
        }
    }
    if (cpu_blocks > 0) {
        CHECK(clEnqueueReadBuffer(cpu_c->q, cpu_out, CL_TRUE, 0, cpu_blocks * worst_blk, cpu_sparse, 0, NULL, NULL));
    }
    compact_off = gpu_comp_total;
    for (size_t i = 0; i < cpu_blocks; ++i) {
        memcpy(compact + compact_off, cpu_sparse + i * worst_blk, lens[gpu_blocks + i]);
        compact_off += lens[gpu_blocks + i];
    }

    out->payload_size = comp_total;
    out->container_size = comp_total + 2 + 16 + nblk * sizeof(uint32_t);
    out->comp_gpu_global = gpu_global;
    out->comp_gpu_dict_slots = gpu_dict_slots;
    out->comp_cpu_global = cpu_global;
    out->payload_ratio_pct = 100.0 * (double)comp_total / (double)in_sz;
    out->container_ratio_pct = 100.0 * (double)out->container_size / (double)in_sz;
    if (gpu_out_sz && accel_ensure_host((void**)&ws->gpu.dec_out, &ws->gpu.dec_out_cap, gpu_out_sz) != 0) goto cleanup;
    if (cpu_out_sz && accel_ensure_host((void**)&ws->cpu.dec_out, &ws->cpu.dec_out_cap, cpu_out_sz) != 0) goto cleanup;
    gpu_dec_out = gpu_out_sz ? ws->gpu.dec_out : NULL;
    cpu_dec_out = cpu_out_sz ? ws->cpu.dec_out : NULL;
    if ((gpu_out_sz && !gpu_dec_out) || (cpu_out_sz && !cpu_dec_out)) goto cleanup;

    if (gpu_blocks > 0) {
        if (accel_ensure_cl_buffer(gpu_d->ctx, &ws->gpu.comp_d, &ws->gpu.comp_cap, gpu_comp_total ? gpu_comp_total : 1, CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(gpu_d->ctx, &ws->gpu.off_d, &ws->gpu.off_cap, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(gpu_d->ctx, &ws->gpu.len_d, &ws->gpu.len_d_cap, gpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(gpu_d->ctx, &ws->gpu.dec_d, &ws->gpu.dec_cap, gpu_out_sz ? gpu_out_sz : 1, CL_MEM_WRITE_ONLY) < 0) goto cleanup;
        gpu_comp_d = ws->gpu.comp_d; gpu_off_d = ws->gpu.off_d; gpu_len_d = ws->gpu.len_d; gpu_dec_d = ws->gpu.dec_d;
    }
    if (cpu_blocks > 0) {
        if (accel_ensure_cl_buffer(cpu_d->ctx, &ws->cpu.comp_d, &ws->cpu.comp_cap, cpu_comp_total ? cpu_comp_total : 1, CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(cpu_d->ctx, &ws->cpu.off_d, &ws->cpu.off_cap, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(cpu_d->ctx, &ws->cpu.len_d, &ws->cpu.len_d_cap, cpu_blocks * sizeof(cl_uint), CL_MEM_READ_ONLY) < 0) goto cleanup;
        if (accel_ensure_cl_buffer(cpu_d->ctx, &ws->cpu.dec_d, &ws->cpu.dec_cap, cpu_out_sz ? cpu_out_sz : 1, CL_MEM_WRITE_ONLY) < 0) goto cleanup;
        cpu_comp_d = ws->cpu.comp_d; cpu_off_d = ws->cpu.off_d; cpu_len_d = ws->cpu.len_d; cpu_dec_d = ws->cpu.dec_d;
    }

    if (gpu_blocks > 0) {
        CHECK(clEnqueueWriteBuffer(gpu_d->q, gpu_comp_d, CL_FALSE, 0, gpu_comp_total, compact, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(gpu_d->q, gpu_off_d, CL_FALSE, 0, gpu_blocks * sizeof(cl_uint), gpu_off, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(gpu_d->q, gpu_len_d, CL_FALSE, 0, gpu_blocks * sizeof(cl_uint), lens, 0, NULL, NULL));
    }
    if (cpu_blocks > 0) {
        CHECK(clEnqueueWriteBuffer(cpu_d->q, cpu_comp_d, CL_FALSE, 0, cpu_comp_total, compact + gpu_comp_total, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(cpu_d->q, cpu_off_d, CL_FALSE, 0, cpu_blocks * sizeof(cl_uint), cpu_off, 0, NULL, NULL));
        CHECK(clEnqueueWriteBuffer(cpu_d->q, cpu_len_d, CL_FALSE, 0, cpu_blocks * sizeof(cl_uint), lens + gpu_blocks, 0, NULL, NULL));
    }
    if (gpu_blocks > 0) clFinish(gpu_d->q);
    if (cpu_blocks > 0) clFinish(cpu_d->q);

    if (gpu_blocks > 0) {
        cl_uint gpu_dec_cus = 0;
        size_t full_dec_global = ((gpu_blocks + gpu_local - 1) / gpu_local) * gpu_local;
        (void)clGetDeviceInfo(gpu_d->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(gpu_dec_cus), &gpu_dec_cus, NULL);
        (void)gpu_dec_cus;
        dec_gpu_global = gpu_blocks;
        dec_gpu_global = ((dec_gpu_global + gpu_local - 1) / gpu_local) * gpu_local;
        if (cpu_blocks == 0 && gpu_d->full_kernel && dec_gpu_global >= full_dec_global) {
            lzo_hybrid_set_dec_full_args(gpu_d->full_kernel, gpu_comp_d, gpu_off_d, gpu_len_d, gpu_dec_d,
                                         (cl_uint)blk, (cl_uint)in_sz, (cl_uint)gpu_blocks);
        } else {
            lzo_hybrid_set_dec_args(gpu_d->kernel, gpu_comp_d, gpu_off_d, gpu_len_d, gpu_dec_d,
                                    (cl_uint)blk, (cl_uint)in_sz, 0U, (cl_uint)gpu_blocks);
        }
    }
    if (cpu_blocks > 0) {
        lzo_hybrid_set_dec_args(cpu_d->kernel, cpu_comp_d, cpu_off_d, cpu_len_d, cpu_dec_d,
                                (cl_uint)blk, (cl_uint)in_sz, (cl_uint)gpu_blocks, (cl_uint)cpu_blocks);
        dec_cpu_global = cpu_global;
    }
    if (gpu_blocks > 0) {
        cl_kernel dec_kernel = (cpu_blocks == 0 && gpu_d->full_kernel && dec_gpu_global >= ((gpu_blocks + gpu_local - 1) / gpu_local) * gpu_local) ? gpu_d->full_kernel : gpu_d->kernel;
        CHECK(clEnqueueNDRangeKernel(gpu_d->q, dec_kernel, 1, NULL, &dec_gpu_global, &gpu_local, 0, NULL, &ev_gpu_d));
    }
    if (cpu_blocks > 0) CHECK(clEnqueueNDRangeKernel(cpu_d->q, cpu_d->kernel, 1, NULL, &dec_cpu_global, &cpu_local, 0, NULL, &ev_cpu_d));
    if (ev_gpu_d) clWaitForEvents(1, &ev_gpu_d);
    if (ev_cpu_d) clWaitForEvents(1, &ev_cpu_d);
    {
        double gpu_ms = event_elapsed_us(ev_gpu_d) / 1000.0;
        double cpu_ms = event_elapsed_us(ev_cpu_d) / 1000.0;
        out->dec_kernel_event_ms = (gpu_ms > cpu_ms) ? gpu_ms : cpu_ms;
    }

    if (gpu_blocks > 0) CHECK(clEnqueueReadBuffer(gpu_d->q, gpu_dec_d, CL_TRUE, 0, gpu_out_sz, gpu_dec_out, 0, NULL, NULL));
    if (cpu_blocks > 0) CHECK(clEnqueueReadBuffer(cpu_d->q, cpu_dec_d, CL_TRUE, 0, cpu_out_sz, cpu_dec_out, 0, NULL, NULL));

    out->dec_gpu_global = dec_gpu_global;
    out->dec_cpu_global = dec_cpu_global;

    if (gpu_out_sz && memcmp(input, gpu_dec_out, gpu_out_sz) != 0) goto cleanup;
    if (cpu_out_sz && memcmp(input + gpu_out_sz, cpu_dec_out, cpu_out_sz) != 0) goto cleanup;
    out->ok = 1;
    rc = 0;

cleanup:
    if (ev_gpu_c) clReleaseEvent(ev_gpu_c);
    if (ev_cpu_c) clReleaseEvent(ev_cpu_c);
    if (ev_gpu_d) clReleaseEvent(ev_gpu_d);
    if (ev_cpu_d) clReleaseEvent(ev_cpu_d);
    return rc;
}

static int run_lzo_accel_bench(const char *in_path, const char *alg_name, int comp_level, double bench_seconds) {
    struct stat st;
    unsigned char* input = NULL;
    size_t in_sz = 0, blk = 0, nblk = 0;
    hybrid_ocl_t gpu_c, cpu_c, gpu_d, cpu_d;
    accel_run_ws_t accel_ws;
    size_t shares[32];
    size_t slots_list[32];
    size_t nshares = 0;
    size_t nslots = 0;
    int need_gpu = 0;
    int need_cpu = 0;
    size_t rounds = 1;
    uint64_t init_start, init_end;
    int rc = 1;

    memset(&gpu_c, 0, sizeof(gpu_c));
    memset(&cpu_c, 0, sizeof(cpu_c));
    memset(&gpu_d, 0, sizeof(gpu_d));
    memset(&cpu_d, 0, sizeof(cpu_d));
    memset(&accel_ws, 0, sizeof(accel_ws));

    if (!in_path || stat(in_path, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "bench error: invalid input file\n");
        return 1;
    }
    if (comp_level < 0) comp_level = LZO_DEFAULT_COMP_LEVEL;
    int gpu_level = (g_cli_gpu_level >= 0) ? g_cli_gpu_level : comp_level;
    int cpu_level = (g_cli_cpu_level >= 0) ? g_cli_cpu_level : comp_level;
    input = (unsigned char*)lzo_read_file(in_path, &in_sz);
    if (!input || in_sz == 0) {
        free(input);
        fprintf(stderr, "bench error: failed to read input\n");
        return 1;
    }
    nshares = parse_accel_cpu_shares(shares, 32);
    if (nshares == 0) {
        fprintf(stderr, "bench error: no accelerator CPU shares configured\n");
        goto cleanup;
    }
    nslots = parse_accel_cpu_slots_list(slots_list, 32);
    if (nslots == 0) {
        fprintf(stderr, "bench error: no accelerator CPU slots configured\n");
        goto cleanup;
    }
    for (size_t i = 0; i < nshares; ++i) {
        if (shares[i] < 100) need_gpu = 1;
        if (shares[i] > 0) need_cpu = 1;
    }

    init_start = now_ns();
    if ((need_gpu && accel_init_kernel_pair(&gpu_c, &gpu_d, CL_DEVICE_TYPE_GPU, "GPU", alg_name, gpu_level) != 0) ||
        (need_cpu && accel_init_kernel_pair(&cpu_c, &cpu_d, CL_DEVICE_TYPE_CPU, "CPU", alg_name, cpu_level) != 0)) {
        fprintf(stderr, "bench error: failed to initialize accelerator CPU/GPU kernels\n");
        goto cleanup;
    }
    init_end = now_ns();

    lzo_choose_blocking_adaptive(input, in_sz, need_gpu ? gpu_c.dev : cpu_c.dev, g_cli_fixed_block_bytes, g_cli_fixed_block_exact, &blk, &nblk, 0);
    if (blk == 0 || nblk == 0) goto cleanup;
    {
        long rounds_env = lzo_env_config()->bench_rounds;
        if (rounds_env <= 0) rounds_env = (long)((bench_seconds > 1.0) ? bench_seconds : 1.0);
        if (rounds_env < 1) rounds_env = 1;
        if (rounds_env > 1000) rounds_env = 1000;
        rounds = (size_t)rounds_env;
    }

    printf("\n==============================================================================\n");
    printf("  LZO ACCELERATOR BENCH (unified GPU/CPU split path)\n");
    printf("==============================================================================\n");
    printf("Input File             : %s\n", in_path);
    printf("Input Size             : %zu bytes (%.2f MB)\n", in_sz, (double)in_sz / (1024.0 * 1024.0));
    printf("Algorithm              : %s (D_BITS: %d, GPU D_BITS: %d, CPU D_BITS: %d)\n", alg_name, comp_level, gpu_level, cpu_level);
    printf("Workload               : %zu blocks (BlockSize: %.2f KB)\n", nblk, (double)blk / 1024.0);
    printf("Init/Build excluded    : %.3f ms\n", (double)(init_end - init_start) / 1000000.0);
    printf("Bench rounds           : %zu (init/build excluded)\n", rounds);
    printf("------------------------------------------------------------------------------\n");
    printf("round,cpu_slots,cpu%%,blocks_gpu,blocks_cpu,comp_gpu_global,comp_gpu_dict_slots,comp_cpu_global,dec_gpu_global,dec_cpu_global,ok,payload_bytes,container_bytes,payload_ratio%%,container_ratio%%,comp_kernel_event_ms,comp_kernel_event_MBps,dec_kernel_event_ms,dec_kernel_event_MBps\n");

    for (size_t li = 0; li < nslots; ++li) {
        g_accel_cpu_slots_override = slots_list[li];
        for (size_t ri = 0; ri < rounds; ++ri) {
            for (size_t si = 0; si < nshares; ++si) {
                accel_bench_sample_t s;
                size_t cpu_blocks = (nblk * shares[si] + 99) / 100;
                if (cpu_blocks > nblk) cpu_blocks = nblk;
                size_t gpu_blocks = nblk - cpu_blocks;
                if (run_accel_one_ratio(input, in_sz, blk, nblk, alg_name, gpu_level, cpu_level, shares[si], &gpu_c, &cpu_c, &gpu_d, &cpu_d, &accel_ws, &s) != 0) {
                    memset(&s, 0, sizeof(s));
                }
                double mb = (double)in_sz / (1024.0 * 1024.0);
                printf("%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%s,%zu,%zu,%.4f,%.4f,%.4f,%.3f,%.4f,%.3f\n",
                       ri + 1, slots_list[li], shares[si], gpu_blocks, cpu_blocks,
                       s.comp_gpu_global, s.comp_gpu_dict_slots, s.comp_cpu_global, s.dec_gpu_global, s.dec_cpu_global,
                       s.ok ? "yes" : "no",
                       s.payload_size, s.container_size,
                       s.payload_ratio_pct, s.container_ratio_pct,
                       s.comp_kernel_event_ms,
                       s.comp_kernel_event_ms > 0 ? mb / (s.comp_kernel_event_ms / 1000.0) : 0.0,
                       s.dec_kernel_event_ms,
                       s.dec_kernel_event_ms > 0 ? mb / (s.dec_kernel_event_ms / 1000.0) : 0.0);
            }
        }
    }
    printf("==============================================================================\n\n");
    rc = 0;

cleanup:
    g_accel_cpu_slots_override = 0;
    accel_release_run_ws(&accel_ws);
    lzo_hybrid_release(&gpu_c);
    lzo_hybrid_release(&cpu_c);
    lzo_hybrid_release(&gpu_d);
    lzo_hybrid_release(&cpu_d);
    free(input);
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
    lzo_print_bench_env(dev, standard_copy, alg_name, comp_level, debug_counters, bench_seconds);

    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
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

    char decomp_base[64];
    /* For decompression path, use decomp-only marker to skip unnecessary comp-kernel validation. */
    snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);
    int decomp_bits = comp_level;
    char decomp_prog_name[64];
    if (lzo_format_program_name(decomp_prog_name, sizeof(decomp_prog_name), decomp_base, debug_counters) != 0) {
        fprintf(stderr, "bench error: decompressor program name too long for %s\n", decomp_base);
        clReleaseKernel(krn_c);
        clReleaseProgram(prog_c);
        if (q) { clReleaseCommandQueue(q); q = NULL; }
        if (ctx) { clReleaseContext(ctx); ctx = NULL; }
        free(input_ref);
        return 1;
    }
    cl_program prog_d = lzo_load_program_with_dbits(ctx, dev, decomp_prog_name, decomp_bits, build_log, sizeof(build_log));
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

    snprintf(krn_name, sizeof(krn_name), "%s_block_decompress_range", alg_name);
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
    int kernel_dec_is_range = 0;
    if (clGetKernelInfo(krn_d, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL) == CL_SUCCESS) {
        kernel_dec_is_range = (krn_num_args == 9U || krn_num_args == 11U);
        kernel_has_dbg_dec = kernel_dec_is_range ? (krn_num_args >= 11U) : (krn_num_args >= 10U);
    }

    lzo_hybrid_workspace_t ws;
    lzo_hybrid_workspace_init(&ws);

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
    unsigned long long dbg_dec_literal_ops_total = 0;
    unsigned long long dbg_dec_match_ops_total = 0;
    unsigned long long dbg_dec_overlap_matches_total = 0;
    unsigned long long dbg_dec_m2_matches_total = 0;
    unsigned long long dbg_dec_m3_matches_total = 0;
    unsigned long long dbg_dec_m4_matches_total = 0;
    unsigned long long dbg_dec_first_literal_run_bytes_total = 0;
    unsigned long long dbg_dec_first_literal_run_ops_total = 0;
    unsigned long long dbg_dec_post_match_literal_bytes_total = 0;
    unsigned long long dbg_dec_post_match_literal_ops_total = 0;
    unsigned long long dbg_dec_eof_markers_total = 0;

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
        int rc = lzo_compress_core(ctx, q, dev, krn_c, in_path, NULL,
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
            if (kernel_dec_is_range) {
                cl_uint block_start = 0;
                CHECK(clSetKernelArg(krn_d, 7, sizeof(cl_uint), &block_start));
                CHECK(clSetKernelArg(krn_d, 8, sizeof(cl_uint), &nblk_cl));
            } else {
                CHECK(clSetKernelArg(krn_d, 7, sizeof(cl_uint), &nblk_cl));
            }

            int dbg_dec_enabled = (debug_counters && kernel_has_dbg_dec);
            if (dbg_dec_enabled) {
                size_t dbg_bytes = nblk * BENCH_LZO_DBG_DEC_N * sizeof(cl_uint);
                d_dbg_dec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dbg_bytes, NULL, &err);
                if (err != CL_SUCCESS || !d_dbg_dec) {
                    dbg_dec_enabled = 0;
                } else {
                    h_dbg_dec = (cl_uint*)calloc(nblk * BENCH_LZO_DBG_DEC_N, sizeof(cl_uint));
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
                cl_uint dbg_arg_index = kernel_dec_is_range ? 9U : 8U;
                CHECK(clSetKernelArg(krn_d, dbg_arg_index, sizeof(cl_mem), &dbg_arg));
                CHECK(clSetKernelArg(krn_d, dbg_arg_index + 1U, sizeof(cl_uint), &dbg_flag));
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
            size_t dbg_bytes = nblk * BENCH_LZO_DBG_DEC_N * sizeof(cl_uint);
            err = clEnqueueReadBuffer(q, d_dbg_dec, CL_TRUE, 0, dbg_bytes, h_dbg_dec, 0, NULL, NULL);
            if (err == CL_SUCCESS) {
                for (size_t i = 0; i < nblk; ++i) {
                    size_t base = i * BENCH_LZO_DBG_DEC_N;
                    dbg_dec_tokens_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_TOKENS];
                    dbg_dec_literals_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_LITERAL_BYTES];
                    dbg_dec_matches_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_MATCH_BYTES];
                    dbg_dec_small_offsets_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_SMALL_OFFSETS];
                    dbg_dec_output_errors_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_OUTPUT_ERROR];
                    dbg_dec_literal_ops_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_LITERAL_OPS];
                    dbg_dec_match_ops_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_MATCH_OPS];
                    dbg_dec_overlap_matches_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_OVERLAP_MATCHES];
                    dbg_dec_m2_matches_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_M2_MATCHES];
                    dbg_dec_m3_matches_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_M3_MATCHES];
                    dbg_dec_m4_matches_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_M4_MATCHES];
                    dbg_dec_first_literal_run_bytes_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_FIRST_LITERAL_RUN_BYTES];
                    dbg_dec_first_literal_run_ops_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_FIRST_LITERAL_RUN_OPS];
                    dbg_dec_post_match_literal_bytes_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_POST_MATCH_LITERAL_BYTES];
                    dbg_dec_post_match_literal_ops_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_POST_MATCH_LITERAL_OPS];
                    dbg_dec_eof_markers_total += h_dbg_dec[base + BENCH_LZO_DBG_DEC_EOF_MARKERS];
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
            printf("[LZO-DBG][DECOMP] tokens=%llu literal_bytes=%llu match_bytes=%llu small_offsets=%llu output_errors=%llu literal_ops=%llu match_ops=%llu overlap_matches=%llu\n",
                dbg_dec_tokens_total,
                dbg_dec_literals_total,
                dbg_dec_matches_total,
                dbg_dec_small_offsets_total,
                dbg_dec_output_errors_total,
                dbg_dec_literal_ops_total,
                dbg_dec_match_ops_total,
                dbg_dec_overlap_matches_total);
            printf("[LZO-DBG][DECOMP][detail] m2_matches=%llu m3_matches=%llu m4_matches=%llu first_literal_run_bytes=%llu first_literal_run_ops=%llu post_match_literal_bytes=%llu post_match_literal_ops=%llu eof_markers=%llu\n",
                dbg_dec_m2_matches_total,
                dbg_dec_m3_matches_total,
                dbg_dec_m4_matches_total,
                dbg_dec_first_literal_run_bytes_total,
                dbg_dec_first_literal_run_ops_total,
                dbg_dec_post_match_literal_bytes_total,
                dbg_dec_post_match_literal_ops_total,
                dbg_dec_eof_markers_total);
        }
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_tp);
    free(dec_tp);
    free(ratio_pct);
    free(input_ref);
    lzo_hybrid_workspace_free(&ws);
    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (krn_c) clReleaseKernel(krn_c);
    if (prog_c) clReleaseProgram(prog_c);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return verify_ok ? 0 : 1;
}


/* Implementations will call into lzo_hybrid_core.c which provides lzo_compress_core/lzo_decompress_core */
#include "lzo_hybrid_core.h"


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
        if (strcmp(arg, "--debug-kernel") == 0) {
            g_cli_debug_kernel = 1;
            continue;
        }
        if (strcmp(arg, "--cpu-threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --cpu-threads requires an argument\n");
                return 1;
            }
            lzo_cli_set_cpu_threads(strtol(argv[++i], NULL, 10));
            continue;
        }
        if (strncmp(arg, "--cpu-threads=", 14) == 0) {
            lzo_cli_set_cpu_threads(strtol(arg + 14, NULL, 10));
            continue;
        }
        if (strcmp(arg, "--gpu-ratio") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --gpu-ratio requires an argument\n");
                return 1;
            }
            lzo_cli_set_gpu_ratio(strtod(argv[++i], NULL));
            continue;
        }
        if (strncmp(arg, "--gpu-ratio=", 12) == 0) {
            lzo_cli_set_gpu_ratio(strtod(arg + 12, NULL));
            continue;
        }
        if (strcmp(arg, "--adaptive") == 0) {
            lzo_cli_enable_adaptive();
            continue;
        }
        if (strcmp(arg, "--gpu-level") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --gpu-level requires an argument\n");
                return 1;
            }
            g_cli_gpu_level = lzo_cli_parse_level_value(argv[++i], "--gpu-level");
            if (g_cli_gpu_level < 0) return 1;
            continue;
        }
        if (strncmp(arg, "--gpu-level=", 12) == 0) {
            g_cli_gpu_level = lzo_cli_parse_level_value(arg + 12, "--gpu-level");
            if (g_cli_gpu_level < 0) return 1;
            continue;
        }
        if (strcmp(arg, "--cpu-level") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --cpu-level requires an argument\n");
                return 1;
            }
            g_cli_cpu_level = lzo_cli_parse_level_value(argv[++i], "--cpu-level");
            if (g_cli_cpu_level < 0) return 1;
            continue;
        }
        if (strncmp(arg, "--cpu-level=", 12) == 0) {
            g_cli_cpu_level = lzo_cli_parse_level_value(arg + 12, "--cpu-level");
            if (g_cli_cpu_level < 0) return 1;
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
            if (comp_level < 11 || comp_level > 20) {
                fprintf(stderr, "error: dictionary size must be between 11 and 20 bits (got %d)\n", comp_level);
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
        if (lzo_accel_bench_requested()) {
            return run_lzo_accel_bench(in_path, alg_name, comp_level, bench_seconds);
        }
        return run_lzo_bench(in_path, alg_name, comp_level, bench_seconds);
    }

    effective_in_path = in_path;
    effective_lz_path = lz_path;
    effective_output_path = output_path;

    if (input_from_stdin) {
        if (create_temp_path(temp_input_path, sizeof(temp_input_path), "/tmp/lzo_hybrid_stdin_XXXXXX") != 0) {
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
        if (create_temp_path(temp_output_path, sizeof(temp_output_path), "/tmp/lzo_hybrid_stdout_XXXXXX") != 0) {
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
