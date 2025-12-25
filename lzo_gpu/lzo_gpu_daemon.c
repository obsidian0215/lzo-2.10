#define _POSIX_C_SOURCE 200809L
#include <time.h>
/*
 * lzo_gpu_daemon.c - LZO GPU守护进程实现
 *
 * 功能: 保持OpenCL上下文和缓冲区常驻内存,通过Unix socket接收压缩请求
 * 性能: 节省549ms/次的初始化开销 (OCL初始化44ms + 缓冲区分配505ms)
 *
 * 使用:
 *   启动守护进程: ./lzo_gpu_daemon
 *   客户端请求:   ./lzo_gpu --daemon <file>
 *   停止守护进程: ./lzo_gpu --daemon-stop
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <errno.h>
#include <sys/file.h> /* for flock */
#include <CL/cl.h>
#include "timing.h"
#include <dirent.h>
#include "lzo_defaults.h"
#include "lzo_gpu_utils.h"
#include "lzo_gpu_core.h"
#include <stddef.h>
#include <libgen.h>

/* Decompress is implemented in lzo_gpu_core: use lzo_decompress_core() */

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"
#define PID_FILE "/tmp/lzo_gpu_daemon.pid"
#define MAX_CLIENTS 5
#define MAX_BUFFER_SIZE (128 * 1024 * 1024)  // 128MB - 足够处理大部分文件

/* 守护进程全局状态 */
typedef struct {
    /* OpenCL资源 - 常驻内存 */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    /* 多kernel支持 - 算法(2) x 字典大小(5) */
    /* Alg: 0=1x, 1=1y */
    /* Bits: 0=10, 1=11, 2=12, 3=13, 4=14 */
    cl_program programs[2][5];
    cl_kernel kernels_comp[2][5];
    /* Optimized instrumented variants cache (unroll/vector tests) */
    cl_program programs_opt[2][5];
    cl_kernel kernels_comp_opt[2][5];
    /* Debug-instrumented variants cache (per-algorithm x bits) */
    cl_program programs_debug[2][5];
    cl_kernel kernels_comp_debug[2][5];

    /* 解压缩kernels (每个算法一个) */
    cl_program prog_decomp[2];
    cl_kernel kernel_decomp[2];

    /* 预分配缓冲区 */
    cl_mem d_input;
    cl_mem d_output;
    cl_mem d_lengths;
    size_t buffer_size;

    /* 统计信息 */
    unsigned long requests;
    unsigned long total_time_ms;
    unsigned long init_time_ms;  // 实际测量的初始化时间

    /* 服务器socket */
    int server_sock;
    volatile int running;
    int pid_fd; /* file descriptor for pidfile lock (if any) */
    FILE* logf; /* optional logfile (duplicate of stdout/stderr) */
} daemon_state_t;

static daemon_state_t g_state = {0};

/* Find the named kernel/source file under the common candidate locations.
 * If found, place the resolved absolute (or relative) path into `out` and return 0.
 * If not found, return -1.
 */
/* file location helpers replaced by shared implementation in lzo_gpu_utils.c */

/* Open a kernel file (binary or source) by searching the common locations. */
/* Replaced open_kernel_file with direct calls to lzo_find_file_path + fopen in callers */

/* Helper to return computed pidfile path. By default this is PID_FILE (/tmp),
 * but we allow override via OUT_DIR or LZO_DAEMON_PID env vars to store runtime
 * artifacts within the repository (e.g. $OUT_DIR).
 */
static const char *daemon_pidfile_path(void)
{
    static char buf[PATH_MAX];
    const char *env = getenv("LZO_DAEMON_PID");
    if (env && env[0]) return env;
    env = getenv("OUT_DIR");
    if (env && env[0]) {
        snprintf(buf, sizeof(buf), "%s/lzo_gpu_daemon.pid", env);
        return buf;
    }
    return PID_FILE;
}

/* Helper to return computed socket path. By default it's SOCKET_PATH (/tmp),
 * but allow override via OUT_DIR or LZO_DAEMON_SOCKET env var so runs can keep
 * socket and related artifacts under the repository's exp_results directory.
 */
static const char *daemon_socket_path(void)
{
    static char buf[PATH_MAX];
    const char *env = getenv("LZO_DAEMON_SOCKET");
    if (env && env[0]) return env;
    env = getenv("OUT_DIR");
    if (env && env[0]) {
        snprintf(buf, sizeof(buf), "%s/lzo_gpu_daemon.sock", env);
        return buf;
    }
    return SOCKET_PATH;
}

/* 请求协议 */
typedef struct {
    char operation;      // 'C'=compress, 'D'=decompress
    char input_path[256];
    char output_path[256];
    int level;           // 压缩级别 (bits 10-14)
    int alg;             // 算法: 0=1x, 1=1y
    size_t input_size;
    /* options from client/env */
    int standard_copy; /* 0/1 */
    int mt_io;        /* 0/1 */
    int mt_threads;   /* number of IO threads */
    int fixed_block_kb; /* 0=no fixed size, else KB */
    int local_size;    /* 0=unspecified; else local workgroup size */
    /* Per-request overrides for coalescing and stdio buffer - -1 = unspecified */
    int coalesce_output;  /* -1 unspecified; 0=off; 1=on */
    int coalesce_chunk_mb;/* -1 unspecified; positive = chunk size in MB */
    int coalesce_max_mb;  /* -1 unspecified; positive = max MB threshold for single coalesce */
    int stdio_buf_mb;     /* -1 unspecified; positive = MB for stdio buffer */
    /* Opt kernel flag (from client) */
    int kernel_opt;       /* 0/1: request optimized instrumented kernel */
    /* Per-request debug flag (0/1): request kernel instrumentation and verbose diagnostics */
    int debug;
} request_t;

typedef struct {
    int status;          // 0=success, -1=error
    size_t output_size;
    unsigned long time_us;  // 总时间 (微秒)

    /* compact timing structure */
    timing_t timing;

    char message[128];
} response_t;

/*
 * 初始化OpenCL资源 (仅在守护进程启动时执行一次)
 */
int init_opencl_resources(void)
{
    cl_int err;
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    printf("[DAEMON] 初始化OpenCL资源...\n");

    // 1. 获取平台和设备
    err = clGetPlatformIDs(1, &g_state.platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "获取平台失败: %d\n", err);
        return -1;
    }

    err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_GPU, 1,
                         &g_state.device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "获取GPU设备失败: %d\n", err);
        return -1;
    }

    // 2. 创建上下文 (常驻)
    g_state.context = clCreateContext(NULL, 1, &g_state.device,
                                      NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建上下文失败: %d\n", err);
        return -1;
    }

    // 3. 创建命令队列，启用 CL_QUEUE_PROFILING_ENABLE，方便后续用事件查询精确的 kernel 执行时间
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g_state.queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device,
                                                        props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建命令队列失败: %d\n", err);
        return -1;
    }

    // 4. 初始化Kernel数组 (按需编译)
    memset(g_state.programs, 0, sizeof(g_state.programs));
    memset(g_state.kernels_comp, 0, sizeof(g_state.kernels_comp));
    memset(g_state.programs_opt, 0, sizeof(g_state.programs_opt));
    memset(g_state.kernels_comp_opt, 0, sizeof(g_state.kernels_comp_opt));
    memset(g_state.programs_debug, 0, sizeof(g_state.programs_debug));
    memset(g_state.kernels_comp_debug, 0, sizeof(g_state.kernels_comp_debug));
    memset(g_state.prog_decomp, 0, sizeof(g_state.prog_decomp));
    memset(g_state.kernel_decomp, 0, sizeof(g_state.kernel_decomp));

    printf("[DAEMON] Kernels将按需编译 (Lazy Loading)...\n");

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_state.init_time_ms = (t_end.tv_sec - t_start.tv_sec) * 1000 +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1000000;

    printf("[DAEMON] ✅ OpenCL资源初始化完成\n");
    printf("[DAEMON]    - 上下文: 常驻内存\n");
    printf("[DAEMON]    - 压缩kernels: 1x/1y (10-14 bits)\n");
    printf("[DAEMON]    - 解压缩kernel: 1x/1y\n");
    printf("[DAEMON]    - 缓冲区: 动态分配 (每次请求)\n");
    printf("[DAEMON]    - 初始化耗时: %lu ms\n", g_state.init_time_ms);

    return 0;
}

/* 获取或编译压缩Kernel */
static cl_kernel get_compress_kernel(int alg, int bits, int kernel_opt, int kernel_debug)
{
    if (alg < 0 || alg > 1 || bits < 10 || bits > 14) return NULL;
    int bit_idx = bits - 10;

    const char* alg_names[] = {"lzo1x", "lzo1y"};
    cl_int err;

    /* Helper macro-ish: build opts */
    char build_opts[128];
    snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -D D_BITS=%d", bits);

    /* If kernel_debug requested, attempt to load/compile debug wrapper variant (<alg>_debug) */
    if (kernel_debug) {
        if (g_state.kernels_comp_debug[alg][bit_idx]) return g_state.kernels_comp_debug[alg][bit_idx];
        /* Try bits-specific debug binary first: <alg>_debug_<bits>.clbin */
        {
            char bin_name[80]; size_t bin_sz; unsigned char* bin = NULL;
            snprintf(bin_name, sizeof(bin_name), "%s_debug_%d.clbin", alg_names[alg], bits);
            bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);
            if (bin) {
                printf("[DAEMON] 加载预编译Kernel: %s\n", bin_name);
                cl_int binary_status;
                cl_program p = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                        &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
                free(bin);
                if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                    err = clBuildProgram(p, 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
                    if (err == CL_SUCCESS) {
                        char kernel_name[64]; snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress_debug", alg_names[alg]);
                        g_state.programs_debug[alg][bit_idx] = p;
                        g_state.kernels_comp_debug[alg][bit_idx] = clCreateKernel(p, kernel_name, &err);
                        if (err == CL_SUCCESS) return g_state.kernels_comp_debug[alg][bit_idx];
                        if (g_state.programs_debug[alg][bit_idx]) { clReleaseProgram(g_state.programs_debug[alg][bit_idx]); g_state.programs_debug[alg][bit_idx] = NULL; }
                    } else {
                        clReleaseProgram(p);
                    }
                }
            }
        }
        /* Try generic debug binary <alg>_debug.clbin */
        {
            char bin_name[80]; size_t bin_sz; unsigned char* bin = NULL;
            snprintf(bin_name, sizeof(bin_name), "%s_debug.clbin", alg_names[alg]);
            bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);
            if (bin) {
                printf("[DAEMON] 加载预编译Kernel: %s\n", bin_name);
                cl_int binary_status;
                cl_program p = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                        &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
                free(bin);
                if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                    err = clBuildProgram(p, 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
                    if (err == CL_SUCCESS) {
                        char kernel_name[64]; snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress_debug", alg_names[alg]);
                        g_state.programs_debug[alg][bit_idx] = p;
                        g_state.kernels_comp_debug[alg][bit_idx] = clCreateKernel(p, kernel_name, &err);
                        if (err == CL_SUCCESS) return g_state.kernels_comp_debug[alg][bit_idx];
                        if (g_state.programs_debug[alg][bit_idx]) { clReleaseProgram(g_state.programs_debug[alg][bit_idx]); g_state.programs_debug[alg][bit_idx] = NULL; }
                    } else {
                        clReleaseProgram(p);
                    }
                }
            }
        }
        /* Fallback: compile base source with debug instrumentation enabled */
        {
            char src_file[64]; snprintf(src_file, sizeof(src_file), "%s.cl", alg_names[alg]);
            printf("[DAEMON] 尝试编译 debug 变体: %s (D_BITS=%d + LZO_GPU_DEBUG)...\n", src_file, bits);
            size_t src_len; char* src = lzo_read_file(src_file, &src_len);
            if (src) {
                g_state.programs_debug[alg][bit_idx] = clCreateProgramWithSource(g_state.context, 1, (const char**)&src, &src_len, &err);
                free(src);
                if (err == CL_SUCCESS) {
                    char build_opts_with_inc[256]; snprintf(build_opts_with_inc, sizeof(build_opts_with_inc), "%s -D LZO_GPU_DEBUG", build_opts);
                    err = clBuildProgram(g_state.programs_debug[alg][bit_idx], 1, &g_state.device, build_opts_with_inc, NULL, NULL);
                    if (err == CL_SUCCESS) {
                        char kernel_name[64]; snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress_debug", alg_names[alg]);
                        g_state.kernels_comp_debug[alg][bit_idx] = clCreateKernel(g_state.programs_debug[alg][bit_idx], kernel_name, &err);
                        if (err == CL_SUCCESS) return g_state.kernels_comp_debug[alg][bit_idx];
                        if (g_state.programs_debug[alg][bit_idx]) { clReleaseProgram(g_state.programs_debug[alg][bit_idx]); g_state.programs_debug[alg][bit_idx] = NULL; }
                    } else {
                        size_t log_sz; clGetProgramBuildInfo(g_state.programs_debug[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                        if (log_sz > 0) { char* log = malloc(log_sz + 1); clGetProgramBuildInfo(g_state.programs_debug[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz] = '\0'; fprintf(stderr, "%s\n", log); free(log); }
                        clReleaseProgram(g_state.programs_debug[alg][bit_idx]); g_state.programs_debug[alg][bit_idx] = NULL;
                    }
                }
            }
        }
        /* If we couldn't load/compile debug variant, fall through to default behavior */
    }

    /* If optimized variant requested */
    if (kernel_opt) {
        if (g_state.kernels_comp_opt[alg][bit_idx]) return g_state.kernels_comp_opt[alg][bit_idx];

        char bin_name[80];
        size_t bin_sz;
        unsigned char* bin = NULL;

        /* Prefer debug+opt only when caller explicitly requests kernel_debug */
        if (kernel_debug) {
            snprintf(bin_name, sizeof(bin_name), "%s_debug_opt_%d.clbin", alg_names[alg], bits);
            bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);
            if (bin) {
                printf("[DAEMON] 加载预编译Kernel: %s\n", bin_name);
                cl_int binary_status;
                g_state.programs_opt[alg][bit_idx] = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                            &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
                free(bin);
                if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                    goto kernel_opt_create_debug;
                }
                if (g_state.programs_opt[alg][bit_idx]) { clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL; }
            }
        }

        /* Try non-debug opt precompiled binary */
        snprintf(bin_name, sizeof(bin_name), "%s_opt_%d.clbin", alg_names[alg], bits);
        bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);
        if (bin) {
            printf("[DAEMON] 加载预编译Kernel: %s\n", bin_name);
            cl_int binary_status;
            g_state.programs_opt[alg][bit_idx] = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                        &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
            free(bin);
            if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                goto kernel_opt_create;
            }
            if (g_state.programs_opt[alg][bit_idx]) { clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL; }
        }

        /* If kernel_debug requested, attempt to compile debug+opt from base source */
        if (kernel_debug) {
            char src_file[64]; snprintf(src_file, sizeof(src_file), "%s.cl", alg_names[alg]);
            printf("[DAEMON] 尝试编译 debug+opt 变体: %s (D_BITS=%d + LZO_USE_UNROLL2 + LZO_GPU_DEBUG)...\n", src_file, bits);
            size_t src_len;
            char* src = lzo_read_file(src_file, &src_len);
            if (src) {
                /* Compute include dir to help OpenCL compilers find includes */
                char resolved_src[PATH_MAX] = {0};
                char include_opt[256] = "";
                if (lzo_find_file_path(src_file, resolved_src, sizeof(resolved_src)) == 0) {
                    char *slash = strrchr(resolved_src, '/');
                    if (slash) { *slash = '\0'; snprintf(include_opt, sizeof(include_opt), " -I%s", resolved_src); }
                }
                g_state.programs_opt[alg][bit_idx] = clCreateProgramWithSource(g_state.context, 1, (const char**)&src, &src_len, &err);
                free(src);
                if (err == CL_SUCCESS) {
                    char build_opts_with_inc[256];
                    snprintf(build_opts_with_inc, sizeof(build_opts_with_inc), "%s%s -D LZO_USE_UNROLL2 -D LZO_GPU_DEBUG", build_opts, include_opt);
                    err = clBuildProgram(g_state.programs_opt[alg][bit_idx], 1, &g_state.device, build_opts_with_inc, NULL, NULL);
                    if (err == CL_SUCCESS) goto kernel_opt_create_debug;
                    /* capture log for debug */
                    size_t log_sz; clGetProgramBuildInfo(g_state.programs_opt[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                    if (log_sz > 0) { char* log = malloc(log_sz + 1); clGetProgramBuildInfo(g_state.programs_opt[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz] = '\0'; fprintf(stderr, "%s\n", log); free(log); }
                    clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL;
                }
            }
        }

        /* Fallback: compile base source with LZO_USE_UNROLL2 (non-debug) */
        {
            char src_file[64]; snprintf(src_file, sizeof(src_file), "%s.cl", alg_names[alg]);
            printf("[DAEMON] 尝试编译 opt 变体: %s (D_BITS=%d + LZO_USE_UNROLL2)...\n", src_file, bits);
            size_t src_len;
            char* src = lzo_read_file(src_file, &src_len);
            if (src) {
                char resolved_src[PATH_MAX] = {0};
                char include_opt[256] = "";
                if (lzo_find_file_path(src_file, resolved_src, sizeof(resolved_src)) == 0) {
                    char *slash = strrchr(resolved_src, '/');
                    if (slash) { *slash = '\0'; snprintf(include_opt, sizeof(include_opt), " -I%s", resolved_src); }
                }
                g_state.programs_opt[alg][bit_idx] = clCreateProgramWithSource(g_state.context, 1, (const char**)&src, &src_len, &err);
                free(src);
                if (err == CL_SUCCESS) {
                    char build_opts_with_inc[256];
                    snprintf(build_opts_with_inc, sizeof(build_opts_with_inc), "%s%s -D LZO_USE_UNROLL2", build_opts, include_opt);
                    err = clBuildProgram(g_state.programs_opt[alg][bit_idx], 1, &g_state.device, build_opts_with_inc, NULL, NULL);
                    if (err == CL_SUCCESS) goto kernel_opt_create;
                    size_t log_sz; clGetProgramBuildInfo(g_state.programs_opt[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                    if (log_sz > 0) { char* log = malloc(log_sz + 1); clGetProgramBuildInfo(g_state.programs_opt[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz] = '\0'; fprintf(stderr, "%s\n", log); free(log); }
                    clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL;
                }
            }
        }
        return NULL;

kernel_opt_create:
        {
            char kernel_name[64];
            snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress", alg_names[alg]);
            g_state.kernels_comp_opt[alg][bit_idx] = clCreateKernel(g_state.programs_opt[alg][bit_idx], kernel_name, &err);
            if (err == CL_SUCCESS) return g_state.kernels_comp_opt[alg][bit_idx];
            if (g_state.programs_opt[alg][bit_idx]) { clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL; }
            return NULL;
        }

kernel_opt_create_debug:
        {
            char kernel_name[64];
            /* debug kernel symbol name is same for both debug & debug+opt variants */
            snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress_debug", alg_names[alg]);
            g_state.kernels_comp_opt[alg][bit_idx] = clCreateKernel(g_state.programs_opt[alg][bit_idx], kernel_name, &err);
            if (err == CL_SUCCESS) return g_state.kernels_comp_opt[alg][bit_idx];
            if (g_state.programs_opt[alg][bit_idx]) { clReleaseProgram(g_state.programs_opt[alg][bit_idx]); g_state.programs_opt[alg][bit_idx] = NULL; }
            return NULL;
        }
    }



try_default:
    if (g_state.kernels_comp[alg][bit_idx]) return g_state.kernels_comp[alg][bit_idx];

    /* Try to load precompiled binary first */
    char bin_name[64];
    snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", alg_names[alg], bits);
    size_t bin_sz;
    unsigned char* bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);

    if (bin) {
        printf("[DAEMON] 加载预编译Kernel: %s\n", bin_name);
        cl_int binary_status;
        g_state.programs[alg][bit_idx] = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                        &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
             err = clBuildProgram(g_state.programs[alg][bit_idx], 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
             if (err == CL_SUCCESS) {
                 goto kernel_create;
             }
        }
        if (g_state.programs[alg][bit_idx]) {
            clReleaseProgram(g_state.programs[alg][bit_idx]);
            g_state.programs[alg][bit_idx] = NULL;
        }
    }

    /* Fallback to source compilation */
    char src_file[32];
    snprintf(src_file, sizeof(src_file), "%s.cl", alg_names[alg]);

    printf("[DAEMON] 编译压缩Kernel: %s (D_BITS=%d)...\n", alg_names[alg], bits);

    size_t src_len;
    char* src = lzo_read_file(src_file, &src_len);
    if (!src) {
        fprintf(stderr, "[DAEMON] 无法读取源文件: %s\n", src_file);
        return NULL;
    }

    g_state.programs[alg][bit_idx] = clCreateProgramWithSource(g_state.context, 1,
                                                    (const char**)&src, &src_len, &err);
    free(src);

    if (err != CL_SUCCESS) return NULL;

    /* Compute include dir to help OpenCL compilers find includes */
    char resolved_src2[PATH_MAX] = {0};
    char include_opt2[256] = "";
    if (lzo_find_file_path(src_file, resolved_src2, sizeof(resolved_src2)) == 0) {
        char *slash = strrchr(resolved_src2, '/');
        if (slash) { *slash = '\0'; snprintf(include_opt2, sizeof(include_opt2), " -I%s", resolved_src2); }
    }
    char build_opts_with_inc2[128];
    snprintf(build_opts_with_inc2, sizeof(build_opts_with_inc2), "-cl-std=CL2.0 -I.%s -D D_BITS=%d", include_opt2, bits);

    err = clBuildProgram(g_state.programs[alg][bit_idx], 1, &g_state.device, build_opts_with_inc2, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(g_state.programs[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        if (log_sz > 0) {
            char* log = malloc(log_sz + 1);
            clGetProgramBuildInfo(g_state.programs[alg][bit_idx], g_state.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            log[log_sz] = '\0';
            fprintf(stderr, "%s\n", log);
            free(log);
        }
        return NULL;
    }

kernel_create:
    char kernel_name[32];
    snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress", alg_names[alg]);
    g_state.kernels_comp[alg][bit_idx] = clCreateKernel(g_state.programs[alg][bit_idx], kernel_name, &err);

    if (err != CL_SUCCESS) return NULL;

    return g_state.kernels_comp[alg][bit_idx];
}

/* 获取或编译解压缩Kernel */
static cl_kernel get_decompress_kernel(int alg)
{
    if (alg < 0 || alg > 1) return NULL;

    if (g_state.kernel_decomp[alg]) return g_state.kernel_decomp[alg];

    const char* alg_names[] = {"lzo1x", "lzo1y"};

    /* Try to load precompiled binary first */
    char bin_name[64];
    snprintf(bin_name, sizeof(bin_name), "%s_decomp.clbin", alg_names[alg]);
    size_t bin_sz;
    unsigned char* bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);

    cl_int err;
    if (bin) {
        printf("[DAEMON] 加载预编译解压Kernel: %s\n", bin_name);
        cl_int binary_status;
        g_state.prog_decomp[alg] = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                                        &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
             err = clBuildProgram(g_state.prog_decomp[alg], 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
             if (err == CL_SUCCESS) {
                 goto kernel_create;
             }
        }
        if (g_state.prog_decomp[alg]) {
            clReleaseProgram(g_state.prog_decomp[alg]);
            g_state.prog_decomp[alg] = NULL;
        }
    }

    /* Fallback to source compilation */
    char src_file[32];
    snprintf(src_file, sizeof(src_file), "%s_decomp.cl", alg_names[alg]);

    printf("[DAEMON] 编译解压缩Kernel: %s...\n", alg_names[alg]);

    size_t src_len;
    char* src = lzo_read_file(src_file, &src_len);
    if (!src) return NULL;

    g_state.prog_decomp[alg] = clCreateProgramWithSource(g_state.context, 1,
                                                    (const char**)&src, &src_len, &err);
    free(src);

    if (err != CL_SUCCESS) return NULL;

    err = clBuildProgram(g_state.prog_decomp[alg], 1, &g_state.device, "-cl-std=CL2.0 -I.", NULL, NULL);
    if (err != CL_SUCCESS) return NULL;

kernel_create:
    char kernel_name[32];
    snprintf(kernel_name, sizeof(kernel_name), "%s_block_decompress", alg_names[alg]);
    g_state.kernel_decomp[alg] = clCreateKernel(g_state.prog_decomp[alg], kernel_name, &err);

    return g_state.kernel_decomp[alg];
}

/*
 * 处理压缩请求 (复用已初始化的资源)
 */
int handle_compress_request(request_t* req, response_t* resp)
{
    printf("[DAEMON] 处理压缩请求: %s -> %s (level=%d)\n",
           req->input_path, req->output_path, req->level);
    /* Diagnostic: print key request parameters to ensure they were received */
        fprintf(stderr, "[DAEMON] Request opts: standard_copy=%d mt_io=%d mt_threads=%d fixed_block_kb=%d local_size=%d coalesce_output=%d coalesce_chunk_mb=%d coalesce_max_mb=%d stdio_buf_mb=%d kernel_opt=%d\n",
            req->standard_copy, req->mt_io, req->mt_threads, req->fixed_block_kb, req->local_size,
            req->coalesce_output, req->coalesce_chunk_mb, req->coalesce_max_mb, req->stdio_buf_mb, req->kernel_opt);

    // 根据alg和level选择合适的kernel
    cl_kernel kernel = get_compress_kernel(req->alg, req->level, req->kernel_opt, req->debug);
    const char* alg_names[] = {"lzo1x", "lzo1y"};

    if (!kernel) {
        resp->status = -1;
        snprintf(resp->message, sizeof(resp->message), "Failed to get kernel for alg=%d level=%d", req->alg, req->level);
        return -1;
    }

    printf("[DAEMON]    - 使用kernel: %s (bits=%d)\n", alg_names[req->alg], req->level);

    unsigned long time_us = 0;
    size_t output_size = 0;
    timing_t t = {0};

    /* Create parameter object from request */
    lzo_compress_params_t params = {
        .level = req->level,
        .alg_id = req->alg,
        .standard_copy = req->standard_copy,
        .mt_io = req->mt_io,
        .mt_threads = req->mt_threads,
        .fixed_block_kb = req->fixed_block_kb,
        .coalesce_output = req->coalesce_output,
        .coalesce_chunk_mb = req->coalesce_chunk_mb,
        .coalesce_max_mb = req->coalesce_max_mb,
        .stdio_buf_mb = req->stdio_buf_mb,
        .local_size_param = req->local_size,
        .debug = req->debug
    };

    // 调用压缩函数,复用OpenCL资源(context/queue/kernel)
    int ret = lzo_compress_core(
        g_state.context,
        g_state.queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path,
        &params,
        &time_us,
        &output_size,
        /* single timing struct */
        &t
    );

    if (ret == 0) {
        resp->status = 0;
        resp->output_size = output_size;
        /* Ensure OCL init time is not part of the per-request total.  The daemon
         * initialises OpenCL at startup, and we explicitly zero that field for
         * per-request timings; but in case some implementation fills it, subtract it.
         */
        unsigned long effective_time_us = time_us;
        if (t.ocl_init_us > 0 && effective_time_us >= t.ocl_init_us) {
            effective_time_us -= t.ocl_init_us;
        }
        resp->time_us = effective_time_us;
        resp->timing = t;
        /* annotate kernel name into the timing struct so clients can print it (include variant) */
        if (sizeof(resp->timing.kernel_name) > 0) {
            char suffix[32] = "";
            if (req->kernel_opt) strncpy(suffix, "_opt", sizeof(suffix)-1);
            memset(resp->timing.kernel_name, 0, sizeof(resp->timing.kernel_name));
            snprintf(resp->timing.kernel_name, sizeof(resp->timing.kernel_name)-1, "%s_%d%s", alg_names[req->alg], req->level, suffix);
        }
        snprintf(resp->message, sizeof(resp->message),
                "Success (saved ~%lums init)", g_state.init_time_ms);

        g_state.requests++;
        g_state.total_time_ms += effective_time_us / 1000;  // 统计用毫秒
    } else {
        resp->status = -1;
        resp->output_size = 0;
        resp->time_us = 0;
        snprintf(resp->message, sizeof(resp->message),
                "Compression failed");
    }

    return ret;
}

/*
 * 处理解压缩请求 (使用预加载的解压缩kernel)
 */
int handle_decompress_request(request_t* req, response_t* resp)
{
    printf("[DAEMON] 处理解压缩请求: %s -> %s\n",
           req->input_path, req->output_path);

    unsigned long time_us;
    size_t output_size;
    timing_t t = {0};

    /* Auto-detect algorithm from file header */
    FILE* f_peek = fopen(req->input_path, "rb");
    if (f_peek) {
        uint16_t magic;
        if (fread(&magic, sizeof(magic), 1, f_peek) == 1 && magic == 0x4C5A) {
            fseek(f_peek, 12, SEEK_CUR); /* skip orig_sz(4), blk_sz(4), nblk(4) */
            uint32_t alg_id;
            if (fread(&alg_id, sizeof(alg_id), 1, f_peek) == 1) {
                if (alg_id <= 1) {
                    req->alg = alg_id;
                    printf("[DAEMON] Auto-detected algorithm from header: %d\n", alg_id);
                }
            }
        }
        fclose(f_peek);
    }

    cl_kernel kernel = get_decompress_kernel(req->alg);
    const char* alg_names[] = {"lzo1x", "lzo1y"};

    if (!kernel) {
        resp->status = -1;
        snprintf(resp->message, sizeof(resp->message), "Failed to get decompress kernel for alg=%d", req->alg);
        return -1;
    }

    printf("[DAEMON]    - 使用解压kernel: %s_decomp\n", alg_names[req->alg]);

    int ret = lzo_decompress_core(
        g_state.context,
        g_state.queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path,
        req->standard_copy,
        req->local_size, req->debug,
        &time_us,
        &output_size,
        &t
    );

    /* Fallback logic removed as we only support scalar for now */
    if (ret != 0) {
        printf("[DAEMON] Decompression failed with ret=%d\n", ret);
    }

    if (ret == 0) {
        resp->status = 0;
        /* Subtract any OCL init time from total so client-facing totals don't include startup init */
        unsigned long effective_time_us = time_us;
        if (t.ocl_init_us > 0 && effective_time_us >= t.ocl_init_us) {
            effective_time_us -= t.ocl_init_us;
        }
        resp->time_us = effective_time_us;
        resp->output_size = output_size;
        /* copy the compact timing structure into the response */
        resp->timing = t;
        snprintf(resp->message, sizeof(resp->message), "OK");
        printf("[DAEMON] 解压缩成功: %zu bytes, %.2f ms\n", output_size, effective_time_us/1000.0);
    } else {
        resp->status = -1;
        snprintf(resp->message, sizeof(resp->message), "Decompression failed");
        printf("[DAEMON] 解压缩失败\n");
    }

    return ret;
}

/*
 * 清理OpenCL资源
 */
void cleanup_opencl_resources(void)
{
    printf("[DAEMON] 清理OpenCL资源...\n");

    /* 清理buffer缓存 (core 提供) */
    cleanup_compress_buffer_cache();
    cleanup_decompress_buffer_cache();

    if (g_state.d_input) clReleaseMemObject(g_state.d_input);
    if (g_state.d_output) clReleaseMemObject(g_state.d_output);
    if (g_state.d_lengths) clReleaseMemObject(g_state.d_lengths);

    // 清理所有压缩kernels和programs
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            if (g_state.kernels_comp[i][j]) clReleaseKernel(g_state.kernels_comp[i][j]);
            if (g_state.programs[i][j]) clReleaseProgram(g_state.programs[i][j]);
            if (g_state.kernels_comp_opt[i][j]) clReleaseKernel(g_state.kernels_comp_opt[i][j]);
            if (g_state.programs_opt[i][j]) clReleaseProgram(g_state.programs_opt[i][j]);
            if (g_state.kernels_comp_debug[i][j]) clReleaseKernel(g_state.kernels_comp_debug[i][j]);
            if (g_state.programs_debug[i][j]) clReleaseProgram(g_state.programs_debug[i][j]);
        }
        // 清理解压缩kernel和program
        if (g_state.kernel_decomp[i]) clReleaseKernel(g_state.kernel_decomp[i]);
        if (g_state.prog_decomp[i]) clReleaseProgram(g_state.prog_decomp[i]);
    }

    if (g_state.queue) clReleaseCommandQueue(g_state.queue);
    if (g_state.context) clReleaseContext(g_state.context);
}

/*
 * 信号处理器
 */
void signal_handler(int sig)
{
    printf("\n[DAEMON] 收到信号 %d,准备退出...\n", sig);
    g_state.running = 0;

    // 关闭server socket以中断accept()阻塞
    if (g_state.server_sock >= 0) {
        shutdown(g_state.server_sock, SHUT_RDWR);
        close(g_state.server_sock);
        g_state.server_sock = -1;
    }
    /* release pidfile lock and remove pid file */
    if (g_state.pid_fd >= 0) {
        flock(g_state.pid_fd, LOCK_UN);
        close(g_state.pid_fd);
        g_state.pid_fd = -1;
        unlink(daemon_pidfile_path());
    }
    /* close log file if we own one */
    if (g_state.logf) {
        fflush(g_state.logf);
        fclose(g_state.logf);
        g_state.logf = NULL;
    }
}

/* Create a pidfile and acquire an exclusive lock to ensure a single daemon instance */
static int create_pidfile(void)
{
    int fd = open(daemon_pidfile_path(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        perror("open pidfile");
        return -1;
    }
    /* Try to acquire an exclusive non-blocking flock */
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        /* Another process holds the lock */
        char buf[64] = {0};
        lseek(fd, 0, SEEK_SET);
        read(fd, buf, sizeof(buf) - 1);
        fprintf(stderr, "[DAEMON] 无法获得pidfile锁，另一个守护进程可能正在运行: %s\n", daemon_pidfile_path());
        if (buf[0]) fprintf(stderr, "[DAEMON] 另一个守护进程PID: %s\n", buf);
        close(fd);
        /* Attempt to interpret the pid in the file: if it's a stale pid (process not found), remove the file and try again once */
        pid_t other_pid = 0;
        if (buf[0]) other_pid = (pid_t)atoi(buf);
        if (other_pid > 1) {
            char proc_comm[128] = {0};
            char comm_path[256];
            snprintf(comm_path, sizeof(comm_path), "/proc/%d/comm", other_pid);
            FILE* pf = fopen(comm_path, "r");
            if (pf) {
                if (fgets(proc_comm, sizeof(proc_comm), pf)) {
                    size_t ln = strlen(proc_comm); if (ln>0 && proc_comm[ln-1]=='\n') proc_comm[ln-1]='\0';
                }
                fclose(pf);
            }
            /* If the process doesn't exist (pf is NULL), or the comm doesn't match, remove the stale pidfile and retry lock once */
            if ((pf == NULL) || (strcmp(proc_comm, "lzo_gpu_daemon") != 0)) {
                /* stale pidfile likely - try removing and retry once */
                fprintf(stderr, "[DAEMON] Found stale pidfile or non-daemon process at PID %d — cleaning up pidfile and retrying\n", other_pid);
                unlink(daemon_pidfile_path());
                close(fd);
                /* reopen and relock once */
                fd = open(daemon_pidfile_path(), O_RDWR | O_CREAT, 0644);
                if (fd < 0) return -1;
                if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                    close(fd);
                    return -1;
                }
                /* success on relock */
            } else {
                /* process exists, so we cannot proceed */
                return -1;
            }
        }
    }
    /* Truncate and write our PID */
    ftruncate(fd, 0);
    char pidbuf[32];
    int n = snprintf(pidbuf, sizeof(pidbuf), "%ld\n", (long)getpid());
    write(fd, pidbuf, n);
    fsync(fd);
    /* Keep fd open (locked) for the lifetime of the process */
    g_state.pid_fd = fd;
    return 0;
}

/*
 * Helper: find if another process with exe name 'lzo_gpu_daemon' (or comm) is running.
 */
static int find_running_daemons(void) {
    DIR *proc = opendir("/proc");
    if (!proc) return 0;
    struct dirent *d;
    pid_t self = getpid();
    int count = 0;
    while ((d = readdir(proc)) != NULL) {
        /* If the dirent name isn't all digits, skip */
        const char *pname = d->d_name;
        int ok = 1;
        for (const char *cp = pname; *cp; ++cp) if (*cp < '0' || *cp > '9') { ok = 0; break; }
        if (!ok) continue;
        pid_t pid = (pid_t)atoi(pname);
        if (pid == 0 || pid == self) continue;
        /* read /proc/<pid>/comm and compare filename */
        char comm_path[256];
        snprintf(comm_path, sizeof(comm_path), "/proc/%d/comm", pid);
        FILE *f = fopen(comm_path, "r");
        if (!f) continue;
        char comm[128];
        if (!fgets(comm, sizeof(comm), f)) { fclose(f); continue; }
        fclose(f);
        /* strip newline */
        size_t ln = strlen(comm);
        if (ln > 0 && comm[ln-1] == '\n') comm[ln-1] = '\0';
        if (strcmp(comm, "lzo_gpu_daemon") == 0) {
            count++;
        } else {
            /* Fallback: read exe link /proc/<pid>/exe and check basename */
            char exe_p[256];
            snprintf(exe_p, sizeof(exe_p), "/proc/%d/exe", pid);
            char exe_path[256];
            ssize_t r = readlink(exe_p, exe_path, sizeof(exe_path)-1);
            if (r > 0) {
                exe_path[r] = '\0';
                const char *base = strrchr(exe_path, '/');
                if (base) base++;
                else base = exe_path;
                if (strcmp(base, "lzo_gpu_daemon") == 0) count++;
            }
        }
    }
    closedir(proc);
    return count;
}

static void remove_pidfile(void)
{
    if (g_state.pid_fd >= 0) {
        flock(g_state.pid_fd, LOCK_UN);
        close(g_state.pid_fd);
        g_state.pid_fd = -1;
        unlink(daemon_pidfile_path());
    }
}

/* Open default log file under /tmp and duplicate stdout/stderr to it so users
 * can inspect output via a known file regardless of how the process is started
 * (foreground, background, system service, etc.).
 */
static int open_default_logfile(void)
{
    /* If stdout is not a TTY, assume the caller has already redirected stdout/stderr
     * (e.g. via shell redirection) and skip duplicating to an extra logfile to avoid
     * writing into /tmp by default.
     */
    if (!isatty(STDOUT_FILENO)) {
        return 0;
    }

    char buf[PATH_MAX];
    const char* path = getenv("LZO_DAEMON_LOG");
    if (!path || path[0] == '\0') {
        const char* out_dir = getenv("OUT_DIR");
        if (out_dir && out_dir[0] != '\0') {
            snprintf(buf, sizeof(buf), "%s/lzo_gpu_daemon.log", out_dir);
            path = buf;
        } else {
            const char* tmpdir = getenv("TMPDIR");
            if (tmpdir && tmpdir[0] != '\0') {
                snprintf(buf, sizeof(buf), "%s/lzo_gpu_daemon.log", tmpdir);
                path = buf;
            } else {
                path = "/tmp/lzo_gpu_daemon.log";
            }
        }
    }

    FILE* f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "[DAEMON] 无法打开日志文件: %s (错误=%d)\n", path, errno);
        return -1;
    }
    /* Duplicate fd so both stdout/stderr are redirected. Keep FILE* open (fclose on exit). */
    if (dup2(fileno(f), STDOUT_FILENO) < 0) {
        fprintf(stderr, "[DAEMON] dup2 stdout failed: %d\n", errno);
        fclose(f);
        return -1;
    }
    if (dup2(fileno(f), STDERR_FILENO) < 0) {
        fprintf(stderr, "[DAEMON] dup2 stderr failed: %d\n", errno);
        fclose(f);
        return -1;
    }
    /* Set line buffering so tail -f sees lines promptly */
    setvbuf(stdout, NULL, _IOLBF, 8192);
    setvbuf(stderr, NULL, _IOLBF, 8192);
    g_state.logf = f;
    fprintf(stdout, "[DAEMON] Logging to %s (fd=%d)\n", path, fileno(f));
    return 0;
}

/*
 * 启动Unix socket服务器
 */
int start_server(void)
{
    struct sockaddr_un addr;

    // 创建socket
    g_state.server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_state.server_sock < 0) {
        perror("socket创建失败");
        return -1;
    }

    // Check for an existing server that may already be bound to the socket.
    // If the socket file exists and connecting succeeds, another daemon is listening.
    if (access(daemon_socket_path(), F_OK) == 0) {
        int csock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (csock >= 0) {
            memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, daemon_socket_path(), sizeof(addr.sun_path) - 1);
            if (connect(csock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                /* Connection succeeded -> socket already in use by a running server */
                fprintf(stderr, "[DAEMON] 另一个守护进程已在运行或socket被占用: %s\n", daemon_socket_path());
                close(csock);
                close(g_state.server_sock);
                return -1;
            }
            close(csock);
            /* If connect failed with ECONNREFUSED/ENOENT or others, continue and unlink stale socket */
        }
    }

    // 删除旧socket文件 (如果已经存在但没有运行中的守护进程)
    unlink(daemon_socket_path());

    // 绑定地址
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, daemon_socket_path(), sizeof(addr.sun_path) - 1);

    if (bind(g_state.server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind失败");
        close(g_state.server_sock);
        return -1;
    }

    // 监听连接
    if (listen(g_state.server_sock, MAX_CLIENTS) < 0) {
        perror("listen失败");
        close(g_state.server_sock);
        unlink(daemon_socket_path());
        return -1;
    }

    printf("[DAEMON] ✅ 服务器启动成功\n");
    printf("[DAEMON]    Socket: %s\n", daemon_socket_path());
    printf("[DAEMON]    PID: %d\n", getpid());

    return 0;
}

/*
 * 主服务循环
 */
void run_server(void)
{
    g_state.running = 1;

    printf("[DAEMON] 等待客户端连接...\n\n");

    while (g_state.running) {
        int client_sock;
        request_t req;
        response_t resp;

        // 接受连接
        client_sock = accept(g_state.server_sock, NULL, NULL);
        if (client_sock < 0) {
            if (errno == EINTR) continue;  // 信号中断
            if (!g_state.running) break;    // 正常退出
            perror("accept失败");
            break;
        }

        // 接收请求
        ssize_t n = recv(client_sock, &req, sizeof(req), 0);
        if (req.debug) {
            fprintf(stderr, "[DAEMON] recv returned n=%zd sizeof(req)=%zu\n", n, sizeof(req)); fflush(stderr);
            fprintf(stderr, "[DAEMON] offsetof: fixed_block_kb=%zu local_size=%zu\n",
                offsetof(request_t, fixed_block_kb), offsetof(request_t, local_size)); fflush(stderr);
        }
        if (n != sizeof(req)) {
            fprintf(stderr, "[DAEMON] 接收请求失败\n");
            close(client_sock);
            continue;
        }
        /* debug: dump the values of key fields to stderr and flush */
        if (req.debug) { fprintf(stderr, "[DAEMON] Received raw request: op=%c fixed_block_kb=%d local_size=%d mt_io=%d\n", req.operation, req.fixed_block_kb, req.local_size, req.mt_io); fflush(stderr); }

         // 处理请求
        memset(&resp, 0, sizeof(resp));
         fprintf(stderr, "[DAEMON] Received request: op=%c input=%s output=%s level=%d fixed_block_kb=%d local_size=%d\n",
             req.operation, req.input_path, req.output_path, req.level, req.fixed_block_kb, req.local_size);
        if (req.operation == 'C') {
            handle_compress_request(&req, &resp);
        } else if (req.operation == 'D') {
            handle_decompress_request(&req, &resp);
        } else {
            resp.status = -1;
            snprintf(resp.message, sizeof(resp.message), "Unknown operation");
        }

        /* 发送响应: 不再将生成的压缩/解压输出文件的内容通过socket发送回客户端。
         * 客户端应仅以 response struct 作为信号，客户端/脚本负责通过daemon返回的元信息/路径完成后续验证/读取。
         */
        ssize_t sent = send(client_sock, &resp, sizeof(resp), 0);
        if (sent != sizeof(resp)) {
            fprintf(stderr, "[DAEMON] 发送响应失败: sent=%zd expected=%zu\n", sent, sizeof(resp));
        }
        close(client_sock);
    }

    printf("\n[DAEMON] 服务循环结束\n");
}

/*
 * 打印统计信息
 */
void print_stats(void)
{
    printf("\n========================================\n");
    printf("守护进程统计信息\n");
    printf("========================================\n");
    printf("总请求数:   %lu\n", g_state.requests);

    if (g_state.requests > 0) {
        unsigned long avg_time = g_state.total_time_ms / g_state.requests;
        unsigned long total_saved = g_state.init_time_ms * g_state.requests;

        printf("初始化耗时: %lu ms (一次性)\n", g_state.init_time_ms);
        printf("平均耗时:   %lu ms/次\n", avg_time);
        printf("每次节省:   %lu ms\n", g_state.init_time_ms);
        printf("累计节省:   %lu ms (%.1f秒)\n",
               total_saved, total_saved / 1000.0);
        printf("性能提升:   %.1f%%\n",
               100.0 * g_state.init_time_ms / (avg_time + g_state.init_time_ms));
    }
    printf("========================================\n");
}

/*
 * 守护进程主函数
 */
int main(int argc, char** argv)
{
    /* simple command-line parsing: support -h/--help */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stdout, "LZO GPU 守护进程 Usage:\n");
            fprintf(stdout, "  %s [options]\n", argv[0]);
            fprintf(stdout, "\nOptions:\n");
            fprintf(stdout, "  -h, --help       Show this help and exit\n");
            fprintf(stdout, "\nEnvironment variables (daemon-level defaults):\n");
            fprintf(stdout, "  LZO_OPENCL_DEVICE=CPU|GPU  Select OpenCL device preference for daemon (env-level)\n");
            fprintf(stdout, "  LZO_DECOMP_CACHE_MB=N      Decompress buffer cache threshold in MB (daemon-level)\n");
            fprintf(stdout, "\nPer-request client options: clients may request behavior (eg. MT I/O, coalesce, stdio buf), but daemon will only honor per-request values at runtime if configured to do so. See client help.\n");
            return 0;
        }
    }
    /* initialize logfile pointer and default stdout/stderr duplication to /tmp */
    g_state.logf = NULL;
    open_default_logfile();

    printf("========================================\n");
    printf("LZO GPU守护进程\n");
    printf("========================================\n\n");

    // 注册信号处理器
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* initialize pidfile fd */
    g_state.pid_fd = -1;
    /* detect existing daemons by name (ensure singleton); if found, abort. */
    int existing = find_running_daemons();
    if (existing > 0) {
        fprintf(stderr, "[DAEMON] 错误: 发现 %d 个 'lzo_gpu_daemon' 进程正在运行; 仅允许单个守护进程。请停止重复进程后重试\n", existing);
        return 1;
    }
    /* Create pidfile and acquire lock to ensure only one daemon instance */
    if (create_pidfile() != 0) {
        fprintf(stderr, "[DAEMON] 无法创建/锁定 pidfile，退出\n");
        return 1;
    }

    // 初始化OpenCL资源 (仅一次)
    if (init_opencl_resources() != 0) {
        fprintf(stderr, "OpenCL初始化失败\n");
        remove_pidfile();
        return 1;
    }

    // 启动服务器
    if (start_server() != 0) {
        cleanup_opencl_resources();
        remove_pidfile();
        return 1;
    }

    // 运行服务循环
    run_server();

    // 清理资源
    if (g_state.server_sock >= 0) {
        close(g_state.server_sock);
        g_state.server_sock = -1;
    }
    unlink(daemon_socket_path());
    cleanup_opencl_resources();
    remove_pidfile();
    /* Close logfile if we opened one */
    if (g_state.logf) {
        fflush(g_state.logf);
        fclose(g_state.logf);
        g_state.logf = NULL;
    }

    // 打印统计信息
    print_stats();

    printf("\n[DAEMON] 已退出\n");
    return 0;
}
