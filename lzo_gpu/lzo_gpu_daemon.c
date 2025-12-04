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

/* 声明daemon_decompress.c中的函数 */
extern int daemon_decompress(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel kernel, int prefer_vec, const char* input_path, const char* output_path,
    unsigned long* time_us_out, size_t* output_size_out, timing_t* t_out
);

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

    /* 多kernel支持 - 压缩级别映射 */
    cl_program programs[4];      // lzo1x_1, 1k, 1l, 1o
    cl_kernel kernels_comp[4];   // 对应的压缩kernel
    cl_program prog_decomp;      // 解压缩program (标量)
    cl_program prog_decomp_vec;  // 解压缩program (向量)
    cl_kernel kernel_decomp;     // 解压缩kernel (标量)
    cl_kernel kernel_decomp_vec; // 解压缩kernel (向量)

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

/* 外部压缩函数声明 */
extern int daemon_compress(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel kernel,
    const char* input_path, const char* output_path,
    int level,
    /* options (from client request) */
    int standard_copy, int mt_io, int mt_threads, int fixed_block_kb, int async_upload,
    /* per-request coalesce/stdio overrides */
    int coalesce_output, int coalesce_chunk_mb, int coalesce_max_mb, int stdio_buf_mb,
    unsigned long* time_us, size_t* output_size,
    timing_t* t_out
);
/* 请求协议 */
typedef struct {
    char operation;      // 'C'=compress, 'D'=decompress
    char input_path[256];
    char output_path[256];
    int level;           // 压缩级别 1-9
    size_t input_size;
    /* options from client/env */
    int standard_copy; /* 0/1 */
    int mt_io;        /* 0/1 */
    int mt_threads;   /* number of IO threads */
    int fixed_block_kb; /* 0=no fixed size, else KB */
    /* force_map deprecated: mapping behavior now controlled by standard_copy */
    int async_upload; /* 0/1 */
    int prefer_cpu;   /* 0=gpu, 1=cpu */
    int decomp_vec;   /* 0=scalar decomp, 1=vector (default) */
    /* Per-request overrides for coalescing and stdio buffer - -1 = unspecified */
    int coalesce_output;  /* -1 unspecified; 0=off; 1=on */
    int coalesce_chunk_mb;/* -1 unspecified; positive = chunk size in MB */
    int coalesce_max_mb;  /* -1 unspecified; positive = max MB threshold for single coalesce */
    int stdio_buf_mb;     /* -1 unspecified; positive = MB for stdio buffer */
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

/* 辅助函数: 读取文件内容 */
static char* read_file_content(const char* path, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = malloc(len + 1);
    if (fread(buf, 1, len, f) != len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[len] = '\0';
    fclose(f);

    if (out_len) *out_len = len;
    return buf;
}

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

    // 3. 创建命令队列 (使用OpenCL 2.0的新API)
    // 启用 CL_QUEUE_PROFILING_ENABLE，方便后续用事件查询精确的 kernel 执行时间
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    g_state.queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device,
                                                        props, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建命令队列失败: %d\n", err);
        return -1;
    }

    // 4. 加载多个压缩kernel (lzo1x_1, 1k, 1l, 1o)
    // 直接使用独立.cl文件,每个有不同的压缩算法实现
    // 不使用lzo1x_comp.cl前端,因为它只适用于lzo1x_1
    const char* compress_bases[] = {"lzo1x_1", "lzo1x_1k", "lzo1x_1l", "lzo1x_1o"};
    const char* compress_sources[] = {"lzo1x_1.cl", "lzo1x_1k.cl", "lzo1x_1l.cl", "lzo1x_1o.cl"};

    printf("[DAEMON] 加载压缩kernels...\n");
    for (int i = 0; i < 4; i++) {
        // 尝试加载binary,失败则编译源码
        char bin_path[256];
        snprintf(bin_path, sizeof(bin_path), "%s.bin", compress_bases[i]);

        FILE* fb = fopen(bin_path, "rb");
        if (fb) {
            fseek(fb, 0, SEEK_END);
            long bsz = ftell(fb);
            fseek(fb, 0, SEEK_SET);

            unsigned char* bin = malloc(bsz);
            if (fread(bin, 1, bsz, fb) == (size_t)bsz) {
                cl_int binary_status;
                g_state.programs[i] = clCreateProgramWithBinary(g_state.context, 1,
                                                                &g_state.device,
                                                                (const size_t*)&bsz,
                                                                (const unsigned char**)&bin,
                                                                &binary_status, &err);

                if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                    err = clBuildProgram(g_state.programs[i], 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
                    if (err == CL_SUCCESS) {
                        printf("[DAEMON]    - %s: 从预编译binary加载 ✅\n", compress_bases[i]);
                        free(bin);
                        fclose(fb);

                        g_state.kernels_comp[i] = clCreateKernel(g_state.programs[i],
                                                                 "lzo1x_block_compress", &err);
                        if (err != CL_SUCCESS) {
                            fprintf(stderr, "创建kernel失败: %s (err=%d)\n", compress_bases[i], err);
                            return -1;
                        }
                        continue;
                    }
                }
                if (g_state.programs[i]) clReleaseProgram(g_state.programs[i]);
            }
            free(bin);
            fclose(fb);
        }

        // 回退到源码编译
        size_t src_len;
        char* src = read_file_content(compress_sources[i], &src_len);
        if (!src) {
            fprintf(stderr, "[DAEMON] 无法读取源文件: %s\n", compress_sources[i]);
            return -1;
        }

        g_state.programs[i] = clCreateProgramWithSource(g_state.context, 1,
                                                        (const char**)&src, &src_len, &err);
        free(src);

        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DAEMON] 创建程序失败: %s (err=%d)\n", compress_sources[i], err);
            return -1;
        }

        err = clBuildProgram(g_state.programs[i], 1, &g_state.device, "-cl-std=CL2.0 -I.", NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DAEMON] 编译内核失败: %s (err=%d)\n", compress_sources[i], err);
            size_t log_sz;
            clGetProgramBuildInfo(g_state.programs[i], g_state.device,
                                 CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            if (log_sz > 0) {
                char* log = malloc(log_sz + 1);
                clGetProgramBuildInfo(g_state.programs[i], g_state.device,
                                     CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
                log[log_sz] = '\0';
                fprintf(stderr, "%s\n", log);
                free(log);
            }
            return -1;
        }

        printf("[DAEMON]    - %s: 从源码编译 ⚠️\n", compress_bases[i]);

        g_state.kernels_comp[i] = clCreateKernel(g_state.programs[i],
                                                 "lzo1x_block_compress", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "创建kernel失败: %s (err=%d)\n", compress_bases[i], err);
            return -1;
        }
    }

decomp_done:
    // 5. 加载解压缩kernel (标量+向量)
    printf("[DAEMON] 加载解压缩kernel (标量+向量)...\n");

    // 标量版本
    FILE* fb_decomp = fopen("lzo1x_decomp.bin", "rb");
    if (fb_decomp) {
        fseek(fb_decomp, 0, SEEK_END);
        long bsz = ftell(fb_decomp);
        fseek(fb_decomp, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin, 1, bsz, fb_decomp) == (size_t)bsz) {
            cl_int binary_status;
            g_state.prog_decomp = clCreateProgramWithBinary(g_state.context, 1,
                                                           &g_state.device,
                                                           (const size_t*)&bsz,
                                                           (const unsigned char**)&bin,
                                                           &binary_status, &err);
            if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                err = clBuildProgram(g_state.prog_decomp, 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
                if (err == CL_SUCCESS) {
                    printf("[DAEMON]    - decompress: 从预编译binary加载 ✅\n");
                    g_state.kernel_decomp = clCreateKernel(g_state.prog_decomp,
                                                          "lzo1x_block_decompress", &err);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "创建解压缩kernel失败 (err=%d)\n", err);
                        return -1;
                    }
                }
            }
            if (g_state.prog_decomp && !g_state.kernel_decomp) clReleaseProgram(g_state.prog_decomp);
        }
        free(bin);
        fclose(fb_decomp);
    }
    // 回退到源码编译
    if (!g_state.kernel_decomp) {
        size_t src_len_decomp;
        char* src_decomp = read_file_content("lzo1x_decomp.cl", &src_len_decomp);
        if (!src_decomp) {
            fprintf(stderr, "[DAEMON] 无法读取源文件: lzo1x_decomp.cl\n");
            return -1;
        }
        g_state.prog_decomp = clCreateProgramWithSource(g_state.context, 1,
                                                        (const char**)&src_decomp,
                                                        &src_len_decomp, &err);
        free(src_decomp);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DAEMON] 创建解压缩程序失败 (err=%d)\n", err);
            return -1;
        }
        err = clBuildProgram(g_state.prog_decomp, 1, &g_state.device, "-cl-std=CL2.0 -I.", NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DAEMON] 编译解压缩内核失败 (err=%d)\n", err);
            size_t log_sz;
            clGetProgramBuildInfo(g_state.prog_decomp, g_state.device,
                                 CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            if (log_sz > 0) {
                char* log = malloc(log_sz + 1);
                clGetProgramBuildInfo(g_state.prog_decomp, g_state.device,
                                     CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
                log[log_sz] = '\0';
                fprintf(stderr, "%s\n", log);
                free(log);
            }
            return -1;
        }
        printf("[DAEMON]    - decompress: 从源码编译 ⚠️\n");
        g_state.kernel_decomp = clCreateKernel(g_state.prog_decomp,
                                              "lzo1x_block_decompress", &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "创建解压缩kernel失败 (err=%d)\n", err);
            return -1;
        }
    }

    // 向量版本
    FILE* fb_decomp_vec = fopen("lzo1x_decomp_vec.bin", "rb");
    if (fb_decomp_vec) {
        fseek(fb_decomp_vec, 0, SEEK_END);
        long bsz = ftell(fb_decomp_vec);
        fseek(fb_decomp_vec, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin, 1, bsz, fb_decomp_vec) == (size_t)bsz) {
            cl_int binary_status;
            g_state.prog_decomp_vec = clCreateProgramWithBinary(g_state.context, 1,
                                                               &g_state.device,
                                                               (const size_t*)&bsz,
                                                               (const unsigned char**)&bin,
                                                               &binary_status, &err);
            if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                err = clBuildProgram(g_state.prog_decomp_vec, 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
                if (err == CL_SUCCESS) {
                    printf("[DAEMON]    - decomp_vec: 从预编译binary加载 ✅\n");
                    /* kernel entry in vec program uses the same kernel name
                     * (lzo1x_block_decompress) as the scalar program; don't suffix with _vec
                     */
                    g_state.kernel_decomp_vec = clCreateKernel(g_state.prog_decomp_vec,
                                                              "lzo1x_block_decompress", &err);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "创建decomp_vec kernel失败 (err=%d)\n", err);
                        g_state.kernel_decomp_vec = NULL;
                    }
                }
            }
            if (g_state.prog_decomp_vec && !g_state.kernel_decomp_vec) clReleaseProgram(g_state.prog_decomp_vec);
        }
        free(bin);
        fclose(fb_decomp_vec);
    }
    // 回退到源码编译
    if (!g_state.kernel_decomp_vec) {
        size_t src_len_decomp_vec;
        char* src_decomp_vec = read_file_content("lzo1x_decomp_vec.cl", &src_len_decomp_vec);
        if (src_decomp_vec) {
            g_state.prog_decomp_vec = clCreateProgramWithSource(g_state.context, 1,
                                                               (const char**)&src_decomp_vec,
                                                               &src_len_decomp_vec, &err);
            free(src_decomp_vec);
            if (err == CL_SUCCESS) {
                err = clBuildProgram(g_state.prog_decomp_vec, 1, &g_state.device, "-cl-std=CL2.0 -I.", NULL, NULL);
                if (err == CL_SUCCESS) {
                    printf("[DAEMON]    - decomp_vec: 从源码编译 ⚠️\n");
                    g_state.kernel_decomp_vec = clCreateKernel(g_state.prog_decomp_vec,
                                                              "lzo1x_block_decompress", &err);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "创建decomp_vec kernel失败 (err=%d)\n", err);
                        g_state.kernel_decomp_vec = NULL;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_state.init_time_ms = (t_end.tv_sec - t_start.tv_sec) * 1000 +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1000000;

    printf("[DAEMON] ✅ OpenCL资源初始化完成\n");
    printf("[DAEMON]    - 上下文: 常驻内存\n");
    printf("[DAEMON]    - 压缩kernels: lzo1x_1/1k/1l/1o\n");
    printf("[DAEMON]    - 解压缩kernel: lzo1x_decomp + lzo1x_decomp_vec\n");
    printf("[DAEMON]    - 缓冲区: 动态分配 (每次请求)\n");
    printf("[DAEMON]    - 初始化耗时: %lu ms\n", g_state.init_time_ms);

    return 0;
}

/* 外部压缩函数声明 */
extern int daemon_compress(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel kernel,
    const char* input_path, const char* output_path,
    int level,
    /* options (from client request) */
    int standard_copy, int mt_io, int mt_threads, int fixed_block_kb, int async_upload,
    /* per-request coalesce/stdio overrides */
    int coalesce_output, int coalesce_chunk_mb, int coalesce_max_mb, int stdio_buf_mb,
    unsigned long* time_us, size_t* output_size,
    /* compact per-stage timing structure */
    timing_t* t_out
);

/* 根据压缩级别选择kernel */
static cl_kernel select_kernel_by_level(int level)
{
    // level映射:
    //   1-3: lzo1x_1  (标准压缩)
    //   4-6: lzo1x_1k (1KB优化)
    //   7-8: lzo1x_1l (轻量级)
    //   9:   lzo1x_1o (最优压缩)
    if (level >= 1 && level <= 3) {
        return g_state.kernels_comp[0];  // lzo1x_1
    } else if (level >= 4 && level <= 6) {
        return g_state.kernels_comp[1];  // lzo1x_1k
    } else if (level >= 7 && level <= 8) {
        return g_state.kernels_comp[2];  // lzo1x_1l
    } else {
        return g_state.kernels_comp[3];  // lzo1x_1o (level 9)
    }
}

/*
 * 处理压缩请求 (复用已初始化的资源)
 */
int handle_compress_request(request_t* req, response_t* resp)
{
    printf("[DAEMON] 处理压缩请求: %s -> %s (level=%d)\n",
           req->input_path, req->output_path, req->level);

    // 根据level选择合适的kernel
    cl_kernel kernel = select_kernel_by_level(req->level);
    const char* kernel_names[] = {"lzo1x_1", "lzo1x_1k", "lzo1x_1l", "lzo1x_1o"};

    // 根据level确定kernel名称
    int kernel_idx;
    if (req->level >= 1 && req->level <= 3) {
        kernel_idx = 0;  // lzo1x_1
    } else if (req->level >= 4 && req->level <= 6) {
        kernel_idx = 1;  // lzo1x_1k
    } else if (req->level >= 7 && req->level <= 8) {
        kernel_idx = 2;  // lzo1x_1l
    } else {
        kernel_idx = 3;  // lzo1x_1o (level 9+)
    }
    printf("[DAEMON]    - 使用kernel: %s\n", kernel_names[kernel_idx]);

    unsigned long time_us = 0;
    size_t output_size = 0;
    timing_t t = {0};

    // 调用压缩函数,复用OpenCL资源(context/queue/kernel)
    int ret = daemon_compress(
        g_state.context,
        g_state.queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path,
        req->level,
        /* options */
        req->standard_copy, req->mt_io, req->mt_threads, req->fixed_block_kb, req->async_upload,
        /* per-request coalesce/stdio overrides */
        req->coalesce_output, req->coalesce_chunk_mb, req->coalesce_max_mb, req->stdio_buf_mb,
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

    int prefer_vec = (g_state.kernel_decomp_vec != NULL) ? 1 : 0;
    cl_kernel kernel = prefer_vec ? g_state.kernel_decomp_vec : g_state.kernel_decomp;
    const char* kernel_name = prefer_vec ? "lzo1x_decomp_vec" : "lzo1x_decomp";
    printf("[DAEMON]    - 使用解压kernel: %s\n", kernel_name);

    int ret = daemon_decompress(
        g_state.context,
        g_state.queue,
        g_state.device,
        kernel,
        prefer_vec,
        req->input_path,
        req->output_path,
        &time_us,
        &output_size,
        &t
    );

    /* 如果向量化kernel因为不可压缩或运行失败导致非0返回，回退到标量kernel并重试（容错） */
    if (ret != 0 && g_state.kernel_decomp && kernel != g_state.kernel_decomp) {
        printf("[DAEMON] vectorized decompressor not suitable or failed, falling back to scalar kernel\n");
        kernel = g_state.kernel_decomp;
        kernel_name = "lzo1x_decomp";
        /* prefer_vec=0 表示标量重试 */
        ret = daemon_decompress(
            g_state.context,
            g_state.queue,
            g_state.device,
            kernel,
            0,
            req->input_path,
            req->output_path,
            &time_us,
            &output_size,
            &t
        );
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

    /* 清理buffer缓存 */
    extern void cleanup_compress_buffer_cache(void);
    extern void cleanup_decompress_buffer_cache(void);
    cleanup_compress_buffer_cache();
    cleanup_decompress_buffer_cache();

    if (g_state.d_input) clReleaseMemObject(g_state.d_input);
    if (g_state.d_output) clReleaseMemObject(g_state.d_output);
    if (g_state.d_lengths) clReleaseMemObject(g_state.d_lengths);

    // 清理所有压缩kernels和programs
    for (int i = 0; i < 4; i++) {
        if (g_state.kernels_comp[i]) clReleaseKernel(g_state.kernels_comp[i]);
        if (g_state.programs[i]) clReleaseProgram(g_state.programs[i]);
    }

    // 清理解压缩kernel和program
    if (g_state.kernel_decomp) clReleaseKernel(g_state.kernel_decomp);
    if (g_state.prog_decomp) clReleaseProgram(g_state.prog_decomp);

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
        if (n != sizeof(req)) {
            fprintf(stderr, "[DAEMON] 接收请求失败\n");
            close(client_sock);
            continue;
        }

        // 处理请求
        memset(&resp, 0, sizeof(resp));
        if (req.operation == 'C') {
            handle_compress_request(&req, &resp);
        } else if (req.operation == 'D') {
            handle_decompress_request(&req, &resp);
        } else {
            resp.status = -1;
            snprintf(resp.message, sizeof(resp.message), "Unknown operation");
        }

        /* 发送响应 (原子 - 仅发 response struct)
         * NOTE: 不再将生成的压缩/解压输出文件的内容通过socket发送回客户端。
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
            fprintf(stdout, "\nEnvironment variables (daemon & client / per-request options):\n");
            fprintf(stdout, "  LZO_ASYNC_UPLOAD=0|1    (per-request) allow daemon to perform asynchronous uploads via non-blocking writes\n");
            fprintf(stdout, "  LZO_STANDARD_COPY=0|1    Use standard host->device copy (default: 0 for zero-copy)\n");
            fprintf(stdout, "  LZO_MT_IO=0|1            Enable multi-threaded I/O (pread) for reads/uploads\n");
            fprintf(stdout, "  LZO_MT_IO_THREADS=N      Threads for multi-threaded I/O (1-32, default: %d)\n", LZO_DEFAULT_MT_IO_THREADS);
            fprintf(stdout, "  LZO_FIXED_BLOCK_SIZE=N   Fixed block size in KB (overrides adaptive choice)\n");
            /* LZO_FORCE_MAP removed: mapping path controlled by LZO_STANDARD_COPY */
            fprintf(stdout, "  LZO_OPENCL_DEVICE=CPU|GPU Select OpenCL device preference for daemon (env-level)\n");
            fprintf(stdout, "  LZO_DECOMP_VEC=0|1       Prefer vectorized decompressor if available (default:1)\n");
            /* LZO_FORCE_NBLK removed: block target heuristics are determined by device and adaptive logic */
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
