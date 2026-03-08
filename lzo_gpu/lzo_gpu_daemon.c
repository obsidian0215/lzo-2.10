#define _POSIX_C_SOURCE 200809L
#include <time.h>
/*
 * lzo_gpu_daemon.c - LZO GPU守护进程实现
 *
 * 功能: 保持OpenCL上下文和缓冲区常驻内存,通过Unix socket接收压缩请求
 * 性能: 节省549ms/次的初始化开销 (OCL初始化44ms + 缓冲区分配505ms)
 *
 * 使用:
 *   启动守护进程: ./lzo_gpu --daemon
 *   客户端请求:   ./lzo_gpu --use-daemon <file>
 *   停止守护进程: ./lzo_gpu --stop-daemon
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
#include <pthread.h>
#include "timing.h"
#include <dirent.h>
#include "lzo_defaults.h"
#include "lzo_gpu_utils.h"
#include "lzo_gpu_core.h"
#include <stddef.h>
#include <libgen.h>

#define core_now_ns lzo_now_ns

/* Decompress is implemented in lzo_gpu_core: use lzo_decompress_core() */

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"
#define PID_FILE "/tmp/lzo_gpu_daemon.pid"
#define MAX_CLIENTS 5
#define MAX_BUFFER_SIZE (256 * 1024 * 1024)  // 256MB - 减小预配避免OOM, 针对超大文件CORE会自动扩容

#define MAX_WORKERS 4

static const char* daemon_socket_path(void) { return lzo_daemon_socket_path(); }
static const char* daemon_pidfile_path(void) { return lzo_daemon_pidfile_path(); }

/* Worker resource */
typedef struct {
    cl_command_queue queue;
    lzo_gpu_workspace_t ws;
    int in_use;
    pthread_mutex_t lock;

    /* Private kernels for this worker to avoid race conditions on clSetKernelArg */
    /* Alg: 0=1x, 1=1y | Bits: 0-8 (10-18) */
    cl_kernel kernels_comp[2][9];
    cl_kernel kernels_decomp[2];
} worker_res_t;

typedef struct {
    int client_sock;
} worker_thread_args_t;

/* 守护进程全局状态 */
typedef struct {
    /* OpenCL资源 - 常驻内存 */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;

    /* Dedicated queues and workspaces for concurrent workers */
    worker_res_t workers[MAX_WORKERS];

    /* 多kernel支持 - 算法(2) x 字典大小(9) */
    /* Alg: 0=1x, 1=1y */
    /* Bits: 0=10, 1=11, 2=12, 3=13, 4=14, 5=15, 6=16, 7=17, 8=18 */
    cl_program programs[2][9];

    /* 解压缩kernels (每个算法一个) */
    cl_program prog_decomp[2];

    /* 统计信息 */
    unsigned long requests;
    unsigned long total_time_ms;
    unsigned long init_time_ms;  // 实际测量的初始化时间

    /* 服务器socket */
    int server_sock;
    volatile int running;
    pthread_mutex_t compile_lock; /* Protects lazy program loading */
    pthread_mutex_t stats_lock;   /* Protects counters */
    int pid_fd; /* file descriptor for pidfile lock (if any) */
    FILE* logf; /* optional logfile (duplicate of stdout/stderr) */
} daemon_state_t;

static daemon_state_t g_state = {0};

/* Forward declarations */
static int create_pidfile(void);
static void remove_pidfile(void);

/* Find the named kernel/source file under the common candidate locations.
 * If found, place the resolved absolute (or relative) path into `out` and return 0.
 * If not found, return -1.
 */
/* file location helpers replaced by shared implementation in lzo_gpu_utils.c */

/* Open a kernel file (binary or source) by searching the common locations. */
/* Replaced open_kernel_file with direct calls to lzo_find_file_path + fopen in callers */


#include "lzo_gpu_protocol.h"

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
        err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_DEFAULT, 1,
                             &g_state.device, NULL);
    }
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(g_state.platform, CL_DEVICE_TYPE_ALL, 1,
                             &g_state.device, NULL);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "获取OpenCL设备失败: %d\n", err);
        return -1;
    }

    // 2. 创建上下文 (常驻)
    g_state.context = clCreateContext(NULL, 1, &g_state.device,
                                      NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建上下文失败: %d\n", err);
        return -1;
    }

    pthread_mutex_init(&g_state.compile_lock, NULL);
    pthread_mutex_init(&g_state.stats_lock, NULL);

    // 3. 为每个Worker初始化资源
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    for (int i = 0; i < MAX_WORKERS; i++) {
        g_state.workers[i].queue = clCreateCommandQueueWithProperties(g_state.context, g_state.device, props, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "创建Worker %d 命令队列失败: %d\n", i, err);
            return -1;
        }
        lzo_gpu_workspace_init(&g_state.workers[i].ws);

        /* 预分配 GPU 缓冲区,实现真正的“输入/输出缓冲常驻” */
        size_t prealloc = MAX_BUFFER_SIZE;
        printf("[DAEMON]    - 为Worker %d 预分配 %.1f MB 缓冲区...\n", i, prealloc / 1024.0 / 1024.0);

        cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR;
        g_state.workers[i].ws.d_in = clCreateBuffer(g_state.context, flags | CL_MEM_READ_ONLY, prealloc, NULL, &err);
        if (err == CL_SUCCESS) g_state.workers[i].ws.in_size = prealloc;

        /* 计算压缩最差情况所需的输出大小 */
        size_t worst_out = prealloc + prealloc / 16 + 64 + 3;
        g_state.workers[i].ws.d_out = clCreateBuffer(g_state.context, flags | CL_MEM_WRITE_ONLY, worst_out, NULL, &err);
        if (err == CL_SUCCESS) g_state.workers[i].ws.out_size = worst_out;

        g_state.workers[i].ws.d_len = clCreateBuffer(g_state.context, flags | CL_MEM_READ_WRITE, (prealloc / 1024) * sizeof(cl_uint), NULL, &err);
        if (err == CL_SUCCESS) g_state.workers[i].ws.len_size = (prealloc / 1024) * sizeof(cl_uint);

        pthread_mutex_init(&g_state.workers[i].lock, NULL);
        g_state.workers[i].in_use = 0;

        memset(g_state.workers[i].kernels_comp, 0, sizeof(g_state.workers[i].kernels_comp));

        memset(g_state.workers[i].kernels_decomp, 0, sizeof(g_state.workers[i].kernels_decomp));
    }

    // 4. 初始化全局Program数组
    memset(g_state.programs, 0, sizeof(g_state.programs));
    memset(g_state.prog_decomp, 0, sizeof(g_state.prog_decomp));

    printf("[DAEMON] Kernels将按需编译 (Lazy Loading)...\n");

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_state.init_time_ms = (t_end.tv_sec - t_start.tv_sec) * 1000 +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1000000;

    printf("[DAEMON] ✅ OpenCL资源初始化完成\n");
    printf("[DAEMON]    - 上下文: 常驻内存\n");
    printf("[DAEMON]    - 压缩kernels: 1x/1y (10-18 bits)\n");
    printf("[DAEMON]    - 解压缩kernel: 1x/1y\n");
    printf("[DAEMON]    - 缓冲区: 预分配 (常驻 %.1f MB/Worker)\n", MAX_BUFFER_SIZE / 1024.0 / 1024.0);
    printf("[DAEMON]    - 初始化耗时: %lu ms\n", g_state.init_time_ms);
    g_ocl_init_us = g_state.init_time_ms * 1000;

    return 0;
}

/* 获取或编译压缩Program (线程安全,全局缓存) */
static cl_program get_compress_program(int alg, int bits)
{
    if (alg < 0 || alg > 1 || bits < 10 || bits > 18) return NULL;
    int bit_idx = bits - 10;
    cl_program *p_prog = &g_state.programs[alg][bit_idx];

    pthread_mutex_lock(&g_state.compile_lock);
    if (*p_prog) {
        pthread_mutex_unlock(&g_state.compile_lock);
        return *p_prog;
    }

    const char* alg_names[] = {"lzo1x", "lzo1y"};
    cl_int err;

    /* 1. Try to load precompiled binary first */
    char bin_name[80];
    size_t bin_sz = 0;
    unsigned char* bin = NULL;

    snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", alg_names[alg], bits);

    uint64_t tk1 = core_now_ns();
    bin = (unsigned char*)lzo_read_file(bin_name, &bin_sz);
    if (bin) {
        printf("[DAEMON] 加载预编译Kernel Program: %s\n", bin_name);
        cl_int binary_status;
        *p_prog = clCreateProgramWithBinary(g_state.context, 1, &g_state.device,
                                           &bin_sz, (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
            clBuildProgram(*p_prog, 1, &g_state.device, "-cl-std=CL2.0", NULL, NULL);
            uint64_t tk2 = core_now_ns();
            g_kernel_load_us += (unsigned long)((tk2 - tk1) / 1000);
            pthread_mutex_unlock(&g_state.compile_lock);
            return *p_prog;
        }
    }

    /* 2. Fallback: Compile from source */
    char src_file[64];
    snprintf(src_file, sizeof(src_file), "%s.cl", alg_names[alg]);
    printf("[DAEMON] 尝试编译 Kernel Program: %s (D_BITS=%d)...\n", src_file, bits);

    size_t src_len;
    char* src = lzo_read_file(src_file, &src_len);
    if (src) {
        char resolved_src[PATH_MAX] = {0};
        char include_opt[256] = "";
        if (lzo_find_file_path(src_file, resolved_src, sizeof(resolved_src)) == 0) {
            char *slash = strrchr(resolved_src, '/');
            if (slash) { *slash = '\0'; snprintf(include_opt, sizeof(include_opt), " -I%s", resolved_src); }
        }

        *p_prog = clCreateProgramWithSource(g_state.context, 1, (const char**)&src, &src_len, &err);
        free(src);
        if (err == CL_SUCCESS) {
            char build_opts_with_inc[512];
            snprintf(build_opts_with_inc, sizeof(build_opts_with_inc), "-cl-std=CL2.0 -I.%s -D D_BITS=%d", include_opt, bits);
            err = clBuildProgram(*p_prog, 1, &g_state.device, build_opts_with_inc, NULL, NULL);
            uint64_t tk2_src = core_now_ns();
            g_kernel_load_us += (unsigned long)((tk2_src - tk1) / 1000);
            if (err == CL_SUCCESS) {
                pthread_mutex_unlock(&g_state.compile_lock);
                return *p_prog;
            }

            /* capture log on error */
            size_t log_sz; clGetProgramBuildInfo(*p_prog, g_state.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            if (log_sz > 0) { char* log = malloc(log_sz + 1); clGetProgramBuildInfo(*p_prog, g_state.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz] = '\0'; fprintf(stderr, "%s\n", log); free(log); }
            clReleaseProgram(*p_prog); *p_prog = NULL;
        }
    }
    pthread_mutex_unlock(&g_state.compile_lock);
    return NULL;
}

static cl_kernel get_compress_kernel_for_worker(worker_res_t* worker, int alg, int bits)
{
    if (alg < 0 || alg > 1 || bits < 10 || bits > 18) return NULL;
    int bit_idx = bits - 10;
    cl_kernel *p_krn = &worker->kernels_comp[alg][bit_idx];

    if (*p_krn) return *p_krn;

    cl_program prog = get_compress_program(alg, bits);
    if (!prog) return NULL;

    const char* alg_names[] = {"lzo1x", "lzo1y"};
    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name), "%s_block_compress", alg_names[alg]);

    cl_int err;
    *p_krn = clCreateKernel(prog, kernel_name, &err);
    return *p_krn;
}

/* 获取或编译解压缩Program (线程安全) */
static cl_program get_decompress_program(int alg)
{
    if (alg < 0 || alg > 1) return NULL;

    pthread_mutex_lock(&g_state.compile_lock);
    if (g_state.prog_decomp[alg]) {
        pthread_mutex_unlock(&g_state.compile_lock);
        return g_state.prog_decomp[alg];
    }

    const char* alg_names[] = {"lzo1x", "lzo1y"};
    char build_log[8192] = {0};
    uint64_t tk1 = core_now_ns();

    printf("[DAEMON] 加载解压缩Program: %s\n", alg_names[alg]);
    g_state.prog_decomp[alg] = lzo_load_program_with_dbits(
        g_state.context,
        g_state.device,
        alg_names[alg],
        LZO_DEFAULT_COMP_LEVEL,
        build_log,
        sizeof(build_log)
    );
    if (g_state.prog_decomp[alg]) {
        uint64_t tk2 = core_now_ns();
        g_kernel_load_us += (unsigned long)((tk2 - tk1) / 1000);
        pthread_mutex_unlock(&g_state.compile_lock);
        return g_state.prog_decomp[alg];
    }

    if (build_log[0]) {
        fprintf(stderr, "[DAEMON] 解压缩Program加载失败: %s\n", build_log);
    }
    pthread_mutex_unlock(&g_state.compile_lock);
    return NULL;
}

static cl_kernel get_decompress_kernel_for_worker(worker_res_t* worker, int alg)
{
    if (alg < 0 || alg > 1) return NULL;
    if (worker->kernels_decomp[alg]) return worker->kernels_decomp[alg];

    cl_program prog = get_decompress_program(alg);
    if (!prog) return NULL;

    const char* alg_names[] = {"lzo1x", "lzo1y"};
    char kernel_name[64];
    snprintf(kernel_name, sizeof(kernel_name), "%s_block_decompress", alg_names[alg]);

    cl_int err;
    worker->kernels_decomp[alg] = clCreateKernel(prog, kernel_name, &err);
    return worker->kernels_decomp[alg];
}

/*
 * 处理压缩请求 (复用已初始化的资源)
 */
int handle_compress_request(request_t* req, response_t* resp, worker_res_t* worker)
{
    /* If output_path is empty, generate a default one: input + ".lzo" */
    if (req->output_path[0] == '\0') {
        snprintf(req->output_path, sizeof(req->output_path), "%s.lzo", req->input_path);
    }

    printf("[DAEMON] 处理压缩请求: %s -> %s (level=%d)\n",
           req->input_path, req->output_path, req->level);

    // 根据alg和level选择合适的worker私有kernel
    cl_kernel kernel = get_compress_kernel_for_worker(worker, req->alg, req->level);
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
        .block_size = req->block_size,
        .local_size_param = req->local_size,
        .debug = 0
    };

    int ret = lzo_compress_core(
        g_state.context,
        worker->queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path,
        &params,
        &worker->ws,
        &time_us,
        &output_size,
        /* single timing struct */
        &t
    );

    if (ret == 0) {
        resp->status = 0;
        resp->out_size = output_size;
        /* Inclusive total time for this request (daemon mode excludes global init) */
        resp->time_us = time_us;
        resp->timing = t;
        /* Per user request: in daemon mode, ocl_setup_us is always 0 for all requests */
        resp->timing.ocl_setup_us = 0;

        snprintf(resp->message, sizeof(resp->message),
                "Success (saved ~%lums init)", g_state.init_time_ms);

        pthread_mutex_lock(&g_state.stats_lock);
        g_state.requests++;
        g_state.total_time_ms += resp->time_us / 1000;  // 统计用毫秒
        pthread_mutex_unlock(&g_state.stats_lock);
    } else {
        resp->status = -1;
        resp->out_size = 0;
        resp->time_us = 0;
        snprintf(resp->message, sizeof(resp->message),
                "Compression failed (or verify FAILED)");
    }

    return ret;
}

static int detect_alg_from_lzo_header(const char* input_path)
{
    FILE* f = fopen(input_path, "rb");
    if (!f) return -1;

    uint16_t magic = 0;
    uint32_t hdr[4] = {0}; /* orig_sz, blk_sz, nblk, alg_id */

    if (fread(&magic, sizeof(magic), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    if (magic != 0x4C5A) {
        fclose(f);
        return -1;
    }
    if (fread(hdr, sizeof(uint32_t), 4, f) != 4) {
        fclose(f);
        return -1;
    }

    fclose(f);
    if (hdr[3] <= 1u) return (int)hdr[3];
    return -1;
}

/*
 * 处理解压缩请求 (使用预加载的解压缩kernel)
 */
int handle_decompress_request(request_t* req, response_t* resp, worker_res_t* worker)
{
    /* If output_path is empty, generate a default one */
    if (req->output_path[0] == '\0') {
        size_t ilen = strlen(req->input_path);
        const char *suf = ".lzo";
        size_t suf_len = strlen(suf);
        if (ilen > suf_len && strcmp(req->input_path + ilen - suf_len, suf) == 0) {
            /* strip suffix */
            memcpy(req->output_path, req->input_path, ilen - suf_len);
            req->output_path[ilen - suf_len] = '\0';
        } else {
            /* append .dec */
            snprintf(req->output_path, sizeof(req->output_path), "%s.dec", req->input_path);
        }
    }

    printf("[DAEMON] 处理解压缩请求: %s -> %s\n",
           req->input_path, req->output_path);

    unsigned long time_us;
    size_t output_size;
    timing_t t = {0};

    int alg = req->alg;
    if (alg < 0 || alg > 1) {
        alg = detect_alg_from_lzo_header(req->input_path);
        if (alg < 0 || alg > 1) {
            resp->status = -1;
            snprintf(resp->message, sizeof(resp->message),
                     "Failed to detect algorithm from input header (req alg=%d)", req->alg);
            return -1;
        }
        printf("[DAEMON]    - 自动检测算法: %s (req alg=%d)\n", alg == 0 ? "lzo1x" : "lzo1y", req->alg);
    }

    cl_kernel kernel = get_decompress_kernel_for_worker(worker, alg);
    const char* alg_names[] = {"lzo1x", "lzo1y"};

    if (!kernel) {
        resp->status = -1;
        snprintf(resp->message, sizeof(resp->message), "Failed to get decompress kernel for alg=%d", alg);
        return -1;
    }

    printf("[DAEMON]    - 使用解压kernel: %s_decomp\n", alg_names[alg]);

    int ret = lzo_decompress_core(
        g_state.context,
        worker->queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path[0] == '\0' ? "/dev/null" : req->output_path,
        &worker->ws,
        req->standard_copy,
        req->local_size, 0,
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
        resp->out_size = output_size;
        resp->time_us = time_us;
        resp->timing = t;
        resp->timing.ocl_setup_us = 0;

        snprintf(resp->message, sizeof(resp->message), "OK");
        printf("[DAEMON] 解压缩成功: %zu bytes, %.2f ms\n", output_size, resp->time_us/1000.0);

        pthread_mutex_lock(&g_state.stats_lock);
        g_state.requests++;
        g_state.total_time_ms += resp->time_us / 1000;
        pthread_mutex_unlock(&g_state.stats_lock);
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

    /* 清理Worker资源 */
    for (int i = 0; i < MAX_WORKERS; i++) {
        pthread_mutex_lock(&g_state.workers[i].lock);
        lzo_gpu_workspace_free(&g_state.workers[i].ws);
        if (g_state.workers[i].queue) clReleaseCommandQueue(g_state.workers[i].queue);

        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 9; b++) {
                if (g_state.workers[i].kernels_comp[a][b]) clReleaseKernel(g_state.workers[i].kernels_comp[a][b]);
            }
            if (g_state.workers[i].kernels_decomp[a]) clReleaseKernel(g_state.workers[i].kernels_decomp[a]);
        }

        pthread_mutex_unlock(&g_state.workers[i].lock);
        pthread_mutex_destroy(&g_state.workers[i].lock);
    }

    // 清理所有全局programs
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 9; j++) {
            if (g_state.programs[i][j]) clReleaseProgram(g_state.programs[i][j]);
        }
        if (g_state.prog_decomp[i]) clReleaseProgram(g_state.prog_decomp[i]);
    }

    if (g_state.context) clReleaseContext(g_state.context);
    pthread_mutex_destroy(&g_state.compile_lock);
    pthread_mutex_destroy(&g_state.stats_lock);
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
            if ((pf == NULL) || (strcmp(proc_comm, "lzo_gpu") != 0)) {
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
 * Worker thread handles a single client connection
 */
static void* worker_thread(void* arg) {
    worker_thread_args_t* args = (worker_thread_args_t*)arg;
    int client_sock = args->client_sock;
    free(args);

    request_t req;
    response_t resp;
    memset(&resp, 0, sizeof(resp));

    // 1. Receive request
    ssize_t n = recv(client_sock, &req, sizeof(req), 0);
    if (n != sizeof(req)) {
        fprintf(stderr, "[DAEMON] Failed to receive request\n");
        close(client_sock);
        return NULL;
    }

    // 2. Find an available worker resource
    worker_res_t* worker = NULL;
    while (g_state.running) {
        for (int i = 0; i < MAX_WORKERS; i++) {
            if (pthread_mutex_trylock(&g_state.workers[i].lock) == 0) {
                if (!g_state.workers[i].in_use) {
                    g_state.workers[i].in_use = 1;
                    worker = &g_state.workers[i];
                    pthread_mutex_unlock(&g_state.workers[i].lock);
                    break;
                }
                pthread_mutex_unlock(&g_state.workers[i].lock);
            }
        }
        if (worker) break;
        struct timespec ts = {0, 1000000}; // 1ms
        nanosleep(&ts, NULL);
    }

    if (!worker) {
        resp.status = -1;
        snprintf(resp.message, sizeof(resp.message), "Daemon shutting down or no workers available");
        send(client_sock, &resp, sizeof(resp), 0);
        close(client_sock);
        return NULL;
    }

    // 3. Process request
    if (req.operation == 'C') {
        handle_compress_request(&req, &resp, worker);
    } else if (req.operation == 'D') {
        handle_decompress_request(&req, &resp, worker);
    } else {
        resp.status = -1;
        snprintf(resp.message, sizeof(resp.message), "Unknown operation");
    }

    // 4. Send response and release worker
    send(client_sock, &resp, sizeof(resp), 0);
    close(client_sock);

    pthread_mutex_lock(&worker->lock);
    worker->in_use = 0;
    pthread_mutex_unlock(&worker->lock);

    return NULL;
}

/*
 * 主服务循环
 */
void run_server(void)
{
    g_state.running = 1;

    printf("[DAEMON] 等待客户端连接 (并发限制: %d)...\n\n", MAX_WORKERS);

    while (g_state.running) {
        int client_sock = accept(g_state.server_sock, NULL, NULL);
        if (client_sock < 0) {
            if (errno == EINTR) continue;
            if (!g_state.running) break;
            perror("accept失败");
            break;
        }

        worker_thread_args_t* args = malloc(sizeof(worker_thread_args_t));
        args->client_sock = client_sock;

        pthread_t tid;
        if (pthread_create(&tid, NULL, worker_thread, args) != 0) {
            fprintf(stderr, "[DAEMON] Failed to create worker thread\n");
            close(client_sock);
            free(args);
        } else {
            pthread_detach(tid);
        }
    }
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
int run_lzo_daemon(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    /* simple command-line parsing: support -h/--help and -s/--socket */
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stdout, "LZO GPU 守护进程 Usage:\n");
            fprintf(stdout, "  %s [options]\n", argv[0]);
            fprintf(stdout, "\nOptions:\n");
            fprintf(stdout, "  -h, --help           Show this help and exit\n");
            fprintf(stdout, "  -s, --socket PATH    Custom socket path\n");
            fprintf(stdout, "  --pid PATH           Custom PID file path\n");
            fprintf(stdout, "  -v, --verbose        Enable global profiling (OpenCL events)\n");
            fprintf(stdout, "\nEnvironment variables (daemon-level defaults):\n");
            fprintf(stdout, "  LZO_DECOMP_CACHE_MB=N      Decompress buffer cache threshold in MB (daemon-level)\n");
            fprintf(stdout, "\nPer-request client options: clients may request behavior (eg. MT I/O, coalesce, stdio buf), but daemon will only honor per-request values at runtime if configured to do so. See client help.\n");
            return 0;
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--socket") == 0) {
            if (i + 1 < argc) lzo_set_daemon_socket_path(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "--pid") == 0) {
            if (i + 1 < argc) lzo_set_daemon_pidfile_path(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            g_verbose = 1;
            continue;
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
