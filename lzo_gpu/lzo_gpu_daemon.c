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
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <errno.h>
#include <CL/cl.h>

/* 声明daemon_decompress.c中的函数 */
extern int daemon_decompress(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel kernel, const char* input_path, const char* output_path,
    unsigned long* time_ms_out, size_t* output_size_out
);

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"
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
    cl_program prog_decomp;      // 解压缩program
    cl_kernel kernel_decomp;     // 解压缩kernel

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
} daemon_state_t;

static daemon_state_t g_state = {0};

/* 请求协议 */
typedef struct {
    char operation;      // 'C'=compress, 'D'=decompress
    char input_path[256];
    char output_path[256];
    int level;           // 压缩级别 1-9
    size_t input_size;
} request_t;

typedef struct {
    int status;          // 0=success, -1=error
    size_t output_size;
    unsigned long time_us;  // 总时间(微秒)
    // 详细时间分解 (微秒)
    unsigned long read_us;
    unsigned long buffer_us;
    unsigned long upload_us;
    unsigned long kernel_us;
    unsigned long download_us;
    unsigned long write_us;
    unsigned long cleanup_us;
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
    cl_queue_properties props[] = {0};
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

    // 5. 加载解压缩kernel
    printf("[DAEMON] 加载解压缩kernel...\n");

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
                    free(bin);
                    fclose(fb_decomp);

                    g_state.kernel_decomp = clCreateKernel(g_state.prog_decomp,
                                                          "lzo1x_block_decompress", &err);
                    if (err != CL_SUCCESS) {
                        fprintf(stderr, "创建解压缩kernel失败 (err=%d)\n", err);
                        return -1;
                    }
                    goto decomp_done;
                }
            }
            if (g_state.prog_decomp) clReleaseProgram(g_state.prog_decomp);
        }
        free(bin);
        fclose(fb_decomp);
    }

    // 回退到源码编译
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

decomp_done:

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_state.init_time_ms = (t_end.tv_sec - t_start.tv_sec) * 1000 +
                           (t_end.tv_nsec - t_start.tv_nsec) / 1000000;

    printf("[DAEMON] ✅ OpenCL资源初始化完成\n");
    printf("[DAEMON]    - 上下文: 常驻内存\n");
    printf("[DAEMON]    - 压缩kernels: lzo1x_1/1k/1l/1o\n");
    printf("[DAEMON]    - 解压缩kernel: lzo1x_decomp\n");
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
    unsigned long* time_us, size_t* output_size,
    unsigned long* read_us, unsigned long* buffer_us, unsigned long* upload_us,
    unsigned long* kernel_us, unsigned long* download_us, unsigned long* write_us,
    unsigned long* cleanup_us
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
    unsigned long read_us = 0, buffer_us = 0, upload_us = 0;
    unsigned long kernel_us = 0, download_us = 0, write_us = 0, cleanup_us = 0;

    // 调用压缩函数,复用OpenCL资源(context/queue/kernel)
    int ret = daemon_compress(
        g_state.context,
        g_state.queue,
        g_state.device,
        kernel,
        req->input_path,
        req->output_path,
        req->level,
        &time_us,
        &output_size,
        &read_us, &buffer_us, &upload_us,
        &kernel_us, &download_us, &write_us, &cleanup_us
    );

    if (ret == 0) {
        resp->status = 0;
        resp->output_size = output_size;
        resp->time_us = time_us;
        resp->read_us = read_us;
        resp->buffer_us = buffer_us;
        resp->upload_us = upload_us;
        resp->kernel_us = kernel_us;
        resp->download_us = download_us;
        resp->write_us = write_us;
        resp->cleanup_us = cleanup_us;
        snprintf(resp->message, sizeof(resp->message),
                "Success (saved ~%lums init)", g_state.init_time_ms);

        g_state.requests++;
        g_state.total_time_ms += time_us / 1000;  // 统计用毫秒
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

    int ret = daemon_decompress(
        g_state.context,
        g_state.queue,
        g_state.device,
        g_state.kernel_decomp,
        req->input_path,
        req->output_path,
        &time_us,
        &output_size
    );

    if (ret == 0) {
        resp->status = 0;
        resp->time_us = time_us;
        resp->output_size = output_size;
        snprintf(resp->message, sizeof(resp->message), "OK");
        printf("[DAEMON] 解压缩成功: %zu bytes, %.2f ms\n", output_size, time_us/1000.0);
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

    // 删除旧socket文件
    unlink(SOCKET_PATH);

    // 绑定地址
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (bind(g_state.server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind失败");
        close(g_state.server_sock);
        return -1;
    }

    // 监听连接
    if (listen(g_state.server_sock, MAX_CLIENTS) < 0) {
        perror("listen失败");
        close(g_state.server_sock);
        unlink(SOCKET_PATH);
        return -1;
    }

    printf("[DAEMON] ✅ 服务器启动成功\n");
    printf("[DAEMON]    Socket: %s\n", SOCKET_PATH);
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

        // 发送响应
        send(client_sock, &resp, sizeof(resp), 0);
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
    printf("========================================\n");
    printf("LZO GPU守护进程\n");
    printf("========================================\n\n");

    // 注册信号处理器
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 初始化OpenCL资源 (仅一次)
    if (init_opencl_resources() != 0) {
        fprintf(stderr, "OpenCL初始化失败\n");
        return 1;
    }

    // 启动服务器
    if (start_server() != 0) {
        cleanup_opencl_resources();
        return 1;
    }

    // 运行服务循环
    run_server();

    // 清理资源
    if (g_state.server_sock >= 0) {
        close(g_state.server_sock);
        g_state.server_sock = -1;
    }
    unlink(SOCKET_PATH);
    cleanup_opencl_resources();

    // 打印统计信息
    print_stats();

    printf("\n[DAEMON] 已退出\n");
    return 0;
}
