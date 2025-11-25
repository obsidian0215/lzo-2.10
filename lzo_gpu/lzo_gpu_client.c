/*
 * lzo_gpu_client.c - LZO GPU守护进程客户端
 *
 * 功能: 通过Unix socket向守护进程发送压缩请求
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"

typedef struct {
    char operation;
    char input_path[256];
    char output_path[256];
    int level;
    size_t input_size;
    /* options passed to daemon (populated from env vars) */
    int standard_copy; /* 0/1 */
    int mt_io;        /* 0/1 */
    int mt_threads;   /* number of IO threads */
    int fixed_block_kb; /* 0=no fixed size, else KB */
    int force_map;    /* 0/1 */
    int prefer_cpu;   /* 0=gpu, 1=cpu */
    int decomp_vec;   /* 0=scalar decomp, 1=vector (default) */
} request_t;

typedef struct {
    int status;
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

/* helper: read client-side environment flags into request */
static void fill_request_env_flags(request_t* req) {
    const char* s;
    s = getenv("LZO_STANDARD_COPY"); req->standard_copy = (s && atoi(s) == 1) ? 1 : 0;
    s = getenv("LZO_MT_IO"); req->mt_io = (s && atoi(s) == 1) ? 1 : 0;
    s = getenv("LZO_MT_IO_THREADS"); req->mt_threads = s ? atoi(s) : 4;
    if (req->mt_threads < 1) req->mt_threads = 1;
    if (req->mt_threads > 32) req->mt_threads = 32;
    s = getenv("LZO_FIXED_BLOCK_SIZE"); req->fixed_block_kb = s ? atoi(s) : 0; /* KB */
    s = getenv("LZO_FORCE_MAP"); req->force_map = (s && atoi(s) == 1) ? 1 : 0;
    s = getenv("LZO_OPENCL_DEVICE"); req->prefer_cpu = (s && strcmp(s, "CPU") == 0) ? 1 : 0;
    s = getenv("LZO_DECOMP_VEC"); req->decomp_vec = (s && atoi(s) == 0) ? 0 : 1; /* default 1 */
}

/*
 * 检查守护进程是否运行
 */
int is_daemon_running(void)
{
    return access(SOCKET_PATH, F_OK) == 0;
}

/*
 * 向守护进程发送解压缩请求
 */
int decompress_with_daemon(const char* input, const char* output)
{
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    struct stat st;

    // 检查守护进程
    if (!is_daemon_running()) {
        fprintf(stderr, "错误: 守护进程未运行\n");
        fprintf(stderr, "请先启动: ./lzo_gpu_daemon\n");
        return -1;
    }

    // 获取文件大小
    if (stat(input, &st) != 0) {
        perror("无法获取文件信息");
        return -1;
    }

    // 创建socket
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket创建失败");
        return -1;
    }

    // 连接守护进程
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        return -1;
    }

    // 构造解压缩请求
    memset(&req, 0, sizeof(req));
    req.operation = 'D';
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.level = 0;  // 解压缩不需要level
    req.input_size = st.st_size;

    /* fill options from env */
    fill_request_env_flags(&req);

    // 发送请求
    if (send(sock, &req, sizeof(req), 0) != sizeof(req)) {
        perror("发送请求失败");
        close(sock);
        return -1;
    }

    // 接收响应
    if (recv(sock, &resp, sizeof(resp), 0) != sizeof(resp)) {
        perror("接收响应失败");
        close(sock);
        return -1;
    }

    close(sock);

    // 处理响应
    if (resp.status == 0) {
        printf("解压缩成功: %s -> %s\n", input, output);
        /*
        printf("  压缩大小: %ld bytes (%.2f MB)\n", req.input_size, req.input_size / 1048576.0);
        printf("  原始大小: %ld bytes (%.2f MB)\n", resp.output_size, resp.output_size / 1048576.0);
        printf("  扩展比:   %.4f:1\n", (double)resp.output_size / req.input_size);
        printf("  耗时:     %.3f ms\n", resp.time_us / 1000.0);
        printf("  吞吐量:   %.2f MB/s\n", (resp.output_size / 1048576.0) / (resp.time_us / 1000000.0));
        printf("  %s\n", resp.message);
        */
        return 0;
    } else {
        fprintf(stderr, "解压缩失败: %s\n", resp.message);
        return -1;
    }
}

/*
 * 向守护进程发送压缩请求
 */
int compress_with_daemon(const char* input, const char* output, int level)
{
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    struct stat st;

    // 检查守护进程
    if (!is_daemon_running()) {
        fprintf(stderr, "错误: 守护进程未运行\n");
        fprintf(stderr, "请先启动: ./lzo_gpu_daemon\n");
        return -1;
    }

    // 获取文件大小
    if (stat(input, &st) != 0) {
        perror("无法获取文件信息");
        return -1;
    }

    // 创建socket
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket创建失败");
        return -1;
    }

    // 连接守护进程
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        return -1;
    }

    // 构造请求
    memset(&req, 0, sizeof(req));
    req.operation = 'C';
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.level = level;
    req.input_size = st.st_size;

    /* fill options from env */
    fill_request_env_flags(&req);

    // 发送请求
    if (send(sock, &req, sizeof(req), 0) != sizeof(req)) {
        perror("发送请求失败");
        close(sock);
        return -1;
    }

    // 接收响应
    if (recv(sock, &resp, sizeof(resp), 0) != sizeof(resp)) {
        perror("接收响应失败");
        close(sock);
        return -1;
    }

    close(sock);

    // 处理响应
    if (resp.status == 0) {
        printf("压缩成功: %s -> %s\n", input, output);
        /*
        printf("  原始大小: %ld bytes (%.2f MB)\n", req.input_size, req.input_size / 1048576.0);
        printf("  压缩大小: %ld bytes (%.2f MB)\n", resp.output_size, resp.output_size / 1048576.0);
        printf("  压缩比:   %.4f:1 (节省 %.2f%%)\n",
               (double)req.input_size / resp.output_size,
               (1.0 - (double)resp.output_size / req.input_size) * 100);
        printf("  总耗时:   %.3f ms (%.2f MB/s)\n",
               resp.time_us / 1000.0,
               (req.input_size / 1048576.0) / (resp.time_us / 1000000.0));
        printf("  [时间分解] 读文件=%.2fms, 缓冲区=%.2fms, 上传=%.2fms, Kernel=%.2fms, 下载=%.2fms, 写文件=%.2fms, 清理=%.2fms\n",
               resp.read_us/1000.0, resp.buffer_us/1000.0, resp.upload_us/1000.0, resp.kernel_us/1000.0,
               resp.download_us/1000.0, resp.write_us/1000.0, resp.cleanup_us/1000.0);
        printf("  %s\n", resp.message);
        */
        return 0;
    } else {
        fprintf(stderr, "压缩失败: %s\n", resp.message);
        return -1;
    }
}

/*
 * 客户端主函数
 * 支持: -l/--level <1|1k|1l|1o> 压缩级别
 *       -d/--decompress 解压缩模式
 *
 * 默认level=1 (lzo1x_1): 基于性能测试,所有变体速度相近,
 * 选择D_BITS=14的标准版本以获得最佳通用性
 */
int main(int argc, char** argv)
{
    const char* input = NULL;
    const char* output = NULL;
    int level = 1;  // 默认: lzo1x_1 (D_BITS=14, 标准配置)
    char operation = 'C';  // 默认压缩

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            input = NULL; output = NULL; // force help print below
            break;
        }
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            operation = 'D';
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--level") == 0) {
            if (i + 1 < argc) {
                i++;
                // 支持 1/1k/1l/1o 格式
                if (strcmp(argv[i], "1") == 0) {
                    level = 1;
                } else if (strcmp(argv[i], "1k") == 0) {
                    level = 5;  // 映射到1k (level 4-6)
                } else if (strcmp(argv[i], "1l") == 0) {
                    level = 7;  // 映射到1l (level 7-8)
                } else if (strcmp(argv[i], "1o") == 0) {
                    level = 9;  // 映射到1o (level 9)
                } else {
                    // 也支持数字1-9
                    level = atoi(argv[i]);
                    if (level < 1 || level > 9) {
                        fprintf(stderr, "错误: level必须是 1/1k/1l/1o 或 1-9\n");
                        return 1;
                    }
                }
            } else {
                fprintf(stderr, "错误: -l/--level 需要参数\n");
                return 1;
            }
        } else if (!input) {
            input = argv[i];
        } else if (!output) {
            output = argv[i];
        } else {
            fprintf(stderr, "错误: 多余的参数 '%s'\n", argv[i]);
            return 1;
        }
    }

    // 检查必需参数
    if (!input || !output) {
        fprintf(stderr, "用法: %s [选项] <input> <output>\n", argv[0]);
        fprintf(stderr, "选项:\n");
        fprintf(stderr, "  -L, -l, --level <1|1k|1l|1o>  压缩级别 (默认: 1l)\n");
        fprintf(stderr, "  -d, --decompress          解压缩模式\n");
        fprintf(stderr, "  -h, --help                Show this help\n\n");

        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  %s input.txt output.lzo           # 使用level=1l压缩\n", argv[0]);
        fprintf(stderr, "  %s -l 1k input.txt output.lzo     # 使用lzo1x_1k压缩\n", argv[0]);
        fprintf(stderr, "  %s -d input.lzo output.txt        # 解压缩\n\n", argv[0]);

        fprintf(stderr, "Client / per-request options (via environment variables):\n");
        fprintf(stderr, "  LZO_STANDARD_COPY=0|1    Request daemon to use standard host->device copy (default: 0=zero-copy)\n");
        fprintf(stderr, "  LZO_FORCE_MAP=0|1        Force the daemon to use mapped pinned buffers (overrides LZO_STANDARD_COPY)\n");
        fprintf(stderr, "  LZO_MT_IO=0|1            Enable multi-threaded I/O for reads/uploads (default: 0)\n");
        fprintf(stderr, "  LZO_MT_IO_THREADS=N      Number of I/O threads (1-32; default: 4)\n");
        fprintf(stderr, "  LZO_FIXED_BLOCK_SIZE=N   Request fixed block size (KB) for daemon (0 = adaptive)\n");
        fprintf(stderr, "  LZO_OPENCL_DEVICE=CPU|GPU Select device preference for daemon (daemon may ignore)\n");
        fprintf(stderr, "  LZO_DECOMP_VEC=0|1       Prefer vectorized decompressor (daemon-side; default:1)\n\n");

        fprintf(stderr, "Notes:\n");
        fprintf(stderr, "  - The client reads these environment variables and sends them per-request to the daemon.\n");
        fprintf(stderr, "  - The daemon may choose to accept or ignore certain preferences (device choice, etc.) depending on its configuration.\n\n");

        return 1;
    }

    if (operation == 'D') {
        return decompress_with_daemon(input, output);
    }

    return compress_with_daemon(input, output, level);
}
