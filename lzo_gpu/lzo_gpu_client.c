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
        printf("  压缩大小: %ld bytes (%.2f MB)\n", req.input_size, req.input_size / 1048576.0);
        printf("  原始大小: %ld bytes (%.2f MB)\n", resp.output_size, resp.output_size / 1048576.0);
        printf("  扩展比:   %.4f:1\n", (double)resp.output_size / req.input_size);
        printf("  耗时:     %.3f ms\n", resp.time_us / 1000.0);
        printf("  吞吐量:   %.2f MB/s\n", (resp.output_size / 1048576.0) / (resp.time_us / 1000000.0));
        printf("  %s\n", resp.message);
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
        if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            operation = 'D';
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--level") == 0) {
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
        fprintf(stderr, "  -l, --level <1|1k|1l|1o>  压缩级别 (默认: 1)\n");
        fprintf(stderr, "                            1  = lzo1x_1  (标准, D_BITS=14)\n");
        fprintf(stderr, "                            1k = lzo1x_1k (紧凑, D_BITS=11)\n");
        fprintf(stderr, "                            1l = lzo1x_1l (轻量, D_BITS=12)\n");
        fprintf(stderr, "                            1o = lzo1x_1o (最优, D_BITS=15)\n");
        fprintf(stderr, "  -d, --decompress          解压缩模式\n");
        fprintf(stderr, "\n示例:\n");
        fprintf(stderr, "  %s input.txt output.lzo           # 使用level=1压缩\n", argv[0]);
        fprintf(stderr, "  %s -l 1k input.txt output.lzo     # 使用lzo1x_1k压缩\n", argv[0]);
        fprintf(stderr, "  %s -d input.lzo output.txt        # 解压缩\n", argv[0]);
        return 1;
    }

    if (operation == 'D') {
        return decompress_with_daemon(input, output);
    }

    return compress_with_daemon(input, output, level);
}
