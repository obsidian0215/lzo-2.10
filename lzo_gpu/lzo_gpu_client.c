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
#include "timing.h"
#include "lzo_defaults.h"
#include <getopt.h>
#include <limits.h>

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"

/* Determine socket path by checking LZO_DAEMON_SOCKET or OUT_DIR env vars,
 * falling back to default SOCKET_PATH. This mirrors lzo_gpu_daemon's behavior so
 * clients can connect to repo-local sockets created by the daemon when OUT_DIR
 * is set (e.g., during experiments).
 */
static const char* client_socket_path(void)
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

/* request_t/response_t - local protocol structs (mirror daemon's definitions) */
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
    int async_upload; /* 0/1 */
    int fixed_block_kb; /* 0=no fixed size, else KB */
    int prefer_cpu;   /* 0=gpu, 1=cpu */
    int decomp_vec;   /* 0=scalar decomp, 1=vector (default) */
    /* Optional overrides for coalescing and stdio buffer (per-request) */
    int coalesce_output; /* -1 unspecified; 0=no coalesce; 1=coalesce */
    int coalesce_chunk_mb; /* -1 unspecified; positive value overrides default chunk size */
    int coalesce_max_mb;   /* -1 unspecified; positive value overrides default max MB to coalesce in one write */
    int stdio_buf_mb;      /* -1 unspecified; positive value overrides stdio buffer size in MB */
} request_t;

typedef struct {
    int status;
    size_t output_size;
    unsigned long time_us;  // 总时间 (微秒)
    timing_t timing;
    char message[128];
} response_t;

/* helper: read client-side environment flags into request */
/* no default struct - the client simply fills environment-driven options */

/* Fill values for unset fields (-1) from environment variables.
 * Only set a field if it is currently -1 (unspecified by CLI).
 */
static void fill_request_env_flags(request_t* req) {
    const char* s;
    /* read essential env vars to populate per-request options. These are
     * the only environment variables preserved to reduce env complexity
     * while keeping important runtime controls.
     */
    s = getenv("LZO_STANDARD_COPY"); req->standard_copy = (s && atoi(s) == 1) ? 1 : 0;
    s = getenv("LZO_MT_IO"); req->mt_io = (s && atoi(s) == 1) ? 1 : 0;
    s = getenv("LZO_MT_IO_THREADS"); req->mt_threads = s ? atoi(s) : LZO_DEFAULT_MT_IO_THREADS;
    if (req->mt_threads < 1) req->mt_threads = 1; if (req->mt_threads > 32) req->mt_threads = 32;
    s = getenv("LZO_FIXED_BLOCK_SIZE"); req->fixed_block_kb = s ? atoi(s) : 0;
    /* LZO_FORCE_MAP has been deprecated; mapping behavior is controlled via LZO_STANDARD_COPY */
    s = getenv("LZO_OPENCL_DEVICE"); req->prefer_cpu = (s && strcmp(s, "CPU") == 0) ? 1 : 0;
    s = getenv("LZO_DECOMP_VEC"); req->decomp_vec = (s && atoi(s) == 0) ? 0 : 1;
    s = getenv("LZO_ASYNC_UPLOAD"); req->async_upload = (s && atoi(s) == 1) ? 1 : 0;
    /* coalesce/stdio flags: support env overrides only if CLI didn't set them. */
    if (req->coalesce_output == -1) { s = getenv("LZO_COALESCE_OUTPUT"); if (s) req->coalesce_output = atoi(s) ? 1 : 0; }
    if (req->coalesce_chunk_mb == -1) { s = getenv("LZO_COALESCE_CHUNK_MB"); if (s) req->coalesce_chunk_mb = atoi(s); }
    if (req->coalesce_max_mb == -1) { s = getenv("LZO_COALESCE_MAX_MB"); if (s) req->coalesce_max_mb = atoi(s); }
    if (req->stdio_buf_mb == -1) { s = getenv("LZO_STDIO_BUF_MB"); if (s) req->stdio_buf_mb = atoi(s); }
    /* profile_writes removed; use LZO_DEBUG for diagnostics */
}

/* Simple CLI parsing for per-request flags. These CLI flags override env vars. */
/* Note: coalescing/stdio buffer control is moved to defaults (see lzo_defaults.h)
 * and is not configured via environment variables to reduce global env complexity.
 */
static request_t g_cli_req_flags; /* global overrides parsed from CLI */
static void parse_client_cli_opts(int argc, char** argv, request_t* req) {
    /* Manual, minimal long-option parsing to read per-request flags that can be provided
     * via --option=value or --option value. This avoids interacting with getopt_long
     * and consuming/printing short-option errors for -l / -o, which the main parser
     * handles.
     */
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strncmp(a, "--mt-io=", 8) == 0) { req->mt_io = atoi(a + 8) ? 1 : 0; continue; }
        if (strcmp(a, "--mt-io") == 0 && i + 1 < argc) { req->mt_io = atoi(argv[++i]) ? 1 : 0; continue; }
        if (strncmp(a, "--mt-io-threads=", 16) == 0) { req->mt_threads = atoi(a + 16); continue; }
        if (strcmp(a, "--mt-io-threads") == 0 && i + 1 < argc) { req->mt_threads = atoi(argv[++i]); continue; }
        if (strncmp(a, "--standard-copy=", 16) == 0) { req->standard_copy = atoi(a + 16) ? 1 : 0; continue; }
        if (strcmp(a, "--standard-copy") == 0 && i + 1 < argc) { req->standard_copy = atoi(argv[++i]) ? 1 : 0; continue; }
        if (strncmp(a, "--async-upload=", 15) == 0) { req->async_upload = atoi(a + 15) ? 1 : 0; continue; }
        if (strcmp(a, "--async-upload") == 0 && i + 1 < argc) { req->async_upload = atoi(argv[++i]) ? 1 : 0; continue; }
        if (strncmp(a, "--fixed-block-size=", 19) == 0) { req->fixed_block_kb = atoi(a + 19); continue; }
        if (strcmp(a, "--fixed-block-size") == 0 && i + 1 < argc) { req->fixed_block_kb = atoi(argv[++i]); continue; }
        if (strncmp(a, "--prefer-cpu=", 13) == 0) { req->prefer_cpu = atoi(a + 13) ? 1 : 0; continue; }
        if (strcmp(a, "--prefer-cpu") == 0 && i + 1 < argc) { req->prefer_cpu = atoi(argv[++i]) ? 1 : 0; continue; }
        if (strncmp(a, "--decomp-vec=", 13) == 0) { req->decomp_vec = atoi(a + 13) ? 1 : 0; continue; }
        if (strcmp(a, "--decomp-vec") == 0 && i + 1 < argc) { req->decomp_vec = atoi(argv[++i]) ? 1 : 0; continue; }
        /* ignore other arguments; main() will parse short options and operands */
    }
}

/* set default request options */
static void set_request_defaults(request_t* req) {
    req->standard_copy = 0;
    req->mt_io = 0;
    req->mt_threads = LZO_DEFAULT_MT_IO_THREADS;
    req->async_upload = 0;
    req->fixed_block_kb = 0;
    /* force_map option removed; no default to set */
    req->prefer_cpu = 0;
    req->decomp_vec = 1;
    /* per-request coalesce/stdio defaults: -1 means unspecified so daemon uses defaults */
    req->coalesce_output = -1;
    req->coalesce_chunk_mb = -1;
    req->coalesce_max_mb = -1;
    req->stdio_buf_mb = -1;
    /* profile_writes removed */
}

/*
 * 检查守护进程是否运行
 */
int is_daemon_running(void)
{
    return access(client_socket_path(), F_OK) == 0;
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
    strncpy(addr.sun_path, client_socket_path(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        return -1;
    }

    // 构造解压缩请求
    memset(&req, 0, sizeof(req));
    set_request_defaults(&req);
    /* inject CLI-specified overrides */
    memcpy(&req, &g_cli_req_flags, sizeof(req));
    req.operation = 'D';
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.level = 0;  // 解压缩不需要level
    req.input_size = st.st_size;

    /* fill options from env for fields still unspecified */
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
        /* Decompression path: output format matching lzo_gpu standalone */
        double orig_mb = resp.output_size / (1024.0 * 1024.0);
        double comp_mb = req.input_size / (1024.0 * 1024.0);
        double total_ms = resp.time_us / 1000.0;
        double kernel_ms = resp.timing.kernel_exec_us / 1000.0;
        double throughput = total_ms > 0 ? orig_mb / (total_ms / 1000.0) : 0;
        double kernel_throughput = kernel_ms > 0 ? orig_mb / (kernel_ms / 1000.0) : 0;

        fprintf(stderr, "\n=== Decompression Statistics ===\n");
        fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", (size_t)req.input_size, comp_mb);
        fprintf(stderr, "Output size      : %zu bytes (%.2f MB)\n", (size_t)resp.output_size, orig_mb);
        fprintf(stderr, "Expansion ratio  : %.2f:1\n", req.input_size > 0 ? (double)resp.output_size / req.input_size : 0);
        fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);
        fprintf(stderr, "==============================\n\n");

        fprintf(stderr, "=== Timing Breakdown ===\n");
        fprintf(stderr, "1. File Read       : %.2f ms (%.1f%%)\n",
                resp.timing.file_read_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.file_read_us / resp.time_us : 0);
        fprintf(stderr, "2. Buffer Alloc    : %.2f ms (%.1f%%)\n",
                resp.timing.buffer_alloc_in_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.buffer_alloc_in_us / resp.time_us : 0);
        fprintf(stderr, "3. Data Upload     : %.2f ms (%.1f%%)\n",
                resp.timing.data_upload_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.data_upload_us / resp.time_us : 0);
        fprintf(stderr, "4. Setup Args      : %.2f ms (%.1f%%)\n",
                resp.timing.setup_args_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.setup_args_us / resp.time_us : 0);
        fprintf(stderr, "5. Kernel Exec     : %.2f ms (%.1f%%)\n",
                resp.timing.kernel_exec_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.kernel_exec_us / resp.time_us : 0);
        fprintf(stderr, "6. Data Download   : %.2f ms (%.1f%%)\n",
                resp.timing.download_total_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.download_total_us / resp.time_us : 0);
        fprintf(stderr, "7. File Write      : %.2f ms (%.1f%%)\n",
                resp.timing.file_write_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.file_write_us / resp.time_us : 0);
        fprintf(stderr, "------------------------\n");
        fprintf(stderr, "TOTAL              : %.2f ms\n", total_ms);
        fprintf(stderr, "Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);

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
    strncpy(addr.sun_path, client_socket_path(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        return -1;
    }

    // 构造请求
    memset(&req, 0, sizeof(req));
    set_request_defaults(&req);
    memcpy(&req, &g_cli_req_flags, sizeof(req));
    req.operation = 'C';
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.level = level;
    req.input_size = st.st_size;

    /* fill options from env for fields still unspecified */
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
        /* Compression path: output format matching lzo_gpu standalone */
        double orig_mb = req.input_size / (1024.0 * 1024.0);
        double comp_mb = resp.output_size / (1024.0 * 1024.0);
        double total_ms = resp.time_us / 1000.0;
        double kernel_ms = resp.timing.kernel_exec_us / 1000.0;
        double throughput = total_ms > 0 ? orig_mb / (total_ms / 1000.0) : 0;
        double kernel_throughput = kernel_ms > 0 ? orig_mb / (kernel_ms / 1000.0) : 0;
        double ratio = resp.output_size > 0 ? (double)req.input_size / resp.output_size : 0;
        double ratio_pct = req.input_size > 0 ? 100.0 * resp.output_size / req.input_size : 0;

        fprintf(stderr, "\n=== Compression Statistics ===\n");
        fprintf(stderr, "Input size       : %zu bytes (%.2f MB)\n", (size_t)req.input_size, orig_mb);
        fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", (size_t)resp.output_size, comp_mb);
        fprintf(stderr, "Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, ratio_pct);
        fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);
        fprintf(stderr, "==============================\n\n");

        fprintf(stderr, "=== Timing Breakdown ===\n");
        fprintf(stderr, "1. File Read       : %.2f ms (%.1f%%)\n",
                resp.timing.file_read_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.file_read_us / resp.time_us : 0);
        fprintf(stderr, "2. Blocking Calc   : %.2f ms (%.1f%%)\n",
                resp.timing.blocking_calc_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.blocking_calc_us / resp.time_us : 0);
        fprintf(stderr, "3. Buffer Alloc    : %.2f ms (%.1f%%)\n",
                resp.timing.buffer_alloc_in_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.buffer_alloc_in_us / resp.time_us : 0);
        fprintf(stderr, "4. Data Upload     : %.2f ms (%.1f%%)\n",
                resp.timing.data_upload_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.data_upload_us / resp.time_us : 0);
        fprintf(stderr, "5. Kernel Exec     : %.2f ms (%.1f%%)\n",
                resp.timing.kernel_exec_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.kernel_exec_us / resp.time_us : 0);
        fprintf(stderr, "6. Data Download   : %.2f ms (%.1f%%)\n",
                resp.timing.download_total_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.download_total_us / resp.time_us : 0);
        fprintf(stderr, "7. File Write      : %.2f ms (%.1f%%)\n",
                resp.timing.file_write_us / 1000.0,
                resp.time_us > 0 ? 100.0 * resp.timing.file_write_us / resp.time_us : 0);
        fprintf(stderr, "------------------------\n");
        fprintf(stderr, "TOTAL              : %.2f ms\n", total_ms);
        fprintf(stderr, "Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);

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
static void show_help(const char* prog) {
    fprintf(stderr, "LZO GPU Client - 通过守护进程进行 GPU 加速压缩/解压\n\n");
    fprintf(stderr, "用法: %s [选项] <input>\n", prog);
    fprintf(stderr, "选项:\n");
    fprintf(stderr, "  -L, -l, --level <1|1k|1l|1o>  压缩级别 (默认: 1l)\n");
    fprintf(stderr, "  -d, --decompress              解压缩模式\n");
    fprintf(stderr, "  -o, --output <file>           指定输出文件\n");
    fprintf(stderr, "  -h, --help                    显示此帮助信息\n\n");
    fprintf(stderr, "示例:\n");
    fprintf(stderr, "  %s input.txt -o output.lzo           # 使用 level=1l 压缩\n", prog);
    fprintf(stderr, "  %s -l 1k input.txt -o output.lzo     # 使用 lzo1x_1k 压缩\n", prog);
    fprintf(stderr, "  %s -d input.lzo -o output.txt        # 解压缩\n", prog);
    fprintf(stderr, "  %s -d input.lzo                      # 解压缩 (自动去除 .lzo 后缀)\n\n", prog);
    fprintf(stderr, "环境变量 (客户端请求选项):\n");
    fprintf(stderr, "  LZO_STANDARD_COPY=0|1    使用标准 host->device 拷贝 (默认: 0=zero-copy)\n");
    fprintf(stderr, "  LZO_MT_IO=0|1            启用多线程 I/O (默认: 0)\n");
    fprintf(stderr, "  LZO_MT_IO_THREADS=N      I/O 线程数 (1-32; 默认: %d)\n", LZO_DEFAULT_MT_IO_THREADS);
    fprintf(stderr, "  LZO_FIXED_BLOCK_SIZE=N   固定块大小 (KB) (0=自适应)\n");
    fprintf(stderr, "  LZO_DECOMP_VEC=0|1       使用向量化解压 (默认: 1)\n");
    fprintf(stderr, "\n注意: 需要先启动 lzo_gpu_daemon 守护进程\n");
}

int main(int argc, char** argv)
{
    const char* input = NULL;
    const char* output = NULL;
    int level = 7;  // 默认: lzo1x_1 (D_BITS=14, 标准配置)
    char operation = 'C';  // 默认压缩
    int show_help_flag = 0;

    // Parse client-level CLI options (per-request overrides) first
    set_request_defaults(&g_cli_req_flags);
    parse_client_cli_opts(argc, argv, &g_cli_req_flags);

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help_flag = 1;
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
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) { output = argv[++i]; } else { fprintf(stderr, "错误: -o/--output 需要参数\n"); return 1; }
        } else if (!input) {
            input = argv[i];
        } else {
            fprintf(stderr, "错误: 仅支持一个位置参数作为 <input>，不要使用位置形式指定输出，请改用 -o/--output\n");
            return 1;
        }
    }

    // 显示帮助
    if (show_help_flag || argc < 2) {
        show_help(argv[0]);
        return show_help_flag ? 0 : 1;
    }

    /* If output not provided, compute sensible defaults:
     * - For compression: input -> input + ".lzo"
     * - For decompression: if input ends with .lzo, strip it; else append ".dec"
     */
    if (!input) {
        fprintf(stderr, "错误: 未指定输入文件\n");
        show_help(argv[0]);
        return 1;
    }

    if (!output) {
        size_t ilen = strlen(input);
        if (operation == 'C') {
            size_t n = ilen + 5 + 1; /* ".lzo" + NUL */
            char *tmp = malloc(n);
            if (!tmp) { perror("malloc"); return 1; }
            snprintf(tmp, n, "%s.lzo", input);
            output = tmp; /* intentional leak until process exit */
        } else { /* decompress default */
            const char *suf = ".lzo";
            size_t suf_len = strlen(suf);
            if (ilen > suf_len && strcmp(input + ilen - suf_len, suf) == 0) {
                /* strip suffix */
                size_t n = ilen - suf_len + 1;
                char *tmp = malloc(n);
                if (!tmp) { perror("malloc"); return 1; }
                memcpy(tmp, input, ilen - suf_len);
                tmp[ilen - suf_len] = '\0';
                output = tmp;
            } else {
                /* append .dec */
                size_t n = ilen + 4 + 1; /* ".dec" + NUL */
                char *tmp = malloc(n);
                if (!tmp) { perror("malloc"); return 1; }
                snprintf(tmp, n, "%s.dec", input);
                output = tmp;
            }
        }
    }

    /* If the operation is decompression, go to decompress path */
    if (operation == 'D') {
        return decompress_with_daemon(input, output);
    }

    return compress_with_daemon(input, output, level);
}
