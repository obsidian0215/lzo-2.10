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
#include "lzo_gpu_utils.h"
#include <getopt.h>
#include <limits.h>
#include <stddef.h>

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

/* parse human-friendly size string and return KB (rounded) */
static int parse_size_kb(const char* s)
{
    if (!s || !*s) return 0;
    char* end;
    double val = strtod(s, &end);
    if (end == s) return 0;
    while (*end == ' ' || *end == '\t') end++;
    if (*end == '\0') {
        /* no unit - assume KB */
        return (int) (val);
    }
    if (strcasecmp(end, "B") == 0 || strcasecmp(end, "BYTES") == 0) {
        return (int) ((val + 1023) / 1024); /* round up to KB */
    }
    if (strcasecmp(end, "K") == 0 || strcasecmp(end, "KB") == 0) {
        return (int) (val);
    }
    if (strcasecmp(end, "M") == 0 || strcasecmp(end, "MB") == 0) {
        return (int) (val * 1024);
    }
    if (strcasecmp(end, "G") == 0 || strcasecmp(end, "GB") == 0) {
        return (int) (val * 1024 * 1024);
    }
    return 0;
}

/* request_t/response_t - local protocol structs (mirror daemon's definitions) */
typedef struct {
    char operation;
    char input_path[256];
    char output_path[256];
    int level;
    int alg;
    size_t input_size;
    /* options passed to daemon (populated from env vars) */
    int standard_copy; /* 0/1 */
    int mt_io;        /* 0/1 */
    int mt_threads;   /* number of IO threads */
    int fixed_block_kb; /* 0=no fixed size, else KB */
    int local_size;     /* 0=unspecified, else local size for kernel */
    /* Optional overrides for coalescing and stdio buffer (per-request) */
    int coalesce_output; /* -1 unspecified; 0=no coalesce; 1=coalesce */
    int coalesce_chunk_mb; /* -1 unspecified; positive value overrides default chunk size */
    int coalesce_max_mb;   /* -1 unspecified; positive value overrides default max MB to coalesce in one write */
    int stdio_buf_mb;      /* -1 unspecified; positive value overrides stdio buffer size in MB */
    /* Opt kernel flag (client -> daemon request) */
    int kernel_opt;    /* 0/1: request optimized instrumented kernel (unrolled vector) */
    /* Per-request debug flag: 0/1 (client requests kernel instrumentation and verbose diagnostics) */
    int debug;
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
    s = getenv("LZO_MT_IO_THREADS"); req->mt_threads = s ? CLAMP(atoi(s), LZO_MIN_MT_IO_THREADS, LZO_MAX_MT_IO_THREADS) : LZO_DEFAULT_MT_IO_THREADS;
    /* coalesce/stdio flags: support env overrides only if CLI didn't set them. */
    if (req->coalesce_output == -1) { s = getenv("LZO_COALESCE_OUTPUT"); if (s) req->coalesce_output = atoi(s) ? 1 : 0; }
    if (req->coalesce_chunk_mb == -1) { s = getenv("LZO_COALESCE_CHUNK_MB"); if (s) req->coalesce_chunk_mb = atoi(s); }
    if (req->coalesce_max_mb == -1) { s = getenv("LZO_COALESCE_MAX_MB"); if (s) req->coalesce_max_mb = atoi(s); }
    if (req->stdio_buf_mb == -1) { s = getenv("LZO_STDIO_BUF_MB"); if (s) req->stdio_buf_mb = atoi(s); }
    /* kernel variant env flags (optional) */
    s = getenv("LZO_KERNEL_OPT"); if (s && atoi(s) == 1) req->kernel_opt = 1; // kernel debug is not supported for daemon clients
}

/* Simple CLI parsing for per-request flags. These CLI flags override env vars. */
/* Note: coalescing/stdio buffer control is moved to defaults (see lzo_defaults.h)
 * and is not configured via environment variables to reduce global env complexity.
 */
static request_t g_cli_req_flags; /* global overrides parsed from CLI */
static void parse_client_cli_opts(int argc, char** argv, request_t* req) {
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strncmp(a, "-B=", 3) == 0) { req->fixed_block_kb = parse_size_kb(a + 3); argv[i] = "--"; continue; }
        if (strcmp(a, "-B") == 0 && i + 1 < argc) { argv[i] = "--"; req->fixed_block_kb = parse_size_kb(argv[++i]); argv[i] = "--"; continue; }
        if (strncmp(a, "--block-size=", 13) == 0) { req->fixed_block_kb = parse_size_kb(a + 13); argv[i] = "--"; continue; }
        if (strcmp(a, "--block-size") == 0 && i + 1 < argc) { argv[i] = "--"; req->fixed_block_kb = parse_size_kb(argv[++i]); argv[i] = "--"; continue; }
        if (strncmp(a, "--local=", 8) == 0) { req->local_size = atoi(a + 8); argv[i] = "--"; continue; }
        if (strcmp(a, "--local") == 0 && i + 1 < argc) { argv[i] = "--"; req->local_size = atoi(argv[++i]); argv[i] = "--"; continue; }
        if (strcmp(a, "--kernel-opt") == 0) { req->kernel_opt = 1; argv[i] = "--"; continue; }
        if (strcmp(a, "--debug") == 0) { req->debug = 1; argv[i] = "--"; continue; }
    }
}

/* set default request options */
static void set_request_defaults(request_t* req) {
    req->standard_copy = 0;
    req->mt_io = 0;
    req->mt_threads = LZO_DEFAULT_MT_IO_THREADS;
    req->fixed_block_kb = 0;
    req->local_size = 0;
    /* per-request coalesce/stdio defaults: -1 means unspecified so daemon uses defaults */
    req->coalesce_output = -1;
    req->coalesce_chunk_mb = -1;
    req->coalesce_max_mb = -1;
    req->stdio_buf_mb = -1;
    /* kernel variant flags */
    req->kernel_opt = 0;
    /* per-request debug (0/1) */
    req->debug = 0;
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
int decompress_with_daemon(const char* input, const char* output, int alg)
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
    req.alg = alg;  // 指定算法
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
        fprintf(stderr, "Block size       : %lu bytes/%lu KB\n",
            (unsigned long)resp.timing.blk_size_bytes,
            (unsigned long)(resp.timing.blk_size_bytes / 1024UL));
        fprintf(stderr, "Kernel           : %s\n",
            resp.timing.kernel_name[0] ? resp.timing.kernel_name : "unknown");
        fprintf(stderr, "Work groups      : global=%lu, local=%lu\n",
            (unsigned long)resp.timing.global_size,
            (unsigned long)resp.timing.local_size);
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
int compress_with_daemon(const char* input, const char* output, int alg, int level)
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
                    /* Diagnostic: print final request fields to stderr to confirm CLI parsing */
                    if (req.debug) {
                    fprintf(stderr, "[CLIENT] Request opts: fixed_block_kb=%d local_size=%d standard_copy=%d mt_io=%d mt_threads=%d kernel_opt=%d debug=%d sizeof(req)=%zu\n",
                        req.fixed_block_kb, req.local_size, req.standard_copy, req.mt_io, req.mt_threads, req.kernel_opt, req.debug, sizeof(req));
                    fprintf(stderr, "[CLIENT] offsetof: fixed_block_kb=%zu local_size=%zu kernel_opt=%zu debug=%zu\n",
                        offsetof(request_t, fixed_block_kb), offsetof(request_t, local_size), offsetof(request_t, kernel_opt), offsetof(request_t, debug));
                    }
    req.operation = 'C';
    strncpy(req.input_path, input, sizeof(req.input_path) - 1);
    strncpy(req.output_path, output, sizeof(req.output_path) - 1);
    req.alg = alg;
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
        fprintf(stderr, "Block size (blocks): %lu bytes/%lu KB (%lu)\n",
            (unsigned long)resp.timing.blk_size_bytes,
            (unsigned long)(resp.timing.blk_size_bytes / 1024UL),
            (unsigned long)resp.timing.nblk);
        fprintf(stderr, "Kernel           : %s\n",
            resp.timing.kernel_name[0] ? resp.timing.kernel_name : "unknown");
        fprintf(stderr, "Work groups      : global=%lu, local=%lu\n",
            (unsigned long)resp.timing.global_size,
            (unsigned long)resp.timing.local_size);
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

static void show_help(const char* prog) {
    fprintf(stderr, "LZO GPU Client - 通过守护进程进行 GPU 加速压缩/解压\n\n");
    fprintf(stderr, "用法: %s [选项] <input>\n", prog);
    fprintf(stderr, "选项:\n");
    fprintf(stderr, "  -h, --help                    Show help and exit.\n");
    fprintf(stderr, "  -L, -l, --level <1|1k|1l|1o>  Set compression level/kernel variant (default: 1l).\n");
    fprintf(stderr, "  -B N, --block-size N          Fixed block size; accepts units (B/KB/MB), e.g., -B 64KB.\n");
    fprintf(stderr, "  --local N                     Local work-group size for kernel (1, 8, 64). Compression kernels require local=1 and will be forced.\n");
    fprintf(stderr, "  -d, --decompress              Decompress mode.\n");
    fprintf(stderr, "  -o, --output <file>           Output file.\n");
    fprintf(stderr, "  --debug                       Enable debug instrumentation and verbose diagnostics (per-request).\n\n");
    fprintf(stderr, "示例:\n");
    fprintf(stderr, "  %s input.txt -o output.lzo\n", prog);
    fprintf(stderr, "  %s -l 1k input.txt -o output.lzo\n", prog);
    fprintf(stderr, "  %s -d input.lzo -o output.txt\n\n", prog);
    fprintf(stderr, "Environment variables (defaults used by client if CLI options not present):\n");
    fprintf(stderr, "  LZO_STANDARD_COPY=0|1       Default copy mode (0=zero-copy, 1=standard copy). Applies to both compression and decompression.\n");
    fprintf(stderr, "  LZO_MT_IO=0|1               Enable multi-threaded I/O by default.\n");
    fprintf(stderr, "  LZO_MT_IO_THREADS=N         Default number of I/O threads (1-32).\n");
    fprintf(stderr, "  LZO_COALESCE_OUTPUT=0|1     Enable output coalescing by default.\n");
    fprintf(stderr, "  LZO_STDIO_BUF_MB=N          Default stdio buffer size in MB for file writes.\n");
    fprintf(stderr, "\n注意: 需要先启动 lzo_gpu_daemon 守护进程\n");
}


/*
 * 客户端主函数
 */
int main(int argc, char** argv)
{
    const char* input = NULL;
    const char* output = NULL;
    int level = 12;  // 默认: 12 bits (4KB dict)
    int alg = 0;     // 默认: lzo1x
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
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--alg") == 0) {
            if (i + 1 < argc) {
                i++;
                if (strcmp(argv[i], "1x") == 0 || strcmp(argv[i], "lzo1x") == 0) alg = 0;
                else if (strcmp(argv[i], "1y") == 0 || strcmp(argv[i], "lzo1y") == 0) alg = 1;
                else {
                    fprintf(stderr, "错误: 未知算法 '%s'. 支持: 1x, 1y\n", argv[i]);
                    return 1;
                }
            } else {
                fprintf(stderr, "错误: --alg 需要参数\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--level") == 0) {
            if (i + 1 < argc) {
                i++;
                // 支持 10-14
                int val = atoi(argv[i]);
                if (val >= 10 && val <= 14) {
                    level = val;
                } else {
                    // 尝试兼容旧参数
                    if (strcmp(argv[i], "1") == 0) level = 12;
                    else if (strcmp(argv[i], "1k") == 0) level = 10;
                    else if (strcmp(argv[i], "1l") == 0) level = 11;
                    else if (strcmp(argv[i], "1o") == 0) level = 14; // or 15? 14 is max for now
                    else {
                        fprintf(stderr, "错误: level必须是 10-14 (bits)\n");
                        return 1;
                    }
                }
            } else {
                fprintf(stderr, "错误: -l/--level 需要参数\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) { output = argv[++i]; } else { fprintf(stderr, "错误: -o/--output 需要参数\n"); return 1; }
        } else if (argv[i][0] == '-') {
            /* unknown/long options are handled by parse_client_cli_opts(), so
             * ignore them here so they are not treated as a positional input
             */
            continue;
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
        return decompress_with_daemon(input, output, alg);
    }

    return compress_with_daemon(input, output, alg, level);
}
