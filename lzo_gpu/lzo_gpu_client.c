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

#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"

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
        /* Decompression path: show consistent breakdown matching daemon_decompress */
        printf("解压缩成功: %s -> %s\n", input, output);
        /* Printers align with daemon_decompress logging so the harness can parse them reliably */
        printf("1. File Read           : %8.3f ms\n", resp.timing.file_read_us/1000.0);
        printf("2. OCL Init            : %8.3f ms\n", resp.timing.ocl_init_us/1000.0);
        printf("3. Kernel Load         : %8.3f ms\n", resp.timing.kernel_load_us/1000.0);
        printf("4. Buffer Alloc        : %8.3f ms\n", resp.timing.buffer_alloc_in_us/1000.0);
        printf("5. Data Upload         : %8.3f ms\n", resp.timing.data_upload_us/1000.0);
        printf("6. Setup Args          : %8.3f ms\n", resp.timing.setup_args_us/1000.0);
        printf("7. Kernel Exec         : %8.3f ms\n", resp.timing.kernel_exec_us/1000.0);
        printf("8. Data Download       : %8.3f ms\n", resp.timing.download_total_us/1000.0);
        printf("9. File Write          : %8.3f ms\n", resp.timing.file_write_us/1000.0);
        printf("TOTAL                : %8.3f ms\n", resp.time_us/1000.0);
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
        /* Compression path: show breakdown matching daemon_compress logging */
        printf("压缩成功: %s -> %s\n", input, output);
        printf("1. File Read           : %8.3f ms\n", resp.timing.file_read_us/1000.0);
        printf("2. Blocking Calc       : %8.3f ms\n", resp.timing.blocking_calc_us/1000.0);
        printf("3. Buffer Alloc (in)   : %8.3f ms\n", resp.timing.buffer_alloc_in_us/1000.0);
        printf("4. Data Upload         : %8.3f ms\n", resp.timing.data_upload_us/1000.0);
        printf("5. Buffer Alloc (out)  : %8.3f ms\n", resp.timing.buffer_alloc_out_us/1000.0);
        printf("6. Buffer Alloc (len)  : %8.3f ms\n", resp.timing.buffer_alloc_len_us/1000.0);
        printf("7. Setup Args          : %8.3f ms\n", resp.timing.setup_args_us/1000.0);
        printf("8. Kernel Exec         : %8.3f ms\n", resp.timing.kernel_exec_us/1000.0);
        printf("9. Download (len)      : %8.3f ms\n", resp.timing.download_len_us/1000.0);
        printf("10. Download (bulk)    : %8.3f ms\n", resp.timing.download_bulk_us/1000.0);
        printf("11. Download Total     : %8.3f ms\n", resp.timing.download_total_us/1000.0);
        printf("12. File Write         : %8.3f ms\n", resp.timing.file_write_us/1000.0);
        printf("13. Cleanup            : %8.3f ms\n", resp.timing.cleanup_us/1000.0);
        printf("TOTAL                  : %8.3f ms\n", resp.time_us/1000.0);
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
    int level = 7;  // 默认: lzo1x_1 (D_BITS=14, 标准配置)
    char operation = 'C';  // 默认压缩

    // Parse client-level CLI options (per-request overrides) first
    set_request_defaults(&g_cli_req_flags);
    parse_client_cli_opts(argc, argv, &g_cli_req_flags);

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
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) { output = argv[++i]; } else { fprintf(stderr, "错误: -o/--output 需要参数\n"); return 1; }
        } else if (!input) {
            input = argv[i];
        } else {
            fprintf(stderr, "错误: 仅支持一个位置参数作为 <input>，不要使用位置形式指定输出，请改用 -o/--output\n");
            return 1;
        }
    }

    /* If output not provided, compute sensible defaults:
     * - For compression: input -> input + ".lzo"
     * - For decompression: if input ends with .lzo, strip it; else append ".dec"
     */
    if (!output) {
        if (!input) {
            fprintf(stderr, "错误: 未指定输入文件\n");
            return 1;
        }
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

    // 检查必需参数
    if (!input || !output) {
        fprintf(stderr, "用法: %s [选项] <input>\n", argv[0]);
        fprintf(stderr, "选项:\n");
        fprintf(stderr, "  -L, -l, --level <1|1k|1l|1o>  压缩级别 (默认: 1l)\n");
        fprintf(stderr, "  -d, --decompress          解压缩模式\n");
        fprintf(stderr, "  -o, --output <out>        指定输出文件 (必须)\n");
        fprintf(stderr, "  -h, --help                Show this help\n\n");

        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  %s input.txt -o output.lzo           # 使用level=1l压缩\n", argv[0]);
        fprintf(stderr, "  %s -l 1k input.txt -o output.lzo     # 使用lzo1x_1k压缩\n", argv[0]);
        fprintf(stderr, "  %s -d input.lzo -o output.txt        # 解压缩 (输出必须使用 -o/--output)\n\n", argv[0]);

        fprintf(stderr, "Client / per-request options (via environment variables):\n");
        fprintf(stderr, "  LZO_STANDARD_COPY=0|1    Request daemon to use standard host->device copy (default: 0=zero-copy)\n");
        /* LZO_FORCE_MAP removed: use LZO_STANDARD_COPY to select mapping */
        fprintf(stderr, "  LZO_MT_IO=0|1            Enable multi-threaded I/O for reads/uploads (default: 0)\n");
        fprintf(stderr, "  LZO_MT_IO_THREADS=N      Number of I/O threads (1-32; default: %d)\n", LZO_DEFAULT_MT_IO_THREADS);
        fprintf(stderr, "  LZO_FIXED_BLOCK_SIZE=N   Request fixed block size (KB) for daemon (0 = adaptive)\n");
        fprintf(stderr, "  LZO_ASYNC_UPLOAD=0|1    Request daemon to use asynchronous uploads (non-blocking clEnqueueWriteBuffer)\n");
        fprintf(stderr, "  LZO_OPENCL_DEVICE=CPU|GPU Select device preference for daemon (daemon may ignore)\n");
        fprintf(stderr, "  LZO_DECOMP_VEC=0|1       Prefer vectorized decompressor (daemon-side; default:1)\n");
        fprintf(stderr, "  LZO_COALESCE_OUTPUT=0|1  Enable coalesced file output (default: same as daemon defaults)\n");
        fprintf(stderr, "  LZO_COALESCE_CHUNK_MB=N   Coalesce chunk size in MB (default from lzo_defaults.h)\n");
        fprintf(stderr, "  LZO_COALESCE_MAX_MB=N     Max MB to coalesce in one write (default from lzo_defaults.h)\n");
        fprintf(stderr, "  LZO_STDIO_BUF_MB=N        Size of stdio write buffer in MB (default from lzo_defaults.h)\n\n");

        fprintf(stderr, "Notes:\n");
        fprintf(stderr, "  - The client reads these environment variables and sends them per-request to the daemon.\n");
        fprintf(stderr, "  - The daemon may choose to accept or ignore certain preferences (device choice, etc.) depending on its configuration.\n\n");

        return 1;
    }

    /* Parse additional client CLI flags (sets per-request options that
     * override env vars). These are non-positional flags that may come
     * anywhere on the command line, so parse them now.
     */
    /* initialize defaults (-1 = unspecified) */
    {
        /* We'll do this on a local request obj later, for now just parse values. */
    }

    /* Perform parsing of per-request flags to override env options */
    // (we parse whole argv but non-positionally so the earlier simple parsing is preserved)
    // note: the parse_client_cli_opts uses getopt_long which doesn't consume positional arguments necessarily

    /* If the operation is decompression, go to decompress path */
    if (operation == 'D') {
        return decompress_with_daemon(input, output);
    }

    return compress_with_daemon(input, output, level);
}
