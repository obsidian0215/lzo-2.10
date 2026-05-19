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
#include "lzo_gpu_core.h"
#include <getopt.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "lzo_gpu_protocol.h"

/* Determin socket path by using helper from protocol.h if needed,
 * but this file currently uses a slightly more complex multi-env lookup.
 * We'll keep the existing client_socket_path for compatibility but wrap it.
 */
#define SOCKET_PATH "/tmp/lzo_gpu_daemon.sock"

/* client_socket_path is now provided by lzo_gpu_protocol.h or lzo_gpu_utils.h */


/* parse human-friendly size string and return KB (rounded) */
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

}


/* Simple CLI parsing for per-request flags. These CLI flags override env vars. */
/* Note: coalescing control is removed from code.
 * stdio buffer control is hardcoded to 1MB.
 */
static request_t g_cli_req_flags; /* global overrides parsed from CLI */
static void parse_client_cli_opts(int argc, char** argv, request_t* req) {
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strcmp(a, "-s") == 0 || strcmp(a, "--socket") == 0) {
            if (i + 1 < argc) {
                setenv("LZO_DAEMON_SOCKET", argv[++i], 1);
                argv[i-1] = "--"; argv[i] = "--";
            }
            continue;
        }
        if (strncmp(a, "-B=", 3) == 0) { req->block_size = (int)lzo_parse_block_size(a + 3); argv[i] = "--"; continue; }
        if (strcmp(a, "-B") == 0 && i + 1 < argc) { argv[i] = "--"; req->block_size = (int)lzo_parse_block_size(argv[++i]); argv[i] = "--"; continue; }
        if (strncmp(a, "--block-size=", 13) == 0) { req->block_size = (int)lzo_parse_block_size(a + 13); argv[i] = "--"; continue; }
        if (strcmp(a, "--block-size") == 0 && i + 1 < argc) { argv[i] = "--"; req->block_size = (int)lzo_parse_block_size(argv[++i]); argv[i] = "--"; continue; }
        if (strncmp(a, "--local=", 8) == 0) { req->local_size = atoi(a + 8); argv[i] = "--"; continue; }
        if (strcmp(a, "--local") == 0 && i + 1 < argc) { argv[i] = "--"; req->local_size = atoi(argv[++i]); argv[i] = "--"; continue; }
        if (strcmp(a, "--zero-copy") == 0) { req->standard_copy = 0; argv[i] = "--"; continue; }
        if (strcmp(a, "--standard-copy") == 0) { req->standard_copy = 1; argv[i] = "--"; continue; }
        if (strcmp(a, "-v") == 0 || strcmp(a, "--verbose") == 0) { g_verbose = 1; argv[i] = "--"; continue; }
    }
}

/* set default request options */
static void set_request_defaults(request_t* req) {
    req->magic = LZO_DAEMON_REQUEST_MAGIC;
    req->version = LZO_DAEMON_REQUEST_VERSION;
    req->standard_copy = 0;
    req->block_size = 0;
    req->local_size = 0;
    /* per-request debug flag removed */
    /* profile_writes removed */
}

/*
 * 检查守护进程是否运行
 */
int is_daemon_running(void)
{
    return access(lzo_daemon_socket_path(), F_OK) == 0;
}

static int client_read_full(int fd, void* buf, size_t len)
{
    unsigned char* p = (unsigned char*)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int client_write_full(int fd, const void* buf, size_t len)
{
    const unsigned char* p = (const unsigned char*)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int stdin_read_u64(uint64_t* out)
{
    size_t n = fread(out, 1, sizeof(*out), stdin);
    if (n == 0 && feof(stdin)) return 1;
    if (n != sizeof(*out)) return -1;
    return 0;
}

static int stdin_read_exact(void* buf, size_t len)
{
    return fread(buf, 1, len, stdin) == len ? 0 : -1;
}

static int stdout_write_u64(uint64_t value)
{
    return fwrite(&value, 1, sizeof(value), stdout) == sizeof(value) ? 0 : -1;
}

static int read_stdin_all(unsigned char** out, size_t* out_len)
{
    size_t cap = 1 << 20;
    size_t len = 0;
    unsigned char* buf = (unsigned char*)malloc(cap);
    if (!buf) return -1;
    for (;;) {
        size_t got;
        if (len == cap) {
            size_t next = cap * 2;
            unsigned char* nb = (unsigned char*)realloc(buf, next);
            if (!nb) { free(buf); return -1; }
            buf = nb;
            cap = next;
        }
        got = fread(buf + len, 1, cap - len, stdin);
        len += got;
        if (got == 0) {
            if (ferror(stdin)) { free(buf); return -1; }
            break;
        }
    }
    *out = buf;
    *out_len = len;
    return 0;
}

static int write_stdout_all(const void* data, size_t len)
{
    return fwrite(data, 1, len, stdout) == len ? 0 : -1;
}

static int raw_with_daemon(char operation, int alg, int level)
{
    int sock;
    struct sockaddr_un addr;
    request_t req;
    response_t resp;
    unsigned char* input = NULL;
    size_t input_len = 0;
    uint64_t out_len = 0;
    unsigned char* out = NULL;
    int rc = -1;

    if (!is_daemon_running()) {
        fprintf(stderr, "错误: 守护进程未运行\n");
        return -1;
    }
    if (read_stdin_all(&input, &input_len) != 0 || input_len == 0) {
        fprintf(stderr, "错误: failed to read stdin raw payload\n");
        free(input);
        return -1;
    }

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) { free(input); return -1; }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, lzo_daemon_socket_path(), sizeof(addr.sun_path) - 1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        free(input);
        return -1;
    }

    memset(&req, 0, sizeof(req));
    set_request_defaults(&req);
    memcpy(&req, &g_cli_req_flags, sizeof(req));
    req.operation = operation;
    req.alg = alg;
    req.level = level;
    req.input_size = input_len;
    req.flags = LZO_DAEMON_FLAG_RAW_BUFFER;
    fill_request_env_flags(&req);

    if (client_write_full(sock, &req, sizeof(req)) != 0 ||
        client_write_full(sock, input, input_len) != 0 ||
        client_read_full(sock, &resp, sizeof(resp)) != 0) {
        perror("daemon raw exchange failed");
        goto out;
    }
    if (resp.status != 0) {
        fprintf(stderr, "daemon raw failed: %s\n", resp.message);
        goto out;
    }
    if (client_read_full(sock, &out_len, sizeof(out_len)) != 0) goto out;
    if (out_len > 0) {
        if (out_len > (uint64_t)SIZE_MAX) goto out;
        out = (unsigned char*)malloc((size_t)out_len);
        if (!out) goto out;
        if (client_read_full(sock, out, (size_t)out_len) != 0) goto out;
        if (write_stdout_all(out, (size_t)out_len) != 0) goto out;
    }
    rc = 0;
out:
    free(out);
    free(input);
    close(sock);
    return rc;
}

static int raw_session_with_daemon(char operation, int alg, int level)
{
    int sock;
    struct sockaddr_un addr;
    int rc = -1;

    if (!is_daemon_running()) {
        fprintf(stderr, "错误: 守护进程未运行\n");
        return -1;
    }

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, lzo_daemon_socket_path(), sizeof(addr.sun_path) - 1);
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("连接守护进程失败");
        close(sock);
        return -1;
    }

    for (;;) {
        uint64_t input_len = 0;
        unsigned char* input = NULL;
        unsigned char* out = NULL;
        uint64_t out_len = 0;
        request_t req;
        response_t resp;

        rc = stdin_read_u64(&input_len);
        if (rc == 1) { rc = 0; break; }
        if (rc != 0 || input_len == 0 || input_len > (uint64_t)SIZE_MAX) {
            fprintf(stderr, "错误: failed to read raw session frame length\n");
            rc = -1;
            break;
        }

        input = (unsigned char*)malloc((size_t)input_len);
        if (!input) { rc = -1; break; }
        if (stdin_read_exact(input, (size_t)input_len) != 0) {
            fprintf(stderr, "错误: failed to read raw session payload\n");
            free(input);
            rc = -1;
            break;
        }

        memset(&req, 0, sizeof(req));
        set_request_defaults(&req);
        memcpy(&req, &g_cli_req_flags, sizeof(req));
        req.operation = operation;
        req.alg = alg;
        req.level = level;
        req.input_size = (size_t)input_len;
        req.flags = LZO_DAEMON_FLAG_RAW_BUFFER;
        fill_request_env_flags(&req);

        if (client_write_full(sock, &req, sizeof(req)) != 0 ||
            client_write_full(sock, input, (size_t)input_len) != 0 ||
            client_read_full(sock, &resp, sizeof(resp)) != 0) {
            perror("daemon raw session exchange failed");
            free(input);
            rc = -1;
            break;
        }
        free(input);
        if (resp.status != 0) {
            fprintf(stderr, "daemon raw failed: %s\n", resp.message);
            rc = -1;
            break;
        }
        if (client_read_full(sock, &out_len, sizeof(out_len)) != 0) {
            rc = -1;
            break;
        }
        if (out_len > (uint64_t)SIZE_MAX) {
            rc = -1;
            break;
        }
        if (out_len > 0) {
            out = (unsigned char*)malloc((size_t)out_len);
            if (!out) { rc = -1; break; }
            if (client_read_full(sock, out, (size_t)out_len) != 0) {
                free(out);
                rc = -1;
                break;
            }
        }
        if (stdout_write_u64(out_len) != 0 ||
            (out_len > 0 && fwrite(out, 1, (size_t)out_len, stdout) != (size_t)out_len)) {
            free(out);
            rc = -1;
            break;
        }
        fflush(stdout);
        free(out);
    }

    close(sock);
    return rc;
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
        fprintf(stderr, "请先启动: ./lzo_gpu --daemon\n");
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
    strncpy(addr.sun_path, lzo_daemon_socket_path(), sizeof(addr.sun_path) - 1);

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
        if (g_verbose) {
            lzo_print_response_stats(&resp, input, 'D', alg);
        } else {
            printf("Decompressed %zu -> %zu in %.2f ms\n", (size_t)st.st_size, (size_t)resp.out_size, resp.time_us / 1000.0);
        }
        return 0;
    } else {
        fprintf(stderr, "解压缩失败: %s\n", resp.message);
        close(sock);
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
        fprintf(stderr, "请先启动: ./lzo_gpu --daemon\n");
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
    strncpy(addr.sun_path, lzo_daemon_socket_path(), sizeof(addr.sun_path) - 1);

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
                    if (g_verbose) {
                        fprintf(stderr, "[CLIENT] Request opts: block_size=%dKB local_size=%d standard_copy=%d sizeof(req)=%zu\n",
                                req.block_size, req.local_size, req.standard_copy, sizeof(req));
                        fprintf(stderr, "[CLIENT] offsetof: block_size=%zuKB local_size=%zu\n",
                                offsetof(request_t, block_size), offsetof(request_t, local_size));
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
        if (g_verbose) {
            lzo_print_response_stats(&resp, input, 'C', alg);
        } else {
             double ratio = resp.out_size > 0 ? (double)req.input_size / (double)resp.out_size : 0.0;
             printf("Compressed %zu -> %zu (%.2f:1) in %.2f ms\n", (size_t)req.input_size, (size_t)resp.out_size, ratio, resp.time_us / 1000.0);
        }
        return 0;
    } else {
        fprintf(stderr, "压缩失败: %s\n", resp.message);
        return -1;
    }
}

static void show_help(const char* prog) {
    fprintf(stderr, "LZO GPU Client - 通过守护进程进行 GPU 加速压缩/解压\n\n");
    fprintf(stderr, "用法: %s [选项] <input1> [input2 ...]\n", prog);
    fprintf(stderr, "选项:\n");
    fprintf(stderr, "  -h, --help                    Show help and exit.\n");
    fprintf(stderr, "  -L, -l, --level <10-15>       Set GPU dictionary bits for the hash-dictionary kernel. Default: 11.\n");
    fprintf(stderr, "  -a, --alg <lzo1x|lzo1y>       Set algorithm. Default: lzo1x.\n");
    fprintf(stderr, "  -d, --decompress              Decompress mode.\n");
    fprintf(stderr, "  -o, --output <file>           Output file (only valid for single input).\n");
    fprintf(stderr, "  -B N, --block-size N          Block size (B/KB/MB, default: 64KB).\n");
    fprintf(stderr, "  --mt-io / --no-mt-io          Enable/disable multi-threaded I/O.\n");
    fprintf(stderr, "  --zero-copy / --standard-copy Enable/disable zero-copy (pinned memory).\n");
    fprintf(stderr, "  --local N                     Local work-group size.\n");
    fprintf(stderr, "  --raw-buffer                  Read request payload from stdin and write output to stdout via daemon raw-buffer protocol.\n");
    fprintf(stderr, "  --raw-buffer-session          Keep one daemon socket open and process multiple length-prefixed payload frames from stdin.\n");
    fprintf(stderr, "示例:\n");
    fprintf(stderr, "  %s input.txt                  # 压缩为 input.txt.lzo\n", prog);
    fprintf(stderr, "  %s -d file1.lzo file2.lzo     # 批量解压\n", prog);
    fprintf(stderr, "  %s -a lzo1y -L 12 bigfile.bin # 使用 LZO1Y-12 压缩\n\n", prog);
    fprintf(stderr, "注意: 需要先启动 ./lzo_gpu --daemon 守护进程\n");
}


/*
 * 客户端主函数
 */
int run_lzo_client(int argc, char** argv)
{
    const char** inputs = malloc(argc * sizeof(char*));
    int input_count = 0;
    const char* output_arg = NULL;
    /* Use -1 as sentinel for unspecified; let server decide */
    int level = -1;  /* 未指定: 让服务端决定 */
    int alg = -1;    /* 未指定: 让服务端决定 */
    char operation = 'C';  /* 默认压缩 */
    int show_help_flag = 0;
    int raw_buffer_mode = 0;
    int raw_buffer_session_mode = 0;



    if (!inputs) { perror("malloc inputs"); return 1; }

    // Parse client-level CLI options (per-request overrides) first
    set_request_defaults(&g_cli_req_flags);
    parse_client_cli_opts(argc, argv, &g_cli_req_flags);

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) continue; /* ignore internal marks */
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            show_help_flag = 1;
            break;
        }

        if (strcmp(argv[i], "--raw-buffer") == 0) {
            raw_buffer_mode = 1;
        } else if (strcmp(argv[i], "--raw-buffer-session") == 0) {
            raw_buffer_session_mode = 1;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decompress") == 0) {
            operation = 'D';
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--alg") == 0) {
            if (i + 1 < argc) {
                i++;
                if (strcmp(argv[i], "1x") == 0 || strcmp(argv[i], "lzo1x") == 0) alg = 0;
                else if (strcmp(argv[i], "1y") == 0 || strcmp(argv[i], "lzo1y") == 0) alg = 1;
                else {
                    fprintf(stderr, "错误: 未知算法 '%s'. 支持: 1x, 1y\n", argv[i]);
                    free(inputs); return 1;
                }
            } else {
                fprintf(stderr, "错误: --alg 需要参数\n");
                free(inputs); return 1;
            }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "-L") == 0 || strcmp(argv[i], "--level") == 0) {
            if (i + 1 < argc) {
                i++;
                int val = atoi(argv[i]);
                if (val >= 11 && val <= 20) {
                    level = val;
                } else {
                    // 尝试兼容旧参数
                    if (strcmp(argv[i], "1") == 0) level = 14;
                    else if (strcmp(argv[i], "1k") == 0) level = 12;
                    else if (strcmp(argv[i], "1l") == 0) level = 13;
                    else if (strcmp(argv[i], "1o") == 0) level = 15;
                    else {
                        fprintf(stderr, "错误: level必须是 11-20 (bits)\n");
                        free(inputs); return 1;
                    }
                }
            } else {
                fprintf(stderr, "错误: -l/--level 需要参数\n");
                free(inputs); return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) { output_arg = argv[++i]; }
            else { fprintf(stderr, "错误: -o/--output 需要参数\n"); free(inputs); return 1; }
        } else if (argv[i][0] == '-') {
            continue;
        } else {
            inputs[input_count++] = argv[i];
        }
    }

    // 显示帮助
    if (show_help_flag || ((!raw_buffer_mode && !raw_buffer_session_mode) && input_count == 0)) {
        show_help(argv[0]);
        free(inputs);
        return show_help_flag ? 0 : 1;
    }

    if (raw_buffer_session_mode) {
        int final_alg = (alg < 0) ? 0 : alg;
        int final_level = (level < 0) ? LZO_DEFAULT_COMP_LEVEL : level;
        int ret = raw_session_with_daemon(operation, final_alg, final_level);
        free(inputs);
        return ret == 0 ? 0 : 1;
    }

    if (raw_buffer_mode) {
        int final_alg = (alg < 0) ? 0 : alg;
        int final_level = (level < 0) ? LZO_DEFAULT_COMP_LEVEL : level;
        int ret = raw_with_daemon(operation, final_alg, final_level);
        free(inputs);
        return ret == 0 ? 0 : 1;
    }



    if (input_count > 1 && output_arg) {
        fprintf(stderr, "警告: 指定了多个输入文件且指定了 -o。所有输出将尝试写入同一个路径(可能失败或覆盖)。建议不要在多文件模式下使用 -o。\n");
    }

    int final_ret = 0;
    for (int idx = 0; idx < input_count; idx++) {
        const char* input = inputs[idx];
        const char* output = output_arg;
        int cleanup_output = 0;

        if (!output) {
            size_t ilen = strlen(input);
            if (operation == 'C') {
                size_t n = ilen + 5 + 1;
                char *tmp = malloc(n);
                if (!tmp) { perror("malloc"); final_ret = 1; break; }
                snprintf(tmp, n, "%s.lzo", input);
                output = tmp;
            } else {
                const char *suf = ".lzo";
                size_t suf_len = strlen(suf);
                if (ilen > suf_len && strcmp(input + ilen - suf_len, suf) == 0) {
                    size_t n = ilen - suf_len + 1;
                    char *tmp = malloc(n);
                    if (!tmp) { perror("malloc"); final_ret = 1; break; }
                    memcpy(tmp, input, ilen - suf_len);
                    tmp[ilen - suf_len] = '\0';
                    output = tmp;
                } else {
                    size_t n = ilen + 4 + 1;
                    char *tmp = malloc(n);
                    if (!tmp) { perror("malloc"); final_ret = 1; break; }
                    snprintf(tmp, n, "%s.dec", input);
                    output = tmp;
                }
            }
            cleanup_output = 1;
        }

        if (input_count > 1) {
            printf("[%d/%d] 处理: %s -> %s\n", idx + 1, input_count, input, output);
        }

        int ret;
        if (operation == 'D') {
            ret = decompress_with_daemon(input, output, alg);
        } else {
            ret = compress_with_daemon(input, output, alg, level);
        }

        if (ret != 0) {
            fprintf(stderr, "处理失败: %s\n", input);
            final_ret = 1;
        }

        if (cleanup_output) {
            free((void*)output);
        }
    }

    free(inputs);
    return final_ret;
}
