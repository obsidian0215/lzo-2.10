#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <libgen.h>
#include <linux/limits.h>
#include <signal.h>
#include "lzo_defaults.h"
#include "timing.h"
#include "lzo_gpu_utils.h"
#include "lzo_gpu_core.h"

/* Forward declarations for daemon and client modules */
#include "lzo_gpu_utils.h"

#define now_ns lzo_now_ns

int run_lzo_daemon(int argc, char** argv);
int run_lzo_client(int argc, char** argv);

/* Helper for multi-threaded pread into a destination buffer.
* 压缩文件格式：
uint16  magic     = 0x4C5A   // 'L''Z'
uint32  orig_size               (≤4 GiB)
uint32  blk_size
uint32  nblk
uint32  len[nblk]               // 每块压缩长度
-----   nblk 个压缩块数据
*/
#define MAGIC  0x4C5A   /* 'L''Z' */
#define D_BITS          11
//#define BLK_SIZE        (32 * 1024)
/* Compression ratio tracking */
#define ENABLE_COMPRESSION_RATIO_TRACKING 1

#if defined(_WIN32) || defined(_WIN64)
     * → counter * 1e9 / freq = 纳秒
     */
    return (uint64_t)counter.QuadPart * (uint64_t)1000000000ULL /
        (uint64_t)freq.QuadPart;
}

#endif

static inline void print_ns(const char* tag, uint64_t ns) {
    unsigned long us = (unsigned long)(ns / 1000ULL);
    /* print_us_tag is provided by timing.h */
    print_us_tag(stdout, tag, us);
}

/* CLI override variables (set by parsing -B/--block-size and --local) */
static size_t g_cli_fixed_block_bytes = 0; /* 0 = not specified */
static int g_cli_fixed_block_exact = 0; /* 1 = user specified bytes (B suffix) -> respect exact */
static size_t g_cli_local_size = 0;       /* 0 = not specified */

#define CHECK(expr)  do{ cl_int _e=(expr);                       \
        if(_e!=CL_SUCCESS){                                      \
            fprintf(stderr,"OpenCL error %d at %s:%d\n",         \
                    _e,__FILE__,__LINE__); exit(1);} }while(0)

static inline size_t lzo_worst(size_t n) {
    return n + n / 16 + 64 + 3;
}

static cl_context  ctx;
static cl_command_queue q;
static cl_device_id dev;

static void ocl_init(void)
{
    uint64_t t1 = now_ns();
    cl_int err;
    cl_platform_id pf = NULL;
    err = clGetPlatformIDs(1, &pf, NULL);
    if (err != CL_SUCCESS || pf == NULL) {
        fprintf(stderr, "OpenCL init failed: clGetPlatformIDs err=%d\n", err);
        ctx = NULL;
        q = NULL;
        return;
    }

    err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_DEFAULT, 1, &dev, NULL);
    }
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL init failed: clGetDeviceIDs err=%d\n", err);
        ctx = NULL;
        q = NULL;
        return;
    }

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || ctx == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateContext err=%d\n", err);
        ctx = NULL;
        q = NULL;
        return;
    }
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
    if (err != CL_SUCCESS || q == NULL) {
        fprintf(stderr, "OpenCL init failed: clCreateCommandQueueWithProperties err=%d\n", err);
        if (ctx) clReleaseContext(ctx);
        ctx = NULL;
        q = NULL;
        return;
    }
    uint64_t t2 = now_ns();
    g_ocl_init_us = (unsigned long)((t2 - t1) / 1000);
}

void print_buildlog(cl_program program, cl_device_id device) {
    char* buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-1);
    }
    buff_erro = malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr, "Build log: \n%s\n", buff_erro); //Be careful with fprint
    free(buff_erro);
    fprintf(stderr, "clBuildProgram failed\n");
}


/* Helper: load program from source file with D_BITS macro */
static cl_program load_prog_with_dbits(const char* alg_name, int bits)
{
    char build_log[8192] = {0};
    cl_program p = lzo_load_program_with_dbits(ctx, dev, alg_name, bits, build_log, sizeof(build_log));
    if (!p) {
        fprintf(stderr, "failed to load/compile kernel %s (D_BITS=%d): %s\n", alg_name, bits, build_log);
    }
    return p;
}

static inline void show_help(char *prog_name)
{
    fprintf(stderr, "Unified LZO GPU Tool\n");
    fprintf(stderr, "Usage Modes:\n");
    fprintf(stderr, "  1. Standalone:   %s [options] <input_file>\n", prog_name);
    fprintf(stderr, "  2. Run Daemon:   %s --daemon [options]\n", prog_name);
    fprintf(stderr, "  3. Use Daemon:   %s --use-daemon [options] <input_file>\n", prog_name);
    fprintf(stderr, "  4. Stop Daemon:  %s --stop-daemon\n", prog_name);

    fprintf(stderr, "\nBasic Options:\n");
    fprintf(stderr, "  -c                   Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress     Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE    Output file (use '-' for stdout)\n");
    fprintf(stderr, "  -a, --alg ALG        Algorithm (lzo1x, lzo1y) (default: lzo1x)\n");
    fprintf(stderr, "  -L, --level LEVEL    Dictionary bits (10-15) (default: 11)\n");
    fprintf(stderr, "  -B, --block-size N   Block size (B/KB/MB) (default: 16KB)\n");
    fprintf(stderr, "  -v, --verbose        Enable performance statistics\n");
    fprintf(stderr, "  --local N            Local work-group size (default: 4)\n");
    fprintf(stderr, "Detailed Help:\n");
    fprintf(stderr, "  %s --daemon -h           # Daemon-specific settings\n", prog_name);
    fprintf(stderr, "  %s --use-daemon -h       # Client-specific settings\n\n", prog_name);

    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  Compress:         %s input.dat -o out.lzo\n", prog_name);
    fprintf(stderr, "  Decompress:       %s -d out.lzo -o out.dec\n", prog_name);
    fprintf(stderr, "  Using Daemon:     %s --use-daemon -a lzo1y bigfile.bin\n", prog_name);
    fprintf(stderr, "  %s -h|--help                                 # show this help\n", prog_name);
    fprintf(stderr, "\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
    fprintf(stderr, "    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload). Applies to both compression and decompression.\n");
}


/* Prototypes for extracted helpers to keep main concise */
static int do_decompress_mode(const char* lz_path, const char* output_path, int output_explicit, int suppress_non_data);
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int suppress_non_data, const char* alg_name, int comp_level);

/* Implementations: wrappers that use the shared core backend (lzo_gpu_core.c)
 * The helpers create short-lived OpenCL contexts, load the appropriate kernels
 * and call lzo_compress_core / lzo_decompress_core to perform the heavy lifting.
 */
static int do_compress_mode(const char* in_path, const char* output_path, int output_explicit, int suppress_non_data, const char* alg_name, int comp_level)
{
    if (!in_path) {
        fprintf(stderr, "error: missing input\n");
        return 1;
    }

    if (comp_level < 0) comp_level = LZO_DEFAULT_COMP_LEVEL;

    ocl_init();
    if (!ctx || !q) {
        fprintf(stderr, "error: failed to initialize OpenCL runtime\n");
        return 1;
    }

    /* load compression kernel */
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    char build_log[8192] = {0};
    int kernel_has_dbg = 0;
    uint64_t tk1 = now_ns();
    if (lzo_load_comp_kernel(ctx, dev, alg_name, comp_level, 0, &prog_c, &krn_c, &kernel_has_dbg, build_log, sizeof(build_log)) != 0) {
        if (build_log[0]) fprintf(stderr, "error: failed to load kernel for %s bits=%d: %s\n", alg_name, comp_level, build_log);
        else fprintf(stderr, "error: failed to load kernel for %s bits=%d\n", alg_name, comp_level);
        return 1;
    }
    uint64_t tk2 = now_ns();
    g_kernel_load_us = (unsigned long)((tk2 - tk1) / 1000);

    int standard_copy = (getenv("LZO_STANDARD_COPY") && atoi(getenv("LZO_STANDARD_COPY")) == 1) ? 1 : 0;
    size_t block_size = g_cli_fixed_block_bytes;
    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;

    lzo_gpu_workspace_t ws;
    lzo_gpu_workspace_init(&ws);

    /* Create parameter object */
    lzo_compress_params_t params = {
        .level = comp_level,
        .alg_id = alg_id,
        .standard_copy = standard_copy,
        .block_size = block_size,
        .local_size_param = (int)g_cli_local_size,
        .debug = 0
    };

    unsigned long time_us = 0;
    size_t output_size = 0;
    timing_t t_out = {0};

    int ret = lzo_compress_core(ctx, q, dev, krn_c, in_path, output_path, &params, &ws, &time_us, &output_size, &t_out);

    if (ret == 0) {
        if (g_verbose) {
            response_t r = {0};
            r.status = 0;
            r.time_us = time_us;
            r.out_size = output_size;
            r.timing = t_out;
            lzo_print_response_stats(&r, in_path, 'C', alg_id);
        } else {
            double ratio = (double)t_out.in_size / (t_out.out_size > 0 ? t_out.out_size : 1);
            printf("%s : %zu -> %zu (%.2f:1) in %.2f ms\n", in_path, t_out.in_size, t_out.out_size, ratio, time_us / 1000.0);
        }
    }
    if (krn_c) clReleaseKernel(krn_c);
    if (prog_c) clReleaseProgram(prog_c);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return ret;
}

static int do_decompress_mode(const char* lz_path, const char* output_path, int output_explicit, int suppress_non_data)
{
    if (!lz_path) { fprintf(stderr, "error: missing input .lzo\n"); return 1; }
    int standard_copy = (getenv("LZO_STANDARD_COPY") && atoi(getenv("LZO_STANDARD_COPY")) == 1) ? 1 : 0;

    FILE* f = fopen(lz_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    uint16_t magic; fread(&magic, 2, 1, f);
    if (magic != 0x4C5A) { fprintf(stderr, "error: magic mismatch\n"); fclose(f); return 1; }
    uint32_t a[4]; fread(a, 4, 4, f); // orig_sz, blk_sz, nblk, alg_id
    fclose(f);

    const char* alg_name = (a[3] == 1) ? "lzo1y" : "lzo1x";
    char decomp_base[64]; snprintf(decomp_base, sizeof(decomp_base), "%s_decomp", alg_name);

    ocl_init();
    if (!ctx || !q) {
        fprintf(stderr, "error: failed to initialize OpenCL runtime\n");
        return 1;
    }
    cl_program prog_d = NULL;
    cl_int err;
    char build_log[8192] = {0};

    uint64_t tk1 = now_ns();
    /* Fallback to base decompressor */
    if (!prog_d) {
        prog_d = lzo_load_program_with_dbits(ctx, dev, decomp_base, 0, build_log, sizeof(build_log));
        if (!prog_d) { fprintf(stderr, "error: unable to load decompressor for %s\n", decomp_base); return 1; }
    }

    char krn_name[64]; snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", alg_name);
    cl_kernel krn_d = clCreateKernel(prog_d, krn_name, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); clReleaseProgram(prog_d); return 1; }
    uint64_t tk2 = now_ns();
    g_kernel_load_us = (unsigned long)((tk2 - tk1) / 1000);

    lzo_gpu_workspace_t ws;
    lzo_gpu_workspace_init(&ws);
    unsigned long time_us = 0; size_t output_size = 0; timing_t t_out = {0};

    int rc = lzo_decompress_core(ctx, q, dev, krn_d, lz_path, output_path, &ws, standard_copy, (int)g_cli_local_size, 0, &time_us, &output_size, &t_out);
    if (rc == 0) {
        if (g_verbose) {
            response_t r = {0};
            r.status = 0;
            r.time_us = time_us;
            r.out_size = output_size;
            r.timing = t_out;
            lzo_print_response_stats(&r, lz_path, 'D', (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0);
        } else {
            printf("%s : %zu -> %zu in %.2f ms\n", lz_path, t_out.in_size, t_out.out_size, time_us / 1000.0);
        }
    }
    lzo_gpu_workspace_free(&ws);

    if (krn_d) clReleaseKernel(krn_d);
    if (prog_d) clReleaseProgram(prog_d);
    if (q) { clReleaseCommandQueue(q); q = NULL; }
    if (ctx) { clReleaseContext(ctx); ctx = NULL; }
    return rc;
}


/* Implementations will call into lzo_gpu_core.c which provides lzo_compress_core/lzo_decompress_core */
#include "lzo_gpu_core.h"


int run_lzo_standalone(int argc, char** argv)
{
    if (argc < 1) { // Changed from 2 to 1 because we might pass 0/1 args if flags stripped
        show_help(argv[0]);
        return 0;
    }

    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int comp_level = -1; /* -1 means adaptive (not explicitly specified by user) */
    int comp_level_specified = 0;
    const char *alg_name = "lzo1x";
    int alg_specified = 0;

    /* parse options and positionals */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            show_help(argv[0]);
            return 0;
        }
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            g_verbose = 1;
            continue;
        }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
                return 1;
            }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "-l") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
                return 1;
            }
            comp_level = atoi(argv[++i]);
            comp_level_specified = 1;
            if (comp_level < 8 || comp_level > 20) {
                fprintf(stderr, "error: dictionary size must be between 8 and 20 bits (got %d)\n", comp_level);
                return 1;
            }
            continue;
        }
        if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing argument for %s\n", arg);
                return 1;
            }
            alg_name = argv[++i];
            alg_specified = 1;
            if (strcmp(alg_name, "1x") == 0 || strcmp(alg_name, "lzo1x") == 0)
                alg_name = "lzo1x";
            else if (strcmp(alg_name, "1y") == 0 || strcmp(alg_name, "lzo1y") == 0)
                alg_name = "lzo1y";
            else {
                fprintf(stderr, "错误: 未知算法 '%s'. 支持: lzo1x, lzo1y (或 1x/1y)\n", alg_name);
                return 1;
            }
            continue;
        }
        if (strcmp(arg, "-c") == 0) {
            decompress_mode = 0;
            continue;
        }
        if (strcmp(arg, "-d") == 0 || strcmp(arg, "--decompress") == 0) {
            decompress_mode = 1;
            continue;
        }
        if (strcmp(arg, "-B") == 0 || strcmp(arg, "--block-size") == 0) {
            if (i + 1 < argc) {
                const char* s = argv[++i];
                g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s);
                size_t b = lzo_parse_block_size(s);
                if (b > 0) g_cli_fixed_block_bytes = b;
            } else {
                fprintf(stderr, "Error: -B requires an argument\n");
                return 1;
            }
            continue;
        }
        if (strncmp(arg, "--block-size=", 13) == 0) {
            const char* s = arg + 13;
            g_cli_fixed_block_exact = lzo_specified_unit_is_bytes(s);
            size_t b = lzo_parse_block_size(s);
            if (b > 0) g_cli_fixed_block_bytes = b;
            continue;
        }
        if (strncmp(arg, "--local=", 8) == 0) {
            g_cli_local_size = (size_t)atoi(arg + 8);
            continue;
        }
        if (strcmp(arg, "--local") == 0) {
            if (i + 1 < argc) g_cli_local_size = (size_t)atoi(argv[++i]);
            else { fprintf(stderr, "Error: --local requires an argument\n"); return 1; }
            continue;
        }
        /* positional or error */
        if (arg[0] == '-') {
            fprintf(stderr, "Error: Unknown option %s\n", arg);
            fprintf(stderr, "Tips: Use -h or --help for help. Standard positional usage: %s <input> [output]\n", argv[0]);
            return 1;
        } else {
            if (!in_path && !lz_path) {
                if (decompress_mode) lz_path = arg; else in_path = arg;
            } else if (!output_explicit) {
                output_path = arg;
                output_explicit = 1;
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                return 1;
            }
        }
    }
    /* Set default output names if not specified */
    char default_output[512];
    if (output_explicit == 0 || output_path == NULL) {
        if (decompress_mode) {
            if (lz_path) {
                size_t ilen = strlen(lz_path);
                const char *suf = ".lzo";
                size_t suf_len = strlen(suf);
                if (ilen > suf_len && strcmp(lz_path + ilen - suf_len, suf) == 0) {
                    /* strip suffix */
                    size_t n = (ilen - suf_len < sizeof(default_output) - 1) ? ilen - suf_len : sizeof(default_output) - 1;
                    memcpy(default_output, lz_path, n);
                    default_output[n] = '\0';
                } else {
                    /* append .dec */
                    snprintf(default_output, sizeof(default_output), "%s.dec", lz_path);
                }
                output_path = default_output;
            }
        } else {
            if (in_path) {
                snprintf(default_output, sizeof(default_output), "%s.lzo", in_path);
                output_path = default_output;
            }
        }
    }

    /* Apply autotune config for compression if present (only change unspecified fields). */
    if (!decompress_mode) {
        request_t ar;
        memset(&ar, 0, sizeof(ar));
        ar.operation = 'C';
        ar.block_size = (g_cli_fixed_block_bytes > 0) ? (int)(g_cli_fixed_block_bytes / 1024) : 0;
        ar.local_size = (uint32_t)g_cli_local_size;
        ar.alg = alg_specified ? ((strcmp(alg_name, "lzo1y") == 0) ? 1 : 0) : -1;
        ar.level = comp_level_specified ? comp_level : -1;
        if (in_path) {
            struct stat st;
            if (stat(in_path, &st) == 0) ar.input_size = st.st_size;
        }

        if (lzo_apply_autotune_config(&ar) == 0) {
            if (!alg_specified && ar.alg >= 0) {
                alg_name = (ar.alg == 1) ? "lzo1y" : "lzo1x";
            }
            if (!comp_level_specified && ar.level > 0) comp_level = ar.level;
            if (g_cli_fixed_block_bytes == 0 && ar.block_size > 0) g_cli_fixed_block_bytes = (size_t)ar.block_size * 1024;
            if (g_cli_local_size == 0 && ar.local_size > 0) g_cli_local_size = (size_t)ar.local_size;
            if (g_verbose) {
                fprintf(stderr, "[AUTOTUNE] Applied autotune suggestions: alg=%d level=%d block_kb=%d local_size=%u\n",
                        ar.alg, ar.level, ar.block_size, ar.local_size);
            }
        } else {
            if (g_verbose) fprintf(stderr, "[AUTOTUNE] No autotune config found in exe dir or LZO_GPU_AUTOTUNE_CONF\n");
        }
    }

    /* Decompress mode */
    if (decompress_mode) {
        return do_decompress_mode(lz_path, output_path, output_explicit, suppress_non_data);
    }

    /* Compress path (simple, fast) */
    return do_compress_mode(in_path, output_path, output_explicit, suppress_non_data, alg_name, comp_level);
}


/* --- Unified Tool Main Entry --- */
static int stop_daemon(void) {
    const char* pid_path = lzo_daemon_pidfile_path();
    const char* sock_path = lzo_daemon_socket_path();
    FILE* f = fopen(pid_path, "r");
    if (!f) {
        printf("守护进程似乎没有运行 (未找到 PID 文件: %s)\n", pid_path);
        /* Check if socket exists anyway */
        if (access(sock_path, F_OK) == 0) unlink(sock_path);
        return 0;
    }
    pid_t pid;
    if (fscanf(f, "%d", &pid) != 1) {
        fclose(f);
        printf("无法读取 PID 文件内容\n");
        return 1;
    }
    fclose(f);

    printf("正在停止守护进程 (PID: %d)...\n", pid);
    if (kill(pid, SIGTERM) == 0) {
        /* Wait up to 5 seconds for it to exit */
        for (int i = 0; i < 50; i++) {
            if (kill(pid, 0) != 0) break;
            usleep(100000);
        }
        if (kill(pid, 0) == 0) {
            printf("警告: 守护进程未能在 5 秒内退出，正在强制终止...\n");
            kill(pid, SIGKILL);
        }
    } else if (errno == ESRCH) {
        printf("进程 %d 已经退出\n", pid);
    } else {
        perror("停止守护进程失败");
    }

    unlink(pid_path);
    unlink(sock_path);
    printf("守护进程清理完成\n");
    return 0;
}

int main(int argc, char** argv) {
    /* Handle global options like --socket/--pid first for all sub-commands */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--socket") == 0) {
            if (i + 1 < argc) lzo_set_daemon_socket_path(argv[i+1]);
        } else if (strcmp(argv[i], "--pid") == 0) {
            if (i + 1 < argc) lzo_set_daemon_pidfile_path(argv[i+1]);
        }
    }

    if (argc >= 2) {
        if (strcmp(argv[1], "--daemon") == 0) {
            return run_lzo_daemon(argc - 1, argv + 1);
        }
        if (strcmp(argv[1], "--stop-daemon") == 0) {
            return stop_daemon();
        }
        if (strcmp(argv[1], "--use-daemon") == 0) {
            /* If --use-daemon is the first arg, we shift args for the client */
            return run_lzo_client(argc - 1, argv + 1);
        }
    }
    /* Default: Standalone mode */
    return run_lzo_standalone(argc, argv);
}
