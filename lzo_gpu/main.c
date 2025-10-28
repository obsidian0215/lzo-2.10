#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static char argv0[256]; // 存储程序名

/*
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
#define OCC_FACTOR        4             /* 每个 CU 期望 OCC_FACTOR 个块，保证高占用 */
#define ALIGN_BYTES       256           /* 向 256 B 对齐，便于内存访问 */

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline uint64_t now_ns(void)
{
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    /*   counter / freq = 秒
     * → counter * 1e9 / freq = 纳秒
     */
    return (uint64_t)counter.QuadPart * (uint64_t)1000000000ULL /
        (uint64_t)freq.QuadPart;
}

#else
#include <time.h>

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

static inline void print_ns(const char* tag, uint64_t ns) {
    printf("%-22s : %8.3f ms\n", tag, ns / 1e6);
}

static void choose_blocking(size_t in_sz, cl_device_id dev,
    size_t* blk_sz_out, size_t* nblk_out)
{
    /* 1. 取 GPU 计算单元数 */
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 2. ⽬标块数：CU × OCC_FACTOR，但不能多于字节数 */
    size_t tgt_blk = (size_t)cu * OCC_FACTOR;
    if (tgt_blk > in_sz) tgt_blk = in_sz;        /* 每块≥1 B */

    /* 3. 初步块⼤⼩ = ceil(in_sz / tgt_blk) */
    size_t blk = (in_sz + tgt_blk - 1) / tgt_blk;

    /* 4. 向 ALIGN_BYTES 对齐，且不得为 0 */
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    /* 5. 重新得到块数；如仍 < CU，则再细分以 nblk = CU 为下限 */
    size_t nblk = (in_sz + blk - 1) / blk;
    if (nblk < cu) {
        nblk = cu;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    }

    /* 6. 尾块太⼩（< blk/4）时，把数据平均回各块 */
    size_t tail = in_sz - blk * (nblk - 1);
    if (nblk > 1 && tail < blk / 4) {
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    }

    *blk_sz_out = blk;
    *nblk_out = (in_sz + blk - 1) / blk;
}

#define CHECK(expr)  do{ cl_int _e=(expr);                       \
        if(_e!=CL_SUCCESS){                                      \
            fprintf(stderr,"OpenCL error %d at %s:%d\n",         \
                    _e,__FILE__,__LINE__); exit(1);} }while(0)

static inline size_t lzo_worst(size_t n) {
    return n + n / 16 + 64 + 3;
}

/* read mem-images/kernel-source */
static char* read_file(const char* path, size_t* sz_out)
{
    FILE* fp;
    char* buf;
    long sz;

    if (strcmp(path, "-") == 0) {
        // 从stdin读取
        fp = stdin;
        // 对于stdin，我们需要动态读取直到EOF
        // 使用一个合理的初始缓冲区大小
        size_t buf_size = 1024 * 1024; // 1MB初始大小
        size_t current_size = 0;
        buf = malloc(buf_size);

        if (!buf) {
            perror("malloc"); exit(1);
        }

        // 逐块读取数据
        size_t bytes_read;
        while ((bytes_read = fread(buf + current_size, 1, 1024, fp)) > 0) {
            current_size += bytes_read;
            // 检查是否需要扩展缓冲区（预留1024字节空间）
            if (current_size + 1024 >= buf_size) {
                // 扩展缓冲区
                size_t new_size = buf_size * 2;
                buf = realloc(buf, new_size);
                if (!buf) {
                    perror("realloc"); exit(1);
                }
                buf_size = new_size;
            }
        }

        sz = (long)current_size;
    } else {
        // 从文件读取
        fp = fopen(path, "rb");
        if (!fp) {
            // 如果相对路径失败，尝试相对于可执行文件的路径
            // 这是为了处理当程序在不同目录运行时的.cl文件查找问题
            char* exec_path = malloc(1024);
            if (exec_path) {
                // 构建相对于可执行文件的路径
                char* exec_dir = malloc(1024);
                if (exec_dir) {
                    // 获取可执行文件目录
                    strcpy(exec_dir, argv0); // argv0是全局变量，存储程序名
                    char* last_slash = strrchr(exec_dir, '/');
                    if (last_slash) {
                        *last_slash = '\0';
                        // 构建新路径：exec_dir/../lzo_gpu/filename
                        char full_path[1024];
                        snprintf(full_path, sizeof(full_path), "%s/../lzo_gpu/%s", exec_dir, path);

                        fp = fopen(full_path, "rb");
                        if (fp) {
                            printf("Found %s at %s\n", path, full_path);
                        }
                        free(exec_dir);
                    }
                }
                free(exec_path);
            }

            if (!fp) {
                perror(path); exit(1);
            }
        }
        fseek(fp, 0, SEEK_END);
        sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        buf = malloc(sz + 1);
        fread(buf, 1, sz, fp);
        fclose(fp);
    }

    if (sz_out)
        *sz_out = (size_t)sz;
    buf[sz] = '\0';
    return buf;
}

static cl_context  ctx;
static cl_command_queue q;
static cl_device_id dev;

/* 程序编译缓存管理 */
#define MAX_CACHE_ENTRIES 16
typedef struct {
    char cl_path[256];
    char kernel_name[256];
    cl_program program;
    int is_loaded;
} program_cache_entry;

static program_cache_entry cache[MAX_CACHE_ENTRIES];
static int cache_count = 0;

static void ocl_init(void)
{
    cl_platform_id pf;
    CHECK(clGetPlatformIDs(1, &pf, NULL));
    CHECK(clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    q = clCreateCommandQueueWithProperties(ctx, dev, props, NULL);
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

/* 查找缓存中的程序 */
static int find_cached_program(const char* cl_path, const char* kernel_name) {
    for (int i = 0; i < cache_count; i++) {
        if (strcmp(cache[i].cl_path, cl_path) == 0 &&
            strcmp(cache[i].kernel_name, kernel_name) == 0) {
            return i;
        }
    }
    return -1;
}

/* 保存程序二进制到文件 */
static void save_program_binary(cl_program program, const char* cl_path) {
    size_t binary_sizes[1];
    unsigned char* binary;

    cl_int err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
        sizeof(size_t), binary_sizes, NULL);
    if (err != CL_SUCCESS) return;

    binary = malloc(binary_sizes[0]);
    if (!binary) return;

    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
        sizeof(unsigned char*), &binary, NULL);
    if (err != CL_SUCCESS) {
        free(binary);
        return;
    }

    /* 创建二进制文件名 - 相对于可执行文件 */
    char binary_path[512];
    char* exec_dir = malloc(1024);
    if (exec_dir) {
        strcpy(exec_dir, argv0);
        char* last_slash = strrchr(exec_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
            snprintf(binary_path, sizeof(binary_path), "%s/../lzo_gpu/%s.bin", exec_dir, cl_path);
        } else {
            snprintf(binary_path, sizeof(binary_path), "%s.bin", cl_path);
        }
        free(exec_dir);
    } else {
        snprintf(binary_path, sizeof(binary_path), "%s.bin", cl_path);
    }

    FILE* fp = fopen(binary_path, "wb");
    if (fp) {
        fwrite(binary, 1, binary_sizes[0], fp);
        fclose(fp);
        // printf("Saved program binary to %s\n", binary_path); // 抑制输出
    }

    free(binary);
}

/* 从文件加载程序二进制 */
static cl_program load_program_binary(const char* cl_path, cl_context context, cl_device_id device) {
    char binary_path[512];

    // 构建相对于可执行文件的路径
    char* exec_dir = malloc(1024);
    if (exec_dir) {
        strcpy(exec_dir, argv0);
        char* last_slash = strrchr(exec_dir, '/');
        if (last_slash) {
            *last_slash = '\0';
            snprintf(binary_path, sizeof(binary_path), "%s/../lzo_gpu/%s.bin", exec_dir, cl_path);
        } else {
            snprintf(binary_path, sizeof(binary_path), "%s.bin", cl_path);
        }
        free(exec_dir);
    } else {
        snprintf(binary_path, sizeof(binary_path), "%s.bin", cl_path);
    }

    FILE* fp = fopen(binary_path, "rb");
    if (!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    size_t binary_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* binary = malloc(binary_size);
    if (!binary) {
        fclose(fp);
        return NULL;
    }

    fread(binary, 1, binary_size, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithBinary(context, 1, &device,
        &binary_size, (const unsigned char**)&binary, NULL, NULL);

    free(binary);

    if (program) {
        cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err == CL_SUCCESS) {
            // printf("Loaded program binary from %s\n", binary_path); // 抑制输出
            return program;
        } else {
            clReleaseProgram(program);
        }
    }

    return NULL;
}

/* 加载或构建程序（带缓存机制） */
static cl_program load_or_build_program(const char* cl_path, const char* kernel_name) {
    /* 先检查缓存 */
    int cache_idx = find_cached_program(cl_path, kernel_name);
    if (cache_idx >= 0) {
        // printf("Using cached program for %s\n", cl_path); // 抑制输出
        return cache[cache_idx].program;
    }

    /* 尝试从文件加载二进制 */
    cl_program program = load_program_binary(cl_path, ctx, dev);
    if (program) {
        /* 保存到缓存 */
        if (cache_count < MAX_CACHE_ENTRIES) {
            strcpy(cache[cache_count].cl_path, cl_path);
            strcpy(cache[cache_count].kernel_name, kernel_name);
            cache[cache_count].program = program;
            cache[cache_count].is_loaded = 1;
            cache_count++;
        }
        return program;
    }

    /* 需要重新编译 */
    // printf("Building program from source: %s\n", cl_path); // 抑制输出
    size_t src_len;
    char* cl_src = read_file(cl_path, &src_len);
    if (!cl_src) {
        fprintf(stderr, "Failed to read %s\n", cl_path);
        exit(1);
    }

    program = clCreateProgramWithSource(ctx, 1, (const char**)&cl_src, &src_len, NULL);
    if (!program) {
        free(cl_src);
        fprintf(stderr, "Failed to create program\n");
        exit(1);
    }

    const char* cl_opts = "-cl-std=CL3.0 -I .";
    cl_int err = clBuildProgram(program, 1, &dev, cl_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        print_buildlog(program, dev);
        clReleaseProgram(program);
        free(cl_src);
        exit(1);
    }

    /* 保存二进制以便下次使用 */
    save_program_binary(program, cl_path);

    /* 保存到缓存 */
    if (cache_count < MAX_CACHE_ENTRIES) {
        strcpy(cache[cache_count].cl_path, cl_path);
        strcpy(cache[cache_count].kernel_name, kernel_name);
        cache[cache_count].program = program;
        cache[cache_count].is_loaded = 0;
        cache_count++;
    }

    free(cl_src);
    return program;
}

int main(int argc, char** argv)
{
    // 保存程序名用于路径解析
    strcpy(argv0, argv[0]);

    int is_stdin = 0;
    int is_stdout = 0;

    if (argc < 2) {
        printf("usage: %s [-1|-2|-3|-4|-d] input_file [output_file]\n", argv[0]);
        printf("  -1: fastest compression (2K dict)\n");
        printf("  -2: fast compression (4K dict)\n");
        printf("  -3/-c: standard compression (16K dict, default)\n");
        printf("  -4: best compression (32K dict)\n");
        printf("  -d: decompress mode\n");
        return 0;
    }

    int mode = 1; // 1=compress (default), 2=decompress
    int compression_level = 3; // 默认标准压缩
    int arg_idx = 1;

    // 检查第一个参数
    if (strcmp(argv[1], "-d") == 0) {
        mode = 2; // decompress mode
        arg_idx = 2;
    } else if (strcmp(argv[1], "-1") == 0) {
        mode = 1; // compress mode
        compression_level = 1;
        arg_idx = 2;
    } else if (strcmp(argv[1], "-2") == 0) {
        mode = 1; // compress mode
        compression_level = 2;
        arg_idx = 2;
    } else if (strcmp(argv[1], "-3") == 0 || strcmp(argv[1], "-c") == 0) {
        mode = 1; // compress mode
        compression_level = 3;
        arg_idx = 2;
    } else if (strcmp(argv[1], "-4") == 0) {
        mode = 1; // compress mode
        compression_level = 4;
        arg_idx = 2;
    } else if (argv[1][0] == '-') {
        // 其他未知参数，丢弃并继续（假设是压缩模式）
        arg_idx = 1;
    }

    const char* in_path = argv[arg_idx];
    const char* out_path = (argc > arg_idx + 1) ? argv[arg_idx + 1] : NULL;

    // 检查是否使用stdin/stdout
    if (strcmp(in_path, "-") == 0) {
        is_stdin = 1;
    }
    if (!out_path) {
        // 如果没有输出参数且输入来自stdin，假设输出到stdout
        if (is_stdin) {
            out_path = "-";
        }
    }
    if (out_path && strcmp(out_path, "-") == 0) {
        is_stdout = 1;
    }

    // 生成默认输出文件名
    if (!out_path) {
        if (is_stdin && !is_stdout) {
            // 输入来自stdin且输出到文件，必须指定输出文件名
            fprintf(stderr, "Error: Input from stdin and output to file, output filename must be specified\n");
            fprintf(stderr, "Usage: %s - [output_file] [compression_level]\n", argv[0]);
            return 1;
        }

        if (is_stdin && is_stdout) {
            // 输入来自stdin且输出到stdout，这是管道模式，设置输出为NULL（使用默认处理）
            out_path = "-";
            is_stdout = 1;
        }

        if (mode == 1) {
            // 压缩模式：输入文件名.lzo
            char* default_out = malloc(strlen(in_path) + 5);
            strcpy(default_out, in_path);
            strcat(default_out, ".lzo");
            out_path = default_out;
        } else {
            // 解压模式：如果输入文件以.lzo结尾，去掉扩展名，否则添加前缀
            char* default_out = malloc(strlen(in_path) + 20);
            if (strlen(in_path) > 4 && strcmp(in_path + strlen(in_path) - 4, ".lzo") == 0) {
                // 去掉.lzo扩展名
                strncpy(default_out, in_path, strlen(in_path) - 4);
                default_out[strlen(in_path) - 4] = '\0';
            } else {
                // 添加decompressed_前缀
                strcpy(default_out, "decompressed_");
                strcat(default_out, in_path);
            }
            out_path = default_out;
        }
    } else if (strcmp(out_path, "-") == 0) {
        is_stdout = 1;
    }

    /* Validate compression level */
    if (compression_level < 1 || compression_level > 4) {
        printf("Invalid compression level: %d (must be 1, 2, 3, or 4)\n", compression_level);
        printf("Using default level 3 (standard)\n");
        compression_level = 3;
    }

    cl_int err;

    if (mode == 1) {
        // 压缩模式
        return compress_data(in_path, out_path, compression_level, is_stdin, is_stdout);
    } else {
        // 解压模式
        return decompress_data(in_path, out_path, is_stdin, is_stdout);
    }
}

int compress_data(const char* in_path, const char* out_path, int compression_level, int is_stdin, int is_stdout)
{
    if (!is_stdout) {
        printf("======== Compress ========\n");
    }
    uint64_t tA0 = now_ns();
    size_t in_sz; unsigned char* in_buf = (unsigned char*)read_file(in_path, &in_sz);
    uint64_t tA1 = now_ns();

    /* 选择合适的内核文件和函数 */
    const char* cl_path;
    const char* kernel_name;

    switch (compression_level) {
        case 1:
            cl_path = "lzo1x_1k.cl";
            kernel_name = "lzo1x_block_compress";
            break;
        case 2:
            cl_path = "lzo1x_1l.cl";
            kernel_name = "lzo1x_block_compress";
            break;
        case 3:
            cl_path = "lzo1x_1.cl";
            kernel_name = "lzo1x_block_compress";
            break;
        case 4:
            cl_path = "lzo1x_1o.cl";
            kernel_name = "lzo1x_block_compress";
            break;
        default:
            cl_path = "lzo1x_1.cl";
            kernel_name = "lzo1x_block_compress";
    }

    /* 2. OpenCL 初始化 & 构建内核 */
    uint64_t tB0 = now_ns();
    ocl_init();

    /* 尝试加载预编译程序，避免重复编译 */
    cl_program prog = load_or_build_program(cl_path, kernel_name);
    cl_int err;
    cl_kernel krn = clCreateKernel(prog, kernel_name, &err); CHECK(err);
    uint64_t tB1 = now_ns();

    //cl_bool gas;
    //clGetDeviceInfo(dev, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
    //    sizeof(gas), &gas, NULL);
    //printf("Generic AS support: %s\n", gas ? "YES" : "NO");

    /* 3. 计算分块数并创建缓冲区 */
    uint64_t tC0 = now_ns();
    //cl_uint blk = BLK_SIZE;
    //size_t nblk = (in_sz + BLK_SIZE - 1) / BLK_SIZE;
    size_t blk, nblk;
    choose_blocking(in_sz, dev, &blk, &nblk);
    if (!is_stdout) {
        printf("\nAuto blocking: blk_sz=%zu , nblk=%zu (CU×%d)\n",
            blk, nblk, OCC_FACTOR);
    }
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;      /* 保证总容量充足 */

    cl_mem c_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        in_sz, in_buf, &err); CHECK(err);
    cl_mem c_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_cap, NULL, &err); CHECK(err);
    cl_mem c_len = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        nblk * sizeof(cl_uint), NULL, &err); CHECK(err);
    uint64_t tC1 = now_ns();

    /* 4. 设置参数并启动kernel */
    CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &c_in));
    CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &c_out));
    CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &c_len));
    CHECK(clSetKernelArg(krn, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(krn, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(krn, 5, sizeof(cl_uint), &worst_blk));

    // 设备本地内存
    //size_t max_lmem;
    //clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,
    //    sizeof(max_lmem), &max_lmem, NULL);

    size_t lsz = 1, gsz = nblk * lsz;
    cl_event evt;
    uint64_t tD0 = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn, 1, NULL, &gsz, &lsz, 0, NULL, &evt));
    clWaitForEvents(1, &evt);
    uint64_t tD1 = now_ns();
    cl_ulong t_kernel_start, t_kernel_end;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
        sizeof(t_kernel_start), &t_kernel_start, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
        sizeof(t_kernel_end), &t_kernel_end, NULL);
    double ck_ns = (double)(t_kernel_end - t_kernel_start);   // ns
    //printf("Compress Kernel execution time: %.3f ms\n", ck_ns / 1e6);
    //CHECK(clFinish(q));

    /* 5. 读取各块长度 */
    uint64_t tE0 = now_ns();
    cl_uint* cblock_len = malloc(nblk * sizeof(cl_uint));
    CHECK(clEnqueueReadBuffer(q, c_len, CL_TRUE, 0, nblk * sizeof(cl_uint), cblock_len, 0, NULL, NULL));

    /* 6. 把压缩结果依次拼到 host 缓冲区 */
    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += cblock_len[i];
    unsigned char* c_buf = malloc(out_sz);

    size_t c_off = 0;
    for (size_t i = 0; i < nblk; ++i) {
        size_t dev_off = i * worst_blk;
        if (dev_off + cblock_len[i] > out_cap) {
            fprintf(stderr, "cblock_len[%zu] overflow !\n", i);
            exit(1);
        }
        CHECK(clEnqueueReadBuffer(q, c_out, CL_TRUE, dev_off, cblock_len[i],
            c_buf + c_off, 0, NULL, NULL));
        c_off += cblock_len[i];
    }
    uint64_t tE1 = now_ns();

    /* 7. 打印性能结果 */
    // puts("\n=== Per-block size ===");
    // for (size_t i = 0; i < nblk; ++i)
    //     printf("block %-3zu -> %7u B\n", i, cblock_len[i]);
    if (!is_stdout) {
        printf("Input %zu B → Output %zu B  (%.2f%%)\n",
            in_sz, out_sz, 100.0 * out_sz / in_sz);

        puts("\n=== Timing summary ===");
        print_ns("A. read input", tA1 - tA0);
        print_ns("B. build program", tB1 - tB0);
        print_ns("C. create+upload", tC1 - tC0);
        print_ns("D. enqueue+wait", tD1 - tD0);     /* host 视角 */
        print_ns("|- device kernel", ck_ns);     /* 纯 device */
        print_ns("E. download result", tE1 - tE0);
        print_ns("Total (A→E)", tE1 - tA0);

        /* 吞吐量 */
        double mb_in = (double)in_sz / 1e6;
        double mb_out = (double)out_sz / 1e6;
        double gpu_mbps = mb_in / (ck_ns * 1e-9);   /* MB / s */
        printf("\nInput %zu B -> Output %zu B (%.2f%%, %.2f:1)\n",
            in_sz, out_sz, 100.0 * out_sz / in_sz, (double)in_sz / out_sz);
        printf("GPU Compress throughput : %.2f MB/s  (%.2f GiB/s)\n",
            gpu_mbps, gpu_mbps / 1024.0);
    }

    /* 8. 写压缩文件 */
    FILE* fo;
    if (is_stdout) {
        fo = stdout;
    } else {
        fo = fopen(out_path, "wb");
        if (!fo) {
            perror(out_path); return 1;
        }
    }

    uint16_t u16 = MAGIC;             fwrite(&u16, 2, 1, fo);   // 魔数
    uint32_t u32 = (uint32_t)in_sz;   fwrite(&u32, 4, 1, fo);   // 原始大小
    u32 = (uint32_t)blk;              fwrite(&u32, 4, 1, fo);   // 分块大小
    u32 = (uint32_t)nblk;             fwrite(&u32, 4, 1, fo);   // 分块数
    fwrite(cblock_len, 4, nblk, fo);                               // 每块压缩长度
    fwrite(c_buf, 1, out_sz, fo);                                  // 压缩数据

    if (!is_stdout) {
        fclose(fo);
    }

    /* 资源释放 */
    free(in_buf); free(c_buf); free(cblock_len);
    clReleaseMemObject(c_in); clReleaseMemObject(c_out); clReleaseMemObject(c_len);
    clReleaseKernel(krn); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);

    return 0;
}

int decompress_data(const char* in_path, const char* out_path, int is_stdin, int is_stdout)
{
    if (!is_stdout) {
        printf("======== Decompress ========\n");
    }

    /* 1. 读取压缩文件 */
    uint64_t tA0 = now_ns();
    size_t lz_sz; unsigned char* lz_buf = (unsigned char*)read_file(in_path, &lz_sz);
    uint64_t tA1 = now_ns();

    const unsigned char* p = lz_buf;
    uint16_t magic = *(uint16_t*)p; p += 2;
    if (magic != MAGIC) {
        fprintf(stderr, "bad magic: expected 0x%04x, got 0x%04x\n", MAGIC, magic);
        free(lz_buf);
        return 1;
    }

    uint32_t orig_sz = *(uint32_t*)p;   p += 4;
    uint32_t blk_sz = *(uint32_t*)p;   p += 4;
    uint32_t nblk = *(uint32_t*)p;   p += 4;

    uint32_t* dblock_len = (uint32_t*)p;   p += 4 * nblk;
    size_t comp_sz = lz_sz - (p - lz_buf);   /* 全部压缩数据字节数 */

    /* 计算每块在压缩流中的起始偏移 */
    uint32_t* block_offset = malloc((nblk + 1) * 4);
    block_offset[0] = 0;
    for (uint32_t i = 0; i < nblk; ++i)
        block_offset[i + 1] = block_offset[i] + dblock_len[i];

    /* 2. OpenCL 初始化 & 构建内核 */
    uint64_t tB0 = now_ns();
    ocl_init();
    cl_program prog = load_or_build_program("lzo1x_1.cl", "lzo1x_block_decompress");
    cl_int err;
    cl_kernel dkrn = clCreateKernel(prog, "lzo1x_block_decompress", &err); CHECK(err);
    uint64_t tB1 = now_ns();

    /* 3. 创建OpenCL缓冲区+设置kernel参数 */
    uint64_t tC0 = now_ns();
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        comp_sz, (void*)p, &err); CHECK(err);
    cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (nblk + 1) * 4, block_offset, &err); CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        orig_sz, NULL, &err); CHECK(err);
    cl_mem d_olen = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

    CHECK(clSetKernelArg(dkrn, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(dkrn, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(dkrn, 2, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(dkrn, 3, sizeof(cl_mem), &d_olen));
    CHECK(clSetKernelArg(dkrn, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(dkrn, 5, sizeof(cl_uint), &orig_sz));
    uint64_t tC1 = now_ns();

    /* 4. 调度执行Decompress Kernel */
    uint64_t tD0 = now_ns();
    size_t gsz = nblk, lsz = 1;
    cl_event evt_d;
    CHECK(clEnqueueNDRangeKernel(q, dkrn, 1, NULL, &gsz, &lsz, 0, NULL, &evt_d));
    clWaitForEvents(1, &evt_d);
    uint64_t tD1 = now_ns();

    cl_ulong d_kernel_start, d_kernel_end;
    clGetEventProfilingInfo(evt_d, CL_PROFILING_COMMAND_START,
        sizeof(d_kernel_start), &d_kernel_start, NULL);
    clGetEventProfilingInfo(evt_d, CL_PROFILING_COMMAND_END,
        sizeof(d_kernel_end), &d_kernel_end, NULL);
    double dk_ns = (double)(d_kernel_end - d_kernel_start);

    /* 5. 取回解压结果 */
    uint64_t tE0 = now_ns();
    unsigned char* dec_buf = malloc(orig_sz);
    CHECK(clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, orig_sz, dec_buf, 0, NULL, NULL));
    cl_uint* dec_len = malloc(nblk * sizeof(cl_uint));
    CHECK(clEnqueueReadBuffer(q, d_olen, CL_TRUE, 0,
        nblk * sizeof(cl_uint), dec_len, 0, NULL, NULL));
    uint64_t tE1 = now_ns();

    /* 6. 校验解压结果 */
    uint64_t tF0 = now_ns();
    size_t sum_len = 0;
    for (uint32_t i = 0; i < nblk; ++i) sum_len += dec_len[i];

    if (sum_len != orig_sz) {
        fprintf(stderr, "[ERR] sum(out_len_arr) = %zu , but orig_sz = %u\n",
            sum_len, orig_sz);
        free(lz_buf); free(dec_buf); free(dec_len); free(block_offset);
        return 1;
    }

    if (!is_stdout) {
        puts("verify OK");
    }
    uint64_t tF1 = now_ns();

    /* 7. 输出性能统计 */
    if (!is_stdout) {
        puts("\n=== Timing summary ===");
        print_ns("A. read input", tA1 - tA0);
        print_ns("B. build program", tB1 - tB0);
        print_ns("C. create+upload", tC1 - tC0);
        print_ns("D. enqueue+wait", tD1 - tD0);
        print_ns("|- device kernel", dk_ns);
        print_ns("E. download result", tE1 - tE0);
        print_ns("F. verify", tF1 - tF0);
        print_ns("Total (A→F)", tF1 - tA0);

        double dk_ms = dk_ns / 1e6;
        double orig_mb = (double)orig_sz / 1e6;
        printf("GPU Decompress throughput: %.2f MB/s (%.3f ms for %u B)\n",
            orig_mb / (dk_ms / 1000.0), dk_ms, orig_sz);
    }

    /* 8. 写解压文件 */
    FILE* fo;
    if (is_stdout) {
        fo = stdout;
    } else {
        fo = fopen(out_path, "wb");
        if (!fo) {
            perror(out_path);
            free(lz_buf); free(dec_buf); free(dec_len); free(block_offset);
            return 1;
        }
    }

    fwrite(dec_buf, 1, orig_sz, fo);

    if (!is_stdout) {
        fclose(fo);
    }

    /* 清理资源 */
    free(lz_buf); free(dec_buf); free(dec_len); free(block_offset);
    clReleaseMemObject(d_in); clReleaseMemObject(d_off);
    clReleaseMemObject(d_out); clReleaseMemObject(d_olen);
    clReleaseKernel(dkrn); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);

    return 0;
}
