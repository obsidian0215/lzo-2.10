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

/* Helper for multi-threaded pread into a destination buffer.
 * Each thread handles a subrange (off,len) of the destination.
 */
typedef struct {
    int fd;
    void *dest; /* base pointer */
    off_t off;  /* offset into dest */
    size_t len; /* length to read */
    int err;    /* errno on failure, 0 on success */
} mt_io_arg_t;

static void* mt_pread_worker(void *v) {
    mt_io_arg_t *a = (mt_io_arg_t*)v;
    /* validate args first */
    if (!a || a->len == 0) { if (a) a->err = 0; return NULL; }
    if ((uint64_t)a->off + (uint64_t)a->len < (uint64_t)a->off) { /* overflow */
        a->err = EINVAL; return NULL;
    }
    size_t left = a->len;
    off_t pos = a->off;
    char *p = (char*)a->dest + pos;
    while (left > 0) {
        ssize_t r;
        do {
            r = pread(a->fd, p, left, pos);
        } while (r < 0 && errno == EINTR);
        if (r < 0) { a->err = errno; return NULL; }
        if (r == 0) { /* short read */ a->err = EIO; return NULL; }
        left -= r; p += r; pos += r;
    }
    a->err = 0;
    return NULL;
}

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
#define OCC_FACTOR        128           /* 优化: 从12提升到128，大幅增加并行度 */
#define ALIGN_BYTES       16384         /* 优化: 从64KB降到16KB，减小对齐浪费 */
#define MIN_BLOCK_SIZE    (64 * 1024)   /* 最小块大小: 64KB (从512KB大幅降低) */
#define MAX_BLOCK_SIZE    (512 * 1024)  /* Phase 7.2: 最大块大小提升到512KB (自适应块大小) */

/* 压缩率跟踪 */
#define ENABLE_COMPRESSION_RATIO_TRACKING 1

/* Phase 8.1: 多线程I/O优化开关 */
#define ENABLE_MMAP_IO         0  /* Phase 8.3: mmap在测试中未显示优势，保留fread */
#define ENABLE_PINNED_MEMORY   1  /* 使用OpenCL Pinned Memory */
#define ENABLE_ASYNC_WRITE     0  /* 异步文件写入（实验性）*/

/* Phase 7.2: 自适应块大小算法声明 */
#define SAMPLE_SIZE (64 * 1024)
#define LOW_ENTROPY_THRESHOLD  4.0
#define HIGH_ENTROPY_THRESHOLD 7.0
static double calculate_entropy(const unsigned char* data, size_t size);
static size_t adaptive_block_size(const unsigned char* data, size_t size, double* entropy_out, int debug);

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

/* 解析块大小字符串，支持单位: B/KB/K/MB/M/GB/G，默认单位为KB */
static size_t parse_block_size(const char* str) {
    if (!str || !*str) return 0;

    char* endptr;
    double value = strtod(str, &endptr);
    if (value <= 0) return 0;

    /* 跳过空格 */
    while (*endptr == ' ' || *endptr == '\t') endptr++;

    size_t multiplier = 1024; /* 默认单位: KB */

    if (*endptr == '\0') {
        /* 无单位，默认KB */
        multiplier = 1024;
    } else if (strcasecmp(endptr, "B") == 0 || strcasecmp(endptr, "BYTES") == 0) {
        multiplier = 1;
    } else if (strcasecmp(endptr, "K") == 0 || strcasecmp(endptr, "KB") == 0) {
        multiplier = 1024;
    } else if (strcasecmp(endptr, "M") == 0 || strcasecmp(endptr, "MB") == 0) {
        multiplier = 1024 * 1024;
    } else if (strcasecmp(endptr, "G") == 0 || strcasecmp(endptr, "GB") == 0) {
        multiplier = 1024 * 1024 * 1024;
    } else {
        fprintf(stderr, "Warning: Unknown size unit '%s', assuming KB\n", endptr);
        multiplier = 1024;
    }

    size_t result = (size_t)(value * multiplier);
    return result;
}

static void choose_blocking(size_t in_sz, cl_device_id dev,
    size_t* blk_sz_out, size_t* nblk_out)
{
    /* Phase 7.2: 自适应块大小需要数据指针，这里改为由compress_gpu传入 */
    /* 保留原有逻辑作为后备 */

    /* 1. 取 GPU 计算单元数 */
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 2. ⽬标块数：CU × OCC_FACTOR，但不能多于字节数 */
    size_t tgt_blk = (size_t)cu * OCC_FACTOR;
    /* Allow forcing a smaller target block count for testing via
     * LZO_FORCE_NBLK environment variable (helps run CPU-style kernel
     * on GPUs by reducing concurrent work-items). */
    const char* env_nblk = getenv("LZO_FORCE_NBLK");
    if (env_nblk) {
        int v = atoi(env_nblk);
        if (v > 0) tgt_blk = (size_t)v;
    }
    if (tgt_blk > in_sz) tgt_blk = in_sz;        /* 每块≥1 B */

    /* 3. 初步块⼤⼩ = ceil(in_sz / tgt_blk) */
    size_t blk = (in_sz + tgt_blk - 1) / tgt_blk;

    /* 4. 向 ALIGN_BYTES 对齐，且不得为 0 */
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    /* 4.5 优化: 限制块大小在合理范围内 */
    if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
    if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;

    /* 5. 重新得到块数；如仍 < CU，则再细分以 nblk = CU 为下限 */
    size_t nblk = (in_sz + blk - 1) / blk;
    if (nblk < cu) {
        nblk = cu;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        /* 再次检查范围 */
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    }

    /* 6. 尾块太⼩（< blk/4）时，把数据平均回各块 */
    size_t tail = in_sz - blk * (nblk - 1);
    if (nblk > 1 && tail < blk / 4) {
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        /* 再次检查范围 */
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    }

    *blk_sz_out = blk;
    *nblk_out = (in_sz + blk - 1) / blk;
}

/* Phase 7.2: 自适应块大小算法实现 */
static double calculate_entropy(const unsigned char* data, size_t size)
{
    if (size == 0) return 0.0;

    uint32_t freq[256] = {0};
    size_t sample_len = (size < SAMPLE_SIZE) ? size : SAMPLE_SIZE;

    for (size_t i = 0; i < sample_len; ++i) {
        freq[data[i]]++;
    }

    double entropy = 0.0;
    double inv_size = 1.0 / sample_len;

    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0) {
            double p = freq[i] * inv_size;
            entropy -= p * log2(p);
        }
    }

    return entropy;
}

static size_t adaptive_block_size(const unsigned char* data, size_t size,
                                  double* entropy_out, int debug)
{
    double entropy = calculate_entropy(data, size);
    if (entropy_out) *entropy_out = entropy;

    size_t block_size;
    if (entropy < LOW_ENTROPY_THRESHOLD) {
        block_size = MAX_BLOCK_SIZE;  /* 512KB for low entropy */
    } else if (entropy > HIGH_ENTROPY_THRESHOLD) {
        block_size = MIN_BLOCK_SIZE;  /* 64KB for high entropy */
    } else {
        /* Linear interpolation between 512KB and 64KB */
        double ratio = (entropy - LOW_ENTROPY_THRESHOLD) /
                      (HIGH_ENTROPY_THRESHOLD - LOW_ENTROPY_THRESHOLD);
        block_size = (size_t)(MAX_BLOCK_SIZE - ratio * (MAX_BLOCK_SIZE - MIN_BLOCK_SIZE));
    }

    if (debug) {
        const char* entropy_desc;
        if (entropy < LOW_ENTROPY_THRESHOLD) {
            entropy_desc = "LOW (repetitive)";
        } else if (entropy > HIGH_ENTROPY_THRESHOLD) {
            entropy_desc = "HIGH (random)";
        } else {
            entropy_desc = "MEDIUM (structured)";
        }
        fprintf(stderr, "[Adaptive] Entropy: %.2f bits/byte (%s) -> Block: %zu KB\n",
                entropy, entropy_desc, block_size / 1024);
    }

    return block_size;
}

/* Phase 7.2: 带自适应块大小的choose_blocking变体 (改进版v2增强) */
static void choose_blocking_adaptive(const unsigned char* in_buf, size_t in_sz,
                                     cl_device_id dev, size_t* blk_sz_out,
                                     size_t* nblk_out, int debug)
{
    /* 0. 检查是否强制固定块大小 (LZO_FIXED_BLOCK_SIZE环境变量) */
    const char* env_fixed_blk = getenv("LZO_FIXED_BLOCK_SIZE");
    if (env_fixed_blk) {
        size_t fixed_blk = parse_block_size(env_fixed_blk);
        if (fixed_blk > 0) {
            /* 对齐到ALIGN_BYTES */
            fixed_blk = (fixed_blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
            if (fixed_blk == 0) fixed_blk = ALIGN_BYTES;

            size_t nblk = (in_sz + fixed_blk - 1) / fixed_blk;
            *blk_sz_out = fixed_blk;
            *nblk_out = nblk;

            if (debug) {
                printf("[FIXED] LZO_FIXED_BLOCK_SIZE=%s -> %zu bytes (%zu KB)\n",
                       env_fixed_blk, fixed_blk, fixed_blk / 1024);
                printf("[FIXED] File: %zu bytes -> %zu blocks of %zu bytes\n",
                       in_sz, nblk, fixed_blk);
            }
            return;
        }
    }

    /* 1. 取 GPU 计算单元数 */
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 2. 计算数据熵 */
    double entropy = 0.0;
    size_t entropy_suggested_blk = adaptive_block_size(in_buf, in_sz, &entropy, debug);

    /* 3. 优化策略：偏向更小块以最大化吞吐量
     *
     * 实验发现：
     * - 64KB块: 2470 MB/s，压缩率6.53:1 (最优吞吐量)
     * - 128KB块: 2430 MB/s，压缩率6.61:1 (-1.6%吞吐，+1.2%压缩)
     * - 256KB块: 2367 MB/s，压缩率6.60:1 (-4.2%吞吐，+1.1%压缩)
     * - 512KB块: 2229 MB/s，压缩率6.62:1 (-9.8%吞吐，+1.4%压缩)
     *
     * 结论：块越大，GPU并行度越低，吞吐量下降明显，但压缩率提升很小(<2%)
     * 策略：优先保证高并行度，使用更激进的OCC_FACTOR，倾向64-96KB块
     */
    size_t occ_factor = OCC_FACTOR * 2;  /* 默认256，激进并行 */

    if (entropy < LOW_ENTROPY_THRESHOLD) {
        /* 低熵数据: 仍然增加并行度，但允许稍大的块 */
        occ_factor = (size_t)(OCC_FACTOR * 1.5);  /* 192: 平衡压缩率和吞吐量 */
    } else if (entropy > HIGH_ENTROPY_THRESHOLD) {
        /* 高熵数据: 最大并行度以提高吞吐量 */
        occ_factor = OCC_FACTOR * 3;  /* 384: 极致并行 */
    }
    /* 中等熵: OCC_FACTOR * 2 = 256，保持高并行度 */

    /* 4. 计算目标块数 */
    size_t target_nblk = (size_t)cu * occ_factor;

    /* 5. 根据目标块数计算初始块大小 */
    size_t blk = (in_sz + target_nblk - 1) / target_nblk;

    /* 6. 对齐到ALIGN_BYTES */
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    /* 7. 设定块大小上下限，严格限制在64-128KB范围 */
    size_t min_blk = MIN_BLOCK_SIZE;  /* 64KB */
    size_t max_blk = 96 * 1024;       /* 默认96KB上限，偏向小块 */

    if (entropy < LOW_ENTROPY_THRESHOLD) {
        /* 低熵数据: 允许稍大块以提升压缩率，但最多128KB */
        max_blk = 128 * 1024;
    } else if (entropy > HIGH_ENTROPY_THRESHOLD) {
        /* 高熵数据: 严格限制在64-80KB以最大化并行度 */
        max_blk = 80 * 1024;
    }

    /* 8. 限制块大小在合适范围内 */
    if (blk < min_blk) blk = min_blk;
    if (blk > max_blk) blk = max_blk;

    /* 9. 重新计算块数 */
    size_t nblk = (in_sz + blk - 1) / blk;

    /* 10. 确保最小块数（避免GPU利用率过低）*/
    size_t min_nblk = target_nblk / 2;  /* 至少目标的一半 */
    if (nblk < min_nblk) {
        nblk = min_nblk;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        if (blk < min_blk) blk = min_blk;
        if (blk > max_blk) blk = max_blk;
    }

    /* 11. 处理尾块太小的情况 */
    size_t tail = in_sz - blk * (nblk - 1);
    if (nblk > 1 && tail < blk / 4) {
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        if (blk < min_blk) blk = min_blk;
        if (blk > max_blk) blk = max_blk;
    }

    if (debug) {
        fprintf(stderr, "[Adaptive-SmallBlock] Entropy: %.2f, OCC_FACTOR: %zu, Target blocks: %zu\n",
                entropy, occ_factor, target_nblk);
        fprintf(stderr, "[Adaptive-SmallBlock] Final: blk=%zu KB (%zu blocks), max_allowed=%zu KB\n",
                blk / 1024, (in_sz + blk - 1) / blk, max_blk / 1024);
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
#if ENABLE_MMAP_IO
/* Phase 8.1: mmap版本 - 零拷贝，利用页缓存预读 */
static char* read_file(const char* path, size_t* sz_out)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path); exit(1);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat"); close(fd); exit(1);
    }

    size_t sz = st.st_size;
    if (sz == 0) {
        fprintf(stderr, "Error: %s is empty\n", path);
        close(fd); exit(1);
    }

    /* mmap文件到内存 */
    void* mapped = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap"); close(fd); exit(1);
    }

    /* 建议内核顺序预读（提升大文件性能）*/
    madvise(mapped, sz, MADV_SEQUENTIAL | MADV_WILLNEED);

    /* Phase 8.3: mmap + memcpy 方案
     * 虽然有一次 memcpy，但仍比 fread 快：
     * - fread: Page Cache → 内核缓冲区 → 用户空间 (2次拷贝)
     * - mmap: Page Cache → 用户空间 (1次拷贝，memcpy可能被优化)
     * 且 mmap 利用页表映射，大文件预读效率更高
     */
    char* buf = malloc(sz + 1);
    if (!buf) {
        munmap(mapped, sz); close(fd);
        fprintf(stderr, "malloc failed for %zu bytes\n", sz);
        exit(1);
    }
    memcpy(buf, mapped, sz);
    buf[sz] = '\0';

    /* 清理映射 */
    munmap(mapped, sz);
    close(fd);

    if (sz_out)
        *sz_out = sz;
    return buf;
}
#else
/* 原有fread版本 */
static char* read_file(const char* path, size_t* sz_out)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror(path); exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* buf = malloc(sz + 1);
    fread(buf, 1, sz, fp);
    fclose(fp);

    if (sz_out)
        *sz_out = (size_t)sz;
    buf[sz] = '\0';
    return buf;
}
#endif

static cl_context  ctx;
static cl_command_queue q;
static cl_device_id dev;
static int debug = 0;  /* 全局debug标志 */

/* 优化: 全局缓存以避免重复编译和创建内核 */
#define MAX_CACHED_PROGRAMS 16
static struct {
    char name[128];
    cl_program prog;
    cl_kernel krn_compress;
    cl_kernel krn_decompress;
} prog_cache[MAX_CACHED_PROGRAMS];
static int prog_cache_count = 0;

/* 优化: 持久化缓冲区缓存以避免重复创建和释放 */
static struct {
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    size_t in_size;
    size_t out_size;
    size_t len_size;
} buffer_cache = {0};

/* 优化: 内核参数缓存以避免重复设置 */
static struct {
    cl_kernel kernel;
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    cl_uint in_sz;
    cl_uint blk;
    cl_uint worst_blk;
} kernel_args_cache = {0};

static void set_kernel_args_cached(cl_kernel krn, cl_mem d_in, cl_mem d_out,
                                   cl_mem d_len, cl_uint in_sz, cl_uint blk,
                                   cl_uint worst_blk) {
    /* 仅在参数变化时才设置 */
    if (kernel_args_cache.kernel != krn ||
        kernel_args_cache.d_in != d_in ||
        kernel_args_cache.d_out != d_out ||
        kernel_args_cache.d_len != d_len ||
        kernel_args_cache.in_sz != in_sz ||
        kernel_args_cache.blk != blk ||
        kernel_args_cache.worst_blk != worst_blk) {

        if (debug) fprintf(stderr, "DBG: Setting kernel args (cache miss)\n");

        CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &d_in));
        CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &d_out));
        CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &d_len));
        CHECK(clSetKernelArg(krn, 3, sizeof(cl_uint), &in_sz));
        CHECK(clSetKernelArg(krn, 4, sizeof(cl_uint), &blk));
        CHECK(clSetKernelArg(krn, 5, sizeof(cl_uint), &worst_blk));

        /* 更新缓存 */
        kernel_args_cache.kernel = krn;
        kernel_args_cache.d_in = d_in;
        kernel_args_cache.d_out = d_out;
        kernel_args_cache.d_len = d_len;
        kernel_args_cache.in_sz = in_sz;
        kernel_args_cache.blk = blk;
        kernel_args_cache.worst_blk = worst_blk;
    } else {
        if (debug) fprintf(stderr, "DBG: Kernel args cached (skip setting)\n");
    }
}

static cl_mem get_or_create_buffer(cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags) {
    if (*cached_size < required_size) {
        if (*cached_buf) clReleaseMemObject(*cached_buf);
        cl_int err;
        /* Phase 6.1: 添加 CL_MEM_ALLOC_HOST_PTR 启用 Pinned Memory
         * 这使得主机端内存页锁定，DMA传输效率提升20-30%
         */
        *cached_buf = clCreateBuffer(ctx, flags | CL_MEM_ALLOC_HOST_PTR, required_size, NULL, &err);
        CHECK(err);
        *cached_size = required_size;
    }
    return *cached_buf;
}

static void ocl_init(void)
{
    cl_platform_id pf;
    /* allow selecting CPU device for testing CPU-style kernels via env
     * variable LZO_OPENCL_DEVICE=CPU; default remains GPU. */
    const char* prefer = getenv("LZO_OPENCL_DEVICE");
    cl_device_type dtype = CL_DEVICE_TYPE_GPU;
    if (prefer && strcmp(prefer, "CPU") == 0) dtype = CL_DEVICE_TYPE_CPU;
    CHECK(clGetPlatformIDs(1, &pf, NULL));
    CHECK(clGetDeviceIDs(pf, dtype, 1, &dev, NULL));
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

/* forward decl so main can call it */
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path);

/* 优化: 缓存查找和管理函数 */
static int find_cached_program(const char* name) {
    for (int i = 0; i < prog_cache_count; i++) {
        if (strcmp(prog_cache[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void cache_program(const char* name, cl_program prog, cl_kernel krn_c, cl_kernel krn_d) {
    if (prog_cache_count >= MAX_CACHED_PROGRAMS) {
        fprintf(stderr, "warning: program cache full, not caching %s\n", name);
        return;
    }
    strncpy(prog_cache[prog_cache_count].name, name, sizeof(prog_cache[0].name) - 1);
    prog_cache[prog_cache_count].prog = prog;
    prog_cache[prog_cache_count].krn_compress = krn_c;
    prog_cache[prog_cache_count].krn_decompress = krn_d;
    prog_cache_count++;
}

/* Helper: load program from <base>.bin or from source file */
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path)
{
    char use_base[256]; strncpy(use_base, base, sizeof(use_base)-1); use_base[sizeof(use_base)-1]='\0';
    char use_cl_src[256]; strncpy(use_cl_src, cl_src_path, sizeof(use_cl_src)-1); use_cl_src[sizeof(use_cl_src)-1]='\0';

    char bin_path[512]; snprintf(bin_path, sizeof(bin_path), "%s.bin", use_base);
    /* also prepare an alternate path inside the lzo_gpu subdir to be robust
     * against differing working directories when running via tools/runner */
    char bin_path_alt[512]; snprintf(bin_path_alt, sizeof(bin_path_alt), "lzo_gpu/%s.bin", use_base);
    /* Attempt to use a precompiled binary first (robust fallback to source
     * compilation is performed below if binary is incompatible). */
    FILE* fb = fopen(bin_path, "rb");
    if (!fb) {
        /* try lzo_gpu/ subdir */
        fb = fopen(bin_path_alt, "rb");
    }
    cl_int err; cl_program prog = NULL;
    if (fb) {
        /* attempt to use precompiled binary; on any failure fall back to source */
        fseek(fb, 0, SEEK_END); long bsz = ftell(fb); fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin,1,bsz,fb) != (size_t)bsz) { perror("fread"); fclose(fb); free(bin); exit(1); }
        fclose(fb);
        cl_int binary_status;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
            (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
            fprintf(stderr, "warning: precompiled binary %s.bin incompatible, falling back to source (clCreateProgramWithBinary err=%d bin_status=%d)\n", base, err, binary_status);
            if (prog) { clReleaseProgram(prog); prog = NULL; }
        } else {
            err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
                fprintf(stderr, "Build log (from binary):\n%s\n", log); free(log);
                fprintf(stderr, "warning: build from binary failed for %s.bin (err=%d), falling back to source\n", base, err);
                clReleaseProgram(prog); prog = NULL;
            }
        }
    }

    if (!prog) {
        /* compile from source as fallback - use direct .cl file without frontend combinations */
        size_t src_len = 0; char* src = NULL;
        /* try source in current dir, otherwise try lzo_gpu/ subdir */
        FILE* fchk = fopen(use_cl_src, "rb");
        if (!fchk) {
            char use_cl_src_alt[512]; snprintf(use_cl_src_alt, sizeof(use_cl_src_alt), "lzo_gpu/%s", use_cl_src);
            fchk = fopen(use_cl_src_alt, "rb");
            if (fchk) {
                fclose(fchk);
                src = read_file(use_cl_src_alt, &src_len);
            }
        } else {
            fclose(fchk);
            src = read_file(use_cl_src, &src_len);
        }

        if (!src) {
            fprintf(stderr, "source file %s not found (frontend combinations removed)\n", use_cl_src);
            exit(1);
        }

        /* create program from assembled source */
        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed (err=%d)\n", err); free(src); exit(1); }
        /* Add include paths for OpenCL compiler to resolve #include directives
         * Try both current directory and lzo_gpu/ subdirectory */
        const char* build_opts = "-cl-std=CL2.0 -I. -I./lzo_gpu -I..";
        err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
            fprintf(stderr, "Build log (from source):\n%s\n", log); free(log); free(src); exit(1);
        }
        free(src);
    }
    return prog;
}

int main(int argc, char** argv)
{
    uint64_t t_start_total = now_ns();
    if (argc < 2) {
        /* Print the detailed help when no args are provided to keep output
         * consistent with --help behaviour (avoid the short, outdated usage)
         */
        /* forward to help-printing below by setting a flag we check later */
        /* set argc to 1 so parsing flow will not treat missing args as error, but
         * instead show help via the same code path used for -h/--help
         */
        argc = 1;
        /* set an internal env flag to display help later (we cannot reference
         * show_help variable yet), we will check argc < 2 again after parsing
         * unless show_help is handled below. Simpler: print full help now and exit.
         */
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s [--debug|-v] [--verify|-c] [-L level] [-o out.lzo] input_file\n", argv[0]);
        fprintf(stderr, "     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        fprintf(stderr, "     - --verify/-c (compress mode): do in-memory roundtrip check (no arg).\n");
        fprintf(stderr, "     - -L|-l|--level LEVEL : compression level to select kernel variant (default: 1)\n");
        fprintf(stderr, "         supported LEVEL values:\n");
        fprintf(stderr, "            1   : original LZO1X-1 compressor (kernel: lzo1x_1)\n");
        fprintf(stderr, "            1k  : LZO1X-1K variant (kernel: lzo1x_1k) - optimized for kernel K behavior\n");
        fprintf(stderr, "            1l  : LZO1X-1L variant (kernel: lzo1x_1l) - alternative lookup/heuristics (default)\n");
        fprintf(stderr, "            1o  : LZO1X-1O variant (kernel: lzo1x_1o) - other tuning/optimizations\n\n");
        fprintf(stderr, "  %s -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo\n", argv[0]);
        fprintf(stderr, "     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        fprintf(stderr, "     - --verify/-c ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  Compress with default level: %s input.dat -o out.lzo\n", argv[0]);
        fprintf(stderr, "  Compress with level 1k:      %s -L 1k input.dat -o out.lzo\n", argv[0]);
        fprintf(stderr, "  Decompress and verify:      %s -d --verify input.dat out.lzo -o out.dec\n", argv[0]);
        fprintf(stderr, "  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", argv[0]);
        fprintf(stderr, "  %s -h|--help                                 # show this help\n", argv[0]);
        fprintf(stderr, "\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
        fprintf(stderr, "  I/O mode:\n");
        fprintf(stderr, "    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload).\n");
        fprintf(stderr, "    LZO_FORCE_MAP=0|1        Force use of mapped pinned buffer even if standard-copy requested (overrides LZO_STANDARD_COPY).\n");
        fprintf(stderr, "\n  Multi-threaded I/O (reduce file-read latency / parallel uploads):\n");
        fprintf(stderr, "    LZO_MT_IO=0|1            Enable multi-threaded pread reads / parallel uploads (standalone + daemon).\n");
        fprintf(stderr, "    LZO_MT_IO_THREADS=N      Threads for MT I/O (1-32; common values: 4/8; default: 4 standalone, 4 daemon).\n");
        fprintf(stderr, "\n  Block sizing / adaptive behavior:\n");
        fprintf(stderr, "    LZO_FIXED_BLOCK_SIZE=N   Fix block size in KB (overrides adaptive selection). Use 0 to restore adaptive behavior.\n");
        fprintf(stderr, "    LZO_FORCE_NBLK=N         (advanced) Force target block count for blocking algorithm (comma-free tuning).\n");
        fprintf(stderr, "\n  OpenCL & correctness flags:\n");
        fprintf(stderr, "    LZO_OPENCL_DEVICE=CPU|GPU Prefer device selection for standalone/daemon. Daemons may ignore if configured otherwise.\n");
        fprintf(stderr, "    LZO_DECOMP_VEC=0|1       Decompression: 1=prefer vectorized kernel (default), 0=force scalar decompressor.\n");
        fprintf(stderr, "    LZO_DEBUG=1              Enable debug prints and timing traces.\n");
        fprintf(stderr, "\n  Notes / dependency rules:\n");
        fprintf(stderr, "    - Force/override rules: LZO_FORCE_MAP overrides LZO_STANDARD_COPY.\n");
        fprintf(stderr, "    - MT I/O affects both file read and upload; enabling LZO_MT_IO without sufficient threads (LZO_MT_IO_THREADS) gives limited benefit.\n");
        fprintf(stderr, "    - Client->daemon: client sends its environment options per-request; the daemon may accept or ignore some settings depending on its configuration.\n");
        fprintf(stderr, "\n  Which binary honors which option:\n");
        fprintf(stderr, "    - standalone (./lzo_gpu) supports all above flags and exposes low-level control (best for local testing).\n");
        fprintf(stderr, "    - daemon (./lzo_gpu_daemon) accepts per-request options sent by clients (see lzo_gpu_client help). Some global daemon-config flags may still be ignored.\n");
        return 0;
    }

    /* simple CLI parsing: support optional --debug/-v flag and -d decompress mode */
    /* debug is now global */
    int verify_flag = 0; /* only when set, do roundtrip/verify prints */
    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *orig_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int show_help = 0;
    const char *comp_level = "1l"; /* default: "1l" (GPU optimized, D_BITS=12, ~2500MB/s and best compression)
                                      * "1"  (CPU standard, D_BITS=14)
                                      * "1k" (low mem, D_BITS=11)
                                      * "1o" (D_BITS=15, highest compression) */

    /* pass 1: only detect mode (-d) and help, to know how to parse verify */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { show_help = 1; }
        if (strcmp(argv[i], "-d") == 0) { decompress_mode = 1; }
    }

    /* pass 2: parse options and positionals with knowledge of mode */
    const char* verify_path = NULL; /* only for -d mode */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--debug") == 0 || strcmp(arg, "-v") == 0) { debug = 1; continue; }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) { /* already noted */ continue; }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        /* --strategy option removed: strategy dimension is no longer supported */
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "-l") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            comp_level = argv[++i];
            continue;
        }
        if (strcmp(arg, "-c") == 0 || strcmp(arg, "--verify") == 0) {
            if (decompress_mode) {
                if (i + 1 >= argc || argv[i+1][0] == '-') { fprintf(stderr, "--verify requires a reference file in -d mode\n"); return 1; }
                verify_path = argv[++i];
            } else {
                verify_flag = 1; /* compress-mode in-memory roundtrip */
            }
            continue;
        }
        if (strcmp(arg, "-d") == 0) { /* already noted */ continue; }
        /* positional */
        if (arg[0] != '-') {
            if (decompress_mode) {
                if (!lz_path) { lz_path = arg; continue; }
                /* ignore extra positionals; verify file should come via --verify */
            } else {
                if (!in_path) { in_path = arg; continue; }
            }
        }
    }

    if (show_help) {
        printf("Usage:\n");
        printf("  %s [--debug|-v] [--verify|-c] [-L level] [-o out.lzo] input_file\n", argv[0]);
        printf("     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        printf("     - --verify/-c (compress mode): do in-memory roundtrip check (no arg).\n");
        printf("     - -L|--level LEVEL : compression level to select kernel variant (default: 1)\n");
        printf("         supported LEVEL values:\n");
        printf("            1   : default LZO1X-1 compressor (kernel: lzo1x_1)\n");
        printf("            1k  : LZO1X-1K variant (kernel: lzo1x_1k) - optimized for kernel K behavior\n");
        printf("            1l  : LZO1X-1L variant (kernel: lzo1x_1l) - alternative lookup/heuristics\n");
        printf("            1o  : LZO1X-1O variant (kernel: lzo1x_1o) - other tuning/optimizations\n");
        printf("\n");
        printf("  %s -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo\n", argv[0]);
        printf("     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        printf("     - --verify/-c ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n");
        printf("\n");
        printf("Examples:\n");
        printf("  Compress with default level: %s input.dat -o out.lzo\n", argv[0]);
        printf("  Compress with level 1k:      %s -L 1k input.dat -o out.lzo\n", argv[0]);
        printf("  Decompress and verify:      %s -d --verify input.dat out.lzo -o out.dec\n", argv[0]);
        printf("  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", argv[0]);
        printf("  %s -h|--help                                 # show this help\n", argv[0]);
        printf("\nEnvironment variables (grouped — standalone / client->daemon / advanced):\n");
        printf("  I/O mode:\n");
        printf("    LZO_STANDARD_COPY=0|1    0=zero-copy (map into pinned buffer), 1=standard host->device copy (explicit upload).\n");
        printf("    LZO_FORCE_MAP=0|1        Force use of mapped pinned buffer even if standard-copy requested (overrides LZO_STANDARD_COPY).\n");
        printf("\n  Multi-threaded I/O (reduce file-read latency / parallel uploads):\n");
        printf("    LZO_MT_IO=0|1            Enable multi-threaded pread reads / parallel uploads (standalone + daemon).\n");
        printf("    LZO_MT_IO_THREADS=N      Threads for MT I/O (1-32; common values: 4/8; default: 4 standalone, 4 daemon).\n");
        printf("\n  Block sizing / adaptive behavior:\n");
        printf("    LZO_FIXED_BLOCK_SIZE=N   Fix block size in KB (overrides adaptive selection). Use 0 to restore adaptive behavior.\n");
        printf("    LZO_FORCE_NBLK=N         (advanced) Force target block count for blocking algorithm (comma-free tuning).\n");
        printf("\n  OpenCL & correctness flags:\n");
        printf("    LZO_OPENCL_DEVICE=CPU|GPU Prefer device selection for standalone/daemon. Daemons may ignore if configured otherwise.\n");
        printf("    LZO_DECOMP_VEC=0|1       Decompression: 1=prefer vectorized kernel (default), 0=force scalar decompressor.\n");
        printf("    LZO_DEBUG=1              Enable debug prints and timing traces.\n");
        printf("\n  Notes / dependency rules:\n");
        printf("    - Force/override rules: LZO_FORCE_MAP overrides LZO_STANDARD_COPY.\n");
        printf("    - MT I/O affects both file read and upload; enabling LZO_MT_IO without sufficient threads (LZO_MT_IO_THREADS) gives limited benefit.\n");
        printf("    - Client->daemon: client sends its environment options per-request; the daemon may accept or ignore some settings depending on its configuration.\n");
        printf("\n  Which binary honors which option:\n");
        printf("    - standalone (./lzo_gpu) supports all above flags and exposes low-level control (best for local testing).\n");
        printf("    - daemon (./lzo_gpu_daemon) accepts per-request options sent by clients (see lzo_gpu_client help). Some global daemon-config flags may still be ignored.\n");
        return 0;
    }

    /* Decompress mode */
    if (decompress_mode) {
        if (!lz_path) { fprintf(stderr, "no input .lzo specified (after -d)\n"); return 1; }
    uint64_t t_io_in = now_ns();
    size_t lz_sz; unsigned char* lz_buf = read_file(lz_path, &lz_sz);
    unsigned char* ref = NULL;
    size_t ref_sz = 0;
        unsigned char* p = lz_buf;
        uint16_t magic = *(uint16_t*)p; p += 2;
        if (magic != MAGIC) { fprintf(stderr, "bad magic\n"); return 1; }
        uint32_t orig_sz = *(uint32_t*)p; p += 4;
        uint32_t blk_sz = *(uint32_t*)p; p += 4;
        uint32_t nblk = *(uint32_t*)p; p += 4;
        uint32_t* len_arr = (uint32_t*)p; p += 4 * nblk;
        size_t comp_sz = lz_sz - (p - lz_buf);

        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (uint32_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

    uint64_t t_io_after = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
        /* 优先使用向量化解压器(性能提升199%)，失败时自动回退到标准版本
         * 可通过环境变量 LZO_DECOMP_VEC=0 强制使用标准解压器
         */
        const char* devec_env = getenv("LZO_DECOMP_VEC");
        int devec_flag = 1;  /* 默认优先使用向量化版本 */
        if (devec_env && strcmp(devec_env, "0") == 0) {
            devec_flag = 0;  /* 显式禁用 */
        }
        const char* decomp_base = devec_flag ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src  = devec_flag ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";

        /* 优化: 检查缓存以避免重复编译和创建内核 */
        cl_program prog_d = NULL;
        cl_kernel krn_d = NULL;
        int cache_idx_d = find_cached_program(decomp_base);

        if (cache_idx_d >= 0) {
            /* 使用缓存的程序和内核 */
            prog_d = prog_cache[cache_idx_d].prog;
            krn_d = prog_cache[cache_idx_d].krn_decompress;
            if (debug) fprintf(stderr, "DBG: using cached decompress program/kernel for %s\n", decomp_base);
        } else {
            /* 首次加载: 编译并缓存，如果向量化版本失败则自动回退到标准版本 */
            if (debug) fprintf(stderr, "DBG: loading and caching decompress program %s\n", decomp_base);

            /* 尝试加载首选kernel */
            int load_failed = 0;
            cl_int err;

            /* 尝试从二进制加载 */
            char bin_path[512];
            snprintf(bin_path, sizeof(bin_path), "%s.bin", decomp_base);
            char bin_path_alt[512];
            snprintf(bin_path_alt, sizeof(bin_path_alt), "lzo_gpu/%s.bin", decomp_base);

            FILE* fb = fopen(bin_path, "rb");
            if (!fb) fb = fopen(bin_path_alt, "rb");

            if (fb) {
                fseek(fb, 0, SEEK_END);
                long bsz = ftell(fb);
                fseek(fb, 0, SEEK_SET);
                unsigned char* bin = malloc(bsz);
                if (fread(bin,1,bsz,fb) == (size_t)bsz) {
                    cl_int binary_status;
                    prog_d = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
                        (const unsigned char**)&bin, &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                        err = clBuildProgram(prog_d, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
                        if (err != CL_SUCCESS) {
                            if (prog_d) { clReleaseProgram(prog_d); prog_d = NULL; }
                            load_failed = 1;
                        }
                    } else {
                        if (prog_d) { clReleaseProgram(prog_d); prog_d = NULL; }
                        load_failed = 1;
                    }
                } else {
                    load_failed = 1;
                }
                free(bin);
                fclose(fb);
            } else {
                load_failed = 1;
            }

            /* 如果二进制加载失败，尝试从源码编译 */
            if (load_failed || !prog_d) {
                prog_d = NULL;
                /* 检查源文件是否存在 */
                FILE* fchk = fopen(decomp_src, "rb");
                if (!fchk) {
                    char decomp_src_alt[512];
                    snprintf(decomp_src_alt, sizeof(decomp_src_alt), "lzo_gpu/%s", decomp_src);
                    fchk = fopen(decomp_src_alt, "rb");
                    if (!fchk) {
                        load_failed = 1;
                    } else {
                        fclose(fchk);
                        /* 尝试编译 */
                        size_t src_len = 0;
                        char* src = read_file(decomp_src_alt, &src_len);
                        prog_d = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
                        if (err == CL_SUCCESS) {
                            const char* build_opts = "-cl-std=CL2.0 -I. -I./lzo_gpu -I..";
                            err = clBuildProgram(prog_d, 1, &dev, build_opts, NULL, NULL);
                            if (err != CL_SUCCESS) {
                                if (prog_d) { clReleaseProgram(prog_d); prog_d = NULL; }
                                load_failed = 1;
                            }
                        } else {
                            load_failed = 1;
                        }
                        free(src);
                    }
                } else {
                    fclose(fchk);
                    size_t src_len = 0;
                    char* src = read_file(decomp_src, &src_len);
                    prog_d = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
                    if (err == CL_SUCCESS) {
                        const char* build_opts = "-cl-std=CL2.0 -I. -I./lzo_gpu -I..";
                        err = clBuildProgram(prog_d, 1, &dev, build_opts, NULL, NULL);
                        if (err != CL_SUCCESS) {
                            if (prog_d) { clReleaseProgram(prog_d); prog_d = NULL; }
                            load_failed = 1;
                        }
                    } else {
                        load_failed = 1;
                    }
                    free(src);
                }
            }

            /* 如果向量化版本加载失败，回退到标准版本 */
            if ((load_failed || !prog_d) && devec_flag) {
                if (!suppress_non_data) {
                    fprintf(stderr, "warning: vectorized decompressor unavailable, falling back to standard version\n");
                }
                decomp_base = "lzo1x_decomp";
                decomp_src = "lzo1x_decomp.cl";
                devec_flag = 0;

                /* 检查标准版本是否已缓存 */
                cache_idx_d = find_cached_program(decomp_base);
                if (cache_idx_d >= 0) {
                    prog_d = prog_cache[cache_idx_d].prog;
                    krn_d = prog_cache[cache_idx_d].krn_decompress;
                    if (debug) fprintf(stderr, "DBG: using cached standard decompress program/kernel\n");
                } else {
                    /* 加载标准版本 */
                    prog_d = load_prog_from_bin_or_src(decomp_base, decomp_src);
                }
            }

            /* 创建kernel */
            if (prog_d && !krn_d) {
                krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err);
                CHECK(err);

                /* 缓存程序和内核供后续使用 */
                cache_program(decomp_base, prog_d, NULL, krn_d);
            }
        }

        cl_int err;

    /* 优化: 使用pinned memory创建所有缓冲区 */
    uint64_t t_buffer_start = now_ns();
    cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, comp_sz, NULL, &err);
    CHECK(err);
    /* 使用map上传压缩数据 */
    void* mapped_comp = clEnqueueMapBuffer(q, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_comp, p, comp_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_comp, mapped_comp, 0, NULL, NULL));

    cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (nblk + 1) * sizeof(cl_uint), NULL, &err);
    CHECK(err);
    void* mapped_off = clEnqueueMapBuffer(q, d_off, CL_TRUE, CL_MAP_WRITE, 0, (nblk + 1) * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_off, off_arr, (nblk + 1) * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(q, d_off, mapped_off, 0, NULL, NULL));

    cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, orig_sz, NULL, &err);
    CHECK(err);
    /* decompressor expects an out_lens buffer as arg 3 */
    cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, nblk * sizeof(cl_uint), NULL, &err);
    CHECK(err);
    uint64_t t_buffer_end = now_ns();

    uint64_t t_upload_start = now_ns();
    CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
    CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));
    uint64_t t_upload_end = now_ns();

    /* 解压优化: local_size=8 可提升20%性能 (测试显示8是最优值)
     * 原因: 解压无需大量私有内存，可利用work-group的cache共享 */
    size_t lsz = 8;  /* 优化: 从1改为8, 解压性能 +20.2% */
    const char* local_size_env = getenv("LZO_LOCAL_SIZE");
    if (local_size_env) {
        lsz = (size_t)atoi(local_size_env);
        if (lsz == 0) lsz = 1;
        if (debug) fprintf(stderr, "DBG: using local_size=%zu from LZO_LOCAL_SIZE\n", lsz);
    }
    size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;  /* round up to multiple of lsz */
    cl_event evt_decomp;
    uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt_decomp));
    clWaitForEvents(1, &evt_decomp);
    uint64_t t_exec_end = now_ns();

    /* 优化: 使用map读取解压数据(零拷贝) */
    unsigned char* out2 = malloc(orig_sz);
    uint64_t t_read_start = now_ns();
    void* mapped_out2 = clEnqueueMapBuffer(q, d_out2, CL_TRUE, CL_MAP_READ, 0, orig_sz,
                                           0, NULL, NULL, &err);
    CHECK(err);
    memcpy(out2, mapped_out2, orig_sz);
    CHECK(clEnqueueUnmapMemObject(q, d_out2, mapped_out2, 0, NULL, NULL));
    uint64_t t_read_end = now_ns();

    /* perform verify first if requested; on failure do not write and exit non-zero */
    if (verify_path) {
        size_t ref_sz; unsigned char* ref = (unsigned char*)read_file(verify_path, &ref_sz);
        if (ref_sz != orig_sz || memcmp(ref, out2, orig_sz) != 0) {
            fprintf(stderr, "decompress verify FAILED!\n");
            for (size_t i = 0; i < orig_sz; ++i) {
                if (ref[i] != out2[i]) { fprintf(stderr, "first_mismatch_offset=%zu (ref=0x%02x out=0x%02x)\n", i, ref[i], out2[i]); break; }
            }
            free(ref);
            /* cleanup and exit */
            clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
            clReleaseKernel(krn_d); clReleaseProgram(prog_d);
            clReleaseCommandQueue(q); clReleaseContext(ctx);
            free(lz_buf); free(off_arr); free(out2);
            return 1;
        }
        free(ref);
        if (!suppress_non_data) puts("verify OK");
    }

    /* decide whether to write output:
       - If user requested --verify (decompress mode) and did not explicitly pass -o,
         do NOT write the decompressed file (only perform in-memory verification).
       - If user explicitly passed -o, honor it and write output as requested. */
    uint64_t t_write_start = now_ns();
    if (verify_path && !output_explicit) {
        /* skip writing decompressed output when verify requested without -o */
        if (!suppress_non_data) puts("verify mode: not writing decompressed output (no -o given)");
    } else {
        /* compute default output path if not explicitly provided */
        if (!output_path) {
            const char* in = lz_path;
            size_t L = strlen(in);
            int ends_lzo = (L >= 4 && strcmp(in + L - 4, ".lzo") == 0);
            size_t outL = ends_lzo ? (L - 4) : (L + 4);
            char* def = (char*)malloc(outL + 1);
            if (ends_lzo) { memcpy(def, in, L - 4); def[L - 4] = '\0'; }
            else { memcpy(def, in, L); memcpy(def + L, ".raw", 4); def[L + 4] = '\0'; }
            output_path = def;
        }

        if (output_path) {
            if (strcmp(output_path, "-") == 0) {
                /* write raw data to stdout (only data) */
                if (fwrite(out2, 1, orig_sz, stdout) != orig_sz) { perror("stdout write"); }
                fflush(stdout);
            } else {
                if (verify_path && strcmp(output_path, verify_path) == 0) {
                    fprintf(stderr, "refusing to write output to the same path as --verify reference: %s\n", output_path);
                    return 1;
                }
                FILE* fo = fopen(output_path, "wb");
                if (!fo) { perror(output_path); return 1; }
                if (fwrite(out2, 1, orig_sz, fo) != orig_sz) { perror("fwrite"); fclose(fo); return 1; }
                fclose(fo);
                if (!suppress_non_data) printf("wrote %s\n", output_path);
            }
        }
    }
    uint64_t t_write_end = now_ns();

        uint64_t t_total_end = now_ns();
        double ms_file_read = (t_io_after - t_io_in)/1e6;
        double ms_ocl_init = (t_ocl_init - t_io_after)/1e6;
        double ms_buffer = (t_buffer_end - t_buffer_start)/1e6;
        double ms_upload = (t_upload_end - t_upload_start)/1e6;
        double ms_kernel = (t_exec_end - t_exec_start)/1e6;
        double ms_download = (t_read_end - t_read_start)/1e6;
        double ms_write = (t_write_end - t_write_start)/1e6;
        double ms_total = (t_total_end - t_start_total)/1e6;
        double ratio = lz_sz > 0 ? (double)orig_sz / (double)lz_sz : 0.0;
        double thrpt = ms_kernel > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;

#if ENABLE_COMPRESSION_RATIO_TRACKING
        /* 输出解压缩统计信息 */
        printf("\n=== Decompression Statistics ===\n");
        printf("Compressed size  : %zu bytes (%.2f MB)\n", (size_t)lz_sz, lz_sz / (1024.0 * 1024.0));
        printf("Output size      : %zu bytes (%.2f MB)\n", (size_t)orig_sz, orig_sz / (1024.0 * 1024.0));
        printf("Expansion ratio  : %.2f:1 (%.2f%% of compressed)\n", ratio, 100.0 * ratio);
        printf("Block size       : %u bytes (%u KB)\n", blk_sz, blk_sz / 1024);
        printf("Number of blocks : %u\n", nblk);
        printf("Kernel           : %s (vectorized=%s)\n",
               decomp_base,
               (strstr(decomp_base, "_vec") != NULL) ? "yes" : "no");
        printf("Work groups      : global=%zu, local=%zu\n", gsz, lsz);
        printf("Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n",
               ((double)orig_sz / (1024.0*1024.0)) / (ms_total/1000.0),
               thrpt);
        printf("==============================\n\n");
#endif

        /* 打印详细的时间分解 */
        printf("\n=== Time Breakdown (Decompression) ===\n");
        print_ns("1. File Read", t_io_after - t_io_in);
        print_ns("2. OCL Init", t_ocl_init - t_io_after);
        print_ns("3. Kernel Load", 0);  /* uses cached kernel in decompress */
        print_ns("4. Buffer Alloc", t_buffer_end - t_buffer_start);
        print_ns("5. Data Upload", t_upload_end - t_upload_start);
        print_ns("6. Kernel Exec", t_exec_end - t_exec_start);
        print_ns("7. Data Download", t_read_end - t_read_start);
        print_ns("8. File Write", t_write_end - t_write_start);
        print_ns("TOTAL", t_total_end - t_start_total);
        printf("\n");

        /* 计算占比 */
        printf("=== Percentage Breakdown ===\n");
        printf("Kernel Exec     : %6.2f%%\n", 100.0 * ms_kernel / ms_total);
        printf("Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
               100.0 * (ms_upload + ms_download) / ms_total,
               100.0 * ms_upload / ms_total,
               100.0 * ms_download / ms_total);
        printf("File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
               100.0 * (ms_file_read + ms_write) / ms_total,
               100.0 * ms_file_read / ms_total,
               100.0 * ms_write / ms_total);
        printf("Buffer Alloc    : %6.2f%%\n",
               100.0 * ms_buffer / ms_total);
        printf("OCL Setup       : %6.2f%%\n",
               100.0 * ms_ocl_init / ms_total);
        printf("\n");

    clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        free(lz_buf); free(off_arr); free(out2);
        if (orig_path) free(ref);
        return 0;
    }

    /* Compress path (simple, fast) */
    if (!in_path) { fprintf(stderr, "no input file specified for compression\n"); return 1; }
    uint64_t t_compress_start = now_ns();

    /* Phase 8.3: 改用Daemon方式 - 直接fread到Pinned Memory，消除中间buffer */
    /* 先获取文件大小 */
    FILE* f_in = fopen(in_path, "rb");
    if (!f_in) {
        perror(in_path);
        return 1;
    }
    fseek(f_in, 0, SEEK_END);
    size_t in_sz = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    uint64_t t_after_fopen = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
    /* select compression kernel variant based on comp_level */
    char kernel_base[64]; char cl_src[128];
    if (strcmp(comp_level, "1") == 0 || strcmp(comp_level, "1x") == 0) {
        strcpy(kernel_base, "lzo1x_1");
    } else if (strcmp(comp_level, "1k") == 0) {
        strcpy(kernel_base, "lzo1x_1k");
    } else if (strcmp(comp_level, "1k_opt") == 0) {
        strcpy(kernel_base, "lzo1x_1k_opt");
    } else if (strcmp(comp_level, "1l") == 0) {
        strcpy(kernel_base, "lzo1x_1l");
    } else if (strcmp(comp_level, "1o") == 0) {
        strcpy(kernel_base, "lzo1x_1o");
    } else {
        fprintf(stderr, "unknown compression level: %s\n", comp_level);
        return 1;
    }

    /* Use standalone kernel (no frontend combinations) */
    snprintf(cl_src, sizeof(cl_src), "%s.cl", kernel_base);

    /* 优化: 检查缓存以避免重复编译和创建内核 */
    uint64_t t_kernel_load_start = now_ns();
    cl_int err;
    cl_program prog_c = NULL;
    cl_kernel krn_c = NULL;
    int cache_idx = find_cached_program(kernel_base);

    if (cache_idx >= 0) {
        /* 使用缓存的程序和内核 */
        prog_c = prog_cache[cache_idx].prog;
        krn_c = prog_cache[cache_idx].krn_compress;
        if (debug) fprintf(stderr, "DBG: using cached program/kernel for %s\n", kernel_base);
    } else {
        /* 首次加载: 编译并缓存 */
        if (debug) fprintf(stderr, "DBG: loading and caching program %s\n", kernel_base);
        prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);

        /* select kernel function name according to the kernel_base we loaded
         * Use canonical exported symbol 'lzo1x_block_compress' from frontends.
         */
        char krn_name[64];
        strcpy(krn_name, "lzo1x_block_compress");
        krn_c = clCreateKernel(prog_c, krn_name, &err);
        if (err != CL_SUCCESS) {
            /* Simplified fallback: report available kernels and force a single
             * source rebuild retry. Avoid multiple name-specific fallbacks — precompile
             * step should produce binaries matching canonical exported symbol.
             */
            if (err == CL_INVALID_KERNEL_NAME) {
                size_t kn_sz = 0;
                clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kn_sz);
                if (kn_sz > 0) {
                    char* kn = malloc(kn_sz + 1);
                    clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, kn_sz, kn, NULL);
                    kn[kn_sz] = '\0';
                    fprintf(stderr, "kernel '%s' not found; available kernels: %s\n", krn_name, kn);
                    free(kn);
                } else {
                    fprintf(stderr, "kernel '%s' not found and program reports no kernel names\n", krn_name);
                }
                /* Force a source-built program and retry once */
                clReleaseProgram(prog_c);
                prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);
                krn_c = clCreateKernel(prog_c, krn_name, &err);
                if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel after source rebuild failed (err=%d)\n", err); exit(1); }
            } else {
                fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); exit(1);
            }
        }

        /* 缓存程序和内核供后续使用 */
        cache_program(kernel_base, prog_c, krn_c, NULL);
    }
    uint64_t t_kernel_load_end = now_ns();

    /* Phase 8.3: File Read — 支持两种复制模式 (zero-copy / standard-copy)
     * 默认: zero-copy (map + fread -> device-accessible pinned memory)
     * 若环境变量 LZO_STANDARD_COPY=1 则使用 standard copy:
     *   fread -> host buffer (posix_memalign) ; clEnqueueWriteBuffer -> device
     * 这可帮助在不同平台（iGPU vs dGPU）对比两种传输路径的性能。
     */
    int standard_copy = 0;
    const char* std_s = getenv("LZO_STANDARD_COPY");
    if (std_s && atoi(std_s) == 1) standard_copy = 1;
    /* Allow forcing map (zero-copy) regardless of standard_copy via LZO_FORCE_MAP=1 */
    const char* force_map_env = getenv("LZO_FORCE_MAP");
    if (force_map_env && atoi(force_map_env) == 1) {
        if (standard_copy && debug) fprintf(stderr, "DBG: LZO_FORCE_MAP=1 overriding LZO_STANDARD_COPY -> forcing zero-copy\n");
        standard_copy = 0;
    }

    /* multi-threaded I/O control */
    int mt_io = 0;
    int mt_threads = 0;
    const char* mt_s = getenv("LZO_MT_IO");
    if (mt_s && atoi(mt_s) == 1) {
        mt_io = 1;
        const char* mt_threads_s = getenv("LZO_MT_IO_THREADS");
        mt_threads = mt_threads_s ? atoi(mt_threads_s) : 4;
        if (mt_threads < 1) mt_threads = 1;
        if (mt_threads > 32) mt_threads = 32;
    }

    uint64_t t_io_read_start = now_ns();
    cl_mem d_in = NULL;
    unsigned long upload_us = 0;
    unsigned long buffer_in_us = 0;
    uint64_t t_io_read_done = 0;
    void* mapped_in = NULL;
    unsigned char* host_in = NULL; /* used only in standard-copy */
    const unsigned char* data_for_blocking = NULL;

    if (standard_copy) {
        if (debug) fprintf(stderr, "DBG: using STANDARD copy path (LZO_STANDARD_COPY=1)\n");
        /* read into host buffer first */
        uint64_t t_file_read_start = now_ns();
        /* reuse outer host_in variable (do not shadow) */
        host_in = NULL;
        /* Align host buffer for DMA/pinned-friendly boundaries */
        if (posix_memalign((void**)&host_in, ALIGN_BYTES, in_sz) != 0) {
            host_in = malloc(in_sz);
            if (!host_in) { perror("malloc"); fclose(f_in); return 1; }
        }
        /* multi-threaded read into host_in (pread) if enabled */
        if (mt_io && mt_threads > 1) {
            int fd = fileno(f_in);
            /* only spawn as many workers as needed (skip zero-length tasks) */
            pthread_t *tids = calloc(mt_threads, sizeof(pthread_t));
            mt_io_arg_t *args = calloc(mt_threads, sizeof(mt_io_arg_t));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t cur = 0;
            int failed = 0;
            int created_count = 0;
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                /* skip creating a thread with no work */
                if (len == 0) { args[i].len = 0; args[i].err = 0; continue; }
                args[i].fd = fd;
                args[i].dest = host_in;
                args[i].off = cur;
                args[i].len = len;
                args[i].err = 0;
                    int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                    if (rc != 0) {
                        /* failed to create thread — mark error and break to fallback */
                        args[i].err = rc;
                        failed = 1; break;
                    }
                    created_count++;
                cur += len;
            }
            for (int i = 0; i < mt_threads; ++i) {
                if (args[i].len == 0) continue; /* no thread created for this slot */
                pthread_join(tids[i], NULL);
                if (args[i].err) failed = 1;
            }
            free(tids); free(args);
            if (failed) {
                if (fseek(f_in, 0, SEEK_SET) != 0) { perror("fseek"); free(host_in); fclose(f_in); return 1; }
                if (fread(host_in, 1, in_sz, f_in) != in_sz) { perror("fread"); free(host_in); fclose(f_in); return 1; }
            }
        } else {
            if (fread(host_in, 1, in_sz, f_in) != in_sz) { perror("fread"); free(host_in); fclose(f_in); return 1; }
        }
        fclose(f_in);
        uint64_t t_file_read_end = now_ns();
        t_io_read_done = t_file_read_end;
        /* keep host_in for blocking calc and upload later */
        data_for_blocking = (const unsigned char*)host_in;
    } else {
        if (debug) fprintf(stderr, "DBG: using ZERO-COPY path (default)\n");
        /* zero-copy: create pinned device buffer then map and fread into it */
        if (debug) fprintf(stderr, "DBG: getting cached d_in size=%zu (pinned)\n", in_sz);
        uint64_t t_buf_in_start = now_ns();
        d_in = get_or_create_buffer(&buffer_cache.d_in, &buffer_cache.in_size,
                                    in_sz, CL_MEM_READ_ONLY);
        uint64_t t_buf_in_end = now_ns();
        /* mapping counted as part of File Read in zero-copy path */
        mapped_in = clEnqueueMapBuffer(q, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz,
                             0, NULL, NULL, &err);
        CHECK(err);

        /* Support multi-threaded pread into mapped device-accessible memory */
        if (mt_io && mt_threads > 1) {
            int fd = fileno(f_in);
            pthread_t *tids = calloc(mt_threads, sizeof(pthread_t));
            mt_io_arg_t *args = calloc(mt_threads, sizeof(mt_io_arg_t));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t cur = 0;
            int failed = 0;
            int created_count2 = 0;
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                /* skip zero-length chunks to avoid creating useless threads */
                if (len == 0) { args[i].len = 0; args[i].err = 0; continue; }
                args[i].fd = fd;
                args[i].dest = mapped_in;
                args[i].off = cur;
                args[i].len = len;
                args[i].err = 0;
                int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                if (rc != 0) {
                    args[i].err = rc; /* mark failure for fallback */
                    failed = 1;
                    break;
                }
                created_count2++;
                cur += len;
            }
            for (int i = 0; i < mt_threads; ++i) {
                if (args[i].len == 0) continue;
                pthread_join(tids[i], NULL);
                if (args[i].err) failed = 1;
            }
            free(tids); free(args);
            if (failed) {
                if (fseek(f_in, 0, SEEK_SET) != 0) { perror("fseek"); clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL); fclose(f_in); return 1; }
                if (fread(mapped_in, 1, in_sz, f_in) != in_sz) { perror("fread"); clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL); fclose(f_in); return 1; }
            }
        } else {
            if (fread(mapped_in, 1, in_sz, f_in) != in_sz) {
                perror("fread");
                clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL);
                fclose(f_in);
                return 1;
            }
        }
        fclose(f_in);  /* 文件读取完成，关闭 */
        uint64_t t_map_fread_end = now_ns();
        /* in zero-copy, buffer alloc was effectively part of file read */
        t_io_read_done = t_map_fread_end;
        buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000; /* small */
        data_for_blocking = (const unsigned char*)mapped_in;
        upload_us = 0;

        /* leave mapped_in accessible for later use (we will unmap after blocking calc) */
        /* store mapped_in to a temporary symbol name used below */
        /* make sure mapped_in is available in the following block; reuse variable name */
        /* We'll keep mapped_in but no-op in other path */
        /* map pointer is kept in the local scope below where used for blocking */
    }

    /* Phase 7.2: 使用自适应块大小 (基于数据熵)
     * 此时数据已在mapped_in中，可用于熵计算
     */
    uint64_t t_blocking_start = now_ns();
    /* timestamps for allocation/upload phases — declared early so the
     * standard-copy branch can write into them; defaults assigned below
     * after blocking is recorded.
     */
    uint64_t t_buffer_alloc_start = 0, t_buffer_alloc_end = 0;
    uint64_t t_upload_start = 0, t_upload_end = 0;
    size_t blk = 0, nblk = 0;
    choose_blocking_adaptive(data_for_blocking, in_sz, dev, &blk, &nblk, debug);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    if (debug) {
        fprintf(stderr, "DBG: choose_blocking -> in_sz=%zu blk=%zu nblk=%zu worst_blk=%zu out_cap=%zu\n",
                in_sz, blk, nblk, worst_blk, out_cap);
    }

    /* Blocking calc finished: record end now so we don't include subsequent
     * buffer allocation or upload timings in the "Blocking Calc" measurement.
     */
    uint64_t t_blocking_end = now_ns();

    /* Unmap输入缓冲区 (数据已就绪且熵计算完成) */
    if (!standard_copy) {
        CHECK(clEnqueueUnmapMemObject(q, d_in, mapped_in, 0, NULL, NULL));
    }
    else {

        /* Standard copy: now upload host_in into d_in (device buffer)
         * Record allocations and upload using the outer timestamps so the
         * final printed breakdown includes these times separately from
         * the blocking calculation.
         */
        t_buffer_alloc_start = now_ns();
        d_in = get_or_create_buffer(&buffer_cache.d_in, &buffer_cache.in_size,
                                    in_sz, CL_MEM_READ_ONLY);
        t_buffer_alloc_end = now_ns();

        if (mt_io && mt_threads > 1) {
            /* parallel non-blocking writes + wait */
            cl_event *events = calloc(mt_threads, sizeof(cl_event));
            size_t base = in_sz / mt_threads;
            size_t rem = in_sz % mt_threads;
            size_t off = 0;
            t_upload_start = now_ns();
            for (int i = 0; i < mt_threads; ++i) {
                size_t len = base + (i == mt_threads-1 ? rem : 0);
                err = clEnqueueWriteBuffer(q, d_in, CL_FALSE, off, len, (char*)host_in + off, 0, NULL, &events[i]);
                CHECK(err);
                off += len;
            }
            CHECK(clWaitForEvents(mt_threads, events));
            t_upload_end = now_ns();
            for (int i = 0; i < mt_threads; ++i) clReleaseEvent(events[i]);
            free(events);
        } else {
            t_upload_start = now_ns();
            err = clEnqueueWriteBuffer(q, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
            CHECK(err);
            t_upload_end = now_ns();
        }

        /* host_in no longer needed */
        free(host_in); host_in = NULL;
    }
    /* NOTE: t_blocking_end already recorded above; any upload / buffer alloc
     * timings are measured separately below and will not be attributed to
     * the blocking calculation.
     */

    /* Data Upload / Buffer alloc timestamps: default to zero-size intervals
     * for zero-copy path (overwrite if standard-copy branch set them).
     */
    if (t_upload_start == 0) t_upload_start = t_blocking_end;
    if (t_upload_end == 0) t_upload_end = t_blocking_end;

    /* Buffer Alloc (in) default to zero (already included in File Read for
     * zero-copy), will be overwritten by standard-copy branch above.
     */
    if (t_buffer_alloc_start == 0) t_buffer_alloc_start = t_io_read_start;
    if (t_buffer_alloc_end == 0) t_buffer_alloc_end = t_io_read_start;

    /* 创建输出缓冲区 (Pinned Memory) */
    uint64_t t_out_buffer_start = now_ns();
    if (debug) fprintf(stderr, "DBG: getting cached d_out size=%zu (pinned)\n", out_cap);
    cl_mem d_out = get_or_create_buffer(&buffer_cache.d_out, &buffer_cache.out_size,
                                        out_cap, CL_MEM_WRITE_ONLY);
    uint64_t t_out_buffer_end = now_ns();

    /* map_mode: 0=default CL_MEM_WRITE_ONLY + clEnqueueReadBuffer
     * 1=ALLOC_HOST_PTR + clEnqueueMapBuffer
     * 2=USE_HOST_PTR with host pointer (posix_memalign)
     */
    int map_mode = 0;
    /* Default to explicit reads (map_mode=0) for all kernels. Individual
     * experiments can still force mapping via LZO_FORCE_MAP=1. */
    /* Allow overriding map_mode via environment for testing/comparison.
     * Set environment variable LZO_FORCE_MAP=0 or =1 to force explicit read
     * or alloc-host+map respectively. Helpful to produce two test binaries
     * that exercise map==0 vs map==1 deterministically. */
    const char* force_map_s = getenv("LZO_FORCE_MAP");
    if (force_map_s) {
        int fm = atoi(force_map_s);
        if (fm == 0 || fm == 1) {
            if (debug) fprintf(stderr, "DBG: forcing map_mode from env LZO_FORCE_MAP=%d\n", fm);
            map_mode = fm;
        } else {
            if (debug) fprintf(stderr, "DBG: LZO_FORCE_MAP='%s' ignored (not 0 or 1)\n", force_map_s);
        }
    }
    /* The 'usehost' strategy has been removed; only map_mode 0/1 are used. */

    /* 优化: 使用缓冲区缓存 */
    uint64_t t_len_buffer_start = now_ns();
    size_t len_bytes = nblk * sizeof(cl_uint);
    if (debug) fprintf(stderr, "DBG: getting cached d_len size=%zu\n", len_bytes);
    cl_mem d_len = get_or_create_buffer(&buffer_cache.d_len, &buffer_cache.len_size,
                                        len_bytes, CL_MEM_READ_WRITE);  /* 优化:移除ALLOC_HOST_PTR */
    uint64_t t_len_buffer_end = now_ns();

    /* 优化: 使用参数缓存避免重复设置 */
    uint64_t t_setup_args_start = now_ns();
    set_kernel_args_cached(krn_c, d_in, d_out, d_len, in_sz, blk, worst_blk);
    uint64_t t_setup_args_end = now_ns();

    /* 压缩: 必须使用local_size=1 (每个work-item需2KB字典，local_size>1会内存溢出)
     * 测试显示local_size=8时性能暴跌94%，因为字典溢出到全局内存 */
    size_t lsz = 1;  /* 压缩必须为1，不可修改 */
    const char* local_size_env = getenv("LZO_LOCAL_SIZE");
    if (local_size_env) {
        lsz = (size_t)atoi(local_size_env);
        if (lsz == 0) lsz = 1;
        if (debug) fprintf(stderr, "DBG: using local_size=%zu from LZO_LOCAL_SIZE (WARNING: >1 may degrade performance)\n", lsz);
    }
    size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;  /* round up to multiple of lsz */
    cl_event evt_compute;
    uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_c, 1, NULL, &gsz, &lsz, 0, NULL, &evt_compute));
    clWaitForEvents(1, &evt_compute);
    uint64_t t_exec_end = now_ns();

    /* 优化: 使用map读取长度数组(零拷贝) */
    uint64_t t_download_start = now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    uint64_t t_len_read_start = now_ns();
    void* mapped_len = clEnqueueMapBuffer(q, d_len, CL_TRUE, CL_MAP_READ, 0, len_bytes,
                                          0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, len_bytes);
    CHECK(clEnqueueUnmapMemObject(q, d_len, mapped_len, 0, NULL, NULL));
    uint64_t t_len_read_end = now_ns();

    if (debug) {
        fprintf(stderr, "Per-block compressed lengths (nblk=%zu):\n", nblk);
        for (size_t i = 0; i < nblk; ++i) {
            fprintf(stderr, "  block %4zu : %u\n", i, len_arr[i]);
        }
    }

    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += len_arr[i];
    unsigned char* out_buf = NULL;
    size_t host_off = 0;
    /* 优化: 使用map读取输出缓冲区(零拷贝) */
    uint64_t t_bulk_read_start = now_ns();
    if (debug) fprintf(stderr, "DBG: about to map d_out size=%zu\n", out_cap);
    unsigned char* dev_out = (unsigned char*)clEnqueueMapBuffer(q, d_out, CL_TRUE, CL_MAP_READ,
                                                                  0, out_cap, 0, NULL, NULL, &err);
    CHECK(err);
    if (debug) fprintf(stderr, "DBG: map completed\n");
    uint64_t t_bulk_read_end = now_ns();
    /* debug: dump first 32 bytes of first block to help diagnose visibility */
    if (debug) {
        fprintf(stderr, "dev_out[0..31]:");
        for (size_t i = 0; i < 32 && i < out_cap; ++i) fprintf(stderr, " %02x", dev_out[i]);
        fprintf(stderr, "\n");
    }

    /* If kernel didn't populate `out_len` (all zeros), try reconstructing per-block
     * lengths from the device output buffer: we expect the kernel to write a little-endian
     * 32-bit length at the start of each block region as a robust fallback. */
    if (out_sz == 0) {
        /* Attempt to recover per-block lengths from device output buffer.
         * Guard against interpreting arbitrary bytes as huge lengths which
         * can lead to oversized allocations and crashes. Accept a length
         * only if it is non-zero and reasonably bounded by `worst_blk` and
         * `out_cap`. Accumulate into a temporary size and check overflow. */
        size_t tmp_out_sz = 0;
        for (size_t i = 0; i < nblk; ++i) {
            size_t dev_off = i * worst_blk;
            if (dev_off + 4 <= out_cap) {
                uint32_t v = (uint32_t)dev_out[dev_off + 0]
                           | ((uint32_t)dev_out[dev_off + 1] << 8)
                           | ((uint32_t)dev_out[dev_off + 2] << 16)
                           | ((uint32_t)dev_out[dev_off + 3] << 24);
                /* sanity checks */
                if (v == 0 || v > worst_blk || v > out_cap) {
                    len_arr[i] = 0;
                } else {
                    /* check overflow before adding */
                    if (tmp_out_sz + (size_t)v < tmp_out_sz) {
                        len_arr[i] = 0;
                    } else {
                        len_arr[i] = v;
                        tmp_out_sz += (size_t)v;
                    }
                }
            } else {
                len_arr[i] = 0;
            }
        }
        out_sz = tmp_out_sz;
        if (out_sz == 0) {
            fprintf(stderr, "ERR: failed to recover per-block lengths from device output; aborting\n");
            free(dev_out);
            free(len_arr);
            /* cleanup and exit with error */
            clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
            clReleaseKernel(krn_c); clReleaseProgram(prog_c);
            clReleaseCommandQueue(q); clReleaseContext(ctx);
            /* Phase 8.3: in_buf不再使用(直接fread到mapped) */
            return 1;
        }
    }

    if (out_sz > out_cap) {
        fprintf(stderr, "ERR: computed total output size (%zu) exceeds device capacity (%zu); aborting\n", out_sz, out_cap);
        free(dev_out);
        free(len_arr);
        clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
        clReleaseKernel(krn_c); clReleaseProgram(prog_c);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        return 1;
    }
    /* Phase 8.2: Zero-Copy 优化 - 移除中间的 memcpy 打包，直接使用 mapped memory */
    /* 不再分配 out_buf，保持 dev_out mapped，稍后直接 scatter-gather 写入 */
    out_buf = NULL;  /* 标记为未使用 */
    /* 优化: 保持 dev_out mapped，等写入文件后再 unmap */
    uint64_t t_download_end = now_ns();

    /* decide output path if not specified: default to input_file.lzo */
    uint64_t t_write_start = now_ns();
    if (!output_path) {
        size_t L = strlen(in_path); char* def = (char*)malloc(L + 4 + 1);
        memcpy(def, in_path, L); memcpy(def + L, ".lzo", 4); def[L + 4] = '\0';
        output_path = def;
    }
    /* write LZO container: magic, orig_size, blk_size, nblk, len[nblk], then data */
    FILE* fo = fopen(output_path, "wb");
    if (!fo) { perror(output_path); return 1; }
    uint16_t magic = MAGIC;
    uint32_t orig_sz32 = (uint32_t)in_sz;
    uint32_t blk32 = (uint32_t)blk;
    uint32_t nblk32 = (uint32_t)nblk;
    if (fwrite(&magic, sizeof(magic), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&orig_sz32, sizeof(orig_sz32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&blk32, sizeof(blk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&nblk32, sizeof(nblk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(len_arr, sizeof(uint32_t), nblk, fo) != nblk) { perror("fwrite"); fclose(fo); return 1; }
    /* Phase 8.2: Zero-Copy 写入 - 直接从 mapped memory (dev_out) scatter-gather 到文件 */
    for (size_t i = 0; i < nblk; ++i) {
        if (len_arr[i] > 0) {
            size_t dev_off = i * worst_blk;
            if (fwrite(dev_out + dev_off, 1, len_arr[i], fo) != len_arr[i]) {
                perror("fwrite block");
                fclose(fo);
                CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
                free(len_arr);
                return 1;
            }
        }
    }
    fclose(fo);
    /* 写入完成后才 unmap */
    CHECK(clEnqueueUnmapMemObject(q, d_out, dev_out, 0, NULL, NULL));
    printf("wrote %s\n", output_path);

    uint64_t t_after_write = now_ns();

    /* 计算各阶段耗时 */
    double ms_file_read = (t_io_read_done - t_io_read_start)/1e6;
    double ms_ocl_init = (t_ocl_init - t_after_fopen)/1e6;
    double ms_kernel_load = (t_kernel_load_end - t_kernel_load_start)/1e6;
    double ms_blocking = (t_blocking_end - t_blocking_start)/1e6;
    double ms_buffer_alloc_in = (t_buffer_alloc_end - t_buffer_alloc_start)/1e6;
    double ms_upload = (t_upload_end - t_upload_start)/1e6;
    double ms_buffer_alloc_out = (t_out_buffer_end - t_out_buffer_start)/1e6;
    double ms_buffer_alloc_len = (t_len_buffer_end - t_len_buffer_start)/1e6;
    double ms_setup_args = (t_setup_args_end - t_setup_args_start)/1e6;
    double ms_kernel = (t_exec_end - t_exec_start)/1e6;
    double ms_len_read = (t_len_read_end - t_len_read_start)/1e6;
    double ms_bulk_read = (t_bulk_read_end - t_bulk_read_start)/1e6;
    double ms_download_total = (t_download_end - t_download_start)/1e6;
    double ms_file_write = (t_after_write - t_write_start)/1e6;
    double ms_total = (t_after_write - t_io_read_start)/1e6;
    double ms_buffer_alloc_total = ms_buffer_alloc_in + ms_buffer_alloc_out + ms_buffer_alloc_len;

    double ratio = out_sz > 0 ? (double)in_sz / (double)out_sz : 0.0;
    double thrpt = ms_kernel > 0 ? ((double)in_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;

#if ENABLE_COMPRESSION_RATIO_TRACKING
    /* 输出压缩统计信息 */
    printf("\n=== Compression Statistics ===\n");
    printf("Input size       : %zu bytes (%.2f MB)\n", in_sz, in_sz / (1024.0 * 1024.0));
    printf("Compressed size  : %zu bytes (%.2f MB)\n", out_sz, out_sz / (1024.0 * 1024.0));
    printf("Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, 100.0 / ratio);
    long long space_diff = (long long)in_sz - (long long)out_sz;
    printf("Space saved      : %lld bytes (%.2f MB, %.1f%%)\n",
           space_diff,
           space_diff / (1024.0 * 1024.0),
           100.0 * (1.0 - 1.0/ratio));
    printf("Block size       : %zu bytes (%zu KB)\n", blk, blk / 1024);
    printf("Number of blocks : %zu\n", nblk);
    printf("Avg block compr  : %.2f:1\n", ratio);
    printf("Kernel           : %s (from %s)\n", kernel_base, cl_src);
    printf("Work groups      : global=%zu, local=%zu\n", gsz, lsz);
    printf("Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n",
           ((double)in_sz / (1024.0*1024.0)) / (ms_total/1000.0),
           thrpt);
    printf("==============================\n\n");
#endif

    /* 打印详细的时间分解 */
    printf("\n=== Time Breakdown (Compression) ===\n");
    print_ns("1. File Read", t_io_read_done - t_io_read_start);
    print_ns("2. OCL Init", t_ocl_init - t_after_fopen);
    print_ns("3. Kernel Load", t_kernel_load_end - t_kernel_load_start);
    print_ns("4. Blocking Calc", t_blocking_end - t_blocking_start);
    print_ns("5. Buffer Alloc (in)", t_buffer_alloc_end - t_buffer_alloc_start);
    print_ns("6. Data Upload", t_upload_end - t_upload_start);
    print_ns("7. Buffer Alloc (out)", t_out_buffer_end - t_out_buffer_start);
    print_ns("8. Buffer Alloc (len)", t_len_buffer_end - t_len_buffer_start);
    print_ns("9. Setup Args", t_setup_args_end - t_setup_args_start);
    print_ns("10. Kernel Exec", t_exec_end - t_exec_start);
    print_ns("11. Download (len)", t_len_read_end - t_len_read_start);
    print_ns("12. Download (bulk)", t_bulk_read_end - t_bulk_read_start);
    print_ns("13. Download Total", t_download_end - t_download_start);
    print_ns("14. File Write", t_after_write - t_write_start);
    print_ns("TOTAL", t_after_write - t_io_read_start);
    printf("\n");

    /* 计算占比 */
    printf("=== Percentage Breakdown ===\n");
    printf("Kernel Exec     : %6.2f%%\n", 100.0 * ms_kernel / ms_total);
    printf("Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
           100.0 * (ms_upload + ms_download_total) / ms_total,
           100.0 * ms_upload / ms_total,
           100.0 * ms_download_total / ms_total);
    printf("File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
           100.0 * (ms_file_read + ms_file_write) / ms_total,
           100.0 * ms_file_read / ms_total,
           100.0 * ms_file_write / ms_total);
    printf("Buffer Alloc    : %6.2f%% (in=%.2f%% + out=%.2f%% + len=%.2f%%)\n",
           100.0 * ms_buffer_alloc_total / ms_total,
           100.0 * ms_buffer_alloc_in / ms_total,
           100.0 * ms_buffer_alloc_out / ms_total,
           100.0 * ms_buffer_alloc_len / ms_total);
    printf("OCL Setup       : %6.2f%% (init=%.2f%% + kernel_load=%.2f%%)\n",
           100.0 * (ms_ocl_init + ms_kernel_load) / ms_total,
           100.0 * ms_ocl_init / ms_total,
           100.0 * ms_kernel_load / ms_total);
    printf("Kernel Args     : %6.2f%%\n",
           100.0 * ms_setup_args / ms_total);
    printf("Other           : %6.2f%%\n",
           100.0 * ms_blocking / ms_total);
    printf("\n");

    /* optional roundtrip verification only when --verify set */
    if (verify_flag) {
        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (size_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

        /* 优先使用向量化解压器进行验证，失败时回退到标准版本 */
        const char* devec2 = getenv("LZO_DECOMP_VEC");
        int devec_flag2 = 1;  /* 默认优先使用向量化版本 */
        if (devec2 && strcmp(devec2, "0") == 0) {
            devec_flag2 = 0;  /* 显式禁用 */
        }
        const char* decomp_base2 = devec_flag2 ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src2  = devec_flag2 ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";

        /* 尝试加载首选kernel，失败则回退 */
        cl_program prog_d = NULL;
        FILE* test_f = fopen(decomp_src2, "rb");
        if (!test_f) {
            char alt_path[512];
            snprintf(alt_path, sizeof(alt_path), "lzo_gpu/%s", decomp_src2);
            test_f = fopen(alt_path, "rb");
        }
        if (test_f) {
            fclose(test_f);
            prog_d = load_prog_from_bin_or_src(decomp_base2, decomp_src2);
        }

        /* 如果向量化版本不可用，回退到标准版本 */
        if (!prog_d && devec_flag2) {
            decomp_base2 = "lzo1x_decomp";
            decomp_src2 = "lzo1x_decomp.cl";
            prog_d = load_prog_from_bin_or_src(decomp_base2, decomp_src2);
        }

        if (!prog_d) {
            fprintf(stderr, "error: unable to load decompressor for verify\n");
            return 1;
        }

        cl_kernel krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err);
        CHECK(err);
        cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_sz, out_buf, &err); CHECK(err);
        cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (nblk + 1) * sizeof(cl_uint), off_arr, &err); CHECK(err);
        cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, in_sz, NULL, &err); CHECK(err);
        cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

        CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
        CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
        CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
        CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
        CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk));
        CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &in_sz));
        CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));

        cl_event evt_verify;
        CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt_verify));
        clWaitForEvents(1, &evt_verify);
        unsigned char* out2 = malloc(in_sz);
        CHECK(clEnqueueReadBuffer(q, d_out2, CL_TRUE, 0, in_sz, out2, 0, NULL, NULL));

        /* Phase 8.3: in_buf不再存在，重新读取原始文件用于验证 */
        size_t verify_sz;
        unsigned char* verify_buf = (unsigned char*)read_file(in_path, &verify_sz);
        if (verify_sz != in_sz) {
            fprintf(stderr, "verify size mismatch: %zu != %zu\n", verify_sz, in_sz);
        } else if (memcmp(verify_buf, out2, in_sz) == 0) {
            printf("verify OK\n");
        } else {
            printf("verify FAILED\n");
            for (size_t i=0; i<in_sz; i++) {
                if (verify_buf[i] != out2[i]) {
                    printf("first mismatch at %zu (0x%02x != 0x%02x)\n", i, verify_buf[i], out2[i]);
                    break;
                }
            }
        }
        free(verify_buf);

        clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        free(off_arr); free(out2);
    }

    /* cleanup */
    clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
    clReleaseKernel(krn_c);
    clReleaseProgram(prog_c);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    /* Phase 8.3: in_buf不再使用(直接fread到mapped) */
    free(len_arr);
    /* Phase 8.2: out_buf 不再使用，已移除 */
    return 0;
}
