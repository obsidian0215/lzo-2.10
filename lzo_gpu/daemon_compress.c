/*
 * daemon_compress.c - 守护进程压缩核心逻辑
 * 从lzo_host.c提取,用于守护进程复用OpenCL资源
 *
 * Phase 7.2: 同步自适应块大小优化
 * - 熵计算
 * - 动态OCC调整
 * - Pinned Memory
 * - Buffer缓存
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>

#define CHECK(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

#define D_BITS 11
/* Phase 7.2: 对齐参数到standalone版本 */
#define ALIGN_BYTES 65536
#define MIN_BLOCK_SIZE (64 * 1024)   // 64KB (与standalone一致)
#define MAX_BLOCK_SIZE (512 * 1024)  // 512KB (与standalone一致)

/* 与lzo_host.c相同的辅助函数 */
static inline size_t lzo_worst(size_t sz) {
    return sz + sz / 16 + 64 + 3;
}

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Multi-threaded pread support (same pattern as lzo_host.c) */
typedef struct {
    int fd;
    void *dest; /* base pointer */
    off_t off;  /* offset into dest */
    size_t len; /* length to read */
    int err;    /* errno on failure, 0 on success */
} mt_io_arg_t;

static void* mt_pread_worker(void *v) {
    mt_io_arg_t *a = (mt_io_arg_t*)v;
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

/* Phase 7.2: 熵计算 (从standalone同步) */
static double calculate_entropy(const unsigned char* data, size_t size) {
    if (size == 0) return 0.0;

    /* 采样策略：大文件每4KB采样1024字节 */
    size_t sample_size = size;
    size_t sample_interval = 1;

    if (size > 1024 * 1024) {  /* >1MB: 采样 */
        sample_size = (size / 4096) * 256;  /* 每4KB采样256字节 */
        if (sample_size > 256 * 1024) sample_size = 256 * 1024;  /* 最多256KB */
        sample_interval = size / sample_size;
    }

    /* 频率统计 */
    unsigned int freq[256] = {0};
    for (size_t i = 0; i < sample_size; i++) {
        freq[data[i * sample_interval]]++;
    }

    /* 熵计算：H = -Σ(p_i * log2(p_i)) */
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            double p = (double)freq[i] / sample_size;
            entropy -= p * log2(p);
        }
    }

    return entropy;
}

/* Phase 7.2: 自适应块大小 (从standalone同步) */
static size_t adaptive_block_size(size_t file_size, double entropy, cl_uint cu) {
    /* 熵分类：
     *   高熵 (>7.5): 随机数据，压缩率低，使用大块减少overhead
     *   中熵 (6-7.5): 一般数据，平衡块大小
     *   低熵 (<6): 重复数据，压缩率高，使用小块提升并行度
     */
    size_t base_block;

    if (entropy > 7.5) {
        /* 高熵：大块优先 (降低kernel启动开销) */
        base_block = 512 * 1024;  /* 512KB */
    } else if (entropy > 6.0) {
        /* 中熵：128-256KB */
        base_block = (entropy > 7.0) ? 256 * 1024 : 192 * 1024;
    } else {
        /* 低熵：Phase 7.2策略 - 小块倾向 */
        if (file_size >= 100 * 1024 * 1024) {
            base_block = 64 * 1024;   /* >=100MB: 64KB (最大并行度) */
        } else if (file_size >= 10 * 1024 * 1024) {
            base_block = 96 * 1024;   /* 10-100MB: 96KB */
        } else {
            base_block = 128 * 1024;  /* <10MB: 128KB */
        }
    }

    /* 确保在范围内 */
    if (base_block < MIN_BLOCK_SIZE) base_block = MIN_BLOCK_SIZE;
    if (base_block > MAX_BLOCK_SIZE) base_block = MAX_BLOCK_SIZE;

    return base_block;
}

/* 读取文件 */
static void* read_file_data(const char* path, size_t* size_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* buf = malloc(sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, sz, f) != (size_t)sz) {
        perror("fread");
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *size_out = sz;
    return buf;
}

/* 写入压缩文件 - 优化: 直接从稀疏的Pinned Memory写入，避免中间buffer */
static int write_compressed_file(const char* path,
                                 size_t orig_size, size_t blk_size,
                                 size_t nblk, const unsigned int* lens,
                                 const void* sparse_data, size_t worst_blk) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    // 写入头部
    unsigned short magic = 0x4C5A;  // 'LZ'
    fwrite(&magic, sizeof(magic), 1, f);

    unsigned int u32;
    u32 = (unsigned int)orig_size;
    fwrite(&u32, sizeof(u32), 1, f);
    u32 = (unsigned int)blk_size;
    fwrite(&u32, sizeof(u32), 1, f);
    u32 = (unsigned int)nblk;
    fwrite(&u32, sizeof(u32), 1, f);

    // 写入长度数组
    fwrite(lens, sizeof(unsigned int), nblk, f);

    // 写入压缩数据 (Scatter-Gather from sparse buffer)
    const unsigned char* dev_out = (const unsigned char*)sparse_data;
    for (size_t i = 0; i < nblk; i++) {
        if (lens[i] > 0) {
            size_t dev_off = i * worst_blk;
            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                perror("fwrite block");
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}

/* Phase 7.2: Buffer缓存机制 (从standalone同步) */
static struct {
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    size_t in_size;
    size_t out_size;
    size_t len_size;
} buffer_cache = {0};

static cl_mem get_or_create_buffer(cl_context ctx, cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags, cl_int* err_out) {
    if (*cached_size < required_size) {
        if (*cached_buf) {
            clReleaseMemObject(*cached_buf);
            *cached_buf = NULL;
            *cached_size = 0;
        }
        /* Phase 6.1: 使用Pinned Memory */
        *cached_buf = clCreateBuffer(ctx, flags | CL_MEM_ALLOC_HOST_PTR,
                                     required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS) {
            *cached_size = required_size;
        }
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

/* 清理buffer缓存 - 外部可调用以在daemon关闭时释放资源 */
void cleanup_compress_buffer_cache(void) {
    if (buffer_cache.d_in) clReleaseMemObject(buffer_cache.d_in);
    if (buffer_cache.d_out) clReleaseMemObject(buffer_cache.d_out);
    if (buffer_cache.d_len) clReleaseMemObject(buffer_cache.d_len);
    buffer_cache.d_in = NULL;
    buffer_cache.d_out = NULL;
    buffer_cache.d_len = NULL;
    buffer_cache.in_size = 0;
    buffer_cache.out_size = 0;
    buffer_cache.len_size = 0;
}

/* 动态块选择 - Phase 7.2自适应版本 (从standalone同步) */
static void choose_blocking_adaptive(size_t in_sz, const unsigned char* data,
                                     cl_device_id dev, int debug,
                                     size_t* blk_sz_out, size_t* nblk_out) {
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 计算熵 */
    double entropy = calculate_entropy(data, in_sz);
    if (debug) {
        fprintf(stderr, "[DAEMON] 文件熵: %.2f (大小=%zu)\n", entropy, in_sz);
    }

    /* 自适应块大小 */
    size_t blk = adaptive_block_size(in_sz, entropy, cu);

    /* 环境变量覆盖 (与standalone一致) */
    const char* fixed_blk_env = getenv("LZO_FIXED_BLOCK_SIZE");
    if (fixed_blk_env) {
        size_t env_blk = atoi(fixed_blk_env) * 1024;
        if (env_blk >= MIN_BLOCK_SIZE && env_blk <= MAX_BLOCK_SIZE) {
            if (debug) fprintf(stderr, "[DAEMON] 使用固定块大小: %zu KB\n", env_blk / 1024);
            blk = env_blk;
        }
    }

    /* 动态OCC调整 */
    size_t occ_factor;
    if (entropy > 7.5) {
        occ_factor = 64;   /* 高熵：降低OCC (减少work-items) */
    } else if (entropy > 6.0) {
        occ_factor = 96;   /* 中熵 */
    } else {
        occ_factor = 128;  /* 低熵：最大并行度 */
    }

    /* 计算块数 */
    size_t tgt_blk = (size_t)cu * occ_factor;
    if (tgt_blk < in_sz / blk) {
        /* 如果目标块数小于自然块数，保持自适应块大小 */
        size_t nblk = (in_sz + blk - 1) / blk;
        *blk_sz_out = blk;
        *nblk_out = nblk;
    } else {
        /* 否则按OCC计算 */
        size_t nblk = (in_sz + tgt_blk - 1) / tgt_blk;
        if (nblk == 0) nblk = 1;
        blk = (in_sz + nblk - 1) / nblk;

        /* 对齐 */
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;

        nblk = (in_sz + blk - 1) / blk;
        *blk_sz_out = blk;
        *nblk_out = nblk;
    }

    if (debug) {
        fprintf(stderr, "[DAEMON] 自适应分块: blk=%zu KB, nblk=%zu, OCC=%zu, entropy=%.2f\n",
                *blk_sz_out / 1024, *nblk_out, occ_factor, entropy);
    }
}

/*
 * 守护进程压缩函数
 * 复用预分配的OpenCL资源,仅执行必要的压缩操作
 *
 * Phase 7.2更新:
 * - 自适应块大小算法
 * - Pinned Memory优化
 * - Buffer缓存机制
 *
 * level参数映射:
 *   1-3: lzo1x_1  (标准压缩)
 *   4-6: lzo1x_1k (1KB优化)
 *   7-8: lzo1x_1l (轻量级)
 *   9:   lzo1x_1o (最优压缩)
 */
int daemon_compress(
    /* OpenCL资源 (已初始化,复用) */
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,         // 根据level选择的kernel
    /* 请求参数 */
    const char* input_path,
    const char* output_path,
    int level,                // 压缩级别 1-9
    /* options */
    int standard_copy, int mt_io, int mt_threads, int fixed_block_kb, int force_map,
    /* 输出统计 */
    unsigned long* time_us_out,  // 总时间(微秒)
    size_t* output_size_out,
    /* 详细时间输出(微秒) */
    unsigned long* read_us_out,
    unsigned long* buffer_us_out,
    unsigned long* upload_us_out,
    unsigned long* kernel_us_out,
    unsigned long* download_us_out,
    unsigned long* write_us_out,
    unsigned long* cleanup_us_out
) {
    cl_int err;
    uint64_t t_total_start = now_ns();

    /* Debug开关 */
    int debug = getenv("LZO_DEBUG") != NULL;

    // 1. 获取文件大小
    uint64_t t_read_start = now_ns();

    FILE* f_in = fopen(input_path, "rb");
    if (!f_in) {
        perror("fopen input");
        return -1;
    }
    fseek(f_in, 0, SEEK_END);
    size_t in_sz = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    uint64_t t_file_open_end = now_ns();

    // 2. 准备输入缓冲区 (Pinned Memory) - 单独计时
    uint64_t t_buf_in_start = now_ns();
    cl_mem d_in = get_or_create_buffer(ctx, &buffer_cache.d_in, &buffer_cache.in_size,
                                       in_sz, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DAEMON] 创建输入缓冲区失败: %d\n", err);
        fclose(f_in);
        return -1;
    }

    /* Decide whether to use standard copy or zero-copy (mapping).
     * client can request standard_copy; force_map overrides
     */
    int use_standard_copy = standard_copy && !force_map ? 1 : 0;

    void* mapped_in = NULL;
    if (!use_standard_copy) {
        /* Map输入缓冲区用于写入 (zero-copy path) */
        mapped_in = clEnqueueMapBuffer(queue, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz,
                                       0, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DAEMON] Map输入缓冲区失败: %d\n", err);
            fclose(f_in);
            return -1;
        }
    }
    uint64_t t_buf_in_end = now_ns();
    unsigned long buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000;

    // 3. 读取文件并根据模式选择(零拷贝 -> mapped_in; 标准拷贝 -> host buffer + upload)
    uint64_t t_read_start2 = now_ns();
    unsigned long read_us = 0;
    unsigned long upload_us = 0;
    void* host_in = NULL;

    if (!use_standard_copy) {
        /* Zero-copy path: read directly into mapped_in. Support mt_io (pread workers) */
        if (mt_io && mt_threads > 1) {
            /* Use pread in multiple threads into mapped_in */
            int fd = open(input_path, O_RDONLY);
            if (fd < 0) {
                perror("open input");
                clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
                fclose(f_in);
                return -1;
            }

            int nthreads = mt_threads;
            if (nthreads < 1) nthreads = 1;
            if (nthreads > 32) nthreads = 32;

            mt_io_arg_t *args = malloc(sizeof(mt_io_arg_t) * nthreads);
            pthread_t *tids = malloc(sizeof(pthread_t) * nthreads);
            size_t piece = (in_sz + nthreads - 1) / nthreads;
            for (int i = 0; i < nthreads; ++i) {
                off_t off = (off_t)i * (off_t)piece;
                size_t len = ((size_t)off + piece > in_sz) ? (in_sz - (size_t)off) : piece;
                args[i].fd = fd;
                args[i].dest = mapped_in;
                args[i].off = off;
                args[i].len = len;
                args[i].err = 0;
                int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                if (rc != 0) {
                    /* fallback to single-thread read */
                    for (int j = 0; j < i; ++j) pthread_join(tids[j], NULL);
                    free(args); free(tids);
                    close(fd);
                    /* fallback to fread */
                    uint64_t tfr = now_ns();
                    if (fread(mapped_in, 1, in_sz, f_in) != in_sz) {
                        perror("fread input");
                        clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
                        fclose(f_in);
                        return -1;
                    }
                    uint64_t tfr_end = now_ns();
                    read_us = (tfr_end - tfr) / 1000;
                    close(fd);
                    goto after_zero_read;
                }
            }
            /* join */
            for (int i = 0; i < nthreads; ++i) pthread_join(tids[i], NULL);
            /* check errors */
            for (int i = 0; i < nthreads; ++i) {
                if (args[i].err != 0) {
                    errno = args[i].err;
                    perror("pread worker");
                    free(args); free(tids);
                    close(fd);
                    clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
                    fclose(f_in);
                    return -1;
                }
            }
            uint64_t t_read_end2 = now_ns();
            read_us = (t_read_end2 - t_read_start2) / 1000;
            free(args); free(tids);
            close(fd);

        } else {
            /* Single-threaded read into mapped_in */
            uint64_t tfr = now_ns();
            if (fread(mapped_in, 1, in_sz, f_in) != in_sz) {
                perror("fread input");
                clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
                fclose(f_in);
                return -1;
            }
            uint64_t tfr_end = now_ns();
            read_us = (tfr_end - tfr) / 1000;
        }

after_zero_read:
        fclose(f_in);
        /* Upload time is zero since we read directly into pinned mapped buffer */
        upload_us = 0;

    } else {
        /* Standard-copy path: read into host buffer then upload via clEnqueueWriteBuffer */
        /* allocate aligned host buffer */
        void* host_in = NULL;
        int rc_mem = posix_memalign(&host_in, ALIGN_BYTES, in_sz);
        if (rc_mem != 0 || host_in == NULL) {
            /* fallback to malloc */
            host_in = malloc(in_sz);
            if (!host_in) {
                perror("malloc host_in");
                fclose(f_in);
                return -1;
            }
        }

        if (mt_io && mt_threads > 1) {
            int fd = open(input_path, O_RDONLY);
            if (fd < 0) {
                perror("open input");
                free(host_in);
                fclose(f_in);
                return -1;
            }
            int nthreads = mt_threads;
            if (nthreads < 1) nthreads = 1;
            if (nthreads > 32) nthreads = 32;
            mt_io_arg_t *args = malloc(sizeof(mt_io_arg_t) * nthreads);
            pthread_t *tids = malloc(sizeof(pthread_t) * nthreads);
            size_t piece = (in_sz + nthreads - 1) / nthreads;
            for (int i = 0; i < nthreads; ++i) {
                off_t off = (off_t)i * (off_t)piece;
                size_t len = ((size_t)off + piece > in_sz) ? (in_sz - (size_t)off) : piece;
                args[i].fd = fd;
                args[i].dest = host_in;
                args[i].off = off;
                args[i].len = len;
                args[i].err = 0;
                int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
                if (rc != 0) {
                    for (int j = 0; j < i; ++j) pthread_join(tids[j], NULL);
                    free(args); free(tids);
                    close(fd);
                    /* fallback to fread into host buffer */
                    uint64_t tfr = now_ns();
                    if (fread(host_in, 1, in_sz, f_in) != in_sz) {
                        perror("fread input");
                        free(host_in);
                        fclose(f_in);
                        return -1;
                    }
                    uint64_t tfr_end = now_ns();
                    read_us = (tfr_end - tfr) / 1000;
                    close(fd);
                    goto after_std_read;
                }
            }
            for (int i = 0; i < nthreads; ++i) pthread_join(tids[i], NULL);
            for (int i = 0; i < nthreads; ++i) {
                if (args[i].err != 0) {
                    errno = args[i].err; perror("pread worker");
                    free(args); free(tids); close(fd);
                    free(host_in); fclose(f_in);
                    return -1;
                }
            }
            uint64_t t_read_end2 = now_ns();
            read_us = (t_read_end2 - t_read_start2) / 1000;
            free(args); free(tids); close(fd);

        } else {
            uint64_t tfr = now_ns();
            if (fread(host_in, 1, in_sz, f_in) != in_sz) {
                perror("fread input");
                free(host_in); fclose(f_in); return -1;
            }
            uint64_t tfr_end = now_ns();
            read_us = (tfr_end - tfr) / 1000;
        }

after_std_read:
        fclose(f_in);

        /* NOTE: do NOT upload yet; we need data available for entropy/blocking calc.
         * Upload will be performed after choose_blocking_adaptive so that the
         * blocking decision uses the raw host data.
         */
    }

    // 4. Phase 7.2: 自适应分块策略
    uint64_t t_blocking_start = now_ns();
    size_t blk, nblk;
    const unsigned char* entropy_ptr = NULL;
    if (use_standard_copy) {
        /* host_in exists in this branch (we kept it for later upload) */
        entropy_ptr = (const unsigned char*)host_in;
    } else {
        entropy_ptr = (const unsigned char*)mapped_in;
    }
    choose_blocking_adaptive(in_sz, entropy_ptr, device, debug, &blk, &nblk);
    uint64_t t_blocking_end = now_ns();
    unsigned long blocking_us = (t_blocking_end - t_blocking_start) / 1000;
    if (debug) {
        fprintf(stderr, "[DEBUG] blk=%zu, nblk=%zu, in_sz=%zu\n", blk, nblk, in_sz);
    }

    /* honor request-level fixed_block_kb if provided (overrides env) */
    if (fixed_block_kb > 0) {
        size_t env_blk = (size_t)fixed_block_kb * 1024;
        if (env_blk >= MIN_BLOCK_SIZE && env_blk <= MAX_BLOCK_SIZE) {
            if (debug) fprintf(stderr, "[DAEMON] 强制固定块大小 (request): %zu KB\n", env_blk / 1024);
            blk = env_blk;
            nblk = (in_sz + blk - 1) / blk;
        }
    }

    // If zero-copy (mapped_in set) we unmap now. If standard-copy, we still need
    // to upload host_in -> d_in so perform upload here.
    if (!use_standard_copy) {
        CHECK(clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL));
        /* upload_us remains 0 for zero-copy */
    } else {
        /* perform upload of host_in -> d_in now (measuring time) */
        uint64_t t_upload_start2 = now_ns();
        if (mt_io && mt_threads > 1) {
            int nthreads = mt_threads;
            if (nthreads < 1) nthreads = 1;
            if (nthreads > 32) nthreads = 32;
            cl_event *evts = malloc(sizeof(cl_event) * nthreads);
            size_t piece = (in_sz + nthreads - 1) / nthreads;
            for (int i = 0; i < nthreads; ++i) {
                size_t off = (size_t)i * piece;
                size_t len = (off + piece > in_sz) ? (in_sz - off) : piece;
                if (len == 0) { evts[i] = NULL; continue; }
                err = clEnqueueWriteBuffer(queue, d_in, CL_FALSE, off, len, (char*)host_in + off, 0, NULL, &evts[i]);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "[DAEMON] clEnqueueWriteBuffer failed: %d\n", err);
                    free(evts); free(host_in);
                    return -1;
                }
            }
            clWaitForEvents(nthreads, evts);
            for (int i = 0; i < nthreads; ++i) if (evts[i]) clReleaseEvent(evts[i]);
            free(evts);
        } else {
            err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "[DAEMON] clEnqueueWriteBuffer failed: %d\n", err);
                free(host_in);
                return -1;
            }
        }
        uint64_t t_upload_end2 = now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
        /* free host buffer */
        free(host_in);
    }

    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t in_needed = nblk * blk;
    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    // 5. Phase 7.2: 使用Buffer缓存 + Pinned Memory (输出和长度缓冲区)
    // d_in 已经准备好了（在步骤2创建）
    // buffer_in_us 已在步骤2计算

    uint64_t t_buf_out_start = now_ns();
    cl_mem d_out = get_or_create_buffer(ctx, &buffer_cache.d_out, &buffer_cache.out_size,
                                        out_needed, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DAEMON] 创建输出缓冲区失败: %d\n", err);
        return -1;
    }
    uint64_t t_buf_out_end = now_ns();
    unsigned long buffer_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    uint64_t t_buf_len_start = now_ns();
    cl_mem d_len = get_or_create_buffer(ctx, &buffer_cache.d_len, &buffer_cache.len_size,
                                        len_needed, CL_MEM_READ_WRITE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DAEMON] 创建长度缓冲区失败: %d\n", err);
        return -1;
    }
    uint64_t t_buf_len_end = now_ns();
    unsigned long buffer_len_us = (t_buf_len_end - t_buf_len_start) / 1000;

    unsigned long buffer_us = buffer_in_us + buffer_out_us + buffer_len_us;

    // 6. 数据上传 (已在步骤3中通过直接读取完成)
    // 保留变量名以兼容后续代码


    // 7. 设置内核参数 (使用cl_uint类型，与standalone完全一致)
    uint64_t t_kernel_start = now_ns();
    cl_uint in_sz_cl = (cl_uint)in_sz;
    cl_uint blk_cl = (cl_uint)blk;
    cl_uint worst_blk_cl = (cl_uint)worst_blk;

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));
    clFinish(queue);
    uint64_t t_kernel_end = now_ns();
    unsigned long kernel_setup_us = (t_kernel_end - t_kernel_start) / 1000;

    // 8. 执行内核
    uint64_t t_exec_start = now_ns();
    /* 压缩kernel必须使用local_size=1 */
    size_t lsz = 1;
    size_t gsz = ((nblk + lsz - 1) / lsz) * lsz;  /* round up to multiple of lsz */

    /*
    fprintf(stderr, "[DEBUG-KERNEL] 准备执行kernel: gsz=%zu, lsz=%zu, nblk=%zu\n", gsz, lsz, nblk);
    fprintf(stderr, "[DEBUG-KERNEL] Kernel参数: d_in=%p, d_out=%p, d_len=%p, in_sz=%u, blk=%u, worst_blk=%u\n",
            (void*)d_in, (void*)d_out, (void*)d_len, in_sz_cl, blk_cl, worst_blk_cl);
    */

    cl_event evt;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                                 0, NULL, &evt);
    CHECK(err);
    // fprintf(stderr, "[DEBUG-KERNEL] clEnqueueNDRangeKernel返回: %d\n", err);

    clWaitForEvents(1, &evt);

    /* 检查event执行状态 */
    cl_int evt_status;
    clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &evt_status, NULL);
    // fprintf(stderr, "[DEBUG-KERNEL] Event执行状态: %d (CL_COMPLETE=%d)\n", evt_status, CL_COMPLETE);

    /* 获取事件profiling信息 */
    cl_ulong ev_start = 0, ev_end = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(ev_start), &ev_start, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(ev_end), &ev_end, NULL);

    clReleaseEvent(evt);
    clFinish(queue);
    uint64_t t_exec_end = now_ns();
    unsigned long kernel_exec_us = (t_exec_end - t_exec_start) / 1000;

    // 9. Phase 6.1: 使用map读取结果 (零拷贝DMA)
    uint64_t t_download_start = now_ns();

    /* 读取长度数组 */
    uint64_t t_len_read_start = now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    void* mapped_len = clEnqueueMapBuffer(queue, d_len, CL_TRUE,
                                         CL_MAP_READ, 0, nblk * sizeof(cl_uint),
                                         0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, nblk * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_len, mapped_len, 0, NULL, NULL));
    uint64_t t_len_read_end = now_ns();
    unsigned long download_len_us = (t_len_read_end - t_len_read_start) / 1000;

    /* 计算总压缩大小 */
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) {
        comp_total += len_arr[i];
    }
    if (debug) {  /* 仅在debug模式输出 */
        fprintf(stderr, "[DEBUG-LENS] comp_total=%zu, out_needed=%zu\n", comp_total, out_needed);
        fprintf(stderr, "[DEBUG-LENS] 前10个block长度: ");
        for (size_t i = 0; i < 10 && i < nblk; i++) {
            fprintf(stderr, "[%zu]=%u ", i, len_arr[i]);
        }
        fprintf(stderr, "\n");
    }

    /* 读取压缩数据并打包 - 优化: 移除中间buffer，直接使用mapped_out */
    /* 注意: d_out是稀疏的(每个block占worst_blk空间) */
    uint64_t t_bulk_read_start = now_ns();
    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE,
                                         CL_MAP_READ, 0, out_needed,
                                         0, NULL, NULL, &err);
    CHECK(err);
    uint64_t t_bulk_read_end = now_ns();
    unsigned long download_bulk_us = (t_bulk_read_end - t_bulk_read_start) / 1000;

    /* 移除 comp_buf 分配和 memcpy */
    // unsigned char* comp_buf = malloc(comp_total);
    // ... memcpy loop removed ...

    uint64_t t_download_end = now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    // 10. 写入输出文件
    uint64_t t_write_start = now_ns();
    int write_ret = write_compressed_file(output_path,
                                         in_sz, blk, nblk, len_arr,
                                         mapped_out, worst_blk);

    /* Unmap after writing */
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));

    uint64_t t_write_end = now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    // 11. 清理临时内存
    uint64_t t_cleanup_start = now_ns();
    // free(in_buf); // 已移除
    free(len_arr);
    // free(comp_buf); // 已移除

    /* 关键修复：强制flush queue，确保所有pending操作完成 */
    clFlush(queue);
    clFinish(queue);

    /* 实验：暂时不释放buffers，保持缓存 (性能优化) */
    // fprintf(stderr, "[DEBUG-CLEANUP] 暂不释放buffers，保持缓存\n");

    /*
    if (buffer_cache.d_in) {
        clReleaseMemObject(buffer_cache.d_in);
        buffer_cache.d_in = NULL;
        buffer_cache.in_size = 0;
    }
    if (buffer_cache.d_out) {
        clReleaseMemObject(buffer_cache.d_out);
        buffer_cache.d_out = NULL;
        buffer_cache.out_size = 0;
    }
    if (buffer_cache.d_len) {
        clReleaseMemObject(buffer_cache.d_len);
        buffer_cache.d_len = NULL;
        buffer_cache.len_size = 0;
    }
    */

    uint64_t t_cleanup_end = now_ns();
    unsigned long cleanup_us = (t_cleanup_end - t_cleanup_start) / 1000;

    // 12. 统计输出
    uint64_t t_total_end = now_ns();
    unsigned long total_us = (t_total_end - t_total_start) / 1000;

    *time_us_out = total_us;
    *output_size_out = comp_total;
    *read_us_out = read_us;
    *buffer_us_out = buffer_us;
    *upload_us_out = upload_us;  // 实际为0（直接 fread 到 Pinned Memory）
    *kernel_us_out = kernel_setup_us + kernel_exec_us;
    *download_us_out = download_us;
    *write_us_out = write_us;
    *cleanup_us_out = cleanup_us;

    /* 输出统计信息 (always print for consistency) */
    double ratio = comp_total > 0 ? (double)in_sz / (double)comp_total : 0.0;
    long long space_diff = (long long)in_sz - (long long)comp_total;

    fprintf(stderr, "\n=== Compression Statistics ===\n");
    fprintf(stderr, "Input size       : %zu bytes (%.2f MB)\n", in_sz, in_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", comp_total, comp_total / (1024.0 * 1024.0));
    fprintf(stderr, "Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, 100.0 / ratio);
    fprintf(stderr, "Space saved      : %lld bytes (%.2f MB, %.1f%%)\n",
           space_diff,
           space_diff / (1024.0 * 1024.0),
           100.0 * (1.0 - 1.0/ratio));
    fprintf(stderr, "Block size       : %zu bytes (%zu KB)\n", blk, blk / 1024);
    fprintf(stderr, "Number of blocks : %zu\n", nblk);
    fprintf(stderr, "Compression level: %d\n", level);
    fprintf(stderr, "Work groups      : global=%zu, local=auto\n", gsz);
    double kernel_thrpt = kernel_exec_us > 0 ? ((double)in_sz / (1024.0*1024.0)) / (kernel_exec_us/1000000.0) : 0.0;
    fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n",
           ((double)in_sz / (1024.0*1024.0)) / (total_us/1000000.0),
           kernel_thrpt);
    fprintf(stderr, "==============================\n\n");

    /* 计算基于事件的内核时间（μs） */
    unsigned long exec_us_ev = 0;
    if (ev_start != 0 && ev_end != 0 && ev_end > ev_start) {
        exec_us_ev = (unsigned long)((ev_end - ev_start) / 1000);
    }

    /* 打印详细的时间分解（对齐 standalone 格式）*/
    fprintf(stderr, "\n=== Time Breakdown (Compression) ===\n");
    fprintf(stderr, "1. File Read           : %8.3f ms\n", read_us / 1000.0);
    fprintf(stderr, "2. Blocking Calc       : %8.3f ms\n", blocking_us / 1000.0);
    fprintf(stderr, "3. Buffer Alloc (in)   : %8.3f ms\n", buffer_in_us / 1000.0);
    fprintf(stderr, "4. Data Upload         : %8.3f ms\n", upload_us / 1000.0);
    fprintf(stderr, "5. Buffer Alloc (out)  : %8.3f ms\n", buffer_out_us / 1000.0);
    fprintf(stderr, "6. Buffer Alloc (len)  : %8.3f ms\n", buffer_len_us / 1000.0);
    fprintf(stderr, "7. Setup Args          : %8.3f ms\n", kernel_setup_us / 1000.0);
    fprintf(stderr, "8. Kernel Exec         : %8.3f ms\n", kernel_exec_us / 1000.0);
    if (exec_us_ev) {
        fprintf(stderr, "   (event profiling)   : %8.3f ms\n", exec_us_ev / 1000.0);
    }
    fprintf(stderr, "9. Download (len)      : %8.3f ms\n", download_len_us / 1000.0);
    fprintf(stderr, "10. Download (bulk)    : %8.3f ms\n", download_bulk_us / 1000.0);
    fprintf(stderr, "11. Download Total     : %8.3f ms\n", download_us / 1000.0);
    fprintf(stderr, "12. File Write         : %8.3f ms\n", write_us / 1000.0);
    fprintf(stderr, "13. Cleanup            : %8.3f ms\n", cleanup_us / 1000.0);
    fprintf(stderr, "TOTAL                  : %8.3f ms\n", total_us / 1000.0);
    fprintf(stderr, "\n");

    /* 计算占比（对齐 standalone 格式）*/
    fprintf(stderr, "=== Percentage Breakdown ===\n");
    fprintf(stderr, "Kernel Exec     : %6.2f%%\n", 100.0 * kernel_exec_us / total_us);
    fprintf(stderr, "Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
           100.0 * (upload_us + download_us) / total_us,
           100.0 * upload_us / total_us,
           100.0 * download_us / total_us);
    fprintf(stderr, "File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
           100.0 * (read_us + write_us) / total_us,
           100.0 * read_us / total_us,
           100.0 * write_us / total_us);
    fprintf(stderr, "Buffer Alloc    : %6.2f%% (in=%.2f%% + out=%.2f%% + len=%.2f%%)\n",
           100.0 * buffer_us / total_us,
           100.0 * buffer_in_us / total_us,
           100.0 * buffer_out_us / total_us,
           100.0 * buffer_len_us / total_us);
    fprintf(stderr, "Setup Args      : %6.2f%%\n", 100.0 * kernel_setup_us / total_us);
    fprintf(stderr, "Blocking Calc   : %6.2f%%\n", 100.0 * blocking_us / total_us);
    fprintf(stderr, "\n");

    return write_ret;
}
