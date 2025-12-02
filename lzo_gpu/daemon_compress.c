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
#include "timing.h"
#include "lzo_defaults.h"
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

/* Async upload context for daemon: same idea as standalone — hold events,
 * host buffer pointer and a completion timestamp pointer, and a reaper
 * thread will wait for write events, free host buffer, then wait for a
 * kernel-enqueue signal before releasing events.
 */
typedef struct async_upload_ctx {
    cl_event *events;
    int n;
    void *host_ptr; /* pointer to host buffer to free when upload finished */
    uint64_t *t_upload_end_ptr;
} async_upload_ctx_t;

static void* async_upload_reaper(void *v) {
    async_upload_ctx_t *ctx = (async_upload_ctx_t*)v;
    if (!ctx) return NULL;
    /* Lightweight debug log to help trace async upload lifecycle */
    fprintf(stderr, "[DAEMON] async_upload_reaper: events=%p n=%d host_ptr=%p t_ptr=%p\n",
            (void*)ctx->events, ctx->n, ctx->host_ptr, (void*)ctx->t_upload_end_ptr);
    if (ctx->n > 0 && ctx->events) {
        if (getenv("LZO_DEBUG")) fprintf(stderr, "[DAEMON][REAPER] waiting for %d events at %p\n", ctx->n, (void*)ctx->events);
        clWaitForEvents(ctx->n, ctx->events);
        if (getenv("LZO_DEBUG")) fprintf(stderr, "[DAEMON][REAPER] wait complete for events %p\n", (void*)ctx->events);
    }
    if (ctx->t_upload_end_ptr) *ctx->t_upload_end_ptr = now_ns();
    if (ctx->host_ptr) {
        fprintf(stderr, "[DAEMON][REAPER] freeing host_ptr=%p\n", ctx->host_ptr);
        free(ctx->host_ptr); ctx->host_ptr = NULL; }
    return NULL;
}

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
                                 const void* sparse_data, size_t worst_blk,
                                 int coalesce, size_t coalesce_chunk_mb, size_t coalesce_max_mb,
                                 size_t stdio_buf_mb) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    /* 配置 stdio 缓冲区大小，降低 fwrite 小块写入带来的 syscalls 波动。
     * 默认 LZO_DEFAULT_STDIO_BUF_MB=4MB。不要从环境中读值以减少 env 变量。
     */
    int debug = getenv("LZO_DEBUG") != NULL;
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) comp_total += lens[i];
    size_t buf_mb = stdio_buf_mb;
    size_t vsize = buf_mb * 1024 * 1024;
    if (vsize == 0) vsize = 4 * 1024 * 1024;
    if (vsize > comp_total && comp_total > 0) vsize = comp_total;
    char *vbuf = NULL;
    if (vsize > 0) {
        vbuf = (char*)malloc(vsize);
        if (vbuf) {
            /* best-effort: if setvbuf fails just continue with defaults */
            if (setvbuf(f, vbuf, _IOFBF, (int)vsize) != 0) {
                free(vbuf);
                vbuf = NULL;
            }
        }
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

    /* Default: enable coalescing; avoid attempting full coalesces for outputs
     * larger than LZO_DEFAULT_COALESCE_MAX_MB to avoid OOM or excessively large
     * allocations. Chunk size is LZO_DEFAULT_COALESCE_CHUNK_MB for fallback.
     */
    /* coalesce/coalesce_max/coalesce_chunk_mb/stdio_buf_mb are passed as parameters */
    /* coalesce variable is already the value of the parameter */
    int profile_writes = debug; /* profile write diagnostics gated by debug flag */

    if (coalesce && comp_total > 0) {
        /* If total compressed size is larger than the permitted single-coalesce threshold,
         * avoid attempting a single large allocation and go straight to chunked coalesce.
         */
        size_t threshold_bytes = coalesce_max_mb * 1024 * 1024;
        unsigned char* contig = NULL;
        int attempted_full_coalesce = 0;
        if (threshold_bytes == 0 || comp_total <= threshold_bytes) {
            attempted_full_coalesce = 1;
            contig = (unsigned char*)malloc(comp_total);
        }
            if (contig) {
                if (debug) fprintf(stderr, "[DAEMON] COALESCE: full contiguous allocation success size=%zu bytes\n", comp_total);
            uint64_t t_copy_start = now_ns();
            size_t pos = 0;
            for (size_t i = 0; i < nblk; ++i) {
                if (lens[i] > 0) {
                    size_t dev_off = i * worst_blk;
                    memcpy(contig + pos, dev_out + dev_off, lens[i]);
                    pos += lens[i];
                }
            }
            uint64_t t_copy_end = now_ns();
            uint64_t t_write_start_blk = now_ns();
            if (fwrite(contig, 1, comp_total, f) != comp_total) {
                perror("fwrite contiguous");
                free(contig);
                fclose(f);
                if (vbuf) { free(vbuf); vbuf = NULL; }
                return -1;
            }
            uint64_t t_write_end_blk = now_ns();
            if (profile_writes) {
                fprintf(stderr, "COALESCE_COPY: %.3f ms\n", (t_copy_end - t_copy_start)/1e6);
                fprintf(stderr, "COALESCE_WRITE: %.3f ms\n", (t_write_end_blk - t_write_start_blk)/1e6);
            }
            free(contig);
        } else {
            if (debug) fprintf(stderr, "[DAEMON] COALESCE: full contiguous allocation failed, attempting chunked coalesce\n");
            /* allocation failed - try a chunked coalesce approach to avoid many small fwrite syscalls
             * This attempts to allocate a modest chunk buffer and stream groups of blocks into it,
             * falling back to per-block writes only if that also fails.
             */
            size_t chunk_mb = coalesce_chunk_mb; /* default chunk size 16MB or override */
            size_t chunk_size = chunk_mb * 1024 * 1024;
            if (chunk_size == 0) chunk_size = 16 * 1024 * 1024;
            unsigned char* chunk = (unsigned char*)malloc(chunk_size);
            if (chunk) {
                if (debug) fprintf(stderr, "[DAEMON] COALESCE: chunk buffer allocation success chunk_mb=%zu chunk_size=%zu bytes\n", chunk_mb, chunk_size);
                size_t used = 0;
                uint64_t t_copy_total_start = now_ns();
                for (size_t i = 0; i < nblk; ++i) {
                    if (lens[i] == 0) continue;
                    size_t dev_off = i * worst_blk;
                    /* if single block larger than chunk and chunk empty, write it directly */
                    if (lens[i] > chunk_size && used == 0) {
                        uint64_t t1 = now_ns();
                        if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                            perror("fwrite block large");
                            free(chunk);
                            fclose(f);
                            if (vbuf) { free(vbuf); vbuf = NULL; }
                            return -1;
                        }
                        uint64_t t2 = now_ns();
                        if (profile_writes) fprintf(stderr, "BLOCK_WRITE %zu len=%u : %.3f ms\n", i, lens[i], (t2 - t1)/1e6);
                        continue;
                    }
                    /* if not enough room, flush current chunk */
                    if (used + lens[i] > chunk_size) {
                        uint64_t t_write_chunk_start = now_ns();
                        if (fwrite(chunk, 1, used, f) != used) {
                            perror("fwrite chunk");
                            free(chunk);
                            fclose(f);
                            if (vbuf) { free(vbuf); vbuf = NULL; }
                            return -1;
                        }
                        uint64_t t_write_chunk_end = now_ns();
                        if (profile_writes) fprintf(stderr, "CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_write_chunk_end - t_write_chunk_start)/1e6, used);
                        used = 0;
                    }
                    memcpy(chunk + used, dev_out + dev_off, lens[i]);
                    used += lens[i];
                }
                /* flush remainder */
                if (used > 0) {
                    uint64_t t_write_chunk_start = now_ns();
                    if (fwrite(chunk, 1, used, f) != used) {
                        perror("fwrite chunk final");
                        free(chunk);
                        fclose(f);
                        if (vbuf) { free(vbuf); vbuf = NULL; }
                        return -1;
                    }
                    uint64_t t_write_chunk_end = now_ns();
                    if (profile_writes) fprintf(stderr, "CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_write_chunk_end - t_write_chunk_start)/1e6, used);
                }
                free(chunk);
            } else {
                if (debug) fprintf(stderr, "[DAEMON] COALESCE: chunk allocation failed, falling back to per-block writes\n");
                /* fallback to scatter writes */
                fprintf(stderr, "[DAEMON] warning: coalesce allocation and chunk alloc both failed (%zu bytes), falling back to per-block writes\n", comp_total);
                for (size_t i = 0; i < nblk; i++) {
                    if (lens[i] > 0) {
                        size_t dev_off = i * worst_blk;
                        if (profile_writes) {
                            uint64_t t1 = now_ns();
                            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                                perror("fwrite block");
                                fclose(f);
                                return -1;
                            }
                            uint64_t t2 = now_ns();
                            fprintf(stderr, "BLOCK_WRITE %zu len=%u : %.3f ms\n", i, lens[i], (t2 - t1)/1e6);
                        } else {
                            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                                perror("fwrite block");
                                fclose(f);
                                return -1;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < nblk; i++) {
            if (lens[i] > 0) {
                size_t dev_off = i * worst_blk;
                if (profile_writes) {
                    uint64_t t1 = now_ns();
                    if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                        perror("fwrite block");
                        fclose(f);
                        return -1;
                    }
                    uint64_t t2 = now_ns();
                    fprintf(stderr, "BLOCK_WRITE %zu len=%u : %.3f ms\n", i, lens[i], (t2 - t1)/1e6);
                } else {
                    if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                        perror("fwrite block");
                        fclose(f);
                        return -1;
                    }
                }
            }
        }
    }

    fclose(f);
    if (vbuf) {
        free(vbuf);
        vbuf = NULL;
    }
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
        /* Always use pinned host mapping for buffers to preserve zero-copy path */
        cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags,
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
    int standard_copy, int mt_io, int mt_threads, int fixed_block_kb, int async_upload,
    /* per-request overrides */
    int coalesce_output, int coalesce_chunk_mb, int coalesce_max_mb, int stdio_buf_mb,
    /* 输出统计 */
    unsigned long* time_us_out,  // 总时间(微秒)
    size_t* output_size_out,
    /* 详细时间输出(微秒) - per-stage fields bundle */
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_total_start = now_ns();

    /* Debug开关 */
    int debug = getenv("LZO_DEBUG") != NULL;

    /* Apply per-request overrides for coalescing and stdio buffer.
     * Pass-through semantics: -1 means unspecified (use daemon-level env override -> compile-time defaults)
     * To allow administrators to alter daemon-default behavior without patching, we honor these
     * environment variables (if present): LZO_COALESCE_OUTPUT, LZO_COALESCE_MAX_MB,
     * LZO_COALESCE_CHUNK_MB, LZO_STDIO_BUF_MB. If not present, fall back to compiled-in defaults.
     */
    const char* sopt = NULL;
    int default_coalesce = LZO_DEFAULT_COALESCE_OUTPUT;
    if ((sopt = getenv("LZO_COALESCE_OUTPUT"))) default_coalesce = (atoi(sopt) != 0) ? 1 : 0;
    size_t default_coalesce_max_mb = (size_t)LZO_DEFAULT_COALESCE_MAX_MB;
    if ((sopt = getenv("LZO_COALESCE_MAX_MB"))) {
        int v = atoi(sopt); if (v > 0) default_coalesce_max_mb = (size_t)v;
    }
    size_t default_coalesce_chunk_mb = (size_t)LZO_DEFAULT_COALESCE_CHUNK_MB;
    if ((sopt = getenv("LZO_COALESCE_CHUNK_MB"))) {
        int v = atoi(sopt); if (v > 0) default_coalesce_chunk_mb = (size_t)v;
    }
    size_t default_stdio_buf_mb = (size_t)LZO_DEFAULT_STDIO_BUF_MB;
    if ((sopt = getenv("LZO_STDIO_BUF_MB"))) {
        int v = atoi(sopt); if (v > 0) default_stdio_buf_mb = (size_t)v;
    }

    int coalesce = (coalesce_output == -1) ? default_coalesce : coalesce_output;
    size_t coalesce_max_mb_local = (coalesce_max_mb == -1) ? default_coalesce_max_mb : (size_t)coalesce_max_mb;
    size_t coalesce_chunk_mb_local = (coalesce_chunk_mb == -1) ? default_coalesce_chunk_mb : (size_t)coalesce_chunk_mb;
    size_t stdio_buf_mb_local = (stdio_buf_mb == -1) ? default_stdio_buf_mb : (size_t)stdio_buf_mb;

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
     * client requests standard_copy (explicit uploads) when set; default is zero-copy (mapping).
     */
    int use_standard_copy = standard_copy ? 1 : 0;

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
    /* async upload bookkeeping (visible across scopes) */
    uint64_t t_upload_start2 = 0;
    int reaper_started = 0;
    pthread_t upload_reaper_tid = 0;
    async_upload_ctx_t *upload_ctx = NULL;
    uint64_t *t_upload_end_ptr = NULL;

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
        /* allocate aligned host buffer into the outer-scoped `host_in` */
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
        /* host_in now points to the allocated buffer (no shadowing) */
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
        t_upload_start2 = now_ns();
        /* async upload bookkeeping: reuse outer-scope variables declared above */
        if (mt_io && mt_threads > 1) {
            int nthreads = mt_threads;
            if (nthreads < 1) nthreads = 1;
            if (nthreads > 32) nthreads = 32;
            cl_event *evts = malloc(sizeof(cl_event) * nthreads);
            int ev_count = 0;
            size_t piece = (in_sz + nthreads - 1) / nthreads;
            for (int i = 0; i < nthreads; ++i) {
                size_t off = (size_t)i * piece;
                size_t len = (off + piece > in_sz) ? (in_sz - off) : piece;
                if (len == 0) { continue; }
                err = clEnqueueWriteBuffer(queue, d_in, CL_FALSE, off, len, (char*)host_in + off, 0, NULL, &evts[ev_count]);
                if (err == CL_SUCCESS) {
                    if (debug) fprintf(stderr, "[DAEMON] enqueued write evt[%d] off=%zu len=%zu\n", ev_count, off, len);
                    ev_count++;
                } else {
                    fprintf(stderr, "[DAEMON] clEnqueueWriteBuffer failed (segment) : %d\n", err);
                    /* immediate cleanup on failure */
                    for (int j = 0; j < ev_count; ++j) if (evts[j]) clReleaseEvent(evts[j]);
                    free(evts);
                    free(host_in);
                    return -1;
                }
            }
                if (async_upload) {
                    if (ev_count == 0) {
                        /* Nothing was enqueued (all zero-length); treat as completed */
                        if (debug) fprintf(stderr, "[DAEMON] async_upload requested but ev_count==0, skipping reaper\n");
                        upload_us = 0;
                        free(evts);
                        free(host_in);
                        host_in = NULL;
                        goto after_std_read_upload_done;
                    }
                /* spawn a reaper thread to wait for enqueued writes and free host buffer */
                t_upload_end_ptr = malloc(sizeof(uint64_t)); *t_upload_end_ptr = 0;
                upload_ctx = calloc(1, sizeof(async_upload_ctx_t));
                /* only pass the real number of events (non-zero partitions) */
                upload_ctx->events = evts;
                upload_ctx->n = ev_count;
                upload_ctx->host_ptr = host_in; /* will be freed by reaper */
                upload_ctx->t_upload_end_ptr = t_upload_end_ptr;
                /* main thread will join/cleanup ctx and events */
                if (pthread_create(&upload_reaper_tid, NULL, async_upload_reaper, upload_ctx) != 0) {
                    /* fallback to blocking behavior */
                    clWaitForEvents(ev_count, evts);
                    *t_upload_end_ptr = now_ns();
                    for (int i = 0; i < ev_count; ++i) if (evts[i]) clReleaseEvent(evts[i]);
                    free(evts);
                    free(t_upload_end_ptr); t_upload_end_ptr = NULL;
                    free(upload_ctx); upload_ctx = NULL;
                } else {
                    reaper_started = 1; /* reaper handles host_in free */
                }
            } else {
                clWaitForEvents(ev_count, evts);
                for (int i = 0; i < ev_count; ++i) if (evts[i]) clReleaseEvent(evts[i]);
                free(evts);
                free(host_in); host_in = NULL;
            }
        } else {
            if (async_upload) {
                cl_event *evts = malloc(sizeof(cl_event));
                err = clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, in_sz, host_in, 0, NULL, &evts[0]);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "[DAEMON] clEnqueueWriteBuffer failed: %d\n", err);
                    free(evts); free(host_in);
                    return -1;
                }
                t_upload_end_ptr = malloc(sizeof(uint64_t)); *t_upload_end_ptr = 0;
                upload_ctx = calloc(1, sizeof(async_upload_ctx_t));
                upload_ctx->events = evts;
                upload_ctx->n = 1;
                upload_ctx->host_ptr = host_in; /* reaper will free */
                upload_ctx->t_upload_end_ptr = t_upload_end_ptr;
                /* main thread will join/cleanup ctx and events */
                if (pthread_create(&upload_reaper_tid, NULL, async_upload_reaper, upload_ctx) != 0) {
                    /* fallback to blocking */
                    clWaitForEvents(1, evts);
                    *t_upload_end_ptr = now_ns();
                    clReleaseEvent(evts[0]); free(evts);
                    free(t_upload_end_ptr); t_upload_end_ptr = NULL;
                    free(upload_ctx); upload_ctx = NULL;
                    free(host_in); host_in = NULL;
                } else {
                    reaper_started = 1;
                }
            } else {
                err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "[DAEMON] clEnqueueWriteBuffer failed: %d\n", err);
                    free(host_in);
                    return -1;
                }
            }
        }
    after_std_read_upload_done:
        uint64_t t_upload_end2 = now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
        /* free host buffer only when a reaper did not take ownership */
        if (!reaper_started && host_in) {
            free(host_in);
            host_in = NULL;
        }
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
    if (upload_ctx && upload_ctx->n > 0 && upload_ctx->events) {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                                     upload_ctx->n, upload_ctx->events, &evt);
        /* reaper will not free events/ctx; main will cleanup after join */
    } else {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                                     0, NULL, &evt);
    }
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
                                         mapped_out, worst_blk,
                                         coalesce, coalesce_chunk_mb_local, coalesce_max_mb_local,
                                         stdio_buf_mb_local);

    /* Unmap after writing */
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));

    uint64_t t_write_end = now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    // 如果我们启动了异步上传 reaper，先 join 它以获取上传完成时间
    if (reaper_started && upload_reaper_tid) {
        pthread_join(upload_reaper_tid, NULL);
        upload_reaper_tid = 0;
        if (t_upload_end_ptr) {
            uint64_t t_upload_end = *t_upload_end_ptr;
            upload_us = (t_upload_end - t_upload_start2) / 1000;
            free(t_upload_end_ptr); t_upload_end_ptr = NULL;
        }
        /* release events & free ctx (reaper freed host buffer) */
        if (upload_ctx) {
            if (upload_ctx->events) {
                for (int _i = 0; _i < upload_ctx->n; ++_i) if (upload_ctx->events[_i]) clReleaseEvent(upload_ctx->events[_i]);
                free(upload_ctx->events); upload_ctx->events = NULL; upload_ctx->n = 0;
            }
            free(upload_ctx); upload_ctx = NULL;
        }
    }

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
    if (t_out) {
        t_out->file_read_us = read_us;
        /* daemon preinitializes OpenCL at startup - do not count OCL init per-request */
        t_out->ocl_init_us = 0;
        /* kernel programs are pre-loaded during daemon startup */
        t_out->kernel_load_us = 0;
        t_out->blocking_calc_us = blocking_us;
        t_out->buffer_alloc_in_us = buffer_in_us;
        t_out->data_upload_us = upload_us;
        t_out->buffer_alloc_out_us = buffer_out_us;
        t_out->buffer_alloc_len_us = buffer_len_us;
        t_out->setup_args_us = kernel_setup_us;
        t_out->kernel_setup_us = kernel_setup_us;
        t_out->kernel_exec_us = kernel_exec_us;
        t_out->download_len_us = download_len_us;
        t_out->download_bulk_us = download_bulk_us;
        t_out->download_total_us = download_us;
        t_out->file_write_us = write_us;
        t_out->cleanup_us = cleanup_us;
    }

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

        /* 计算占比（对齐 standalone 格式） - 保护除以零 */
        double denom = (total_us > 0) ? (double)total_us : 1.0;
        int zero_total = (total_us == 0);
        fprintf(stderr, "=== Percentage Breakdown ===\n");
        fprintf(stderr, "Kernel Exec     : %6.2f%%\n", zero_total ? 0.0 : 100.0 * kernel_exec_us / denom);
        fprintf(stderr, "Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
            zero_total ? 0.0 : 100.0 * (upload_us + download_us) / denom,
            zero_total ? 0.0 : 100.0 * upload_us / denom,
            zero_total ? 0.0 : 100.0 * download_us / denom);
        fprintf(stderr, "File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
            zero_total ? 0.0 : 100.0 * (read_us + write_us) / denom,
            zero_total ? 0.0 : 100.0 * read_us / denom,
            zero_total ? 0.0 : 100.0 * write_us / denom);
        fprintf(stderr, "Buffer Alloc    : %6.2f%% (in=%.2f%% + out=%.2f%% + len=%.2f%%)\n",
            zero_total ? 0.0 : 100.0 * buffer_us / denom,
            zero_total ? 0.0 : 100.0 * buffer_in_us / denom,
            zero_total ? 0.0 : 100.0 * buffer_out_us / denom,
            zero_total ? 0.0 : 100.0 * buffer_len_us / denom);
        fprintf(stderr, "Setup Args      : %6.2f%%\n", zero_total ? 0.0 : 100.0 * kernel_setup_us / denom);
        fprintf(stderr, "Blocking Calc   : %6.2f%%\n", zero_total ? 0.0 : 100.0 * blocking_us / denom);
    fprintf(stderr, "\n");

    return write_ret;
}
