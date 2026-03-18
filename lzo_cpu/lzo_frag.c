/*
 * lzo_frag.c -- CPU driver mirroring lzo_gpu CLI semantics
 * Supports runtime compression level selection, threaded compression
 * and decompression, containerized I/O compatible with the GPU tool,
 * and an opt-in benchmark mode built from the original test harness.
 */

#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdbool.h>
#include <time.h>
#include <stdatomic.h>
#include <unistd.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#include <malloc.h>
#endif

#include <lzo/lzo1x.h>
#include <lzo/lzo1y.h>
#include "lzo_levels.h"

#define MAGIC_TAG            0x4C5A       /* 'L''Z' */
#define DEFAULT_THREAD_COUNT 0  /* 0 = auto-detect at runtime */
#define MIN_BLOCK_SIZE       (64u * 1024u)
#define MAX_BLOCK_SIZE       (512u * 1024u)
#define LZO_WORK_MEM_SIZE    LZO1X_1_MEM_COMPRESS

typedef struct {
    const unsigned char *in;
    size_t in_size;
    unsigned char *comp;
    size_t comp_size;
    size_t offset;
    unsigned char *out;
    uint64_t proc_ns; /* per-chunk processing time in nanoseconds (filled when --debug-metrics enabled) */
} chunk_t;

/* Global algorithm specifier set from -L. When non-NULL it overrides numeric
 * compression level selection inside compress_block_level(). Expected values
 * are labels like "1x", "1k", "1o", "1l".
 */
static const char *g_alg_spec = NULL;
static int cpu_debug_metrics_enabled = 0; /* set by --debug-metrics */
typedef enum {
    ALG_1X = 0,
    ALG_1Y = 1,
} alg_t;

typedef enum {
    VAR_1 = 1,
    VAR_1K = 2,
    VAR_1L = 3,
    VAR_1O = 4,
    VAR_999 = 999,
} variant_t;

static alg_t g_alg = ALG_1X;
static variant_t g_variant = VAR_1;

static alg_t alg_from_string(const char *s) {
    if (!s) return ALG_1X;
    if (strcasecmp(s, "1x") == 0 || strcasecmp(s, "lzo1x") == 0) return ALG_1X;
    if (strcasecmp(s, "1y") == 0 || strcasecmp(s, "lzo1y") == 0) return ALG_1Y;
    return ALG_1X;
}

static variant_t variant_from_string(const char *s) {
    if (!s) return VAR_1;
    /* Accept string variants */
    if (strcasecmp(s, "1") == 0) return VAR_1;
    if (strcasecmp(s, "1k") == 0) return VAR_1K;
    if (strcasecmp(s, "1l") == 0) return VAR_1L;
    if (strcasecmp(s, "1o") == 0) return VAR_1O;
    if (strcasecmp(s, "999") == 0) return VAR_999;
    /* Accept numeric level strings: 11->1k, 12->1l, 13->1o, 14->1 */
    if (strcasecmp(s, "11") == 0) return VAR_1K;
    if (strcasecmp(s, "12") == 0) return VAR_1L;
    if (strcasecmp(s, "13") == 0) return VAR_1O;
    if (strcasecmp(s, "14") == 0) return VAR_1;
    return VAR_1;
}

static size_t workmem_size_for(alg_t alg, variant_t variant) {
    if (variant == VAR_999) {
        return (alg == ALG_1Y) ? (size_t)LZO1Y_999_MEM_COMPRESS : (size_t)LZO1X_999_MEM_COMPRESS;
    }
    if (alg == ALG_1Y) return (size_t)LZO1Y_MEM_COMPRESS;
    if (variant == VAR_1K) return (size_t)LZO1X_1_12_MEM_COMPRESS;
    if (variant == VAR_1L) return (size_t)LZO1X_1_11_MEM_COMPRESS;
    if (variant == VAR_1O) return (size_t)LZO1X_1_15_MEM_COMPRESS;
    return (size_t)LZO1X_1_MEM_COMPRESS;
}

static const char *alg_to_str(alg_t a) {
    switch (a) {
        case ALG_1X: return "1x";
        case ALG_1Y: return "1y";
        default: return "unknown";
    }
}
static const char *alg_to_str(alg_t a);

typedef struct {
    chunk_t *chunks;
    size_t chunk_count;
    _Atomic size_t next_index;
    alg_t compression_alg;
    variant_t variant;
    _Atomic int status;
    pthread_mutex_t lock;
} compress_job_t;

typedef struct {
    chunk_t *chunks;
    size_t chunk_count;
    _Atomic size_t next_index;
    _Atomic int status;
    pthread_mutex_t lock;
    alg_t alg;
} decompress_job_t;

#define HEAP_ALLOC(var, size) \
    lzo_align_t __LZO_MMODEL var[((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t)]

/* Global algorithm specifier set from -L. When non-NULL it overrides numeric
 * compression level selection inside compress_block_level(). Expected values
 * are labels like "1x", "1k", "1o", "1l".
 */
/* Definitions for algorithm helpers (moved above) */




static uint16_t read_u16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static void write_u16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8) & 0xFFu);
}

static uint32_t read_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void write_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);
    p[3] = (unsigned char)((v >> 24) & 0xFFu);
}

/* Monotonic timespec diff in milliseconds */
static double diff_ms_ts(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

/* Helper to format a millisecond value into ms or us string depending on magnitude.
 * Writes a compact string into 'buf' (buflen should be at least 32).
 * For ms >= 1.0, shows '%.3f ms'; for 0 < ms < 1.0 shows '%.0f us'; for 0, prints 'N/A'.
 */
static void format_ms_or_us(char *buf, size_t buflen, double ms) {
    if (!buf || buflen == 0) return;
    if (ms <= 0.0) {
        snprintf(buf, buflen, "N/A");
    } else if (ms >= 1.0) {
        snprintf(buf, buflen, "%.3f ms", ms);
    } else {
        snprintf(buf, buflen, "%.0f us", ms * 1000.0);
    }
}

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_double(const double *vals, size_t n) {
    if (!vals || n == 0) return 0.0;
    double *tmp = (double *)malloc(n * sizeof(double));
    if (!tmp) return 0.0;
    memcpy(tmp, vals, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_asc);
    double out = (n % 2 == 0) ? (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0 : tmp[n / 2];
    free(tmp);
    return out;
}

static size_t g_cli_fixed_block_bytes = 0;

static long get_online_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
#endif
}

static int lzo_cpu_aligned_alloc(void **ptr, size_t align, size_t size) {
#ifdef _WIN32
    *ptr = _aligned_malloc(size, align);
    return (*ptr != NULL) ? 0 : -1;
#else
    return posix_memalign(ptr, align, size);
#endif
}

static void lzo_cpu_aligned_free(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int auto_detect_threads(void) {
    long n = get_online_cpu_count();
    return (n > 0) ? (int)n : 4;
}

static size_t choose_block_size(size_t total_bytes, int threads) {
    if (g_cli_fixed_block_bytes > 0) {
        size_t blk = g_cli_fixed_block_bytes;
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
        return blk;
    }

    /* Dynamic block size selection */
    if (total_bytes < 1024 * 1024) {
        return 64 * 1024; /* 64KB for small files */
    }

    if (threads < 1) threads = 1;
    size_t blk = (threads > 0) ? (total_bytes + (size_t)threads - 1u) / (size_t)threads : total_bytes;
    if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
    if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    if (blk > total_bytes) blk = total_bytes;
    if (blk == 0 && total_bytes == 0) blk = MIN_BLOCK_SIZE;
    return blk;
}

static unsigned char *read_entire(const char *path, size_t *size_out) {
    if (!path || !size_out) return NULL;

    FILE *fp;
    int from_stdin = 0;
    if (strcmp(path, "-") == 0) {
        fp = stdin;
        from_stdin = 1;
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
    } else {
        fp = fopen(path, "rb");
    }

    if (!fp) {
        perror(path);
        return NULL;
    }

    unsigned char *buf = NULL;
    size_t size = 0;

    if (from_stdin) {
        size_t cap = 1u << 18; /* 256 KiB */
        buf = (unsigned char *)malloc(cap);
        if (!buf) {
            fprintf(stderr, "malloc failed\n");
            return NULL;
        }
        while (1) {
            if (size == cap) {
                size_t new_cap = cap * 2u;
                unsigned char *tmp = (unsigned char *)realloc(buf, new_cap);
                if (!tmp) {
                    free(buf);
                    fprintf(stderr, "realloc failed\n");
                    return NULL;
                }
                buf = tmp;
                cap = new_cap;
            }
            size_t chunk = fread(buf + size, 1u, cap - size, fp);
            size += chunk;
            if (chunk == 0u) break;
        }
        if (ferror(fp)) {
            fprintf(stderr, "stdin read error\n");
            free(buf);
            return NULL;
        }
    } else {
        if (fseek(fp, 0, SEEK_END) != 0) {
            perror(path);
            fclose(fp);
            return NULL;
        }
        long sz = ftell(fp);
        if (sz < 0) {
            perror(path);
            fclose(fp);
            return NULL;
        }
        if (fseek(fp, 0, SEEK_SET) != 0) {
            perror(path);
            fclose(fp);
            return NULL;
        }
        buf = (unsigned char *)malloc((size_t)sz);
        if (!buf) {
            fprintf(stderr, "malloc failed\n");
            fclose(fp);
            return NULL;
        }
        size = fread(buf, 1u, (size_t)sz, fp);
        if (size != (size_t)sz) {
            fprintf(stderr, "short read from %s\n", path);
            free(buf);
            fclose(fp);
            return NULL;
        }
        fclose(fp);
    }

    *size_out = size;
    return buf;
}

static int write_entire(const char *path, const unsigned char *buf, size_t len) {
    if (!buf && len > 0) return -1;
    int to_stdout = (path && strcmp(path, "-") == 0);
    FILE *fp;
    if (to_stdout) {
        fp = stdout;
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    } else {
        fp = fopen(path, "wb");
    }
    if (!fp) {
        perror(path ? path : "(null)");
        return -1;
    }

    if (len > 0) {
        size_t written = fwrite(buf, 1u, len, fp);
        if (written != len) {
            fprintf(stderr, "short write to %s\n", to_stdout ? "stdout" : path);
            if (!to_stdout) fclose(fp);
            return -1;
        }
    }

    if (!to_stdout) fclose(fp);
    return 0;
}

static int compress_block_level(const unsigned char *in, size_t in_size,
                                unsigned char **out, size_t *out_size,
                                alg_t compression_alg, void *wrkmem_in) {
    /* This function is deprecated and replaced by compress_block_into */
    return LZO_E_ERROR;
}

/* forward decl for the prealloc variant used by workers */
static int compress_block_into(const unsigned char *in, size_t in_size,
                               unsigned char *out, size_t out_cap, size_t *out_size,
                               alg_t compression_alg, variant_t variant, void *wrkmem_in);

static int decompress_block(const unsigned char *in, size_t in_size,
                            unsigned char *out, size_t orig_size, alg_t alg) {
    lzo_uint dst_len = (lzo_uint)orig_size;
    int rc;
    if (alg == ALG_1Y) {
        rc = lzo1y_decompress_safe(in, (lzo_uint)in_size, out, &dst_len, NULL);
    } else {
        rc = lzo1x_decompress_safe(in, (lzo_uint)in_size, out, &dst_len, NULL);
    }
    if (rc != LZO_E_OK || dst_len != (lzo_uint)orig_size) {
        fprintf(stderr, "decompress_block failed: rc=%d dst_len=%lu orig_size=%lu alg=%d\n", rc, (unsigned long)dst_len, (unsigned long)orig_size, alg);
    }
    return (rc == LZO_E_OK && dst_len == (lzo_uint)orig_size) ? LZO_E_OK : rc;
}

static void free_compression_chunks(chunk_t *chunks, size_t chunk_count) {
    if (!chunks) return;
    for (size_t i = 0; i < chunk_count; ++i) {
        free(chunks[i].comp);
    }
    free(chunks);
}

static void *compress_worker(void *opaque) {
    compress_job_t *job = (compress_job_t *)opaque;
    lzo_align_t *thread_wrkmem = NULL;
    int have_wrkmem = 0;
    size_t wrkmem_sz = workmem_size_for(job->compression_alg, job->variant);
    if (lzo_cpu_aligned_alloc((void **)&thread_wrkmem, sizeof(lzo_align_t), wrkmem_sz) == 0) {
        have_wrkmem = 1;
    } else {
        thread_wrkmem = NULL;
        have_wrkmem = 0;
    }
    while (1) {
        /* atomic scheduling: fetch next index without lock */
        size_t idx = atomic_fetch_add(&job->next_index, (size_t)1);
        if (idx >= job->chunk_count) break;
        if (atomic_load(&job->status) != LZO_E_OK) break;

        chunk_t *ck = &job->chunks[idx];
        size_t out_len = 0;
        int rc;
        if (ck->comp) {
            /* compress into preallocated buffer */
            size_t cap = ck->in_size + ck->in_size / 16u + 64u + 3u;
            struct timespec _t0, _t1;
#ifdef CLOCK_MONOTONIC_RAW
            clock_gettime(CLOCK_MONOTONIC_RAW, &_t0);
#else
            clock_gettime(CLOCK_MONOTONIC, &_t0);
#endif
            rc = compress_block_into(ck->in, ck->in_size, ck->comp, cap, &out_len, job->compression_alg, job->variant, thread_wrkmem);
#ifdef CLOCK_MONOTONIC_RAW
            clock_gettime(CLOCK_MONOTONIC_RAW, &_t1);
#else
            clock_gettime(CLOCK_MONOTONIC, &_t1);
#endif
            ck->proc_ns = (uint64_t)(_t1.tv_sec - _t0.tv_sec) * 1000000000ULL + (uint64_t)(_t1.tv_nsec - _t0.tv_nsec);
            if (rc != LZO_E_OK) {
                atomic_store(&job->status, rc);
                break;
            }
            ck->comp_size = out_len;
        } else {
            /* compress_block_level is deprecated/removed in favor of compress_block_into with malloc */
            /* But for now let's just use compress_block_into with malloc */
            size_t cap = ck->in_size + ck->in_size / 16u + 64u + 3u;
            unsigned char *out = (unsigned char *)malloc(cap);
            if (!out) {
                atomic_store(&job->status, LZO_E_OUT_OF_MEMORY);
                break;
            }
            struct timespec _t0, _t1;
#ifdef CLOCK_MONOTONIC_RAW
            clock_gettime(CLOCK_MONOTONIC_RAW, &_t0);
#else
            clock_gettime(CLOCK_MONOTONIC, &_t0);
#endif
            rc = compress_block_into(ck->in, ck->in_size, out, cap, &out_len, job->compression_alg, job->variant, thread_wrkmem);
#ifdef CLOCK_MONOTONIC_RAW
            clock_gettime(CLOCK_MONOTONIC_RAW, &_t1);
#else
            clock_gettime(CLOCK_MONOTONIC, &_t1);
#endif
            ck->proc_ns = (uint64_t)(_t1.tv_sec - _t0.tv_sec) * 1000000000ULL + (uint64_t)(_t1.tv_nsec - _t0.tv_nsec);
            if (rc != LZO_E_OK) {
                free(out);
                atomic_store(&job->status, rc);
                break;
            }
            ck->comp = out;
            ck->comp_size = out_len;
        }
    }
    if (have_wrkmem && thread_wrkmem) lzo_cpu_aligned_free(thread_wrkmem);
    return NULL;
}

static int compress_multi(const unsigned char *input, size_t input_size,
                          size_t block_size, int threads, int level,
                          chunk_t **chunks_out, size_t *chunk_count_out,
                          double *elapsed_ms, size_t *total_comp_out) {
    if (threads < 1) threads = 1;
    size_t chunk_count = (block_size == 0 || input_size == 0)
        ? (input_size == 0 ? 0 : 1)
        : (input_size + block_size - 1u) / block_size;

    chunk_t *chunks = NULL;
    if (chunk_count > 0) {
        chunks = (chunk_t *)calloc(chunk_count, sizeof(chunk_t));
        if (!chunks) return LZO_E_OUT_OF_MEMORY;
        for (size_t i = 0; i < chunk_count; ++i) {
            size_t off = i * block_size;
            size_t left = input_size - off;
            chunks[i].in = input + off;
            chunks[i].in_size = (left < block_size || block_size == 0) ? left : block_size;
            chunks[i].offset = off;
        }
    }

    /* Preallocate per-chunk output buffers to avoid malloc/free in workers. */
    if (chunk_count > 0) {
        for (size_t i = 0; i < chunk_count; ++i) {
            size_t in_sz = chunks[i].in_size;
            size_t cap = in_sz + in_sz / 16u + 64u + 3u;
            chunks[i].comp = (unsigned char *)malloc(cap);
            if (!chunks[i].comp) {
                free_compression_chunks(chunks, chunk_count);
                return LZO_E_OUT_OF_MEMORY;
            }
            chunks[i].comp_size = 0;
        }
    }

    compress_job_t job;
    job.chunks = chunks;
    job.chunk_count = chunk_count;
    atomic_store(&job.next_index, (size_t)0);
    /* prefer explicit algorithm selection; fall back to numeric mapping */
    job.compression_alg = g_alg;
    job.variant = g_variant;
    atomic_store(&job.status, LZO_E_OK);
    pthread_mutex_init(&job.lock, NULL);

    /* use monotonic clock to avoid wall-clock adjustments */
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clk = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clk = CLOCK_MONOTONIC;
#endif
    struct timespec ts_start, ts_end;
    clock_gettime(clk, &ts_start);

    pthread_t *workers = NULL;
    if (chunk_count > 0) {
        workers = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
        if (!workers) {
            pthread_mutex_destroy(&job.lock);
            free_compression_chunks(chunks, chunk_count);
            return LZO_E_OUT_OF_MEMORY;
        }
        for (int i = 0; i < threads; ++i)
            pthread_create(&workers[i], NULL, compress_worker, &job);
        for (int i = 0; i < threads; ++i)
            pthread_join(workers[i], NULL);
    }

    clock_gettime(clk, &ts_end);
    if (elapsed_ms) *elapsed_ms = diff_ms_ts(&ts_start, &ts_end);

    int status = atomic_load(&job.status);
    pthread_mutex_destroy(&job.lock);
    free(workers);

    if (status != LZO_E_OK) {
        free_compression_chunks(chunks, chunk_count);
        return status;
    }

    size_t total_comp = 0;
    for (size_t i = 0; i < chunk_count; ++i)
        total_comp += chunks[i].comp_size;

    if (total_comp_out) *total_comp_out = total_comp;
    if (chunks_out) *chunks_out = chunks;
    else free_compression_chunks(chunks, chunk_count);
    if (chunk_count_out) *chunk_count_out = chunk_count;

    if (cpu_debug_metrics_enabled) {
        fprintf(stderr, "LZO_CPU_DEBUG per-chunk metrics (nblk=%zu):\n", chunk_count);
        for (size_t i = 0; i < chunk_count; ++i) {
            double ms = (double)chunks[i].proc_ns / 1e6;
            fprintf(stderr, "LZO_CPU_DEBUG BLOCK %4zu IN %6zu OUT %6zu CPU_MS %.3f\n", i, chunks[i].in_size, chunks[i].comp_size, ms);
        }
    }

    return LZO_E_OK;
}

static void *decompress_worker(void *opaque) {
    decompress_job_t *job = (decompress_job_t *)opaque;
    while (1) {
        size_t idx = atomic_fetch_add(&job->next_index, (size_t)1);
        if (idx >= job->chunk_count) break;
        if (atomic_load(&job->status) != LZO_E_OK) break;

        chunk_t *ck = &job->chunks[idx];
        int rc = decompress_block(ck->comp, ck->comp_size, ck->out, ck->in_size, job->alg);
        if (rc != LZO_E_OK) {
            atomic_store(&job->status, rc);
            break;
        }
    }
    return NULL;
}

static int decompress_multi(chunk_t *chunks, size_t chunk_count,
                            int threads, alg_t alg, double *elapsed_ms) {
    if (threads < 1) threads = 1;
    if (chunk_count == 0) {
        if (elapsed_ms) *elapsed_ms = 0.0;
        return LZO_E_OK;
    }

    decompress_job_t job;
    job.chunks = chunks;
    job.chunk_count = chunk_count;
    job.alg = alg;
    atomic_store(&job.next_index, (size_t)0);
    atomic_store(&job.status, LZO_E_OK);
    pthread_mutex_init(&job.lock, NULL);

    pthread_t *workers = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!workers) {
        pthread_mutex_destroy(&job.lock);
        return LZO_E_OUT_OF_MEMORY;
    }

    struct timespec ts_start, ts_end;
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clk = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clk = CLOCK_MONOTONIC;
#endif
    clock_gettime(clk, &ts_start);
    for (int i = 0; i < threads; ++i)
        pthread_create(&workers[i], NULL, decompress_worker, &job);
    for (int i = 0; i < threads; ++i)
        pthread_join(workers[i], NULL);
    clock_gettime(clk, &ts_end);

    if (elapsed_ms) *elapsed_ms = diff_ms_ts(&ts_start, &ts_end);
    int status = atomic_load(&job.status);
    pthread_mutex_destroy(&job.lock);
    free(workers);
    return status;
}

static void run_benchmark(const unsigned char *data, size_t size,
                          int level, int threads) {
    if (!data) return;
    if (size == 0) {
        fprintf(stderr, "\n== Benchmark ==\nInput is empty; skipping benchmark.\n");
        return;
    }
    fprintf(stderr, "\n== Benchmark ==\n");

    struct timespec t0, t1;
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clk = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clk = CLOCK_MONOTONIC;
#endif

    alg_t use_alg = g_alg;

    size_t block_size = choose_block_size(size, threads);
    chunk_t *chunks = NULL;
    size_t chunk_count = 0;
    double multi_comp_ms = 0.0;
    size_t total_comp = 0;
    double comp_total_ms = 0.0;
    double comp_io_ms = 0.0;
    double multi_decomp_ms = 0.0;
    double decomp_total_ms = 0.0;
    int verify_ok = 0;
    unsigned char *container = NULL;
    unsigned char *container_read = NULL;
    chunk_t *dchunks = NULL;
    unsigned char *multi_out = NULL;
    FILE *comp_tmp = NULL;
    FILE *out_tmp = NULL;

    clock_gettime(clk, &t0);
    int rc = compress_multi(data, size, block_size, threads, level,
                            &chunks, &chunk_count, &multi_comp_ms, &total_comp);
    clock_gettime(clk, &t1);
    comp_total_ms = diff_ms_ts(&t0, &t1);

    if (rc != LZO_E_OK) {
        fprintf(stderr, "multi compress failed: %d\n", rc);
        return;
    }

    {
        size_t header_size = 2u + 4u + 4u + 4u + 4u + chunk_count * 4u;
        size_t container_size = header_size + total_comp;
        size_t cursor = 0;

        container = (unsigned char *)malloc(container_size ? container_size : 1u);
        if (!container) {
            fprintf(stderr, "malloc failed\n");
            free_compression_chunks(chunks, chunk_count);
            return;
        }

        write_u16(container + cursor, MAGIC_TAG); cursor += 2u;
        write_u32(container + cursor, (uint32_t)size); cursor += 4u;
        write_u32(container + cursor, (uint32_t)block_size); cursor += 4u;
        write_u32(container + cursor, (uint32_t)chunk_count); cursor += 4u;
        write_u32(container + cursor, (uint32_t)use_alg); cursor += 4u;
        for (size_t i = 0; i < chunk_count; ++i) {
            write_u32(container + cursor, (uint32_t)chunks[i].comp_size);
            cursor += 4u;
        }
        for (size_t i = 0; i < chunk_count; ++i) {
            memcpy(container + cursor, chunks[i].comp, chunks[i].comp_size);
            cursor += chunks[i].comp_size;
        }

        comp_tmp = tmpfile();
        if (!comp_tmp) {
            fprintf(stderr, "tmpfile failed\n");
            free(container);
            free_compression_chunks(chunks, chunk_count);
            return;
        }

        clock_gettime(clk, &t0);
        if (fwrite(container, 1u, container_size, comp_tmp) != container_size) {
            fprintf(stderr, "benchmark tmp write failed\n");
            fclose(comp_tmp);
            free(container);
            free_compression_chunks(chunks, chunk_count);
            return;
        }
        fflush(comp_tmp);
        rewind(comp_tmp);
        clock_gettime(clk, &t1);
        comp_io_ms = diff_ms_ts(&t0, &t1);
        comp_total_ms += comp_io_ms;

        container_read = (unsigned char *)malloc(container_size ? container_size : 1u);
        if (!container_read) {
            fprintf(stderr, "malloc failed\n");
            fclose(comp_tmp);
            free(container);
            free_compression_chunks(chunks, chunk_count);
            return;
        }

        clock_gettime(clk, &t0);
        if (fread(container_read, 1u, container_size, comp_tmp) != container_size) {
            fprintf(stderr, "benchmark tmp read failed\n");
            clock_gettime(clk, &t1);
            decomp_total_ms = diff_ms_ts(&t0, &t1);
        } else {
            uint16_t magic = 0;
            uint32_t orig_sz = 0, blk_sz = 0, nblk = 0, alg_val = 0;
            size_t rcur = 0;
            int parse_ok = 1;

            magic = read_u16(container_read + rcur); rcur += 2u;
            orig_sz = read_u32(container_read + rcur); rcur += 4u;
            blk_sz = read_u32(container_read + rcur); rcur += 4u;
            nblk = read_u32(container_read + rcur); rcur += 4u;
            alg_val = read_u32(container_read + rcur); rcur += 4u;

            if (magic != MAGIC_TAG || orig_sz != (uint32_t)size || alg_val != (uint32_t)use_alg) {
                parse_ok = 0;
            }

            if (parse_ok) {
                size_t lens_bytes = (size_t)nblk * 4u;
                if (rcur + lens_bytes > container_size) parse_ok = 0;

                if (parse_ok) {
                    const unsigned char *lens_ptr = container_read + rcur;
                    const unsigned char *payload = lens_ptr + lens_bytes;
                    const unsigned char *payload_end = container_read + container_size;
                    size_t out_off = 0;

                    dchunks = (chunk_t *)calloc((size_t)nblk ? (size_t)nblk : 1u, sizeof(chunk_t));
                    multi_out = (unsigned char *)malloc(size ? size : 1u);
                    if (!dchunks || !multi_out) {
                        parse_ok = 0;
                    } else {
                        const unsigned char *p = payload;
                        for (uint32_t i = 0; i < nblk; ++i) {
                            uint32_t clen = read_u32(lens_ptr + i * 4u);
                            size_t orig_chunk = (i == nblk - 1u) ? ((size_t)orig_sz - out_off) : (size_t)blk_sz;
                            if (p + clen > payload_end) {
                                parse_ok = 0;
                                break;
                            }
                            dchunks[i].comp = (unsigned char *)p;
                            dchunks[i].comp_size = (size_t)clen;
                            dchunks[i].in_size = orig_chunk;
                            dchunks[i].offset = out_off;
                            dchunks[i].out = multi_out + out_off;
                            out_off += orig_chunk;
                            p += clen;
                        }

                        if (parse_ok) {
                            rc = decompress_multi(dchunks, (size_t)nblk, threads, use_alg, &multi_decomp_ms);
                            verify_ok = (rc == LZO_E_OK && memcmp(multi_out, data, size) == 0);

                            out_tmp = tmpfile();
                            if (out_tmp) {
                                if (fwrite(multi_out, 1u, size, out_tmp) != size) {
                                    verify_ok = 0;
                                }
                                fflush(out_tmp);
                            } else {
                                verify_ok = 0;
                            }
                        }
                    }
                }
            }

            clock_gettime(clk, &t1);
            decomp_total_ms = diff_ms_ts(&t0, &t1);
            if (!parse_ok) {
                fprintf(stderr, "benchmark parse/decompress setup failed\n");
            }
        }
    }

    {
        char comp_k_s[32], comp_t_s[32], dec_k_s[32], dec_t_s[32];
        double comp_kernel_tp = (multi_comp_ms > 0.0) ? (size / 1048576.0) / (multi_comp_ms / 1000.0) : 0.0;
        double comp_total_tp  = (comp_total_ms > 0.0) ? (size / 1048576.0) / (comp_total_ms / 1000.0) : 0.0;
        double dec_kernel_tp  = (multi_decomp_ms > 0.0) ? (size / 1048576.0) / (multi_decomp_ms / 1000.0) : 0.0;
        double dec_total_tp   = (decomp_total_ms > 0.0) ? (size / 1048576.0) / (decomp_total_ms / 1000.0) : 0.0;

        format_ms_or_us(comp_k_s, sizeof(comp_k_s), multi_comp_ms);
        format_ms_or_us(comp_t_s, sizeof(comp_t_s), comp_total_ms);
        format_ms_or_us(dec_k_s, sizeof(dec_k_s), multi_decomp_ms);
        format_ms_or_us(dec_t_s, sizeof(dec_t_s), decomp_total_ms);

        fprintf(stderr,
                "Compress   : kernel=%s total=%s kernel_tp=%.2f MB/s total_tp=%.2f MB/s blocks=%zu\n",
                comp_k_s, comp_t_s, comp_kernel_tp, comp_total_tp, chunk_count);
        fprintf(stderr,
                "Decompress : kernel=%s total=%s kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n",
                dec_k_s, dec_t_s, dec_kernel_tp, dec_total_tp, verify_ok ? "OK" : "FAIL");
    }

    if (out_tmp) fclose(out_tmp);
    if (comp_tmp) fclose(comp_tmp);
    free(multi_out);
    free(dchunks);
    free(container_read);
    free(container);
    free_compression_chunks(chunks, chunk_count);
}

static int run_stable_kernel_bench(const unsigned char *data, size_t size,
                                   int level, int threads, double bench_seconds) {
    if (!data || size == 0) {
        fprintf(stderr, "Bench error: empty input\n");
        return 1;
    }
    if (bench_seconds <= 0.0) bench_seconds = 3.0;

#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clk = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clk = CLOCK_MONOTONIC;
#endif

    size_t cap = 16;
    size_t n = 0;
    double *comp_tp = (double *)malloc(cap * sizeof(double));
    double *dec_tp = (double *)malloc(cap * sizeof(double));
    double *comp_total_tp = (double *)malloc(cap * sizeof(double));
    double *dec_total_tp = (double *)malloc(cap * sizeof(double));
    double *ratio_pct = (double *)malloc(cap * sizeof(double));
    if (!comp_tp || !dec_tp || !comp_total_tp || !dec_total_tp || !ratio_pct) {
        free(comp_tp);
        free(dec_tp);
        free(comp_total_tp);
        free(dec_total_tp);
        free(ratio_pct);
        fprintf(stderr, "Bench error: malloc failed\n");
        return 1;
    }

    int verify_ok = 1;
    struct timespec ts_start, ts_now;
    clock_gettime(clk, &ts_start);

    while (1) {
        size_t block_size = choose_block_size(size, threads);
        chunk_t *chunks = NULL;
        size_t chunk_count = 0;
        double comp_ms = 0.0;
        size_t total_comp = 0;

        struct timespec ts_iter_comp_start, ts_iter_comp_end;
        clock_gettime(clk, &ts_iter_comp_start);
        int rc = compress_multi(data, size, block_size, threads, level,
                                &chunks, &chunk_count, &comp_ms, &total_comp);
        clock_gettime(clk, &ts_iter_comp_end);
        if (rc != LZO_E_OK) {
            fprintf(stderr, "Bench error: compress failed (%d)\n", rc);
            free_compression_chunks(chunks, chunk_count);
            verify_ok = 0;
            break;
        }
        double comp_wall_ms = diff_ms_ts(&ts_iter_comp_start, &ts_iter_comp_end);

        unsigned char *out = (unsigned char *)malloc(size ? size : 1u);
        if (!out) {
            fprintf(stderr, "Bench error: malloc failed\n");
            free_compression_chunks(chunks, chunk_count);
            verify_ok = 0;
            break;
        }

        for (size_t i = 0; i < chunk_count; ++i) {
            chunks[i].out = out + chunks[i].offset;
        }

        double decomp_ms = 0.0;
        struct timespec ts_iter_dec_start, ts_iter_dec_end;
        clock_gettime(clk, &ts_iter_dec_start);
        rc = decompress_multi(chunks, chunk_count, threads, g_alg, &decomp_ms);
        clock_gettime(clk, &ts_iter_dec_end);
        if (rc != LZO_E_OK || memcmp(out, data, size) != 0) {
            verify_ok = 0;
        }
        double dec_wall_ms = diff_ms_ts(&ts_iter_dec_start, &ts_iter_dec_end);

        if (n == cap) {
            size_t new_cap = cap * 2;
            double *new_comp = (double *)realloc(comp_tp, new_cap * sizeof(double));
            if (!new_comp) {
                free(out);
                free_compression_chunks(chunks, chunk_count);
                verify_ok = 0;
                break;
            }
            comp_tp = new_comp;

            double *new_dec = (double *)realloc(dec_tp, new_cap * sizeof(double));
            if (!new_dec) {
                free(out);
                free_compression_chunks(chunks, chunk_count);
                verify_ok = 0;
                break;
            }
            dec_tp = new_dec;

            double *new_comp_total = (double *)realloc(comp_total_tp, new_cap * sizeof(double));
            if (!new_comp_total) {
                free(out);
                free_compression_chunks(chunks, chunk_count);
                verify_ok = 0;
                break;
            }
            comp_total_tp = new_comp_total;

            double *new_dec_total = (double *)realloc(dec_total_tp, new_cap * sizeof(double));
            if (!new_dec_total) {
                free(out);
                free_compression_chunks(chunks, chunk_count);
                verify_ok = 0;
                break;
            }
            dec_total_tp = new_dec_total;

            double *new_ratio = (double *)realloc(ratio_pct, new_cap * sizeof(double));
            if (!new_ratio) {
                free(out);
                free_compression_chunks(chunks, chunk_count);
                verify_ok = 0;
                break;
            }
            ratio_pct = new_ratio;
            cap = new_cap;
        }

        comp_tp[n] = (comp_ms > 0.0) ? (size / 1048576.0) / (comp_ms / 1000.0) : 0.0;
        dec_tp[n] = (decomp_ms > 0.0) ? (size / 1048576.0) / (decomp_ms / 1000.0) : 0.0;
        comp_total_tp[n] = (comp_wall_ms > 0.0) ? (size / 1048576.0) / (comp_wall_ms / 1000.0) : 0.0;
        dec_total_tp[n] = (dec_wall_ms > 0.0) ? (size / 1048576.0) / (dec_wall_ms / 1000.0) : 0.0;
        ratio_pct[n] = size ? (100.0 * (double)total_comp / (double)size) : 0.0;
        n++;

        free(out);
        free_compression_chunks(chunks, chunk_count);

        if (!verify_ok) break;

        clock_gettime(clk, &ts_now);
        double elapsed = diff_ms_ts(&ts_start, &ts_now) / 1000.0;
        if (elapsed >= bench_seconds && n > 0) break;
    }

    clock_gettime(clk, &ts_now);
    double elapsed = diff_ms_ts(&ts_start, &ts_now) / 1000.0;

    if (n == 0) {
        free(comp_tp);
        free(dec_tp);
        free(comp_total_tp);
        free(dec_total_tp);
        free(ratio_pct);
        fprintf(stderr, "Bench error: no valid iterations\n");
        return 1;
    }

    double comp_med = median_double(comp_tp, n);
    double dec_med = median_double(dec_tp, n);
    double comp_total_med = median_double(comp_total_tp, n);
    double dec_total_med = median_double(dec_total_tp, n);
    double ratio_med = median_double(ratio_pct, n);

    fprintf(stderr, "Bench Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n", comp_med, comp_total_med, ratio_med);
    fprintf(stderr, "Bench Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n", dec_med, dec_total_med, verify_ok ? "OK" : "FAIL");
    fprintf(stderr, "Bench Summary : iterations=%zu seconds=%.2f\n", n, elapsed);

    free(comp_tp);
    free(dec_tp);
    free(comp_total_tp);
    free(dec_total_tp);
    free(ratio_pct);
    return verify_ok ? 0 : 1;
}

static int run_stable_kernel_bench_file(const char *input_path,
                                        int level, int threads, double bench_seconds) {
    size_t input_size = 0;
    unsigned char *input = read_entire(input_path, &input_size);
    if (!input && input_size != 0) return 1;

    int rc = run_stable_kernel_bench(input, input_size, level, threads, bench_seconds);
    free(input);
    return rc;
}

static int compress_file(const char *input_path, const char *output_path,
                         int level, int threads, int do_bench, int verify_only) {
    struct timespec t_total_start, t_total_end;
    clock_gettime(CLOCK_MONOTONIC, &t_total_start);
    struct timespec t_read_start, t_read_end;
    clock_gettime(CLOCK_MONOTONIC, &t_read_start);

    size_t input_size = 0;
    unsigned char *input = read_entire(input_path, &input_size);
    if (!input && input_size != 0) return 1;

    clock_gettime(CLOCK_MONOTONIC, &t_read_end);
    double read_ms = diff_ms_ts(&t_read_start, &t_read_end);

    if (input_size > UINT32_MAX) {
        fprintf(stderr, "input larger than 4 GiB is not supported\n");
        free(input);
        return 1;
    }

    size_t block_size = choose_block_size(input_size, threads);
    chunk_t *chunks = NULL;
    size_t chunk_count = 0;
    double comp_ms = 0.0;
    size_t total_comp = 0;
    int rc = compress_multi(input, input_size, block_size, threads, level,
                            &chunks, &chunk_count, &comp_ms, &total_comp);
    if (rc != LZO_E_OK) {
        fprintf(stderr, "compress failed: %d\n", rc);
        free(input);
        return 1;
    }

    alg_t used_alg = g_alg;

    struct timespec t_prepare_start, t_prepare_end;
    struct timespec t_write_start, t_write_end;
    double write_ms = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &t_prepare_start);

    size_t header_size = 2u + 4u + 4u + 4u + 4u + chunk_count * 4u;
    size_t total_size = header_size + total_comp;
    unsigned char *out_buf = (unsigned char *)malloc(total_size ? total_size : 1u);
    if (!out_buf) {
        fprintf(stderr, "malloc failed\n");
        free(input);
        free_compression_chunks(chunks, chunk_count);
        return 1;
    }

    size_t cursor = 0;
    write_u16(out_buf + cursor, MAGIC_TAG); cursor += 2u;
    write_u32(out_buf + cursor, (uint32_t)input_size); cursor += 4u;
    write_u32(out_buf + cursor, (uint32_t)block_size); cursor += 4u;
    write_u32(out_buf + cursor, (uint32_t)chunk_count); cursor += 4u;
    write_u32(out_buf + cursor, (uint32_t)used_alg); cursor += 4u;
    for (size_t i = 0; i < chunk_count; ++i) {
        write_u32(out_buf + cursor, (uint32_t)chunks[i].comp_size);
        cursor += 4u;
    }
    for (size_t i = 0; i < chunk_count; ++i) {
        memcpy(out_buf + cursor, chunks[i].comp, chunks[i].comp_size);
        cursor += chunks[i].comp_size;
    }

    if (verify_only) {
        /* Perform in-memory decompression from chunks and verify equality */
        unsigned char *multi_out = (unsigned char *)malloc(input_size ? input_size : 1u);
        if (!multi_out) {
            fprintf(stderr, "malloc failed\n");
            free(out_buf);
            free(input);
            free_compression_chunks(chunks, chunk_count);
            return 1;
        }
        for (size_t i = 0; i < chunk_count; ++i)
            chunks[i].out = multi_out + chunks[i].offset;
        double multi_decomp_ms = 0.0;
        int rc = decompress_multi(chunks, chunk_count, threads, used_alg, &multi_decomp_ms);
        if (rc != LZO_E_OK) {
            fprintf(stderr, "verify decompress failed: %d\n", rc);
            free(multi_out);
            free(out_buf);
            free(input);
            free_compression_chunks(chunks, chunk_count);
            return 1;
        }
        if (memcmp(multi_out, input, input_size) != 0) {
            fprintf(stderr, "verify failed: decompressed data differs\n");
            free(multi_out);
            free(out_buf);
            free(input);
            free_compression_chunks(chunks, chunk_count);
            return 1;
        }
        {
            char comp_s[32], decomp_s[32];
            format_ms_or_us(comp_s, sizeof(comp_s), comp_ms);
            format_ms_or_us(decomp_s, sizeof(decomp_s), multi_decomp_ms);
            fprintf(stderr, "Verify OK: in=%zu out=%zu ratio=%.2f%% comp_time=%s decomp_time=%s\n",
                input_size, total_comp, input_size ? (100.0 * total_comp / input_size) : 0.0,
                comp_s, decomp_s);
        }
        free(multi_out);
        /* skip writing output file when verifying */
    } else {
        clock_gettime(CLOCK_MONOTONIC, &t_write_start);

        if (write_entire(output_path, out_buf, total_size) != 0) {
            fprintf(stderr, "failed to write output\n");
            free(out_buf);
            free(input);
            free_compression_chunks(chunks, chunk_count);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &t_write_end);
        write_ms = diff_ms_ts(&t_write_start, &t_write_end);

        clock_gettime(CLOCK_MONOTONIC, &t_total_end);
    }

    if (!do_bench) {
        alg_t used_alg = g_alg;

        // Calculate prepare time (time between compression and write)
        clock_gettime(CLOCK_MONOTONIC, &t_prepare_end);
        double prepare_ms = diff_ms_ts(&t_prepare_start, &t_prepare_end);

        // Calculate total time if not in verify mode
        double total_ms = 0.0;
        double write_ms = 0.0;
        if (!verify_only) {
            total_ms = diff_ms_ts(&t_total_start, &t_total_end);
            write_ms = diff_ms_ts(&t_write_start, &t_write_end);
        }

        if (verify_only) {
            char comp_s[32];
            format_ms_or_us(comp_s, sizeof(comp_s), comp_ms);
            fprintf(stderr,
                "Compressed %zu bytes -> %zu bytes (%.2f%%) blocks=%zu block_sz=%zu threads=%d alg=%s time=%s (%.2f MB/s)\n",
                input_size,
                total_comp,
                input_size ? (100.0 * total_comp / input_size) : 0.0,
                chunk_count,
                block_size,
                threads,
                alg_to_str(used_alg),
                comp_s,
                comp_ms > 0.0 ? (input_size / 1048576.0) / (comp_ms / 1000.0) : 0.0);
        } else {
            // Calculate throughput: overall and kernel
            double throughput = total_ms > 0.0 ? (input_size / 1048576.0) / (total_ms / 1000.0) : 0.0;
            double kernel_throughput = comp_ms > 0.0 ? (input_size / 1048576.0) / (comp_ms / 1000.0) : 0.0;
            double ratio_pct = input_size ? (100.0 * total_comp / input_size) : 0.0;

            fprintf(stderr,
                    "Compressed %zu bytes -> %zu bytes (%.2f%%) blocks=%zu block_sz=%zu threads=%d alg=%s\n",
                    input_size,
                    total_comp,
                    ratio_pct,
                    chunk_count,
                    block_size,
                    threads,
                    alg_to_str(used_alg));

            // Output timing in GPU-compatible format
            fprintf(stderr, "=== Timing Breakdown ===\n");
            fprintf(stderr, "1. File Read       : %.2f ms (%.1f%%)\n", read_ms, total_ms > 0 ? 100.0*read_ms/total_ms : 0);
            fprintf(stderr, "2. Compress        : %.2f ms (%.1f%%)\n", comp_ms, total_ms > 0 ? 100.0*comp_ms/total_ms : 0);
            fprintf(stderr, "3. Prepare Output  : %.2f ms (%.1f%%)\n", prepare_ms, total_ms > 0 ? 100.0*prepare_ms/total_ms : 0);
            fprintf(stderr, "4. File Write      : %.2f ms (%.1f%%)\n", write_ms, total_ms > 0 ? 100.0*write_ms/total_ms : 0);
            fprintf(stderr, "------------------------\n");
            fprintf(stderr, "TOTAL              : %.2f ms\n", total_ms);
            fprintf(stderr, "Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);
            fprintf(stderr, "Compression ratio  : %.2f%% (%.2f : 1)\n", ratio_pct, ratio_pct > 0 ? 100.0/ratio_pct : 0);
        }
    }

    if (do_bench) run_benchmark(input, input_size, level, threads);

    free(out_buf);
    free(input);
    free_compression_chunks(chunks, chunk_count);
    return 0;
}

static int decompress_file(const char *input_path, const char *output_path,
                           int threads, int verify_only) {
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clk = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clk = CLOCK_MONOTONIC;
#endif
    struct timespec t_total_start, t_read_end, t_write_start, t_write_end, t_total_end;
    clock_gettime(clk, &t_total_start);
    size_t comp_size = 0;
    unsigned char *comp = read_entire(input_path, &comp_size);
    clock_gettime(clk, &t_read_end);
    if (!comp && comp_size != 0) return 1;

    if (comp_size < 14u) {
        fprintf(stderr, "input too small\n");
        free(comp);
        return 1;
    }

    size_t cursor = 0;
    uint16_t magic = read_u16(comp + cursor); cursor += 2u;
    if (magic != MAGIC_TAG) {
        fprintf(stderr, "bad magic 0x%04x\n", magic);
        free(comp);
        return 1;
    }
    uint32_t orig_sz = read_u32(comp + cursor); cursor += 4u;
    uint32_t blk_sz = read_u32(comp + cursor); cursor += 4u;
    uint32_t nblk = read_u32(comp + cursor); cursor += 4u;
    /* Read algorithm type */
    uint32_t alg_val = read_u32(comp + cursor); cursor += 4u;
    g_alg = (alg_t)alg_val;

    size_t lengths_bytes = (size_t)nblk * 4u;
    if (cursor + lengths_bytes > comp_size) {
        fprintf(stderr, "truncated length table\n");
        free(comp);
        return 1;
    }

    const unsigned char *lengths_ptr = comp + cursor;
    cursor += lengths_bytes;
    const unsigned char *payload = comp + cursor;
    size_t payload_size = comp_size - cursor;

    size_t total_comp = 0;
    for (uint32_t i = 0; i < nblk; ++i)
        total_comp += read_u32(lengths_ptr + i * 4u);
    if (total_comp > payload_size) {
        fprintf(stderr, "truncated payload\n");
        free(comp);
        return 1;
    }

    size_t output_size = orig_sz;
    unsigned char *output = (unsigned char *)malloc(output_size ? output_size : 1u);
    if (!output && output_size != 0) {
        fprintf(stderr, "malloc failed\n");
        free(comp);
        return 1;
    }

    chunk_t *chunks = NULL;
    if (nblk > 0) {
        chunks = (chunk_t *)calloc(nblk, sizeof(chunk_t));
        if (!chunks) {
            fprintf(stderr, "calloc failed\n");
            free(output);
            free(comp);
            return 1;
        }

        const unsigned char *blk_ptr = payload;
        size_t offset = 0;
        for (uint32_t i = 0; i < nblk; ++i) {
            uint32_t clen = read_u32(lengths_ptr + i * 4u);
            size_t orig_chunk = (i == nblk - 1u) ? (size_t)orig_sz - offset : (size_t)blk_sz;
            if (blk_ptr + clen > payload + payload_size) {
                fprintf(stderr, "chunk overflow\n");
                free(output);
                free(comp);
                free(chunks);
                return 1;
            }
            chunks[i].comp = (unsigned char *)blk_ptr;
            chunks[i].comp_size = clen;
            chunks[i].in_size = orig_chunk;
            chunks[i].offset = offset;
            chunks[i].out = output + offset;
            blk_ptr += clen;
            offset += orig_chunk;
        }
    }

    double decomp_ms = 0.0;
    int rc = decompress_multi(chunks, nblk, threads, g_alg, &decomp_ms);
    if (rc != LZO_E_OK) {
        fprintf(stderr, "decompress failed: %d\n", rc);
        free(output);
        free(comp);
        free(chunks);
        return 1;
    }

    double read_ms = diff_ms_ts(&t_total_start, &t_read_end);
    double write_ms = 0.0;

    if (verify_only) {
        clock_gettime(clk, &t_total_end);
        double total_ms = diff_ms_ts(&t_total_start, &t_total_end);
        double throughput = total_ms > 0.0 ? (orig_sz / 1048576.0) / (total_ms / 1000.0) : 0.0;
        double kernel_throughput = decomp_ms > 0.0 ? (orig_sz / 1048576.0) / (decomp_ms / 1000.0) : 0.0;

        fprintf(stderr, "Verify decompress OK\n");
        fprintf(stderr, "\n=== Decompression Statistics ===\n");
        fprintf(stderr, "Compressed size    : %zu bytes (%.2f MB)\n", total_comp, total_comp / 1048576.0);
        fprintf(stderr, "Output size        : %u bytes (%.2f MB)\n", orig_sz, orig_sz / 1048576.0);
        fprintf(stderr, "Block size (blocks): %u bytes/%u KB (%u)\n", blk_sz, blk_sz / 1024, nblk);
        fprintf(stderr, "Threads            : %d\n", threads);
        fprintf(stderr, "Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);
        fprintf(stderr, "==============================\n\n");

        fprintf(stderr, "=== Time Breakdown (Decompression) ===\n");
        fprintf(stderr, "1. File Read           : %8.3f ms\n", read_ms);
        fprintf(stderr, "2. Decompress          : %8.3f ms\n", decomp_ms);
        fprintf(stderr, "TOTAL                  : %8.3f ms\n", total_ms);
        fprintf(stderr, "\n=== Percentage Breakdown ===\n");
        double denom = total_ms > 0.0 ? total_ms : 1.0;
        fprintf(stderr, "Decompress      : %6.2f%%\n", 100.0 * decomp_ms / denom);
        fprintf(stderr, "File Read       : %6.2f%%\n", 100.0 * read_ms / denom);
        fprintf(stderr, "\n");
    } else {
        clock_gettime(clk, &t_write_start);
        if (write_entire(output_path, output, output_size) != 0) {
            fprintf(stderr, "failed to write output\n");
            free(output);
            free(comp);
            free(chunks);
            return 1;
        }
        clock_gettime(clk, &t_write_end);
        clock_gettime(clk, &t_total_end);

        write_ms = diff_ms_ts(&t_write_start, &t_write_end);
        double total_ms = diff_ms_ts(&t_total_start, &t_total_end);
        double throughput = total_ms > 0.0 ? (orig_sz / 1048576.0) / (total_ms / 1000.0) : 0.0;
        double kernel_throughput = decomp_ms > 0.0 ? (orig_sz / 1048576.0) / (decomp_ms / 1000.0) : 0.0;

        fprintf(stderr, "wrote %s\n", output_path);
        fprintf(stderr, "\n=== Decompression Statistics ===\n");
        fprintf(stderr, "Compressed size    : %zu bytes (%.2f MB)\n", total_comp, total_comp / 1048576.0);
        fprintf(stderr, "Output size        : %u bytes (%.2f MB)\n", orig_sz, orig_sz / 1048576.0);
        fprintf(stderr, "Block size (blocks): %u bytes/%u KB (%u)\n", blk_sz, blk_sz / 1024, nblk);
        fprintf(stderr, "Threads            : %d\n", threads);
        fprintf(stderr, "Throughput         : %.2f MB/s (kernel: %.2f MB/s)\n", throughput, kernel_throughput);
        fprintf(stderr, "==============================\n\n");

        fprintf(stderr, "=== Time Breakdown (Decompression) ===\n");
        fprintf(stderr, "1. File Read           : %8.3f ms\n", read_ms);
        fprintf(stderr, "2. Decompress          : %8.3f ms\n", decomp_ms);
        fprintf(stderr, "3. File Write          : %8.3f ms\n", write_ms);
        fprintf(stderr, "TOTAL                  : %8.3f ms\n", total_ms);
        fprintf(stderr, "\n=== Percentage Breakdown ===\n");
        double denom = total_ms > 0.0 ? total_ms : 1.0;
        fprintf(stderr, "Decompress      : %6.2f%%\n", 100.0 * decomp_ms / denom);
        fprintf(stderr, "File I/O        : %6.2f%% (read=%.2f%% + write=%.2f%%)\n",
            100.0 * (read_ms + write_ms) / denom,
            100.0 * read_ms / denom,
            100.0 * write_ms / denom);
        fprintf(stderr, "\n");
    }

    free(output);
    free(comp);
    free(chunks);
    return 0;
}

static size_t parse_size_bytes(const char *s) {
    if (!s || !*s) return 0;
    char *end;
    double val = strtod(s, &end);
    if (end == s) return 0;
    while (*end == ' ' || *end == '\t') end++;
    if (*end == '\0') return (size_t)val;
    if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0) return (size_t)(val * 1024);
    if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0) return (size_t)(val * 1024 * 1024);
    return (size_t)val;
}

static int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') return -1;
    if (v < 0 || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static void print_usage(const char *prog) {
        fprintf(stderr,
            "Usage: %s [options] <input>\n"
            "Options:\n"
            "  -d              Decompress instead of compress\n"
            "  -t <threads>    Worker thread count (0=auto, default: auto)\n"
            "  --verify        Verify round-trip instead of writing outputs\n"
            "  -a <alg>        Select algorithm (1x, 1y). Default 1x.\n"
            "  -l <level>      Select variant: numeric (11/12/13/14/999) or string (1/1k/1l/1o/999). Default 14.\n"
            "                   11=1k, 12=1l, 13=1o, 14=1 (standard), 999=slow compression\n"
            "  -L              Alias for -l\n"
            "  -B, --block-size <N>  Block size with suffix B/KB/MB (default: 256KB)\n"
            "  -o <path>       Output path (- for stdout). If omitted a default is generated from input\n"
            "  --bench [N]     Run stable kernel benchmark for optional N seconds (default 3)\n"
            "  --benchmark     Run benchmark metrics after operation\n"
            "  --debug-metrics Enable per-block CPU timing/metrics (debug only)\n"
            "  -h, --help      Show this help\n"
            "  Use '-' for stdin/stdout. Output defaults to input with .lzo (compress)\n"
            "  or stripped .lzo extension (decompress).\n",
            prog);
}

int main(int argc, char **argv) {
    if (lzo_init() != LZO_E_OK) {
        fprintf(stderr, "lzo_init failed\n");
        return 1;
    }

    int level = 3;
    int mode_decompress = 0;
    int threads = DEFAULT_THREAD_COUNT;
    int do_bench = 0;
    int bench_mode = 0; /* concise bench output (compression ratio, throughput) */
    double bench_seconds = 3.0;
    int verbose = 0;
    int verify_only = 0;
    char *kernel_spec = NULL;

    const char *input = NULL;
    const char *output = NULL;
    char *auto_output = NULL;

    int arg_idx = 1;
    /* Support compact numeric flags like -1/-2/-3/-4 to select common strategy, or -d for decompress.
     * These are accepted as the first argument, mirroring the behavior of lzo_gpu and other tools.
     */
    if (argc >= 2) {
        if (strcmp(argv[1], "-d") == 0) {
            mode_decompress = 1; arg_idx = 2;
        } else if (strcmp(argv[1], "-1") == 0) {
            level = 1; arg_idx = 2;
        } else if (strcmp(argv[1], "-2") == 0) {
            level = 2; arg_idx = 2;
        } else if (strcmp(argv[1], "-3") == 0 || strcmp(argv[1], "-c") == 0) {
            level = 3; arg_idx = 2;
        } else if (strcmp(argv[1], "-4") == 0) {
            level = 4; arg_idx = 2;
        }
    }

    for (int i = arg_idx; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "-d") == 0) {
            if (kernel_spec) {
                fprintf(stderr, "-l cannot be used with -d (decompress mode)\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            mode_decompress = 1;
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(arg, "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') bench_seconds = atof(argv[++i]);
        } else if (strcmp(arg, "-B") == 0 || strcmp(arg, "--block-size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "-B requires an argument\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            g_cli_fixed_block_bytes = parse_size_bytes(argv[++i]);
            if (g_cli_fixed_block_bytes == 0) {
                 fprintf(stderr, "invalid block size\n");
                 return 1;
            }
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || parse_int(argv[i + 1], &threads) != 0) {
                fprintf(stderr, "invalid thread count\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            ++i;
        } else if (strcmp(arg, "--benchmark") == 0) {
            do_bench = 1;
        } else if (strcmp(arg, "--verify") == 0) {
            verify_only = 1;
        } else if (strcmp(arg, "--debug-metrics") == 0) {
            cpu_debug_metrics_enabled = 1;
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "-o requires an argument\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            output = argv[++i];
        } else if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "-a requires an argument\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            const char *alg_str = argv[++i];
            g_alg = alg_from_string(alg_str);
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "-L") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "-l requires an argument\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            if (mode_decompress) {
                fprintf(stderr, "-l cannot be used with -d (decompress mode)\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            kernel_spec = argv[++i];
            g_variant = variant_from_string(kernel_spec);
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            free(auto_output);
            return 0;
        } else if (arg[0] == '-' && strcmp(arg, "-") != 0) {
            fprintf(stderr, "unknown option: %s\n", arg);
            print_usage(argv[0]);
            free(auto_output);
            return 1;
        } else {
            if (!input) {
                input = arg;
            } else if (!output) {
                output = arg;
            } else {
                fprintf(stderr, "too many positional arguments\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
        }
    }

    if (!input) {
        print_usage(argv[0]);
        free(auto_output);
        return 1;
    }

    /* Enforce variant restrictions */
    if (g_alg == ALG_1Y && g_variant != VAR_1 && g_variant != VAR_999) {
        g_variant = VAR_1;
    }

    if (threads <= 0) threads = auto_detect_threads();

    if (!output) {
        if (strcmp(input, "-") == 0) {
            output = "-";
        } else if (!mode_decompress) {
            size_t len = strlen(input);
            auto_output = (char *)malloc(len + 5u);
            if (!auto_output) {
                fprintf(stderr, "malloc failed\n");
                return 1;
            }
            strcpy(auto_output, input);
            strcat(auto_output, ".lzo");
            output = auto_output;
        } else {
            size_t len = strlen(input);
            if (len > 4 && strcmp(input + len - 4, ".lzo") == 0) {
                auto_output = (char *)malloc(len - 3u);
                if (!auto_output) {
                    fprintf(stderr, "malloc failed\n");
                    return 1;
                }
                memcpy(auto_output, input, len - 4);
                auto_output[len - 4] = '\0';
            } else {
                auto_output = (char *)malloc(len + 15u);
                if (!auto_output) {
                    fprintf(stderr, "malloc failed\n");
                    return 1;
                }
                strcpy(auto_output, "decompressed_");
                strcat(auto_output, input);
            }
            output = auto_output;
        }
    }

    int rc;

    if (bench_mode && do_bench) {
        fprintf(stderr, "--bench and --benchmark are mutually exclusive\n");
        free(auto_output);
        return 1;
    }
    /* Only set a default algorithm label when compressing; do not set/print
     * a default when in decompress mode, otherwise decompress runs without
     * an explicit -L will still print a misleading default label. */

    if (!mode_decompress && !do_bench && !bench_mode) {
        fprintf(stderr, "Using algorithm: %s, variant: %s\n",
            alg_to_str(g_alg),
            (g_variant == VAR_1) ? "1" :
            (g_variant == VAR_1K) ? "1k" :
            (g_variant == VAR_1L) ? "1l" :
            (g_variant == VAR_1O) ? "1o" :
            (g_variant == VAR_999) ? "999" : "unknown");
    }

    if (mode_decompress) {
        rc = decompress_file(input, output, threads, verify_only);
    } else if (bench_mode) {
        rc = run_stable_kernel_bench_file(input, level, threads, bench_seconds);
    } else {
        rc = compress_file(input, output, level, threads, do_bench, verify_only);
    }

    free(auto_output);
    return rc;
}

/* Compress into a caller-provided buffer `out` with capacity `out_cap`.
 * Returns LZO_E_OK on success and sets *out_size to the compressed length.
 */
static int compress_block_into(const unsigned char *in, size_t in_size,
                               unsigned char *out, size_t out_cap, size_t *out_size,
                               alg_t compression_alg, variant_t variant, void *wrkmem_in) {
    if (!out || out_cap == 0) return LZO_E_OUT_OF_MEMORY;
    lzo_align_t *wrkmem_ptr = NULL;
    lzo_align_t *wrkmem_heap = NULL;
    if (wrkmem_in) {
        wrkmem_ptr = (lzo_align_t *)wrkmem_in;
    } else {
        size_t wrkmem_sz = workmem_size_for(compression_alg, variant);
        if (lzo_cpu_aligned_alloc((void **)&wrkmem_heap, sizeof(lzo_align_t), wrkmem_sz) != 0) {
            wrkmem_heap = NULL;
        }
        if (!wrkmem_heap) return LZO_E_OUT_OF_MEMORY;
        memset(wrkmem_heap, 0, wrkmem_sz);
        wrkmem_ptr = wrkmem_heap;
    }

    lzo_uint dst_len = (lzo_uint)out_cap;
    int rc;
    switch (compression_alg) {
        case ALG_1X:
            if (variant == VAR_999)
                rc = lzo1x_999_compress_level(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr, NULL, 0, NULL, 9);
            else if (variant == VAR_1K)
                rc = lzo1x_1_12_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            else if (variant == VAR_1L)
                rc = lzo1x_1_11_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            else if (variant == VAR_1O)
                rc = lzo1x_1_15_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            else
                rc = lzo1x_1_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            break;
        case ALG_1Y:
            if (variant == VAR_999)
                rc = lzo1y_999_compress_level(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr, NULL, 0, NULL, 9);
            else
                rc = lzo1y_1_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            break;
        default:
            rc = lzo1x_1_compress(in, (lzo_uint)in_size, out, &dst_len, wrkmem_ptr);
            break;
    }
    if (wrkmem_heap) lzo_cpu_aligned_free(wrkmem_heap);
    if (rc != LZO_E_OK) return rc;
    *out_size = (size_t)dst_len;
    return LZO_E_OK;
}

/* Helper: print a timing value in microseconds with a friendly unit.
 * If value is zero, print N/A. For >=1000 us, print ms with fractional parts;
 * otherwise print integer microseconds.
 */
static inline void print_us_tag(FILE *f, const char *tag, unsigned long us) {
    if (!f) return;
    if (us == 0) {
        fprintf(f, "%-22s : %8s\n", tag, "N/A");
    } else if (us >= 1000UL) {
        fprintf(f, "%-22s : %8.3f ms\n", tag, us / 1000.0);
    } else {
        fprintf(f, "%-22s : %8lu us\n", tag, us);
    }
}
