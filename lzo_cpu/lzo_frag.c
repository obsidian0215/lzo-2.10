/*
 * lzo_frag.c -- CPU driver mirroring lzo_gpu CLI semantics
 * Supports runtime compression level selection, threaded compression
 * and decompression, containerized I/O compatible with the GPU tool,
 * and an opt-in benchmark mode built from the original test harness.
 */

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

#include <lzo/lzo1x.h>
#include "lzo_levels.h"

#define MAGIC_TAG            0x4C5A       /* 'L''Z' */
#define DEFAULT_THREAD_COUNT 4
#define MIN_BLOCK_SIZE       (64u * 1024u)
#define MAX_BLOCK_SIZE       (1024u * 1024u)
#define LZO_WORK_MEM_SIZE    LZO1X_1_MEM_COMPRESS

typedef struct {
    const unsigned char *in;
    size_t in_size;
    unsigned char *comp;
    size_t comp_size;
    size_t offset;
    unsigned char *out;
} chunk_t;

typedef struct {
    chunk_t *chunks;
    size_t chunk_count;
    size_t next_index;
    int compression_level;
    int status;
    pthread_mutex_t lock;
} compress_job_t;

typedef struct {
    chunk_t *chunks;
    size_t chunk_count;
    size_t next_index;
    int status;
    pthread_mutex_t lock;
} decompress_job_t;

#define HEAP_ALLOC(var, size) \
    lzo_align_t __LZO_MMODEL var[((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t)]

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

static double diff_ms(const struct timeval *start, const struct timeval *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_usec - start->tv_usec) / 1000.0;
}

static size_t choose_block_size(size_t total_bytes, int threads) {
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
                                int compression_level) {
    size_t cap = in_size + in_size / 16u + 64u + 3u;
    *out = (unsigned char *)malloc(cap);
    if (!*out) return LZO_E_OUT_OF_MEMORY;
    HEAP_ALLOC(wrkmem, LZO_WORK_MEM_SIZE);

    lzo_uint dst_len = (lzo_uint)cap;
    int rc;
    switch (compression_level) {
        case 1: rc = lzo1x_1_11_compress(in, (lzo_uint)in_size, *out, &dst_len, wrkmem); break;
        case 2: rc = lzo1x_1_12_compress(in, (lzo_uint)in_size, *out, &dst_len, wrkmem); break;
        case 4: rc = lzo1x_1_15_compress(in, (lzo_uint)in_size, *out, &dst_len, wrkmem); break;
        case 3:
        default: rc = lzo1x_1_compress(in, (lzo_uint)in_size, *out, &dst_len, wrkmem); break;
    }
    if (rc != LZO_E_OK) {
        free(*out);
        *out = NULL;
        return rc;
    }
    *out_size = (size_t)dst_len;
    return LZO_E_OK;
}

static int decompress_block(const unsigned char *in, size_t in_size,
                            unsigned char *out, size_t orig_size) {
    lzo_uint dst_len = (lzo_uint)orig_size;
    int rc = lzo1x_decompress(in, (lzo_uint)in_size, out, &dst_len, NULL);
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
    while (1) {
        size_t idx;
        pthread_mutex_lock(&job->lock);
        if (job->status != LZO_E_OK) {
            pthread_mutex_unlock(&job->lock);
            break;
        }
        if (job->next_index >= job->chunk_count) {
            pthread_mutex_unlock(&job->lock);
            break;
        }
        idx = job->next_index++;
        pthread_mutex_unlock(&job->lock);

        chunk_t *ck = &job->chunks[idx];
        unsigned char *out = NULL;
        size_t out_len = 0;
        int rc = compress_block_level(ck->in, ck->in_size, &out, &out_len, job->compression_level);
        if (rc != LZO_E_OK) {
            pthread_mutex_lock(&job->lock);
            job->status = rc;
            pthread_mutex_unlock(&job->lock);
            break;
        }
        ck->comp = out;
        ck->comp_size = out_len;
    }
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

    compress_job_t job;
    job.chunks = chunks;
    job.chunk_count = chunk_count;
    job.next_index = 0;
    job.compression_level = level;
    job.status = LZO_E_OK;
    pthread_mutex_init(&job.lock, NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);

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

    gettimeofday(&end, NULL);

    if (elapsed_ms) *elapsed_ms = diff_ms(&start, &end);

    int status = job.status;
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

    return LZO_E_OK;
}

static void *decompress_worker(void *opaque) {
    decompress_job_t *job = (decompress_job_t *)opaque;
    while (1) {
        size_t idx;
        pthread_mutex_lock(&job->lock);
        if (job->status != LZO_E_OK) {
            pthread_mutex_unlock(&job->lock);
            break;
        }
        if (job->next_index >= job->chunk_count) {
            pthread_mutex_unlock(&job->lock);
            break;
        }
        idx = job->next_index++;
        pthread_mutex_unlock(&job->lock);

        chunk_t *ck = &job->chunks[idx];
        int rc = decompress_block(ck->comp, ck->comp_size, ck->out, ck->in_size);
        if (rc != LZO_E_OK) {
            pthread_mutex_lock(&job->lock);
            job->status = rc;
            pthread_mutex_unlock(&job->lock);
            break;
        }
    }
    return NULL;
}

static int decompress_multi(chunk_t *chunks, size_t chunk_count,
                            int threads, double *elapsed_ms) {
    if (threads < 1) threads = 1;
    if (chunk_count == 0) {
        if (elapsed_ms) *elapsed_ms = 0.0;
        return LZO_E_OK;
    }

    decompress_job_t job;
    job.chunks = chunks;
    job.chunk_count = chunk_count;
    job.next_index = 0;
    job.status = LZO_E_OK;
    pthread_mutex_init(&job.lock, NULL);

    pthread_t *workers = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!workers) {
        pthread_mutex_destroy(&job.lock);
        return LZO_E_OUT_OF_MEMORY;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < threads; ++i)
        pthread_create(&workers[i], NULL, decompress_worker, &job);
    for (int i = 0; i < threads; ++i)
        pthread_join(workers[i], NULL);
    gettimeofday(&end, NULL);

    if (elapsed_ms) *elapsed_ms = diff_ms(&start, &end);
    int status = job.status;
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

    struct timeval t0, t1;
    unsigned char *single_comp = NULL;
    size_t single_comp_len = 0;
    gettimeofday(&t0, NULL);
    int rc = compress_block_level(data, size, &single_comp, &single_comp_len, level);
    gettimeofday(&t1, NULL);
    if (rc != LZO_E_OK) {
        fprintf(stderr, "single-block compress failed: %d\n", rc);
        return;
    }
    double single_comp_ms = diff_ms(&t0, &t1);

    unsigned char *single_out = (unsigned char *)malloc(size ? size : 1u);
    if (!single_out) {
        fprintf(stderr, "malloc failed\n");
        free(single_comp);
        return;
    }
    gettimeofday(&t0, NULL);
    rc = decompress_block(single_comp, single_comp_len, single_out, size);
    gettimeofday(&t1, NULL);
    double single_decomp_ms = diff_ms(&t0, &t1);
    fprintf(stderr, "Single  Compress : %.3f ms (%.2f MB/s)\n",
            single_comp_ms, size ? (size / 1048576.0) / (single_comp_ms / 1000.0) : 0.0);
    fprintf(stderr, "Single  Decompress: %.3f ms (%.2f MB/s) verify=%s\n",
            single_decomp_ms,
            size ? (size / 1048576.0) / (single_decomp_ms / 1000.0) : 0.0,
            (rc == LZO_E_OK && memcmp(single_out, data, size) == 0) ? "OK" : "FAIL");

    free(single_out);

    size_t block_size = choose_block_size(size, threads);
    chunk_t *chunks = NULL;
    size_t chunk_count = 0;
    double multi_comp_ms = 0.0;
    size_t total_comp = 0;
    rc = compress_multi(data, size, block_size, threads, level,
                        &chunks, &chunk_count, &multi_comp_ms, &total_comp);
    if (rc != LZO_E_OK) {
        fprintf(stderr, "multi compress failed: %d\n", rc);
        free(single_comp);
        return;
    }

    fprintf(stderr, "Multi   Compress : %.3f ms (%zu blocks, %.2f MB/s)\n",
            multi_comp_ms,
            chunk_count,
            size ? (size / 1048576.0) / (multi_comp_ms / 1000.0) : 0.0);

    unsigned char *multi_out = (unsigned char *)malloc(size ? size : 1u);
    if (!multi_out) {
        fprintf(stderr, "malloc failed\n");
        free(single_comp);
        free_compression_chunks(chunks, chunk_count);
        return;
    }
    for (size_t i = 0; i < chunk_count; ++i)
        chunks[i].out = multi_out + chunks[i].offset;

    double multi_decomp_ms = 0.0;
    rc = decompress_multi(chunks, chunk_count, threads, &multi_decomp_ms);
    fprintf(stderr, "Multi   Decompress: %.3f ms (%.2f MB/s) verify=%s\n",
            multi_decomp_ms,
            size ? (size / 1048576.0) / (multi_decomp_ms / 1000.0) : 0.0,
            (rc == LZO_E_OK && memcmp(multi_out, data, size) == 0) ? "OK" : "FAIL");

    free(multi_out);
    free(single_comp);
    free_compression_chunks(chunks, chunk_count);
}

static int compress_file(const char *input_path, const char *output_path,
                         int level, int threads, int do_bench) {
    size_t input_size = 0;
    unsigned char *input = read_entire(input_path, &input_size);
    if (!input && input_size != 0) return 1;

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

    size_t header_size = 2u + 4u + 4u + 4u + chunk_count * 4u;
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
    for (size_t i = 0; i < chunk_count; ++i) {
        write_u32(out_buf + cursor, (uint32_t)chunks[i].comp_size);
        cursor += 4u;
    }
    for (size_t i = 0; i < chunk_count; ++i) {
        memcpy(out_buf + cursor, chunks[i].comp, chunks[i].comp_size);
        cursor += chunks[i].comp_size;
    }

    if (write_entire(output_path, out_buf, total_size) != 0) {
        fprintf(stderr, "failed to write output\n");
        free(out_buf);
        free(input);
        free_compression_chunks(chunks, chunk_count);
        return 1;
    }

    fprintf(stderr,
            "Compressed %zu bytes -> %zu bytes (%.2f%%) blocks=%zu block_sz=%zu threads=%d level=%d time=%.3f ms (%.2f MB/s)\n",
            input_size,
            total_comp,
            input_size ? (100.0 * total_comp / input_size) : 0.0,
            chunk_count,
            block_size,
            threads,
            level,
            comp_ms,
            comp_ms > 0.0 ? (input_size / 1048576.0) / (comp_ms / 1000.0) : 0.0);

    if (do_bench) run_benchmark(input, input_size, level, threads);

    free(out_buf);
    free(input);
    free_compression_chunks(chunks, chunk_count);
    return 0;
}

static int decompress_file(const char *input_path, const char *output_path,
                           int threads) {
    size_t comp_size = 0;
    unsigned char *comp = read_entire(input_path, &comp_size);
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
    int rc = decompress_multi(chunks, nblk, threads, &decomp_ms);
    if (rc != LZO_E_OK) {
        fprintf(stderr, "decompress failed: %d\n", rc);
        free(output);
        free(comp);
        free(chunks);
        return 1;
    }

    if (write_entire(output_path, output, output_size) != 0) {
        fprintf(stderr, "failed to write output\n");
        free(output);
        free(comp);
        free(chunks);
        return 1;
    }

    fprintf(stderr,
            "Decompressed %zu bytes -> %u bytes (blocks=%u block_sz=%u threads=%d time=%.3f ms %.2f MB/s)\n",
            total_comp,
            orig_sz,
            nblk,
            blk_sz,
            threads,
            decomp_ms,
            decomp_ms > 0.0 ? (orig_sz / 1048576.0) / (decomp_ms / 1000.0) : 0.0);

    free(output);
    free(comp);
    free(chunks);
    return 0;
}

static int parse_int(const char *s, int *out) {
    if (!s || !out) return -1;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end != '\0') return -1;
    if (v <= 0 || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options] <input> [output]\n"
            "Options:\n"
            "  -1|-2|-3|-4     Select compression level (default -3)\n"
            "  -d              Decompress instead of compress\n"
            "  -t <threads>    Worker thread count (default %d)\n"
            "  --benchmark     Run benchmark metrics after operation\n"
            "  -h, --help      Show this help\n"
            "  Use '-' for stdin/stdout. Output defaults to input with .lzo (compress)\n"
            "  or stripped .lzo extension (decompress).\n",
            prog, DEFAULT_THREAD_COUNT);
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

    const char *input = NULL;
    const char *output = NULL;
    char *auto_output = NULL;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "-d") == 0) {
            mode_decompress = 1;
        } else if (strcmp(arg, "-1") == 0) {
            level = 1;
        } else if (strcmp(arg, "-2") == 0) {
            level = 2;
        } else if (strcmp(arg, "-3") == 0 || strcmp(arg, "-c") == 0) {
            level = 3;
        } else if (strcmp(arg, "-4") == 0) {
            level = 4;
        } else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || parse_int(argv[i + 1], &threads) != 0) {
                fprintf(stderr, "invalid thread count\n");
                print_usage(argv[0]);
                free(auto_output);
                return 1;
            }
            ++i;
        } else if (strcmp(arg, "--benchmark") == 0 || strcmp(arg, "-b") == 0) {
            do_bench = 1;
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
    if (mode_decompress) {
        rc = decompress_file(input, output, threads);
    } else {
        rc = compress_file(input, output, level, threads, do_bench);
    }

    free(auto_output);
    return rc;
}