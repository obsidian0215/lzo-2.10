#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <lzo/lzo1x.h>
#include "lzo_levels.h"
#include <sys/time.h>

#define NUM_THREADS_DEFAULT 4
#define LZO_WORK_MEM_SIZE   LZO1X_1_MEM_COMPRESS  // 每个线程使用的LZO工作内存大小
#define HEAP_ALLOC(var, size) \
    lzo_align_t __LZO_MMODEL var[((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t)]

typedef struct {
    unsigned char *in;
    size_t in_size;
    unsigned char *comp;
    size_t comp_size;
    size_t offset;
    unsigned char *out;
    int    index;
} chunk_t;

static double get_time_diff_ms(struct timeval *s, struct timeval *e) {
    return (e->tv_sec - s->tv_sec) * 1000.0 + (e->tv_usec - s->tv_usec) / 1000.0;
}

// 单块压缩
// compress a single block at a given compression level
// compression_level: 1..4 (higher = better compression, slower)
int compress_block_level(const unsigned char *in, size_t in_size,
                   unsigned char **out, size_t *out_size, int compression_level) {
    /* Choose the compress function according to compression_level.
     * We map levels to the corresponding lzo1x variant compiled from ../src/:
     * 1 -> lzo1x_1_11_compress
     * 2 -> lzo1x_1_12_compress
     * 3 -> lzo1x_1_compress
     * 4 -> lzo1x_1_15_compress
     */
    size_t max_comp = in_size + in_size / 16 + 64 + 3;
    *out = malloc(max_comp);
    if (!*out) return -1;
    HEAP_ALLOC(wrkmem, LZO_WORK_MEM_SIZE);
    /*
     * NOTE:
     * The upstream miniLZO in this tree exposes `lzo1x_1_compress` only.
     * To support multiple runtime compression levels similar to lzo_gpu,
     * we would either need additional compiled variants (with different
     * D_BITS) or a parameterized compressor implementation. As a safe
     * incremental change we keep using the existing compressor here and
     * expose the `compression_level` parameter in the API so callers can
     * select different implementations in the future.
     */
    int r = LZO_E_ERROR;
    switch (compression_level) {
        case 1:
            r = lzo1x_1_11_compress(in, in_size, *out, &max_comp, wrkmem);
            break;
        case 2:
            r = lzo1x_1_12_compress(in, in_size, *out, &max_comp, wrkmem);
            break;
        case 4:
            r = lzo1x_1_15_compress(in, in_size, *out, &max_comp, wrkmem);
            break;
        case 3:
        default:
            r = lzo1x_1_compress(in, in_size, *out, &max_comp, wrkmem);
            break;
    }
    if (r != LZO_E_OK) { free(*out); return r; }
    *out_size = max_comp;
    return LZO_E_OK;
}

// 单块解压
int decompress_block(const unsigned char *in, size_t in_size,
                     unsigned char *out, size_t orig_size) {
    size_t outlen = orig_size;
    int r = lzo1x_decompress(in, in_size, out, &outlen, NULL);
    return (r == LZO_E_OK && outlen == orig_size) ? LZO_E_OK : r;
}

// 解压线程
void *thread_decompress(void *arg) {
    chunk_t *ck = (chunk_t*)arg;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    int r = decompress_block(ck->comp, ck->comp_size, ck->out, ck->in_size);
    gettimeofday(&t1, NULL);
    double decomp_ms = get_time_diff_ms(&t0, &t1);
        if (r != LZO_E_OK)
            fprintf(stderr, "[Decompress Th%d] ERROR %d\n", ck->index & 0xFFFFFF, r);
        else
            printf("[Decompress Th%d] %.3f ms, offset=%zu, size=%zu\n",
                     ck->index & 0xFFFFFF, decomp_ms,
                     ck->offset,
                     ck->in_size);

    return NULL;
}

// 读取整个文件
unsigned char* read_file(const char *fname, size_t *sz) {
    FILE *f = fopen(fname, "rb");
    if (!f) {
        perror(fname);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    *sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*sz);
    fread(buf, 1, *sz, f);
    fclose(f);
    return buf;
}

// 压缩线程
// compress thread: honor compression level in chunk->index (or separate field)
void* thread_compress(void *arg) {
    chunk_t *ck = (chunk_t*)arg;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    /* ck->index remains the chunk index; we store level in comp_size as temporary
     * The caller now sets ck->offset as byte offset and uses ck->index for chunk id.
     * To keep api backward-compatible we pass a default level if not provided.
     */
    int level = (int)(ck->index >> 24); /* if caller encoded level here */
    if (level < 1 || level > 4) level = 3; /* default balanced */
    /* real compression call (level-aware) */
    compress_block_level(ck->in, ck->in_size, &ck->comp, &ck->comp_size, level);
    gettimeofday(&t1, NULL);
    double dur = get_time_diff_ms(&t0, &t1);
    long start_ms = t0.tv_sec * 1000L + t0.tv_usec / 1000;
    long end_ms   = t1.tv_sec * 1000L + t1.tv_usec / 1000;
    printf("[Compress Thread %d] start=%ld ms, end=%ld ms, duration=%.3f ms\n",
           ck->index & 0xFFFFFF, start_ms, end_ms, dur);
    return NULL;
}

// 多线程压缩
// multi-thread compress with runtime compression level (1..4)
int compress_multi(unsigned char *input, size_t fsize,
                   int num_threads, int compression_level,
                   chunk_t **chunks, int *chunk_cnt,
                   double *elapsed_ms, size_t *total_comp) {
    struct timeval st, et;
    gettimeofday(&st, NULL);
    // 计算每个线程的块大小：每个线程大致处理 fsize / num_threads 大小的块
    size_t block_size = (fsize + num_threads - 1) / num_threads;

    // 计算数据块数目
    int max_chunks = (fsize + block_size - 1) / block_size;
    *chunk_cnt = max_chunks;
    *chunks = calloc(max_chunks, sizeof(chunk_t));

    // 如果线程数大于数据块数，调整线程数为块数
    if (num_threads > max_chunks) {
        num_threads = max_chunks;
    }

    pthread_t *ths = malloc(max_chunks * sizeof(pthread_t));  // 为每个块分配线程

    // 为每个数据块创建线程
    for (int i = 0; i < max_chunks; i++) {
        size_t off = i * block_size;
        (*chunks)[i].in      = input + off;
        (*chunks)[i].in_size = (off + block_size <= fsize) ? block_size : (fsize - off);
        /* encode compression level into the high byte of index to pass it
         * into the thread function without changing the struct layout.
         * lower 24 bits keep the chunk id.
         */
        (*chunks)[i].index   = (compression_level << 24) | (i & 0xFFFFFF);
        (*chunks)[i].offset   = off;
        //pthread_create(&ths[i % num_threads], NULL, thread_compress, &(*chunks)[i]);
        pthread_create(&ths[i], NULL, thread_compress, &(*chunks)[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < max_chunks; i++) {
        pthread_join(ths[i], NULL);
    }
    gettimeofday(&et, NULL);

    *elapsed_ms = get_time_diff_ms(&st, &et);
    *total_comp = 0;
    for (int i = 0; i < max_chunks; i++) {
        *total_comp += (*chunks)[i].comp_size;
    }

    free(ths);
    return 0;
}

// 多线程解压
int decompress_multi(chunk_t *chunks, int cnt,
                     unsigned char *recon, size_t fsize,
                     double *total_ms) {
    pthread_t *ths = malloc(cnt * sizeof(pthread_t));
    struct timeval st, en;
    // 设置 out 指针
    for (int i = 0; i < cnt; i++) {
        chunks[i].out = recon + chunks[i].offset;
    }
    gettimeofday(&st, NULL);
    for (int i = 0; i < cnt; i++) {
        pthread_create(&ths[i], NULL, thread_decompress, &chunks[i]);
    }
    for (int i = 0; i < cnt; i++) pthread_join(ths[i], NULL);
    gettimeofday(&en, NULL);
    *total_ms = get_time_diff_ms(&st, &en);
    free(ths);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
      fprintf(stderr, "Usage: %s [-1|-2|-3|-4|-d] <file> [threads]\n", argv[0]);
      return 1;
    }

    int mode = 1; /* 1 = compress, 2 = decompress */
    int compression_level = 3; /* default balanced */
    int arg_idx = 1;

    if (strcmp(argv[1], "-d") == 0) {
        mode = 2; arg_idx = 2;
    } else if (strcmp(argv[1], "-1") == 0) {
        compression_level = 1; arg_idx = 2;
    } else if (strcmp(argv[1], "-2") == 0) {
        compression_level = 2; arg_idx = 2;
    } else if (strcmp(argv[1], "-3") == 0) {
        compression_level = 3; arg_idx = 2;
    } else if (strcmp(argv[1], "-4") == 0) {
        compression_level = 4; arg_idx = 2;
    }

    int threads = (argc > arg_idx + 1) ? atoi(argv[arg_idx + 1]) : NUM_THREADS_DEFAULT;
    size_t fsize;
    unsigned char *in = read_file(argv[arg_idx], &fsize);
    if (!in)
      return fprintf(stderr,"read error\n"),1;

    // 单线程
    struct timeval a,b;
    gettimeofday(&a,NULL);
    unsigned char *scomp; size_t scsz;
    if (mode == 1) {
        /* single-block compress honoring level */
        compress_block_level(in,fsize,&scomp,&scsz, compression_level);
    } else {
        fprintf(stderr, "decompress mode: use -d <file>\n");
        return 1;
    }
    gettimeofday(&b,NULL);
    double c1 = get_time_diff_ms(&a,&b);
    printf("Single Compress: %.3f ms, %zu bytes, ratio=%.2f%%, thpt=%.2f MB/s\n",
           c1, scsz, scsz*100.0/fsize, (fsize/1048576.0)/(c1/1000));

    gettimeofday(&a,NULL);
    unsigned char *sdec = malloc(fsize);
    decompress_block(scomp,scsz,sdec,fsize);
    gettimeofday(&b,NULL);
    double d1 = get_time_diff_ms(&a,&b);
    printf("Single Decompress: %.3f ms, thpt=%.2f MB/s, verify=%s\n",
           d1, (fsize/1048576.0)/(d1/1000), memcmp(sdec,in,fsize)==0?"OK":"FAIL");

    free(sdec); free(scomp);

    // 多线程
    chunk_t *cks; int cnt;
    double cm_ms; size_t totc;
    compress_multi(in,fsize,threads, compression_level, &cks,&cnt,&cm_ms,&totc);
    printf("Multi Compress: threads=%d, %.3f ms, %zu bytes, ratio=%.2f%%, thpt=%.2f MB/s\n",
           threads, cm_ms, totc, totc*100.0/fsize, (fsize/1048576.0)/(cm_ms/1000));

    unsigned char *recon = malloc(fsize);
    double dm_ms;
    decompress_multi(cks,cnt,recon,fsize,&dm_ms);
    printf("Multi Decompress: threads=%d, %.3f ms, thpt=%.2f MB/s, verify=%s\n",
           threads, dm_ms, (fsize/1048576.0)/(dm_ms/1000), memcmp(recon,in,fsize)==0?"OK":"FAIL");

    // cleanup
    for (int i=0;i<cnt;i++)
      free(cks[i].comp);
    free(cks);
    free(recon);
    free(in);
    return 0;
}