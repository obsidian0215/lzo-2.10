/*
 * lzo_gpu_io.c - Optimized I/O implementation for LZO GPU
 */

#include "lzo_gpu_io.h"
#include "lzo_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

/* Internal worker thread argument structure */
typedef struct {
    int fd;          /* File descriptor to read from */
    void *dest;      /* Base destination buffer */
    off_t off;       /* Offset within dest (and file position) */
    size_t len;      /* Number of bytes to read */
    int err;         /* errno value on failure, 0 on success */
} mt_io_worker_arg_t;

/* Extended worker argument for reading with separate file/dest offsets */
typedef struct {
    int fd;          /* File descriptor to read from */
    void *dest;      /* Base destination buffer */
    off_t file_off;  /* File position to read from */
    off_t dest_off;  /* Offset within dest buffer */
    size_t len;      /* Number of bytes to read */
    int err;         /* errno value on failure, 0 on success */
} mt_io_worker_arg_ex_t;

/* Get current time in nanoseconds */
static inline uint64_t mt_io_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Worker thread function for multi-threaded pread */
static void* mt_pread_worker(void *arg) {
    mt_io_worker_arg_t *a = (mt_io_worker_arg_t*)arg;

    if (!a || a->len == 0) {
        if (a) a->err = 0;
        return NULL;
    }

    /* Check for potential offset overflow */
    if ((uint64_t)a->off + (uint64_t)a->len < (uint64_t)a->off) {
        a->err = EINVAL;
        return NULL;
    }

    size_t left = a->len;
    off_t pos = a->off;
    char *p = (char*)a->dest + pos;

    while (left > 0) {
        ssize_t r;
        do {
            r = pread(a->fd, p, left, pos);
        } while (r < 0 && errno == EINTR);

        if (r < 0) {
            a->err = errno;
            return NULL;
        }
        if (r == 0) {
            /* Unexpected EOF */
            a->err = EIO;
            return NULL;
        }

        left -= r;
        p += r;
        pos += r;
    }

    a->err = 0;
    return NULL;
}

/* Extended worker for reading with separate file/dest offsets */
static void* mt_pread_worker_ex(void *arg) {
    mt_io_worker_arg_ex_t *a = (mt_io_worker_arg_ex_t*)arg;

    if (!a || a->len == 0) {
        if (a) a->err = 0;
        return NULL;
    }

    /* Check for potential offset overflow */
    if ((uint64_t)a->file_off + (uint64_t)a->len < (uint64_t)a->file_off ||
        (uint64_t)a->dest_off + (uint64_t)a->len < (uint64_t)a->dest_off) {
        a->err = EINVAL;
        return NULL;
    }

    size_t left = a->len;
    off_t file_pos = a->file_off;
    char *p = (char*)a->dest + a->dest_off;

    while (left > 0) {
        ssize_t r;
        do {
            r = pread(a->fd, p, left, file_pos);
        } while (r < 0 && errno == EINTR);

        if (r < 0) {
            a->err = errno;
            return NULL;
        }

        if (r == 0) {
            /* Unexpected EOF */
            a->err = EIO;
            return NULL;
        }

        left -= (size_t)r;
        p += r;
        file_pos += r;
    }

    a->err = 0;
    return NULL;
}

int lzo_mt_read_file(const char* path, void* dest, size_t size, int num_threads, unsigned long* read_us_out) {
    if (!path || !dest || size == 0) {
        errno = EINVAL;
        return -1;
    }

    /* Clamp thread count to tested optimal range */
    num_threads = lzo_clamp_int(num_threads, LZO_MIN_MT_IO_THREADS, LZO_MAX_MT_IO_THREADS);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    uint64_t t_start = mt_io_now_ns();

    /* Allocate worker arguments and thread IDs */
    mt_io_worker_arg_t *args = malloc(sizeof(mt_io_worker_arg_t) * num_threads);
    pthread_t *tids = malloc(sizeof(pthread_t) * num_threads);

    if (!args || !tids) {
        free(args);
        free(tids);
        close(fd);
        errno = ENOMEM;
        return -1;
    }

    /* Divide work among threads */
    size_t piece = (size + num_threads - 1) / num_threads;

    /* Create worker threads */
    for (int i = 0; i < num_threads; ++i) {
        off_t off = (off_t)i * (off_t)piece;
        size_t len = ((size_t)off + piece > size) ? (size - (size_t)off) : piece;

        args[i].fd = fd;
        args[i].dest = dest;
        args[i].off = off;
        args[i].len = len;
        args[i].err = 0;

        int rc = pthread_create(&tids[i], NULL, mt_pread_worker, &args[i]);
        if (rc != 0) {
            /* Failed to create thread - join already created threads and fallback */
            for (int j = 0; j < i; ++j) {
                pthread_join(tids[j], NULL);
            }
            free(args);
            free(tids);
            close(fd);
            errno = rc;
            return -1;
        }
    }

    /* Wait for all threads to complete */
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(tids[i], NULL);
    }

    /* Check for errors in any worker */
    int result = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (args[i].err != 0) {
            errno = args[i].err;
            result = -1;
            break;
        }
    }

    uint64_t t_end = mt_io_now_ns();

    if (read_us_out) {
        *read_us_out = (unsigned long)((t_end - t_start) / 1000);
    }

    free(args);
    free(tids);
    close(fd);

    return result;
}

/* Multi-threaded read from file descriptor at specific offset */
int lzo_mt_read_file_fd(int fd, void* dest, size_t size, off_t file_offset, int num_threads, unsigned long* read_us_out) {
    if (!dest || size == 0) {
        errno = EINVAL;
        return -1;
    }

    /* Clamp thread count to tested optimal range */
    num_threads = lzo_clamp_int(num_threads, LZO_MIN_MT_IO_THREADS, LZO_MAX_MT_IO_THREADS);

    /* For small files, use single thread */
    if (size < LZO_MT_IO_SIZE_THRESHOLD || num_threads == 1) {
        uint64_t t_start = mt_io_now_ns();
        ssize_t nread = pread(fd, dest, size, file_offset);
        uint64_t t_end = mt_io_now_ns();

        if (nread != (ssize_t)size) {
            return -1;
        }

        if (read_us_out) {
            *read_us_out = (unsigned long)((t_end - t_start) / 1000);
        }
        return 0;
    }

    uint64_t t_start = mt_io_now_ns();

    /* Allocate thread data */
    pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    mt_io_worker_arg_ex_t* args = (mt_io_worker_arg_ex_t*)malloc(sizeof(mt_io_worker_arg_ex_t) * num_threads);

    if (!tids || !args) {
        if (tids) free(tids);
        if (args) free(args);
        errno = ENOMEM;
        return -1;
    }

    /* Divide work among threads */
    size_t piece = (size + num_threads - 1) / num_threads;

    /* Create worker threads */
    for (int i = 0; i < num_threads; ++i) {
        off_t dest_off = (off_t)i * (off_t)piece;
        size_t len = ((size_t)dest_off + piece > size) ? (size - (size_t)dest_off) : piece;

        args[i].fd = fd;
        args[i].dest = dest;
        args[i].file_off = file_offset + dest_off;  /* File position */
        args[i].dest_off = dest_off;                /* Dest buffer offset */
        args[i].len = len;
        args[i].err = 0;

        int rc = pthread_create(&tids[i], NULL, mt_pread_worker_ex, &args[i]);
        if (rc != 0) {
            /* Failed to create thread - join already created threads */
            for (int j = 0; j < i; ++j) {
                pthread_join(tids[j], NULL);
            }
            free(args);
            free(tids);
            errno = rc;
            return -1;
        }
    }

    /* Wait for all threads to complete */
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(tids[i], NULL);
    }

    /* Check for errors in any worker */
    int result = 0;
    for (int i = 0; i < num_threads; ++i) {
        if (args[i].err != 0) {
            errno = args[i].err;
            result = -1;
            break;
        }
    }

    uint64_t t_end = mt_io_now_ns();

    if (read_us_out) {
        *read_us_out = (unsigned long)((t_end - t_start) / 1000);
    }

    free(args);
    free(tids);

    return result;
}

int lzo_st_read_file(const char* path, void* dest, size_t size, unsigned long* read_us_out) {
    if (!path || !dest || size == 0) {
        errno = EINVAL;
        return -1;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    uint64_t t_start = mt_io_now_ns();

    size_t nread = fread(dest, 1, size, f);

    uint64_t t_end = mt_io_now_ns();

    if (read_us_out) {
        *read_us_out = (unsigned long)((t_end - t_start) / 1000);
    }

    int result = 0;
    if (nread != size) {
        result = -1;
    }

    fclose(f);
    return result;
}

int lzo_read_file_with_mode(const char* path, void* dest, size_t size,
                            int enable_mt, int num_threads, unsigned long* read_us_out) {
    if (enable_mt && num_threads > 1) {
        return lzo_mt_read_file(path, dest, size, num_threads, read_us_out);
    } else {
        return lzo_st_read_file(path, dest, size, read_us_out);
    }
}

/* Write strategies: extracted functions for better maintainability */

/* Strategy 1: Full coalesce - copy all blocks to contiguous buffer then write once */
static int write_full_coalesce(FILE* f, const unsigned char* dev_out,
                               size_t nblk, const unsigned int* lens,
                               size_t worst_blk, size_t comp_total,
                               int profile_writes) {
    unsigned char* contig = (unsigned char*)malloc(comp_total);
    if (!contig) {
        return -1; /* Signal fallback needed */
    }

    fprintf(stderr, "[IO] COALESCE: full contiguous allocation success size=%zu bytes\n", comp_total);

    /* Copy all blocks to contiguous buffer */
    uint64_t t_copy_start = mt_io_now_ns();
    size_t pos = 0;
    for (size_t i = 0; i < nblk; ++i) {
        if (lens[i] > 0) {
            size_t dev_off = i * worst_blk;
            memcpy(contig + pos, dev_out + dev_off, lens[i]);
            pos += lens[i];
        }
    }
    uint64_t t_copy_end = mt_io_now_ns();

    /* Write entire buffer at once */
    uint64_t t_write_start = mt_io_now_ns();
    size_t written = fwrite(contig, 1, comp_total, f);
    uint64_t t_write_end = mt_io_now_ns();

    if (profile_writes) {
        fprintf(stderr, "COALESCE_COPY: %.3f ms\n", (t_copy_end - t_copy_start)/1e6);
        fprintf(stderr, "COALESCE_WRITE: %.3f ms\n", (t_write_end - t_write_start)/1e6);
    }

    free(contig);

    if (written != comp_total) {
        perror("fwrite contiguous");
        return -2; /* Write error */
    }

    return 0; /* Success */
}

/* Strategy 2: Chunked coalesce - write in large chunks to balance memory and I/O */
static int write_chunked(FILE* f, const unsigned char* dev_out,
                        size_t nblk, const unsigned int* lens,
                        size_t worst_blk, size_t chunk_size,
                        int profile_writes) {
    if (chunk_size == 0) {
        chunk_size = LZO_WRITE_CHUNK_DEFAULT_MB * 1024 * 1024;
    }

    unsigned char* chunk = (unsigned char*)malloc(chunk_size);
    if (!chunk) {
        return -1; /* Signal fallback needed */
    }

    fprintf(stderr, "[IO] COALESCE: chunk buffer size=%zu bytes\n", chunk_size);

    size_t used = 0;
    for (size_t i = 0; i < nblk; ++i) {
        if (lens[i] == 0) continue;

        size_t dev_off = i * worst_blk;

        /* Write oversized blocks directly */
        if (lens[i] > chunk_size && used == 0) {
            uint64_t t1 = mt_io_now_ns();
            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                perror("fwrite block large");
                free(chunk);
                return -2;
            }
            uint64_t t2 = mt_io_now_ns();
            if (profile_writes) {
                fprintf(stderr, "BLOCK_WRITE %zu len=%u : %.3f ms\n", i, lens[i], (t2 - t1)/1e6);
            }
            continue;
        }

        /* Flush chunk if adding this block would overflow */
        if (used + lens[i] > chunk_size) {
            uint64_t t_start = mt_io_now_ns();
            if (fwrite(chunk, 1, used, f) != used) {
                perror("fwrite chunk");
                free(chunk);
                return -2;
            }
            uint64_t t_end = mt_io_now_ns();
            if (profile_writes) {
                fprintf(stderr, "CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_end - t_start)/1e6, used);
            }
            used = 0;
        }

        /* Add block to chunk */
        memcpy(chunk + used, dev_out + dev_off, lens[i]);
        used += lens[i];
    }

    /* Write remaining data in chunk */
    int result = 0;
    if (used > 0) {
        uint64_t t_start = mt_io_now_ns();
        if (fwrite(chunk, 1, used, f) != used) {
            perror("fwrite chunk final");
            result = -2;
        } else {
            uint64_t t_end = mt_io_now_ns();
            if (profile_writes) {
                fprintf(stderr, "CHUNK_WRITE: %.3f ms (bytes=%zu)\n", (t_end - t_start)/1e6, used);
            }
        }
    }

    free(chunk);
    return result;
}

/* Strategy 3: Direct block writes - no coalescing, simple per-block fwrite */
static int write_direct_blocks(FILE* f, const unsigned char* dev_out,
                               size_t nblk, const unsigned int* lens,
                               size_t worst_blk, int profile_writes) {
    for (size_t i = 0; i < nblk; i++) {
        if (lens[i] == 0) continue;

        size_t dev_off = i * worst_blk;

        if (profile_writes) {
            uint64_t t1 = mt_io_now_ns();
            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                perror("fwrite block");
                return -1;
            }
            uint64_t t2 = mt_io_now_ns();
            fprintf(stderr, "BLOCK_WRITE %zu len=%u : %.3f ms\n", i, lens[i], (t2 - t1)/1e6);
        } else {
            if (fwrite(dev_out + dev_off, 1, lens[i], f) != lens[i]) {
                perror("fwrite block");
                return -1;
            }
        }
    }

    return 0;
}

/* Write compressed LZO file with coalescing optimization */
int lzo_write_compressed_file(const char* path,
                              size_t orig_size, size_t blk_size,
                              size_t nblk, const unsigned int* lens,
                              const void* sparse_data, size_t worst_blk,
                              int enable_coalesce, size_t coalesce_chunk_mb,
                              size_t coalesce_max_mb, size_t stdio_buf_mb,
                              int alg_id, int debug) {
    if (!path || !lens || !sparse_data || nblk == 0) {
        if (debug) fprintf(stderr, "[IO] write_compressed_file: invalid parameters\n");
        errno = EINVAL;
        return -1;
    }

    if (debug) fprintf(stderr, "[IO] Opening file: %s\n", path);
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("[IO] fopen");
        return -1;
    }

    if (debug) fprintf(stderr, "[IO] File opened successfully\n");

    /* Calculate total compressed size */
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) {
        comp_total += lens[i];
    }

    /* Set up stdio buffer */
    size_t vsize = stdio_buf_mb * 1024 * 1024;
    if (vsize == 0) vsize = 4 * 1024 * 1024; /* Default 4MB */
    if (vsize > comp_total && comp_total > 0) vsize = comp_total;

    char *vbuf = NULL;
    if (vsize > 0) {
        vbuf = (char*)malloc(vsize);
        if (vbuf) {
            if (setvbuf(f, vbuf, _IOFBF, (int)vsize) != 0) {
                free(vbuf);
                vbuf = NULL;
            }
        }
    }

    /* Write LZO file header */
    unsigned short magic = 0x4C5A;  /* 'LZ' */
    if (fwrite(&magic, sizeof(magic), 1, f) != 1) {
        fclose(f);
        if (vbuf) free(vbuf);
        return -1;
    }

    unsigned int u32;
    u32 = (unsigned int)orig_size;
    if (fwrite(&u32, sizeof(u32), 1, f) != 1) { fclose(f); if (vbuf) free(vbuf); return -1; }
    u32 = (unsigned int)blk_size;
    if (fwrite(&u32, sizeof(u32), 1, f) != 1) { fclose(f); if (vbuf) free(vbuf); return -1; }
    u32 = (unsigned int)nblk;
    if (fwrite(&u32, sizeof(u32), 1, f) != 1) { fclose(f); if (vbuf) free(vbuf); return -1; }
    u32 = (unsigned int)alg_id;
    if (fwrite(&u32, sizeof(u32), 1, f) != 1) { fclose(f); if (vbuf) free(vbuf); return -1; }

    /* Write block length array */
    if (fwrite(lens, sizeof(unsigned int), nblk, f) != nblk) {
        fclose(f);
        if (vbuf) free(vbuf);
        return -1;
    }

    const unsigned char* dev_out = (const unsigned char*)sparse_data;
    int profile_writes = debug;

    /* Write compressed data using appropriate strategy */
    int write_result = 0;

    if (enable_coalesce && comp_total > 0) {
        size_t threshold_bytes = coalesce_max_mb * 1024 * 1024;

        /* Strategy 1: Try full coalesce first if within threshold */
        if (threshold_bytes == 0 || comp_total <= threshold_bytes) {
            write_result = write_full_coalesce(f, dev_out, nblk, lens, worst_blk, comp_total, profile_writes);

            if (write_result == 0) {
                /* Success */
            } else if (write_result == -1) {
                /* Allocation failed, try chunked */
                if (debug) fprintf(stderr, "[IO] COALESCE: full allocation failed, using chunked coalesce\n");
                write_result = write_chunked(f, dev_out, nblk, lens, worst_blk, coalesce_chunk_mb * 1024 * 1024, profile_writes);
            } else {
                /* Write error (-2) */
                fclose(f);
                if (vbuf) free(vbuf);
                return -1;
            }
        } else {
            /* File too large, use chunked directly */
            if (debug) fprintf(stderr, "[IO] COALESCE: file too large (>%zuMB), using chunked coalesce\n", coalesce_max_mb);
            write_result = write_chunked(f, dev_out, nblk, lens, worst_blk, coalesce_chunk_mb * 1024 * 1024, profile_writes);
        }

        /* Strategy 3: Fallback to direct writes if chunked also failed */
        if (write_result == -1) {
            if (debug) fprintf(stderr, "[IO] COALESCE: chunk allocation failed, using per-block writes\n");
            write_result = write_direct_blocks(f, dev_out, nblk, lens, worst_blk, profile_writes);
        }

        if (write_result != 0) {
            fclose(f);
            if (vbuf) free(vbuf);
            return -1;
        }
    } else {
        /* No coalescing requested: direct block writes */
        write_result = write_direct_blocks(f, dev_out, nblk, lens, worst_blk, profile_writes);
        if (write_result != 0) {
            fclose(f);
            if (vbuf) free(vbuf);
            return -1;
        }
    }

    fclose(f);
    if (vbuf) free(vbuf);

    return 0;
}
