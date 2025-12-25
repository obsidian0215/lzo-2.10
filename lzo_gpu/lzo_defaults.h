/*
 * lzo_defaults.h - Default settings for lzo gpu host and daemon
 *
 * Centralize commonly used defaults (stdio buffer, coalesce behaviour)
 * so we can reduce the number of environment variables and default-enable
 * useful behavior such as coalescing and stdio buffering.
 */
#ifndef LZO_DEFAULTS_H
#define LZO_DEFAULTS_H

#include <stdlib.h>  /* for getenv, atoi */

#define LZO_DEFAULT_STDIO_BUF_MB 4
#define LZO_DEFAULT_COALESCE_OUTPUT 1
#define LZO_DEFAULT_COALESCE_MAX_MB 256
#define LZO_DEFAULT_COALESCE_CHUNK_MB 16
#define DEFAULT_DECOMP_CACHE_MB 256

/* Multi-threaded I/O configuration (based on performance testing) */
#define LZO_DEFAULT_MT_IO_THREADS 4   /* Default: 4 threads (good balance) */
#define LZO_MIN_MT_IO_THREADS 1       /* Minimum: 1 thread */
#define LZO_MAX_MT_IO_THREADS 8       /* Maximum: 8 threads (tested optimal range 4-6) */
#define LZO_MT_IO_SIZE_THRESHOLD (16 * 1024 * 1024)  /* 16MB: Use MT-IO for files larger than this */

/* Write coalescing configuration */
#define LZO_WRITE_CHUNK_DEFAULT_MB 16  /* Default chunk size for partial coalescing */
#define LZO_WRITE_CHUNK_MIN_MB 1       /* Minimum chunk size */

/* Block size and GPU parallelism configuration */
#define LZO_OCC_FACTOR_DEFAULT 256
#define LZO_ALIGN_BYTES_DEFAULT 1024  /* 1KB alignment for block sizes */
#define LZO_MIN_BLOCK_BYTES_DEFAULT (1 * 1024)   /* 1KB minimum */
#define LZO_MAX_BLOCK_BYTES_DEFAULT (64 * 1024)  /* 64KB maximum */
#define LZO_MAX_NBLOCKS_DEFAULT (16 * 1024)  /* Maximum blocks per job */

/* Entropy calculation configuration */
#define LZO_ADAPTIVE_SAMPLE_SIZE (64 * 1024)  /* 64KB sample for entropy */
#define LZO_ADAPTIVE_LOW_ENTROPY 4.0
#define LZO_ADAPTIVE_HIGH_ENTROPY 7.0
#define LZO_ADAPTIVE_ENTROPY_ENABLED 0  /* 0=size-only (fast), 1=entropy-aware */

/* OpenCL configuration */
#define LZO_LOCAL_SIZE_DEFAULT 1  /* Compression kernels require local_size=1 */

/* Memory alignment */
#ifndef ALIGN_BYTES
#define ALIGN_BYTES LZO_ALIGN_BYTES_DEFAULT  /* Page-aligned for better DMA performance */
#endif

/* Minimum block size */
#ifndef MIN_BLOCK_SIZE
#define MIN_BLOCK_SIZE LZO_MIN_BLOCK_BYTES_DEFAULT
#endif

/* Clamp value to range [min, max] */
static inline int lzo_clamp_int(int val, int min, int max) {
	return (val < min) ? min : (val > max) ? max : val;
}

/* Convenience helper to parse int env vars with a default value */
static inline int lzo_env_get_int(const char* name, int default_val) {
	const char* s = getenv(name);
	if (!s || *s == '\0') return default_val;
	return atoi(s);
}

/* Note: profile writes are gated by per-run / per-request debug flag (use --debug). */

/* Adaptive blocking: Enable entropy-based block size calculation (default: disabled for speed) */
#define LZO_ADAPTIVE_ENTROPY_ENABLED 0  /* 0=size-only (fast), 1=entropy-aware (slower but better) */

#endif /* LZO_DEFAULTS_H */
