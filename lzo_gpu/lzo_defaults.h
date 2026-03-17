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

#define DEFAULT_DECOMP_CACHE_MB 256

/* Compression configuration */
/* Default compression level (bits). */
#define LZO_DEFAULT_COMP_LEVEL 14

/* Default block size (in KB) when no adaptive/training result is available */
#define LZO_DEFAULT_BLOCK_KB 64

/* Block size and GPU parallelism configuration */
#define LZO_OCC_FACTOR_DEFAULT 256
#define LZO_ALIGN_BYTES_DEFAULT 4096  /* 4KB alignment for block sizes for memory efficiency */
#define LZO_MIN_BLOCK_BYTES_DEFAULT (4 * 1024)   /* 1KB minimum */
#define LZO_MAX_BLOCK_BYTES_DEFAULT (256 * 1024)  /* 256KB maximum */
#define LZO_MAX_NBLOCKS_DEFAULT (64 * 1024)  /* Maximum blocks per job */
/* Entropy calculation configuration */
#define LZO_ADAPTIVE_SAMPLE_SIZE (64 * 1024)  /* 64KB sample for entropy */
#define LZO_ADAPTIVE_LOW_ENTROPY 4.0
#define LZO_ADAPTIVE_HIGH_ENTROPY 7.0
#define LZO_ADAPTIVE_ENTROPY_ENABLED 1  /* 0=size-only (fast), 1=entropy-aware */

/* OpenCL configuration */
/* Default work-group size. Set to 4 (empirically best balance in our param-scan) */
#define LZO_LOCAL_SIZE_DEFAULT 1

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

#endif /* LZO_DEFAULTS_H */
