/*
 * lzo_defaults.h - Default settings for lzo gpu host and daemon
 *
 * Centralize commonly used defaults (stdio buffer, coalesce behaviour)
 * so we can reduce the number of environment variables and default-enable
 * useful behavior such as coalescing and stdio buffering.
 */
#ifndef LZO_DEFAULTS_H
#define LZO_DEFAULTS_H

#define LZO_DEFAULT_STDIO_BUF_MB 4
#define LZO_DEFAULT_COALESCE_OUTPUT 1
#define LZO_DEFAULT_COALESCE_MAX_MB 256
#define LZO_DEFAULT_COALESCE_CHUNK_MB 16
/* default value for multi-threaded I/O threads when enabled */
#define LZO_DEFAULT_MT_IO_THREADS 2

/* Convenience helper to parse int env vars with a default value */
static inline int lzo_env_get_int(const char* name, int default_val) {
	const char* s = getenv(name);
	if (!s || *s == '\0') return default_val;
	return atoi(s);
}

/* Note: profile writes are gated by debug (LZO_DEBUG) and no longer by a default env var. */

#endif /* LZO_DEFAULTS_H */
