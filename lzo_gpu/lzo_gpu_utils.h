#ifndef LZO_GPU_UTILS_H
#define LZO_GPU_UTILS_H

#include <stddef.h>
#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Locate a file by name with priority: exe_dir, exe_dir/../lzo_gpu, LZO_GPU_DIR, OUT_DIR, cwd, raw
 * Returns 0 on success and writes resolved path to out. Returns -1 on failure.
 */
int lzo_find_file_path(const char *name, char *out, size_t outlen);

/* Read a file into a malloc'd buffer. If path is "-", read from stdin.
 * On success, returns pointer (caller must free) and sets *sz_out. On failure returns NULL.
 */
char* lzo_read_file(const char *path, size_t *sz_out);

/* Choose block size and number of blocks for GPU kernels (adapts to CU count)
 * Based on logic used across lzo_host and daemon.
 */
/* Adaptive/dynamic blocking helpers */
/* Entropy thresholds and alignment defaults (can be overridden by env vars) */
#define LZO_OCC_FACTOR_DEFAULT 128
#define LZO_ALIGN_BYTES_DEFAULT 16384
#define LZO_MIN_BLOCK_BYTES_DEFAULT (4 * 1024)
#define LZO_MAX_BLOCK_BYTES_DEFAULT (256 * 1024)
#define LZO_ADAPTIVE_SAMPLE_SIZE (64 * 1024)
#define LZO_ADAPTIVE_LOW_ENTROPY 4.0
#define LZO_ADAPTIVE_HIGH_ENTROPY 7.0
#define MIN_BLOCK_SIZE LZO_MIN_BLOCK_BYTES_DEFAULT
#define MAX_BLOCK_SIZE LZO_MAX_BLOCK_BYTES_DEFAULT
#define ALIGN_BYTES LZO_ALIGN_BYTES_DEFAULT
#define OCC_FACTOR LZO_OCC_FACTOR_DEFAULT
#define SAMPLE_SIZE LZO_ADAPTIVE_SAMPLE_SIZE
#define LOW_ENTROPY_THRESHOLD LZO_ADAPTIVE_LOW_ENTROPY
#define HIGH_ENTROPY_THRESHOLD LZO_ADAPTIVE_HIGH_ENTROPY
#define LZO_LOCAL_SIZE_DEFAULT 1

double lzo_calc_entropy(const unsigned char* data, size_t size);
size_t lzo_parse_block_size(const char* str);
size_t lzo_adaptive_block_size(const unsigned char* data, size_t in_sz, cl_uint cu, double* entropy_out, int debug);
size_t lzo_adaptive_block_size_from_entropy(size_t in_sz, double entropy, cl_uint cu, int debug);
/* data may be NULL if caller cannot provide a memory pointer — in that case
 * the function will fallback to count-based strategy using device CU only.
 * fixed_blk_bytes forces a specific block size if > 0.
 */
void lzo_choose_blocking_adaptive(const unsigned char* data, size_t in_sz, cl_device_id dev, size_t fixed_blk_bytes, size_t* blk_sz_out, size_t* nblk_out, int debug);

#ifdef __cplusplus
}
#endif

#endif /* LZO_GPU_UTILS_H */
