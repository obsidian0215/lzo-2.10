#ifndef LZO_GPU_UTILS_H
#define LZO_GPU_UTILS_H

#include <stddef.h>
#include <CL/cl.h>
#include "lzo_defaults.h"
#include "lzo_gpu_protocol.h"

/* 通用工具宏 */
#ifndef CLAMP
#define CLAMP(val, min_val, max_val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>
#include <stdint.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <malloc.h>
static inline uint64_t lzo_now_ns(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)counter.QuadPart * 1000000000ULL / (uint64_t)freq.QuadPart;
}

static inline int lzo_aligned_alloc_portable(void** ptr, size_t alignment, size_t size) {
    void* p = _aligned_malloc(size, alignment);
    if (!p) {
        *ptr = NULL;
        return -1;
    }
    *ptr = p;
    return 0;
}

static inline void lzo_aligned_free_portable(void* ptr) {
    _aligned_free(ptr);
}
#else
static inline uint64_t lzo_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline int lzo_aligned_alloc_portable(void** ptr, size_t alignment, size_t size) {
    return posix_memalign(ptr, alignment, size);
}

static inline void lzo_aligned_free_portable(void* ptr) {
    free(ptr);
}
#endif

/* Locate a file by name with priority: exe_dir, exe_dir/../lzo_gpu, LZO_GPU_DIR, OUT_DIR, cwd, raw
 * Returns 0 on success and writes resolved path to out. Returns -1 on failure.
 */
int lzo_find_file_path(const char *name, char *out, size_t outlen);

/* Read a file into a malloc'd buffer. If path is "-", read from stdin.
 * On success, returns pointer (caller must free) and sets *sz_out. On failure returns NULL.
 */
char* lzo_read_file(const char *path, size_t *sz_out);

/* Adaptive block size calculation and entropy helpers */
double lzo_calc_entropy(const unsigned char* data, size_t size);
size_t lzo_parse_block_size(const char* str);
/* Adaptive block size calculation functions */
size_t lzo_adaptive_block_size(size_t in_sz, cl_uint cu);
size_t lzo_adaptive_block_size_with_entropy(const unsigned char* data, size_t in_sz, cl_uint cu, double* entropy_out, int debug);
size_t lzo_adaptive_block_size_from_entropy(size_t in_sz, double entropy, cl_uint cu, int debug);
int lzo_specified_unit_is_bytes(const char* s);
/* data may be NULL if caller cannot provide a memory pointer — in that case
 * the function will fallback to count-based strategy using device CU only.
 * blk_bytes forces a specific block size if > 0. If the caller sets
 * fixed_exact==1 then the requested value is used as-is (no alignment/minimum).
 */
void lzo_choose_blocking_adaptive(const unsigned char* data, size_t in_sz, cl_device_id dev, size_t blk_bytes, int fixed_exact, size_t* blk_sz_out, size_t* nblk_out, int debug);

/* Load an OpenCL program from a precompiled binary (<prog>_<bits>.clbin) or compile from source (<prog>.cl) with -D D_BITS=<bits>.
 * On success returns a built cl_program; on failure returns NULL and, if build_log is provided, writes build output into it.
 */
cl_program lzo_load_program_with_dbits(cl_context ctx, cl_device_id dev, const char *alg_name, int bits, char *build_log, size_t build_log_len);

/* Load and create a compression kernel for the given algorithm and D_BITS.
 * Compiles/loads the base kernel (<alg>.cl / <alg>_<bits>.clbin). LZO_USE_UNROLL2 is applied by default.
 * On success returns 0 and fills out_prog/out_krn and sets *kernel_has_dbg (1 if kernel takes a debug buffer arg; usually 0 in production builds).
 * On failure returns non-zero and, if provided, writes a short message into build_log.
 */
int lzo_load_comp_kernel(cl_context ctx, cl_device_id dev, const char *alg_name, int comp_level, int debug, cl_program *out_prog, cl_kernel *out_krn, int *kernel_has_dbg, char *build_log, size_t build_log_len);

/* Debug parsing and instrumentation helpers removed from production build. */

/* Daemon paths */
void lzo_set_daemon_socket_path(const char* path);
void lzo_set_daemon_pidfile_path(const char* path);
const char* lzo_daemon_socket_path(void);
const char* lzo_daemon_pidfile_path(void);

/* Optimized IO helpers */
int lzo_read_file_to_buf(const char* path, void* dest, size_t size, unsigned long* read_us_out);

int lzo_write_compressed_file(const char* path,
                              size_t orig_size, size_t blk_size,
                              size_t nblk, const unsigned int* lens,
                              const void* sparse_data, size_t worst_blk,
                              int alg_id, int debug);

void lzo_print_response_stats(const response_t* resp, const char* input_path, int operation, int alg);

#ifdef __cplusplus
}
#endif

#endif /* LZO_GPU_UTILS_H */
