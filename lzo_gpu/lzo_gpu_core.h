#ifndef LZO_GPU_CORE_H
#define LZO_GPU_CORE_H

#include <CL/cl.h>
#include "timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parameter object to reduce lzo_compress_core argument count from 18 to 6 */
typedef struct {
    /* Algorithm and quality settings */
    int level;              /* Dictionary size in bits (10-14) */
    int alg_id;             /* Algorithm ID (0=lzo1x, 1=lzo1y) */

    /* I/O and memory settings */
    int standard_copy;      /* 0=zero-copy (map), 1=standard copy */
    int mt_io;              /* Enable multi-threaded I/O */
    int mt_threads;         /* Number of MT-IO threads */
    int fixed_block_kb;     /* Fixed block size in KB (0=adaptive) */

    /* Output coalescing settings */
    int coalesce_output;    /* Enable output coalescing */
    int coalesce_chunk_mb;  /* Chunk size for chunked coalesce (MB) */
    int coalesce_max_mb;    /* Max size for full coalesce (MB) */
    int stdio_buf_mb;       /* stdio buffer size (MB) */

    /* Debug and profiling */
    int local_size_param;   /* OpenCL local work-group size */
    int debug;              /* Enable debug output */
} lzo_compress_params_t;

int lzo_compress_core(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    const lzo_compress_params_t* params,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
);

/* Legacy function signature - kept for backward compatibility during transition */
int lzo_compress_core_legacy(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    int level,
    int alg_id,
    int standard_copy, int mt_io, int mt_threads, int fixed_block_kb,
    int coalesce_output, int coalesce_chunk_mb, int coalesce_max_mb, int stdio_buf_mb,
    int local_size_param, int debug,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
);

int lzo_decompress_core(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    int standard_copy,
    int local_size_param, int debug,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
);

/* Cleanup exported so daemon can call on exit */
void cleanup_compress_buffer_cache(void);
void cleanup_decompress_buffer_cache(void);

#ifdef __cplusplus
}
#endif

#endif /* LZO_GPU_CORE_H */
