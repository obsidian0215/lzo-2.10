#ifndef LZO_GPU_CORE_H
#define LZO_GPU_CORE_H

#include <CL/cl.h>
#include <stdint.h>
#include "timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Workspace for thread-safe buffer caching */
typedef struct {
    /* Compression buffers */
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    cl_mem d_dict;
    size_t in_size;
    size_t out_size;
    size_t len_size;
    size_t dict_size;
    uint32_t comp_epoch_base;

    /* Decompression buffers */
    cl_mem d_comp;
    cl_mem d_off;
    cl_mem d_comp_lens;
    cl_mem d_decomp_out;
    cl_mem d_out_lens;
    size_t comp_size;
    size_t off_size;
    size_t comp_lens_size;
    size_t decomp_out_size;
    size_t lens_size;

    int comp_kernel_args_set;
    cl_kernel comp_cached_kernel;
    cl_mem comp_cached_d_in;
    cl_mem comp_cached_d_out;
    cl_mem comp_cached_d_len;
    cl_mem comp_cached_d_dict;
    cl_uint comp_cached_in_sz;
    cl_uint comp_cached_blk;
    cl_uint comp_cached_worst_blk;
    cl_uint comp_cached_pool_size;

    size_t comp_cached_input_size;
    size_t comp_cached_blk_size;
    size_t comp_cached_nblk;
} lzo_gpu_workspace_t;

void lzo_gpu_workspace_init(lzo_gpu_workspace_t* ws);
void lzo_gpu_workspace_free(lzo_gpu_workspace_t* ws);

/* Global verbosity control */
extern int g_verbose;
extern unsigned long g_ocl_init_us;
extern unsigned long g_kernel_load_us;

/* Parameter object to reduce lzo_compress_core argument count from 18 to 6 */
typedef struct {
    /* Algorithm and quality settings */
    int level;              /* Dictionary size in bits (11-20) */
    int alg_id;             /* Algorithm ID (0=lzo1x, 1=lzo1y) */

    /* I/O and memory settings */
    int standard_copy;      /* 0=zero-copy (map), 1=standard copy */
    size_t block_size;      /* Fixed block size in Bytes (0=adaptive) */

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
    int skip_input_upload,
    lzo_gpu_workspace_t* ws,      /* Thread-safe workspace */
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
    lzo_gpu_workspace_t* ws,      /* Thread-safe workspace */
    int standard_copy,
    int local_size_param, int debug,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
);

#ifdef __cplusplus
}
#endif

#endif /* LZO_GPU_CORE_H */
