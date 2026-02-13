#ifndef LZO_GPU_CORE_H
#define LZO_GPU_CORE_H

#include <CL/cl.h>
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

    /* Decompression buffers */
    cl_mem d_comp;
    cl_mem d_off;
    cl_mem d_decomp_out;
    cl_mem d_out_lens;
    size_t comp_size;
    size_t off_size;
    size_t decomp_out_size;
    size_t lens_size;
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
    int level;              /* Dictionary size in bits (10-14) */
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
