#ifndef LZO_HYBRID_CORE_H
#define LZO_HYBRID_CORE_H

#include <CL/cl.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- timing ---- */
typedef struct {
    unsigned long gpu_kernel_us;    /* GPU kernel execution time */
    unsigned long cpu_kernel_us;    /* CPU worker execution time */
    unsigned long total_us;         /* Wall-clock total (incl. host overhead) */
    unsigned long file_read_us;
    unsigned long file_write_us;
    unsigned long upload_us;
    unsigned long download_us;

    size_t in_size;
    size_t out_size;                /* compressed size */
    size_t blk_size;
    size_t nblk;
    size_t gpu_blocks;
    size_t cpu_blocks;
} hybrid_timing_t;

/* ---- split strategy ---- */
typedef enum {
    HYBRID_SPLIT_FIXED,      /* fixed ratio (default) */
    HYBRID_SPLIT_ADAPTIVE,   /* per-file entropy-based split */
} hybrid_split_mode_t;

/* ---- configuration ---- */
typedef struct {
    /* Algorithm */
    int alg_id;              /* 0=lzo1x, 1=lzo1y */
    int comp_level;          /* D_BITS (10-14) */
    size_t block_size;       /* 0 = adaptive */

    /* Hybrid split */
    hybrid_split_mode_t split_mode;
    double gpu_ratio;        /* GPU fraction [0.0, 1.0], default 0.8 */
    size_t adaptive_sample_blocks;

    /* CPU worker */
    int cpu_threads;         /* number of CPU worker threads */

    /* OpenCL */
    int local_size;          /* work-group size */
    int standard_copy;       /* 0=zero-copy, 1=standard */
    int debug;
} hybrid_params_t;

/* ---- workspace (reusable across bench iterations) ---- */
typedef struct {
    /* GPU compression buffers */
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    cl_mem d_packed_out;
    cl_mem d_packed_off;
    cl_mem d_dict;
    size_t in_cap;
    size_t out_cap;
    size_t len_cap;
    size_t packed_out_cap;
    size_t packed_off_cap;
    size_t dict_cap;
    uint32_t comp_epoch_base;

    /* GPU decompression buffers */
    cl_mem d_comp;
    cl_mem d_off;
    cl_mem d_comp_lens;
    cl_mem d_decomp_out;
    cl_mem d_out_lens;
    size_t comp_cap;
    size_t off_cap;
    size_t comp_lens_cap;
    size_t decomp_out_cap;
    size_t out_lens_cap;
} hybrid_workspace_t;

void hybrid_workspace_init(hybrid_workspace_t* ws);
void hybrid_workspace_free(hybrid_workspace_t* ws);

/* ---- core API ---- */

/*
 * Hybrid compress: read input file, split blocks between GPU and CPU,
 * run both concurrently, write output .lzo file.
 * Returns 0 on success.
 */
int hybrid_compress(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel gpu_kernel,
    cl_kernel pack_kernel,
    const char* input_path,
    const char* output_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
);

/*
 * Hybrid decompress: read .lzo, split blocks between GPU and CPU,
 * run both concurrently, write output file.
 * Returns 0 on success.
 */
int hybrid_decompress(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel gpu_kernel,
    const char* input_path,
    const char* output_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
);

/*
 * Hybrid bench mode: compress → extract → decompress → verify, repeat.
 * Prints "Bench Compress : ..." / "Bench Decompress : ..." lines.
 * Returns 0 on success, 1 on verify failure.
 */
int hybrid_bench(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel comp_kernel,
    cl_kernel pack_kernel,
    cl_kernel dec_kernel,
    const char* input_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    double bench_seconds,
    int include_file_io
);

#ifdef __cplusplus
}
#endif

#endif /* LZO_HYBRID_CORE_H */
