/*
 * lzo_hybrid_core.c — Hybrid GPU+CPU LZO compression/decompression core.
 *
 * Partitions blocks between GPU (OpenCL kernel) and CPU (liblzo2 pthreads).
 * Both run concurrently. Results are merged into the standard .lzo container.
 *
 * Block format is identical to lzo_gpu: MAGIC 0x4C5A, header, per-block lengths, data.
 * GPU and CPU produce format-compatible compressed blocks.
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "lzo_hybrid_core.h"
#include "../lzo_gpu/lzo_gpu_utils.h"
#include "../lzo_gpu/lzo_defaults.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>

/* CPU LZO library */
#include <lzo/lzoconf.h>
#include <lzo/lzo1x.h>
#include <lzo/lzo1y.h>

#define MAGIC 0x4C5A
#define CHECK_CL(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

static int g_lzo_initialized = 0;

static inline uint64_t hybrid_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline size_t lzo_worst_size(size_t n) {
    return n + n / 16 + 64 + 3;
}

static void ensure_lzo_init(void) {
    if (!g_lzo_initialized) {
        if (lzo_init() != LZO_E_OK) {
            fprintf(stderr, "lzo_init() failed\n");
        }
        g_lzo_initialized = 1;
    }
}

/* ---- workspace management ---- */

void hybrid_workspace_init(hybrid_workspace_t* ws) {
    memset(ws, 0, sizeof(*ws));
    ws->comp_epoch_base = 1;
}

void hybrid_workspace_free(hybrid_workspace_t* ws) {
    if (ws->d_in) clReleaseMemObject(ws->d_in);
    if (ws->d_out) clReleaseMemObject(ws->d_out);
    if (ws->d_len) clReleaseMemObject(ws->d_len);
    if (ws->d_dict) clReleaseMemObject(ws->d_dict);
    if (ws->d_comp) clReleaseMemObject(ws->d_comp);
    if (ws->d_off) clReleaseMemObject(ws->d_off);
    if (ws->d_decomp_out) clReleaseMemObject(ws->d_decomp_out);
    if (ws->d_out_lens) clReleaseMemObject(ws->d_out_lens);
    memset(ws, 0, sizeof(*ws));
}

static cl_mem grow_buffer(cl_context ctx, cl_mem* buf, size_t* cap,
                          size_t needed, cl_mem_flags flags, cl_int* err) {
    if (*buf && *cap >= needed) {
        *err = CL_SUCCESS;
        return *buf;
    }
    if (*buf) clReleaseMemObject(*buf);
    *buf = clCreateBuffer(ctx, flags, needed, NULL, err);
    *cap = (*err == CL_SUCCESS && *buf) ? needed : 0;
    return *buf;
}

static int zero_buffer(cl_command_queue q, cl_mem buf, size_t sz) {
    cl_uchar zero = 0;
    cl_int err = clEnqueueFillBuffer(q, buf, &zero, 1, 0, sz, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    clFinish(q);
    return 0;
}

/* ---- CPU worker thread data ---- */

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    uint32_t* lengths;
    size_t block_size;
    size_t in_size;
    size_t nblk;
    size_t worst_blk;
    const size_t* block_indices;
    size_t num_assigned;
    int alg_id;
    int rc;
} cpu_compress_job_t;

typedef struct {
    /* shared across all workers */
    cpu_compress_job_t* job;
    _Atomic size_t next_idx;
} cpu_compress_pool_t;

static void* cpu_compress_worker(void* arg) {
    cpu_compress_pool_t* pool = (cpu_compress_pool_t*)arg;
    cpu_compress_job_t* job = pool->job;

    /* Per-thread work memory for LZO */
    void* wrkmem = NULL;
    size_t wrkmem_sz = (job->alg_id == 1) ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
    if (posix_memalign(&wrkmem, 64, wrkmem_sz) != 0) {
        wrkmem = malloc(wrkmem_sz);
    }
    if (!wrkmem) {
        job->rc = -1;
        return NULL;
    }
    memset(wrkmem, 0, wrkmem_sz);

    for (;;) {
        size_t wi = atomic_fetch_add(&pool->next_idx, 1);
        if (wi >= job->num_assigned) break;

        size_t blk_idx = job->block_indices[wi];
        size_t offset = blk_idx * job->block_size;
        size_t this_blk = job->block_size;
        if (offset + this_blk > job->in_size) this_blk = job->in_size - offset;

        const unsigned char* src = job->input + offset;
        unsigned char* dst = job->output + blk_idx * job->worst_blk;
        lzo_uint dst_len = (lzo_uint)job->worst_blk;

        int rc;
        if (job->alg_id == 1) {
            rc = lzo1y_1_compress(src, (lzo_uint)this_blk, dst, &dst_len, wrkmem);
        } else {
            rc = lzo1x_1_compress(src, (lzo_uint)this_blk, dst, &dst_len, wrkmem);
        }

        if (rc != LZO_E_OK) {
            job->rc = -1;
        }
        job->lengths[blk_idx] = (uint32_t)dst_len;
    }

    free(wrkmem);
    return NULL;
}

/* ---- CPU decompression worker ---- */

typedef struct {
    const unsigned char* comp_data;
    const uint32_t* offsets;
    const uint32_t* comp_lengths;
    unsigned char* output;
    uint32_t* out_lengths;
    size_t block_size;
    size_t orig_size;
    size_t nblk;
    const size_t* block_indices;
    size_t num_assigned;
    int alg_id;
    int rc;
} cpu_decompress_job_t;

typedef struct {
    cpu_decompress_job_t* job;
    _Atomic size_t next_idx;
} cpu_decompress_pool_t;

static void* cpu_decompress_worker(void* arg) {
    cpu_decompress_pool_t* pool = (cpu_decompress_pool_t*)arg;
    cpu_decompress_job_t* job = pool->job;

    for (;;) {
        size_t wi = atomic_fetch_add(&pool->next_idx, 1);
        if (wi >= job->num_assigned) break;

        size_t blk_idx = job->block_indices[wi];
        size_t out_offset = blk_idx * job->block_size;
        size_t this_blk = job->block_size;
        if (out_offset + this_blk > job->orig_size) this_blk = job->orig_size - out_offset;

        uint32_t comp_off = job->offsets[blk_idx];
        uint32_t comp_len = job->comp_lengths[blk_idx];

        const unsigned char* src = job->comp_data + comp_off;
        unsigned char* dst = job->output + out_offset;
        lzo_uint dst_len = (lzo_uint)this_blk;

        int rc;
        if (job->alg_id == 1) {
            rc = lzo1y_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
        } else {
            rc = lzo1x_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
        }

        if (rc != LZO_E_OK || dst_len != (lzo_uint)this_blk) {
            job->rc = -1;
        }
        job->out_lengths[blk_idx] = (uint32_t)dst_len;
    }

    return NULL;
}

/* ---- block index partitioning ---- */

static void partition_blocks(size_t nblk, double gpu_ratio,
                             size_t** gpu_indices, size_t* gpu_count,
                             size_t** cpu_indices, size_t* cpu_count) {
    size_t gn = (size_t)(nblk * gpu_ratio + 0.5);
    if (gn > nblk) gn = nblk;
    size_t cn = nblk - gn;

    *gpu_count = gn;
    *cpu_count = cn;

    /* GPU gets the first gn blocks, CPU gets the rest.
     * GPU processes all its blocks in one kernel dispatch. Contiguous is efficient. */
    *gpu_indices = (size_t*)malloc(gn * sizeof(size_t));
    *cpu_indices = (size_t*)malloc(cn * sizeof(size_t));

    for (size_t i = 0; i < gn; i++) (*gpu_indices)[i] = i;
    for (size_t i = 0; i < cn; i++) (*cpu_indices)[i] = gn + i;
}

/* ---- internal buffer-based compress (no file I/O) ---- */

static int hybrid_compress_buf(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel gpu_kernel,
    const unsigned char* input_buf, size_t in_sz,
    unsigned char* out_buf, uint32_t* lengths,
    size_t blk, size_t nblk, size_t worst_blk,
    const hybrid_params_t* params, hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    cl_int err;
    hybrid_timing_t timing = {0};
    uint64_t t_start = hybrid_now_ns();

    timing.in_size = in_sz;
    timing.blk_size = blk;
    timing.nblk = nblk;

    /* Partition blocks */
    size_t* gpu_idx = NULL;
    size_t* cpu_idx = NULL;
    size_t gpu_count = 0, cpu_count = 0;
    double gpu_ratio = params->gpu_ratio;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    partition_blocks(nblk, gpu_ratio, &gpu_idx, &gpu_count, &cpu_idx, &cpu_count);
    timing.gpu_blocks = gpu_count;
    timing.cpu_blocks = cpu_count;

    /* Launch CPU workers */
    cpu_compress_job_t cpu_job = {
        .input = input_buf,
        .output = out_buf,
        .lengths = lengths,
        .block_size = blk,
        .in_size = in_sz,
        .nblk = nblk,
        .worst_blk = worst_blk,
        .block_indices = cpu_idx,
        .num_assigned = cpu_count,
        .alg_id = params->alg_id,
        .rc = 0,
    };
    cpu_compress_pool_t cpu_pool = {
        .job = &cpu_job,
        .next_idx = 0,
    };

    int n_cpu_threads = (cpu_count > 0) ? params->cpu_threads : 0;
    if (n_cpu_threads > (int)cpu_count) n_cpu_threads = (int)cpu_count;
    pthread_t* cpu_tids = NULL;
    uint64_t t_cpu_start = hybrid_now_ns();
    if (n_cpu_threads > 0) {
        cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
        for (int i = 0; i < n_cpu_threads; i++)
            pthread_create(&cpu_tids[i], NULL, cpu_compress_worker, &cpu_pool);
    }

    /* GPU compression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;

    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();

        /* Upload input — only the portion covering GPU blocks (contiguous at start) */
        size_t gpu_input_sz = gpu_count * blk;
        if (gpu_input_sz > in_sz) gpu_input_sz = in_sz;
        grow_buffer(ctx, &ws->d_in, &ws->in_cap, gpu_input_sz, CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) goto fail;
        err = clEnqueueWriteBuffer(queue, ws->d_in, CL_FALSE, 0, gpu_input_sz, input_buf, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;

        size_t out_needed = gpu_count * worst_blk;
        grow_buffer(ctx, &ws->d_out, &ws->out_cap, out_needed, CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) goto fail;

        size_t len_needed = gpu_count * sizeof(cl_uint);
        grow_buffer(ctx, &ws->d_len, &ws->len_cap, len_needed, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) goto fail;

        /* Dictionary pool */
        size_t dict_per_block = (1ULL << params->comp_level) * sizeof(uint32_t);
        size_t pool_size = gpu_count;
        size_t total_dict = pool_size * dict_per_block;
        size_t prev_dict = ws->dict_cap;
        grow_buffer(ctx, &ws->d_dict, &ws->dict_cap, total_dict, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) goto fail;
        if (ws->dict_cap != prev_dict) {
            zero_buffer(queue, ws->d_dict, ws->dict_cap);
        }

        /* Epoch management */
        if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
        if ((uint32_t)gpu_count + 2U >= 4095U ||
            ws->comp_epoch_base + (uint32_t)gpu_count + 1U > 4095U) {
            zero_buffer(queue, ws->d_dict, ws->dict_cap);
            ws->comp_epoch_base = 1;
        }
        uint32_t epoch_base = ws->comp_epoch_base;
        ws->comp_epoch_base += (uint32_t)gpu_count + 1U;

        /* Set kernel args */
        cl_uint gpu_in_sz = (cl_uint)gpu_input_sz;
        cl_uint blk_cl = (cl_uint)blk;
        cl_uint worst_blk_cl = (cl_uint)worst_blk;
        cl_uint pool_size_cl = (cl_uint)pool_size;
        CHECK_CL(clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &ws->d_in));
        CHECK_CL(clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &ws->d_out));
        CHECK_CL(clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &ws->d_len));
        CHECK_CL(clSetKernelArg(gpu_kernel, 3, sizeof(cl_uint), &gpu_in_sz));
        CHECK_CL(clSetKernelArg(gpu_kernel, 4, sizeof(cl_uint), &blk_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 5, sizeof(cl_uint), &worst_blk_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 6, sizeof(cl_mem), &ws->d_dict));
        CHECK_CL(clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &pool_size_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &epoch_base));

        /* Handle optional debug args */
        cl_uint krn_num_args = 0;
        clGetKernelInfo(gpu_kernel, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL);
        if (krn_num_args >= 11U) {
            cl_uint dbg_flag = 0U;
            CHECK_CL(clSetKernelArg(gpu_kernel, 9, sizeof(cl_mem), &ws->d_len));
            CHECK_CL(clSetKernelArg(gpu_kernel, 10, sizeof(cl_uint), &dbg_flag));
        }

        clFinish(queue);
        uint64_t t_up1 = hybrid_now_ns();
        timing.upload_us = (unsigned long)((t_up1 - t_up0) / 1000);

        size_t local_size = (params->local_size > 0) ? (size_t)params->local_size : 1;
        size_t global_size = gpu_count;
        if (local_size > global_size) local_size = 1;
        global_size = ((global_size + local_size - 1) / local_size) * local_size;
        if (global_size == 0) global_size = 1;

        t_gpu_k0 = hybrid_now_ns();
        err = clEnqueueNDRangeKernel(queue, gpu_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        clFinish(queue);
        t_gpu_k1 = hybrid_now_ns();

        /* Read GPU lengths */
        uint64_t t_dl0 = hybrid_now_ns();
        void* mapped_len = clEnqueueMapBuffer(queue, ws->d_len, CL_TRUE, CL_MAP_READ,
                                               0, gpu_count * sizeof(cl_uint), 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) goto fail;
        memcpy(lengths, mapped_len, gpu_count * sizeof(cl_uint));
        clEnqueueUnmapMemObject(queue, ws->d_len, mapped_len, 0, NULL, NULL);

        /* Read GPU compressed output — only copy actual compressed data per block */
        void* mapped_out = clEnqueueMapBuffer(queue, ws->d_out, CL_TRUE, CL_MAP_READ,
                                               0, gpu_count * worst_blk, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) goto fail;
        for (size_t i = 0; i < gpu_count; i++) {
            memcpy(out_buf + i * worst_blk,
                   (unsigned char*)mapped_out + i * worst_blk,
                   lengths[i]);
        }
        clEnqueueUnmapMemObject(queue, ws->d_out, mapped_out, 0, NULL, NULL);
        clFinish(queue);
        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU workers */
    uint64_t t_cpu_end = t_cpu_start;
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++)
            pthread_join(cpu_tids[i], NULL);
        t_cpu_end = hybrid_now_ns();
        free(cpu_tids);
        cpu_tids = NULL;
    }
    timing.cpu_kernel_us = (unsigned long)((t_cpu_end - t_cpu_start) / 1000);
    timing.gpu_kernel_us = (t_gpu_k1 > t_gpu_k0) ? (unsigned long)((t_gpu_k1 - t_gpu_k0) / 1000) : 0;

    if (cpu_job.rc != 0) goto fail;

    /* Compute total compressed size */
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) comp_total += lengths[i];
    timing.out_size = comp_total;

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    free(gpu_idx); free(cpu_idx);
    return 0;

fail:
    free(gpu_idx); free(cpu_idx);
    if (cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        free(cpu_tids);
    }
    return -1;
}

/* ---- internal buffer-based decompress (no file I/O) ---- */

static int hybrid_decompress_buf(
    cl_context ctx, cl_command_queue queue,
    cl_kernel gpu_kernel,
    const unsigned char* packed, const uint32_t* offsets, const uint32_t* comp_lengths,
    size_t packed_sz, uint32_t orig_sz, uint32_t blk_sz, uint32_t nblk, int alg_id,
    unsigned char* output_buf, uint32_t* out_lengths,
    const hybrid_params_t* params, hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    cl_int err;
    hybrid_timing_t timing = {0};
    uint64_t t_start = hybrid_now_ns();

    timing.in_size = orig_sz;
    timing.blk_size = blk_sz;
    timing.nblk = nblk;

    /* Partition */
    size_t* gpu_idx = NULL;
    size_t* cpu_idx = NULL;
    size_t gpu_count = 0, cpu_count = 0;
    double gpu_ratio = params->gpu_ratio;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    partition_blocks(nblk, gpu_ratio, &gpu_idx, &gpu_count, &cpu_idx, &cpu_count);
    timing.gpu_blocks = gpu_count;
    timing.cpu_blocks = cpu_count;

    memset(out_lengths, 0, nblk * sizeof(uint32_t));

    /* Launch CPU decompression */
    cpu_decompress_job_t cpu_job = {
        .comp_data = packed,
        .offsets = offsets,
        .comp_lengths = comp_lengths,
        .output = output_buf,
        .out_lengths = out_lengths,
        .block_size = blk_sz,
        .orig_size = orig_sz,
        .nblk = nblk,
        .block_indices = cpu_idx,
        .num_assigned = cpu_count,
        .alg_id = alg_id,
        .rc = 0,
    };
    cpu_decompress_pool_t cpu_pool = {
        .job = &cpu_job,
        .next_idx = 0,
    };

    int n_cpu_threads = (cpu_count > 0) ? params->cpu_threads : 0;
    if (n_cpu_threads > (int)cpu_count) n_cpu_threads = (int)cpu_count;
    pthread_t* cpu_tids = NULL;
    uint64_t t_cpu_start = hybrid_now_ns();
    if (n_cpu_threads > 0) {
        cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
        for (int i = 0; i < n_cpu_threads; i++)
            pthread_create(&cpu_tids[i], NULL, cpu_decompress_worker, &cpu_pool);
    }

    /* GPU decompression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;
    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();
        grow_buffer(ctx, &ws->d_comp, &ws->comp_cap, packed_sz, CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) goto dfail;
        grow_buffer(ctx, &ws->d_off, &ws->off_cap, (nblk + 1) * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) goto dfail;
        grow_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_cap, orig_sz, CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) goto dfail;
        grow_buffer(ctx, &ws->d_out_lens, &ws->out_lens_cap, nblk * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) goto dfail;

        err = clEnqueueWriteBuffer(queue, ws->d_comp, CL_FALSE, 0, packed_sz, packed, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, ws->d_off, CL_FALSE, 0, (nblk + 1) * sizeof(cl_uint), offsets, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto dfail;
        clFinish(queue);
        uint64_t t_up1 = hybrid_now_ns();
        timing.upload_us = (unsigned long)((t_up1 - t_up0) / 1000);

        cl_uint blk_sz_cl = blk_sz;
        cl_uint orig_sz_cl = orig_sz;
        cl_uint nblk_cl = nblk;
        CHECK_CL(clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &ws->d_comp));
        CHECK_CL(clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &ws->d_off));
        CHECK_CL(clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &ws->d_decomp_out));
        CHECK_CL(clSetKernelArg(gpu_kernel, 3, sizeof(cl_mem), &ws->d_out_lens));
        CHECK_CL(clSetKernelArg(gpu_kernel, 4, sizeof(cl_uint), &blk_sz_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 5, sizeof(cl_uint), &orig_sz_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 6, sizeof(cl_uint), &nblk_cl));

        cl_uint krn_num_args = 0;
        clGetKernelInfo(gpu_kernel, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL);
        if (krn_num_args >= 9U) {
            cl_uint dbg_flag = 0U;
            CHECK_CL(clSetKernelArg(gpu_kernel, 7, sizeof(cl_mem), &ws->d_out_lens));
            CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &dbg_flag));
        }

        size_t local_size = (params->local_size > 0) ? (size_t)params->local_size : 1;
        /* Only dispatch GPU for the gpu_count blocks (contiguous at start) */
        size_t global_size = gpu_count;
        if (local_size > global_size) local_size = 1;
        global_size = ((global_size + local_size - 1) / local_size) * local_size;

        t_gpu_k0 = hybrid_now_ns();
        err = clEnqueueNDRangeKernel(queue, gpu_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto dfail;
        clFinish(queue);
        t_gpu_k1 = hybrid_now_ns();

        /* Read GPU output for GPU-assigned blocks only */
        uint64_t t_dl0 = hybrid_now_ns();
        /* Map only the portion covering GPU blocks instead of full orig_sz */
        size_t gpu_out_sz = gpu_count * blk_sz;
        if (gpu_out_sz > orig_sz) gpu_out_sz = orig_sz;
        void* mapped = clEnqueueMapBuffer(queue, ws->d_decomp_out, CL_TRUE, CL_MAP_READ,
                                            0, gpu_out_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) goto dfail;
        for (size_t i = 0; i < gpu_count; i++) {
            size_t bi = gpu_idx[i];
            size_t off = bi * blk_sz;
            size_t this_blk = blk_sz;
            if (off + this_blk > orig_sz) this_blk = orig_sz - off;
            memcpy(output_buf + off, (unsigned char*)mapped + off, this_blk);
        }
        clEnqueueUnmapMemObject(queue, ws->d_decomp_out, mapped, 0, NULL, NULL);
        clFinish(queue);
        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU */
    uint64_t t_cpu_end = t_cpu_start;
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        t_cpu_end = hybrid_now_ns();
        free(cpu_tids);
        cpu_tids = NULL;
    }
    timing.gpu_kernel_us = (t_gpu_k1 > t_gpu_k0) ? (unsigned long)((t_gpu_k1 - t_gpu_k0) / 1000) : 0;
    timing.cpu_kernel_us = (unsigned long)((t_cpu_end - t_cpu_start) / 1000);

    if (cpu_job.rc != 0) goto dfail;

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    free(gpu_idx); free(cpu_idx);
    return 0;

dfail:
    if (cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        free(cpu_tids);
    }
    free(gpu_idx); free(cpu_idx);
    return -1;
}

/* ---- public hybrid compress (file-based) ---- */

int hybrid_compress(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel gpu_kernel,
    const char* input_path,
    const char* output_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    ensure_lzo_init();
    uint64_t t_start = hybrid_now_ns();

    struct stat st;
    if (stat(input_path, &st) != 0 || st.st_size <= 0) return -1;
    size_t in_sz = (size_t)st.st_size;

    /* Read input */
    uint64_t t_read0 = hybrid_now_ns();
    unsigned char* input_buf = NULL;
    if (posix_memalign((void**)&input_buf, 4096, in_sz) != 0) {
        input_buf = (unsigned char*)malloc(in_sz);
        if (!input_buf) return -1;
    }
    unsigned long read_us = 0;
    if (lzo_read_file_to_buf(input_path, input_buf, in_sz, &read_us) != 0) {
        free(input_buf);
        return -1;
    }
    uint64_t t_read1 = hybrid_now_ns();

    /* Block calculation */
    size_t blk = 0, nblk = 0;
    size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
    lzo_choose_blocking_adaptive(input_buf, in_sz, device, blk_bytes, 0, &blk, &nblk, params->debug);
    size_t worst_blk = lzo_worst_size(blk);

    /* Allocate output arrays */
    uint32_t* lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    unsigned char* out_buf = (unsigned char*)malloc(nblk * worst_blk);
    if (!lengths || !out_buf) {
        free(input_buf); free(lengths); free(out_buf);
        return -1;
    }

    /* Compress using buffer-based function */
    hybrid_timing_t timing = {0};
    int rc = hybrid_compress_buf(ctx, queue, device, gpu_kernel,
                                  input_buf, in_sz, out_buf, lengths,
                                  blk, nblk, worst_blk, params, ws, &timing);
    if (rc != 0) {
        free(input_buf); free(lengths); free(out_buf);
        return -1;
    }

    timing.file_read_us = (unsigned long)((t_read1 - t_read0) / 1000);

    /* Write output file */
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        uint64_t t_w0 = hybrid_now_ns();
        int wr = lzo_write_compressed_file(output_path, in_sz, blk, nblk,
                                           (unsigned int*)lengths, out_buf,
                                           worst_blk, params->alg_id, params->debug);
        uint64_t t_w1 = hybrid_now_ns();
        timing.file_write_us = (unsigned long)((t_w1 - t_w0) / 1000);
        if (wr != 0) { free(input_buf); free(lengths); free(out_buf); return -1; }
    }

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    free(input_buf); free(lengths); free(out_buf);
    return 0;
}

/* ---- public hybrid decompress (file-based) ---- */

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
) {
    ensure_lzo_init();
    uint64_t t_start = hybrid_now_ns();
    hybrid_timing_t timing = {0};

    /* Read .lzo header */
    FILE* f = fopen(input_path, "rb");
    if (!f) return -1;

    uint16_t magic;
    fread(&magic, 2, 1, f);
    if (magic != MAGIC) { fclose(f); return -1; }

    uint32_t orig_sz, blk_sz, nblk, alg_id;
    fread(&orig_sz, 4, 1, f);
    fread(&blk_sz, 4, 1, f);
    fread(&nblk, 4, 1, f);
    fread(&alg_id, 4, 1, f);

    uint32_t* comp_lengths = (uint32_t*)malloc(nblk * sizeof(uint32_t));
    if (!comp_lengths) { fclose(f); return -1; }
    fread(comp_lengths, sizeof(uint32_t), nblk, f);

    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    size_t comp_data_sz = (size_t)(ftell(f) - data_start);
    fseek(f, data_start, SEEK_SET);

    unsigned char* comp_data = (unsigned char*)malloc(comp_data_sz);
    if (!comp_data) { free(comp_lengths); fclose(f); return -1; }
    fread(comp_data, 1, comp_data_sz, f);
    fclose(f);

    /* Build per-block offsets into packed compressed data */
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    offsets[0] = 0;
    for (uint32_t i = 0; i < nblk; i++) offsets[i + 1] = offsets[i] + comp_lengths[i];

    unsigned char* output_buf = (unsigned char*)malloc(orig_sz);
    uint32_t* out_lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    if (!output_buf || !out_lengths) goto dec_fail;

    /* Decompress using buffer-based function */
    int rc = hybrid_decompress_buf(ctx, queue, gpu_kernel,
                                    comp_data, offsets, comp_lengths,
                                    comp_data_sz, orig_sz, blk_sz, nblk, (int)alg_id,
                                    output_buf, out_lengths,
                                    params, ws, &timing);
    if (rc != 0) goto dec_fail;

    /* Write output */
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        uint64_t t_w0 = hybrid_now_ns();
        FILE* fout = fopen(output_path, "wb");
        if (!fout) goto dec_fail;
        fwrite(output_buf, 1, orig_sz, fout);
        fclose(fout);
        uint64_t t_w1 = hybrid_now_ns();
        timing.file_write_us = (unsigned long)((t_w1 - t_w0) / 1000);
    }

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    timing.out_size = comp_data_sz;
    if (timing_out) *timing_out = timing;

    free(comp_data); free(comp_lengths); free(offsets);
    free(output_buf); free(out_lengths);
    return 0;

dec_fail:
    free(comp_data); free(comp_lengths); free(offsets);
    free(output_buf); free(out_lengths);
    return -1;
}

/* ---- Bench mode (in-memory, no file I/O in hot loop) ---- */

static int cmp_double_asc(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static double median_double(const double* vals, size_t n) {
    if (!vals || n == 0) return 0.0;
    double* tmp = (double*)malloc(n * sizeof(double));
    memcpy(tmp, vals, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_asc);
    double out = (n % 2 == 0) ? (tmp[n/2 - 1] + tmp[n/2]) / 2.0 : tmp[n/2];
    free(tmp);
    return out;
}

static double elapsed_sec(const struct timespec* s, const struct timespec* e) {
    return (double)(e->tv_sec - s->tv_sec) + (double)(e->tv_nsec - s->tv_nsec) / 1e9;
}

int hybrid_bench(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel comp_kernel,
    cl_kernel dec_kernel,
    const char* input_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    double bench_seconds
) {
    ensure_lzo_init();
    if (bench_seconds <= 0.0) bench_seconds = 3.0;

    struct stat st;
    if (stat(input_path, &st) != 0 || st.st_size <= 0) return 1;
    size_t in_size = (size_t)st.st_size;

    /* Read input once */
    size_t ref_sz = 0;
    unsigned char* input_ref = (unsigned char*)lzo_read_file(input_path, &ref_sz);
    if (!input_ref || ref_sz != in_size) { free(input_ref); return 1; }

    /* Block calculation (done once) */
    size_t blk = 0, nblk = 0;
    size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
    lzo_choose_blocking_adaptive(input_ref, in_size, device, blk_bytes, 0, &blk, &nblk, params->debug);
    size_t worst_blk = lzo_worst_size(blk);

    /* Pre-allocate reusable buffers */
    uint32_t* lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    unsigned char* out_buf = (unsigned char*)malloc(nblk * worst_blk);
    unsigned char* packed = NULL;
    size_t packed_cap = 0;
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    unsigned char* dec_buf = (unsigned char*)malloc(in_size);
    uint32_t* out_lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    if (!lengths || !out_buf || !offsets || !dec_buf || !out_lengths) {
        free(lengths); free(out_buf); free(packed); free(offsets);
        free(dec_buf); free(out_lengths); free(input_ref);
        return 1;
    }

    size_t cap = 16, n = 0;
    double* comp_ktp = (double*)malloc(cap * sizeof(double));
    double* dec_ktp = (double*)malloc(cap * sizeof(double));
    double* comp_ttp = (double*)malloc(cap * sizeof(double));
    double* dec_ttp = (double*)malloc(cap * sizeof(double));
    double* ratio_pct = (double*)malloc(cap * sizeof(double));
    int verify_ok = 1;

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    while (verify_ok) {
        /* ---- COMPRESS (in-memory) ---- */
        memset(lengths, 0, nblk * sizeof(uint32_t));
        hybrid_timing_t tc = {0};
        int rc = hybrid_compress_buf(ctx, queue, device, comp_kernel,
                                      input_ref, in_size, out_buf, lengths,
                                      blk, nblk, worst_blk, params, ws, &tc);
        if (rc != 0) { verify_ok = 0; break; }

        size_t comp_total = tc.out_size;
        if (comp_total == 0 || nblk == 0) { verify_ok = 0; break; }

        /* Kernel throughput = max(gpu_kernel, cpu_kernel) — they run in parallel */
        unsigned long comp_kernel_us = tc.gpu_kernel_us;
        if (tc.cpu_kernel_us > comp_kernel_us) comp_kernel_us = tc.cpu_kernel_us;

        /* Pack compressed data (strip worst_blk padding) */
        if (comp_total > packed_cap) {
            free(packed);
            packed = (unsigned char*)malloc(comp_total);
            packed_cap = packed ? comp_total : 0;
            if (!packed) { verify_ok = 0; break; }
        }
        offsets[0] = 0;
        size_t co = 0;
        for (size_t i = 0; i < nblk; i++) {
            size_t csz = (size_t)lengths[i];
            memcpy(packed + co, out_buf + i * worst_blk, csz);
            co += csz;
            offsets[i + 1] = (uint32_t)co;
        }

        /* ---- DECOMPRESS (in-memory) ---- */
        uint64_t dec_total_start = hybrid_now_ns();
        hybrid_timing_t td = {0};
        rc = hybrid_decompress_buf(ctx, queue, dec_kernel,
                                    packed, offsets, lengths,
                                    comp_total, (uint32_t)in_size, (uint32_t)blk, (uint32_t)nblk,
                                    params->alg_id,
                                    dec_buf, out_lengths,
                                    params, ws, &td);
        uint64_t dec_total_end = hybrid_now_ns();

        if (rc != 0) { verify_ok = 0; break; }

        /* Verify */
        if (memcmp(dec_buf, input_ref, in_size) != 0) {
            verify_ok = 0;
            break;
        }

        unsigned long dec_kernel_us = td.gpu_kernel_us;
        if (td.cpu_kernel_us > dec_kernel_us) dec_kernel_us = td.cpu_kernel_us;
        double dec_total_us = (double)(dec_total_end - dec_total_start) / 1000.0;

        /* Record */
        if (n == cap) {
            cap *= 2;
            comp_ktp = (double*)realloc(comp_ktp, cap * sizeof(double));
            dec_ktp = (double*)realloc(dec_ktp, cap * sizeof(double));
            comp_ttp = (double*)realloc(comp_ttp, cap * sizeof(double));
            dec_ttp = (double*)realloc(dec_ttp, cap * sizeof(double));
            ratio_pct = (double*)realloc(ratio_pct, cap * sizeof(double));
        }

        double in_mb = (double)in_size / (1024.0 * 1024.0);
        comp_ktp[n] = (comp_kernel_us > 0) ? (in_mb * 1e6 / (double)comp_kernel_us) : 0.0;
        dec_ktp[n] = (dec_kernel_us > 0) ? (in_mb * 1e6 / (double)dec_kernel_us) : 0.0;
        comp_ttp[n] = (tc.total_us > 0) ? (in_mb * 1e6 / (double)tc.total_us) : 0.0;
        dec_ttp[n] = (dec_total_us > 0.0) ? (in_mb * 1e6 / dec_total_us) : 0.0;
        ratio_pct[n] = (in_size > 0) ? (100.0 * (double)comp_total / (double)in_size) : 0.0;
        n++;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double sec = elapsed_sec(&ts0, &ts1);

    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n",
               median_double(comp_ktp, n), median_double(comp_ttp, n), median_double(ratio_pct, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n",
               median_double(dec_ktp, n), median_double(dec_ttp, n), verify_ok ? "OK" : "FAIL");
        printf("Bench Summary : iterations=%zu seconds=%.2f\n", n, sec);
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_ktp); free(dec_ktp); free(comp_ttp); free(dec_ttp); free(ratio_pct);
    free(input_ref); free(lengths); free(out_buf);
    free(packed); free(offsets); free(dec_buf); free(out_lengths);
    return verify_ok ? 0 : 1;
}
