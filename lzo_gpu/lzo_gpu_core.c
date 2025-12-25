/* lzo_gpu_core.c - shared backend for compression/decompression
 * Extracted from daemon_compress.c / daemon_decompress.c so both daemon
 * and standalone can reuse same implementation.
 */

#include <CL/cl.h>
#include "timing.h"
#include "lzo_defaults.h"
#include "lzo_gpu_core.h"
#include "lzo_gpu_utils.h"
#include "lzo_gpu_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>

#define CHECK(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

#define D_BITS 11

static inline size_t core_lzo_worst(size_t sz) {
    return sz + sz / 16 + 64 + 3;
}

static inline uint64_t core_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Write compressed file - delegates to lzo_gpu_io module */
static int core_write_compressed_file(const char* path,
                                 size_t orig_size, size_t blk_size,
                                 size_t nblk, const unsigned int* lens,
                                 const void* sparse_data, size_t worst_blk,
                                 int coalesce, size_t coalesce_chunk_mb, size_t coalesce_max_mb,
                                 size_t stdio_buf_mb, int alg_id, int debug) {
    /* Delegate to unified IO module */
    return lzo_write_compressed_file(path, orig_size, blk_size, nblk, lens,
                                     sparse_data, worst_blk, coalesce,
                                     coalesce_chunk_mb, coalesce_max_mb,
                                     stdio_buf_mb, alg_id, debug);
}

/* Compression buffer cache */
static struct {
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    size_t in_size;
    size_t out_size;
    size_t len_size;
} compress_buffer_cache = {0};

static cl_mem core_get_or_create_buffer(cl_context ctx, cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags, cl_int* err_out) {
    if (*cached_size < required_size) {
        if (*cached_buf) {
            clReleaseMemObject(*cached_buf);
            *cached_buf = NULL;
            *cached_size = 0;
        }
        cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags, required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS) {
            *cached_size = required_size;
        }
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

void cleanup_compress_buffer_cache(void) {
    if (compress_buffer_cache.d_in) clReleaseMemObject(compress_buffer_cache.d_in);
    if (compress_buffer_cache.d_out) clReleaseMemObject(compress_buffer_cache.d_out);
    if (compress_buffer_cache.d_len) clReleaseMemObject(compress_buffer_cache.d_len);
    compress_buffer_cache.d_in = NULL; compress_buffer_cache.d_out = NULL; compress_buffer_cache.d_len = NULL;
    compress_buffer_cache.in_size = 0; compress_buffer_cache.out_size = 0; compress_buffer_cache.len_size = 0;
}

/* Decompression buffer cache */
static struct {
    cl_mem d_comp;
    cl_mem d_off;
    cl_mem d_out;
    cl_mem d_out_lens;
    size_t comp_size;
    size_t off_size;
    size_t out_size;
    size_t lens_size;
} decomp_buffer_cache = {0};

static cl_mem core_get_or_create_decomp_buffer(cl_context ctx, cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags, cl_int* err_out) {
    if (*cached_size < required_size) {
        if (*cached_buf) {
            clReleaseMemObject(*cached_buf);
            *cached_buf = NULL;
            *cached_size = 0;
        }
        cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags, required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS)
            *cached_size = required_size;
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

void cleanup_decompress_buffer_cache(void) {
    if (decomp_buffer_cache.d_comp) clReleaseMemObject(decomp_buffer_cache.d_comp);
    if (decomp_buffer_cache.d_off) clReleaseMemObject(decomp_buffer_cache.d_off);
    if (decomp_buffer_cache.d_out) clReleaseMemObject(decomp_buffer_cache.d_out);
    if (decomp_buffer_cache.d_out_lens) clReleaseMemObject(decomp_buffer_cache.d_out_lens);
    decomp_buffer_cache.d_comp = NULL; decomp_buffer_cache.d_off = NULL; decomp_buffer_cache.d_out = NULL; decomp_buffer_cache.d_out_lens = NULL;
    decomp_buffer_cache.comp_size = 0; decomp_buffer_cache.off_size = 0; decomp_buffer_cache.out_size = 0; decomp_buffer_cache.lens_size = 0;
}

/* Core compression implementation */
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
) {
    cl_int err;
    uint64_t t_total_start = core_now_ns();

    int use_standard_copy = params->standard_copy ? 1 : 0;

    /* 1. 获取文件大小并读取 */
    FILE* f_in = fopen(input_path, "rb");
    if (!f_in) { perror("fopen input"); return -1; }
    fseek(f_in, 0, SEEK_END);
    size_t in_sz = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    /* 2. 准备输入缓冲区 */
    uint64_t t_buf_in_start = core_now_ns();
    cl_mem d_in = core_get_or_create_buffer(ctx, &compress_buffer_cache.d_in, &compress_buffer_cache.in_size, in_sz, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create input buffer: %d\n", err);
        fclose(f_in);
        return -1;
    }

    void* mapped_in = NULL;
    if (!use_standard_copy) {
        mapped_in = clEnqueueMapBuffer(queue, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] Map input buffer failed: %d\n", err);
            fclose(f_in);
            return -1;
        }
    }
    uint64_t t_buf_in_end = core_now_ns();
    unsigned long buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000;

    unsigned long read_us = 0;
    unsigned long upload_us = 0;
    void* host_in = NULL;

    if (!use_standard_copy) {
        /* Use unified MT-IO module */
        if (lzo_read_file_with_mode(input_path, mapped_in, in_sz, params->mt_io, params->mt_threads, &read_us) != 0) {
            perror("lzo_read_file_with_mode");
            clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
            fclose(f_in);
            return -1;
        }
        fclose(f_in);
        /* upload_us will be measured after unmap in blocking section */
    } else {
        int rc_mem = posix_memalign(&host_in, ALIGN_BYTES, in_sz);
        if (rc_mem != 0 || host_in == NULL) {
            host_in = malloc(in_sz);
            if (!host_in) {
                perror("malloc host_in");
                fclose(f_in);
                return -1;
            }
        }

        /* Use unified MT-IO module */
        if (lzo_read_file_with_mode(input_path, host_in, in_sz, params->mt_io, params->mt_threads, &read_us) != 0) {
            perror("lzo_read_file_with_mode");
            free(host_in);
            fclose(f_in);
            return -1;
        }
        fclose(f_in);
        /* upload after blocking calc */
    }

    /* 3. Blocking calculation */
    uint64_t t_blocking_start = core_now_ns();
    size_t blk = 0, nblk = 0;
    const unsigned char* entropy_ptr = NULL;
    if (use_standard_copy)
        entropy_ptr = (const unsigned char*)host_in;
    else
        entropy_ptr = (const unsigned char*)mapped_in;

    size_t fixed_blk_bytes = (params->fixed_block_kb > 0) ? (size_t)params->fixed_block_kb * 1024 : 0;
    lzo_choose_blocking_adaptive(entropy_ptr, in_sz, device, fixed_blk_bytes, 0, &blk, &nblk, params->debug);
    if (params->debug) fprintf(stderr, "[CORE][DBG] lzo_choose_blocking_adaptive determined blk=%zu nblk=%zu (in_sz=%zu)\n", blk, nblk, in_sz);

    uint64_t t_blocking_end = core_now_ns();
    unsigned long blocking_us = (t_blocking_end - t_blocking_start) / 1000;

    if (!use_standard_copy) {
        uint64_t t_unmap_start = core_now_ns();
        CHECK(clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL));
        clFinish(queue); /* Ensure unmap completes */
        uint64_t t_unmap_end = core_now_ns();
        upload_us = (t_unmap_end - t_unmap_start) / 1000;
    } else {
        uint64_t t_upload_start2 = core_now_ns();
        if (params->mt_io && params->mt_threads > 1) {
            int nthreads = params->mt_threads;
            if (nthreads < 1)   nthreads = 1;
            if (nthreads > 32)  nthreads = 32;
            cl_event *evts = malloc(sizeof(cl_event) * nthreads);
            int ev_count = 0;
            size_t piece = (in_sz + nthreads - 1) / nthreads;
            for (int i = 0; i < nthreads; ++i) {
                size_t off = (size_t)i * piece;
                size_t len = (off + piece > in_sz) ? (in_sz - off) : piece;
                if (len == 0) continue;
                err = clEnqueueWriteBuffer(queue, d_in, CL_FALSE, off, len, (char*)host_in + off, 0, NULL, &evts[ev_count]);
                if (err == CL_SUCCESS) {
                    if (params->debug) fprintf(stderr, "[CORE] enqueued write evt[%d] off=%zu len=%zu\n", ev_count, off, len);
                    ev_count++;
                } else {
                    fprintf(stderr, "[CORE] clEnqueueWriteBuffer failed (segment) : %d\n", err);
                    for (int j = 0; j < ev_count; ++j)
                        if (evts[j])
                            clReleaseEvent(evts[j]);
                    free(evts);
                    free(host_in);
                    return -1;
                }
            }
            if (ev_count == 0) {
                if (params->debug)
                fprintf(stderr, "[CORE] write segments enqueued ev_count==0, skipping wait\n");
                upload_us = 0;
                free(evts);
                free(host_in);
                host_in = NULL;
                goto after_std_read_upload_done_core;
            }
            clWaitForEvents(ev_count, evts);
            for (int i = 0; i < ev_count; ++i)
                if (evts[i])
                clReleaseEvent(evts[i]);
            free(evts);
            free(host_in);
            host_in = NULL;
        } else {
            err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "[CORE] clEnqueueWriteBuffer failed: %d\n", err);
                free(host_in);
                return -1;
            }
            free(host_in);
            host_in = NULL;
        }
after_std_read_upload_done_core:
        uint64_t t_upload_end2 = core_now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
    }

    size_t worst_blk = core_lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &compress_buffer_cache.d_out, &compress_buffer_cache.out_size, out_needed, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create output buffer: %d\n", err);
        return -1;
    }
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buffer_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    uint64_t t_buf_len_start = core_now_ns();
    cl_mem d_len = core_get_or_create_buffer(ctx, &compress_buffer_cache.d_len, &compress_buffer_cache.len_size, len_needed, CL_MEM_READ_WRITE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create len buffer: %d\n", err);
        return -1;
    }
    uint64_t t_buf_len_end = core_now_ns();
    unsigned long buffer_len_us = (t_buf_len_end - t_buf_len_start) / 1000;

    cl_uint in_sz_cl = (cl_uint)in_sz, blk_cl = (cl_uint)blk, worst_blk_cl = (cl_uint)worst_blk;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));

    cl_mem d_dbg = NULL;
    int kernel_has_dbg_arg = 0;
    /* Probe kernel for number of args; if it expects a debug buffer (arg index 6), allocate and attach one.
     * When the caller requested debug instrumentation (via per-job flag) and the kernel supports the
     * debug argument, this buffer will be used to capture per-block debug metrics.
     */
    {
        cl_uint num_args = 0;
        cl_int rc = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
        if (rc == CL_SUCCESS && num_args >= 7) {
            kernel_has_dbg_arg = 1;
            /* Allocate a debug buffer: 7 uints per block */
            const size_t dbg_fields = 7;
            size_t per_blk = dbg_fields * sizeof(cl_uint);
            if (nblk > 0 && nblk <= (SIZE_MAX / per_blk)) {
                size_t dbg_bytes = nblk * per_blk;
                d_dbg = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, dbg_bytes, NULL, &err);
                if (err == CL_SUCCESS) {
                    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dbg));
                } else {
                    fprintf(stderr, "[CORE] failed to create debug buffer: %d\n", err);
                    /* Try setting a NULL arg to at least avoid missing-arg errors */
                    d_dbg = NULL;
                    cl_int perr = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dbg);
                    if (perr != CL_SUCCESS) fprintf(stderr, "[CORE] clSetKernelArg(6,NULL) failed: %d\n", perr);
                }
            } else {
                fprintf(stderr, "[CORE] dbg buffer size overflow or nblk==0; setting kernel arg 6 to NULL\n");
                d_dbg = NULL;
                cl_int perr = clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dbg);
                if (perr != CL_SUCCESS) fprintf(stderr, "[CORE] clSetKernelArg(6,NULL) failed: %d\n", perr);
            }
        }
    }

    /* Compression kernels require local_size=1 (defined as LZO_LOCAL_SIZE_DEFAULT) */
    const size_t local_size = LZO_LOCAL_SIZE_DEFAULT;
    if (params->local_size_param > 0 && params->local_size_param != local_size) {
        fprintf(stderr, "[CORE] WARNING: --local=%d ignored for compression (kernel requires local=%zu)\n",
                params->local_size_param, local_size);
    }
    size_t global_size = nblk;  /* One work-item per block */
    cl_event evt_compute;
    uint64_t t_exec_start = core_now_ns();
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &evt_compute));
    clWaitForEvents(1, &evt_compute);
    uint64_t t_exec_end = core_now_ns();

    uint64_t t_download_start = core_now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    void* mapped_len = clEnqueueMapBuffer(queue, d_len, CL_TRUE, CL_MAP_READ, 0, nblk * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, nblk * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_len, mapped_len, 0, NULL, NULL));

    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++)
        comp_total += len_arr[i];
    if (params->debug) {
        fprintf(stderr, "[CORE][DEBUG-LENS] comp_total=%zu, out_needed=%zu\n", comp_total, out_needed);
    }

    /* If kernel supports debug instrumentation and created a debug buffer, map and print it when debug mode is enabled. */
    if (kernel_has_dbg_arg && d_dbg != NULL) {
        const size_t dbg_fields = 7;
        size_t dbg_bytes = nblk * dbg_fields * sizeof(cl_uint);
        cl_int err_dbg;
        if (params->debug) {
            uint32_t *dbg_map = clEnqueueMapBuffer(queue, d_dbg, CL_TRUE, CL_MAP_READ, 0, dbg_bytes, 0, NULL, NULL, &err_dbg);
            if (err_dbg == CL_SUCCESS && dbg_map) {
                int abort_debug_sanity = lzo_parse_and_print_debug_buffer(dbg_map, dbg_fields, nblk, blk, worst_blk);
                CHECK(clEnqueueUnmapMemObject(queue, d_dbg, dbg_map, 0, NULL, NULL));
                if (abort_debug_sanity) {
                    fprintf(stderr, "[CORE] Debug sanity checks failed, aborting compression job\n");
                    /* cleanup and return error */
                    free(len_arr);
                    if (d_dbg) { clReleaseMemObject(d_dbg); d_dbg = NULL; }
                    return -1;
                }
            } else {
                fprintf(stderr, "[CORE] Failed to map debug buffer: %d\n", err_dbg);
            }
        }
    }

    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, out_needed, 0, NULL, NULL, &err); CHECK(err);
    uint64_t t_download_end = core_now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    uint64_t t_write_start = core_now_ns();
    int write_ret = core_write_compressed_file(output_path, in_sz, blk, nblk, len_arr, mapped_out, worst_blk, params->coalesce_output, params->coalesce_chunk_mb, params->coalesce_max_mb, params->stdio_buf_mb, params->alg_id, params->debug);
    uint64_t t_write_end = core_now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    if (write_ret != 0) {
        fprintf(stderr, "[CORE] Failed to write compressed file\n");
        CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));
        free(len_arr);
        if (d_dbg) {
            clReleaseMemObject(d_dbg);
            d_dbg = NULL;
        }
        return -1;
    }

    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));

    uint64_t t_cleanup_start = core_now_ns();
    free(len_arr);
    clFlush(queue);
    clFinish(queue); /* release debug buffer if allocated */
    if (d_dbg) {
        clReleaseMemObject(d_dbg);
        d_dbg = NULL;
    }
    uint64_t t_cleanup_end = core_now_ns();
    unsigned long cleanup_us = (t_cleanup_end - t_cleanup_start) / 1000;

    uint64_t t_total_end = core_now_ns();
    unsigned long total_us = (t_total_end - t_total_start) / 1000;
    *time_us_out = total_us; *output_size_out = comp_total;

    if (t_out) {
        t_out->file_read_us = read_us;
        t_out->ocl_init_us = 0;
        t_out->kernel_load_us = 0;
        t_out->blocking_calc_us = blocking_us;
        t_out->buffer_alloc_in_us = buffer_in_us;
        t_out->data_upload_us = upload_us;
        t_out->buffer_alloc_out_us = buffer_out_us;
        t_out->buffer_alloc_len_us = buffer_len_us;
        t_out->kernel_exec_us = (unsigned long)((t_exec_end - t_exec_start)/1000);
        t_out->download_total_us = download_us;
        t_out->file_write_us = write_us;
        t_out->cleanup_us = cleanup_us;
        t_out->blk_size_bytes = (unsigned long)blk;
        t_out->nblk = (unsigned long)nblk;
        t_out->global_size = (unsigned long)global_size;
        t_out->local_size = (unsigned long)local_size;
        memset(t_out->kernel_name, 0, sizeof(t_out->kernel_name));
    }

    double ratio = comp_total > 0 ? (double)in_sz / (double)comp_total : 0.0;
    fprintf(stderr, "\n=== Compression Statistics ===\n");
    fprintf(stderr, "Input size       : %zu bytes (%.2f MB)\n", in_sz, in_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", comp_total, comp_total / (1024.0 * 1024.0));
    fprintf(stderr, "Compression ratio: %.2f:1 (%.2f%% of original)\n", ratio, 100.0 / ratio);
    fprintf(stderr, "Block size       : %zu bytes (%zu KB)\n", blk, blk / 1024);
    fprintf(stderr, "Number of blocks : %zu\n", nblk);
    fprintf(stderr, "Compression level: %d\n", params->level);
    fprintf(stderr, "Work groups      : global=%zu, local=%zu\n", global_size, local_size);
    double kernel_thrpt = (t_out && t_out->kernel_exec_us > 0) ? ((double)in_sz / (1024.0*1024.0)) / (t_out->kernel_exec_us/1000000.0) : 0.0;
    fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", ((double)in_sz / (1024.0*1024.0)) / (total_us/1000000.0), kernel_thrpt);
    fprintf(stderr, "==============================\n\n");

    fprintf(stderr, "\n=== Time Breakdown (Compression) ===\n");
    print_us_tag(stderr, "1. File Read", read_us);
    print_us_tag(stderr, "2. Blocking Calc", blocking_us);
    print_us_tag(stderr, "3. Buffer Alloc (in)", buffer_in_us);
    print_us_tag(stderr, "4. GPU Upload", upload_us);
    print_us_tag(stderr, "5. Buffer Alloc (out)", buffer_out_us);
    print_us_tag(stderr, "6. Buffer Alloc (len)", buffer_len_us);
    print_us_tag(stderr, "7. Kernel Exec", (unsigned long)((t_exec_end - t_exec_start)/1000));
    print_us_tag(stderr, "8. GPU Download", download_us);
    print_us_tag(stderr, "9. File Write", write_us);
    print_us_tag(stderr, "10. Cleanup", cleanup_us);
    print_us_tag(stderr, "TOTAL", total_us);
    fprintf(stderr, "\n");

    return write_ret;
}

/* Core decompression implementation (adapted from daemon_decompress.c) */
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
) {
    cl_int err;
    uint64_t t_start = core_now_ns();

    /* Resource tracking for cleanup */
    FILE* f_in = NULL;
    uint32_t* len_arr = NULL;
    uint32_t* off_arr = NULL;
    void* comp_host = NULL;
    void* mapped_out = NULL;
    FILE* fout = NULL;
    int ret = -1; /* Default failure */

    f_in = fopen(input_path, "rb");
    if (!f_in) {
        perror("fopen input");
        goto cleanup;
    }

    uint16_t magic;
    if (fread(&magic, sizeof(magic), 1, f_in) != 1) {
        perror("fread magic");
        goto cleanup;
    }
    if (magic != 0x4C5A) {
        fprintf(stderr, "[DECOMP] wrong file magic\n");
        goto cleanup;
    }

    uint32_t orig_sz, blk_sz, nblk, alg_id;
    if (fread(&orig_sz, sizeof(orig_sz), 1, f_in) != 1 ||
        fread(&blk_sz, sizeof(blk_sz), 1, f_in) != 1 ||
        fread(&nblk, sizeof(nblk), 1, f_in) != 1 ||
        fread(&alg_id, sizeof(alg_id), 1, f_in) != 1) {
        perror("fread header");
        goto cleanup;
    }

    len_arr = malloc(nblk * sizeof(uint32_t));
    if (!len_arr) {
        perror("malloc len_arr");
        goto cleanup;
    }
    if (fread(len_arr, sizeof(uint32_t), nblk, f_in) != nblk) {
        perror("fread len_arr");
        goto cleanup;
    }

    long current_pos = ftell(f_in); fseek(f_in, 0, SEEK_END); long file_sz = ftell(f_in); fseek(f_in, current_pos, SEEK_SET);
    size_t comp_sz = file_sz - current_pos;

    fprintf(stderr, "[DECOMP] file summary: orig=%u blk=%u nblk=%u comp=%zu\n", orig_sz, blk_sz, nblk, comp_sz);

    off_arr = malloc((nblk + 1) * sizeof(uint32_t));
    off_arr[0] = 0;
    for (uint32_t i = 0; i < nblk; ++i)
        off_arr[i+1] = off_arr[i] + len_arr[i];
    free(len_arr);
    len_arr = NULL; /* Mark as freed */

    int use_standard_copy_decomp = standard_copy ? 1 : 0;

    uint64_t t_buf_comp_start = core_now_ns();
    /* For zero-copy decompression, use ALLOC_HOST_PTR to get pinned memory */
    cl_mem_flags comp_flags = use_standard_copy_decomp ? CL_MEM_READ_ONLY : (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
    cl_mem d_comp = core_get_or_create_decomp_buffer(ctx, &decomp_buffer_cache.d_comp, &decomp_buffer_cache.comp_size, comp_sz, comp_flags, &err); CHECK(err);
    uint64_t t_buf_comp_end = core_now_ns();
    unsigned long buf_comp_us = (t_buf_comp_end - t_buf_comp_start) / 1000;

    uint64_t t_buf_off_start = core_now_ns();
    cl_mem d_off = core_get_or_create_decomp_buffer(ctx, &decomp_buffer_cache.d_off, &decomp_buffer_cache.off_size, (nblk + 1) * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); CHECK(err);
    uint64_t t_buf_off_end = core_now_ns();
    unsigned long buf_off_us = (t_buf_off_end - t_buf_off_start) / 1000;

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_decomp_buffer(ctx, &decomp_buffer_cache.d_out, &decomp_buffer_cache.out_size, orig_sz, CL_MEM_WRITE_ONLY, &err); CHECK(err);
    cl_mem d_out_lens = core_get_or_create_decomp_buffer(ctx, &decomp_buffer_cache.d_out_lens, &decomp_buffer_cache.lens_size, nblk * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err); CHECK(err);
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buf_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    unsigned long buf_us = buf_comp_us + buf_off_us + buf_out_us;

    /* Read compressed data with MT-IO optimization */
    uint64_t t_file_read_start = core_now_ns();
    unsigned long file_read_us = 0;

    /* Check if MT-IO is enabled via environment */
    int use_mt_io = 0;
    int mt_threads_decomp = 4; /* default */
    const char* mt_env = getenv("LZO_MT_IO");
    if (mt_env && atoi(mt_env) > 0) {
        use_mt_io = 1;
        const char* threads_env = getenv("LZO_MT_THREADS");
        if (threads_env) {
            mt_threads_decomp = atoi(threads_env);
            if (mt_threads_decomp < 1) mt_threads_decomp = 1;
            if (mt_threads_decomp > 32) mt_threads_decomp = 32;
        }
    }

    /* Allocate host buffer for compressed data */
    if (use_standard_copy_decomp) {
        /* Standard copy: allocate regular host memory */
        int rc_mem = posix_memalign(&comp_host, ALIGN_BYTES, comp_sz);
        if (rc_mem != 0 || comp_host == NULL) {
            comp_host = malloc(comp_sz);
            if (!comp_host) {
                perror("malloc comp_host");
                goto cleanup;
            }
        }
    } else {
        /* Zero-copy: map the device buffer to get pinned host memory */
        comp_host = clEnqueueMapBuffer(queue, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !comp_host) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_comp failed: %d, falling back to standard copy\n", err);
            use_standard_copy_decomp = 1;
            comp_host = malloc(comp_sz);
            if (!comp_host) {
                perror("malloc comp_host fallback");
                goto cleanup;
            }
        }
    }

    /* Read file with MT-IO if enabled */
    long pos_before_data = ftell(f_in);
    if (use_mt_io && comp_sz > LZO_MT_IO_SIZE_THRESHOLD) { /* Use MT-IO for files > 1MB */
        fclose(f_in);
        f_in = NULL; /* MT-IO will reopen */

        /* Read from file at offset */
        int fd = open(input_path, O_RDONLY);
        if (fd < 0) {
            perror("open for MT-IO");
            goto cleanup;
        }

        if (lzo_mt_read_file_fd(fd, comp_host, comp_sz, pos_before_data, mt_threads_decomp, &file_read_us) != 0) {
            perror("lzo_mt_read_file_fd");
            close(fd);
            goto cleanup;
        }
        close(fd);
    } else {
        uint64_t t_fread_start = core_now_ns();
        if (fread(comp_host, 1, comp_sz, f_in) != comp_sz) {
            perror("fread compressed data");
            goto cleanup;
        }
        fclose(f_in);
        f_in = NULL;
        uint64_t t_fread_end = core_now_ns();
        file_read_us = (t_fread_end - t_fread_start) / 1000;
    }
    uint64_t t_file_read_end = core_now_ns();
    unsigned long total_file_read_us = (t_file_read_end - t_file_read_start) / 1000;

    /* Now upload to GPU */
    uint64_t t_upload_start = core_now_ns();

    if (use_standard_copy_decomp) {
        /* Standard copy: explicit upload with async write */
        cl_event evt_write_comp;
        err = clEnqueueWriteBuffer(queue, d_comp, CL_FALSE, 0, comp_sz, comp_host, 0, NULL, &evt_write_comp);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueWriteBuffer d_comp failed: %d\n", err);
            goto cleanup;
        }

        /* Upload offset array with async write */
        cl_event evt_write_off;
        err = clEnqueueWriteBuffer(queue, d_off, CL_FALSE, 0, (nblk + 1) * sizeof(cl_uint), off_arr, 0, NULL, &evt_write_off);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueWriteBuffer d_off failed: %d\n", err);
            clReleaseEvent(evt_write_comp);
            goto cleanup;
        }

        /* Wait for both uploads to complete */
        cl_event upload_events[2] = {evt_write_comp, evt_write_off};
        clWaitForEvents(2, upload_events);
        clReleaseEvent(evt_write_comp);
        clReleaseEvent(evt_write_off);
    } else {
        /* Zero-copy: data is already in pinned memory, just unmap */
        err = clEnqueueUnmapMemObject(queue, d_comp, comp_host, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_comp failed: %d\n", err);
            goto cleanup;
        }
        comp_host = NULL; /* Unmapped, no longer valid */

        /* For offset array, also use zero-copy: map, copy, unmap */
        void* mapped_off = clEnqueueMapBuffer(queue, d_off, CL_TRUE, CL_MAP_WRITE, 0, (nblk + 1) * sizeof(cl_uint), 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_off failed: %d\n", err);
            goto cleanup;
        }
        memcpy(mapped_off, off_arr, (nblk + 1) * sizeof(cl_uint));
        err = clEnqueueUnmapMemObject(queue, d_off, mapped_off, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_off failed: %d\n", err);
            goto cleanup;
        }
    }

/* Free comp_host only if we allocated it (standard copy mode) */
    if (use_standard_copy_decomp && comp_host) {
        free(comp_host);
    }
    comp_host = NULL;
    free(off_arr);
    off_arr = NULL;

    uint64_t t_upload_end = core_now_ns();

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &nblk));

    size_t local_size = (local_size_param > 0) ? (size_t)local_size_param : LZO_LOCAL_SIZE_DEFAULT;
    size_t global_size = ((size_t)nblk + local_size - 1) / local_size * local_size;
    uint64_t t_exec_start = core_now_ns();
    fprintf(stderr, "[DECOMP] launching kernel: nblk=%u global_size=%zu local_size=%zu\n", nblk, global_size, local_size);

    cl_event evt_exec = NULL;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &evt_exec);
    CHECK(err);

    clWaitForEvents(1, &evt_exec);
    clFinish(queue);
    uint64_t t_exec_end = core_now_ns();

    cl_ulong ev_start = 0, ev_end = 0;
    if (evt_exec) {
        cl_int perr = CL_SUCCESS;
        perr = clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_START, sizeof(ev_start), &ev_start, NULL);
        if (perr != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] profiling start info not available (err=%d)\n", perr);
        } else {
            cl_int perr2 = clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_END, sizeof(ev_end), &ev_end, NULL);
            if (perr2 != CL_SUCCESS)
                fprintf(stderr, "[DECOMP] profiling end info not available (err=%d)\n", perr2);
        }
        // fprintf(stderr, "[DECOMP] event profiling: start=%llu end=%llu\n", (unsigned long long)ev_start, (unsigned long long)ev_end);
        clReleaseEvent(evt_exec);
    } else {
        fprintf(stderr, "[DECOMP] no event returned from clEnqueueNDRangeKernel\n");
    }

    uint64_t t_download_start = core_now_ns();
    mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, orig_sz, 0, NULL, NULL, &err);
    CHECK(err);
    uint64_t t_download_end = core_now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    uint64_t t_write_start2 = core_now_ns();
    fout = fopen(output_path, "wb");
    if (!fout) {
        perror("fopen output");
        goto cleanup;
    }
    if (fwrite(mapped_out, 1, orig_sz, fout) != orig_sz) {
        perror("fwrite");
        goto cleanup;
    }
    fclose(fout);
    fout = NULL;
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));
    mapped_out = NULL;
    uint64_t t_write_end2 = core_now_ns();
    unsigned long write_us = (t_write_end2 - t_write_start2) / 1000;

    size_t cache_threshold_mb = DEFAULT_DECOMP_CACHE_MB;
    const char* cache_env = getenv("LZO_DECOMP_CACHE_MB");
    if (cache_env)
        cache_threshold_mb = (size_t)atoi(cache_env);
    size_t cache_threshold = cache_threshold_mb * 1024 * 1024;
    if (orig_sz > cache_threshold) {
        if (decomp_buffer_cache.d_out) {
            clReleaseMemObject(decomp_buffer_cache.d_out);
            decomp_buffer_cache.d_out = NULL;
            decomp_buffer_cache.out_size = 0;
        }
        if (decomp_buffer_cache.d_out_lens) {
            clReleaseMemObject(decomp_buffer_cache.d_out_lens);
            decomp_buffer_cache.d_out_lens = NULL;
            decomp_buffer_cache.lens_size = 0;
        }
    }

    free(off_arr);
    off_arr = NULL;

    uint64_t t_end = core_now_ns();
    *time_us_out = (t_end - t_start) / 1000;
    *output_size_out = orig_sz;

    unsigned long upload_us_local = (t_upload_end - t_upload_start) / 1000;
    unsigned long exec_host_us = (t_exec_end - t_exec_start) / 1000;
    unsigned long download_us_local = download_us;
    unsigned long write_us_local = write_us;
    unsigned long exec_us_ev = 0;
    if (ev_start != 0 && ev_end != 0 && ev_end > ev_start)
        exec_us_ev = (unsigned long)((ev_end - ev_start) / 1000);

    fprintf(stderr, "\n=== Decompression Statistics ===\n");
    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", comp_sz, comp_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Output size      : %u bytes (%.2f MB)\n", orig_sz, orig_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Block size       : %u bytes (%u KB)\n", blk_sz, blk_sz / 1024);
    fprintf(stderr, "Work groups      : global=%zu, local=%zu\n", global_size, local_size);
    if (use_mt_io && comp_sz > 1024*1024) {
        fprintf(stderr, "MT-IO enabled    : %d threads\n", mt_threads_decomp);
    }
    double kernel_thrpt = exec_host_us > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (exec_host_us/1000000.0) : 0.0;
    fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", ((double)orig_sz / (1024.0*1024.0)) / (*time_us_out/1000000.0), kernel_thrpt);
    fprintf(stderr, "==============================\n\n");

    fprintf(stderr, "\n=== Time Breakdown (Decompression) ===\n");
    print_us_tag(stderr, "1. File Read", total_file_read_us);
    if (use_mt_io && comp_sz > 1024*1024) {
        print_us_tag(stderr, "   (MT-IO overhead)", total_file_read_us - file_read_us);
    }
    print_us_tag(stderr, "2. Buffer Alloc (comp)", buf_comp_us);
    print_us_tag(stderr, "3. Buffer Alloc (off)", buf_off_us);
    print_us_tag(stderr, "4. Buffer Alloc (out)", buf_out_us);
    print_us_tag(stderr, "5. GPU Upload", upload_us_local);
    print_us_tag(stderr, "6. Kernel Exec", exec_host_us);
    if (exec_us_ev) print_us_tag(stderr, "   (event profiling)", exec_us_ev);
    print_us_tag(stderr, "7. GPU Download", download_us_local);
    print_us_tag(stderr, "8. File Write", write_us_local);
    print_us_tag(stderr, "TOTAL", (unsigned long)(*time_us_out));
    fprintf(stderr, "\n");

    if (t_out) {
        t_out->file_read_us = (t_buf_comp_start - t_start) / 1000;
        t_out->ocl_init_us = 0;
        t_out->kernel_load_us = 0;
        t_out->buffer_alloc_in_us = buf_us;
        t_out->data_upload_us = upload_us_local;
        t_out->setup_args_us = 0;
        t_out->kernel_exec_us = exec_us_ev ? exec_us_ev : exec_host_us;
        t_out->download_total_us = download_us_local;
        t_out->file_write_us = write_us_local;
        t_out->global_size = global_size; t_out->local_size = local_size; t_out->blk_size_bytes = blk_sz; t_out->nblk = nblk;
        memset(t_out->kernel_name, 0, sizeof(t_out->kernel_name)); strncpy(t_out->kernel_name, "lzo1x_decomp", sizeof(t_out->kernel_name) - 1);
    }

    ret = 0; /* Success */

cleanup:
    /* Unified cleanup: close files, free resources, unmap buffers */
    if (f_in) fclose(f_in);
    if (len_arr) free(len_arr);
    if (off_arr) free(off_arr);
    if (comp_host) {
        /* Only free if we allocated it ourselves (standard copy) */
        if (use_standard_copy_decomp) {
            free(comp_host);
        }
        /* If zero-copy and still mapped (error path), unmap it */
        /* Note: in success path, comp_host is already NULL after unmap */
    }
    if (fout) fclose(fout);
    if (mapped_out) clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL);

    return ret;
}
