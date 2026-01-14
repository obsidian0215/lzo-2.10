/* lzo_gpu_core.c - shared backend for compression/decompression
 * Extracted from daemon_compress.c / daemon_decompress.c so both daemon
 * and standalone can reuse same implementation.
 */

#include <CL/cl.h>
#include "timing.h"
#include "lzo_defaults.h"
#include "lzo_gpu_core.h"
#include "lzo_gpu_utils.h"
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

int g_verbose = 0;

static inline size_t core_lzo_worst(size_t sz) {
    return sz + sz / 16 + 64 + 3;
}

static inline uint64_t core_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void lzo_gpu_workspace_init(lzo_gpu_workspace_t* ws) {
    memset(ws, 0, sizeof(lzo_gpu_workspace_t));
}

void lzo_gpu_workspace_free(lzo_gpu_workspace_t* ws) {
    if (ws->d_in) clReleaseMemObject(ws->d_in);
    if (ws->d_out) clReleaseMemObject(ws->d_out);
    if (ws->d_len) clReleaseMemObject(ws->d_len);
    if (ws->d_dict) clReleaseMemObject(ws->d_dict);
    if (ws->d_comp) clReleaseMemObject(ws->d_comp);
    if (ws->d_off) clReleaseMemObject(ws->d_off);
    if (ws->d_decomp_out) clReleaseMemObject(ws->d_decomp_out);
    if (ws->d_out_lens) clReleaseMemObject(ws->d_out_lens);
    memset(ws, 0, sizeof(lzo_gpu_workspace_t));
}

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

/* Core compression implementation */
int lzo_compress_core(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    const lzo_compress_params_t* params,
    lzo_gpu_workspace_t* ws,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_total_start = core_now_ns();

    int use_standard_copy = params->standard_copy ? 1 : 0;

    /* 1. 获取文件大小 (使用 stat 替代 fopen/fseek, 避免重复打开) */
    struct stat st;
    if (stat(input_path, &st) != 0) {
        perror("stat input");
        return -1;
    }
    size_t in_sz = (size_t)st.st_size;
    if (in_sz == 0) {
        /* Handle empty file */
        return -1;
    }

    /* 2. 准备输入缓冲区 (如果是 Daemon 模式,此处通常会命中预分配好的常驻缓冲) */
    uint64_t t_buf_in_start = core_now_ns();
    cl_mem d_in = core_get_or_create_buffer(ctx, &ws->d_in, &ws->in_size, in_sz, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create input buffer: %d\n", err);
        return -1;
    }

    void* mapped_in = NULL;
    if (!use_standard_copy) {
        /* Map with Non-blocking if possible, but CORE uses CL_TRUE for simplicity */
        mapped_in = clEnqueueMapBuffer(queue, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] Map input buffer failed: %d\n", err);
            return -1;
        }
    }
    uint64_t t_buf_in_end = core_now_ns();
    unsigned long buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000;

    unsigned long read_us = 0;
    unsigned long upload_us = 0;
    void* host_in = NULL;

    if (!use_standard_copy) {
        /* 直接读入映射好的显存空间 (Zero-copy 核心) */
        if (lzo_read_file_to_buf(input_path, mapped_in, in_sz, &read_us) != 0) {
            perror("lzo_read_file_to_buf");
            clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
            return -1;
        }
        /* upload_us will be measured after unmap */
    } else {
        int rc_mem = posix_memalign(&host_in, ALIGN_BYTES, in_sz);
        if (rc_mem != 0 || host_in == NULL) {
            host_in = malloc(in_sz);
            if (!host_in) {
                perror("malloc host_in");
                return -1;
            }
        }

        /* Use unified IO module */
        if (lzo_read_file_to_buf(input_path, host_in, in_sz, &read_us) != 0) {
            perror("lzo_read_file_to_buf");
            free(host_in);
            return -1;
        }
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

    uint64_t t_blocking_end = core_now_ns();
    unsigned long blocking_us = (t_blocking_end - t_blocking_start) / 1000;

    if (!use_standard_copy) {
        uint64_t t_unmap_start = core_now_ns();
        CHECK(clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL));
        /* clFinish is removed here for better performance, OpenCL queue maintains ordering */
        uint64_t t_unmap_end = core_now_ns();
        upload_us = (t_unmap_end - t_unmap_start) / 1000;
    } else {
        uint64_t t_upload_start2 = core_now_ns();
        err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] clEnqueueWriteBuffer failed: %d\n", err);
            free(host_in);
            return -1;
        }
        free(host_in);
        host_in = NULL;
        uint64_t t_upload_end2 = core_now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
    }

    size_t worst_blk = core_lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_out, &ws->out_size, out_needed, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create output buffer: %d\n", err);
        return -1;
    }
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buffer_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    /* Buffer Alloc (len) timing removed per request; keep allocation but don't time it */
    cl_mem d_len = core_get_or_create_buffer(ctx, &ws->d_len, &ws->len_size, len_needed, CL_MEM_READ_WRITE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create len buffer: %d\n", err);
        return -1;
    }

    cl_uint in_sz_cl = (cl_uint)in_sz, blk_cl = (cl_uint)blk, worst_blk_cl = (cl_uint)worst_blk;
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));

    cl_mem d_dbg = NULL;
    int kernel_has_dbg_arg = 0;

    /* Dictionary Pool Architecture:
     * We use a pool of dictionaries in global memory, scaled by the number of CUs.
     */
    cl_uint cus = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    if (err != CL_SUCCESS || cus == 0) cus = 32; // Fallback

    /* multiplier: how many workgroups per CU. 4-8 is usually good for latency hiding.
     */
    uint32_t pool_size = cus * 4;
    if (pool_size > 2048) pool_size = 2048; // Cap at 2048 for now
    if (pool_size < 128) pool_size = 128;   // Minimum floor for small GPUs

    if (params->debug) {
        fprintf(stderr, "[CORE] CU count: %u, dict_pool_size: %u\n", cus, pool_size);
    }
    {
        size_t dict_per_block = (1ULL << params->level) * sizeof(unsigned short);
        size_t total_dict_size = (size_t)pool_size * dict_per_block;

        cl_mem d_dict = core_get_or_create_buffer(ctx, &ws->d_dict, &ws->dict_size, total_dict_size, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] failed to create dictionary buffer: %d\n", err);
            return -1;
        }
        CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
        CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));

        /* If it's a debug kernel, debug buffer is at index 8 */
        cl_uint num_args = 0;
        cl_int rc = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
        if (rc == CL_SUCCESS && num_args >= 9) {
            kernel_has_dbg_arg = 1;
            const size_t dbg_fields = 7;
            size_t per_blk = dbg_fields * sizeof(cl_uint);
            size_t dbg_bytes = nblk * per_blk;
            d_dbg = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, dbg_bytes, NULL, &err);
            if (err == CL_SUCCESS) {
                CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_dbg));
            } else {
                d_dbg = NULL;
                CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_dbg));
            }
        }
    }

    size_t local_size = LZO_LOCAL_SIZE_DEFAULT;
    if (params->local_size_param > 0) {
        local_size = params->local_size_param;
        if (params->debug) fprintf(stderr, "[CORE] using local_size=%zu\n", local_size);
    }

    // Launch at most pool_size workgroups.
    size_t active_groups = (nblk < pool_size) ? nblk : pool_size;
    size_t global_size = active_groups * local_size;

    uint64_t t_exec_start = 0, t_exec_end = 0;
    cl_event evt_compute;
    cl_event* p_evt = g_verbose ? &evt_compute : NULL;

    if (g_verbose) t_exec_start = core_now_ns();
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, p_evt));
    if (g_verbose) {
        clWaitForEvents(1, &evt_compute);
        t_exec_end = core_now_ns();
    }

    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    void* mapped_len = clEnqueueMapBuffer(queue, d_len, CL_TRUE, CL_MAP_READ, 0, nblk * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, nblk * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_len, mapped_len, 0, NULL, NULL));

    size_t comp_total = 0;
    cl_uint* off_arr = malloc(nblk * sizeof(cl_uint));
    for (size_t i = 0; i < nblk; i++) {
        off_arr[i] = (cl_uint)comp_total;
        comp_total += len_arr[i];
    }

    unsigned long download_us = 0;
    void* host_comp = NULL;

    /* Map the whole output buffer */
    if (off_arr) free(off_arr);
    uint64_t t_down_start = core_now_ns();
    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, out_needed, 0, NULL, NULL, &err); CHECK(err);
    host_comp = mapped_out;
    uint64_t t_down_end = core_now_ns();
    download_us = (unsigned long)((t_down_end - t_down_start) / 1000);

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
                    CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));
                    return -1;
                }
            } else {
                fprintf(stderr, "[CORE] Failed to map debug buffer: %d\n", err_dbg);
            }
        }
    }

    uint64_t t_write_start = core_now_ns();
    int write_ret = 0;
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        write_ret = lzo_write_compressed_file(output_path, in_sz, blk, nblk, len_arr, host_comp,
                                               worst_blk, params->alg_id, params->debug);
    }
    uint64_t t_write_end = core_now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    if (write_ret != 0) {
        fprintf(stderr, "[CORE] Failed to write compressed file\n");
        CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));
        free(len_arr);
        if (d_dbg) { clReleaseMemObject(d_dbg); d_dbg = NULL; }
        return -1;
    }

    CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));

    free(len_arr);
    clFlush(queue);
    clFinish(queue); /* release debug buffer if allocated */
    if (d_dbg) {
        clReleaseMemObject(d_dbg);
        d_dbg = NULL;
    }

    uint64_t t_total_end = core_now_ns();
    unsigned long total_us = (t_total_end - t_total_start) / 1000;
    *time_us_out = total_us; *output_size_out = comp_total;

    unsigned long buffer_alloc_us = buffer_in_us + buffer_out_us;
    if (t_out) {
        t_out->file_read_us = read_us;
        t_out->ocl_init_us = 0;
        t_out->kernel_load_us = 0;
        t_out->blocking_calc_us = blocking_us;
        t_out->buffer_alloc_us = buffer_alloc_us;
        t_out->data_upload_us = upload_us;
        /* buffer_alloc_out_us / buffer_alloc_len_us removed */
        t_out->kernel_exec_us = (g_verbose) ? (unsigned long)((t_exec_end - t_exec_start)/1000) : 0;
        t_out->download_total_us = download_us;
        t_out->file_write_us = write_us;
        t_out->blk_size_bytes = (unsigned long)blk;
        t_out->nblk = (unsigned long)nblk;
        t_out->global_size = (unsigned long)global_size;
        t_out->local_size = (unsigned long)local_size;
        memset(t_out->kernel_name, 0, sizeof(t_out->kernel_name));
    }

    double ratio = comp_total > 0 ? (double)in_sz / (double)comp_total : 0.0;
    if (g_verbose) {
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
        unsigned long buffer_alloc_us = buffer_in_us + buffer_out_us;
        print_us_tag(stderr, "3. Buffer Alloc", buffer_alloc_us);
        print_us_tag(stderr, "4. GPU Upload", upload_us);
        print_us_tag(stderr, "5. Kernel Exec", (unsigned long)((t_exec_end - t_exec_start)/1000));
        print_us_tag(stderr, "6. GPU Download", download_us);
        print_us_tag(stderr, "7. File Write", write_us);
        print_us_tag(stderr, "TOTAL", total_us);
        fprintf(stderr, "\n");
    } else {
        /* Minimal output for non-verbose mode: simple summary with total time */
        printf("Compressed %zu -> %zu (%.2f:1) in %.2f ms\n", in_sz, comp_total, ratio, total_us / 1000.0);
    }

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
    lzo_gpu_workspace_t* ws,
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
    cl_mem d_comp = core_get_or_create_buffer(ctx, &ws->d_comp, &ws->comp_size, comp_sz, comp_flags, &err); CHECK(err);
    uint64_t t_buf_comp_end = core_now_ns();
    unsigned long buf_comp_us = (t_buf_comp_end - t_buf_comp_start) / 1000;

    /* Buffer Alloc (off) timing removed per request; keep allocation but don't time it */
    cl_mem d_off = core_get_or_create_buffer(ctx, &ws->d_off, &ws->off_size, (nblk + 1) * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); CHECK(err);

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_size, orig_sz, CL_MEM_WRITE_ONLY, &err); CHECK(err);
    cl_mem d_out_lens = core_get_or_create_buffer(ctx, &ws->d_out_lens, &ws->lens_size, nblk * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err); CHECK(err);
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buf_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    unsigned long buf_us = buf_comp_us + buf_out_us;

    /* Allocate host buffer for compressed data */
    if (use_standard_copy_decomp) {
        /* Standard copy */
        comp_host = malloc(comp_sz);
        if (!comp_host) {
            perror("malloc comp_host");
            goto cleanup;
        }
    } else {
        /* Zero-copy mapping */
        comp_host = clEnqueueMapBuffer(queue, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !comp_host) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_comp failed: %d\n", err);
            goto cleanup;
        }
    }

    /* Read compressed data */
    uint64_t t_file_read_start = core_now_ns();
    if (fread(comp_host, 1, comp_sz, f_in) != comp_sz) {
        perror("fread compressed data");
        goto cleanup;
    }
    fclose(f_in);
    f_in = NULL;
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
    uint64_t t_exec_start = 0, t_exec_end = 0;
    if (g_verbose) {
        t_exec_start = core_now_ns();
        fprintf(stderr, "[DECOMP] launching kernel: nblk=%u global_size=%zu local_size=%zu\n", nblk, global_size, local_size);
    }

    cl_event evt_exec = NULL;
    cl_event* p_evt = g_verbose ? &evt_exec : NULL;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, p_evt);
    CHECK(err);

    if (g_verbose && evt_exec) {
        clWaitForEvents(1, &evt_exec);
        t_exec_end = core_now_ns();
    }

    cl_ulong ev_start = 0, ev_end = 0;
    if (g_verbose && evt_exec) {
        cl_int perr = CL_SUCCESS;
        perr = clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_START, sizeof(ev_start), &ev_start, NULL);
        if (perr == CL_SUCCESS) {
            clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_END, sizeof(ev_end), &ev_end, NULL);
        }
        clReleaseEvent(evt_exec);
    }

    uint64_t t_download_start = core_now_ns();
    mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, orig_sz, 0, NULL, NULL, &err);
    CHECK(err);
    uint64_t t_download_end = core_now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    uint64_t t_write_start2 = core_now_ns();
    int write_ret = 0;
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
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
    }
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
        if (ws->d_decomp_out) {
            clReleaseMemObject(ws->d_decomp_out);
            ws->d_decomp_out = NULL;
            ws->decomp_out_size = 0;
        }
        if (ws->d_out_lens) {
            clReleaseMemObject(ws->d_out_lens);
            ws->d_out_lens = NULL;
            ws->lens_size = 0;
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

    if (g_verbose) {
        fprintf(stderr, "\n=== Decompression Statistics ===\n");
        fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", comp_sz, comp_sz / (1024.0 * 1024.0));
        fprintf(stderr, "Output size      : %u bytes (%.2f MB)\n", orig_sz, orig_sz / (1024.0 * 1024.0));
        fprintf(stderr, "Block size       : %u bytes (%u KB)\n", blk_sz, blk_sz / 1024);
        fprintf(stderr, "Work groups      : global=%zu, local=%zu\n", global_size, local_size);
        double kernel_thrpt = exec_host_us > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (exec_host_us/1000000.0) : 0.0;
        fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n", ((double)orig_sz / (1024.0*1024.0)) / (*time_us_out/1000000.0), kernel_thrpt);
        fprintf(stderr, "==============================\n\n");

        fprintf(stderr, "\n=== Time Breakdown (Decompression) ===\n");
        print_us_tag(stderr, "1. File Read", total_file_read_us);
        print_us_tag(stderr, "2. Buffer Alloc", buf_us);
        print_us_tag(stderr, "3. GPU Upload", upload_us_local);
        print_us_tag(stderr, "4. Kernel Exec", exec_host_us);
        if (exec_us_ev) print_us_tag(stderr, "   (event profiling)", exec_us_ev);
        print_us_tag(stderr, "5. GPU Download", download_us_local);
        print_us_tag(stderr, "6. File Write", write_us_local);
        print_us_tag(stderr, "TOTAL", (unsigned long)(*time_us_out));
        fprintf(stderr, "\n");
    } else {
        printf("Decompressed %zu -> %u in %.2f ms\n", comp_sz, orig_sz, (*time_us_out) / 1000.0);
    }

    if (t_out) {
        t_out->file_read_us = (t_buf_comp_start - t_start) / 1000;
        t_out->ocl_init_us = 0;
        t_out->kernel_load_us = 0;
        t_out->buffer_alloc_us = buf_us;
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
