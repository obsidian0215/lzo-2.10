/*
 * daemon_decompress.c - 守护进程解压缩核心逻辑
 *
 * Phase 7.2: 同步Pinned Memory和Buffer缓存优化
 */

#include <CL/cl.h>
#include "timing.h"
#include "lzo_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define CHECK(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

#define MAGIC 0x4C5A  // 'L''Z' - LZO文件魔数

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Phase 7.2: Buffer缓存机制 */
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

static cl_mem get_or_create_buffer(cl_context ctx, cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags, cl_int* err_out) {
    if (*cached_size < required_size) {
        if (*cached_buf) {
            clReleaseMemObject(*cached_buf);
            *cached_buf = NULL;
            *cached_size = 0;
        }
           /* Always use pinned host mapping for buffers to restore HEAD behaviour */
           cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags,
                 required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS) {
            *cached_size = required_size;
        }
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

/* 读取文件 */
static void* read_file_data(const char* path, size_t* size_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    void* buf = malloc(sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (fread(buf, 1, sz, f) != (size_t)sz) {
        perror("fread");
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *size_out = sz;
    return buf;
}

/*
 * 守护进程解压缩函数
 * 复用预分配的OpenCL资源,仅执行必要的解压缩操作
 *
 * Phase 7.2更新:
 * - Pinned Memory优化
 * - Buffer缓存机制
 */
int daemon_decompress(
    /* OpenCL资源 (已初始化,复用) */
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,         // 解压缩kernel (标量或向量)
    int prefer_vec,           // caller indicates this call was intended for vectorized kernel
    /* 请求参数 */
    const char* input_path,
    const char* output_path,
    /* 输出统计 */
    unsigned long* time_us_out,
    size_t* output_size_out,
    /* optional detailed timings (microseconds) in a compact struct */
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_start = now_ns();

    // 1. 打开文件并读取头部
    FILE* f_in = fopen(input_path, "rb");
    if (!f_in) {
        perror("fopen input");
        return -1;
    }

    // 读取魔数
    uint16_t magic;
    if (fread(&magic, sizeof(magic), 1, f_in) != 1) {
        perror("fread magic");
        fclose(f_in);
        return -1;
    }
    if (magic != MAGIC) {
        fprintf(stderr, "[DECOMP] 错误的文件格式 (magic=0x%04x, 期望=0x%04x)\n", magic, MAGIC);
        fclose(f_in);
        return -1;
    }

    // 读取头部信息
    uint32_t orig_sz, blk_sz, nblk;
    if (fread(&orig_sz, sizeof(orig_sz), 1, f_in) != 1 ||
        fread(&blk_sz, sizeof(blk_sz), 1, f_in) != 1 ||
        fread(&nblk, sizeof(nblk), 1, f_in) != 1) {
        perror("fread header");
        fclose(f_in);
        return -1;
    }

    // 读取长度数组
    uint32_t* len_arr = malloc(nblk * sizeof(uint32_t));
    if (!len_arr) {
        perror("malloc len_arr");
        fclose(f_in);
        return -1;
    }
    if (fread(len_arr, sizeof(uint32_t), nblk, f_in) != nblk) {
        perror("fread len_arr");
        free(len_arr);
        fclose(f_in);
        return -1;
    }

    // 计算压缩数据大小
    long current_pos = ftell(f_in);
    fseek(f_in, 0, SEEK_END);
    long file_sz = ftell(f_in);
    fseek(f_in, current_pos, SEEK_SET);
    size_t comp_sz = file_sz - current_pos;

    /* Always print file summary so tests / logs have consistent output (not debug-only) */
    fprintf(stderr, "[DECOMP] 文件信息: 原始=%u, 块大小=%u, 块数=%u, 压缩数据=%zu\n",
            orig_sz, blk_sz, nblk, comp_sz);

    // 3. 计算偏移数组
    uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t));
    off_arr[0] = 0;
    for (uint32_t i = 0; i < nblk; ++i) {
        off_arr[i+1] = off_arr[i] + len_arr[i];
    }
    free(len_arr); // 长度数组不再需要

    // 4. Phase 7.2: 使用Buffer缓存 + Pinned Memory
    uint64_t t_buf_start = now_ns();

    cl_mem d_comp = get_or_create_buffer(ctx, &decomp_buffer_cache.d_comp,
                                         &decomp_buffer_cache.comp_size,
                                         comp_sz, CL_MEM_READ_ONLY, &err);
    CHECK(err);

    cl_mem d_off = get_or_create_buffer(ctx, &decomp_buffer_cache.d_off,
                                        &decomp_buffer_cache.off_size,
                                        (nblk + 1) * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
    CHECK(err);

    cl_mem d_out = get_or_create_buffer(ctx, &decomp_buffer_cache.d_out,
                                        &decomp_buffer_cache.out_size,
                                        orig_sz, CL_MEM_WRITE_ONLY, &err);
    CHECK(err);

    cl_mem d_out_lens = get_or_create_buffer(ctx, &decomp_buffer_cache.d_out_lens,
                                             &decomp_buffer_cache.lens_size,
                                             nblk * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err);
    CHECK(err);

    uint64_t t_buf_end = now_ns();

    // 5. Phase 6.1: 使用map上传数据 (零拷贝DMA)
    uint64_t t_upload_start = now_ns();

    // 直接读取压缩数据到Pinned Memory
    void* mapped_comp = clEnqueueMapBuffer(queue, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz,
                                           0, NULL, NULL, &err);
    CHECK(err);

    if (fread(mapped_comp, 1, comp_sz, f_in) != comp_sz) {
        perror("fread compressed data");
        clEnqueueUnmapMemObject(queue, d_comp, mapped_comp, 0, NULL, NULL);
        free(off_arr);
        fclose(f_in);
        return -1;
    }
    CHECK(clEnqueueUnmapMemObject(queue, d_comp, mapped_comp, 0, NULL, NULL));
    fclose(f_in); // 文件读取完成

    // 上传偏移数组
    void* mapped_off = clEnqueueMapBuffer(queue, d_off, CL_TRUE, CL_MAP_WRITE, 0,
                                          (nblk + 1) * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(mapped_off, off_arr, (nblk + 1) * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_off, mapped_off, 0, NULL, NULL));

    clFinish(queue);
    uint64_t t_upload_end = now_ns();

    // 6. 设置kernel参数
    uint64_t t_setup_start = now_ns();
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &nblk));
    uint64_t t_setup_end = now_ns();

    // 7. 执行kernel
    /* 优化: local_size默认使用8（与 standalone 保持一致），可通过环境变量 LZO_LOCAL_SIZE 覆盖
       使用较大的local_size能让解压向量化/标量kernel在同一work-group共享资源，提高吞吐 */
    size_t local_size = 8;
    const char* ls_env = getenv("LZO_LOCAL_SIZE");
    if (ls_env) {
        size_t v = (size_t)atoi(ls_env);
        if (v >= 1) local_size = v;
    }
    size_t global_size = ((size_t)nblk + local_size - 1) / local_size * local_size;
    uint64_t t_exec_start = now_ns();
    /* Always log kernel launch configuration so profiler traces are visible in daemon logs */
    fprintf(stderr, "[DECOMP] launching kernel: nblk=%u global_size=%zu local_size=%zu\n", nblk, global_size, local_size);
    cl_event evt_exec = NULL;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size,
                                0, NULL, &evt_exec);
    CHECK(err);
    /* 等待kernel完成并获取事件级精确时间（若队列启用了profiling） */
    clWaitForEvents(1, &evt_exec);
    clFinish(queue);  /* 确保所有GPU操作完成后再读取结果 */
    uint64_t t_exec_end = now_ns();

    /* 事件profiling信息（可选） */
    cl_ulong ev_start = 0, ev_end = 0;
    if (evt_exec) {
        cl_int perr = CL_SUCCESS;
        perr = clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_START, sizeof(ev_start), &ev_start, NULL);
        if (perr != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] profiling start info not available (err=%d)\n", perr);
        } else {
            cl_int perr2 = clGetEventProfilingInfo(evt_exec, CL_PROFILING_COMMAND_END, sizeof(ev_end), &ev_end, NULL);
            if (perr2 != CL_SUCCESS) {
                fprintf(stderr, "[DECOMP] profiling end info not available (err=%d)\n", perr2);
            }
        }
        fprintf(stderr, "[DECOMP] event profiling: start=%llu end=%llu\n", (unsigned long long)ev_start, (unsigned long long)ev_end);
        clReleaseEvent(evt_exec);
    } else {
        fprintf(stderr, "[DECOMP] no event returned from clEnqueueNDRangeKernel\n");
    }

    // 8. Phase 6.1: 使用map下载解压数据 (零拷贝DMA)
    /* 优化: 移除 out_buf 分配，直接从 mapped_out 写入文件 */
    // unsigned char* out_buf = malloc(orig_sz);

    /* 如果evt profiling 有效，使用它计算kernel执行时间（ns）否则回退到wall-clock */

    uint64_t t_download_start = now_ns();
    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, orig_sz,
                                          0, NULL, NULL, &err);
    CHECK(err);
    // memcpy(out_buf, mapped_out, orig_sz); // 移除memcpy
    uint64_t t_download_end = now_ns();

    // 9. 写入输出文件
    uint64_t t_write_start = now_ns();
    FILE* fout = fopen(output_path, "wb");
    if (!fout) {
        perror("fopen output");
        clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL);
        // free(out_buf);
        // free(lz_buf); // 已移除
        free(off_arr);
        return -1;
    }

    if (fwrite(mapped_out, 1, orig_sz, fout) != orig_sz) {
        perror("fwrite");
        fclose(fout);
        clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL);
        // free(out_buf);
        // free(lz_buf); // 已移除
        free(off_arr);
        return -1;
    }

    fclose(fout);

    /* Unmap after writing */
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));

    uint64_t t_write_end = now_ns();

    // 10. 清理输出buffer以避免GPU内存耗尽 (保留输入buffer缓存以提升性能)
    /* 解压缩的输出buffer每次都可能很大(800MB+)，如果缓存会导致GPU内存不足
     * 因此每次都释放输出buffer，只保留输入相关的buffer缓存 */
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

    // free(lz_buf); // 已移除
    free(off_arr);
    // free(out_buf);    // 已移除
    uint64_t t_end = now_ns();
    *time_us_out = (t_end - t_start) / 1000;
    *output_size_out = orig_sz;

    /* 计算各阶段耗时（微秒）*/
    unsigned long buf_us = (t_buf_end - t_buf_start) / 1000;
    unsigned long upload_us = (t_upload_end - t_upload_start) / 1000;
    unsigned long setup_us = (t_setup_end - t_setup_start) / 1000;
    unsigned long exec_host_us = (t_exec_end - t_exec_start) / 1000;
    unsigned long download_us = (t_download_end - t_download_start) / 1000;
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    /* 计算基于事件的内核时间（μs），如果事件不可用则为0 */
    unsigned long exec_us_ev = 0;
    if (ev_start != 0 && ev_end != 0 && ev_end > ev_start) {
        exec_us_ev = (unsigned long)((ev_end - ev_start) / 1000);
    }

    /* 输出解压缩统计信息 (always print for consistency) */
    double ratio = comp_sz > 0 ? (double)orig_sz / (double)comp_sz : 0.0;
    fprintf(stderr, "\n=== Decompression Statistics ===\n");
    fprintf(stderr, "Compressed size  : %zu bytes (%.2f MB)\n", comp_sz, comp_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Output size      : %u bytes (%.2f MB)\n", orig_sz, orig_sz / (1024.0 * 1024.0));
    fprintf(stderr, "Expansion ratio  : %.2f:1 (%.2f%% of compressed)\n", ratio, 100.0 * ratio);
    fprintf(stderr, "Block size       : %u bytes (%u KB)\n", blk_sz, blk_sz / 1024);
    fprintf(stderr, "Number of blocks : %u\n", nblk);
    fprintf(stderr, "Kernel           : %s (vectorized=%s)\n",
           prefer_vec ? "lzo1x_decomp_vec" : "lzo1x_decomp",
           prefer_vec ? "yes" : "no");
    fprintf(stderr, "Work groups      : global=%zu, local=%zu\n", global_size, local_size);
    double kernel_thrpt = exec_host_us > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (exec_host_us/1000000.0) : 0.0;
    fprintf(stderr, "Throughput       : %.2f MB/s (kernel: %.2f MB/s)\n",
           ((double)orig_sz / (1024.0*1024.0)) / (*time_us_out/1000000.0),
           kernel_thrpt);
    fprintf(stderr, "==============================\n\n");    /* 打印详细的时间分解（与standalone格式一致）*/
    fprintf(stderr, "\n=== Time Breakdown (Decompression) ===\n");
    fprintf(stderr, "1. Buffer Alloc      : %8.3f ms\n", buf_us / 1000.0);
    fprintf(stderr, "2. Data Upload       : %8.3f ms\n", upload_us / 1000.0);
    fprintf(stderr, "3. Setup Args        : %8.3f ms\n", setup_us / 1000.0);
    fprintf(stderr, "4. Kernel Exec       : %8.3f ms\n", exec_host_us / 1000.0);
    if (exec_us_ev) {
        fprintf(stderr, "   (event profiling) : %8.3f ms\n", exec_us_ev / 1000.0);
    }
    fprintf(stderr, "5. Data Download     : %8.3f ms\n", download_us / 1000.0);
    fprintf(stderr, "6. File Write        : %8.3f ms\n", write_us / 1000.0);
    fprintf(stderr, "TOTAL                : %8.3f ms\n", *time_us_out / 1000.0);
    fprintf(stderr, "\n");

        /* 计算占比（与standalone格式一致），保护除以零 */
        double denom = (*time_us_out > 0) ? (double)*time_us_out : 1.0;
        int zero_total = (*time_us_out == 0);
        fprintf(stderr, "=== Percentage Breakdown ===\n");
        fprintf(stderr, "Kernel Exec     : %6.2f%%\n", zero_total ? 0.0 : 100.0 * exec_host_us / denom);
        fprintf(stderr, "Data Transfer   : %6.2f%% (upload=%.2f%% + download=%.2f%%)\n",
            zero_total ? 0.0 : 100.0 * (upload_us + download_us) / denom,
            zero_total ? 0.0 : 100.0 * upload_us / denom,
            zero_total ? 0.0 : 100.0 * download_us / denom);
        fprintf(stderr, "File I/O        : %6.2f%% (write=%.2f%%)\n",
            zero_total ? 0.0 : 100.0 * write_us / denom,
            zero_total ? 0.0 : 100.0 * write_us / denom);
        fprintf(stderr, "Buffer Alloc    : %6.2f%%\n",
            zero_total ? 0.0 : 100.0 * buf_us / denom);
        fprintf(stderr, "Setup Args      : %6.2f%%\n",
            zero_total ? 0.0 : 100.0 * setup_us / denom);
    fprintf(stderr, "\n");

    /* Prefer event-based execution time if available, otherwise use host wall-clock kernel time */
    unsigned long kernel_us = exec_us_ev ? exec_us_ev : exec_host_us;

    /* map computed timing pieces into the compact timing_t (if requested) */
    if (t_out) {
        t_out->file_read_us = (t_buf_start - t_start) / 1000; /* time spent before buffer alloc */
        t_out->ocl_init_us = 0; /* daemon typically has OCL init at startup */
        t_out->kernel_load_us = 0; /* kernel load usually cached at daemon startup */
        t_out->buffer_alloc_in_us = buf_us;
        t_out->data_upload_us = upload_us;
        t_out->setup_args_us = setup_us;
        t_out->kernel_exec_us = kernel_us;
        t_out->download_total_us = download_us;
        t_out->file_write_us = write_us;
        /* fields not applicable for decompression - set to 0 to maintain deterministic layout */
        t_out->blocking_calc_us = 0;
        t_out->buffer_alloc_out_us = 0;
        t_out->buffer_alloc_len_us = 0;
        t_out->kernel_setup_us = 0;
        t_out->download_len_us = 0;
        t_out->download_bulk_us = 0;
        t_out->cleanup_us = 0;
    }

    return 0;
}

/* 清理解压缩buffer缓存 - 外部可调用以在daemon关闭时释放资源 */
void cleanup_decompress_buffer_cache(void) {
    if (decomp_buffer_cache.d_comp) clReleaseMemObject(decomp_buffer_cache.d_comp);
    if (decomp_buffer_cache.d_off) clReleaseMemObject(decomp_buffer_cache.d_off);
    if (decomp_buffer_cache.d_out) clReleaseMemObject(decomp_buffer_cache.d_out);
    if (decomp_buffer_cache.d_out_lens) clReleaseMemObject(decomp_buffer_cache.d_out_lens);
    decomp_buffer_cache.d_comp = NULL;
    decomp_buffer_cache.d_off = NULL;
    decomp_buffer_cache.d_out = NULL;
    decomp_buffer_cache.d_out_lens = NULL;
    decomp_buffer_cache.comp_size = 0;
    decomp_buffer_cache.off_size = 0;
    decomp_buffer_cache.out_size = 0;
    decomp_buffer_cache.lens_size = 0;
}
