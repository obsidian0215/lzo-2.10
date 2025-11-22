/*
 * daemon_decompress.c - 守护进程解压缩核心逻辑
 */

#include <CL/cl.h>
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
 */
int daemon_decompress(
    /* OpenCL资源 (已初始化,复用) */
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,         // 解压缩kernel
    /* 请求参数 */
    const char* input_path,
    const char* output_path,
    /* 输出统计 */
    unsigned long* time_us_out,
    size_t* output_size_out
) {
    cl_int err;
    uint64_t t_start = now_ns();

    // 1. 读取压缩文件
    size_t lz_sz;
    unsigned char* lz_buf = read_file_data(input_path, &lz_sz);
    if (!lz_buf) {
        fprintf(stderr, "[DECOMP] 读取文件失败: %s\n", input_path);
        return -1;
    }

    // 2. 解析LZO文件头
    unsigned char* p = lz_buf;
    uint16_t magic = *(uint16_t*)p; p += 2;
    if (magic != MAGIC) {
        fprintf(stderr, "[DECOMP] 错误的文件格式 (magic=0x%04x, 期望=0x%04x)\n", magic, MAGIC);
        free(lz_buf);
        return -1;
    }

    uint32_t orig_sz = *(uint32_t*)p; p += 4;
    uint32_t blk_sz = *(uint32_t*)p; p += 4;
    uint32_t nblk = *(uint32_t*)p; p += 4;
    uint32_t* len_arr = (uint32_t*)p; p += 4 * nblk;
    size_t comp_sz = lz_sz - (p - lz_buf);

    printf("[DECOMP] 文件信息: 原始=%u, 块大小=%u, 块数=%u, 压缩数据=%zu\n",
           orig_sz, blk_sz, nblk, comp_sz);

    // 3. 计算偏移数组
    uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t));
    off_arr[0] = 0;
    for (uint32_t i = 0; i < nblk; ++i) {
        off_arr[i+1] = off_arr[i] + len_arr[i];
    }

    // 4. 创建OpenCL缓冲区
    uint64_t t_buf_start = now_ns();

    cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY, comp_sz, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DECOMP] 创建压缩数据缓冲区失败: %d\n", err);
        free(lz_buf);
        free(off_arr);
        return -1;
    }

    cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (nblk + 1) * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DECOMP] 创建偏移缓冲区失败: %d\n", err);
        clReleaseMemObject(d_comp);
        free(lz_buf);
        free(off_arr);
        return -1;
    }

    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, orig_sz, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DECOMP] 创建输出缓冲区失败: %d\n", err);
        clReleaseMemObject(d_comp);
        clReleaseMemObject(d_off);
        free(lz_buf);
        free(off_arr);
        return -1;
    }

    cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[DECOMP] 创建输出长度缓冲区失败: %d\n", err);
        clReleaseMemObject(d_comp);
        clReleaseMemObject(d_off);
        clReleaseMemObject(d_out);
        free(lz_buf);
        free(off_arr);
        return -1;
    }

    uint64_t t_buf_end = now_ns();

    // 5. 上传数据
    uint64_t t_upload_start = now_ns();
    err = clEnqueueWriteBuffer(queue, d_comp, CL_FALSE, 0, comp_sz, p, 0, NULL, NULL);
    CHECK(err);

    err = clEnqueueWriteBuffer(queue, d_off, CL_FALSE, 0, (nblk + 1) * sizeof(cl_uint),
                               off_arr, 0, NULL, NULL);
    CHECK(err);

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
    size_t global_size = nblk;
    size_t local_size = 1;
    uint64_t t_exec_start = now_ns();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size,
                                0, NULL, NULL);
    CHECK(err);
    clFinish(queue);
    uint64_t t_exec_end = now_ns();

    // 8. 下载解压数据
    unsigned char* out_buf = malloc(orig_sz);
    if (!out_buf) {
        fprintf(stderr, "[DECOMP] 分配输出缓冲区失败\n");
        goto cleanup;
    }

    uint64_t t_download_start = now_ns();
    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, orig_sz, out_buf, 0, NULL, NULL);
    CHECK(err);
    uint64_t t_download_end = now_ns();

    // 9. 写入输出文件
    uint64_t t_write_start = now_ns();
    FILE* fout = fopen(output_path, "wb");
    if (!fout) {
        perror("fopen output");
        free(out_buf);
        goto cleanup;
    }

    if (fwrite(out_buf, 1, orig_sz, fout) != orig_sz) {
        perror("fwrite");
        fclose(fout);
        free(out_buf);
        goto cleanup;
    }

    fclose(fout);
    uint64_t t_write_end = now_ns();

    // 10. 清理
    uint64_t t_cleanup_start = now_ns();
    clReleaseMemObject(d_comp);
    clReleaseMemObject(d_off);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_out_lens);
    uint64_t t_cleanup_end = now_ns();

    free(lz_buf);
    free(off_arr);
    free(out_buf);

    uint64_t t_end = now_ns();

    // 11. 输出统计
    unsigned long t_buf = (t_buf_end - t_buf_start) / 1000000;
    unsigned long t_upload = (t_upload_end - t_upload_start) / 1000000;
    unsigned long t_setup = (t_setup_end - t_setup_start) / 1000000;
    unsigned long t_exec = (t_exec_end - t_exec_start) / 1000000;
    unsigned long t_download = (t_download_end - t_download_start) / 1000000;
    unsigned long t_write = (t_write_end - t_write_start) / 1000000;
    unsigned long t_cleanup = (t_cleanup_end - t_cleanup_start) / 1000000;
    unsigned long t_total = (t_end - t_start) / 1000000;

    printf("[TIMING] 总耗时=%lums: 缓冲区创建=%lums, 上传=%lums, Kernel设置=%lums, "
           "Kernel执行=%lums, 下载=%lums, 写文件=%lums, 清理=%lums\n",
           t_total, t_buf, t_upload, t_setup, t_exec, t_download, t_write, t_cleanup);

    *time_us_out = t_total * 1000;  // 纳秒转微秒
    *output_size_out = orig_sz;

    return 0;

cleanup:
    clReleaseMemObject(d_comp);
    clReleaseMemObject(d_off);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_out_lens);
    free(lz_buf);
    free(off_arr);
    return -1;
}
