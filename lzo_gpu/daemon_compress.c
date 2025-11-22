/*
 * daemon_compress.c - 守护进程压缩核心逻辑
 * 从lzo_host.c提取,用于守护进程复用OpenCL资源
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CHECK(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

#define D_BITS 11
#define OCC_FACTOR 128  // 大幅增加并行因子，让GPU有足够的work-items
#define ALIGN_BYTES 65536
#define MIN_BLOCK_SIZE (16 * 1024)   // 降到16KB，最大化并行度
#define MAX_BLOCK_SIZE (128 * 1024)  // 降到128KB

/* 与lzo_host.c相同的辅助函数 */
static inline size_t lzo_worst(size_t sz) {
    return sz + sz / 16 + 64 + 3;
}

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

/* 写入压缩文件 */
static int write_compressed_file(const char* path, const void* data,
                                 size_t orig_size, size_t blk_size,
                                 size_t nblk, const unsigned int* lens,
                                 const void* comp_data, size_t comp_total) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    // 写入头部
    unsigned short magic = 0x4C5A;  // 'LZ'
    fwrite(&magic, sizeof(magic), 1, f);

    unsigned int u32;
    u32 = (unsigned int)orig_size;
    fwrite(&u32, sizeof(u32), 1, f);
    u32 = (unsigned int)blk_size;
    fwrite(&u32, sizeof(u32), 1, f);
    u32 = (unsigned int)nblk;
    fwrite(&u32, sizeof(u32), 1, f);

    // 写入长度数组
    fwrite(lens, sizeof(unsigned int), nblk, f);

    // 写入压缩数据
    fwrite(comp_data, 1, comp_total, f);

    fclose(f);
    return 0;
}

/* 动态块选择 (与lzo_host.c相同) */
static void choose_blocking(size_t in_sz, cl_device_id dev,
                           size_t* blk_sz_out, size_t* nblk_out) {
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    size_t tgt_blk = (size_t)cu * OCC_FACTOR;
    if (tgt_blk > in_sz) tgt_blk = in_sz;

    size_t blk = (in_sz + tgt_blk - 1) / tgt_blk;
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    size_t nblk = (in_sz + blk - 1) / blk;
    if (nblk > 1 && blk < MIN_BLOCK_SIZE) {
        blk = MIN_BLOCK_SIZE;
        nblk = (in_sz + blk - 1) / blk;
    }
    if (blk > MAX_BLOCK_SIZE) {
        blk = MAX_BLOCK_SIZE;
        nblk = (in_sz + blk - 1) / blk;
    }
    if (nblk == 1) {
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
        if (blk < MIN_BLOCK_SIZE) blk = MIN_BLOCK_SIZE;
        if (blk > MAX_BLOCK_SIZE) blk = MAX_BLOCK_SIZE;
    }

    *blk_sz_out = blk;
    *nblk_out = nblk;
}

/*
 * 守护进程压缩函数
 * 复用预分配的OpenCL资源,仅执行必要的压缩操作
 *
 * level参数映射:
 *   1-3: lzo1x_1  (标准压缩)
 *   4-6: lzo1x_1k (1KB优化)
 *   7-8: lzo1x_1l (轻量级)
 *   9:   lzo1x_1o (最优压缩)
 */
int daemon_compress(
    /* OpenCL资源 (已初始化,复用) */
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,         // 根据level选择的kernel
    /* 请求参数 */
    const char* input_path,
    const char* output_path,
    int level,                // 压缩级别 1-9
    /* 输出统计 */
    unsigned long* time_us_out,  // 总时间(微秒)
    size_t* output_size_out,
    /* 详细时间输出(微秒) */
    unsigned long* read_us_out,
    unsigned long* buffer_us_out,
    unsigned long* upload_us_out,
    unsigned long* kernel_us_out,
    unsigned long* download_us_out,
    unsigned long* write_us_out,
    unsigned long* cleanup_us_out
) {
    cl_int err;
    uint64_t t_total_start = now_ns();

    // 1. 读取输入文件
    uint64_t t_read_start = now_ns();
    size_t in_sz;
    unsigned char* in_buf = read_file_data(input_path, &in_sz);
    if (!in_buf) {
        return -1;
    }
    uint64_t t_read_end = now_ns();
    unsigned long read_us = (t_read_end - t_read_start) / 1000;

    // 2. 确定分块策略
    size_t blk, nblk;
    choose_blocking(in_sz, device, &blk, &nblk);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t in_needed = nblk * blk;
    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    // 3. 动态创建缓冲区 (每次请求)
    uint64_t t_buf_start = now_ns();
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, in_needed, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建输入缓冲区失败: %d\n", err);
        free(in_buf);
        return -1;
    }

    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_needed, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建输出缓冲区失败: %d\n", err);
        clReleaseMemObject(d_in);
        free(in_buf);
        return -1;
    }

    cl_mem d_len = clCreateBuffer(ctx, CL_MEM_READ_WRITE, len_needed, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "创建长度缓冲区失败: %d\n", err);
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        free(in_buf);
        return -1;
    }

    uint64_t t_buf_end = now_ns();
    unsigned long buf_create_us = (t_buf_end - t_buf_start) / 1000;

    // 4. 上传数据到缓冲区
    uint64_t t_upload_start = now_ns();
    // 优化: 直接上传原始数据,不需要padding
    // Kernel只读取前in_sz字节,后面的padding不影响结果
    CHECK(clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0,
                              in_sz, in_buf, 0, NULL, NULL));
    clFinish(queue);  // 确保上传完成

    uint64_t t_upload_end = now_ns();
    unsigned long upload_us = (t_upload_end - t_upload_start) / 1000;

    // 5. 设置内核参数
    uint64_t t_kernel_start = now_ns();
    // 5a. 转换为unsigned int (kernel参数要求)
    unsigned int in_sz_u = (unsigned int)in_sz;
    unsigned int blk_u = (unsigned int)blk;
    unsigned int worst_blk_u = (unsigned int)worst_blk;

    // 5b. 清零长度缓冲区
    cl_uint* zero_buffer = calloc(nblk, sizeof(cl_uint));
    CHECK(clEnqueueWriteBuffer(queue, d_len, CL_TRUE, 0, nblk * sizeof(cl_uint),
                              zero_buffer, 0, NULL, NULL));
    free(zero_buffer);

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(unsigned int), &in_sz_u));
    CHECK(clSetKernelArg(kernel, 4, sizeof(unsigned int), &blk_u));
    CHECK(clSetKernelArg(kernel, 5, sizeof(unsigned int), &worst_blk_u));
    clFinish(queue);  // 确保参数设置完成

    uint64_t t_kernel_end = now_ns();
    unsigned long kernel_setup_us = (t_kernel_end - t_kernel_start) / 1000;

    // 6. 执行内核
    uint64_t t_exec_start = now_ns();
    size_t gsz = nblk;
    cl_event evt;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, NULL,
                                 0, NULL, &evt);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] Kernel execution failed: %d\n", err);
        free(in_buf);
        return -1;
    }
    err = clWaitForEvents(1, &evt);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] clWaitForEvents failed: %d\n", err);

        // 获取详细的执行状态
        cl_int exec_status;
        clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &exec_status, NULL);
        fprintf(stderr, "[ERROR] Event execution status: %d\n", exec_status);

        clReleaseEvent(evt);
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_len);
        free(in_buf);
        return -1;
    }

    // 检查kernel执行状态
    cl_int exec_status;
    clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &exec_status, NULL);
    if (exec_status < 0) {
        fprintf(stderr, "[ERROR] Kernel execution failed with status: %d\n", exec_status);
        clReleaseEvent(evt);
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_len);
        free(in_buf);
        return -1;
    }

    clReleaseEvent(evt);
    clFinish(queue);  // 确保kernel真正执行完成

    uint64_t t_exec_end = now_ns();
    unsigned long kernel_exec_us = (t_exec_end - t_exec_start) / 1000;

    // 7. 读取长度数组
    uint64_t t_download_start = now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    void* mapped_len = clEnqueueMapBuffer(queue, d_len, CL_TRUE,
                                         CL_MAP_READ, 0, nblk * sizeof(cl_uint),
                                         0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_len);
        free(in_buf);
        free(len_arr);
        return -1;
    }
    memcpy(len_arr, mapped_len, nblk * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_len, mapped_len, 0, NULL, NULL));

    // 8. 计算总压缩大小
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) {
        comp_total += len_arr[i];
    }

    // 9. 读取压缩数据
    unsigned char* comp_buf = malloc(comp_total);
    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE,
                                         CL_MAP_READ, 0, comp_total,
                                         0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_out);
        clReleaseMemObject(d_len);
        free(in_buf);
        free(len_arr);
        free(comp_buf);
        return -1;
    }

    // 重新组织输出数据 (按块拷贝)
    size_t offset = 0;
    for (size_t i = 0; i < nblk; i++) {
        memcpy(comp_buf + offset,
               (unsigned char*)mapped_out + i * worst_blk,
               len_arr[i]);
        offset += len_arr[i];
    }
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));
    clFinish(queue);  // 确保下载完成

    uint64_t t_download_end = now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    // 10. 写入输出文件
    uint64_t t_write_start = now_ns();
    int ret = write_compressed_file(output_path, NULL, in_sz, blk, nblk,
                                    len_arr, comp_buf, comp_total);

    uint64_t t_write_end = now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    // 清理
    uint64_t t_cleanup_start = now_ns();
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseMemObject(d_len);
    free(in_buf);
    free(len_arr);
    free(comp_buf);

    uint64_t t_cleanup_end = now_ns();
    unsigned long cleanup_us = (t_cleanup_end - t_cleanup_start) / 1000;

    uint64_t t_end = now_ns();
    *time_us_out = (t_end - t_total_start) / 1000;  // 使用t_total_start而非t_start
    *output_size_out = comp_total;

    // 填充详细时间输出
    *read_us_out = read_us;
    *buffer_us_out = buf_create_us;
    *upload_us_out = upload_us;
    *kernel_us_out = kernel_setup_us + kernel_exec_us;  // 合并kernel时间
    *download_us_out = download_us;
    *write_us_out = write_us;
    *cleanup_us_out = cleanup_us;

    // 打印详细时间分解 (包含所有环节,微秒)
    fprintf(stderr, "[TIMING] 总耗时=%luμs (%.2fms): 读文件=%luμs, 缓冲区=%luμs, 上传=%luμs, Kernel设置=%luμs, Kernel执行=%luμs, 下载=%luμs, 写文件=%luμs, 清理=%luμs\n",
            *time_us_out, *time_us_out/1000.0, read_us, buf_create_us, upload_us, kernel_setup_us, kernel_exec_us, download_us, write_us, cleanup_us);

    return ret;
}
