#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
#if defined(CL_VERSION_2_0)
    {
        const cl_queue_properties qprops[] = {
            CL_QUEUE_PROPERTIES,
            0,
            0
        };
        queue = clCreateCommandQueueWithProperties(context, device, qprops, NULL);
    }
#else
    queue = clCreateCommandQueue(context, device, 0, NULL);
#endif

    // 测试不同大小的内存拷贝带宽
    size_t sizes[] = {1<<20, 1<<22, 1<<24, 1<<26}; // 1MB, 4MB, 16MB, 64MB

    printf("GPU Memory Bandwidth Test\n");
    printf("==========================\n");

    for (int i = 0; i < 4; i++) {
        size_t size = sizes[i];
        char *host_data = malloc(size);
        for (size_t j = 0; j < size; j++) host_data[j] = j & 0xFF;

        cl_mem dev_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

        // 测试上传带宽
        double t0 = get_time();
        for (int trial = 0; trial < 10; trial++) {
            clEnqueueWriteBuffer(queue, dev_buf, CL_TRUE, 0, size, host_data, 0, NULL, NULL);
        }
        double t1 = get_time();
        double upload_bw = (size * 10.0) / (t1 - t0) / 1e6;

        // 测试下载带宽
        t0 = get_time();
        for (int trial = 0; trial < 10; trial++) {
            clEnqueueReadBuffer(queue, dev_buf, CL_TRUE, 0, size, host_data, 0, NULL, NULL);
        }
        t1 = get_time();
        double download_bw = (size * 10.0) / (t1 - t0) / 1e6;

        printf("Size: %3zu MB | Upload: %7.2f MB/s | Download: %7.2f MB/s\n",
               size >> 20, upload_bw, download_bw);

        clReleaseMemObject(dev_buf);
        free(host_data);
    }

    // 测试 GPU 内部拷贝带宽 (kernel)
    const char *kernel_src =
        "__kernel void memcpy_test(__global const uchar *src, __global uchar *dst, uint n) {\n"
        "    uint gid = get_global_id(0);\n"
        "    if (gid < n) dst[gid] = src[gid];\n"
        "}\n"
        "__kernel void memcpy_vec16(__global const uchar *src, __global uchar *dst, uint n) {\n"
        "    uint gid = get_global_id(0) * 16;\n"
        "    if (gid + 16 <= n) {\n"
        "        uchar16 v = vload16(0, src + gid);\n"
        "        vstore16(v, 0, dst + gid);\n"
        "    }\n"
        "}\n";

    cl_program prog = clCreateProgramWithSource(context, 1, &kernel_src, NULL, NULL);
    clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    cl_kernel k_scalar = clCreateKernel(prog, "memcpy_test", NULL);
    cl_kernel k_vec16 = clCreateKernel(prog, "memcpy_vec16", NULL);

    printf("\nGPU Internal Copy Bandwidth\n");
    printf("============================\n");

    for (int i = 0; i < 4; i++) {
        size_t size = sizes[i];
        cl_mem src = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);
        cl_mem dst = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

        // 标量拷贝
        clSetKernelArg(k_scalar, 0, sizeof(cl_mem), &src);
        clSetKernelArg(k_scalar, 1, sizeof(cl_mem), &dst);
        cl_uint n = size;
        clSetKernelArg(k_scalar, 2, sizeof(cl_uint), &n);

        size_t global = size;
        double t0 = get_time();
        for (int trial = 0; trial < 10; trial++) {
            clEnqueueNDRangeKernel(queue, k_scalar, 1, NULL, &global, NULL, 0, NULL, NULL);
        }
        clFinish(queue);
        double t1 = get_time();
        double scalar_bw = (size * 10.0) / (t1 - t0) / 1e6;

        // 向量拷贝
        clSetKernelArg(k_vec16, 0, sizeof(cl_mem), &src);
        clSetKernelArg(k_vec16, 1, sizeof(cl_mem), &dst);
        clSetKernelArg(k_vec16, 2, sizeof(cl_uint), &n);

        global = size / 16;
        t0 = get_time();
        for (int trial = 0; trial < 10; trial++) {
            clEnqueueNDRangeKernel(queue, k_vec16, 1, NULL, &global, NULL, 0, NULL, NULL);
        }
        clFinish(queue);
        t1 = get_time();
        double vec16_bw = (size * 10.0) / (t1 - t0) / 1e6;

        printf("Size: %3zu MB | Scalar: %7.2f MB/s | Vec16: %7.2f MB/s (%.2fx)\n",
               size >> 20, scalar_bw, vec16_bw, vec16_bw / scalar_bw);

        clReleaseMemObject(src);
        clReleaseMemObject(dst);
    }

    clReleaseKernel(k_scalar);
    clReleaseKernel(k_vec16);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
