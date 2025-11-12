/*
 * build_kernel.c
 * Small utility to build an OpenCL program from source and write the
 * device binary to a file. Usage:
 *   ./build_kernel <source.cl> <out.bin>
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* read_file(const char* path, size_t* sz_out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = malloc(sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, sz, fp) != (size_t)sz) { free(buf); fclose(fp); return NULL; }
    buf[sz] = '\0';
    fclose(fp);
    if (sz_out) *sz_out = (size_t)sz;
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <source.cl> <out.bin>\n", argv[0]);
        return 2;
    }
    const char* src_path = argv[1];
    const char* out_path = argv[2];

    cl_int err;
    cl_platform_id pf;
    if (clGetPlatformIDs(1, &pf, NULL) != CL_SUCCESS) { fprintf(stderr, "no OpenCL platform\n"); return 1; }
    cl_device_id dev;
    if (clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) {
        if (clGetDeviceIDs(pf, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { fprintf(stderr, "no OpenCL device\n"); return 1; }
    }

    size_t src_len; char* src = read_file(src_path, &src_len);
    if (!src) { fprintf(stderr, "failed to read source %s\n", src_path); return 1; }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (!ctx || err != CL_SUCCESS) { fprintf(stderr, "clCreateContext failed: %d\n", err); free(src); return 1; }

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
    if (!prog || err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed: %d\n", err); clReleaseContext(ctx); free(src); return 1; }

    /* request OpenCL 2.0/C2.0 for kernels that use generic address space */
    const char* build_opts = "-cl-std=CL2.0";
    err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = malloc(log_sz + 1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        log[log_sz] = '\0';
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        clReleaseProgram(prog); clReleaseContext(ctx); free(src);
        return 1;
    }

    /* query binary sizes (we only keep the first device binary to avoid redundant allocations) */
    size_t num_devices = 0;
    clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &num_devices, NULL);
    if (num_devices == 0) { fprintf(stderr, "no devices for program\n"); clReleaseProgram(prog); clReleaseContext(ctx); free(src); return 1; }
    if (num_devices > 1) {
        fprintf(stderr, "Note: program built for %zu devices, will write binary for device 0 only\n", num_devices);
        num_devices = 1; /* only handle first device to avoid redundant allocations */
    }

    size_t bin_size = 0;
    clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, NULL);
    unsigned char* bin = malloc(bin_size);
    unsigned char* bin_ptr = bin;
    clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin_ptr, NULL);

    /* write the first binary to out_path */
    FILE* fo = fopen(out_path, "wb");
    if (!fo) { perror(out_path); free(bin); clReleaseProgram(prog); clReleaseContext(ctx); free(src); return 1; }
    if (fwrite(bin, 1, bin_size, fo) != bin_size) { perror("fwrite"); fclose(fo); free(bin); clReleaseProgram(prog); clReleaseContext(ctx); free(src); return 1; }
    fclose(fo);
    free(bin);
    clReleaseProgram(prog); clReleaseContext(ctx); free(src);

    printf("Wrote kernel binary to %s\n", out_path);
    return 0;
}
