#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
* 压缩文件格式：
uint16  magic     = 0x4C5A   // 'L''Z'
uint32  orig_size               (≤4 GiB)
uint32  blk_size
uint32  nblk
uint32  len[nblk]               // 每块压缩长度
-----   nblk 个压缩块数据
*/
#define MAGIC  0x4C5A   /* 'L''Z' */
#define D_BITS          11
//#define BLK_SIZE        (32 * 1024)
#define OCC_FACTOR        4             /* 每个 CU 期望 OCC_FACTOR 个块，保证高占用 */
#define ALIGN_BYTES       256           /* 向 256 B 对齐，便于内存访问 */

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline uint64_t now_ns(void)
{
    static LARGE_INTEGER freq = { 0 };
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    /*   counter / freq = 秒
     * → counter * 1e9 / freq = 纳秒
     */
    return (uint64_t)counter.QuadPart * (uint64_t)1000000000ULL /
        (uint64_t)freq.QuadPart;
}

#else
#include <time.h>

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

static inline void print_ns(const char* tag, uint64_t ns) {
    printf("%-22s : %8.3f ms\n", tag, ns / 1e6);
}

static void choose_blocking(size_t in_sz, cl_device_id dev,
    size_t* blk_sz_out, size_t* nblk_out)
{
    /* 1. 取 GPU 计算单元数 */
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* 2. ⽬标块数：CU × OCC_FACTOR，但不能多于字节数 */
    size_t tgt_blk = (size_t)cu * OCC_FACTOR;
    if (tgt_blk > in_sz) tgt_blk = in_sz;        /* 每块≥1 B */

    /* 3. 初步块⼤⼩ = ceil(in_sz / tgt_blk) */
    size_t blk = (in_sz + tgt_blk - 1) / tgt_blk;

    /* 4. 向 ALIGN_BYTES 对齐，且不得为 0 */
    blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    if (blk == 0) blk = ALIGN_BYTES;

    /* 5. 重新得到块数；如仍 < CU，则再细分以 nblk = CU 为下限 */
    size_t nblk = (in_sz + blk - 1) / blk;
    if (nblk < cu) {
        nblk = cu;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    }

    /* 6. 尾块太⼩（< blk/4）时，把数据平均回各块 */
    size_t tail = in_sz - blk * (nblk - 1);
    if (nblk > 1 && tail < blk / 4) {
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (ALIGN_BYTES - 1)) & ~(ALIGN_BYTES - 1);
    }

    *blk_sz_out = blk;
    *nblk_out = (in_sz + blk - 1) / blk;
}

#define CHECK(expr)  do{ cl_int _e=(expr);                       \
        if(_e!=CL_SUCCESS){                                      \
            fprintf(stderr,"OpenCL error %d at %s:%d\n",         \
                    _e,__FILE__,__LINE__); exit(1);} }while(0)

static inline size_t lzo_worst(size_t n) {
    return n + n / 16 + 64 + 3;
}

/* read mem-images/kernel-source */
static char* read_file(const char* path, size_t* sz_out)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror(path); exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* buf = malloc(sz + 1);
    fread(buf, 1, sz, fp);
    fclose(fp);

    if (sz_out)
        *sz_out = (size_t)sz;
    buf[sz] = '\0';
    return buf;
}

static cl_context  ctx;
static cl_command_queue q;
static cl_device_id dev;

static void ocl_init(void)
{
    cl_platform_id pf;
    CHECK(clGetPlatformIDs(1, &pf, NULL));
    CHECK(clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    q = clCreateCommandQueueWithProperties(ctx, dev, props, NULL);
}

void print_buildlog(cl_program program, cl_device_id device) {
    char* buff_erro;
    cl_int errcode;
    size_t build_log_len;
    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-1);
    }
    buff_erro = malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr, "Build log: \n%s\n", buff_erro); //Be careful with fprint
    free(buff_erro);
    fprintf(stderr, "clBuildProgram failed\n");
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("usage: %s input_file [compressed.out]\n", argv[0]);
        return 0;
    }
    const char* cl_path = "compress.cl";
    const char* in_path = argv[1];
    const char* out_path = (argc >= 3) ? argv[2] : "compressed.out";
    cl_int err;

    /* Decompress path: argv[1] == "-d"
    usage: lzo_gpu -d input.lzo original_file [decompressed.out]
     */
    if (argc >= 2 && strcmp(argv[1], "-d") == 0) {
		if (argc < 4) {
			fprintf(stderr, "usage: %s -d input.lzo original_file [decompressed.out]\n", argv[0]);
			return 1;
		}
        const char* lz_path = argv[2];
        const char* orig_path = argv[3];
        const char* out_path = (argc >= 5) ? argv[4] : "decompressed.out";

        /* A. 读取压缩文件全部内容 --------------- */
        size_t lz_sz; unsigned char* lz_buf = read_file(lz_path, &lz_sz);

        const unsigned char* p = lz_buf;
        uint16_t magic = *(uint16_t*)p; p += 2;
        if (magic != MAGIC) { fprintf(stderr, "bad magic\n"); exit(1); }

        uint32_t orig_sz = *(uint32_t*)p;   p += 4;
        uint32_t blk_sz = *(uint32_t*)p;   p += 4;
        uint32_t nblk = *(uint32_t*)p;   p += 4;

        uint32_t* len_arr = (uint32_t*)p;   p += 4 * nblk;

        size_t comp_sz = lz_sz - (p - lz_buf);   /* 全部压缩数据字节数 */

        /* 预计算 prefix offset[nblk+1]，少一次 kernel 前缀和 */
        uint32_t* off_arr = malloc((nblk + 1) * 4);
        off_arr[0] = 0;
        for (uint32_t i = 0; i < nblk; ++i) off_arr[i + 1] = off_arr[i] + len_arr[i];

        /* B. OpenCL 初始化 & 构建 kernel --------- */
        ocl_init();
        size_t src_len; char* cl_src = read_file(cl_path, &src_len);
        cl_program prog = clCreateProgramWithSource(
            ctx, 1, (const char**)&cl_src, &src_len, &err); CHECK(err);
        err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
        if (err) print_buildlog(prog, dev);

        cl_kernel krn = clCreateKernel(prog, "lzo1x_block_decompress", &err); CHECK(err);

        /* C. 创建缓冲区并上传 ------------------- */
        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            comp_sz, (void*)p, &err); CHECK(err);
        cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            (nblk + 1) * 4, off_arr, &err); CHECK(err);
        cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            orig_sz, NULL, &err); CHECK(err);

        /* D. 设置 kernel 参数 ------------------- */
        CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &d_in));
        CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &d_off));
        CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &d_out));
        CHECK(clSetKernelArg(krn, 3, sizeof(cl_uint), &blk_sz));
        CHECK(clSetKernelArg(krn, 4, sizeof(cl_uint), &orig_sz));

        size_t gsz = nblk, lsz = 1;  cl_event evt;
        uint64_t t0 = now_ns();
        CHECK(clEnqueueNDRangeKernel(q, krn, 1, NULL, &gsz, &lsz, 0, NULL, &evt));
        clWaitForEvents(1, &evt);
        uint64_t t1 = now_ns();

        /* E. 取回解压结果 ----------------------- */
        unsigned char* out_buf = malloc(orig_sz);
        CHECK(clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, orig_sz, out_buf, 0, NULL, NULL));

        /* F. 与原文件比对 ----------------------- */
        size_t ref_sz; unsigned char* ref = read_file(orig_path, &ref_sz);
        if (ref_sz != orig_sz || memcmp(ref, out_buf, orig_sz) != 0) {
            fprintf(stderr, "decompress verify FAILED!\n"); exit(1);
        }
        puts("verify OK");

        /* G. 输出性能 --------------------------- */
        double ms = (t1 - t0) / 1e6;
        double mb = (double)orig_sz / 1e6;
        printf("Decompress throughput: %.2f MB/s (%.3f ms for %u B)\n",
            mb / (ms / 1000.0), ms, orig_sz);

        /* H. 可选写文件 ------------------------- */
        FILE* fo = fopen(out_path, "wb");
        fwrite(out_buf, 1, orig_sz, fo); fclose(fo);

        /* I. 清理 ------------------------------- */
        free(cl_src); free(lz_buf); free(ref); free(out_buf); free(off_arr);
        clReleaseMemObject(d_in); clReleaseMemObject(d_off); clReleaseMemObject(d_out);
        clReleaseKernel(krn); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        return 0;
    }

    /* Compress path:
    usage: lzo_gpu input_file [compressed.out]
    */

    /* 1. 读输入文件 */
    uint64_t tA0 = now_ns();
    size_t in_sz; unsigned char* in_buf = (unsigned char*)read_file(in_path, &in_sz);
    uint64_t tA1 = now_ns();

    /* 2. OpenCL 初始化 & 构建内核 */
    uint64_t tB0 = now_ns();
    ocl_init();
    size_t src_len; char* cl_src = read_file(cl_path, &src_len);
    cl_program prog = clCreateProgramWithSource(ctx, 1,
        (const char**)&cl_src, &src_len, &err); CHECK(err);
    err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
    if (err != CL_SUCCESS)
        print_buildlog(prog, dev);
    cl_kernel krn = clCreateKernel(prog, "lzo1x_block_compress", &err); CHECK(err);
    uint64_t tB1 = now_ns();

    /* 3. 计算分块数并创建缓冲区 */
    uint64_t tC0 = now_ns();
    //cl_uint blk = BLK_SIZE;
    //size_t nblk = (in_sz + BLK_SIZE - 1) / BLK_SIZE;
    size_t blk, nblk;
    choose_blocking(in_sz, dev, &blk, &nblk);
    printf("\nAuto blocking: blk_sz=%zu , nblk=%zu (CU×%d)\n",
        blk, nblk, OCC_FACTOR);
    size_t worst_blk = lzo_worst(blk);   /*  64 KiB →  68 707 B */
    size_t out_cap = nblk * worst_blk;      /* 保证总容量充足 */

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        in_sz, in_buf, &err); CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_cap, NULL, &err); CHECK(err);
    cl_mem d_len = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        nblk * sizeof(cl_uint), NULL, &err); CHECK(err);
    uint64_t tC1 = now_ns();

    /* 4. 设置参数并启动kernel */
    CHECK(clSetKernelArg(krn, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(krn, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(krn, 2, sizeof(cl_mem), &d_len));
	CHECK(clSetKernelArg(krn, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(krn, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(krn, 5, sizeof(cl_uint), &worst_blk));

    // 设备本地内存
    //size_t max_lmem;
    //clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,
    //    sizeof(max_lmem), &max_lmem, NULL);

    size_t lsz = 1, gsz = nblk * lsz;
    cl_event evt;
    uint64_t tD0 = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn, 1, NULL, &gsz, &lsz, 0, NULL, &evt));
	clWaitForEvents(1, &evt);
    uint64_t tD1 = now_ns();
    cl_ulong t_kernel_start, t_kernel_end;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,
        sizeof(t_kernel_start), &t_kernel_start, NULL);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,
        sizeof(t_kernel_end), &t_kernel_end, NULL);
    double kernel_ns = (double)(t_kernel_end - t_kernel_start);   // ns
	printf("Kernel execution time: %.3f ms\n", kernel_ns / 1e6);
    //CHECK(clFinish(q));

    /* 5. 读取各块长度 */
    uint64_t tE0 = now_ns();
    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    CHECK(clEnqueueReadBuffer(q, d_len, CL_TRUE, 0, nblk * sizeof(cl_uint), len_arr, 0, NULL, NULL));

    /* 6. 把压缩结果依次拼到 host 缓冲区 */
    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += len_arr[i];
    unsigned char* out_buf = malloc(out_sz);

    size_t host_off = 0;
    for (size_t i = 0; i < nblk; ++i) {
        size_t dev_off = i * worst_blk;
        if (dev_off + len_arr[i] > out_cap) {
            fprintf(stderr, "len_arr[%zu] overflow !\n", i);
            exit(1);
        }
        CHECK(clEnqueueReadBuffer(q, d_out, CL_TRUE, dev_off, len_arr[i],
            out_buf + host_off, 0, NULL, NULL));
        host_off += len_arr[i];
    }
    uint64_t tE1 = now_ns();

    /* 7. 打印性能结果 */
    puts("\n=== Per-block size ===");
    for (size_t i = 0; i < nblk; ++i)
        printf("block %-3zu -> %7u B\n", i, len_arr[i]);
    printf("Input %zu B → Output %zu B  (%.2f%%)\n",
        in_sz, out_sz, 100.0 * out_sz / in_sz);

    puts("\n=== Timing summary ===");
    print_ns("A. read input", tA1 - tA0);
    print_ns("B. build program", tB1 - tB0);
    print_ns("C. create+upload", tC1 - tC0);
    print_ns("D. enqueue+wait", tD1 - tD0);     /* host 视角 */
    print_ns("|- device kernel", kernel_ns);     /* 纯 device */
    print_ns("E. download result", tE1 - tE0);
    print_ns("Total (A→E)", tE1 - tA0);

    /* 吞吐量 */
    double mb_in = (double)in_sz / 1e6;
    double mb_out = (double)out_sz / 1e6;
    double gpu_mbps = mb_in / (kernel_ns * 1e-9);   /* MB / s */
    printf("\nInput %zu B -> Output %zu B (%.2f%%, %.2f:1)\n",
        in_sz, out_sz, 100.0 * out_sz / in_sz, (double)in_sz / out_sz);
    printf("GPU kernel throughput : %.2f MB/s  (%.2f GiB/s)\n",
        gpu_mbps, gpu_mbps / 1024.0);

    /* 8. 写压缩文件 */
    FILE* fo = fopen(out_path, "wb");
    uint16_t u16 = MAGIC;             fwrite(&u16, 2, 1, fo);   // 魔数
    uint32_t u32 = (uint32_t)in_sz;   fwrite(&u32, 4, 1, fo);   // 原始大小
	u32 = (uint32_t)blk;              fwrite(&u32, 4, 1, fo);   // 分块大小
	u32 = (uint32_t)nblk;             fwrite(&u32, 4, 1, fo);   // 分块数
	fwrite(len_arr, 4, nblk, fo);                               // 每块压缩长度
	fwrite(out_buf, 1, out_sz, fo);                             // 压缩数据
    fclose(fo);

    /* 资源释放 */
    free(cl_src); free(in_buf); free(out_buf); free(len_arr);
    clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
    clReleaseKernel(krn); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}