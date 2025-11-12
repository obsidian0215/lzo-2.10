#define _POSIX_C_SOURCE 199309L
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

/* forward decl so main can call it */
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path);

/* Helper: load program from <base>.bin or from source file */
static cl_program load_prog_from_bin_or_src(const char* base, const char* cl_src_path)
{
    char bin_path[512]; snprintf(bin_path, sizeof(bin_path), "%s.bin", base);
    FILE* fb = fopen(bin_path, "rb");
    cl_int err; cl_program prog = NULL;
    if (fb) {
        fseek(fb, 0, SEEK_END); long bsz = ftell(fb); fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin,1,bsz,fb) != (size_t)bsz) { perror("fread"); fclose(fb); free(bin); exit(1); }
        fclose(fb);
        cl_int binary_status;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
            (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithBinary failed\n"); exit(1); }
        err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
            fprintf(stderr, "Build log:\n%s\n", log); free(log); exit(1);
        }
    } else {
        size_t src_len; char* src = read_file(cl_src_path, &src_len);
        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err); CHECK(err);
        err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
            fprintf(stderr, "Build log:\n%s\n", log); free(log); exit(1);
        }
        free(src);
    }
    return prog;
}

int main(int argc, char** argv)
{
    uint64_t t_start_total = now_ns();
    if (argc < 2) { fprintf(stderr, "usage: %s [--debug|-v] input_file (or -d [--debug|-v] lzfile orig_file)\n", argv[0]); return 1; }

    /* simple CLI parsing: support optional --debug/-v flag and -d decompress mode */
    int debug = 0;
    int verify_flag = 0; /* only when set, do roundtrip/verify prints */
    int decompress_mode = 0;
    const char *in_path = NULL;
    const char *lz_path = NULL;
    const char *orig_path = NULL;
    const char *output_path = NULL;
    int output_explicit = 0; /* whether -o/--output was explicitly provided */
    int suppress_non_data = 0; /* when writing to stdout (-), suppress non-data prints */
    int show_help = 0;
    const char *comp_level = "1"; /* compression level: "1", "1k", "1l", "1o" */

    /* pass 1: only detect mode (-d) and help, to know how to parse verify */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { show_help = 1; }
        if (strcmp(argv[i], "-d") == 0) { decompress_mode = 1; }
    }

    /* pass 2: parse options and positionals with knowledge of mode */
    const char* verify_path = NULL; /* only for -d mode */
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--debug") == 0 || strcmp(arg, "-v") == 0) { debug = 1; continue; }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) { /* already noted */ continue; }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            output_path = argv[++i];
            output_explicit = 1;
            if (strcmp(output_path, "-") == 0) suppress_non_data = 1;
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "--level") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            comp_level = argv[++i];
            continue;
        }
        if (strcmp(arg, "-c") == 0 || strcmp(arg, "--verify") == 0) {
            if (decompress_mode) {
                if (i + 1 >= argc || argv[i+1][0] == '-') { fprintf(stderr, "--verify requires a reference file in -d mode\n"); return 1; }
                verify_path = argv[++i];
            } else {
                verify_flag = 1; /* compress-mode in-memory roundtrip */
            }
            continue;
        }
        if (strcmp(arg, "-d") == 0) { /* already noted */ continue; }
        /* positional */
        if (arg[0] != '-') {
            if (decompress_mode) {
                if (!lz_path) { lz_path = arg; continue; }
                /* ignore extra positionals; verify file should come via --verify */
            } else {
                if (!in_path) { in_path = arg; continue; }
            }
        }
    }

    if (show_help) {
        printf("Usage:\n");
        printf("  %s [--debug|-v] [--verify|-c] [-L level] [-o out.lzo] input_file\n", argv[0]);
        printf("     - compress input_file. If -o is omitted, writes to input_file.lzo\n");
        printf("     - --verify/-c (compress mode): do in-memory roundtrip check (no arg).\n");
        printf("     - -L|--level LEVEL : compression level to select kernel variant (default: 1)\n");
        printf("         supported LEVEL values:\n");
        printf("            1   : default LZO1X-1 compressor (kernel: lzo1x_1)\n");
        printf("            1k  : LZO1X-1K variant (kernel: lzo1x_1k) - optimized for kernel K behavior\n");
        printf("            1l  : LZO1X-1L variant (kernel: lzo1x_1l) - alternative lookup/heuristics\n");
        printf("            1o  : LZO1X-1O variant (kernel: lzo1x_1o) - other tuning/optimizations\n");
        printf("\n");
        printf("  %s -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo\n", argv[0]);
        printf("     - decompress input.lzo. If -o is omitted, writes to input with .lzo removed or .raw appended.\n");
        printf("     - --verify/-c ORIG (decompress mode): verify output equals ORIG. Without -o, no file is written.\n");
        printf("\n");
        printf("Examples:\n");
        printf("  Compress with default level: %s input.dat -o out.lzo\n", argv[0]);
        printf("  Compress with level 1k:      %s -L 1k input.dat -o out.lzo\n", argv[0]);
        printf("  Decompress and verify:      %s -d --verify input.dat out.lzo -o out.dec\n", argv[0]);
        printf("  Stream decompressed to stdout: %s -d out.lzo -o - | sha256sum\n", argv[0]);
        printf("  %s -h|--help                                 # show this help\n", argv[0]);
        return 0;
    }

    /* Decompress mode */
    if (decompress_mode) {
        if (!lz_path) { fprintf(stderr, "no input .lzo specified (after -d)\n"); return 1; }
    uint64_t t_io_in = now_ns();
    size_t lz_sz; unsigned char* lz_buf = read_file(lz_path, &lz_sz);
    unsigned char* ref = NULL;
    size_t ref_sz = 0;
        unsigned char* p = lz_buf;
        uint16_t magic = *(uint16_t*)p; p += 2;
        if (magic != MAGIC) { fprintf(stderr, "bad magic\n"); return 1; }
        uint32_t orig_sz = *(uint32_t*)p; p += 4;
        uint32_t blk_sz = *(uint32_t*)p; p += 4;
        uint32_t nblk = *(uint32_t*)p; p += 4;
        uint32_t* len_arr = (uint32_t*)p; p += 4 * nblk;
        size_t comp_sz = lz_sz - (p - lz_buf);

        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (uint32_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

    uint64_t t_io_after = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
        cl_program prog_d = load_prog_from_bin_or_src("lzo1x_decompress", "lzo1x_decompress.cl");
        cl_int err; cl_kernel krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err); CHECK(err);

    cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, comp_sz, p, &err); CHECK(err);
    cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (nblk + 1) * sizeof(cl_uint), off_arr, &err); CHECK(err);
    cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, orig_sz, NULL, &err); CHECK(err);
    /* decompressor expects an out_lens buffer as arg 3 */
    cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

    CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
    CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));

    size_t gsz = nblk, lsz = 1; cl_event evt; uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt)); clWaitForEvents(1, &evt);
    uint64_t t_exec_end = now_ns();

    unsigned char* out2 = malloc(orig_sz); uint64_t t_read_start = now_ns();
    CHECK(clEnqueueReadBuffer(q, d_out2, CL_TRUE, 0, orig_sz, out2, 0, NULL, NULL));
    uint64_t t_read_end = now_ns();

    /* perform verify first if requested; on failure do not write and exit non-zero */
    if (verify_path) {
        size_t ref_sz; unsigned char* ref = (unsigned char*)read_file(verify_path, &ref_sz);
        if (ref_sz != orig_sz || memcmp(ref, out2, orig_sz) != 0) {
            fprintf(stderr, "decompress verify FAILED!\n");
            for (size_t i = 0; i < orig_sz; ++i) {
                if (ref[i] != out2[i]) { fprintf(stderr, "first_mismatch_offset=%zu (ref=0x%02x out=0x%02x)\n", i, ref[i], out2[i]); break; }
            }
            free(ref);
            /* cleanup and exit */
            clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
            clReleaseKernel(krn_d); clReleaseProgram(prog_d);
            clReleaseCommandQueue(q); clReleaseContext(ctx);
            free(lz_buf); free(off_arr); free(out2);
            return 1;
        }
        free(ref);
        if (!suppress_non_data) puts("verify OK");
    }

    /* decide whether to write output: compute default path if necessary */
    if (!output_path) {
        const char* in = lz_path;
        size_t L = strlen(in);
        int ends_lzo = (L >= 4 && strcmp(in + L - 4, ".lzo") == 0);
        size_t outL = ends_lzo ? (L - 4) : (L + 4);
        char* def = (char*)malloc(outL + 1);
        if (ends_lzo) { memcpy(def, in, L - 4); def[L - 4] = '\0'; }
        else { memcpy(def, in, L); memcpy(def + L, ".raw", 4); def[L + 4] = '\0'; }
        output_path = def;
    }

    if (output_path) {
        if (strcmp(output_path, "-") == 0) {
            /* write raw data to stdout (only data) */
            if (fwrite(out2, 1, orig_sz, stdout) != orig_sz) { perror("stdout write"); }
            fflush(stdout);
        } else {
            if (verify_path && strcmp(output_path, verify_path) == 0) {
                fprintf(stderr, "refusing to write output to the same path as --verify reference: %s\n", output_path);
                return 1;
            }
            FILE* fo = fopen(output_path, "wb");
            if (!fo) { perror(output_path); return 1; }
            if (fwrite(out2, 1, orig_sz, fo) != orig_sz) { perror("fwrite"); fclose(fo); return 1; }
            fclose(fo);
            if (!suppress_non_data) printf("wrote %s\n", output_path);
        }
    }

        uint64_t t_total_end = now_ns();
        double ms_total = (t_total_end - t_start_total)/1e6;
        double ms_io = (t_io_after - t_io_in)/1e6;
        double ms_kernel = (t_exec_end - t_exec_start)/1e6;
        double ms_read = (t_read_end - t_read_start)/1e6;
        double ratio = lz_sz > 0 ? (double)orig_sz / (double)lz_sz : 0.0;
        double thrpt = ms_kernel > 0 ? ((double)orig_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;
        printf("[DECOMP] orig=%zu comp=%zu blocks=%u blk_size=%u ratio=%.3f kernel=%.3f ms io=%.3f ms read=%.3f ms total=%.3f ms thrpt=%.2f MB/s\n",
                (size_t)orig_sz, (size_t)lz_sz, nblk, blk_sz, ratio, ms_kernel, ms_io, ms_read, ms_total, thrpt);

    clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        free(lz_buf); free(off_arr); free(out2);
        if (orig_path) free(ref);
        return 0;
    }

    /* Compress path (simple, fast) */
    if (!in_path) { fprintf(stderr, "no input file specified for compression\n"); return 1; }
    uint64_t t_io_in = now_ns();
    size_t in_sz; unsigned char* in_buf = (unsigned char*)read_file(in_path, &in_sz);

    uint64_t t_io_read_done = now_ns();
    ocl_init();
    uint64_t t_ocl_init = now_ns();
    /* select compression kernel variant based on comp_level */
    char kernel_base[64]; char cl_src[128];
    if (strcmp(comp_level, "1") == 0 || strcmp(comp_level, "1x") == 0) {
        strcpy(kernel_base, "lzo1x_1");
    } else if (strcmp(comp_level, "1k") == 0) {
        strcpy(kernel_base, "lzo1x_1k");
    } else if (strcmp(comp_level, "1l") == 0) {
        strcpy(kernel_base, "lzo1x_1l");
    } else if (strcmp(comp_level, "1o") == 0) {
        strcpy(kernel_base, "lzo1x_1o");
    } else {
        fprintf(stderr, "unknown compression level: %s\n", comp_level);
        return 1;
    }
    snprintf(cl_src, sizeof(cl_src), "%s.cl", kernel_base);
    cl_program prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);
    cl_int err; cl_kernel krn_c = clCreateKernel(prog_c, "lzo1x_block_compress", &err); CHECK(err);

    /* choose blocking dynamically (uses GPU CU count and ALIGN_BYTES) */
    size_t blk = 0, nblk = 0;
    choose_blocking(in_sz, dev, &blk, &nblk);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_sz, in_buf, &err); CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_cap, NULL, &err); CHECK(err);
    cl_mem d_len = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

    CHECK(clSetKernelArg(krn_c, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(krn_c, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(krn_c, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(krn_c, 3, sizeof(cl_uint), &in_sz));
    CHECK(clSetKernelArg(krn_c, 4, sizeof(cl_uint), &blk));
    CHECK(clSetKernelArg(krn_c, 5, sizeof(cl_uint), &worst_blk));

    size_t gsz = nblk, lsz = 1; cl_event evt; uint64_t t_exec_start = now_ns();
    CHECK(clEnqueueNDRangeKernel(q, krn_c, 1, NULL, &gsz, &lsz, 0, NULL, &evt)); clWaitForEvents(1, &evt);
    uint64_t t_exec_end = now_ns();

    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint)); uint64_t t_len_read_start = now_ns();
    CHECK(clEnqueueReadBuffer(q, d_len, CL_TRUE, 0, nblk * sizeof(cl_uint), len_arr, 0, NULL, NULL));
    uint64_t t_len_read_end = now_ns();

    if (debug) {
        fprintf(stderr, "Per-block compressed lengths (nblk=%zu):\n", nblk);
        for (size_t i = 0; i < nblk; ++i) {
            fprintf(stderr, "  block %4zu : %u\n", i, len_arr[i]);
        }
    }

    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += len_arr[i];
    unsigned char* out_buf = malloc(out_sz);
    size_t host_off = 0;
    /* bulk-read entire device output buffer once to avoid many small PCIe transfers */
    unsigned char* dev_out = malloc(out_cap); uint64_t t_bulk_read_start = now_ns();
    CHECK(clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out_cap, dev_out, 0, NULL, NULL));
    uint64_t t_bulk_read_end = now_ns();
    for (size_t i = 0; i < nblk; ++i) {
        size_t dev_off = i * worst_blk;
        memcpy(out_buf + host_off, dev_out + dev_off, len_arr[i]);
        host_off += len_arr[i];
    }
    free(dev_out);

    /* decide output path if not specified: default to input_file.lzo */
    if (!output_path) {
        size_t L = strlen(in_path); char* def = (char*)malloc(L + 4 + 1);
        memcpy(def, in_path, L); memcpy(def + L, ".lzo", 4); def[L + 4] = '\0';
        output_path = def;
    }
    /* write LZO container: magic, orig_size, blk_size, nblk, len[nblk], then data */
    FILE* fo = fopen(output_path, "wb");
    if (!fo) { perror(output_path); return 1; }
    uint16_t magic = MAGIC;
    uint32_t orig_sz32 = (uint32_t)in_sz;
    uint32_t blk32 = (uint32_t)blk;
    uint32_t nblk32 = (uint32_t)nblk;
    if (fwrite(&magic, sizeof(magic), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&orig_sz32, sizeof(orig_sz32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&blk32, sizeof(blk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(&nblk32, sizeof(nblk32), 1, fo) != 1) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(len_arr, sizeof(uint32_t), nblk, fo) != nblk) { perror("fwrite"); fclose(fo); return 1; }
    if (fwrite(out_buf, 1, out_sz, fo) != out_sz) { perror("fwrite"); fclose(fo); return 1; }
    fclose(fo);
    printf("wrote %s\n", output_path);

    uint64_t t_after_write = now_ns();
    double ms_kernel = (t_exec_end - t_exec_start)/1e6;
    double ms_len_read = (t_len_read_end - t_len_read_start)/1e6;
    double ms_bulk_read = (t_bulk_read_end - t_bulk_read_start)/1e6;
    double ms_io = (t_io_read_done - t_io_in)/1e6;
    double ms_total = (t_after_write - t_start_total)/1e6;
    double ratio = out_sz > 0 ? (double)in_sz / (double)out_sz : 0.0;
    double thrpt = ms_kernel > 0 ? ((double)in_sz / (1024.0*1024.0)) / (ms_kernel/1000.0) : 0.0;
    printf("[COMP ] orig=%zu comp=%zu blocks=%zu blk_size=%zu ratio=%.3f kernel=%.3f ms len_rd=%.3f ms bulk_rd=%.3f ms io=%.3f ms total=%.3f ms thrpt=%.2f MB/s\n",
           in_sz, out_sz, nblk, blk, ratio, ms_kernel, ms_len_read, ms_bulk_read, ms_io, ms_total, thrpt);

    /* optional roundtrip verification only when --verify set */
    if (verify_flag) {
        uint32_t* off_arr = malloc((nblk + 1) * sizeof(uint32_t)); off_arr[0] = 0;
        for (size_t i = 0; i < nblk; ++i) off_arr[i+1] = off_arr[i] + len_arr[i];

        cl_program prog_d = load_prog_from_bin_or_src("lzo1x_decompress", "lzo1x_decompress.cl");
        cl_kernel krn_d = clCreateKernel(prog_d, "lzo1x_block_decompress", &err); CHECK(err);
        cl_mem d_comp = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, out_sz, out_buf, &err); CHECK(err);
        cl_mem d_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (nblk + 1) * sizeof(cl_uint), off_arr, &err); CHECK(err);
        cl_mem d_out2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, in_sz, NULL, &err); CHECK(err);
        cl_mem d_out_lens = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nblk * sizeof(cl_uint), NULL, &err); CHECK(err);

        CHECK(clSetKernelArg(krn_d, 0, sizeof(cl_mem), &d_comp));
        CHECK(clSetKernelArg(krn_d, 1, sizeof(cl_mem), &d_off));
        CHECK(clSetKernelArg(krn_d, 2, sizeof(cl_mem), &d_out2));
        CHECK(clSetKernelArg(krn_d, 3, sizeof(cl_mem), &d_out_lens));
        CHECK(clSetKernelArg(krn_d, 4, sizeof(cl_uint), &blk));
        CHECK(clSetKernelArg(krn_d, 5, sizeof(cl_uint), &in_sz));
        CHECK(clSetKernelArg(krn_d, 6, sizeof(cl_uint), &nblk));

        CHECK(clEnqueueNDRangeKernel(q, krn_d, 1, NULL, &gsz, &lsz, 0, NULL, &evt)); clWaitForEvents(1, &evt);
        unsigned char* out2 = malloc(in_sz);
        CHECK(clEnqueueReadBuffer(q, d_out2, CL_TRUE, 0, in_sz, out2, 0, NULL, NULL));

        if (memcmp(in_buf, out2, in_sz) == 0) printf("verify OK\n"); else { printf("verify FAILED\n"); for (size_t i=0;i<in_sz;i++){ if (in_buf[i]!=out2[i]){ printf("first mismatch at %zu (0x%02x != 0x%02x)\n", i, in_buf[i], out2[i]); break;} } }

        clReleaseMemObject(d_comp); clReleaseMemObject(d_off); clReleaseMemObject(d_out2); clReleaseMemObject(d_out_lens);
        clReleaseKernel(krn_d); clReleaseProgram(prog_d);
        free(off_arr); free(out2);
    }

    /* cleanup */
    clReleaseMemObject(d_in); clReleaseMemObject(d_out); clReleaseMemObject(d_len);
    clReleaseKernel(krn_c);
    clReleaseProgram(prog_c);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    free(in_buf); free(len_arr); free(out_buf);
    return 0;
}
