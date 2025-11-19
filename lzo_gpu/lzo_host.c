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
    /* Allow forcing a smaller target block count for testing via
     * LZO_FORCE_NBLK environment variable (helps run CPU-style kernel
     * on GPUs by reducing concurrent work-items). */
    const char* env_nblk = getenv("LZO_FORCE_NBLK");
    if (env_nblk) {
        int v = atoi(env_nblk);
        if (v > 0) tgt_blk = (size_t)v;
    }
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
    /* allow selecting CPU device for testing CPU-style kernels via env
     * variable LZO_OPENCL_DEVICE=CPU; default remains GPU. */
    const char* prefer = getenv("LZO_OPENCL_DEVICE");
    cl_device_type dtype = CL_DEVICE_TYPE_GPU;
    if (prefer && strcmp(prefer, "CPU") == 0) dtype = CL_DEVICE_TYPE_CPU;
    CHECK(clGetPlatformIDs(1, &pf, NULL));
    CHECK(clGetDeviceIDs(pf, dtype, 1, &dev, NULL));
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
    /* Normalize names: never attempt to load or print the historic
     * "decompress" spelling for kernel/base names. Always prefer the
     * shorter "decomp" token for file and binary names. This avoids the
     * host producing or searching for 'lzo1x_decompress' binaries/sources. */
    char use_base[256]; strncpy(use_base, base, sizeof(use_base)-1); use_base[sizeof(use_base)-1]='\0';
    char use_cl_src[256]; strncpy(use_cl_src, cl_src_path, sizeof(use_cl_src)-1); use_cl_src[sizeof(use_cl_src)-1]='\0';
    /* replace "decompress" -> "decomp" in base (if present) */
    char* p = strstr(use_base, "decompress");
    if (p) {
        size_t tail_len = strlen(p + strlen("decompress"));
        /* write 'decomp' then append trailing part */
        strcpy(p, "decomp");
        strcpy(p + strlen("decomp"), p + strlen("decomp") + tail_len);
    }
    /* replace "decompress.cl" -> "decomp.cl" in cl src path if present */
    char* q = strstr(use_cl_src, "decompress.cl");
    if (q) {
        size_t tail_len2 = strlen(q + strlen("decompress.cl"));
        strcpy(q, "decomp.cl");
        strcpy(q + strlen("decomp.cl"), q + strlen("decomp.cl") + tail_len2);
    }

    char bin_path[512]; snprintf(bin_path, sizeof(bin_path), "%s.bin", use_base);
    /* also prepare an alternate path inside the lzo_gpu subdir to be robust
     * against differing working directories when running via tools/runner */
    char bin_path_alt[512]; snprintf(bin_path_alt, sizeof(bin_path_alt), "lzo_gpu/%s.bin", use_base);
    /* Attempt to use a precompiled binary first (robust fallback to source
     * compilation is performed below if binary is incompatible). */
    FILE* fb = fopen(bin_path, "rb");
    if (!fb) {
        /* try lzo_gpu/ subdir */
        fb = fopen(bin_path_alt, "rb");
    }
    cl_int err; cl_program prog = NULL;
    if (fb) {
        /* attempt to use precompiled binary; on any failure fall back to source */
        fseek(fb, 0, SEEK_END); long bsz = ftell(fb); fseek(fb, 0, SEEK_SET);
        unsigned char* bin = malloc(bsz);
        if (fread(bin,1,bsz,fb) != (size_t)bsz) { perror("fread"); fclose(fb); free(bin); exit(1); }
        fclose(fb);
        cl_int binary_status;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
            (const unsigned char**)&bin, &binary_status, &err);
        free(bin);
        if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
            fprintf(stderr, "warning: precompiled binary %s.bin incompatible, falling back to source (clCreateProgramWithBinary err=%d bin_status=%d)\n", base, err, binary_status);
            if (prog) { clReleaseProgram(prog); prog = NULL; }
        } else {
            err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
                fprintf(stderr, "Build log (from binary):\n%s\n", log); free(log);
                fprintf(stderr, "warning: build from binary failed for %s.bin (err=%d), falling back to source\n", base, err);
                clReleaseProgram(prog); prog = NULL;
            }
        }
    }

    if (!prog) {
        /* compile from source as a robust fallback
         * If the explicit <base>.cl does not exist (common when using combined
         * names like "lzo1x_1_comp_atomic"), try to locate and combine the
         * core and frontend sources: e.g. "lzo1x_1.cl" + "lzo1x_comp_atomic.cl".
         */
        size_t src_len = 0; char* src = NULL;
        /* try source in current dir, otherwise try lzo_gpu/ subdir */
        FILE* fchk = fopen(use_cl_src, "rb");
        if (!fchk) {
            char use_cl_src_alt[512]; snprintf(use_cl_src_alt, sizeof(use_cl_src_alt), "lzo_gpu/%s", use_cl_src);
            fchk = fopen(use_cl_src_alt, "rb");
            if (fchk) {
                fclose(fchk);
                src = read_file(use_cl_src_alt, &src_len);
            }
        } else {
            fclose(fchk);
            src = read_file(use_cl_src, &src_len);
        }
        if (!src) {
            /* If this base is a decompressor base (only two decompressor kernels exist),
             * do NOT attempt to split and combine core+frontend — simply report missing
             * source and exit. This prevents the loader from trying to construct
             * combined names for decompressor kernels. */
            if (strncmp(use_base, "lzo1x_decomp", strlen("lzo1x_decomp")) == 0) {
                fprintf(stderr, "source file %s not found and decompressor base will not be combined\n", use_cl_src);
                exit(1);
            }
            /* attempt to split base into core_front and construct filenames */
            char base_copy[256]; strncpy(base_copy, use_base, sizeof(base_copy)-1); base_copy[sizeof(base_copy)-1]='\0';
            char* sep = strchr(base_copy, '_');
            if (sep) {
                *sep = '\0'; char* core = base_copy; char* front = sep + 1;
                char core_fname[256]; char frontend_fname[256];
                snprintf(core_fname, sizeof(core_fname), "%s.cl", core);
                /* frontend in kernel_base was formed by stripping a leading
                 * "lzo1x_" from frontend names; restore it when trying file
                 * names (i.e. try "lzo1x_%s.cl"). */
                snprintf(frontend_fname, sizeof(frontend_fname), "lzo1x_%s.cl", front);
                /* try reading core + frontend */
                size_t core_len=0, front_len=0; char* core_src = NULL; char* front_src = NULL;
                FILE* fcore = fopen(core_fname, "rb");
                FILE* ffront = fopen(frontend_fname, "rb");
                /* try lzo_gpu/ prefixed filenames if not found in cwd */
                if (!fcore) {
                    char core_fname_alt[256]; snprintf(core_fname_alt, sizeof(core_fname_alt), "lzo_gpu/%s", core_fname);
                    fcore = fopen(core_fname_alt, "rb");
                    if (fcore) { fclose(fcore); core_src = read_file(core_fname_alt, &core_len); }
                } else { fclose(fcore); core_src = read_file(core_fname, &core_len); }
                if (!ffront) {
                    char frontend_fname_alt[256]; snprintf(frontend_fname_alt, sizeof(frontend_fname_alt), "lzo_gpu/%s", frontend_fname);
                    ffront = fopen(frontend_fname_alt, "rb");
                    if (ffront) { fclose(ffront); front_src = read_file(frontend_fname_alt, &front_len); }
                } else { fclose(ffront); front_src = read_file(frontend_fname, &front_len); }
                if (!core_src || !front_src) {
                    if (core_src) free(core_src);
                    if (front_src) free(front_src);
                    fprintf(stderr, "source files not found for combined kernel: %s and %s\n", core_fname, frontend_fname);
                    exit(1);
                }
                /* remove any #include "lzo1x_*.cl" lines from frontend to avoid
                 * re-including a default core; produce a cleaned frontend body. */
                char* cleaned = malloc(front_len + 1);
                size_t outp = 0; size_t pos = 0;
                while (pos < front_len) {
                    /* find line end */
                    size_t line_start = pos; while (pos < front_len && front_src[pos] != '\n') pos++;
                    size_t line_len = pos - line_start;
                    /* check if line contains #include and lzo1x_ */
                    if (!(line_len > 0 && strstr(front_src + line_start, "#include") && strstr(front_src + line_start, "lzo1x_"))) {
                        memcpy(cleaned + outp, front_src + line_start, line_len);
                        outp += line_len;
                        if (pos < front_len && front_src[pos] == '\n') { cleaned[outp++] = '\n'; pos++; }
                    } else {
                        /* skip this include line including trailing newline if present */
                        if (pos < front_len && front_src[pos] == '\n') pos++;
                    }
                }
                cleaned[outp] = '\0';
                /* build combined source: core first, then cleaned frontend */
                src_len = core_len + outp + 32;
                src = malloc(src_len);
                snprintf(src, src_len, "/* combined: %s + %s */\n%s\n/* frontend (cleaned) */\n%s", core_fname, frontend_fname, core_src, cleaned);
                free(core_src); free(front_src); free(cleaned);
            } else {
                fprintf(stderr, "source file %s not found and base not splittable\n", cl_src_path);
                exit(1);
            }
        } else {
            /* src was loaded from current dir or lzo_gpu/; nothing to do */
        }

        /* create program from assembled source */
        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
        if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource failed (err=%d)\n", err); free(src); exit(1); }
        err = clBuildProgram(prog, 1, &dev, "", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz+1); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL); log[log_sz]='\0';
            fprintf(stderr, "Build log (from source):\n%s\n", log); free(log); free(src); exit(1);
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
    const char *strategy = "none"; /* publish strategy: none, atomic, usehost */

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
        if (strcmp(arg, "--strategy") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing argument for %s\n", arg); return 1; }
            strategy = argv[++i];
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
        /* allow selecting a vectorized decompressor via env var LZO_DECOMP_VEC=1
         * If not set, auto-detect from device capabilities: require
         * CL_DEVICE_MEM_BASE_ADDR_ALIGN >= 128 (bits => 16 bytes) and
         * CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR >= 16 to enable vec path.
         */
        const char* devec_env = getenv("LZO_DECOMP_VEC");
        int devec_flag = 0;
        if (devec_env) {
            devec_flag = (strcmp(devec_env, "1") == 0);
        } else {
            cl_uint mem_align_bits = 0, pref_char = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align_bits), &mem_align_bits, NULL);
            clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(pref_char), &pref_char, NULL);
            if (mem_align_bits >= 128 && pref_char >= 16) devec_flag = 1;
        }
        const char* decomp_base = devec_flag ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src  = devec_flag ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";
        /* Emit stable, parseable identifiers for aggregation tools */
        if (!suppress_non_data) {
            printf("KERNEL=%s\n", decomp_base);
            printf("STRATEGY=%s\n", devec_flag ? "vec" : "scalar");
        }
        cl_program prog_d = load_prog_from_bin_or_src(decomp_base, decomp_src);
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

    /* decide whether to write output:
       - If user requested --verify (decompress mode) and did not explicitly pass -o,
         do NOT write the decompressed file (only perform in-memory verification).
       - If user explicitly passed -o, honor it and write output as requested. */
    if (verify_path && !output_explicit) {
        /* skip writing decompressed output when verify requested without -o */
        if (!suppress_non_data) puts("verify mode: not writing decompressed output (no -o given)");
    } else {
        /* compute default output path if not explicitly provided */
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
    /* If a strategy was requested, select the corresponding comp frontend.
       To allow using precompiled binaries for core×frontend combinations
       we form a combined kernel base name: <core>_<frontend>. For example
       'lzo1x_1' + 'lzo1x_comp_atomic' -> 'lzo1x_1_lzo1x_comp_atomic'. */
    char core_base[64]; strcpy(core_base, kernel_base);
    /* Always select a compression frontend: treat 'none' as the generic
     * comp frontend (lzo1x_comp). Map 'usehost' to the delayed frontend
     * variant which provides host-assisted behavior in some setups. */
    {
        const char *frontend = NULL;
        if (strcmp(strategy, "none") == 0) {
            frontend = "lzo1x_comp";
        } else if (strcmp(strategy, "atomic") == 0) {
            frontend = "lzo1x_comp_atomic";
        } else if (strcmp(strategy, "usehost") == 0) {
            /* usehost: use same frontend as 'none' (host-backed buffers but same frontend) */
            frontend = "lzo1x_comp";
        } else {
            fprintf(stderr, "unknown strategy: %s\n", strategy); return 1;
        }

        /* combine core and frontend so the loader will find precompiled combo binaries
           strip a leading "lzo1x_" from the frontend name to avoid duplicated prefix */
        const char *frontend_short = frontend;
        if (strncmp(frontend, "lzo1x_", 6) == 0) frontend_short = frontend + 6;
        snprintf(kernel_base, sizeof(kernel_base), "%s_%s", core_base, frontend_short);
    }
    snprintf(cl_src, sizeof(cl_src), "%s.cl", kernel_base);
    /* Emit stable, parseable identifiers for aggregation tools */
    {
        const char* strat = (strcmp(strategy, "none") == 0) ? "scalar" : strategy;
        if (!debug && !suppress_non_data) {
            /* keep non-verbose prints consistent: print kernel and strategy */
            printf("KERNEL=%s\n", kernel_base);
            printf("STRATEGY=%s\n", strat);
        } else if (!suppress_non_data) {
            printf("KERNEL=%s\n", kernel_base);
            printf("STRATEGY=%s\n", strat);
        }
    }
    cl_program prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);
    cl_int err;
    /* select kernel function name according to the kernel_base we loaded
     * Use canonical exported symbol 'lzo1x_block_compress' from frontends.
     */
    char krn_name[64];
    strcpy(krn_name, "lzo1x_block_compress");
    cl_kernel krn_c = clCreateKernel(prog_c, krn_name, &err);
    if (err != CL_SUCCESS) {
        /* Simplified fallback: report available kernels and force a single
         * source rebuild retry. Avoid multiple name-specific fallbacks — precompile
         * step should produce binaries matching canonical exported symbol.
         */
        if (err == CL_INVALID_KERNEL_NAME) {
            size_t kn_sz = 0;
            clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &kn_sz);
            if (kn_sz > 0) {
                char* kn = malloc(kn_sz + 1);
                clGetProgramInfo(prog_c, CL_PROGRAM_KERNEL_NAMES, kn_sz, kn, NULL);
                kn[kn_sz] = '\0';
                fprintf(stderr, "kernel '%s' not found; available kernels: %s\n", krn_name, kn);
                free(kn);
            } else {
                fprintf(stderr, "kernel '%s' not found and program reports no kernel names\n", krn_name);
            }
            /* Force a source-built program and retry once */
            clReleaseProgram(prog_c);
            prog_c = load_prog_from_bin_or_src(kernel_base, cl_src);
            krn_c = clCreateKernel(prog_c, krn_name, &err);
            if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel after source rebuild failed (err=%d)\n", err); exit(1); }
        } else {
            fprintf(stderr, "clCreateKernel failed for %s (err=%d)\n", krn_name, err); exit(1);
        }
    }

    /* choose blocking dynamically (uses GPU CU count and ALIGN_BYTES) */
    size_t blk = 0, nblk = 0;
    choose_blocking(in_sz, dev, &blk, &nblk);
    size_t worst_blk = lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_sz, in_buf, &err); CHECK(err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, out_cap, NULL, &err); CHECK(err);
    /* map_mode: 0=default CL_MEM_WRITE_ONLY + clEnqueueReadBuffer
     * 1=ALLOC_HOST_PTR + clEnqueueMapBuffer
     * 2=USE_HOST_PTR with host pointer (posix_memalign)
     */
    int map_mode = 0;
    /* comp frontends use the mapped/host-len approach */
    /* comp frontends use the mapped/host-len approach */
    if (strncmp(kernel_base, "lzo1x_comp", strlen("lzo1x_comp")) == 0 ||
        strncmp(kernel_base, "lzo1x_gpu_port", strlen("lzo1x_gpu_port")) == 0) map_mode = 1;
    /* special-case: if strategy==usehost, request USE_HOST_PTR backing */
    if (strcmp(strategy, "usehost") == 0) map_mode = 2;

    cl_mem d_len;
    void* host_len_ptr = NULL;
    size_t len_bytes = nblk * sizeof(cl_uint);
    if (map_mode == 1) {
        d_len = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, len_bytes, NULL, &err); CHECK(err);
    } else if (map_mode == 2) {
        /* allocate host pointer aligned to 64 bytes */
        int rc = posix_memalign(&host_len_ptr, 64, len_bytes);
        if (rc != 0) host_len_ptr = malloc(len_bytes);
        memset(host_len_ptr, 0, len_bytes);
        d_len = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, len_bytes, host_len_ptr, &err); CHECK(err);
    } else {
        d_len = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, len_bytes, NULL, &err); CHECK(err);
    }



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
    uint64_t t_len_read_end = t_len_read_start;
    if (map_mode == 1) {
        void* mapped = clEnqueueMapBuffer(q, d_len, CL_TRUE, CL_MAP_READ, 0, len_bytes, 0, NULL, NULL, &err);
        CHECK(err);
        memcpy(len_arr, mapped, len_bytes);
        CHECK(clEnqueueUnmapMemObject(q, d_len, mapped, 0, NULL, NULL));
        t_len_read_end = now_ns();
    } else if (map_mode == 2) {
        /* host_len_ptr contains the backing store for d_len */
        if (host_len_ptr) memcpy(len_arr, host_len_ptr, len_bytes);
        t_len_read_end = now_ns();
    } else {
        CHECK(clEnqueueReadBuffer(q, d_len, CL_TRUE, 0, len_bytes, len_arr, 0, NULL, NULL));
        t_len_read_end = now_ns();
    }

    if (debug) {
        fprintf(stderr, "Per-block compressed lengths (nblk=%zu):\n", nblk);
        for (size_t i = 0; i < nblk; ++i) {
            fprintf(stderr, "  block %4zu : %u\n", i, len_arr[i]);
        }
    }

    size_t out_sz = 0; for (size_t i = 0; i < nblk; ++i) out_sz += len_arr[i];
    unsigned char* out_buf = NULL;
    size_t host_off = 0;
    /* bulk-read entire device output buffer once to avoid many small PCIe transfers */
    unsigned char* dev_out = malloc(out_cap); uint64_t t_bulk_read_start = now_ns();
    CHECK(clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out_cap, dev_out, 0, NULL, NULL));
    uint64_t t_bulk_read_end = now_ns();
    /* debug: dump first 32 bytes of first block to help diagnose visibility */
    if (debug) {
        fprintf(stderr, "dev_out[0..31]:");
        for (size_t i = 0; i < 32 && i < out_cap; ++i) fprintf(stderr, " %02x", dev_out[i]);
        fprintf(stderr, "\n");
    }

    /* If kernel didn't populate `out_len` (all zeros), try reconstructing per-block
     * lengths from the device output buffer: we expect the kernel to write a
     * little-endian 32-bit length at the start of each block region as a
     * robust fallback (the `lzo1x_gpu_port` variant does this). */
    if (out_sz == 0) {
        for (size_t i = 0; i < nblk; ++i) {
            size_t dev_off = i * worst_blk;
            if (dev_off + 4 <= out_cap) {
                uint32_t v = (uint32_t)dev_out[dev_off + 0]
                           | ((uint32_t)dev_out[dev_off + 1] << 8)
                           | ((uint32_t)dev_out[dev_off + 2] << 16)
                           | ((uint32_t)dev_out[dev_off + 3] << 24);
                len_arr[i] = v;
                out_sz += v;
            } else {
                len_arr[i] = 0;
            }
        }
    }

    out_buf = malloc(out_sz);
    for (size_t i = 0; i < nblk; ++i) {
        size_t dev_off = i * worst_blk;
        if (len_arr[i] > 0) {
            memcpy(out_buf + host_off, dev_out + dev_off, len_arr[i]);
            host_off += len_arr[i];
        }
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

        /* choose vectorized decompress kernel if requested for verify */
        const char* devec2 = getenv("LZO_DECOMP_VEC");
        int devec_flag2 = 0;
        if (devec2) {
            devec_flag2 = (strcmp(devec2, "1") == 0);
        } else {
            cl_uint mem_align_bits2 = 0, pref_char2 = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(mem_align_bits2), &mem_align_bits2, NULL);
            clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(pref_char2), &pref_char2, NULL);
            if (mem_align_bits2 >= 128 && pref_char2 >= 16) devec_flag2 = 1;
        }
        const char* decomp_base2 = devec_flag2 ? "lzo1x_decomp_vec" : "lzo1x_decomp";
        const char* decomp_src2  = devec_flag2 ? "lzo1x_decomp_vec.cl" : "lzo1x_decomp.cl";
        cl_program prog_d = load_prog_from_bin_or_src(decomp_base2, decomp_src2);
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
