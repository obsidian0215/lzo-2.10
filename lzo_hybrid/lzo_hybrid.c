#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#include "../lzo_gpu/lzo_defaults.h"
#include "../lzo_gpu/lzo_gpu_utils.h"
#include "lzo_hybrid_core.h"

static cl_context g_ctx;
static cl_command_queue g_queue;
static cl_device_id g_dev;

static cl_device_type preferred_opencl_device_type(void) {
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int ocl_init(void) {
    cl_int err;
    cl_platform_id pf = NULL;
    cl_device_type pref_type = preferred_opencl_device_type();
    err = clGetPlatformIDs(1, &pf, NULL);
    if (err != CL_SUCCESS || !pf) {
        fprintf(stderr, "OpenCL: clGetPlatformIDs failed (err=%d)\n", err);
        return -1;
    }
    err = clGetDeviceIDs(pf, pref_type, 1, &g_dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_GPU)
        err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &g_dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_DEFAULT)
        err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_DEFAULT, 1, &g_dev, NULL);
    if (err != CL_SUCCESS && pref_type != CL_DEVICE_TYPE_ALL)
        err = clGetDeviceIDs(pf, CL_DEVICE_TYPE_ALL, 1, &g_dev, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: clGetDeviceIDs failed (err=%d)\n", err);
        return -1;
    }
    g_ctx = clCreateContext(NULL, 1, &g_dev, NULL, NULL, &err);
    if (err != CL_SUCCESS || !g_ctx) {
        fprintf(stderr, "OpenCL: clCreateContext failed (err=%d)\n", err);
        return -1;
    }
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    g_queue = clCreateCommandQueueWithProperties(g_ctx, g_dev, props, &err);
    if (err != CL_SUCCESS || !g_queue) {
        fprintf(stderr, "OpenCL: clCreateCommandQueue failed (err=%d)\n", err);
        clReleaseContext(g_ctx); g_ctx = NULL;
        return -1;
    }
    return 0;
}

static void ocl_cleanup(void) {
    if (g_queue) { clReleaseCommandQueue(g_queue); g_queue = NULL; }
    if (g_ctx)   { clReleaseContext(g_ctx); g_ctx = NULL; }
}

static void show_help(const char* prog) {
    fprintf(stderr, "LZO Hybrid (GPU+CPU) Compressor\n");
    fprintf(stderr, "Usage: %s [options] <input_file>\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -c                   Compress mode (default)\n");
    fprintf(stderr, "  -d, --decompress     Decompress mode\n");
    fprintf(stderr, "  -o, --output FILE    Output file\n");
    fprintf(stderr, "  -a, --alg ALG        Algorithm: lzo1x (default), lzo1y\n");
     fprintf(stderr, "  -L, --level LEVEL    Dictionary bits (8-20, 99=enhanced, 999=optimal, default: 12)\n");
    fprintf(stderr, "  -B, --block-size N   Block size (B/KB/MB, default: adaptive)\n");
    fprintf(stderr, "  --local N            OpenCL work-group size (default: 1)\n");
    fprintf(stderr, "  --cpu-threads N      CPU worker threads (default: auto = all cores)\n");
    fprintf(stderr, "  --gpu-ratio F        GPU block fraction 0.0-1.0 (default: 0.8)\n");
    fprintf(stderr, "  --adaptive           Enable adaptive per-file CPU/GPU split\n");
    fprintf(stderr, "  --sample-blocks N    Adaptive sample block count (default: 8)\n");
    fprintf(stderr, "  --bench [SECONDS]    Benchmark mode (compress+decompress+verify)\n");
    fprintf(stderr, "  --bench-io           Include file write/read in bench total throughput\n");
    fprintf(stderr, "  -v, --verbose        Verbose output\n");
    fprintf(stderr, "  -h, --help           Show this help\n");
}

int main(int argc, char** argv) {
    cl_int err;
    int decompress_mode = 0;
    int bench_mode = 0;
    int bench_include_io = 0;
    int verbose = 0;
    double bench_seconds = 3.0;
    const char* in_path = NULL;
    const char* output_path = NULL;
    const char* alg_name = "lzo1x";
    int comp_level = LZO_DEFAULT_COMP_LEVEL;
    size_t block_size = 0;
    int local_size = 0;
    int cpu_threads = 0;  /* 0 = auto-detect */
    double gpu_ratio = 0.8;
    int adaptive_mode = 0;
    size_t adaptive_sample_blocks = 8;
    int debug = 0;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            show_help(argv[0]); return 0;
        }
        if (strcmp(arg, "-c") == 0) { decompress_mode = 0; continue; }
        if (strcmp(arg, "-d") == 0 || strcmp(arg, "--decompress") == 0) {
            decompress_mode = 1; continue;
        }
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            verbose = 1; continue;
        }
        if (strcmp(arg, "--bench") == 0) {
            bench_mode = 1;
            if (i + 1 < argc && argv[i+1][0] != '-') bench_seconds = atof(argv[++i]);
            continue;
        }
        if (strcmp(arg, "--bench-io") == 0) {
            bench_include_io = 1;
            continue;
        }
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: -o requires argument\n"); return 1; }
            output_path = argv[i]; continue;
        }
        if (strcmp(arg, "-a") == 0 || strcmp(arg, "--alg") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: -a requires argument\n"); return 1; }
            if (strcmp(argv[i], "lzo1x") == 0 || strcmp(argv[i], "1x") == 0) alg_name = "lzo1x";
            else if (strcmp(argv[i], "lzo1y") == 0 || strcmp(argv[i], "1y") == 0) alg_name = "lzo1y";
            else { fprintf(stderr, "Error: unknown algorithm '%s'\n", argv[i]); return 1; }
            continue;
        }
        if (strcmp(arg, "-L") == 0 || strcmp(arg, "--level") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: -L requires argument\n"); return 1; }
             comp_level = atoi(argv[i]);
             if (comp_level != 999 && comp_level != 99 && (comp_level < 11 || comp_level > 16)) {
                 fprintf(stderr, "Error: level must be 11-16, 99, or 999 (got %d)\n", comp_level); return 1;
             }
            continue;
        }
        if (strcmp(arg, "-B") == 0 || strcmp(arg, "--block-size") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: -B requires argument\n"); return 1; }
            block_size = lzo_parse_block_size(argv[i]);
            continue;
        }
        if (strcmp(arg, "--local") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: --local requires argument\n"); return 1; }
            local_size = atoi(argv[i]); continue;
        }
        if (strncmp(arg, "--local=", 8) == 0) { local_size = atoi(arg + 8); continue; }
        if (strcmp(arg, "--cpu-threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: --cpu-threads requires argument\n"); return 1; }
            cpu_threads = atoi(argv[i]); continue;
        }
        if (strncmp(arg, "--cpu-threads=", 14) == 0) { cpu_threads = atoi(arg + 14); continue; }
        if (strcmp(arg, "--gpu-ratio") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: --gpu-ratio requires argument\n"); return 1; }
            gpu_ratio = atof(argv[i]); continue;
        }
        if (strncmp(arg, "--gpu-ratio=", 12) == 0) { gpu_ratio = atof(arg + 12); continue; }
        if (strcmp(arg, "--adaptive") == 0) {
            adaptive_mode = 1;
            continue;
        }
        if (strcmp(arg, "--sample-blocks") == 0) {
            if (++i >= argc) { fprintf(stderr, "Error: --sample-blocks requires argument\n"); return 1; }
            adaptive_sample_blocks = (size_t)strtoull(argv[i], NULL, 10);
            continue;
        }
        if (strncmp(arg, "--sample-blocks=", 16) == 0) {
            adaptive_sample_blocks = (size_t)strtoull(arg + 16, NULL, 10);
            continue;
        }
        if (arg[0] == '-') {
            fprintf(stderr, "Error: unknown option '%s'\n", arg); return 1;
        }
        if (!in_path) in_path = arg;
        else if (!output_path) output_path = arg;
        else { fprintf(stderr, "Error: too many positional arguments\n"); return 1; }
    }

    if (!in_path) {
        show_help(argv[0]);
        return 1;
    }

    if (gpu_ratio < 0.0) gpu_ratio = 0.0;
    if (gpu_ratio > 1.0) gpu_ratio = 1.0;
    if (cpu_threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        cpu_threads = (n > 0) ? (int)n : 4;
    }

    int alg_id = (strcmp(alg_name, "lzo1y") == 0) ? 1 : 0;

    hybrid_params_t params = {
        .alg_id = alg_id,
        .comp_level = comp_level,
        .block_size = block_size,
        .split_mode = adaptive_mode ? HYBRID_SPLIT_ADAPTIVE : HYBRID_SPLIT_FIXED,
        .gpu_ratio = gpu_ratio,
        .adaptive_sample_blocks = adaptive_sample_blocks,
        .cpu_threads = cpu_threads,
        .local_size = local_size,
        .standard_copy = 0,
        .debug = debug,
    };

    uint64_t t0 = now_ns();

    char build_log[8192] = {0};
    cl_program comp_prog = NULL;
    cl_kernel comp_krn = NULL;
    cl_kernel pack_krn = NULL;
    cl_program dec_prog = NULL;
    cl_kernel dec_krn = NULL;
    int kernel_has_dbg = 0;
    int use_opencl = !((gpu_ratio <= 0.0) && !adaptive_mode && cpu_threads > 0);
    if (use_opencl) {
        if (ocl_init() != 0) {
            fprintf(stderr, "Error: OpenCL initialization failed\n");
            return 1;
        }

        if (lzo_load_comp_kernel(g_ctx, g_dev, alg_name, comp_level, debug,
                                 &comp_prog, &comp_krn, &kernel_has_dbg,
                                 build_log, sizeof(build_log)) != 0) {
            fprintf(stderr, "Error: failed to load compression kernel: %s\n", build_log);
            ocl_cleanup();
            return 1;
        }

        pack_krn = clCreateKernel(comp_prog, "lzo_pack_compressed_blocks", &err);
        if (err != CL_SUCCESS) {
            pack_krn = NULL;
            if (verbose) {
                fprintf(stderr, "Warning: lzo_pack_compressed_blocks unavailable, hybrid compaction disabled (err=%d)\n", err);
            }
        }

        dec_prog = lzo_load_program_with_dbits(g_ctx, g_dev, alg_name, comp_level,
                                               build_log, sizeof(build_log));
        if (!dec_prog) {
            fprintf(stderr, "Error: failed to load decompression program: %s\n", build_log);
            if (pack_krn) clReleaseKernel(pack_krn);
            clReleaseKernel(comp_krn); clReleaseProgram(comp_prog);
            ocl_cleanup();
            return 1;
        }

        {
            char krn_name[64];
            snprintf(krn_name, sizeof(krn_name), "%s_block_decompress", alg_name);
            dec_krn = clCreateKernel(dec_prog, krn_name, &err);
            if (err != CL_SUCCESS || !dec_krn) {
                fprintf(stderr, "Error: failed to create decompression kernel '%s' (err=%d)\n", krn_name, err);
                if (pack_krn) clReleaseKernel(pack_krn);
                clReleaseProgram(dec_prog); clReleaseKernel(comp_krn); clReleaseProgram(comp_prog);
                ocl_cleanup();
                return 1;
            }
        }
    }

    hybrid_workspace_t ws;
    hybrid_workspace_init(&ws);

    int rc = 0;

    if (bench_mode) {
        rc = hybrid_bench(use_opencl ? g_ctx : NULL,
                          use_opencl ? g_queue : NULL,
                          use_opencl ? g_dev : NULL,
                          comp_krn, pack_krn, dec_krn,
                          in_path, &params, &ws, bench_seconds, bench_include_io);
    } else if (decompress_mode) {
        char default_out[512];
        if (!output_path) {
            size_t len = strlen(in_path);
            if (len > 4 && strcmp(in_path + len - 4, ".lzo") == 0) {
                snprintf(default_out, sizeof(default_out), "%.*s", (int)(len - 4), in_path);
            } else {
                snprintf(default_out, sizeof(default_out), "%s.dec", in_path);
            }
            output_path = default_out;
        }
        hybrid_timing_t timing = {0};
        rc = hybrid_decompress(use_opencl ? g_ctx : NULL,
                               use_opencl ? g_queue : NULL,
                               use_opencl ? g_dev : NULL,
                               dec_krn,
                               in_path, output_path, &params, &ws, &timing);
        uint64_t t1 = now_ns();
        if (rc == 0) {
            printf("%s : decompressed in %.2f ms (gpu_blocks=%zu cpu_blocks=%zu)\n",
                   in_path, (double)(t1 - t0) / 1e6, timing.gpu_blocks, timing.cpu_blocks);
            if (verbose) {
                printf("  gpu_kernel=%lu us  cpu_kernel=%lu us  upload=%lu us  download=%lu us\n",
                       timing.gpu_kernel_us, timing.cpu_kernel_us, timing.upload_us, timing.download_us);
            }
        }
    } else {
        char default_out[512];
        if (!output_path) {
            snprintf(default_out, sizeof(default_out), "%s.lzo", in_path);
            output_path = default_out;
        }
        hybrid_timing_t timing = {0};
        rc = hybrid_compress(use_opencl ? g_ctx : NULL,
                             use_opencl ? g_queue : NULL,
                             use_opencl ? g_dev : NULL,
                             comp_krn, pack_krn,
                             in_path, output_path, &params, &ws, &timing);
        uint64_t t1 = now_ns();
        if (rc == 0) {
            double ratio = (double)timing.in_size / (timing.out_size > 0 ? (double)timing.out_size : 1.0);
            printf("%s : %zu -> %zu (%.2f:1) in %.2f ms (gpu=%zu cpu=%zu blocks)\n",
                   in_path, timing.in_size, timing.out_size, ratio,
                   (double)(t1 - t0) / 1e6, timing.gpu_blocks, timing.cpu_blocks);
            if (verbose) {
                printf("  gpu_kernel=%lu us  cpu_kernel=%lu us  upload=%lu us  download=%lu us\n",
                       timing.gpu_kernel_us, timing.cpu_kernel_us, timing.upload_us, timing.download_us);
            }
        }
    }

    hybrid_workspace_free(&ws);
    if (pack_krn) clReleaseKernel(pack_krn);
    if (dec_krn) clReleaseKernel(dec_krn);
    if (dec_prog) clReleaseProgram(dec_prog);
    if (comp_krn) clReleaseKernel(comp_krn);
    if (comp_prog) clReleaseProgram(comp_prog);
    if (use_opencl) ocl_cleanup();

    return rc;
}
