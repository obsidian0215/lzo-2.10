#include "lzo_gpu_utils.h"
#include "lzo_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#else
#include <windows.h>
#endif
#include <limits.h>
#include <sys/stat.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int lzo_copy_string(char* dst, size_t dst_sz, const char* src) {
    size_t n;
    if (!dst || dst_sz == 0 || !src) return -1;
    n = strlen(src);
    if (n >= dst_sz) return -1;
    memcpy(dst, src, n + 1);
    return 0;
}

static int lzo_join_path2(char* out, size_t out_sz, const char* dir, const char* name) {
    size_t dl, nl;
    if (!out || out_sz == 0 || !dir || !name) return -1;
    dl = strlen(dir);
    nl = strlen(name);
    if (dl == 0 || nl == 0) return -1;
    if (dl + 1 + nl >= out_sz) return -1;
    memcpy(out, dir, dl);
    out[dl] = '/';
    memcpy(out + dl + 1, name, nl);
    out[dl + 1 + nl] = '\0';
    return 0;
}

static int lzo_get_executable_path(char* out, size_t outlen) {
    if (!out || outlen == 0) return -1;
#if defined(_WIN32) || defined(_WIN64)
    {
        DWORD n = GetModuleFileNameA(NULL, out, (DWORD)outlen);
        if (n == 0 || n >= outlen) return -1;
        out[n] = '\0';
        return 0;
    }
#else
    {
        ssize_t n = readlink("/proc/self/exe", out, outlen - 1);
        if (n <= 0) return -1;
        out[n] = '\0';
        return 0;
    }
#endif
}

static void lzo_strip_suffix_if_present(char* s, const char* suffix) {
    size_t slen;
    size_t suflen;
    if (!s || !suffix) return;
    slen = strlen(s);
    suflen = strlen(suffix);
    if (slen >= suflen && strcmp(s + slen - suflen, suffix) == 0) {
        s[slen - suflen] = '\0';
    }
}

static void lzo_compose_source_filename(char* out, size_t out_sz, const char* source_alg, int want_debug) {
    if (!out || out_sz == 0 || !source_alg || !*source_alg) return;
    if (want_debug) {
        snprintf(out, out_sz, "%s_debug.cl", source_alg);
    } else {
        snprintf(out, out_sz, "%s.cl", source_alg);
    }
}

static int lzo_stat_mtime(const char* path, time_t* out_mtime) {
    struct stat st;
    if (!path || !*path || !out_mtime) return -1;
    if (stat(path, &st) != 0) return -1;
    *out_mtime = st.st_mtime;
    return 0;
}

static int lzo_source_is_newer_than_binary(const char* source_path, const char* binary_path) {
    time_t src_mtime;
    time_t bin_mtime;
    if (lzo_stat_mtime(source_path, &src_mtime) != 0) return 0;
    if (lzo_stat_mtime(binary_path, &bin_mtime) != 0) return 0;
    return src_mtime > bin_mtime;
}

static int lzo_validate_program_kernels(cl_program prog, const char* source_alg, int decomp_only) {
    cl_int err = CL_SUCCESS;
    cl_kernel kcomp = NULL;
    cl_kernel kdec = NULL;
    char comp_name[96];
    char dec_name[96];

    if (!prog || !source_alg || !*source_alg) return 0;

    snprintf(dec_name, sizeof(dec_name), "%s_block_decompress", source_alg);

    if (!decomp_only) {
        snprintf(comp_name, sizeof(comp_name), "%s_block_compress", source_alg);

        kcomp = clCreateKernel(prog, comp_name, &err);
        if (err != CL_SUCCESS || !kcomp) {
            if (kcomp) clReleaseKernel(kcomp);
            return 0;
        }
    }

    kdec = clCreateKernel(prog, dec_name, &err);
    if (kcomp) clReleaseKernel(kcomp);
    if (err != CL_SUCCESS || !kdec) {
        if (kdec) clReleaseKernel(kdec);
        return 0;
    }
    clReleaseKernel(kdec);
    return 1;
}

static int lzo_resolve_output_binary_path(const char* bin_name, const char* resolved_src, char* out_path, size_t out_path_sz) {
    char found_path[PATH_MAX];
    if (!bin_name || !*bin_name || !out_path || out_path_sz == 0) return -1;

    if (lzo_find_file_path(bin_name, found_path, sizeof(found_path)) == 0) {
        if (lzo_copy_string(out_path, out_path_sz, found_path) != 0) return -1;
        return 0;
    }

    if (resolved_src && *resolved_src) {
        if (lzo_copy_string(out_path, out_path_sz, resolved_src) != 0) return -1;
        {
            char* slash = strrchr(out_path, '/');
            if (slash) {
                *(slash + 1) = '\0';
                if (strlen(out_path) + strlen(bin_name) >= out_path_sz) return -1;
                strcat(out_path, bin_name);
                return 0;
            }
        }
    }

    if (lzo_copy_string(out_path, out_path_sz, bin_name) != 0) return -1;
    return 0;
}

static void lzo_refresh_program_binary(cl_program prog, const char* bin_path) {
    size_t bin_sz = 0;
    unsigned char* blob = NULL;
    unsigned char* bins[1] = { NULL };
    FILE* bf = NULL;

    if (!prog || !bin_path || !*bin_path) return;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_sz, NULL) != CL_SUCCESS) return;
    if (bin_sz == 0) return;

    blob = (unsigned char*)malloc(bin_sz);
    if (!blob) return;
    bins[0] = blob;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bins), bins, NULL) != CL_SUCCESS) {
        free(blob);
        return;
    }

    bf = fopen(bin_path, "wb");
    if (bf) {
        (void)fwrite(blob, 1, bin_sz, bf);
        fclose(bf);
    }
    free(blob);
}

unsigned long g_ocl_init_us = 0;
unsigned long g_kernel_load_us = 0;
unsigned long g_ocl_setup_us = 0; /* Not used but for safety */

static cl_int lzo_try_get_device_shared(cl_platform_id *platforms,
                                        cl_uint num_platforms,
                                        cl_device_type dtype,
                                        cl_device_id *out_dev,
                                        cl_platform_id *out_pf)
{
    for (cl_uint pi = 0; pi < num_platforms; pi++) {
        cl_device_id tmp_dev = NULL;
        cl_int r = clGetDeviceIDs(platforms[pi], dtype, 1, &tmp_dev, NULL);
        if (r == CL_SUCCESS && tmp_dev != NULL) {
            *out_dev = tmp_dev;
            *out_pf = platforms[pi];
            return CL_SUCCESS;
        }
    }
    return CL_DEVICE_NOT_FOUND;
}

static cl_device_type lzo_preferred_opencl_device_type_shared(void)
{
    const char* pref = getenv("FORCE_OPENCL_DEVICE");
    if (!pref || !*pref) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "CPU") == 0) return CL_DEVICE_TYPE_CPU;
    if (strcasecmp(pref, "GPU") == 0) return CL_DEVICE_TYPE_GPU;
    if (strcasecmp(pref, "DEFAULT") == 0) return CL_DEVICE_TYPE_DEFAULT;
    if (strcasecmp(pref, "ALL") == 0) return CL_DEVICE_TYPE_ALL;
    return CL_DEVICE_TYPE_GPU;
}

cl_int lzo_select_opencl_platform_device(cl_platform_id* out_pf, cl_device_id* out_dev)
{
    cl_uint num_platforms = 0;
    cl_platform_id* platforms = NULL;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_int r = CL_DEVICE_NOT_FOUND;
    cl_device_type pref_type = lzo_preferred_opencl_device_type_shared();

    if (!out_pf || !out_dev) return CL_INVALID_VALUE;
    *out_pf = NULL;
    *out_dev = NULL;

    if (err != CL_SUCCESS || num_platforms == 0) return CL_DEVICE_NOT_FOUND;

    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms) return CL_OUT_OF_HOST_MEMORY;

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return err;
    }

    if (pref_type == CL_DEVICE_TYPE_GPU) {
        r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
    } else if (pref_type == CL_DEVICE_TYPE_CPU) {
        r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
    } else if (pref_type == CL_DEVICE_TYPE_DEFAULT) {
        r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_DEFAULT, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_GPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_CPU, out_dev, out_pf);
        if (r != CL_SUCCESS) r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
    } else {
        r = lzo_try_get_device_shared(platforms, num_platforms, CL_DEVICE_TYPE_ALL, out_dev, out_pf);
    }

    free(platforms);
    return r;
}

const char* format_size_lzo(size_t size) {
    static char buf[32];
    if (size < 1024) snprintf(buf, sizeof(buf), "%zu B", size);
    else if (size < 1024 * 1024) snprintf(buf, sizeof(buf), "%.2f KB", size / 1024.0);
    else if (size < 1024 * 1024 * 1024) snprintf(buf, sizeof(buf), "%.2f MB", size / (1024.0 * 1024.0));
    else snprintf(buf, sizeof(buf), "%.2f GB", size / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

static void print_time_row(FILE *f, const char *label, unsigned long us, unsigned long total_us) {
    double ms = us / 1000.0;
    double pct = total_us > 0 ? (100.0 * us / total_us) : 0.0;
    fprintf(f, "%-22s : %10.3f ms (%6.2f%%)\n", label, ms, pct);
}

/* Implementation of lzo_find_file_path */
int lzo_find_file_path(const char *name, char *out, size_t outlen)
{
    char path[PATH_MAX];
    char base[PATH_MAX];
    /* 1: LZO_GPU_DIR (Highest priority for testing) */
    const char *env = getenv("LZO_GPU_DIR");
    if (env && env[0]) {
        if (lzo_join_path2(path, sizeof(path), env, name) == 0 && access(path, R_OK) == 0) {
            if (lzo_copy_string(out, outlen, path) == 0) return 0;
        }
    }
    /* 2: exe dir and exe_dir/../lzo_gpu */
    char exe_path[PATH_MAX] = {0};
    if (lzo_get_executable_path(exe_path, sizeof(exe_path)) == 0) {
        char *slash = strrchr(exe_path,
#if defined(_WIN32) || defined(_WIN64)
                              '\\'
#else
                              '/'
#endif
        );
        if (!slash) slash = strrchr(exe_path, '/');
        if (slash) {
            *slash = '\0';
            if (lzo_join_path2(path, sizeof(path), exe_path, name) == 0 && access(path, R_OK) == 0) {
                if (lzo_copy_string(out, outlen, path) == 0) return 0;
            }
            snprintf(path, sizeof(path), "%s/../lzo_gpu/%s", exe_path, name);
            if (access(path, R_OK) == 0) { if (lzo_copy_string(out, outlen, path) == 0) return 0; }
        }
    }
    /* 3: OUT_DIR */
    env = getenv("OUT_DIR");
    if (env && env[0]) {
        if (lzo_join_path2(path, sizeof(path), env, name) == 0 && access(path, R_OK) == 0) {
            if (lzo_copy_string(out, outlen, path) == 0) return 0;
        }
    }
    /* 4: cwd */
    if (getcwd(base, sizeof(base)) != NULL) {
        if (lzo_join_path2(path, sizeof(path), base, name) == 0 && access(path, R_OK) == 0) {
            if (lzo_copy_string(out, outlen, path) == 0) return 0;
        }
    }
    /* 5: fallback: raw name in cwd */
    if (access(name, R_OK) == 0) { if (lzo_copy_string(out, outlen, name) == 0) return 0; }
    return -1;
}

/* Read a file into malloc'd buffer. Return NULL on error. */
char* lzo_read_file(const char *path, size_t *sz_out)
{
    FILE* fp;
    char* buf;
    long sz;

    if (!path) return NULL;
    if (strcmp(path, "-") == 0) {
        /* read from stdin */
        fp = stdin;
        size_t buf_size = 1024 * 1024; /* 1MB */
        size_t current_size = 0;
        buf = malloc(buf_size);
        if (!buf) return NULL;
        size_t bytes_read = 0;
        while ((bytes_read = fread(buf + current_size, 1, 1024, fp)) > 0) {
            current_size += bytes_read;
            if (current_size + 1024 >= buf_size) {
                size_t new_size = buf_size * 2;
                char *tmp = realloc(buf, new_size);
                if (!tmp) { free(buf); return NULL; }
                buf = tmp; buf_size = new_size;
            }
        }
        if (sz_out) *sz_out = current_size;
        buf[current_size] = '\0';
        return buf;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        char resolved[PATH_MAX];
        if (lzo_find_file_path(path, resolved, sizeof(resolved)) == 0) {
            fp = fopen(resolved, "rb");
            if (!fp) {
                return NULL;
            }
        } else {
            return NULL;
        }
    }
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf = malloc(sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, sz, fp) != (size_t) sz) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    buf[sz] = '\0';
    if (sz_out) *sz_out = (size_t)sz;
    return buf;
}
static char g_override_socket_path[512] = {0};
static char g_override_pid_path[512] = {0};

void lzo_set_daemon_socket_path(const char* path) {
    if (path) strncpy(g_override_socket_path, path, sizeof(g_override_socket_path)-1);
}

void lzo_set_daemon_pidfile_path(const char* path) {
    if (path) strncpy(g_override_pid_path, path, sizeof(g_override_pid_path)-1);
}

const char* lzo_daemon_socket_path(void) {
    if (g_override_socket_path[0]) return g_override_socket_path;
    const char* env = getenv("LZO_DAEMON_SOCKET");
    if (env && env[0]) return env;
    return "/tmp/lzo_gpu_daemon.sock";
}

const char* lzo_daemon_pidfile_path(void) {
    if (g_override_pid_path[0]) return g_override_pid_path;
    const char* env = getenv("LZO_DAEMON_PID");
    if (env && env[0]) return env;
    return "/tmp/lzo_gpu_daemon.pid";
}

/* lzo_utils: adaptive entropy/blocking helpers */
double lzo_calc_entropy(const unsigned char* data, size_t size)
{
    if (!data || size == 0) return 0.0;

    /* Optimization: Use continuous sampling from file beginning instead of strided sampling
     * across entire file. This avoids cache misses from jumping across 48MB+ files.
     * For entropy estimation, continuous head sample is representative and much faster.
     */
    size_t sample_size = (size < LZO_ADAPTIVE_SAMPLE_SIZE) ? size : LZO_ADAPTIVE_SAMPLE_SIZE;

    unsigned int freq[256] = {0};
    for (size_t i = 0; i < sample_size; ++i) {
        freq[data[i]]++;
    }

    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / (double)sample_size;
        entropy -= p * log2(p);
    }
    return entropy;
}

size_t lzo_parse_block_size(const char* str) {
    if (!str || !*str) return 0;
    char* end;
    double val = strtod(str, &end);
    if (end == str) return 0;
    while (*end == ' ' || *end == '\t') end++;
    size_t multiplier = 1; /* default to Bytes if pure numeric, for precision */
    if (*end == '\0') {
        multiplier = 1;
    } else if (strcasecmp(end, "B") == 0 || strcasecmp(end, "BYTES") == 0) {
        multiplier = 1;
    } else if (strcasecmp(end, "K") == 0 || strcasecmp(end, "KB") == 0) {
        multiplier = 1024;
    } else if (strcasecmp(end, "M") == 0 || strcasecmp(end, "MB") == 0) {
        multiplier = 1024 * 1024;
    } else if (strcasecmp(end, "G") == 0 || strcasecmp(end, "GB") == 0) {
        multiplier = 1024 * 1024 * 1024;
    }
    size_t result = (size_t)(val * (double)multiplier);
    return result;
}

/* Detect whether the user-supplied size string explicitly specified bytes
 * (e.g., "256B" or "256bytes"). Returns 1 if the unit is bytes, 0 otherwise.
 */
int lzo_specified_unit_is_bytes(const char* str) {
    if (!str || !*str) return 0;
    char* end;
    (void)strtod(str, &end);
    while (*end == ' ' || *end == '\t') end++;
    if (*end == '\0') return 1; /* Pure numeric counts as explicitly bytes */
    if (strcasecmp(end, "B") == 0 || strcasecmp(end, "BYTES") == 0) return 1;
    return 0;
}

/* Calculate adaptive block size based ONLY on file size and CU count (fast, no entropy)
 *
 * Updated based on comprehensive kernel testing (64 samples, 1482 configs):
 * - Small (<5MB): 8KB balanced, Level 10
 * - Medium (5-47MB): 16KB balanced, Level 10
 * - Large (47-134MB): 32KB optimal, Level 11 (peak: 1736 MB/s)
 * - XLarge (>134MB): 32KB optimal, Level 12
 *
 * Parallelism increased to 128-256x CU minimum (was 256x) for better GPU utilization.
 */
size_t lzo_adaptive_block_size(size_t in_sz, cl_uint cu)
{
    if (cu == 0) cu = 1;

    /* Target: 192x CU count (middle ground between 128-256x for good occupancy) */
    size_t target_nblk = (size_t)cu * LZO_OCC_FACTOR_DEFAULT;

    /* Calculate block size from target block count */
    size_t base_block = (in_sz + target_nblk - 1) / target_nblk;

    /* Apply size-based bounds matching Level 12-18 test results (matrix_sweep 2026/01/11) */
    size_t size_based_min, size_based_max;

    if (in_sz >= 8 * 1024 * 1024) {
        /* Large files (8MB+): 64KB is the sweet spot for speed (339MB/s) and ratio */
        size_based_min = 64 * 1024; size_based_max = 64 * 1024;
    } else if (in_sz >= 1 * 1024 * 1024) {
        /* Medium files (1-8MB): 32KB yields best compression depth based on sweep */
        size_based_min = 32 * 1024; size_based_max = 32 * 1024;
    } else if (in_sz >= 128 * 1024) {
        /* Small files (128KB - 1MB): 16KB for balanced efficiency */
        size_based_min = 16 * 1024; size_based_max = 16 * 1024;
    } else {
        /* Tiny files (<128KB): 8KB to minimize transfer latency and allow small parallelism */
        size_based_min = 8 * 1024; size_based_max = 8 * 1024;
    }
    /* Clamp to size-based range */
    if (base_block < size_based_min) base_block = size_based_min;
    if (base_block > size_based_max) base_block = size_based_max;

    /* Align to specified boundary (4KB) for optimal coalesced memory access */
    base_block = (base_block + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);

    /* Ensure within absolute bounds */
    if (base_block < LZO_MIN_BLOCK_BYTES_DEFAULT) base_block = LZO_MIN_BLOCK_BYTES_DEFAULT;
    if (base_block > LZO_MAX_BLOCK_BYTES_DEFAULT) base_block = LZO_MAX_BLOCK_BYTES_DEFAULT;

    return base_block;
}

/* Calculate adaptive block size WITH entropy calculation (slower but more optimal) */
size_t lzo_adaptive_block_size_with_entropy(const unsigned char* data, size_t in_sz, cl_uint cu, double* entropy_out, int debug)
{
    if (cu == 0) cu = 1;

    double entropy = 0.0;
    if (data && in_sz > 0) {
        entropy = lzo_calc_entropy(data, in_sz);
        if (debug) {
            fprintf(stderr, "[ADAPTIVE] Calculated entropy: %.4f\n", entropy);
        }
    }
    if (entropy_out) *entropy_out = entropy;

    /* Target block count based on GPU parallelism and entropy:
     * Increased from previous (64x/256x/512x) to (128x/256x/512x) for better utilization
     * - High entropy (incompressible): fewer larger blocks but still 128x CU minimum
     * - Low entropy (highly compressible): more smaller blocks (512x CU) for better parallelism
     */
    size_t target_nblk;
    if (entropy > LZO_ADAPTIVE_HIGH_ENTROPY) {
        target_nblk = (size_t)cu * 128; /* High entropy -> 128x minimum (was 64x) */
    } else if (entropy > (LZO_ADAPTIVE_LOW_ENTROPY + LZO_ADAPTIVE_HIGH_ENTROPY) / 2.0) {
        target_nblk = (size_t)cu * 256; /* Medium entropy -> balanced */
    } else {
        target_nblk = (size_t)cu * 512; /* Low entropy -> smaller blocks, more parallelism */
    }

    /* Calculate block size from target count */
    size_t base_block = (in_sz + target_nblk - 1) / target_nblk;

    /* Apply size-based bounds matching test results (kernel_analysis_20251225_034050) */
    size_t size_based_min, size_based_max;

    if (in_sz >= 50 * 1024 * 1024) {
        /* XLarge/Large files (50MB+): Allow up to 128KB+ for consistency */
        size_based_min = 32 * 1024; size_based_max = 128 * 1024;
        /* For files > 100MB, we bias more towards large blocks */
        if (in_sz >= 100 * 1024 * 1024) {
             size_based_min = 64 * 1024;
             size_based_max = 256 * 1024;
        }
    } else if (in_sz >= 10 * 1024 * 1024) {
        /* Large files (10-50MB): 32KB optimal balance */
        size_based_min = 24 * 1024; size_based_max = 36 * 1024;
    } else if (in_sz >= 1 * 1024 * 1024) {
        /* Medium files (1-10MB): 8-16KB balanced (test: 8KB→978 MB/s, 16KB→1346 MB/s) */
        size_based_min = 8 * 1024; size_based_max = 20 * 1024;
    } else {
        /* Very small files (<1MB): 1-4KB blocks (test: 1KB→313 MB/s fastest) */
        size_based_min = 1 * 1024; size_based_max = 4 * 1024;
    }

    /* Entropy-based adjustment: high entropy prefers larger blocks within the range */
    if (entropy > LZO_ADAPTIVE_HIGH_ENTROPY) {
        /* High entropy: use upper 75% of the range */
        size_t mid_high = size_based_min + ((size_based_max - size_based_min) * 3) / 4;
        if (base_block < mid_high) base_block = mid_high;
    } else if (entropy < LZO_ADAPTIVE_LOW_ENTROPY) {
        /* Low entropy: use lower 75% of the range */
        size_t mid_low = size_based_min + ((size_based_max - size_based_min) * 1) / 4;
        if (base_block > mid_low) base_block = mid_low;
    }

    /* Clamp to size-based range */
    if (base_block < size_based_min) base_block = size_based_min;
    if (base_block > size_based_max) base_block = size_based_max;

    /* Align to 4KB boundary */
    base_block = (base_block + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);

    /* Ensure within absolute bounds */
    if (base_block < LZO_MIN_BLOCK_BYTES_DEFAULT) base_block = LZO_MIN_BLOCK_BYTES_DEFAULT;
    if (base_block > LZO_MAX_BLOCK_BYTES_DEFAULT) base_block = LZO_MAX_BLOCK_BYTES_DEFAULT;

    return base_block;
}

void lzo_choose_blocking_adaptive(const unsigned char* data, size_t in_sz, cl_device_id dev, size_t blk_bytes, int fixed_exact, size_t* blk_sz_out, size_t* nblk_out, int debug)
{
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* CLI or env override: fixed block size */
    if (blk_bytes > 0) {
        size_t blk = blk_bytes;
        int allow_unaligned = fixed_exact;
        /* Environment override: allow explicit unaligned blocks by setting LZO_ALLOW_UNALIGNED_BLOCKS=1 */
        const char* env = getenv("LZO_ALLOW_UNALIGNED_BLOCKS");
        if (!allow_unaligned && env && atoi(env) == 1) allow_unaligned = 1;

        if (!allow_unaligned) {
            /* Align to the configured alignment granularity and enforce sensible minimums */
            size_t align = LZO_ALIGN_BYTES_DEFAULT;
            blk = (blk + (align - 1)) & ~(align - 1);
            if (blk < LZO_MIN_BLOCK_BYTES_DEFAULT) blk = LZO_MIN_BLOCK_BYTES_DEFAULT;
        } else {
            /* exact bytes requested; accept small values (but guard against zero) */
            if (blk == 0) blk = 1;
        }

        if (blk > LZO_MAX_BLOCK_BYTES_DEFAULT) blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
        size_t nblk = (in_sz + blk - 1) / blk;
        *blk_sz_out = blk;
        *nblk_out = nblk;
        return;
    }

    /* Check if entropy-based calculation is enabled */
    int use_entropy = LZO_ADAPTIVE_ENTROPY_ENABLED;
    const char* entropy_env = getenv("LZO_ADAPTIVE_ENTROPY");
    if (entropy_env) {
        use_entropy = atoi(entropy_env);
    }

    /* Calculate optimal block size using adaptive algorithms */
    size_t blk;
    if (use_entropy && data && in_sz > 0) {
        /* Entropy-aware calculation (slower but potentially better block size) */
        double entropy = 0.0;
        blk = lzo_adaptive_block_size_with_entropy(data, in_sz, cu, &entropy, debug);
        if (debug) {
            fprintf(stderr, "[ADAPTIVE] Using entropy-aware heuristic (entropy=%.4f)\n", entropy);
        }
    } else {
        /* Fast size-only calculation (default) */
        blk = lzo_adaptive_block_size(in_sz, cu);
        if (debug) {
            fprintf(stderr, "[ADAPTIVE] Using size-only heuristic (entropy calculation disabled)\n");
        }
    }

    /* Calculate number of blocks and apply safety bounds */
    size_t nblk = (in_sz + blk - 1) / blk;

    /* Enforce absolute limits */
    if (nblk > LZO_MAX_NBLOCKS_DEFAULT) {
        nblk = LZO_MAX_NBLOCKS_DEFAULT;
        blk = (in_sz + nblk - 1) / nblk;
        blk = (blk + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);
        if (blk < LZO_MIN_BLOCK_BYTES_DEFAULT) blk = LZO_MIN_BLOCK_BYTES_DEFAULT;
        if (blk > LZO_MAX_BLOCK_BYTES_DEFAULT) blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
        nblk = (in_sz + blk - 1) / blk;
    }

    /* Handle tiny tail blocks by redistributing */
    if (nblk > 1) {
        size_t last_blk_size = in_sz - blk * (nblk - 1);
        if (last_blk_size < blk / 4) {
            /* Last block too small, recalculate with one less block */
            nblk--;
            if (nblk > 0) {
                blk = (in_sz + nblk - 1) / nblk;
                blk = (blk + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);
                if (blk < LZO_MIN_BLOCK_BYTES_DEFAULT) blk = LZO_MIN_BLOCK_BYTES_DEFAULT;
                if (blk > LZO_MAX_BLOCK_BYTES_DEFAULT) blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
                nblk = (in_sz + blk - 1) / blk;
            }
        }
    }

    if (debug) {
        fprintf(stderr, "[ADAPTIVE] Final: block_size=%zu blocks=%zu (file=%zu CUs=%u)\n",
                blk, nblk, in_sz, cu);
    }

    *blk_sz_out = blk;
    *nblk_out = nblk;
}

int lzo_dict_u16_clear_for_block(size_t block_size, int bits)
{
    return (bits <= 16 && block_size > 0 && block_size <= 64U * 1024U) ? 1 : 0;
}

size_t lzo_dict_entry_bytes_for_block(size_t block_size, int bits)
{
    return lzo_dict_u16_clear_for_block(block_size, bits) ? sizeof(uint16_t) : sizeof(uint32_t);
}

/* Load an OpenCL program from an available precompiled binary or compile from source.
 * Does NOT exit on failure; returns NULL on error and fills build_log (if provided).
 */
cl_program lzo_load_program_with_dbits_and_block(cl_context ctx, cl_device_id dev, const char* alg_name, int bits, size_t block_size, char *build_log, size_t build_log_len)
{
    cl_int err;
    cl_program prog = NULL;
    int want_debug = 0;
    int decomp_only = 0;
    int allow_clbin = 0;
    char base_name[64];
    char source_alg_name[64];
    char source_name_for_stale[128] = {0};
    char source_path_for_stale[PATH_MAX] = {0};
    int source_path_for_stale_ok = 0;

    strncpy(base_name, alg_name, sizeof(base_name) - 1);
    base_name[sizeof(base_name) - 1] = '\0';

    /* Guard against invalid D_BITS, which can cause unstable kernel builds. */
    if (bits < 11 || bits > 15) {
        bits = LZO_DEFAULT_COMP_LEVEL;
    }

    {
        char *debug_pos = strstr(base_name, "_debug");
        if (debug_pos) {
            *debug_pos = '\0';
            want_debug = 1;
        }
    }

    {
        char *decomp_pos = strstr(base_name, "_decomp");
        if (decomp_pos && decomp_pos[7] == '\0') {
            *decomp_pos = '\0';
            decomp_only = 1;
        }
    }

    strncpy(source_alg_name, base_name, sizeof(source_alg_name) - 1);
    source_alg_name[sizeof(source_alg_name) - 1] = '\0';
    lzo_strip_suffix_if_present(source_alg_name, "_decomp");
    lzo_compose_source_filename(source_name_for_stale, sizeof(source_name_for_stale), source_alg_name, want_debug);
    if (lzo_find_file_path(source_name_for_stale, source_path_for_stale, sizeof(source_path_for_stale)) == 0) {
        source_path_for_stale_ok = 1;
    }

    {
        const char* use_clbin_env = getenv("LZO_GPU_USE_CLBIN");
        allow_clbin = (!want_debug && use_clbin_env && strcmp(use_clbin_env, "1") == 0);
    }

    /* Try bits-specific binary first: <alg>_<bits>.clbin */
    if (allow_clbin) {
        char bin_name[128];
        if (bits > 0) snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", base_name, bits);
        else snprintf(bin_name, sizeof(bin_name), "%s.clbin", base_name);

        char resolved_bin[PATH_MAX];
        if (lzo_find_file_path(bin_name, resolved_bin, sizeof(resolved_bin)) == 0) {
            if (!(source_path_for_stale_ok && lzo_source_is_newer_than_binary(source_path_for_stale, resolved_bin))) {
                FILE* fb = fopen(resolved_bin, "rb");
                if (fb) {
                    fseek(fb, 0, SEEK_END);
                    long bsz = ftell(fb);
                    fseek(fb, 0, SEEK_SET);
                    if (bsz > 0) {
                        size_t bin_size = (size_t)bsz;
                        unsigned char* bin = malloc(bin_size);
                        if (bin && fread(bin, 1, bin_size, fb) == bin_size) {
                            cl_int binary_status;
                            const unsigned char* bin_ptr = bin;
                            prog = clCreateProgramWithBinary(ctx, 1, &dev, &bin_size,
                                                            &bin_ptr, &binary_status, &err);
                            if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                                /* Rebuild with the same fast-math flags used by build_kernel */
                                const char *opts = "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable";
                                err = clBuildProgram(prog, 1, &dev, opts, NULL, NULL);
                                if (err == CL_SUCCESS && lzo_validate_program_kernels(prog, source_alg_name, decomp_only)) {
                                    if (build_log) build_log[0] = '\0';
                                    free(bin); fclose(fb); return prog;
                                } else {
                                    /* capture build log */
                                    size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                                    if (log_sz && build_log) {
                                        size_t toread = (log_sz < build_log_len - 1) ? log_sz : build_log_len - 1;
                                        char *tmp = malloc(toread + 1);
                                        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, toread, tmp, NULL);
                                        tmp[toread] = '\0';
                                        snprintf(build_log, build_log_len, "Binary build log: %s", tmp);
                                        free(tmp);
                                    } else if (build_log) {
                                        snprintf(build_log, build_log_len, "Binary build failed or missing kernels (err=%d)", (int)err);
                                    }
                                    clReleaseProgram(prog); prog = NULL;
                                }
                            }
                        } else if (build_log) {
                            snprintf(build_log, build_log_len, "Failed to read binary kernel %s", resolved_bin);
                        }
                        if (bin) free(bin);
                    }
                    fclose(fb);
                }
            }
        }
    }

    /* Fallback: compile from unified source.
     * We map both "lzo1x" and "lzo1x_decomp" to lzo1x(.debug).cl,
     * same for lzo1y, so compression/decompression stay in one source unit.
     */
    {
        char resolved_src[PATH_MAX]; size_t src_len = 0; char* src = NULL;
        int dbg_flag = want_debug;
        char source_alg[64];
        char base_src[128];
        int found_src = 0;

        strncpy(source_alg, base_name, sizeof(source_alg) - 1);
        source_alg[sizeof(source_alg) - 1] = '\0';
        {
            char* decomp_pos = strstr(source_alg, "_decomp");
            if (decomp_pos && decomp_pos[7] == '\0') {
                *decomp_pos = '\0';
            }
        }

        snprintf(base_src, sizeof(base_src), "%s.cl", source_alg);

        /* Try to find and load the base source */
        if (lzo_find_file_path(base_src, resolved_src, sizeof(resolved_src)) == 0) {
            src = lzo_read_file(resolved_src, &src_len);
            found_src = 1;
        }

        if (!found_src) {
            if (build_log) snprintf(build_log, build_log_len, "source file %s not found", base_src);
            return NULL;
        }

        if (!src) {
            if (build_log) snprintf(build_log, build_log_len, "failed to read %s", base_src);
            return NULL;
        } else {
            /* source loaded */
        }

        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
        if (err != CL_SUCCESS) { if (build_log) snprintf(build_log, build_log_len, "clCreateProgramWithSource failed (err=%d)", err); free(src); return NULL; }

        /* Build with appropriate macros based on variant flags.
         * Use the resolved source directory as the primary -I path so that
         * #include "lzo_gpu.h" etc. are found regardless of CWD. */
        char src_dir[PATH_MAX];
        strncpy(src_dir, resolved_src, sizeof(src_dir) - 1);
        src_dir[sizeof(src_dir) - 1] = '\0';
        {
            char *sl = strrchr(src_dir, '/');
            if (sl) *sl = '\0'; else src_dir[0] = '.', src_dir[1] = '\0';
        }
        char build_opts[512];
        int dict_u16_clear = lzo_dict_u16_clear_for_block(block_size, bits);
        if (dbg_flag) {
           snprintf(build_opts, sizeof(build_opts),
               "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable -I\"%s\" -I. -I./lzo_gpu -I../lzo_gpu -I.. -D D_BITS=%d -D LZO_DICT_U16_CLEAR=%d -D LZO_GPU_DEBUG_COUNTERS_RUNTIME=1",
               src_dir, bits, dict_u16_clear);
        } else {
           snprintf(build_opts, sizeof(build_opts),
               "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable -I\"%s\" -I. -I./lzo_gpu -I../lzo_gpu -I.. -D D_BITS=%d -D LZO_DICT_U16_CLEAR=%d",
               src_dir, bits, dict_u16_clear);
        }
        err = clBuildProgram(prog, 1, &dev, build_opts, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
            char* log = malloc(log_sz + 1);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            log[log_sz] = '\0';
            if (build_log) {
                size_t tocopy = (log_sz < build_log_len - 1) ? log_sz : build_log_len - 1;
                strncpy(build_log, log, tocopy); build_log[tocopy] = '\0';
            }
            free(log); free(src); clReleaseProgram(prog);
            return NULL;
        }
        free(src);

        if (allow_clbin && !dbg_flag) {
            char bin_name[128];
            char bin_path[PATH_MAX];
            if (bits > 0) snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", source_alg, bits);
            else snprintf(bin_name, sizeof(bin_name), "%s.clbin", source_alg);
            if (lzo_resolve_output_binary_path(bin_name, resolved_src, bin_path, sizeof(bin_path)) == 0) {
                lzo_refresh_program_binary(prog, bin_path);
            }
        }

        if (build_log) build_log[0] = '\0';
        return prog;
    }
}

cl_program lzo_load_program_with_dbits(cl_context ctx, cl_device_id dev, const char* alg_name, int bits, char *build_log, size_t build_log_len)
{
    return lzo_load_program_with_dbits_and_block(ctx, dev, alg_name, bits, 0, build_log, build_log_len);
}

/* Load and create a compression kernel (see header for behavior). */
int lzo_load_comp_kernel(cl_context ctx, cl_device_id dev, const char *alg_name, int comp_level, int debug, cl_program *out_prog, cl_kernel *out_krn, int *kernel_has_dbg, char *build_log, size_t build_log_len)
{
    return lzo_load_comp_kernel_for_block(ctx, dev, alg_name, comp_level, 0, debug, out_prog, out_krn, kernel_has_dbg, build_log, build_log_len);
}

int lzo_load_comp_kernel_for_block(cl_context ctx, cl_device_id dev, const char *alg_name, int comp_level, size_t block_size, int debug, cl_program *out_prog, cl_kernel *out_krn, int *kernel_has_dbg, char *build_log, size_t build_log_len)
{
    cl_int err;
    cl_program prog = NULL;
    cl_kernel krn = NULL;
    char effective_alg[128];
    char prog_base_name[128];

    strncpy(prog_base_name, alg_name, sizeof(prog_base_name)-1);
    prog_base_name[sizeof(prog_base_name)-1] = '\0';
    char* p_dbg = strstr(prog_base_name, "_debug");
    if (p_dbg) *p_dbg = '\0';

    if (debug) {
        static const char debug_suffix[] = "_debug";
        size_t base_len = strlen(prog_base_name);
        size_t suffix_len = sizeof(debug_suffix) - 1;
        if (base_len + suffix_len >= sizeof(effective_alg)) {
            if (build_log && build_log_len > 0) {
                snprintf(build_log, build_log_len, "debug program name too long for %s", prog_base_name);
            }
            return -1;
        }
        memcpy(effective_alg, prog_base_name, base_len);
        memcpy(effective_alg + base_len, debug_suffix, suffix_len + 1);
    } else {
        strncpy(effective_alg, prog_base_name, sizeof(effective_alg)-1);
        effective_alg[sizeof(effective_alg)-1] = '\0';
    }

    /* Load base algorithm program */
    prog = lzo_load_program_with_dbits_and_block(ctx, dev, effective_alg, comp_level, block_size, build_log, build_log_len);
    if (!prog) {
        if (build_log && build_log[0] == '\0') snprintf(build_log, build_log_len, "failed to build kernel for %s", effective_alg);
        return -1;
    }

    char krn_name[192];
    {
        const char* suffix = "_block_compress";
        size_t base_len = strlen(prog_base_name);
        size_t suffix_len = strlen(suffix);
        if (base_len + suffix_len >= sizeof(krn_name)) {
            if (build_log && build_log[0] == '\0') {
                snprintf(build_log, build_log_len, "kernel name too long for %s", prog_base_name);
            }
            clReleaseProgram(prog);
            return -1;
        }
        memcpy(krn_name, prog_base_name, base_len);
        memcpy(krn_name + base_len, suffix, suffix_len + 1);
    }

    krn = clCreateKernel(prog, krn_name, &err);
    if (err != CL_SUCCESS) {
        if (build_log && build_log[0] == '\0') snprintf(build_log, build_log_len, "clCreateKernel failed for %s (err=%d)", krn_name, (int)err);
        clReleaseProgram(prog);
        return -1;
    }

    cl_uint num_args = 0;
    cl_int rc_g = clGetKernelInfo(krn, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (rc_g == CL_SUCCESS && num_args >= 11U) *kernel_has_dbg = 1;
    else *kernel_has_dbg = 0;

    *out_prog = prog; *out_krn = krn;
    return 0;
}

/* Debug parsing removed in production build. */


/* Optimized IO helpers */
#include <time.h>
#include <stdint.h>

static inline uint64_t utils_now_ns(void) {
    return lzo_now_ns();
}

int lzo_read_file_to_buf(const char* path, void* dest, size_t size, unsigned long* read_us_out) {
    if (!path || !dest || size == 0) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint64_t t_start = utils_now_ns();
    size_t nread = fread(dest, 1, size, f);
    uint64_t t_end = utils_now_ns();

    if (read_us_out) *read_us_out = (unsigned long)((t_end - t_start) / 1000);
    fclose(f);
    return (nread == size) ? 0 : -1;
}

int lzo_write_compressed_file(const char* path,
                              size_t orig_size, size_t blk_size,
                              size_t nblk, const unsigned int* lens,
                              const void* sparse_data, size_t worst_blk,
                              int alg_id, int debug) {
    if (!path || !lens || !sparse_data || nblk == 0) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    /* Use 2MB buffer for glibc buffered writing */
    char* vbuf = (char*)malloc(2 * 1024 * 1024);
    if (vbuf) setvbuf(f, vbuf, _IOFBF, 2 * 1024 * 1024);

    uint16_t magic = 0x4C5A;
    unsigned int header[4] = {(unsigned int)orig_size, (unsigned int)blk_size, (unsigned int)nblk, (unsigned int)alg_id};

    if (fwrite(&magic, 1, 2, f) != 2) goto err;
    if (fwrite(header, 4, 4, f) != 4) goto err;
    if (fwrite(lens, 4, nblk, f) != nblk) goto err;

    const unsigned char* dev_out = (const unsigned char*)sparse_data;
    if (worst_blk == 0) {
        size_t total = 0;
        for(size_t i=0; i<nblk; i++) total += lens[i];
        if (fwrite(dev_out, 1, total, f) != total) goto err;
    } else {
        /* Reduce fwrite syscall count by packing sparse block payloads. */
        size_t pack_kb = 1024;
        const char* env_pack_kb = getenv("LZO_GPU_PACK_WRITE_KB");
        if (env_pack_kb && *env_pack_kb) {
            long v = strtol(env_pack_kb, NULL, 10);
            if (v > 64 && v <= 16384) pack_kb = (size_t)v;
        }
        size_t pack_cap = pack_kb * 1024;
        unsigned char* pack = (unsigned char*)malloc(pack_cap);
        size_t fill = 0;

        if (!pack) {
            /* Fallback: sequential fwrite */
            for (size_t i = 0; i < nblk; i++) {
                if (lens[i] > 0) {
                    if (fwrite(dev_out + i * worst_blk, 1, lens[i], f) != lens[i]) goto err;
                }
            }
        } else {
            for (size_t i = 0; i < nblk; i++) {
                size_t clen = (size_t)lens[i];
                if (clen == 0) continue;
                const unsigned char* src = dev_out + i * worst_blk;

                if (clen > pack_cap) {
                    if (fill > 0) {
                        if (fwrite(pack, 1, fill, f) != fill) {
                            free(pack);
                            goto err;
                        }
                        fill = 0;
                    }
                    if (fwrite(src, 1, clen, f) != clen) {
                        free(pack);
                        goto err;
                    }
                    continue;
                }

                if (fill + clen > pack_cap) {
                    if (fwrite(pack, 1, fill, f) != fill) {
                        free(pack);
                        goto err;
                    }
                    fill = 0;
                }

                memcpy(pack + fill, src, clen);
                fill += clen;
            }

            if (fill > 0) {
                if (fwrite(pack, 1, fill, f) != fill) {
                    free(pack);
                    goto err;
                }
            }
            free(pack);
        }
    }

    fclose(f);
    if (vbuf) free(vbuf);
    return 0;

err:
    if (f) fclose(f);
    if (vbuf) free(vbuf);
    return -1;
}

void lzo_print_response_stats(const response_t* resp, const char* input_path, int operation, int alg) {
    const timing_t* t = &resp->timing;
    size_t in_size = t->in_size;
    if (in_size == 0) {
        struct stat st;
        if (stat(input_path, &st) == 0) in_size = st.st_size;
    }

    const char* mode_str = (operation == 'C' ? "Compress" : "Decompress");

    printf("\n==============================================================================\n");
    printf("  LZO GPU PERFORMANCE REPORT (%s)\n", mode_str);
    printf("==============================================================================\n");
    printf("%-22s : %s\n", "Input File", input_path);
    printf("%-22s : %zu bytes (%s)\n", "Input Size", in_size, format_size_lzo(in_size));

    if (operation == 'C') {
        double ratio_pct = in_size > 0 ? (100.0 * (double)t->out_size / in_size) : 0;
        printf("%-22s : %zu bytes (%s) (%.2f%% ratio)\n", "Output Size", (size_t)t->out_size, format_size_lzo(t->out_size), ratio_pct);
    } else {
        printf("%-22s : %zu bytes (%s)\n", "Output Size", (size_t)t->out_size, format_size_lzo(t->out_size));
    }

    const char* alg_name = (alg == 1 ? "lzo1y" : "lzo1x");
    printf("%-22s : %s (Level: %d)\n", "Algorithm", alg_name, t->algo_config);
    printf("%-22s : %lu blocks (BlockSize: %s)\n", "Workload", (unsigned long)t->nblk, format_size_lzo(t->blk_size_bytes));
    printf("%-22s : Global: %lu, Local: %lu\n", "Grid Size", (unsigned long)t->global_size, (unsigned long)t->local_size);

    printf("------------------------------------------------------------------------------\n");
    printf("  Detailed Timing Breakdown\n");
    printf("------------------------------------------------------------------------------\n");

    unsigned long total_us = resp->time_us;
    print_time_row(stdout, "File Read", t->file_read_us, total_us);
    print_time_row(stdout, "OCI Setup", t->ocl_setup_us, total_us);
    print_time_row(stdout, "Buffer Alloc", t->buffer_alloc_us, total_us);
    print_time_row(stdout, "Data Upload/Map", t->data_upload_us, total_us);
    print_time_row(stdout, "Kernel Execution", t->kernel_exec_us, total_us);
    print_time_row(stdout, "Data Download/Unmap", t->download_total_us, total_us);
    print_time_row(stdout, "File Write", t->file_write_us, total_us);

    printf("------------------------------------------------------------------------------\n");
    printf("%-22s : %10.3f ms\n", "TOTAL INCLUSIVE", total_us / 1000.0);
    printf("------------------------------------------------------------------------------\n");

    size_t throughput_bytes = (operation == 'C') ? in_size : (size_t)t->out_size;
    double in_mb = (double)throughput_bytes / (1024.0 * 1024.0);
    if (total_us > 0) {
        double mb_s = in_mb / ((double)total_us / 1000000.0);
        printf("%-22s : %10.2f MB/s\n", "Inclusive Throughput", mb_s);
    }
    if (t->kernel_exec_us > 0) {
        double mb_s = in_mb / ((double)t->kernel_exec_us / 1000000.0);
        printf("%-22s : %10.2f MB/s\n", "Kernel Throughput", mb_s);
    }
    printf("==============================================================================\n\n");
}
