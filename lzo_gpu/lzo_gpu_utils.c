#include "lzo_gpu_utils.h"
#include "lzo_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>

/* Implementation of lzo_find_file_path */
int lzo_find_file_path(const char *name, char *out, size_t outlen)
{
    char path[PATH_MAX];
    char base[PATH_MAX];
    /* 1: exe dir and exe_dir/../lzo_gpu */
    char exe_path[PATH_MAX] = {0};
    ssize_t r = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (r > 0) {
        exe_path[r] = '\0';
        char *slash = strrchr(exe_path, '/');
        if (slash) {
            *slash = '\0';
            snprintf(path, sizeof(path), "%s/%s", exe_path, name);
            if (access(path, R_OK) == 0) { strncpy(out, path, outlen - 1); out[outlen - 1] = '\0'; return 0; }
            snprintf(path, sizeof(path), "%s/../lzo_gpu/%s", exe_path, name);
            if (access(path, R_OK) == 0) { strncpy(out, path, outlen - 1); out[outlen - 1] = '\0'; return 0; }
        }
    }
    /* 2: LZO_GPU_DIR */
    const char *env = getenv("LZO_GPU_DIR");
    if (env && env[0]) {
        snprintf(path, sizeof(path), "%s/%s", env, name);
        if (access(path, R_OK) == 0) { strncpy(out, path, outlen - 1); out[outlen - 1] = '\0'; return 0; }
    }
    /* 3: OUT_DIR */
    env = getenv("OUT_DIR");
    if (env && env[0]) {
        snprintf(path, sizeof(path), "%s/%s", env, name);
        if (access(path, R_OK) == 0) { strncpy(out, path, outlen - 1); out[outlen - 1] = '\0'; return 0; }
    }
    /* 4: cwd */
    if (getcwd(base, sizeof(base)) != NULL) {
        snprintf(path, sizeof(path), "%s/%s", base, name);
        if (access(path, R_OK) == 0) { strncpy(out, path, outlen - 1); out[outlen - 1] = '\0'; return 0; }
    }
    /* 5: fallback: raw name in cwd */
    if (access(name, R_OK) == 0) { strncpy(out, name, outlen - 1); out[outlen - 1] = '\0'; return 0; }
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
    size_t multiplier = 1024; /* default KB */
    if (*end == '\0') {
        multiplier = 1024;
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
    if (*end == '\0') return 0;
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
    size_t target_nblk = (size_t)cu * 192;

    /* Calculate block size from target block count */
    size_t base_block = (in_sz + target_nblk - 1) / target_nblk;

    /* Apply size-based bounds matching test results (kernel_analysis_20251225_034050) */
    size_t size_based_min, size_based_max;

    if (in_sz >= 134 * 1024 * 1024) {
        /* XLarge files (134MB+): 32KB optimal (test: 988 MB/s with L12) */
        size_based_min = 28 * 1024; size_based_max = 36 * 1024;
    } else if (in_sz >= 47 * 1024 * 1024) {
        /* Large files (47-134MB): 32KB optimal (test peak: 1736 MB/s with L11) */
        size_based_min = 28 * 1024; size_based_max = 36 * 1024;
    } else if (in_sz >= 10 * 1024 * 1024) {
        /* Medium-large files (10-47MB): 16KB balanced (test: 1346 MB/s with L10) */
        size_based_min = 12 * 1024; size_based_max = 20 * 1024;
    } else if (in_sz >= 5 * 1024 * 1024) {
        /* Medium files (5-10MB): 8-16KB blocks (test: 8KB→978 MB/s, 16KB→1346 MB/s) */
        size_based_min = 8 * 1024; size_based_max = 16 * 1024;
    } else if (in_sz >= 1 * 1024 * 1024) {
        /* Small files (1-5MB): 8KB balanced (test: 268 MB/s, CR: 59x) */
        size_based_min = 6 * 1024; size_based_max = 10 * 1024;
    } else {
        /* Very small files (<1MB): 1-4KB blocks (test: 1KB→313 MB/s fastest) */
        size_based_min = 1 * 1024; size_based_max = 4 * 1024;
    }
    /* Clamp to size-based range */
    if (base_block < size_based_min) base_block = size_based_min;
    if (base_block > size_based_max) base_block = size_based_max;

    /* Align to 1KB boundary for optimal memory access */
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

    if (in_sz >= 134 * 1024 * 1024) {
        /* XLarge files (134MB+): 32KB optimal (test: 988 MB/s with L12) */
        size_based_min = 28 * 1024; size_based_max = 36 * 1024;
    } else if (in_sz >= 47 * 1024 * 1024) {
        /* Large files (47-134MB): 32KB optimal (test peak: 1736 MB/s with L11) */
        size_based_min = 28 * 1024; size_based_max = 36 * 1024;
    } else if (in_sz >= 10 * 1024 * 1024) {
        /* Medium-large files (10-47MB): 16KB balanced (test: 1346 MB/s with L10) */
        size_based_min = 12 * 1024; size_based_max = 20 * 1024;
    } else if (in_sz >= 5 * 1024 * 1024) {
        /* Medium files (5-10MB): 8-16KB blocks (test: 8KB→978 MB/s, 16KB→1346 MB/s) */
        size_based_min = 8 * 1024; size_based_max = 16 * 1024;
    } else if (in_sz >= 1 * 1024 * 1024) {
        /* Small files (1-5MB): 8KB balanced (test: 268 MB/s, CR: 59x) */
        size_based_min = 6 * 1024; size_based_max = 10 * 1024;
    } else if (in_sz >= 512 * 1024) {
        /* Very small files (512KB-1MB): 4-8KB blocks */
        size_based_min = 4 * 1024; size_based_max = 8 * 1024;
    } else if (in_sz >= 256 * 1024) {
        /* Tiny files (256KB-512KB): 2-4KB blocks */
        size_based_min = 2 * 1024; size_based_max = 4 * 1024;
    } else {
        /* Minimal files (<256KB): 1-4KB blocks (test: 1KB→313 MB/s fastest) */
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

void lzo_choose_blocking_adaptive(const unsigned char* data, size_t in_sz, cl_device_id dev, size_t fixed_blk_bytes, int fixed_exact, size_t* blk_sz_out, size_t* nblk_out, int debug)
{
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    if (cu == 0) cu = 1;

    /* CLI or env override: fixed block size */
    if (fixed_blk_bytes > 0) {
        size_t blk = fixed_blk_bytes;
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

/* Adaptive compression level selection based on file size and entropy.
 *
 * Based on comprehensive kernel testing (64 samples, 1482 configs):
 * - Small (<5MB): Level 10 (best speed-compression balance)
 * - Medium (5-47MB): Level 10 (L10→L11: -26% speed, +1.4% compression ratio, not worth it)
 * - Large (47-134MB): Level 11 (L11 is sweet spot: 1736 MB/s peak)
 * - XLarge (>134MB): Level 12 (L12 reaches compression ceiling)
 *
 * Entropy adjustments:
 * - High entropy (incompressible): prefer lower levels (speed more important)
 * - Low entropy (highly compressible): can use higher levels (better compression payoff)
 *
 * Only called when user hasn't explicitly specified compression level.
 */
int lzo_adaptive_compression_level(size_t in_sz, double entropy, int debug)
{
    int base_level = 11; /* Default to balanced */

    /* Size-based selection matching test results */
    if (in_sz < 5 * 1024 * 1024) {
        /* Small files (<5MB): Level 10 optimal */
        base_level = 10;
    } else if (in_sz < 47 * 1024 * 1024) {
        /* Medium files (5-47MB): Level 10 optimal (L11 too expensive) */
        base_level = 10;
    } else if (in_sz < 134 * 1024 * 1024) {
        /* Large files (47-134MB): Level 11 is peak (1736 MB/s) */
        base_level = 11;
    } else {
        /* XLarge files (>134MB): Level 12 for best compression */
        base_level = 12;
    }

    /* Entropy-based adjustment */
    if (entropy > 0.0) {
        if (entropy > LZO_ADAPTIVE_HIGH_ENTROPY) {
            /* High entropy (incompressible): reduce level by 1 for speed */
            if (base_level > 10) base_level--;
        } else if (entropy < LZO_ADAPTIVE_LOW_ENTROPY) {
            /* Low entropy (highly compressible): increase level by 1 for better compression */
            if (base_level < 14) base_level++;
        }
        /* Medium entropy: keep base level */
    }

    /* Clamp to valid range (10-14 for LZO1X) */
    if (base_level < 10) base_level = 10;
    if (base_level > 14) base_level = 14;

    if (debug) {
        fprintf(stderr, "[ADAPTIVE] Compression level: %d (size=%zu entropy=%.4f)\n",
                base_level, in_sz, entropy);
    }

    return base_level;
}

/* Load an OpenCL program from an available precompiled binary or compile from source.
 * Does NOT exit on failure; returns NULL on error and fills build_log (if provided).
 */
cl_program lzo_load_program_with_dbits(cl_context ctx, cl_device_id dev, const char* alg_name, int bits, char *build_log, size_t build_log_len)
{
    cl_int err;
    cl_program prog = NULL;

    /* Try bits-specific binary first: <alg>_<bits>.clbin */
    {
        char bin_name[64];
        if (bits > 0) snprintf(bin_name, sizeof(bin_name), "%s_%d.clbin", alg_name, bits);
        else snprintf(bin_name, sizeof(bin_name), "%s.clbin", alg_name);

        char resolved_bin[PATH_MAX];
        if (lzo_find_file_path(bin_name, resolved_bin, sizeof(resolved_bin)) == 0) {
            FILE* fb = fopen(resolved_bin, "rb");
            if (fb) {
                fseek(fb, 0, SEEK_END);
                long bsz = ftell(fb);
                fseek(fb, 0, SEEK_SET);
                unsigned char* bin = malloc(bsz);
                if (bin && fread(bin, 1, bsz, fb) == (size_t)bsz) {
                    cl_int binary_status;
                    prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
                                                    (const unsigned char**)&bin, &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                        err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
                        if (err == CL_SUCCESS) {
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
                                snprintf(build_log, build_log_len, "Binary build failed (err=%d)", (int)err);
                            }
                            clReleaseProgram(prog); prog = NULL;
                        }
                    }
                }
                if (bin) free(bin);
                fclose(fb);
            }
        }
    }

    /* Try generic binary <alg>.clbin when bits-specific not used or failed */
    if (bits > 0) {
        char bin_name[64]; snprintf(bin_name, sizeof(bin_name), "%s.clbin", alg_name);
        char resolved_bin[PATH_MAX];
        if (lzo_find_file_path(bin_name, resolved_bin, sizeof(resolved_bin)) == 0) {
            FILE* fb = fopen(resolved_bin, "rb");
            if (fb) {
                fseek(fb, 0, SEEK_END);
                long bsz = ftell(fb);
                fseek(fb, 0, SEEK_SET);
                unsigned char* bin = malloc(bsz);
                if (bin && fread(bin, 1, bsz, fb) == (size_t)bsz) {
                    cl_int binary_status;
                    prog = clCreateProgramWithBinary(ctx, 1, &dev, (const size_t*)&bsz,
                                                    (const unsigned char**)&bin, &binary_status, &err);
                    if (err == CL_SUCCESS && binary_status == CL_SUCCESS) {
                        err = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", NULL, NULL);
                        if (err == CL_SUCCESS) {
                            if (build_log) build_log[0] = '\0';
                            free(bin); fclose(fb); return prog;
                        } else {
                            size_t log_sz = 0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
                            if (log_sz && build_log) {
                                size_t toread = (log_sz < build_log_len - 1) ? log_sz : build_log_len - 1;
                                char *tmp = malloc(toread + 1);
                                clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, toread, tmp, NULL);
                                tmp[toread] = '\0';
                                snprintf(build_log, build_log_len, "Binary build log: %s", tmp);
                                free(tmp);
                            } else if (build_log) {
                                snprintf(build_log, build_log_len, "Binary build failed (err=%d)", (int)err);
                            }
                            clReleaseProgram(prog); prog = NULL;
                        }
                    }
                }
                if (bin) free(bin);
                fclose(fb);
            }
        }
    }

    /* Fallback: compile from source. When <alg> contains "_debug" or "_opt",
     * strip these suffixes to find the base source (e.g., lzo1x.cl or lzo1y.cl),
     * then compile with appropriate macros (-D LZO_GPU_DEBUG and/or -D LZO_USE_UNROLL2).
     */
    {
        char resolved_src[PATH_MAX]; size_t src_len = 0; char* src = NULL;
        int want_opt = 0;
        int want_debug = 0;

        /* Determine the base algorithm name by stripping _debug and _opt suffixes */
        char base_name[64];
        strncpy(base_name, alg_name, sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';

        /* Check for _opt suffix */
        char *opt_pos = strstr(base_name, "_opt");
        if (opt_pos) {
            *opt_pos = '\0';  /* Remove _opt suffix */
            want_opt = 1;
        }

        /* Check for _debug suffix */
        char *debug_pos = strstr(base_name, "_debug");
        if (debug_pos) {
            *debug_pos = '\0';  /* Remove _debug suffix */
            want_debug = 1;
        }

        /* Now base_name contains just "lzo1x" or "lzo1y" */
        char base_src[128];
        snprintf(base_src, sizeof(base_src), "%s.cl", base_name);

        /* Try to find and load the base source */
        if (lzo_find_file_path(base_src, resolved_src, sizeof(resolved_src)) == 0) {
            src = lzo_read_file(resolved_src, &src_len);
        } else {
            if (build_log) snprintf(build_log, build_log_len, "source file %s not found", base_src);
            return NULL;
        }
        if (!src) { if (build_log) snprintf(build_log, build_log_len, "failed to read %s", base_src); return NULL; }

        prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
        if (err != CL_SUCCESS) { if (build_log) snprintf(build_log, build_log_len, "clCreateProgramWithSource failed (err=%d)", err); free(src); return NULL; }

        /* Build with appropriate macros based on variant flags */
        char build_opts[256];
        if (want_opt && want_debug) {
            snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -I./lzo_gpu -I.. -D D_BITS=%d -D LZO_USE_UNROLL2 -D LZO_GPU_DEBUG", bits);
        } else if (want_opt) {
            snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -I./lzo_gpu -I.. -D D_BITS=%d -D LZO_USE_UNROLL2", bits);
        } else if (want_debug) {
            snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -I./lzo_gpu -I.. -D D_BITS=%d -D LZO_GPU_DEBUG", bits);
        } else {
            snprintf(build_opts, sizeof(build_opts), "-cl-std=CL2.0 -I. -I./lzo_gpu -I.. -D D_BITS=%d", bits);
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
        if (build_log) build_log[0] = '\0';
        return prog;
    }
}

/* Load and create a compression kernel (see header for behavior). */
int lzo_load_comp_kernel(cl_context ctx, cl_device_id dev, const char *alg_name, int comp_level, int kernel_debug, int kernel_opt, int debug, cl_program *out_prog, cl_kernel *out_krn, int *kernel_has_dbg, char *build_log, size_t build_log_len)
{
    cl_int err;
    cl_program prog = NULL;
    cl_kernel krn = NULL;
    int use_opt_debug_variant = 0, use_opt_prod_variant = 0, use_debug_variant = 0;
    char prog_base_name[64]; char alt_base_name[64];

    /* Try kernel_opt variants first (when enabled).
     * Opt variants are implemented by compiling the base/source with -D LZO_USE_UNROLL2
     * so we attempt program names like <alg>_debug_opt and <alg>_opt (may be precompiled binaries
     * or compiled from base sources with the extra macro).
     */
    if (kernel_opt) {
        if (debug) {
            snprintf(prog_base_name, sizeof(prog_base_name), "%s_debug_opt", alg_name);
            prog = lzo_load_program_with_dbits(ctx, dev, prog_base_name, comp_level, build_log, build_log_len);
            if (prog) {
                use_opt_debug_variant = 1;
                if (debug) fprintf(stderr, "DBG: kernel_opt+debug requested; loaded %s (bits=%d)\n", prog_base_name, comp_level);
            } else if (debug) fprintf(stderr, "DBG: kernel_opt+debug: not found or build failed for %s, trying %s_opt\n", prog_base_name, alg_name);
        }
        if (!prog) {
            snprintf(alt_base_name, sizeof(alt_base_name), "%s_opt", alg_name);
            prog = lzo_load_program_with_dbits(ctx, dev, alt_base_name, comp_level, build_log, build_log_len);
            if (prog) {
                use_opt_prod_variant = 1;
                (void)use_opt_prod_variant; /* Suppress unused warning */
                if (debug) fprintf(stderr, "DBG: kernel_opt requested; loaded %s (bits=%d)\n", alt_base_name, comp_level);
            } else if (debug) fprintf(stderr, "DBG: kernel_opt prod: not found or build failed for %s, falling back to base\n", alt_base_name);
        }
        if (!prog && debug) {
            fprintf(stderr, "DBG: kernel_opt requested but no opt variant found for %s (bits=%d); falling back to base\n", alg_name, comp_level);
        }
    }

    /* Next try debug variant if requested (debug flag required) */
    if (!prog && kernel_debug && debug) {
        snprintf(prog_base_name, sizeof(prog_base_name), "%s_debug", alg_name);
        /* Try loading debug binary directly - don't require .cl source to exist */
        prog = lzo_load_program_with_dbits(ctx, dev, prog_base_name, comp_level, build_log, build_log_len);
        if (prog) {
            use_debug_variant = 1;
            if (debug) fprintf(stderr, "DBG: debug requested; loaded %s binary (bits=%d)\n", prog_base_name, comp_level);
        } else if (debug) {
            fprintf(stderr, "DBG: debug requested but %s binary not found; falling back to base\n", prog_base_name);
        }
    }

    /* Fallback to base algorithm */
    if (!prog) {
        prog = lzo_load_program_with_dbits(ctx, dev, alg_name, comp_level, build_log, build_log_len);
        if (!prog) {
            if (build_log && build_log[0] == '\0') snprintf(build_log, build_log_len, "failed to build kernel variants for %s", alg_name);
            return -1;
        }
    }

    char krn_name[96];
    /* unified kernel names: opt variants reuse the same kernel symbol as their base (debug or non-debug)
     * so we only select between debug vs non-debug kernel symbol names here.
     */
    if (use_opt_debug_variant || use_debug_variant) snprintf(krn_name, sizeof(krn_name), "%s_block_compress_debug", alg_name);
    else snprintf(krn_name, sizeof(krn_name), "%s_block_compress", alg_name);

    krn = clCreateKernel(prog, krn_name, &err);
    if (err != CL_SUCCESS) {
        if (build_log && build_log[0] == '\0') snprintf(build_log, build_log_len, "clCreateKernel failed for %s (err=%d)", krn_name, (int)err);
        clReleaseProgram(prog);
        return -1;
    }

    cl_uint num_args = 0;
    cl_int rc_g = clGetKernelInfo(krn, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);
    if (rc_g == CL_SUCCESS && num_args >= 7) *kernel_has_dbg = 1;
    else *kernel_has_dbg = 0;
    if (debug) fprintf(stderr, "DBG: kernel num_args=%u kernel_has_dbg_arg=%d\n", num_args, *kernel_has_dbg);

    *out_prog = prog; *out_krn = krn;
    return 0;
}

/* Parse and print an instrumented debug buffer. Returns 0 on OK, 1 if a sanity failure occurred. */
int lzo_parse_and_print_debug_buffer(const uint32_t *dbg_map, size_t dbg_fields, size_t nblk, size_t blk, size_t worst_blk)
{
    int abort_debug_sanity = 0;

    /* Detect whether verbose fields are present by sampling first few blocks */
    int verbose = 0;
    size_t sample_n = nblk < 8 ? nblk : 8;
    for (size_t i = 0; i < sample_n; ++i) {
        uint32_t a3 = dbg_map[i * dbg_fields + 3];
        uint32_t a4 = dbg_map[i * dbg_fields + 4];
        uint32_t a5 = dbg_map[i * dbg_fields + 5];
        uint32_t a6 = dbg_map[i * dbg_fields + 6];
        if (a3 != 0 || a4 != 0 || a5 != 0 || a6 != 0) { verbose = 1; break; }
    }

    for (size_t i = 0; i < nblk; ++i) {
        uint32_t in_len_dbg = dbg_map[i * dbg_fields + 0];
        uint32_t out_len_dbg = dbg_map[i * dbg_fields + 1];
        uint32_t flag_dbg = dbg_map[i * dbg_fields + 2];

        if (!verbose) {
            /* Basic 3-field sanity checks */
            if (in_len_dbg > (uint32_t)blk ||
                out_len_dbg > (uint32_t)worst_blk ||
                (flag_dbg != 0 && flag_dbg != 1)) {
                fprintf(stderr, "ERR: suspicious debug data at block %zu: IN=%u OUT=%u FLAG=%u\n",
                        i, in_len_dbg, out_len_dbg, flag_dbg);
                /* write short dump */
                char dump_path[256];
                snprintf(dump_path, sizeof(dump_path), "/tmp/lzo_dbg_dump_%d.txt", getpid());
                FILE *fd = fopen(dump_path, "w");
                if (fd) {
                    fprintf(fd, "nblk=%zu blk=%zu worst_blk=%zu\n", nblk, blk, worst_blk);
                    size_t dump_n = nblk < 64 ? nblk : 64;
                    for (size_t bi = 0; bi < dump_n; ++bi) {
                        uint32_t a0 = dbg_map[bi*dbg_fields + 0];
                        uint32_t a1 = dbg_map[bi*dbg_fields + 1];
                        uint32_t a2 = dbg_map[bi*dbg_fields + 2];
                        fprintf(fd, "BLOCK %4zu: %u %u %u\n", bi, a0, a1, a2);
                    }
                    fclose(fd);
                    fprintf(stderr, "ERR: debug dump written to %s\n", dump_path);
                }
                abort_debug_sanity = 1;
                break;
            }
            fprintf(stderr, "LZO_GPU_DEBUG BLOCK %4zu IN %6u OUT %6u FLAG %u\n",
                    i, in_len_dbg, out_len_dbg, flag_dbg);
        } else {
            uint32_t lookups_dbg = dbg_map[i * dbg_fields + 3];
            uint32_t hits_dbg = dbg_map[i * dbg_fields + 4];
            uint32_t matched_bytes_dbg = dbg_map[i * dbg_fields + 5];
            uint32_t updates_dbg = dbg_map[i * dbg_fields + 6];

            /* Verbose sanity checks */
            if (in_len_dbg > (uint32_t)blk ||
                out_len_dbg > (uint32_t)worst_blk ||
                (flag_dbg != 0 && flag_dbg != 1) ||
                lookups_dbg > (1u<<24) ||
                hits_dbg > lookups_dbg ||
                matched_bytes_dbg > in_len_dbg ||
                updates_dbg > lookups_dbg) {
                fprintf(stderr, "ERR: suspicious debug data at block %zu: IN=%u OUT=%u FLAG=%u LOOKUPS=%u HITS=%u MATCH_BYTES=%u UPDATES=%u\n",
                        i, in_len_dbg, out_len_dbg, flag_dbg, lookups_dbg, hits_dbg, matched_bytes_dbg, updates_dbg);
                /* full dump */
                char dump_path[256];
                snprintf(dump_path, sizeof(dump_path), "/tmp/lzo_dbg_dump_%d.txt", getpid());
                FILE *fd = fopen(dump_path, "w");
                if (fd) {
                    fprintf(fd, "nblk=%zu blk=%zu worst_blk=%zu\n", nblk, blk, worst_blk);
                    size_t dump_n = nblk < 64 ? nblk : 64;
                    for (size_t bi = 0; bi < dump_n; ++bi) {
                        uint32_t a0 = dbg_map[bi*dbg_fields + 0];
                        uint32_t a1 = dbg_map[bi*dbg_fields + 1];
                        uint32_t a2 = dbg_map[bi*dbg_fields + 2];
                        uint32_t a3 = dbg_map[bi*dbg_fields + 3];
                        uint32_t a4 = dbg_map[bi*dbg_fields + 4];
                        uint32_t a5 = dbg_map[bi*dbg_fields + 5];
                        uint32_t a6 = dbg_map[bi*dbg_fields + 6];
                        fprintf(fd, "BLOCK %4zu: %u %u %u %u %u %u %u\n", bi, a0, a1, a2, a3, a4, a5, a6);
                    }
                    fclose(fd);
                    fprintf(stderr, "ERR: debug dump written to %s\n", dump_path);
                }
                abort_debug_sanity = 1;
                break;
            }
            fprintf(stderr, "LZO_GPU_DEBUG BLOCK %4zu IN %6u OUT %6u FLAG %u LOOKUPS %u HITS %u MATCH_BYTES %u UPDATES %u\n",
                    i, in_len_dbg, out_len_dbg, flag_dbg, lookups_dbg, hits_dbg, matched_bytes_dbg, updates_dbg);
        }
    }
    return abort_debug_sanity;
}

