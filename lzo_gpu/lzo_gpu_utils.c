#include "lzo_gpu_utils.h"
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
    size_t sample = (size < LZO_ADAPTIVE_SAMPLE_SIZE) ? size : LZO_ADAPTIVE_SAMPLE_SIZE;
    size_t step = (sample < size) ? (size / sample) : 1;
    unsigned int freq[256] = {0};
    size_t count = 0;
    for (size_t i = 0; i < size; i += step) {
        freq[data[i]]++;
        count++;
        if (count >= sample) break;
    }
    double entropy = 0.0;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / (double)count;
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

size_t lzo_adaptive_block_size(const unsigned char* data, size_t in_sz, cl_uint cu, double* entropy_out, int debug)
{
    double entropy = 0.0;
    if (data && in_sz > 0) entropy = lzo_calc_entropy(data, in_sz);
    if (entropy_out) *entropy_out = entropy;
    /* Basic heuristics following host/daemon code logic */
    size_t base_block;
    if (entropy > LZO_ADAPTIVE_HIGH_ENTROPY) {
        base_block = LZO_MAX_BLOCK_BYTES_DEFAULT; /* high entropy -> large blocks */
    } else if (entropy > (LZO_ADAPTIVE_LOW_ENTROPY + LZO_ADAPTIVE_HIGH_ENTROPY) / 2.0) {
        base_block = (size_t)((LZO_MAX_BLOCK_BYTES_DEFAULT + LZO_MIN_BLOCK_BYTES_DEFAULT) / 2);
    } else {
        /* low entropy -> smaller blocks for higher parallelism */
        if (in_sz >= 100 * 1024 * 1024) base_block = 64 * 1024;
        else if (in_sz >= 10 * 1024 * 1024) base_block = 96 * 1024;
        else base_block = 128 * 1024;
    }
    if (base_block < LZO_MIN_BLOCK_BYTES_DEFAULT) base_block = LZO_MIN_BLOCK_BYTES_DEFAULT;
    if (base_block > LZO_MAX_BLOCK_BYTES_DEFAULT) base_block = LZO_MAX_BLOCK_BYTES_DEFAULT;
    return base_block;
}

size_t lzo_adaptive_block_size_from_entropy(size_t in_sz, double entropy, cl_uint cu, int debug)
{
    /* reuse adaptive heuristics but using precomputed entropy */
    double entropy_local = entropy;
    size_t base_block;
    if (entropy_local > LZO_ADAPTIVE_HIGH_ENTROPY) {
        base_block = LZO_MAX_BLOCK_BYTES_DEFAULT;
    } else if (entropy_local > (LZO_ADAPTIVE_LOW_ENTROPY + LZO_ADAPTIVE_HIGH_ENTROPY) / 2.0) {
        base_block = (size_t)((LZO_MAX_BLOCK_BYTES_DEFAULT + LZO_MIN_BLOCK_BYTES_DEFAULT) / 2);
    } else {
        if (in_sz >= 100 * 1024 * 1024) base_block = 64 * 1024;
        else if (in_sz >= 10 * 1024 * 1024) base_block = 96 * 1024;
        else base_block = 128 * 1024;
    }
    if (base_block < LZO_MIN_BLOCK_BYTES_DEFAULT) base_block = LZO_MIN_BLOCK_BYTES_DEFAULT;
    if (base_block > LZO_MAX_BLOCK_BYTES_DEFAULT) base_block = LZO_MAX_BLOCK_BYTES_DEFAULT;
    return base_block;
}

void lzo_choose_blocking_adaptive(const unsigned char* data, size_t in_sz, cl_device_id dev, size_t fixed_blk_bytes, size_t* blk_sz_out, size_t* nblk_out, int debug)
{
    cl_uint cu = 0; clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL); if (cu == 0) cu = 1;
    /* CLI or env override: fixed block size */
    if (fixed_blk_bytes > 0) {
        size_t blk = fixed_blk_bytes;
        /* align */
        size_t align = LZO_ALIGN_BYTES_DEFAULT;
        blk = (blk + (align - 1)) & ~(align - 1);
        if (blk < LZO_MIN_BLOCK_BYTES_DEFAULT) blk = LZO_MIN_BLOCK_BYTES_DEFAULT;
        if (blk > LZO_MAX_BLOCK_BYTES_DEFAULT) blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
        size_t nblk = (in_sz + blk - 1) / blk;
        *blk_sz_out = blk; *nblk_out = nblk; return;
    }
    double entropy = 0.0;
    if (data && in_sz > 0) entropy = lzo_calc_entropy(data, in_sz);
    size_t blk = lzo_adaptive_block_size(data, in_sz, cu, &entropy, debug);
    /* occ factor selection based on entropy */
    size_t occ_factor = LZO_OCC_FACTOR_DEFAULT;
    if (entropy < LZO_ADAPTIVE_LOW_ENTROPY) occ_factor = (size_t)(LZO_OCC_FACTOR_DEFAULT * 1.5);
    if (entropy > LZO_ADAPTIVE_HIGH_ENTROPY) occ_factor = LZO_OCC_FACTOR_DEFAULT * 3;
    size_t target_nblk = (size_t)cu * occ_factor;
    if (target_nblk > in_sz) target_nblk = in_sz; /* guard */
    /* compute block size and align */
    size_t calc_blk = (in_sz + target_nblk - 1) / target_nblk;
    calc_blk = (calc_blk + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);
    if (calc_blk == 0) calc_blk = LZO_ALIGN_BYTES_DEFAULT;

    size_t min_limit = LZO_MIN_BLOCK_BYTES_DEFAULT;
    /* For small files (< 8MB), allow smaller blocks (down to alignment size) to increase concurrency */
    if (in_sz < 8 * 1024 * 1024) {
        min_limit = LZO_ALIGN_BYTES_DEFAULT;
    }

    if (calc_blk < min_limit) calc_blk = min_limit;
    if (calc_blk > LZO_MAX_BLOCK_BYTES_DEFAULT) calc_blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
    size_t nblk = (in_sz + calc_blk - 1) / calc_blk;
    if (nblk < cu) {
        nblk = cu;
        calc_blk = (in_sz + nblk - 1) / nblk;
        calc_blk = (calc_blk + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);
        if (calc_blk < min_limit) calc_blk = min_limit;
        if (calc_blk > LZO_MAX_BLOCK_BYTES_DEFAULT) calc_blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
    }
    size_t tail = in_sz - calc_blk * (nblk - 1);
    if (nblk > 1 && tail < calc_blk / 4) {
        calc_blk = (in_sz + nblk - 1) / nblk;
        calc_blk = (calc_blk + (LZO_ALIGN_BYTES_DEFAULT - 1)) & ~(LZO_ALIGN_BYTES_DEFAULT - 1);
        if (calc_blk < min_limit) calc_blk = min_limit;
        if (calc_blk > LZO_MAX_BLOCK_BYTES_DEFAULT) calc_blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
    }
    *blk_sz_out = calc_blk; *nblk_out = (in_sz + calc_blk - 1) / calc_blk;
}
