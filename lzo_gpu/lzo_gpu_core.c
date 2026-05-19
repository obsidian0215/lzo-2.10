/* lzo_gpu_core.c - shared backend for compression/decompression
 * Extracted from daemon_compress.c / daemon_decompress.c so both daemon
 * and standalone can reuse same implementation.
 */

#include <CL/cl.h>
#include "timing.h"
#include "lzo_defaults.h"
#include "lzo_gpu_core.h"
#include "lzo_gpu_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <limits.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>

#define CHECK(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

int g_verbose = 0;

static int lzo_env_flag_enabled(const char* name) {
    const char* v = getenv(name);
    if (!v || !*v) return 0;
    return (strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0);
}

enum {
    LZO_DBG_COMP_SEARCH_ITERS = 0,
    LZO_DBG_COMP_MATCH_FOUND,
    LZO_DBG_COMP_LITERAL_BYTES,
    LZO_DBG_COMP_MATCH_BYTES,
    LZO_DBG_COMP_INPUT_BYTES,
    LZO_DBG_COMP_OUTPUT_BYTES,
    LZO_DBG_COMP_DICT_LOOKUPS,
    LZO_DBG_COMP_DICT_STORES,
    LZO_DBG_COMP_EPOCH_VALID_HITS,
    LZO_DBG_COMP_EPOCH_MISMATCH_MISS,
    LZO_DBG_COMP_LIVE_SLOT_OVERWRITES,
    LZO_DBG_COMP_STALE_SLOT_OVERWRITES,
    LZO_DBG_COMP_LITERAL_OPS,
    LZO_DBG_COMP_MATCH_OPS,
    LZO_DBG_COMP_M2_MATCHES,
    LZO_DBG_COMP_M3_MATCHES,
    LZO_DBG_COMP_M4_MATCHES,
    LZO_DBG_COMP_TAIL_LITERAL_BYTES,
    LZO_DBG_COMP_MATCH_MISS_AFTER_VALID_HIT,
    LZO_DBG_COMP_MATCH_MISS_AFTER_EPOCH_MISMATCH,
    LZO_DBG_COMP_SHARED_OWNER_BLOCKS,
    LZO_DBG_COMP_NOSHARE_FASTPATH_BLOCKS,
    LZO_DBG_COMP_SLOT_OVERWRITE_SAME_OWNER,
    LZO_DBG_COMP_SLOT_OVERWRITE_CROSS_OWNER,
    LZO_DBG_COMP_SHARED_TABLE_PROBE_COUNT,
    LZO_DBG_COMP_SHARED_TABLE_WRITE_COUNT,
    LZO_DBG_COMP_PREFIX_BYTES_CONSUMED,
    LZO_DBG_COMP_PREFIX_LITERAL_BYTES,
    LZO_DBG_COMP_PREFIX_MATCH_BYTES,
    LZO_DBG_COMP_PREFIX_LITERAL_OPS,
    LZO_DBG_COMP_PREFIX_MATCH_OPS,
    LZO_DBG_COMP_PREFIX_PREWARM_STORES,
    LZO_DBG_COMP_SUFFIX_INPUT_BYTES,
    LZO_DBG_COMP_SUFFIX_LITERAL_BYTES,
    LZO_DBG_COMP_SUFFIX_MATCH_BYTES,
    LZO_DBG_COMP_SUFFIX_LITERAL_OPS,
    LZO_DBG_COMP_SUFFIX_MATCH_OPS,
    LZO_DBG_COMP_CORE_VECTOR_BATCHES,
    LZO_DBG_COMP_CORE_SLOW_STEPS,
    LZO_DBG_COMP_CORE_EXTEND8_ITERS,
    LZO_DBG_COMP_CORE_EXTEND4_ITERS,
    LZO_DBG_COMP_CORE_EXTEND1_ITERS,
    LZO_DBG_COMP_CORE_SKIP_ADVANCE,
    LZO_DBG_COMP_SUFFIX_CORE_CALLS,
    LZO_DBG_COMP_N
};

enum {
    LZO_DBG_DEC_TOKENS = 0,
    LZO_DBG_DEC_LITERAL_BYTES,
    LZO_DBG_DEC_MATCH_BYTES,
    LZO_DBG_DEC_SMALL_OFFSETS,
    LZO_DBG_DEC_OUTPUT_ERROR,
    LZO_DBG_DEC_LITERAL_OPS,
    LZO_DBG_DEC_MATCH_OPS,
    LZO_DBG_DEC_OVERLAP_MATCHES,
    LZO_DBG_DEC_M2_MATCHES,
    LZO_DBG_DEC_M3_MATCHES,
    LZO_DBG_DEC_M4_MATCHES,
    LZO_DBG_DEC_FIRST_LITERAL_RUN_BYTES,
    LZO_DBG_DEC_FIRST_LITERAL_RUN_OPS,
    LZO_DBG_DEC_POST_MATCH_LITERAL_BYTES,
    LZO_DBG_DEC_POST_MATCH_LITERAL_OPS,
    LZO_DBG_DEC_EOF_MARKERS,
    LZO_DBG_DEC_N
};

static int lzo_debug_counters_enabled(void) {
    const char* v = getenv("LZO_GPU_DEBUG");
    if (!v || !*v) return 0;
    return (strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 ||
            strcasecmp(v, "yes") == 0 || strcasecmp(v, "on") == 0);
}

static void lzo_print_comp_debug_stats(const uint32_t* stats, size_t num_blocks, const char* tag) {
    unsigned long long search_iters = 0;
    unsigned long long match_found = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long input_bytes = 0;
    unsigned long long output_bytes = 0;
    unsigned long long dict_lookups = 0;
    unsigned long long dict_stores = 0;
    unsigned long long epoch_valid_hits = 0;
    unsigned long long epoch_mismatch_miss = 0;
    unsigned long long live_slot_overwrites = 0;
    unsigned long long stale_slot_overwrites = 0;
    unsigned long long literal_ops = 0;
    unsigned long long match_ops = 0;
    unsigned long long m2_matches = 0;
    unsigned long long m3_matches = 0;
    unsigned long long m4_matches = 0;
    unsigned long long tail_literal_bytes = 0;
    unsigned long long match_miss_after_valid_hit = 0;
    unsigned long long match_miss_after_epoch_mismatch = 0;
    unsigned long long shared_owner_blocks = 0;
    unsigned long long noshare_fastpath_blocks = 0;
    unsigned long long slot_overwrite_same_owner = 0;
    unsigned long long slot_overwrite_cross_owner = 0;
    unsigned long long shared_table_probe_count = 0;
    unsigned long long shared_table_write_count = 0;

    if (!stats || num_blocks == 0) return;

    for (size_t i = 0; i < num_blocks; ++i) {
        const size_t base = i * LZO_DBG_COMP_N;
        search_iters += stats[base + LZO_DBG_COMP_SEARCH_ITERS];
        match_found += stats[base + LZO_DBG_COMP_MATCH_FOUND];
        literal_bytes += stats[base + LZO_DBG_COMP_LITERAL_BYTES];
        match_bytes += stats[base + LZO_DBG_COMP_MATCH_BYTES];
        input_bytes += stats[base + LZO_DBG_COMP_INPUT_BYTES];
        output_bytes += stats[base + LZO_DBG_COMP_OUTPUT_BYTES];
        dict_lookups += stats[base + LZO_DBG_COMP_DICT_LOOKUPS];
        dict_stores += stats[base + LZO_DBG_COMP_DICT_STORES];
        epoch_valid_hits += stats[base + LZO_DBG_COMP_EPOCH_VALID_HITS];
        epoch_mismatch_miss += stats[base + LZO_DBG_COMP_EPOCH_MISMATCH_MISS];
        live_slot_overwrites += stats[base + LZO_DBG_COMP_LIVE_SLOT_OVERWRITES];
        stale_slot_overwrites += stats[base + LZO_DBG_COMP_STALE_SLOT_OVERWRITES];
        literal_ops += stats[base + LZO_DBG_COMP_LITERAL_OPS];
        match_ops += stats[base + LZO_DBG_COMP_MATCH_OPS];
        m2_matches += stats[base + LZO_DBG_COMP_M2_MATCHES];
        m3_matches += stats[base + LZO_DBG_COMP_M3_MATCHES];
        m4_matches += stats[base + LZO_DBG_COMP_M4_MATCHES];
        tail_literal_bytes += stats[base + LZO_DBG_COMP_TAIL_LITERAL_BYTES];
        match_miss_after_valid_hit += stats[base + LZO_DBG_COMP_MATCH_MISS_AFTER_VALID_HIT];
        match_miss_after_epoch_mismatch += stats[base + LZO_DBG_COMP_MATCH_MISS_AFTER_EPOCH_MISMATCH];
        shared_owner_blocks += stats[base + LZO_DBG_COMP_SHARED_OWNER_BLOCKS];
        noshare_fastpath_blocks += stats[base + LZO_DBG_COMP_NOSHARE_FASTPATH_BLOCKS];
        slot_overwrite_same_owner += stats[base + LZO_DBG_COMP_SLOT_OVERWRITE_SAME_OWNER];
        slot_overwrite_cross_owner += stats[base + LZO_DBG_COMP_SLOT_OVERWRITE_CROSS_OWNER];
        shared_table_probe_count += stats[base + LZO_DBG_COMP_SHARED_TABLE_PROBE_COUNT];
        shared_table_write_count += stats[base + LZO_DBG_COMP_SHARED_TABLE_WRITE_COUNT];
    }

    fprintf(stderr,
            "%s blocks=%zu search_iters=%llu match_found=%llu literal_bytes=%llu match_bytes=%llu input_bytes=%llu output_bytes=%llu dict_lookups=%llu dict_stores=%llu epoch_valid_hits=%llu epoch_mismatch_miss=%llu live_slot_overwrites=%llu stale_slot_overwrites=%llu\n",
            (tag ? tag : "[LZO-DBG][COMP]"),
            num_blocks,
            search_iters,
            match_found,
            literal_bytes,
            match_bytes,
            input_bytes,
            output_bytes,
            dict_lookups,
            dict_stores,
            epoch_valid_hits,
            epoch_mismatch_miss,
            live_slot_overwrites,
            stale_slot_overwrites);
    fprintf(stderr,
            "%s detail literal_ops=%llu match_ops=%llu m2_matches=%llu m3_matches=%llu m4_matches=%llu tail_literal_bytes=%llu match_miss_after_valid_hit=%llu match_miss_after_epoch_mismatch=%llu shared_owner_blocks=%llu noshare_fastpath_blocks=%llu slot_overwrite_same_owner=%llu slot_overwrite_cross_owner=%llu shared_table_probe_count=%llu shared_table_write_count=%llu\n",
            (tag ? tag : "[LZO-DBG][COMP]"),
            literal_ops,
            match_ops,
            m2_matches,
            m3_matches,
            m4_matches,
            tail_literal_bytes,
            match_miss_after_valid_hit,
            match_miss_after_epoch_mismatch,
            shared_owner_blocks,
            noshare_fastpath_blocks,
            slot_overwrite_same_owner,
            slot_overwrite_cross_owner,
            shared_table_probe_count,
            shared_table_write_count);
}

static void lzo_print_dec_debug_stats(const uint32_t* stats, size_t num_blocks, const char* tag) {
    unsigned long long tokens = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long small_offsets = 0;
    unsigned long long output_errors = 0;
    unsigned long long literal_ops = 0;
    unsigned long long match_ops = 0;
    unsigned long long overlap_matches = 0;
    unsigned long long m2_matches = 0;
    unsigned long long m3_matches = 0;
    unsigned long long m4_matches = 0;
    unsigned long long first_literal_run_bytes = 0;
    unsigned long long first_literal_run_ops = 0;
    unsigned long long post_match_literal_bytes = 0;
    unsigned long long post_match_literal_ops = 0;
    unsigned long long eof_markers = 0;

    if (!stats || num_blocks == 0) return;

    for (size_t i = 0; i < num_blocks; ++i) {
        const size_t base = i * LZO_DBG_DEC_N;
        tokens += stats[base + LZO_DBG_DEC_TOKENS];
        literal_bytes += stats[base + LZO_DBG_DEC_LITERAL_BYTES];
        match_bytes += stats[base + LZO_DBG_DEC_MATCH_BYTES];
        small_offsets += stats[base + LZO_DBG_DEC_SMALL_OFFSETS];
        output_errors += stats[base + LZO_DBG_DEC_OUTPUT_ERROR];
        literal_ops += stats[base + LZO_DBG_DEC_LITERAL_OPS];
        match_ops += stats[base + LZO_DBG_DEC_MATCH_OPS];
        overlap_matches += stats[base + LZO_DBG_DEC_OVERLAP_MATCHES];
        m2_matches += stats[base + LZO_DBG_DEC_M2_MATCHES];
        m3_matches += stats[base + LZO_DBG_DEC_M3_MATCHES];
        m4_matches += stats[base + LZO_DBG_DEC_M4_MATCHES];
        first_literal_run_bytes += stats[base + LZO_DBG_DEC_FIRST_LITERAL_RUN_BYTES];
        first_literal_run_ops += stats[base + LZO_DBG_DEC_FIRST_LITERAL_RUN_OPS];
        post_match_literal_bytes += stats[base + LZO_DBG_DEC_POST_MATCH_LITERAL_BYTES];
        post_match_literal_ops += stats[base + LZO_DBG_DEC_POST_MATCH_LITERAL_OPS];
        eof_markers += stats[base + LZO_DBG_DEC_EOF_MARKERS];
    }

    fprintf(stderr,
            "%s blocks=%zu tokens=%llu literal_bytes=%llu match_bytes=%llu small_offsets=%llu output_errors=%llu literal_ops=%llu match_ops=%llu overlap_matches=%llu\n",
            (tag ? tag : "[LZO-DBG][DECOMP]"),
            num_blocks,
            tokens,
            literal_bytes,
            match_bytes,
            small_offsets,
            output_errors,
            literal_ops,
            match_ops,
            overlap_matches);
    fprintf(stderr,
            "%s detail m2_matches=%llu m3_matches=%llu m4_matches=%llu first_literal_run_bytes=%llu first_literal_run_ops=%llu post_match_literal_bytes=%llu post_match_literal_ops=%llu eof_markers=%llu\n",
            (tag ? tag : "[LZO-DBG][DECOMP]"),
            m2_matches,
            m3_matches,
            m4_matches,
            first_literal_run_bytes,
            first_literal_run_ops,
            post_match_literal_bytes,
            post_match_literal_ops,
            eof_markers);
}

static int lzo_zero_buffer(cl_command_queue queue, cl_mem buf, size_t bytes) {
    if (!buf || bytes == 0) return 0;
#if defined(CL_VERSION_1_2)
    {
        static const cl_uint zero = 0;
        cl_int ferr = clEnqueueFillBuffer(queue, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
        if (ferr == CL_SUCCESS) return 0;
    }
#endif
    {
        cl_int err;
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || mapped == NULL) return -1;
        memset(mapped, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
    }
    return 0;
}

static int lzo_zero_buffer_range(cl_command_queue queue, cl_mem buf, size_t offset, size_t bytes) {
    if (!buf || bytes == 0) return 0;
#if defined(CL_VERSION_1_2)
    {
        static const cl_uint zero = 0;
        cl_int ferr = clEnqueueFillBuffer(queue, buf, &zero, sizeof(zero), offset, bytes, 0, NULL, NULL);
        if (ferr == CL_SUCCESS) return 0;
    }
#endif
    {
        cl_int err;
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, offset, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || mapped == NULL) return -1;
        memset(mapped, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
    }
    return 0;
}

static inline size_t core_lzo_worst(size_t sz) {
    return sz + sz / 16 + 64 + 3;
}

#define core_now_ns lzo_now_ns

void lzo_gpu_workspace_init(lzo_gpu_workspace_t* ws) {
    memset(ws, 0, sizeof(lzo_gpu_workspace_t));
    ws->comp_epoch_base = 1;
}

void lzo_gpu_workspace_free(lzo_gpu_workspace_t* ws) {
    if (ws->d_in) clReleaseMemObject(ws->d_in);
    if (ws->d_out) clReleaseMemObject(ws->d_out);
    if (ws->d_len) clReleaseMemObject(ws->d_len);
    if (ws->d_dict) clReleaseMemObject(ws->d_dict);
    if (ws->d_comp) clReleaseMemObject(ws->d_comp);
    if (ws->d_off) clReleaseMemObject(ws->d_off);
    if (ws->d_comp_lens) clReleaseMemObject(ws->d_comp_lens);
    if (ws->d_decomp_out) clReleaseMemObject(ws->d_decomp_out);
    if (ws->d_out_lens) clReleaseMemObject(ws->d_out_lens);
    memset(ws, 0, sizeof(lzo_gpu_workspace_t));
}

static cl_mem core_get_or_create_buffer(cl_context ctx, cl_mem* cached_buf, size_t* cached_size,
                                    size_t required_size, cl_mem_flags flags, cl_int* err_out) {
    if (*cached_size < required_size) {
        if (*cached_buf) {
            clReleaseMemObject(*cached_buf);
            *cached_buf = NULL;
            *cached_size = 0;
        }
        *cached_buf = clCreateBuffer(ctx, flags, required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS) {
            *cached_size = required_size;
        }
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

static int lzo_read_buffer_auto(cl_command_queue queue, cl_mem buf, void* dst, size_t bytes, int standard_copy) {
    cl_int err;
    void* mapped;

    if (!buf || !dst || bytes == 0) return 0;

    if (standard_copy) {
        err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
        return (err == CL_SUCCESS) ? 0 : -1;
    }

    mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &err);
    if (err == CL_SUCCESS && mapped) {
        memcpy(dst, mapped, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        return 0;
    }

    err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

static unsigned long lzo_event_elapsed_us(cl_event ev) {
    cl_ulong st = 0;
    cl_ulong en = 0;
    if (!ev) return 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return 0;
    if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return 0;
    if (en <= st) return 0;
    return (unsigned long)((en - st) / 1000ULL);
}

static int lzo_readback_to_file_chunked(cl_command_queue queue,
                                        cl_mem buf,
                                        size_t total_bytes,
                                        FILE* fout,
                                        size_t chunk_bytes,
                                        unsigned long* download_us_acc,
                                        unsigned long* write_us_acc) {
    unsigned char* staging;
    size_t off = 0;

    if (!queue || !buf || !fout) return -1;
    if (total_bytes == 0) return 0;
    if (chunk_bytes < 256U * 1024U) chunk_bytes = 256U * 1024U;

    staging = (unsigned char*)malloc(chunk_bytes);
    if (!staging) return -1;

    while (off < total_bytes) {
        size_t step = total_bytes - off;
        cl_int err;
        uint64_t t0;
        if (step > chunk_bytes) step = chunk_bytes;

        t0 = core_now_ns();
        err = clEnqueueReadBuffer(queue, buf, CL_TRUE, off, step, staging, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            free(staging);
            return -1;
        }
        if (download_us_acc) {
            *download_us_acc += (unsigned long)((core_now_ns() - t0) / 1000ULL);
        }

        t0 = core_now_ns();
        if (fwrite(staging, 1, step, fout) != step) {
            free(staging);
            return -1;
        }
        if (write_us_acc) {
            *write_us_acc += (unsigned long)((core_now_ns() - t0) / 1000ULL);
        }

        off += step;
    }

    free(staging);
    return 0;
}

static int lzo_env_flag_value(const char* name, int* is_set) {
    const char* env = getenv(name);
    if (is_set) *is_set = 0;
    if (!env || !*env) return 0;
    if (is_set) *is_set = 1;
    if (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 || strcasecmp(env, "yes") == 0 || strcasecmp(env, "on") == 0) return 1;
    if (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0 || strcasecmp(env, "off") == 0) return 0;
    return atoi(env) != 0;
}

static unsigned lzo_env_unsigned_value(const char* name, unsigned defv) {
    const char* env = getenv(name);
    char* end = NULL;
    unsigned long parsed;
    if (!env || !*env) return defv;
    parsed = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || parsed > UINT_MAX) return defv;
    return (unsigned)parsed;
}

static size_t lzo_device_max_wg_size(cl_device_id device) {
    static cl_device_id cached_dev = NULL;
    static size_t cached_wg = 256;
    size_t max_wg = cached_wg;
    if (device && device != cached_dev) {
        max_wg = 256;
        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
        cached_dev = device;
        cached_wg = max_wg;
    }
    if (max_wg == 0) max_wg = 1;
    return max_wg;
}

static size_t lzo_sanitize_local_size(cl_device_id device, size_t requested, size_t upper_items) {
    size_t l = requested;
    size_t max_wg;
    if (upper_items == 0) return 1;

    if (l == 0) {
        if (upper_items >= 4096) l = 256;
        else if (upper_items >= 1024) l = 128;
        else if (upper_items >= 256) l = 64;
        else l = 32;
    }

    max_wg = lzo_device_max_wg_size(device);
    if (l > max_wg) l = max_wg;
    if (l > upper_items) l = upper_items;
    if (l == 0) l = 1;

    {
        size_t p2 = 1;
        while ((p2 << 1) <= l) p2 <<= 1;
        l = p2;
    }
    return l;
}

/* Core compression implementation */
int lzo_compress_core(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    const lzo_compress_params_t* params,
    int skip_input_upload,
    lzo_gpu_workspace_t* ws,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_total_start = core_now_ns();
    int debug_sched = lzo_env_flag_enabled("LZO_GPU_DEBUG_SCHED");
    int debug_counters = (params->debug || lzo_debug_counters_enabled()) ? 1 : 0;

    int use_standard_copy = params->standard_copy ? 1 : 0;
    cl_mem d_in = NULL;

    /* 1. 获取文件大小 (使用 stat 替代 fopen/fseek, 避免重复打开) */
    struct stat st;
    if (stat(input_path, &st) != 0) {
        perror("stat input");
        return -1;
    }
    size_t in_sz = (size_t)st.st_size;
    if (in_sz == 0) {
        /* Handle empty file */
        return -1;
    }

    /* 2. 准备输入缓冲区 (如果是 Daemon 模式,此处通常会命中预分配好的常驻缓冲) */
    uint64_t t_buf_in_start = core_now_ns();
    if (!skip_input_upload) {
        cl_mem_flags in_flags = CL_MEM_READ_ONLY;
        if (!use_standard_copy) in_flags |= CL_MEM_ALLOC_HOST_PTR;
        d_in = core_get_or_create_buffer(ctx, &ws->d_in, &ws->in_size, in_sz, in_flags, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] failed to create input buffer: %d\n", err);
            return -1;
        }
    } else {
        if (!ws->d_in || ws->in_size < in_sz) {
            fprintf(stderr, "[CORE] skip_input_upload requested but cached input buffer is unavailable\n");
            return -1;
        }
        d_in = ws->d_in;
    }

    void* mapped_in = NULL;
    if (!skip_input_upload && !use_standard_copy) {
        /* Map with Non-blocking if possible, but CORE uses CL_TRUE for simplicity */
        mapped_in = clEnqueueMapBuffer(queue, d_in, CL_TRUE, CL_MAP_WRITE, 0, in_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] Map input buffer failed: %d\n", err);
            return -1;
        }
    }
    uint64_t t_buf_in_end = core_now_ns();
    unsigned long buffer_in_us = (t_buf_in_end - t_buf_in_start) / 1000;

    unsigned long read_us = 0;
    unsigned long upload_us = 0;
    void* host_in = NULL;

    if (!skip_input_upload && !use_standard_copy) {
        /* 直接读入映射好的显存空间 (Zero-copy 核心) */
        if (lzo_read_file_to_buf(input_path, mapped_in, in_sz, &read_us) != 0) {
            perror("lzo_read_file_to_buf");
            clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
            return -1;
        }
        /* upload_us will be measured after unmap */
    } else if (!skip_input_upload) {
        int rc_mem = lzo_aligned_alloc_portable(&host_in, ALIGN_BYTES, in_sz);
        if (rc_mem != 0 || host_in == NULL) {
            host_in = malloc(in_sz);
            if (!host_in) {
                perror("malloc host_in");
                return -1;
            }
        }

        /* Use unified IO module */
        if (lzo_read_file_to_buf(input_path, host_in, in_sz, &read_us) != 0) {
            perror("lzo_read_file_to_buf");
            lzo_aligned_free_portable(host_in);
            return -1;
        }
        /* upload after blocking calc */
    }

    /* 3. Blocking calculation */
    size_t blk = 0, nblk = 0;
    if (skip_input_upload && ws->comp_cached_input_size == in_sz &&
        ws->comp_cached_blk_size > 0 && ws->comp_cached_nblk > 0) {
        blk = ws->comp_cached_blk_size;
        nblk = ws->comp_cached_nblk;
    } else {
        const unsigned char* entropy_ptr = NULL;
        if (use_standard_copy)
            entropy_ptr = (const unsigned char*)host_in;
        else
            entropy_ptr = (const unsigned char*)mapped_in;

        size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
        lzo_choose_blocking_adaptive(entropy_ptr, in_sz, device, blk_bytes, 0, &blk, &nblk, params->debug);
    }


    if (!skip_input_upload && !use_standard_copy) {
        uint64_t t_unmap_start = core_now_ns();
        CHECK(clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL));
        uint64_t t_unmap_end = core_now_ns();
        upload_us = (t_unmap_end - t_unmap_start) / 1000;
    } else if (!skip_input_upload) {
        uint64_t t_upload_start2 = core_now_ns();
        err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] clEnqueueWriteBuffer failed: %d\n", err);
            lzo_aligned_free_portable(host_in);
            return -1;
        }
        lzo_aligned_free_portable(host_in);
        host_in = NULL;
        uint64_t t_upload_end2 = core_now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
    }

    size_t worst_blk = core_lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_out, &ws->out_size, out_needed,
                                             use_standard_copy ? CL_MEM_WRITE_ONLY : (CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                             &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create output buffer: %d\n", err);
        return -1;
    }
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buffer_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    /* Buffer Alloc (len) timing removed per request; keep allocation but don't time it */
    cl_mem d_len = core_get_or_create_buffer(ctx, &ws->d_len, &ws->len_size, len_needed,
                                             use_standard_copy ? CL_MEM_READ_WRITE : (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR),
                                             &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create len buffer: %d\n", err);
        return -1;
    }

    cl_uint kernel_num_args = 0;
    int kernel_has_dbg = 0;
    cl_mem dbg_comp_buf = NULL;
    int dbg_comp_enabled = 0;
    if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
        kernel_has_dbg = (kernel_num_args >= 11U);
    }
    dbg_comp_enabled = (debug_counters && kernel_has_dbg);
    if (debug_counters && !kernel_has_dbg) {
        fprintf(stderr, "[LZO-DBG][COMP] warning: kernel has no debug args, counters disabled\n");
    }

    cl_uint in_sz_cl = (cl_uint)in_sz, blk_cl = (cl_uint)blk, worst_blk_cl = (cl_uint)worst_blk;

    size_t local_size = 1;
    local_size = (params->local_size_param > 0) ? (size_t)params->local_size_param : 1;
    if (params->local_size_param > 0 && params->debug) {
        fprintf(stderr, "[CORE] using local_size=%zu\n", local_size);
    }

    size_t target_items = nblk;
    if (target_items == 0) target_items = 1;
    {
        cl_uint cus = 0;
        cl_ulong global_mem = 0;
        cl_ulong max_alloc = 0;
        size_t dict_per_block = (1ULL << params->level) * lzo_dict_entry_bytes_for_block(params->block_size, params->level);
        size_t occ_cap = 0;
        size_t mem_cap = 0;
        size_t safe_mem = 0;
        size_t safe_alloc = 0;

        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
        occ_cap = (cus > 0) ? ((size_t)cus * (size_t)LZO_OCC_FACTOR_DEFAULT) : 4096U;
        if (occ_cap < 1024U) occ_cap = 1024U;

        if (dict_per_block > 0) {
            size_t cap_by_global = SIZE_MAX;
            size_t cap_by_alloc = SIZE_MAX;

            if (global_mem > 0) {
                /* Reserve ~1/6 global memory for dict pool to avoid OOM. */
                safe_mem = (size_t)(global_mem / 6ULL);
                cap_by_global = safe_mem / dict_per_block;
            }
            if (max_alloc > 0) {
                /* Reserve headroom to avoid hitting driver hard limits exactly. */
                safe_alloc = (size_t)(max_alloc * 9ULL / 10ULL);
                cap_by_alloc = safe_alloc / dict_per_block;
            }
            mem_cap = (cap_by_global < cap_by_alloc) ? cap_by_global : cap_by_alloc;
        }
        if (mem_cap == 0) mem_cap = 1;

        if (target_items > occ_cap) target_items = occ_cap;
        if (target_items > mem_cap) target_items = mem_cap;
    }
    local_size = lzo_sanitize_local_size(device, local_size, target_items);
    size_t global_size = ((target_items + local_size - 1) / local_size) * local_size;
    if (global_size == 0) {
        global_size = 1;
        local_size = 1;
    }

    uint32_t pool_size = (uint32_t)global_size;

    if (params->debug) {
        fprintf(stderr, "[CORE] work_items=%zu, dict_pool_size=%u\n", global_size, pool_size);
    }

    if (debug_sched) {
        double blk_per_wi = (target_items > 0) ? ((double)nblk / (double)target_items) : 0.0;
        fprintf(stderr,
                "[LZO-SCHED] alg=%d level=%d blocks=%zu bs=%zu lsz=%zu target_wi=%zu g=%zu blk_per_wi=%.2f\n",
                params->alg_id, params->level, nblk, blk, local_size, target_items, global_size, blk_per_wi);
    }

    if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
    if ((uint32_t)nblk + 2U >= 4095U || ws->comp_epoch_base + (uint32_t)nblk + 1U > 4095U) {
        if (ws->d_dict && ws->dict_size > 0) {
            (void)lzo_zero_buffer(queue, ws->d_dict, ws->dict_size);
        }
        ws->comp_epoch_base = 1;
    }
    uint32_t epoch_base = ws->comp_epoch_base;
    ws->comp_epoch_base += (uint32_t)nblk + 1U;

    {
        size_t dict_per_block = (1ULL << params->level) * lzo_dict_entry_bytes_for_block(params->block_size, params->level);
        size_t total_dict_size = (size_t)pool_size * dict_per_block + sizeof(cl_uint);
        size_t prev_dict_size = ws->dict_size;

        cl_mem d_dict = core_get_or_create_buffer(ctx, &ws->d_dict, &ws->dict_size, total_dict_size, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] failed to create dictionary buffer: %d\n", err);
            return -1;
        }
        if (ws->dict_size > prev_dict_size) {
            (void)lzo_zero_buffer_range(queue, d_dict, prev_dict_size, ws->dict_size - prev_dict_size);
        }
        {
            size_t counter_offset = (size_t)pool_size * dict_per_block;
            (void)lzo_zero_buffer_range(queue, d_dict, counter_offset, sizeof(cl_uint));
        }

        int need_set_stable_args = 1;
        if (skip_input_upload &&
            ws->comp_kernel_args_set &&
            ws->comp_cached_kernel == kernel &&
            ws->comp_cached_d_in == d_in &&
            ws->comp_cached_d_out == d_out &&
            ws->comp_cached_d_len == d_len &&
            ws->comp_cached_d_dict == d_dict &&
            ws->comp_cached_in_sz == in_sz_cl &&
            ws->comp_cached_blk == blk_cl &&
            ws->comp_cached_worst_blk == worst_blk_cl &&
            ws->comp_cached_pool_size == (cl_uint)pool_size) {
            need_set_stable_args = 0;
        }

        if (need_set_stable_args) {
            CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
            CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
            CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
            CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
            CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
            CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));
            CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
            CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));

            ws->comp_kernel_args_set = 1;
            ws->comp_cached_kernel = kernel;
            ws->comp_cached_d_in = d_in;
            ws->comp_cached_d_out = d_out;
            ws->comp_cached_d_len = d_len;
            ws->comp_cached_d_dict = d_dict;
            ws->comp_cached_in_sz = in_sz_cl;
            ws->comp_cached_blk = blk_cl;
            ws->comp_cached_worst_blk = worst_blk_cl;
            ws->comp_cached_pool_size = (cl_uint)pool_size;
            ws->comp_cached_input_size = in_sz;
            ws->comp_cached_blk_size = blk;
            ws->comp_cached_nblk = nblk;
        }

        CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &epoch_base));

        if (dbg_comp_enabled) {
            size_t dbg_comp_bytes = nblk * LZO_DBG_COMP_N * sizeof(uint32_t);
            dbg_comp_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dbg_comp_bytes, NULL, &err);
            if (err != CL_SUCCESS || !dbg_comp_buf || lzo_zero_buffer(queue, dbg_comp_buf, dbg_comp_bytes) != 0) {
                if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
                dbg_comp_buf = NULL;
                dbg_comp_enabled = 0;
                fprintf(stderr, "[LZO-DBG][COMP] warning: failed to enable debug counters\n");
            }
        }
        if (kernel_has_dbg) {
            cl_mem dbg_arg = dbg_comp_enabled ? dbg_comp_buf : d_len;
            cl_uint dbg_flag = dbg_comp_enabled ? 1U : 0U;
            CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &dbg_arg));
            CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uint), &dbg_flag));
        }
    }

    uint64_t t_exec_start = core_now_ns();
    {
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &ev));
        clWaitForEvents(1, &ev);
        if (debug_sched) {
            cl_ulong q = 0, s = 0, st = 0, e = 0;
            if (clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(q), &q, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(s), &s, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL) == CL_SUCCESS) {
                fprintf(stderr,
                        "[LZO-PROF] queue_to_submit=%.3f us submit_to_start=%.3f us device_exec=%.3f us\n",
                        (double)(s - q) / 1000.0,
                        (double)(st - s) / 1000.0,
                        (double)(e - st) / 1000.0);
            }
        }
        clReleaseEvent(ev);
    }
    clFinish(queue);
    uint64_t t_exec_end = core_now_ns();
    unsigned long exec_us_host = (unsigned long)((t_exec_end - t_exec_start) / 1000);

    cl_uint* len_arr = malloc(nblk * sizeof(cl_uint));
    if (!len_arr || lzo_read_buffer_auto(queue, d_len, len_arr, nblk * sizeof(cl_uint), use_standard_copy) != 0) {
        fprintf(stderr, "[CORE] failed to read len buffer\n");
        if (len_arr) free(len_arr);
        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
        return -1;
    }

    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) {
        comp_total += len_arr[i];
    }

    if (dbg_comp_enabled && dbg_comp_buf) {
        size_t dbg_comp_bytes = nblk * LZO_DBG_COMP_N * sizeof(uint32_t);
        uint32_t* dbg_comp_stats = (uint32_t*)malloc(dbg_comp_bytes);
        if (dbg_comp_stats) {
            if (clEnqueueReadBuffer(queue, dbg_comp_buf, CL_TRUE, 0, dbg_comp_bytes, dbg_comp_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lzo_print_comp_debug_stats(dbg_comp_stats, nblk, "[LZO-DBG][COMP]");
            }
            free(dbg_comp_stats);
        }
    }

    unsigned long download_us = 0;
    void* host_comp = NULL;

    uint64_t t_down_start = core_now_ns();
    if (use_standard_copy) {
        host_comp = malloc(out_needed);
        if (!host_comp) {
            fprintf(stderr, "[CORE] failed to allocate output staging buffer\n");
            if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
            free(len_arr);
            return -1;
        }
        if (lzo_read_buffer_auto(queue, d_out, host_comp, out_needed, 1) != 0) {
            fprintf(stderr, "[CORE] failed to download output buffer\n");
            if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
            free(len_arr);
            free(host_comp);
            return -1;
        }
    } else {
        void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, out_needed, 0, NULL, NULL, &err);
        CHECK(err);
        host_comp = mapped_out;
    }
    uint64_t t_down_end = core_now_ns();
    download_us = (unsigned long)((t_down_end - t_down_start) / 1000);

    if (params->debug) {
        fprintf(stderr, "[CORE][DEBUG-LENS] comp_total=%zu, out_needed=%zu\n", comp_total, out_needed);
    }



    uint64_t t_write_start = core_now_ns();
    int write_ret = 0;
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        write_ret = lzo_write_compressed_file(output_path, in_sz, blk, nblk, len_arr, host_comp,
                                               worst_blk, params->alg_id, params->debug);
    }
    uint64_t t_write_end = core_now_ns();
    unsigned long write_us = (t_write_end - t_write_start) / 1000;

    if (write_ret != 0) {
        fprintf(stderr, "[CORE] Failed to write compressed file\n");
        if (!use_standard_copy) {
            CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));
        } else {
            free(host_comp);
        }
        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
        free(len_arr);
        return -1;
    }

    if (!use_standard_copy) {
        CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));
    } else {
        free(host_comp);
    }

    free(len_arr);
    if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
    clFlush(queue);
    clFinish(queue);

    uint64_t t_total_end = core_now_ns();
    unsigned long total_us = (t_total_end - t_total_start) / 1000;
    *time_us_out = total_us; *output_size_out = comp_total;

    unsigned long buffer_alloc_us = buffer_in_us + buffer_out_us;
    if (t_out) {
        t_out->in_size = (unsigned long long)in_sz;
        t_out->out_size = (unsigned long long)comp_total;
        t_out->algo_config = (unsigned long long)params->level;
        t_out->file_read_us = read_us;
        t_out->ocl_setup_us = g_ocl_init_us + g_kernel_load_us;
        t_out->buffer_alloc_us = buffer_alloc_us;
        t_out->data_upload_us = upload_us;
        /* buffer_alloc_out_us / buffer_alloc_len_us removed */
        t_out->kernel_exec_us = exec_us_host;
        t_out->download_total_us = download_us;
        t_out->file_write_us = write_us;
        t_out->blk_size_bytes = (unsigned long)blk;
        t_out->nblk = (unsigned long)nblk;
        t_out->global_size = (unsigned long)global_size;
        t_out->local_size = (unsigned long)local_size;
    }

    if (g_verbose) {
        /* verbose output is handled by lzo_gpu.c wrapper to ensure
         * standalone total includes OpenCL init/kernel load.
         */
    }

    return write_ret;
}

/* Core decompression implementation (adapted from daemon_decompress.c) */
int lzo_decompress_core(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    lzo_gpu_workspace_t* ws,
    int standard_copy,
    int local_size_param, int debug,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_start = core_now_ns();

    /* Resource tracking for cleanup */
    FILE* f_in = NULL;
    uint32_t* len_arr = NULL;
    void* comp_host = NULL;
    void* mapped_out = NULL;
    cl_mem dbg_dec_buf = NULL;
    FILE* fout = NULL;
    int ret = -1; /* Default failure */

    f_in = fopen(input_path, "rb");
    if (!f_in) {
        perror("fopen input");
        goto cleanup;
    }

    uint16_t magic;
    if (fread(&magic, sizeof(magic), 1, f_in) != 1) {
        perror("fread magic");
        goto cleanup;
    }
    if (magic != 0x4C5A) {
        fprintf(stderr, "[DECOMP] wrong file magic\n");
        goto cleanup;
    }

    uint32_t orig_sz, blk_sz, nblk, alg_id;
    if (fread(&orig_sz, sizeof(orig_sz), 1, f_in) != 1 ||
        fread(&blk_sz, sizeof(blk_sz), 1, f_in) != 1 ||
        fread(&nblk, sizeof(nblk), 1, f_in) != 1 ||
        fread(&alg_id, sizeof(alg_id), 1, f_in) != 1) {
        perror("fread header");
        goto cleanup;
    }

    len_arr = malloc(nblk * sizeof(uint32_t));
    if (!len_arr) {
        perror("malloc len_arr");
        goto cleanup;
    }
    if (fread(len_arr, sizeof(uint32_t), nblk, f_in) != nblk) {
        perror("fread len_arr");
        goto cleanup;
    }

    long current_pos = ftell(f_in); fseek(f_in, 0, SEEK_END); long file_sz = ftell(f_in); fseek(f_in, current_pos, SEEK_SET);
    size_t comp_sz = file_sz - current_pos;
    if (g_verbose) {
        fprintf(stderr, "[DECOMP] file summary: orig=%u blk=%u nblk=%u comp=%zu\n", orig_sz, blk_sz, nblk, comp_sz);
    }

    int use_standard_copy_decomp = standard_copy ? 1 : 0;

    uint64_t t_buf_comp_start = core_now_ns();
    /* For zero-copy decompression, use ALLOC_HOST_PTR to get pinned memory */
    cl_mem_flags comp_flags = use_standard_copy_decomp ? CL_MEM_READ_ONLY : (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
    cl_mem d_comp = core_get_or_create_buffer(ctx, &ws->d_comp, &ws->comp_size, comp_sz, comp_flags, &err); CHECK(err);
    uint64_t t_buf_comp_end = core_now_ns();
    unsigned long buf_comp_us = (t_buf_comp_end - t_buf_comp_start) / 1000;

    /* Buffer Alloc (off) timing removed per request; keep allocation but don't time it */
    cl_mem d_off = core_get_or_create_buffer(ctx, &ws->d_off, &ws->off_size, nblk * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); CHECK(err);
    cl_mem d_comp_lens = core_get_or_create_buffer(ctx, &ws->d_comp_lens, &ws->comp_lens_size, nblk * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); CHECK(err);

    /* Build offsets directly into device-backed buffer to avoid host off_arr allocation/upload. */
    {
        void* mapped_off = clEnqueueMapBuffer(queue, d_off, CL_TRUE, CL_MAP_WRITE,
                                              0, nblk * sizeof(cl_uint),
                                              0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !mapped_off) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_off failed: %d\n", err);
            goto cleanup;
        }
        void* mapped_lens = clEnqueueMapBuffer(queue, d_comp_lens, CL_TRUE, CL_MAP_WRITE,
                                               0, nblk * sizeof(cl_uint),
                                               0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !mapped_lens) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_comp_lens failed: %d\n", err);
            clEnqueueUnmapMemObject(queue, d_off, mapped_off, 0, NULL, NULL);
            goto cleanup;
        }
        {
            cl_uint* off_dev = (cl_uint*)mapped_off;
            cl_uint* lens_dev = (cl_uint*)mapped_lens;
            cl_uint off = 0;
            for (uint32_t i = 0; i < nblk; ++i) {
                off_dev[i] = off;
                lens_dev[i] = len_arr[i];
                off += len_arr[i];
            }
        }
        err = clEnqueueUnmapMemObject(queue, d_off, mapped_off, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_off failed: %d\n", err);
            goto cleanup;
        }
        err = clEnqueueUnmapMemObject(queue, d_comp_lens, mapped_lens, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_comp_lens failed: %d\n", err);
            goto cleanup;
        }
    }
    free(len_arr);
    len_arr = NULL;

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_size, orig_sz,
                                             use_standard_copy_decomp ? CL_MEM_WRITE_ONLY : (CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                             &err); CHECK(err);

    {
        int track_out_lens_cfg = lzo_env_flag_enabled("LZO_GPU_DECOMP_TRACK_OUT_LENS");
        if (!track_out_lens_cfg && ws->d_out_lens) {
            clReleaseMemObject(ws->d_out_lens);
            ws->d_out_lens = NULL;
            ws->lens_size = 0;
        }
    }

    cl_mem d_out_lens = NULL;

    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buf_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    unsigned long buf_us = buf_comp_us + buf_out_us;

    /* Allocate host buffer for compressed data */
    if (use_standard_copy_decomp) {
        /* Standard copy */
        comp_host = malloc(comp_sz);
        if (!comp_host) {
            perror("malloc comp_host");
            goto cleanup;
        }
    } else {
        /* Zero-copy mapping */
        comp_host = clEnqueueMapBuffer(queue, d_comp, CL_TRUE, CL_MAP_WRITE, 0, comp_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !comp_host) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_comp failed: %d\n", err);
            goto cleanup;
        }
    }

    /* Read compressed data */
    if (fread(comp_host, 1, comp_sz, f_in) != comp_sz) {
        perror("fread compressed data");
        goto cleanup;
    }
    fclose(f_in);
    f_in = NULL;
    /* Now upload to GPU */
    uint64_t t_upload_start = core_now_ns();

    if (use_standard_copy_decomp) {
        /* Standard copy: explicit upload with async write */
        cl_event evt_write_comp;
        err = clEnqueueWriteBuffer(queue, d_comp, CL_FALSE, 0, comp_sz, comp_host, 0, NULL, &evt_write_comp);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueWriteBuffer d_comp failed: %d\n", err);
            goto cleanup;
        }

        /* Wait for compressed payload upload to complete */
        clWaitForEvents(1, &evt_write_comp);
        clReleaseEvent(evt_write_comp);
    } else {
        /* Zero-copy: data is already in pinned memory, just unmap */
        err = clEnqueueUnmapMemObject(queue, d_comp, comp_host, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_comp failed: %d\n", err);
            goto cleanup;
        }
        comp_host = NULL; /* Unmapped, no longer valid */
    }

/* Free comp_host only if we allocated it (standard copy mode) */
    if (use_standard_copy_decomp && comp_host) {
        free(comp_host);
    }
    comp_host = NULL;

    uint64_t t_upload_end = core_now_ns();

    int track_out_lens = lzo_env_flag_enabled("LZO_GPU_DECOMP_TRACK_OUT_LENS");
    if (track_out_lens) {
        d_out_lens = core_get_or_create_buffer(ctx, &ws->d_out_lens, &ws->lens_size, nblk * sizeof(cl_uint),
                                               use_standard_copy_decomp ? CL_MEM_WRITE_ONLY : (CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR),
                                               &err);
        CHECK(err);
    }

    {
        /*
         * Stage-2 architecture policy:
         * out_lens is not consumed by current host writeback path, so default to
         * disabling kernel-side out_lens stores to reduce global-memory traffic.
         * Set LZO_GPU_DECOMP_TRACK_OUT_LENS=1 to re-enable for diagnostics.
         */
    }
    cl_mem out_lens_arg = track_out_lens ? d_out_lens : (cl_mem)NULL;

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_comp_lens));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_lens_arg));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &nblk));

    {
        cl_uint kernel_num_args = 0;
        int kernel_has_dbg = 0;
        int dbg_dec_enabled = (debug || lzo_debug_counters_enabled()) ? 1 : 0;
        if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
            kernel_has_dbg = (kernel_num_args >= 10U);
        }
        if (dbg_dec_enabled && !kernel_has_dbg) {
            fprintf(stderr, "[LZO-DBG][DECOMP] warning: kernel has no debug args, counters disabled\n");
            dbg_dec_enabled = 0;
        }
        if (dbg_dec_enabled) {
            size_t dbg_dec_bytes = (size_t)nblk * LZO_DBG_DEC_N * sizeof(uint32_t);
            dbg_dec_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, dbg_dec_bytes, NULL, &err);
            if (err != CL_SUCCESS || !dbg_dec_buf || lzo_zero_buffer(queue, dbg_dec_buf, dbg_dec_bytes) != 0) {
                if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);
                dbg_dec_buf = NULL;
                dbg_dec_enabled = 0;
                fprintf(stderr, "[LZO-DBG][DECOMP] warning: failed to enable debug counters\n");
            }
        }
        if (kernel_has_dbg) {
            cl_mem dbg_arg = dbg_dec_enabled ? dbg_dec_buf : out_lens_arg;
            cl_uint dbg_flag = dbg_dec_enabled ? 1U : 0U;
            CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &dbg_arg));
            CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uint), &dbg_flag));
        }
    }

    size_t local_size = lzo_sanitize_local_size(device,
                                                (local_size_param > 0) ? (size_t)local_size_param : 1,
                                                (size_t)nblk);
    size_t global_size = ((size_t)nblk + local_size - 1) / local_size * local_size;

    uint64_t t_exec_start = core_now_ns();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK(err);
    clFinish(queue);
    uint64_t t_exec_end = core_now_ns();

    if (dbg_dec_buf) {
        size_t dbg_dec_bytes = (size_t)nblk * LZO_DBG_DEC_N * sizeof(uint32_t);
        uint32_t* dbg_dec_stats = (uint32_t*)malloc(dbg_dec_bytes);
        if (dbg_dec_stats) {
            if (clEnqueueReadBuffer(queue, dbg_dec_buf, CL_TRUE, 0, dbg_dec_bytes, dbg_dec_stats, 0, NULL, NULL) == CL_SUCCESS) {
                lzo_print_dec_debug_stats(dbg_dec_stats, (size_t)nblk, "[LZO-DBG][DECOMP]");
            }
            free(dbg_dec_stats);
        }
    }

    unsigned long download_us = 0;
    unsigned long write_us = 0;
    {
        size_t chunk_threshold_kb = (size_t)lzo_env_unsigned_value("LZO_GPU_DECOMP_CHUNKED_THRESHOLD_KB", 16384U);
        size_t chunk_readback_kb = (size_t)lzo_env_unsigned_value("LZO_GPU_DECOMP_READBACK_KB", 8192U);
        int force_chunked_set = 0;
        int force_chunked = lzo_env_flag_value("LZO_GPU_DECOMP_FORCE_CHUNKED", &force_chunked_set);
        int disable_chunked = lzo_env_flag_enabled("LZO_GPU_DECOMP_DISABLE_CHUNKED");
        /*
         * Architecture policy:
         * - Windows has shown consistent real-file readback/write benefit from chunked output
         *   on both iGPU-style and dGPU-style paths, so auto enables it for large outputs;
         * - Linux remains conservative and keeps chunked disabled by default unless forced.
         */
        int chunked_backend_ok =
#if defined(_WIN32) || defined(_WIN64)
            1;
#else
            0;
#endif
        int use_chunked_output = 0;

        if (force_chunked_set) {
            use_chunked_output = force_chunked && output_path != NULL;
        } else if (!disable_chunked) {
            use_chunked_output = (output_path != NULL && chunked_backend_ok && orig_sz >= chunk_threshold_kb * 1024ULL);
        }

        if (use_chunked_output) {
            fout = fopen(output_path, "wb");
            if (!fout) {
                perror("fopen output");
                goto cleanup;
            }
            if (lzo_readback_to_file_chunked(queue,
                                             d_out,
                                             orig_sz,
                                             fout,
                                             chunk_readback_kb * 1024ULL,
                                             &download_us,
                                             &write_us) != 0) {
                fprintf(stderr, "[DECOMP] chunked readback/write failed\n");
                goto cleanup;
            }
            fclose(fout);
            fout = NULL;
        } else {
            uint64_t t_download_start = core_now_ns();
            if (use_standard_copy_decomp) {
                mapped_out = malloc(orig_sz);
                if (!mapped_out) {
                    fprintf(stderr, "[DECOMP] malloc output staging buffer failed\n");
                    goto cleanup;
                }
                if (lzo_read_buffer_auto(queue, d_out, mapped_out, orig_sz, 1) != 0) {
                    fprintf(stderr, "[DECOMP] download output buffer failed\n");
                    goto cleanup;
                }
            } else {
                mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, orig_sz, 0, NULL, NULL, &err);
                CHECK(err);
            }
            download_us = (unsigned long)((core_now_ns() - t_download_start) / 1000ULL);

            if (output_path) {
                uint64_t t_write_start2 = core_now_ns();
                fout = fopen(output_path, "wb");
                if (!fout) {
                    perror("fopen output");
                    goto cleanup;
                }
                if (fwrite(mapped_out, 1, orig_sz, fout) != orig_sz) {
                    perror("fwrite");
                    goto cleanup;
                }
                fclose(fout);
                fout = NULL;
                write_us = (unsigned long)((core_now_ns() - t_write_start2) / 1000ULL);
            }

            if (use_standard_copy_decomp) {
                free(mapped_out);
                mapped_out = NULL;
            } else {
                CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));
                mapped_out = NULL;
            }
        }
    }

    size_t cache_threshold_mb = DEFAULT_DECOMP_CACHE_MB;
    size_t cache_threshold = cache_threshold_mb * 1024 * 1024;
    if (orig_sz > cache_threshold) {
        if (ws->d_decomp_out) {
            clReleaseMemObject(ws->d_decomp_out);
            ws->d_decomp_out = NULL;
            ws->decomp_out_size = 0;
        }
        if (ws->d_out_lens) {
            clReleaseMemObject(ws->d_out_lens);
            ws->d_out_lens = NULL;
            ws->lens_size = 0;
        }
    }

    uint64_t t_end = core_now_ns();
    *time_us_out = (t_end - t_start) / 1000;
    *output_size_out = orig_sz;

    unsigned long upload_us_local = (t_upload_end - t_upload_start) / 1000;
    unsigned long exec_host_us = (t_exec_end - t_exec_start) / 1000;
    unsigned long download_us_local = download_us;
    unsigned long write_us_local = write_us;

    if (t_out) {
        t_out->in_size = comp_sz;
        t_out->out_size = orig_sz;
        t_out->file_read_us = (t_buf_comp_start - t_start) / 1000;
        t_out->ocl_setup_us = g_ocl_init_us + g_kernel_load_us;
        t_out->buffer_alloc_us = buf_us;
        t_out->data_upload_us = upload_us_local;
        t_out->kernel_exec_us = exec_host_us;
        t_out->download_total_us = download_us_local;
        t_out->file_write_us = write_us_local;
        t_out->global_size = global_size; t_out->local_size = local_size; t_out->blk_size_bytes = blk_sz; t_out->nblk = nblk;
        t_out->algo_config = 0;
    }

    if (g_verbose) {
        /* verbose output is handled by lzo_gpu.c wrapper to ensure
         * standalone total includes OpenCL init/kernel load.
         */
    }

    ret = 0; /* Success */

cleanup:
    /* Unified cleanup: close files, free resources, unmap buffers */
    if (f_in) fclose(f_in);
    if (len_arr) free(len_arr);
    if (comp_host) {
        /* Only free if we allocated it ourselves (standard copy) */
        if (use_standard_copy_decomp) {
            free(comp_host);
        }
        /* If zero-copy and still mapped (error path), unmap it */
        /* Note: in success path, comp_host is already NULL after unmap */
    }
    if (fout) fclose(fout);
    if (mapped_out) {
        if (use_standard_copy_decomp) free(mapped_out);
        else clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL);
    }
    if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);

    return ret;
}
