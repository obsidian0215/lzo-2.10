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

static unsigned lzo_env_u32(const char* name, unsigned defv) {
    const char* v = getenv(name);
    char* endp = NULL;
    unsigned long n;
    if (!v || !*v) return defv;
    n = strtoul(v, &endp, 10);
    if (endp == v || *endp != '\0') return defv;
    if (n == 0 || n > UINT_MAX) return defv;
    return (unsigned)n;
}

static double lzo_env_f64(const char* name, double defv) {
    const char* v = getenv(name);
    char* endp = NULL;
    double d;
    if (!v || !*v) return defv;
    d = strtod(v, &endp);
    if (endp == v || *endp != '\0' || !isfinite(d)) return defv;
    return d;
}

enum {
    LZO_DBG_COMP_SEARCH_ITERS = 0,
    LZO_DBG_COMP_MATCH_FOUND,
    LZO_DBG_COMP_LITERAL_BYTES,
    LZO_DBG_COMP_MATCH_BYTES,
    LZO_DBG_COMP_INPUT_BYTES,
    LZO_DBG_COMP_OUTPUT_BYTES,
    LZO_DBG_COMP_N
};

enum {
    LZO_DBG_DEC_TOKENS = 0,
    LZO_DBG_DEC_LITERAL_BYTES,
    LZO_DBG_DEC_MATCH_BYTES,
    LZO_DBG_DEC_SMALL_OFFSETS,
    LZO_DBG_DEC_OUTPUT_ERROR,
    LZO_DBG_DEC_N
};

static int lzo_debug_counters_enabled(void) {
    const char* env = getenv("LZO_GPU_DEBUG_COUNTERS");
    if (!env || !*env) return 0;
    return strcmp(env, "0") != 0;
}

static void lzo_print_comp_debug_stats(const uint32_t* stats, size_t num_blocks, const char* tag) {
    unsigned long long search_iters = 0;
    unsigned long long match_found = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long input_bytes = 0;
    unsigned long long output_bytes = 0;

    if (!stats || num_blocks == 0) return;

    for (size_t i = 0; i < num_blocks; ++i) {
        const size_t base = i * LZO_DBG_COMP_N;
        search_iters += stats[base + LZO_DBG_COMP_SEARCH_ITERS];
        match_found += stats[base + LZO_DBG_COMP_MATCH_FOUND];
        literal_bytes += stats[base + LZO_DBG_COMP_LITERAL_BYTES];
        match_bytes += stats[base + LZO_DBG_COMP_MATCH_BYTES];
        input_bytes += stats[base + LZO_DBG_COMP_INPUT_BYTES];
        output_bytes += stats[base + LZO_DBG_COMP_OUTPUT_BYTES];
    }

    fprintf(stderr,
            "%s blocks=%zu search_iters=%llu match_found=%llu literal_bytes=%llu match_bytes=%llu input_bytes=%llu output_bytes=%llu\n",
            (tag ? tag : "[LZO-DBG][COMP]"),
            num_blocks,
            search_iters,
            match_found,
            literal_bytes,
            match_bytes,
            input_bytes,
            output_bytes);
}

static void lzo_print_dec_debug_stats(const uint32_t* stats, size_t num_blocks, const char* tag) {
    unsigned long long tokens = 0;
    unsigned long long literal_bytes = 0;
    unsigned long long match_bytes = 0;
    unsigned long long small_offsets = 0;
    unsigned long long output_errors = 0;

    if (!stats || num_blocks == 0) return;

    for (size_t i = 0; i < num_blocks; ++i) {
        const size_t base = i * LZO_DBG_DEC_N;
        tokens += stats[base + LZO_DBG_DEC_TOKENS];
        literal_bytes += stats[base + LZO_DBG_DEC_LITERAL_BYTES];
        match_bytes += stats[base + LZO_DBG_DEC_MATCH_BYTES];
        small_offsets += stats[base + LZO_DBG_DEC_SMALL_OFFSETS];
        output_errors += stats[base + LZO_DBG_DEC_OUTPUT_ERROR];
    }

    fprintf(stderr,
            "%s blocks=%zu tokens=%llu literal_bytes=%llu match_bytes=%llu small_offsets=%llu output_errors=%llu\n",
            (tag ? tag : "[LZO-DBG][DECOMP]"),
            num_blocks,
            tokens,
            literal_bytes,
            match_bytes,
            small_offsets,
            output_errors);
}

static double lzo_estimate_file_entropy_prefix(const char* path, size_t sample_bytes) {
    int fd = -1;
    unsigned char* buf = NULL;
    const size_t chunk = 64 * 1024;
    size_t freq[256] = {0};
    size_t remain = sample_bytes;
    size_t total = 0;
    double entropy = -1.0;

    if (!path || sample_bytes == 0) return -1.0;

    fd = open(path, O_RDONLY);
    if (fd < 0) goto cleanup;

    buf = (unsigned char*)malloc(chunk);
    if (!buf) goto cleanup;

    while (remain > 0) {
        size_t want = (remain < chunk) ? remain : chunk;
        ssize_t nr = read(fd, buf, want);
        size_t i;
        if (nr < 0) {
            if (errno == EINTR) continue;
            goto cleanup;
        }
        if (nr == 0) break;
        for (i = 0; i < (size_t)nr; i++) {
            freq[buf[i]]++;
        }
        total += (size_t)nr;
        remain -= (size_t)nr;
    }

    if (total > 0) {
        const double inv_log2 = 1.0 / log(2.0);
        size_t i;
        entropy = 0.0;
        for (i = 0; i < 256; i++) {
            if (freq[i] != 0) {
                double p = (double)freq[i] / (double)total;
                entropy -= p * (log(p) * inv_log2);
            }
        }
    }

cleanup:
    if (fd >= 0) close(fd);
    if (buf) free(buf);
    return entropy;
}

static int lzo_zero_buffer(cl_command_queue queue, cl_mem buf, size_t bytes) {
    if (!buf || bytes == 0) return 0;
#if defined(CL_VERSION_1_2)
    {
        static const cl_uint zero = 0;
        cl_int ferr = clEnqueueFillBuffer(queue, buf, &zero, sizeof(zero), 0, bytes, 0, NULL, NULL);
        if (ferr == CL_SUCCESS) {
            (void)clFinish(queue);
            return 0;
        }
    }
#endif
    {
        cl_int err;
        void* mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS || mapped == NULL) return -1;
        memset(mapped, 0, bytes);
        err = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
        (void)clFinish(queue);
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
        cl_mem_flags create_flags = flags | CL_MEM_ALLOC_HOST_PTR;
        *cached_buf = clCreateBuffer(ctx, create_flags, required_size, NULL, err_out);
        if (*err_out == CL_SUCCESS) {
            *cached_size = required_size;
        }
    } else {
        *err_out = CL_SUCCESS;
    }
    return *cached_buf;
}

typedef struct {
    cl_mem d_in;
    cl_mem d_out;
    cl_mem d_len;
    size_t in_cap;
    size_t out_cap;
    size_t len_cap;
    cl_event kernel_ev;
    int inflight;
    size_t in_size;
    size_t block_count;
    size_t block_base;
} lzo_pipeline_slot_t;

static int lzo_env_flag_default_on(const char* name) {
    const char* v = getenv(name);
    if (!v || !*v) return 1;
    return (strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0);
}

static size_t lzo_env_size_mb_to_bytes(const char* name, size_t default_mb) {
    unsigned mb = lzo_env_u32(name, (unsigned)default_mb);
    return (size_t)mb * 1024ULL * 1024ULL;
}

static void lzo_pipeline_slot_release(lzo_pipeline_slot_t* slot) {
    if (!slot) return;
    if (slot->kernel_ev) {
        clReleaseEvent(slot->kernel_ev);
        slot->kernel_ev = NULL;
    }
    if (slot->d_in) clReleaseMemObject(slot->d_in);
    if (slot->d_out) clReleaseMemObject(slot->d_out);
    if (slot->d_len) clReleaseMemObject(slot->d_len);
    memset(slot, 0, sizeof(*slot));
}

static int lzo_read_exact_fd(int fd, unsigned char* dst, size_t need, unsigned long* read_us_accum) {
    size_t done = 0;
    uint64_t t_read_start = core_now_ns();
    while (done < need) {
        ssize_t nr = read(fd, dst + done, need - done);
        if (nr < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (nr == 0) return -1;
        done += (size_t)nr;
    }
    if (read_us_accum) {
        *read_us_accum += (unsigned long)((core_now_ns() - t_read_start) / 1000ULL);
    }
    return 0;
}

static int lzo_pipeline_open_output_file(const char* output_path,
                                         size_t orig_size,
                                         size_t blk_size,
                                         size_t nblk,
                                         int alg_id,
                                         FILE** out_fp,
                                         long* lens_offset_out) {
    FILE* fout;
    uint16_t magic = 0x4C5A;
    uint32_t header[4];
    uint32_t zeros[4096] = {0};
    size_t remain = nblk;

    if (!output_path || !out_fp || !lens_offset_out) return -1;

    fout = fopen(output_path, "wb");
    if (!fout) return -1;

    header[0] = (uint32_t)orig_size;
    header[1] = (uint32_t)blk_size;
    header[2] = (uint32_t)nblk;
    header[3] = (uint32_t)alg_id;

    if (fwrite(&magic, sizeof(magic), 1, fout) != 1) {
        fclose(fout);
        return -1;
    }
    if (fwrite(header, sizeof(uint32_t), 4, fout) != 4) {
        fclose(fout);
        return -1;
    }

    *lens_offset_out = ftell(fout);
    if (*lens_offset_out < 0) {
        fclose(fout);
        return -1;
    }

    while (remain > 0) {
        size_t step = (remain < (size_t)4096) ? remain : (size_t)4096;
        if (fwrite(zeros, sizeof(uint32_t), step, fout) != step) {
            fclose(fout);
            return -1;
        }
        remain -= step;
    }

    *out_fp = fout;
    return 0;
}

static int lzo_pipeline_write_final_file_legacy(const char* output_path,
                                                size_t orig_size,
                                                size_t blk_size,
                                                size_t nblk,
                                                const cl_uint* lens,
                                                FILE* data_fp,
                                                int alg_id) {
    FILE* fout = fopen(output_path, "wb");
    if (!fout) return -1;

    {
        uint16_t magic = 0x4C5A;
        uint32_t header[4];
        header[0] = (uint32_t)orig_size;
        header[1] = (uint32_t)blk_size;
        header[2] = (uint32_t)nblk;
        header[3] = (uint32_t)alg_id;

        if (fwrite(&magic, sizeof(magic), 1, fout) != 1) {
            fclose(fout);
            return -1;
        }
        if (fwrite(header, sizeof(uint32_t), 4, fout) != 4) {
            fclose(fout);
            return -1;
        }
        if (fwrite(lens, sizeof(uint32_t), nblk, fout) != nblk) {
            fclose(fout);
            return -1;
        }
    }

    if (!data_fp) {
        fclose(fout);
        return -1;
    }

    rewind(data_fp);
    {
        unsigned char* copy_buf = (unsigned char*)malloc(1024 * 1024);
        if (!copy_buf) {
            fclose(fout);
            return -1;
        }
        for (;;) {
            size_t nr = fread(copy_buf, 1, 1024 * 1024, data_fp);
            if (nr == 0) break;
            if (fwrite(copy_buf, 1, nr, fout) != nr) {
                free(copy_buf);
                fclose(fout);
                return -1;
            }
        }
        free(copy_buf);
    }

    fclose(fout);
    return 0;
}

static int lzo_pipeline_finalize_output_file(FILE* out_fp,
                                             long lens_offset,
                                             size_t nblk,
                                             const cl_uint* lens) {
    int ret = 0;

    if (!out_fp || !lens) return -1;

    if (fseek(out_fp, lens_offset, SEEK_SET) != 0) {
        ret = -1;
    } else if (fwrite(lens, sizeof(uint32_t), nblk, out_fp) != nblk) {
        ret = -1;
    }

    if (fflush(out_fp) != 0) {
        ret = -1;
    }
    if (fclose(out_fp) != 0) {
        ret = -1;
    }
    return ret;
}

static int lzo_pipeline_drain_slot(
    cl_command_queue queue,
    lzo_pipeline_slot_t* slot,
    size_t worst_blk,
    int debug_sched,
    int discard_output,
    FILE* out_fp,
    cl_uint* lens_all,
    size_t nblk_total,
    size_t* comp_total_out,
    unsigned long* kernel_us_accum,
    unsigned long* download_us_accum,
    unsigned long* write_us_accum
) {
    cl_int err;
    if (!slot || !slot->inflight) return 0;
    if (!slot->kernel_ev) return -1;

    clWaitForEvents(1, &slot->kernel_ev);
    {
        cl_ulong q = 0, s = 0, st = 0, e = 0;
        if (clGetEventProfilingInfo(slot->kernel_ev, CL_PROFILING_COMMAND_QUEUED, sizeof(q), &q, NULL) == CL_SUCCESS &&
            clGetEventProfilingInfo(slot->kernel_ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(s), &s, NULL) == CL_SUCCESS &&
            clGetEventProfilingInfo(slot->kernel_ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) == CL_SUCCESS &&
            clGetEventProfilingInfo(slot->kernel_ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL) == CL_SUCCESS) {
            if (kernel_us_accum) {
                *kernel_us_accum += (unsigned long)((e - st) / 1000ULL);
            }
            if (debug_sched) {
                fprintf(stderr,
                        "[LZO-PROF-PIPE] queue_to_submit=%.3f us submit_to_start=%.3f us device_exec=%.3f us blocks=%zu\n",
                        (double)(s - q) / 1000.0,
                        (double)(st - s) / 1000.0,
                        (double)(e - st) / 1000.0,
                        slot->block_count);
            }
        }
    }

    {
        uint64_t t_down_start = core_now_ns();
        void* mapped_len = clEnqueueMapBuffer(queue, slot->d_len, CL_TRUE, CL_MAP_READ,
                                              0, slot->block_count * sizeof(cl_uint),
                                              0, NULL, NULL, &err);
        if (err != CL_SUCCESS || mapped_len == NULL) {
            return -1;
        }
        if (download_us_accum) {
            *download_us_accum += (unsigned long)((core_now_ns() - t_down_start) / 1000ULL);
        }

        if (!discard_output) {
            uint64_t t_out_down_start = core_now_ns();
            void* mapped_out = clEnqueueMapBuffer(queue, slot->d_out, CL_TRUE, CL_MAP_READ,
                                                  0, slot->block_count * worst_blk,
                                                  0, NULL, NULL, &err);
            if (err != CL_SUCCESS || mapped_out == NULL) {
                clEnqueueUnmapMemObject(queue, slot->d_len, mapped_len, 0, NULL, NULL);
                return -1;
            }
            if (download_us_accum) {
                *download_us_accum += (unsigned long)((core_now_ns() - t_out_down_start) / 1000ULL);
            }

            {
                uint64_t t_write_start = core_now_ns();
                cl_uint* lens_chunk = (cl_uint*)mapped_len;
                unsigned char* out_chunk = (unsigned char*)mapped_out;
                size_t i;
                for (i = 0; i < slot->block_count; i++) {
                    size_t global_idx = slot->block_base + i;
                    cl_uint clen = lens_chunk[i];
                    if (global_idx < nblk_total) {
                        lens_all[global_idx] = clen;
                    }
                    if (comp_total_out) *comp_total_out += (size_t)clen;
                    if (clen > 0 && out_fp) {
                        if (fwrite(out_chunk + i * worst_blk, 1, clen, out_fp) != clen) {
                            clEnqueueUnmapMemObject(queue, slot->d_out, mapped_out, 0, NULL, NULL);
                            clEnqueueUnmapMemObject(queue, slot->d_len, mapped_len, 0, NULL, NULL);
                            return -1;
                        }
                    }
                }
                if (write_us_accum) {
                    *write_us_accum += (unsigned long)((core_now_ns() - t_write_start) / 1000ULL);
                }
            }

            err = clEnqueueUnmapMemObject(queue, slot->d_out, mapped_out, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                clEnqueueUnmapMemObject(queue, slot->d_len, mapped_len, 0, NULL, NULL);
                return -1;
            }
        } else {
            cl_uint* lens_chunk = (cl_uint*)mapped_len;
            size_t i;
            for (i = 0; i < slot->block_count; i++) {
                size_t global_idx = slot->block_base + i;
                cl_uint clen = lens_chunk[i];
                if (global_idx < nblk_total) {
                    lens_all[global_idx] = clen;
                }
                if (comp_total_out) *comp_total_out += (size_t)clen;
            }
        }

        err = clEnqueueUnmapMemObject(queue, slot->d_len, mapped_len, 0, NULL, NULL);
        if (err != CL_SUCCESS) return -1;
    }

    if (slot->kernel_ev) {
        clReleaseEvent(slot->kernel_ev);
        slot->kernel_ev = NULL;
    }
    slot->inflight = 0;
    return 0;
}

static int lzo_compress_core_pipeline(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel kernel,
    const char* input_path,
    const char* output_path,
    const lzo_compress_params_t* params,
    lzo_gpu_workspace_t* ws,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out,
    size_t in_sz,
    size_t blk,
    size_t nblk,
    uint64_t t_total_start,
    int debug_sched
) {
    cl_int err = CL_SUCCESS;
    int ret = -1;
    int fd = -1;
    int use_standard_copy = params->standard_copy ? 1 : 0;
    int discard_output = (!output_path || strcmp(output_path, "/dev/null") == 0);
    int legacy_write = lzo_env_flag_enabled("LZO_PIPELINE_LEGACY_WRITE");

    size_t chunk_blocks_cfg = (size_t)lzo_env_u32("LZO_PIPELINE_CHUNK_BLOCKS", 512U);
    if (chunk_blocks_cfg < 64) chunk_blocks_cfg = 64;
    if (chunk_blocks_cfg > 4096) chunk_blocks_cfg = 4096;
    if (chunk_blocks_cfg > nblk) chunk_blocks_cfg = nblk;

    size_t worst_blk = core_lzo_worst(blk);
    size_t chunk_bytes_max = chunk_blocks_cfg * blk;

    unsigned long read_us = 0;
    unsigned long upload_us = 0;
    unsigned long kernel_exec_us = 0;
    unsigned long download_us = 0;
    unsigned long write_us = 0;
    unsigned long buffer_alloc_us = 0;
    size_t comp_total = 0;

    unsigned char* host_stage[2] = {NULL, NULL};
    lzo_pipeline_slot_t slots[2];
    cl_uint* lens_all = NULL;
    FILE* tmp_data_fp = NULL;
    FILE* out_fp = NULL;
    long out_lens_offset = 0;
    cl_mem d_dict = NULL;
    size_t d_dict_cap = 0;

    size_t local_size = 1;
    size_t target_items = chunk_blocks_cfg;
    size_t global_size = 1;
    uint32_t pool_size = 1;
    uint32_t epoch_base_start;

    memset(slots, 0, sizeof(slots));

    if (!discard_output) {
        if (legacy_write) {
            tmp_data_fp = tmpfile();
            if (!tmp_data_fp) {
                perror("tmpfile");
                goto cleanup;
            }
        } else {
            if (lzo_pipeline_open_output_file(output_path, in_sz, blk, nblk,
                                              params->alg_id, &out_fp,
                                              &out_lens_offset) != 0) {
                perror("pipeline output open");
                goto cleanup;
            }
        }
    }

    lens_all = (cl_uint*)malloc(nblk * sizeof(cl_uint));
    if (!lens_all) {
        perror("malloc lens_all");
        goto cleanup;
    }

    {
        int i;
        for (i = 0; i < 2; i++) {
            int rc = posix_memalign((void**)&host_stage[i], ALIGN_BYTES, chunk_bytes_max);
            if (rc != 0 || host_stage[i] == NULL) {
                host_stage[i] = (unsigned char*)malloc(chunk_bytes_max);
            }
            if (!host_stage[i]) {
                perror("malloc host_stage");
                goto cleanup;
            }
        }
    }

    fd = open(input_path, O_RDONLY);
    if (fd < 0) {
        perror("open input");
        goto cleanup;
    }

    {
        cl_uint cus = 0;
        cl_ulong global_mem = 0;
        cl_ulong max_alloc = 0;
        size_t dict_per_block = (1ULL << params->level) * sizeof(uint64_t);
        size_t occ_cap = 0;
        size_t mem_cap = 0;
        size_t safe_mem = 0;
        size_t safe_alloc = 0;

        if (params->local_size_param > 0) {
            local_size = (size_t)params->local_size_param;
        } else {
            local_size = 1;
        }
        if (local_size == 0) local_size = 1;

        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);

        {
            unsigned sched_wi_per_cu = lzo_env_u32("LZO_GPU_WI_PER_CU", 384U);
            if (sched_wi_per_cu < 32U) sched_wi_per_cu = 32U;
            if (sched_wi_per_cu > 1024U) sched_wi_per_cu = 1024U;
            occ_cap = (cus > 0) ? ((size_t)cus * (size_t)sched_wi_per_cu) : 4096U;
            if (occ_cap < 1024U) occ_cap = 1024U;
        }

        if (dict_per_block > 0) {
            size_t cap_by_global = SIZE_MAX;
            size_t cap_by_alloc = SIZE_MAX;

            if (global_mem > 0) {
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
        if (target_items == 0) target_items = 1;

        if (local_size > target_items) local_size = 1;
        global_size = ((target_items + local_size - 1) / local_size) * local_size;
        if (global_size == 0) {
            global_size = 1;
            local_size = 1;
        }
        pool_size = (uint32_t)global_size;

        {
            uint64_t t_ba_start = core_now_ns();
            d_dict = core_get_or_create_buffer(ctx, &ws->d_dict, &ws->dict_size,
                                               global_size * dict_per_block,
                                               CL_MEM_READ_WRITE, &err);
            if (err != CL_SUCCESS || !d_dict) {
                fprintf(stderr, "[PIPE] failed to allocate dictionary buffer: %d\n", err);
                goto cleanup;
            }
            d_dict_cap = ws->dict_size;
            buffer_alloc_us += (unsigned long)((core_now_ns() - t_ba_start) / 1000ULL);
        }
    }

    if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
    if (ws->comp_epoch_base > (uint32_t)(UINT32_MAX - (uint32_t)nblk - 2U)) {
        if (d_dict && d_dict_cap > 0) {
            (void)lzo_zero_buffer(queue, d_dict, d_dict_cap);
        }
        ws->comp_epoch_base = 1;
    }
    epoch_base_start = ws->comp_epoch_base;
    ws->comp_epoch_base += (uint32_t)nblk + 1U;

    if (debug_sched) {
        double blk_per_wi = (target_items > 0) ? ((double)chunk_blocks_cfg / (double)target_items) : 0.0;
        fprintf(stderr,
                "[LZO-SCHED-PIPE] level=%d total_blocks=%zu chunk_blocks=%zu bs=%zu lsz=%zu target_wi=%zu g=%zu blk_per_wi=%.2f\n",
                params->level, nblk, chunk_blocks_cfg, blk, local_size, target_items, global_size, blk_per_wi);
    }

    {
        size_t block_cursor = 0;
        size_t chunk_idx = 0;

        while (block_cursor < nblk) {
            size_t slot_idx = chunk_idx & 1U;
            lzo_pipeline_slot_t* slot = &slots[slot_idx];
            size_t chunk_blocks = chunk_blocks_cfg;
            size_t chunk_bytes;
            cl_event ev_upload = NULL;
            uint64_t t_ba_start;

            if (chunk_blocks > (nblk - block_cursor)) {
                chunk_blocks = nblk - block_cursor;
            }

            if (slot->inflight) {
                if (lzo_pipeline_drain_slot(queue, slot, worst_blk, debug_sched,
                                            discard_output, (legacy_write ? tmp_data_fp : out_fp), lens_all,
                                            nblk, &comp_total,
                                            &kernel_exec_us, &download_us, &write_us) != 0) {
                    fprintf(stderr, "[PIPE] failed draining inflight slot\n");
                    goto cleanup;
                }
            }

            chunk_bytes = chunk_blocks * blk;
            if (block_cursor + chunk_blocks == nblk) {
                chunk_bytes = in_sz - block_cursor * blk;
            }

            if (lzo_read_exact_fd(fd, host_stage[slot_idx], chunk_bytes, &read_us) != 0) {
                fprintf(stderr, "[PIPE] failed reading chunk at block %zu\n", block_cursor);
                goto cleanup;
            }

            t_ba_start = core_now_ns();
            slot->d_in = core_get_or_create_buffer(ctx, &slot->d_in, &slot->in_cap,
                                                   chunk_bytes, CL_MEM_READ_ONLY, &err);
            if (err != CL_SUCCESS || !slot->d_in) {
                fprintf(stderr, "[PIPE] failed allocating slot input buffer: %d\n", err);
                goto cleanup;
            }
            slot->d_out = core_get_or_create_buffer(ctx, &slot->d_out, &slot->out_cap,
                                                    chunk_blocks * worst_blk, CL_MEM_WRITE_ONLY, &err);
            if (err != CL_SUCCESS || !slot->d_out) {
                fprintf(stderr, "[PIPE] failed allocating slot output buffer: %d\n", err);
                goto cleanup;
            }
            slot->d_len = core_get_or_create_buffer(ctx, &slot->d_len, &slot->len_cap,
                                                    chunk_blocks * sizeof(cl_uint), CL_MEM_READ_WRITE, &err);
            if (err != CL_SUCCESS || !slot->d_len) {
                fprintf(stderr, "[PIPE] failed allocating slot len buffer: %d\n", err);
                goto cleanup;
            }
            buffer_alloc_us += (unsigned long)((core_now_ns() - t_ba_start) / 1000ULL);

            if (use_standard_copy) {
                uint64_t t_up_start = core_now_ns();
                err = clEnqueueWriteBuffer(queue, slot->d_in, CL_FALSE, 0, chunk_bytes,
                                           host_stage[slot_idx], 0, NULL, &ev_upload);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "[PIPE] clEnqueueWriteBuffer failed: %d\n", err);
                    goto cleanup;
                }
                upload_us += (unsigned long)((core_now_ns() - t_up_start) / 1000ULL);
            } else {
                uint64_t t_up_start = core_now_ns();
                void* mapped_in = clEnqueueMapBuffer(queue, slot->d_in, CL_TRUE, CL_MAP_WRITE,
                                                     0, chunk_bytes, 0, NULL, NULL, &err);
                if (err != CL_SUCCESS || mapped_in == NULL) {
                    fprintf(stderr, "[PIPE] clEnqueueMapBuffer failed: %d\n", err);
                    goto cleanup;
                }
                memcpy(mapped_in, host_stage[slot_idx], chunk_bytes);
                err = clEnqueueUnmapMemObject(queue, slot->d_in, mapped_in, 0, NULL, &ev_upload);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "[PIPE] clEnqueueUnmapMemObject failed: %d\n", err);
                    goto cleanup;
                }
                upload_us += (unsigned long)((core_now_ns() - t_up_start) / 1000ULL);
            }

            {
                cl_uint in_sz_cl = (cl_uint)chunk_bytes;
                cl_uint blk_cl = (cl_uint)blk;
                cl_uint worst_blk_cl = (cl_uint)worst_blk;
                cl_uint epoch_base = epoch_base_start + (cl_uint)block_cursor;
                cl_event ev_kernel = NULL;

                CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &slot->d_in));
                CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &slot->d_out));
                CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &slot->d_len));
                CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
                CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
                CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));
                CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
                CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));
                CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &epoch_base));

                err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                             &global_size, &local_size,
                                             (ev_upload ? 1 : 0), (ev_upload ? &ev_upload : NULL),
                                             &ev_kernel);
                if (err != CL_SUCCESS || ev_kernel == NULL) {
                    fprintf(stderr, "[PIPE] clEnqueueNDRangeKernel failed: %d\n", err);
                    if (ev_upload) clReleaseEvent(ev_upload);
                    goto cleanup;
                }

                if (ev_upload) clReleaseEvent(ev_upload);

                slot->kernel_ev = ev_kernel;
                slot->inflight = 1;
                slot->in_size = chunk_bytes;
                slot->block_count = chunk_blocks;
                slot->block_base = block_cursor;
            }

            block_cursor += chunk_blocks;
            chunk_idx++;
        }
    }

    {
        while (slots[0].inflight || slots[1].inflight) {
            int drain_idx;
            if (!slots[0].inflight) {
                drain_idx = 1;
            } else if (!slots[1].inflight) {
                drain_idx = 0;
            } else {
                /* Preserve global block order in output stream. */
                drain_idx = (slots[0].block_base <= slots[1].block_base) ? 0 : 1;
            }

            if (lzo_pipeline_drain_slot(queue, &slots[drain_idx], worst_blk, debug_sched,
                                        discard_output, (legacy_write ? tmp_data_fp : out_fp), lens_all,
                                        nblk, &comp_total,
                                        &kernel_exec_us, &download_us, &write_us) != 0) {
                fprintf(stderr, "[PIPE] failed draining tail slot\n");
                goto cleanup;
            }
        }
    }

    if (!discard_output) {
        uint64_t t_write_start = core_now_ns();
        if (legacy_write) {
            if (lzo_pipeline_write_final_file_legacy(output_path, in_sz, blk, nblk,
                                                     lens_all, tmp_data_fp,
                                                     params->alg_id) != 0) {
                fprintf(stderr, "[PIPE] failed writing final compressed file (legacy)\n");
                goto cleanup;
            }
        } else {
            if (lzo_pipeline_finalize_output_file(out_fp, out_lens_offset, nblk, lens_all) != 0) {
                fprintf(stderr, "[PIPE] failed finalizing compressed file\n");
                goto cleanup;
            }
            out_fp = NULL;
        }
        write_us += (unsigned long)((core_now_ns() - t_write_start) / 1000ULL);
    }

    clFinish(queue);

    {
        uint64_t t_total_end = core_now_ns();
        unsigned long total_us = (unsigned long)((t_total_end - t_total_start) / 1000ULL);
        if (time_us_out) *time_us_out = total_us;
        if (output_size_out) *output_size_out = comp_total;

        if (t_out) {
            t_out->in_size = (unsigned long long)in_sz;
            t_out->out_size = (unsigned long long)comp_total;
            t_out->algo_config = (unsigned long long)params->level;
            t_out->file_read_us = read_us;
            t_out->ocl_setup_us = g_ocl_init_us + g_kernel_load_us;
            t_out->buffer_alloc_us = buffer_alloc_us;
            t_out->data_upload_us = upload_us;
            t_out->kernel_exec_us = kernel_exec_us;
            t_out->download_total_us = download_us;
            t_out->file_write_us = write_us;
            t_out->blk_size_bytes = (unsigned long)blk;
            t_out->nblk = (unsigned long)nblk;
            t_out->global_size = (unsigned long)global_size;
            t_out->local_size = (unsigned long)local_size;
        }

    }

    ret = 0;

cleanup:
    if (fd >= 0) close(fd);
    if (tmp_data_fp) fclose(tmp_data_fp);
    if (out_fp) fclose(out_fp);
    if (lens_all) free(lens_all);
    if (host_stage[0]) free(host_stage[0]);
    if (host_stage[1]) free(host_stage[1]);
    lzo_pipeline_slot_release(&slots[0]);
    lzo_pipeline_slot_release(&slots[1]);
    if (ret != 0 && !discard_output && output_path && strcmp(output_path, "/dev/null") != 0) {
        unlink(output_path);
    }
    return ret;
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
    lzo_gpu_workspace_t* ws,
    unsigned long* time_us_out,
    size_t* output_size_out,
    timing_t* t_out
) {
    cl_int err;
    uint64_t t_total_start = core_now_ns();
    int debug_sched = lzo_env_flag_enabled("LZO_GPU_DEBUG_SCHED");
    int debug_counters = lzo_debug_counters_enabled();

    int use_standard_copy = params->standard_copy ? 1 : 0;

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

    {
        size_t blk_probe = 0, nblk_probe = 0;
        size_t blk_bytes_probe = (params->block_size > 0) ? params->block_size : 0;
        size_t pipeline_threshold = lzo_env_size_mb_to_bytes("LZO_PIPELINE_THRESHOLD_MB", 64);
        int pipeline_enabled = lzo_env_flag_enabled("LZO_PIPELINE_ENABLE");
        int entropy_gate_enabled = lzo_env_flag_enabled("LZO_PIPELINE_ENTROPY_ENABLE");
        size_t entropy_sample_kb = (size_t)lzo_env_u32("LZO_PIPELINE_ENTROPY_SAMPLE_KB", 256U);
        double entropy_max = lzo_env_f64("LZO_PIPELINE_ENTROPY_MAX", 7.60);
        int allow_pipeline;

        lzo_choose_blocking_adaptive(NULL, in_sz, device, blk_bytes_probe, 0,
                                     &blk_probe, &nblk_probe, params->debug);

        if (entropy_sample_kb < 16) entropy_sample_kb = 16;
        if (entropy_sample_kb > 4096) entropy_sample_kb = 4096;
        if (entropy_max < 0.0) entropy_max = 0.0;
        if (entropy_max > 8.0) entropy_max = 8.0;

        allow_pipeline = (pipeline_enabled && in_sz >= pipeline_threshold && nblk_probe >= 2 && !debug_counters);

        if (debug_counters && debug_sched) {
            fprintf(stderr, "[LZO-PIPE-GATE] disabled because LZO_GPU_DEBUG_COUNTERS=1\n");
        }

        if (allow_pipeline && entropy_gate_enabled) {
            double entropy = lzo_estimate_file_entropy_prefix(input_path, entropy_sample_kb * 1024ULL);
            if (entropy >= 0.0) {
                if (debug_sched) {
                    fprintf(stderr,
                            "[LZO-PIPE-GATE] entropy=%.3f max=%.3f sample_kb=%zu decision=%s\n",
                            entropy, entropy_max, entropy_sample_kb,
                            (entropy <= entropy_max) ? "pipeline" : "fallback");
                }
                if (entropy > entropy_max) {
                    allow_pipeline = 0;
                }
            } else if (debug_sched) {
                fprintf(stderr,
                        "[LZO-PIPE-GATE] entropy sampling failed, keep size-threshold decision\n");
            }
        }

        if (allow_pipeline) {
            return lzo_compress_core_pipeline(
                ctx,
                queue,
                device,
                kernel,
                input_path,
                output_path,
                params,
                ws,
                time_us_out,
                output_size_out,
                t_out,
                in_sz,
                blk_probe,
                nblk_probe,
                t_total_start,
                debug_sched
            );
        }
    }

    /* 2. 准备输入缓冲区 (如果是 Daemon 模式,此处通常会命中预分配好的常驻缓冲) */
    uint64_t t_buf_in_start = core_now_ns();
    cl_mem d_in = core_get_or_create_buffer(ctx, &ws->d_in, &ws->in_size, in_sz, CL_MEM_READ_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create input buffer: %d\n", err);
        return -1;
    }

    void* mapped_in = NULL;
    if (!use_standard_copy) {
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

    if (!use_standard_copy) {
        /* 直接读入映射好的显存空间 (Zero-copy 核心) */
        if (lzo_read_file_to_buf(input_path, mapped_in, in_sz, &read_us) != 0) {
            perror("lzo_read_file_to_buf");
            clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL);
            return -1;
        }
        /* upload_us will be measured after unmap */
    } else {
        int rc_mem = posix_memalign(&host_in, ALIGN_BYTES, in_sz);
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
            free(host_in);
            return -1;
        }
        /* upload after blocking calc */
    }

    /* 3. Blocking calculation */
    uint64_t t_blocking_start = core_now_ns();
    size_t blk = 0, nblk = 0;
    const unsigned char* entropy_ptr = NULL;
    if (use_standard_copy)
        entropy_ptr = (const unsigned char*)host_in;
    else
        entropy_ptr = (const unsigned char*)mapped_in;

    size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
    lzo_choose_blocking_adaptive(entropy_ptr, in_sz, device, blk_bytes, 0, &blk, &nblk, params->debug);


    uint64_t t_blocking_end = core_now_ns();
    unsigned long blocking_us = (t_blocking_end - t_blocking_start) / 1000;

    if (!use_standard_copy) {
        uint64_t t_unmap_start = core_now_ns();
        CHECK(clEnqueueUnmapMemObject(queue, d_in, mapped_in, 0, NULL, NULL));
        uint64_t t_unmap_end = core_now_ns();
        upload_us = (t_unmap_end - t_unmap_start) / 1000;
    } else {
        uint64_t t_upload_start2 = core_now_ns();
        err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, in_sz, host_in, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] clEnqueueWriteBuffer failed: %d\n", err);
            free(host_in);
            return -1;
        }
        free(host_in);
        host_in = NULL;
        uint64_t t_upload_end2 = core_now_ns();
        upload_us = (t_upload_end2 - t_upload_start2) / 1000;
    }

    size_t worst_blk = core_lzo_worst(blk);
    size_t out_cap = nblk * worst_blk;

    size_t out_needed = out_cap;
    size_t len_needed = nblk * sizeof(cl_uint);

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_out, &ws->out_size, out_needed, CL_MEM_WRITE_ONLY, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CORE] failed to create output buffer: %d\n", err);
        return -1;
    }
    uint64_t t_buf_out_end = core_now_ns();
    unsigned long buffer_out_us = (t_buf_out_end - t_buf_out_start) / 1000;

    /* Buffer Alloc (len) timing removed per request; keep allocation but don't time it */
    cl_mem d_len = core_get_or_create_buffer(ctx, &ws->d_len, &ws->len_size, len_needed, CL_MEM_READ_WRITE, &err);
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
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_len));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &in_sz_cl));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_cl));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &worst_blk_cl));

    /* Dictionary pool follows active work-items for one-work-item-per-block scheduling. */
    size_t local_size = 1;
    unsigned sched_wi_per_cu = 256U;
    if (params->local_size_param > 0) {
        local_size = (size_t)params->local_size_param;
        if (params->debug) fprintf(stderr, "[CORE] using local_size=%zu\n", local_size);
    }

    size_t target_items = nblk;
    if (target_items == 0) target_items = 1;
    {
        cl_uint cus = 0;
        cl_ulong global_mem = 0;
        cl_ulong max_alloc = 0;
        size_t dict_per_block = (1ULL << params->level) * sizeof(uint64_t);
        size_t occ_cap = 0;
        size_t mem_cap = 0;
        size_t safe_mem = 0;
        size_t safe_alloc = 0;

        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
        (void)clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
        if (local_size == 0) local_size = 1;

        sched_wi_per_cu = lzo_env_u32("LZO_GPU_WI_PER_CU", 384U);
        if (sched_wi_per_cu < 32U) sched_wi_per_cu = 32U;
        if (sched_wi_per_cu > 1024U) sched_wi_per_cu = 1024U;
        occ_cap = (cus > 0) ? ((size_t)cus * (size_t)sched_wi_per_cu) : 4096U;
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
    if (local_size > target_items) local_size = 1;
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
                "[LZO-SCHED] alg=%d level=%d blocks=%zu bs=%zu lsz=%zu target_wi=%zu g=%zu wi_per_cu=%u blk_per_wi=%.2f\n",
                params->alg_id, params->level, nblk, blk, local_size, target_items, global_size, sched_wi_per_cu, blk_per_wi);
    }

    if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
    if (ws->comp_epoch_base > (uint32_t)(UINT32_MAX - (uint32_t)nblk - 2U)) {
        if (ws->d_dict && ws->dict_size > 0) {
            (void)lzo_zero_buffer(queue, ws->d_dict, ws->dict_size);
        }
        ws->comp_epoch_base = 1;
    }
    uint32_t epoch_base = ws->comp_epoch_base;
    ws->comp_epoch_base += (uint32_t)nblk + 1U;

    {
        /* Packed dictionary stores {epoch, entry} in one 64-bit slot per hash entry. */
        size_t dict_per_block = (1ULL << params->level) * sizeof(uint64_t);
        size_t total_dict_size = (size_t)pool_size * dict_per_block;
        size_t prev_dict_size = ws->dict_size;

        cl_mem d_dict = core_get_or_create_buffer(ctx, &ws->d_dict, &ws->dict_size, total_dict_size, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[CORE] failed to create dictionary buffer: %d\n", err);
            return -1;
        }
        if (ws->dict_size != prev_dict_size) {
            (void)lzo_zero_buffer(queue, d_dict, ws->dict_size);
        }
        CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_dict));
        CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &pool_size));
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
    void* mapped_len = clEnqueueMapBuffer(queue, d_len, CL_TRUE, CL_MAP_READ, 0, nblk * sizeof(cl_uint), 0, NULL, NULL, &err);
    CHECK(err);
    memcpy(len_arr, mapped_len, nblk * sizeof(cl_uint));
    CHECK(clEnqueueUnmapMemObject(queue, d_len, mapped_len, 0, NULL, NULL));

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

    /* Map the whole output buffer */
    uint64_t t_down_start = core_now_ns();
    void* mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, out_needed, 0, NULL, NULL, &err); CHECK(err);
    host_comp = mapped_out;
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
        CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));
        if (dbg_comp_buf) clReleaseMemObject(dbg_comp_buf);
        free(len_arr);
        return -1;
    }

    CHECK(clEnqueueUnmapMemObject(queue, d_out, host_comp, 0, NULL, NULL));

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
    cl_mem d_off = core_get_or_create_buffer(ctx, &ws->d_off, &ws->off_size, (nblk + 1) * sizeof(cl_uint), CL_MEM_READ_ONLY, &err); CHECK(err);

    /* Build offsets directly into device-backed buffer to avoid host off_arr allocation/upload. */
    {
        void* mapped_off = clEnqueueMapBuffer(queue, d_off, CL_TRUE, CL_MAP_WRITE,
                                              0, (nblk + 1) * sizeof(cl_uint),
                                              0, NULL, NULL, &err);
        if (err != CL_SUCCESS || !mapped_off) {
            fprintf(stderr, "[DECOMP] clEnqueueMapBuffer d_off failed: %d\n", err);
            goto cleanup;
        }
        {
            cl_uint* off_dev = (cl_uint*)mapped_off;
            off_dev[0] = 0;
            for (uint32_t i = 0; i < nblk; ++i)
                off_dev[i + 1] = off_dev[i] + len_arr[i];
        }
        err = clEnqueueUnmapMemObject(queue, d_off, mapped_off, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[DECOMP] clEnqueueUnmapMemObject d_off failed: %d\n", err);
            goto cleanup;
        }
    }
    free(len_arr);
    len_arr = NULL;

    uint64_t t_buf_out_start = core_now_ns();
    cl_mem d_out = core_get_or_create_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_size, orig_sz, CL_MEM_WRITE_ONLY, &err); CHECK(err);
    cl_mem d_out_lens = core_get_or_create_buffer(ctx, &ws->d_out_lens, &ws->lens_size, nblk * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err); CHECK(err);
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
    uint64_t t_file_read_start = core_now_ns();
    if (fread(comp_host, 1, comp_sz, f_in) != comp_sz) {
        perror("fread compressed data");
        goto cleanup;
    }
    fclose(f_in);
    f_in = NULL;
    uint64_t t_file_read_end = core_now_ns();
    unsigned long total_file_read_us = (t_file_read_end - t_file_read_start) / 1000;
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

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_comp));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_off));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_out));
    CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_out_lens));
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &blk_sz));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &orig_sz));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &nblk));

    {
        cl_uint kernel_num_args = 0;
        int kernel_has_dbg = 0;
        int dbg_dec_enabled = lzo_debug_counters_enabled();
        if (clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(kernel_num_args), &kernel_num_args, NULL) == CL_SUCCESS) {
            kernel_has_dbg = (kernel_num_args >= 9U);
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
            cl_mem dbg_arg = dbg_dec_enabled ? dbg_dec_buf : d_out_lens;
            cl_uint dbg_flag = dbg_dec_enabled ? 1U : 0U;
            CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &dbg_arg));
            CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uint), &dbg_flag));
        }
    }

    size_t local_size = 1;
    if (local_size_param > 0) {
        local_size = (size_t)local_size_param;
    } else {
        local_size = 1;
    }
    if (local_size == 0) local_size = 1;
    if (local_size > (size_t)nblk) local_size = 1;
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

    uint64_t t_download_start = core_now_ns();
    mapped_out = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0, orig_sz, 0, NULL, NULL, &err);
    CHECK(err);
    uint64_t t_download_end = core_now_ns();
    unsigned long download_us = (t_download_end - t_download_start) / 1000;

    uint64_t t_write_start2 = core_now_ns();
    if (output_path) {
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
    }
    CHECK(clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL));
    mapped_out = NULL;
    uint64_t t_write_end2 = core_now_ns();
    unsigned long write_us = (t_write_end2 - t_write_start2) / 1000;

    size_t cache_threshold_mb = DEFAULT_DECOMP_CACHE_MB;
    const char* cache_env = getenv("LZO_DECOMP_CACHE_MB");
    if (cache_env)
        cache_threshold_mb = (size_t)atoi(cache_env);
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
    if (mapped_out) clEnqueueUnmapMemObject(queue, d_out, mapped_out, 0, NULL, NULL);
    if (dbg_dec_buf) clReleaseMemObject(dbg_dec_buf);

    return ret;
}
