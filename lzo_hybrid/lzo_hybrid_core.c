/*
 * lzo_hybrid_core.c — Hybrid GPU+CPU LZO compression/decompression core.
 *
 * Partitions blocks between GPU (OpenCL kernel) and CPU (liblzo2 pthreads).
 * Both run concurrently. Results are merged into the standard .lzo container.
 *
 * Block format is identical to lzo_gpu: MAGIC 0x4C5A, header, per-block lengths, data.
 * GPU and CPU produce format-compatible compressed blocks.
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "lzo_hybrid_core.h"
#include "../lzo_gpu/lzo_gpu_utils.h"
#include "../lzo_gpu/lzo_defaults.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* CPU LZO library */
#include <lzo/lzoconf.h>
#include <lzo/lzo1x.h>
#include <lzo/lzo1y.h>

#define MAGIC 0x4C5A
#define CHECK_CL(err) do { if ((err) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__, __LINE__); \
    return -1; \
}} while(0)

static int g_lzo_initialized = 0;

static inline uint64_t hybrid_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline size_t lzo_worst_size(size_t n) {
    return n + n / 16 + 64 + 3;
}

static long get_online_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
#endif
}

static size_t block_input_size(size_t total_size, size_t block_size, size_t block_idx) {
    size_t start = block_idx * block_size;
    if (start >= total_size) return 0;
    {
        size_t rem = total_size - start;
        return rem < block_size ? rem : block_size;
    }
}

static size_t sampled_block_index(size_t sample_pos, size_t sample_count, size_t num_blocks) {
    if (num_blocks == 0) return 0;
    if (sample_count <= 1 || num_blocks <= 1) return 0;
    return (sample_pos * (num_blocks - 1)) / (sample_count - 1);
}

static void hybrid_choose_blocking(const unsigned char* data,
                                   size_t in_sz,
                                   cl_device_id dev,
                                   cl_kernel gpu_kernel,
                                   size_t blk_bytes,
                                   int debug,
                                   size_t* blk_out,
                                   size_t* nblk_out) {
    if (dev && gpu_kernel) {
        lzo_choose_blocking_adaptive(data, in_sz, dev, blk_bytes, 0, blk_out, nblk_out, debug);
        return;
    }

    if (blk_bytes > 0) {
        size_t blk = blk_bytes;
        if (blk < LZO_MIN_BLOCK_BYTES_DEFAULT) blk = LZO_MIN_BLOCK_BYTES_DEFAULT;
        if (blk > LZO_MAX_BLOCK_BYTES_DEFAULT) blk = LZO_MAX_BLOCK_BYTES_DEFAULT;
        *blk_out = blk;
        *nblk_out = (in_sz + blk - 1) / blk;
    } else {
        size_t blk = (size_t)LZO_DEFAULT_BLOCK_KB * 1024U;
        if (blk == 0) blk = 16 * 1024;
        *blk_out = blk;
        *nblk_out = (in_sz + blk - 1) / blk;
    }
}

/* ---- Device-aware adaptive scheduler infrastructure ---- */

static void ensure_lzo_init(void);
static cl_mem grow_buffer(cl_context ctx, cl_mem* buf, size_t* cap,
                          size_t needed, cl_mem_flags flags, cl_int* err);
static int zero_buffer(cl_command_queue q, cl_mem buf, size_t sz);

typedef struct {
    double cpu_throughput;    /* single-thread CPU throughput (bytes/s) */
    double gpu_throughput;    /* GPU throughput (bytes/s) */
    double gpu_overhead_s;    /* GPU fixed overhead (seconds) */
    double cpu_energy_per_byte; /* CPU energy cost (J/byte), 0 if unavailable */
    double gpu_energy_per_byte; /* GPU energy cost (J/byte), 0 if unavailable */
    int    is_unified_memory;
    int    valid;
} lzo_device_profile_t;

static lzo_device_profile_t g_lzo_dev_profile = {0};

static double lzo_read_sysfs_double(const char* path) {
    FILE* f = fopen(path, "r");
    double val = -1.0;
    if (f) { if (fscanf(f, "%lf", &val) != 1) val = -1.0; fclose(f); }
    return val;
}

static uint64_t lzo_read_rapl_energy_uj(const char* domain_path) {
    FILE* f = fopen(domain_path, "r");
    uint64_t val = 0;
    if (f) { if (fscanf(f, "%lu", &val) != 1) val = 0; fclose(f); }
    return val;
}

static double lzo_read_cpu_availability(void) {
    static uint64_t prev_total = 0, prev_idle = 0;
    FILE* f = fopen("/proc/stat", "r");
    char buf[256];
    uint64_t user, nice, sys, idle, iowait, irq, softirq, steal;
    uint64_t total, diff_total, diff_idle;
    double avail;
    if (!f) return 1.0;
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 1.0; }
    fclose(f);
    if (sscanf(buf, "cpu %lu %lu %lu %lu %lu %lu %lu %lu",
               &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal) < 4)
        return 1.0;
    total = user + nice + sys + idle + iowait + irq + softirq + steal;
    diff_total = total - prev_total;
    diff_idle = idle - prev_idle;
    prev_total = total;
    prev_idle = idle;
    if (diff_total == 0) return 1.0;
    avail = (double)diff_idle / (double)diff_total;
    if (avail < 0.05) avail = 0.05;
    if (avail > 1.0) avail = 1.0;
    return avail;
}

static double lzo_read_gpu_availability(void) {
    double busy = lzo_read_sysfs_double("/sys/class/drm/card0/device/gpu_busy_percent");
    if (busy < 0.0) busy = lzo_read_sysfs_double("/sys/class/drm/card1/device/gpu_busy_percent");
    if (busy < 0.0) return 1.0;
    if (busy > 100.0) busy = 100.0;
    return 1.0 - busy / 100.0;
}

static void lzo_calibrate_device_profile(cl_context ctx, cl_command_queue queue,
                                         cl_device_id device, cl_kernel gpu_kernel,
                                         const hybrid_params_t* params,
                                         hybrid_workspace_t* ws) {
    if (g_lzo_dev_profile.valid) return;

    {
        cl_bool unified = CL_FALSE;
        if (device) {
            clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
        }
        g_lzo_dev_profile.is_unified_memory = (unified == CL_TRUE) ? 1 : 0;
    }

    /* CPU calibration: 2MB buffer, 3 reps LZO compress */
    {
        size_t cal_size = 2 * 1024 * 1024;
        unsigned char* cal_buf = (unsigned char*)malloc(cal_size);
        unsigned char* cal_out = NULL;
        void* wrkmem = NULL;
        size_t wrkmem_sz;
        uint64_t t0, t1;

        if (!cal_buf) {
            g_lzo_dev_profile.cpu_throughput = 300e6;
            g_lzo_dev_profile.gpu_throughput = 1500e6;
            g_lzo_dev_profile.gpu_overhead_s = 0.001;
            g_lzo_dev_profile.valid = 1;
            return;
        }
        memset(cal_buf, 0xAB, cal_size);
        for (size_t i = 0; i < cal_size; i += 97) cal_buf[i] = (unsigned char)(i & 0xFF);

        cal_out = (unsigned char*)malloc(lzo_worst_size(cal_size));
        wrkmem_sz = (params->alg_id == 1) ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
        wrkmem = malloc(wrkmem_sz);
        if (!cal_out || !wrkmem) {
            free(cal_buf); free(cal_out); free(wrkmem);
            g_lzo_dev_profile.cpu_throughput = 300e6;
            g_lzo_dev_profile.gpu_throughput = 1500e6;
            g_lzo_dev_profile.gpu_overhead_s = 0.001;
            g_lzo_dev_profile.valid = 1;
            return;
        }
        memset(wrkmem, 0, wrkmem_sz);
        ensure_lzo_init();

        {
            uint64_t e0 = lzo_read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:0/energy_uj");
            t0 = hybrid_now_ns();
            for (int rep = 0; rep < 3; rep++) {
                lzo_uint out_len = (lzo_uint)lzo_worst_size(cal_size);
                if (params->alg_id == 1)
                    lzo1y_1_compress(cal_buf, (lzo_uint)cal_size, cal_out, &out_len, wrkmem);
                else
                    lzo1x_1_compress(cal_buf, (lzo_uint)cal_size, cal_out, &out_len, wrkmem);
            }
            t1 = hybrid_now_ns();
            g_lzo_dev_profile.cpu_throughput = (3.0 * (double)cal_size) / ((double)(t1 - t0) * 1e-9);

            {
                uint64_t e1 = lzo_read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:0/energy_uj");
                if (e0 > 0 && e1 > e0) {
                    double energy_j = (double)(e1 - e0) * 1e-6;
                    g_lzo_dev_profile.cpu_energy_per_byte = energy_j / (3.0 * (double)cal_size);
                }
            }
        }

        free(cal_out);
        free(wrkmem);

        /* GPU calibration: compress the same 2MB through the kernel */
        if (gpu_kernel && ws && ctx && queue && device) {
            size_t blk = (params->block_size > 0) ? params->block_size : 65536;
            size_t cal_nblk = (cal_size + blk - 1) / blk;
            size_t worst_blk = lzo_worst_size(blk);
            cl_int err;

            grow_buffer(ctx, &ws->d_in, &ws->in_cap, cal_size, CL_MEM_READ_ONLY, &err);
            if (err == CL_SUCCESS) {
                err = clEnqueueWriteBuffer(queue, ws->d_in, CL_TRUE, 0, cal_size, cal_buf, 0, NULL, NULL);
            }
            if (err == CL_SUCCESS) {
                size_t out_needed = cal_nblk * worst_blk;
                grow_buffer(ctx, &ws->d_out, &ws->out_cap, out_needed, CL_MEM_WRITE_ONLY, &err);
            }
            if (err == CL_SUCCESS) {
                size_t len_needed = cal_nblk * sizeof(cl_uint);
                grow_buffer(ctx, &ws->d_len, &ws->len_cap, len_needed, CL_MEM_READ_WRITE, &err);
            }
            if (err == CL_SUCCESS) {
                int comp_level = (params->comp_level > 0) ? params->comp_level : 14;
                int is_999 = (comp_level == 999);
                size_t dict_per_block;
                if (is_999) {
                    dict_per_block = 458752ULL;  /* SWD_POOL_STRIDE */
                } else {
                    dict_per_block = (1ULL << comp_level) * sizeof(uint32_t);
                }
                size_t total_dict = cal_nblk * dict_per_block;
                grow_buffer(ctx, &ws->d_dict, &ws->dict_cap, total_dict, CL_MEM_READ_WRITE, &err);
                if (err == CL_SUCCESS) zero_buffer(queue, ws->d_dict, ws->dict_cap);
            }
            if (err == CL_SUCCESS) {
                int comp_level = (params->comp_level > 0) ? params->comp_level : 14;
                int is_999 = (comp_level == 999);
                cl_uint cal_in_sz = (cl_uint)cal_size;
                cl_uint blk_cl = (cl_uint)blk;
                cl_uint worst_blk_cl = (cl_uint)worst_blk;
                cl_uint pool_size_cl = (cl_uint)cal_nblk;

                clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &ws->d_in);
                clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &ws->d_out);
                clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &ws->d_len);
                clSetKernelArg(gpu_kernel, 3, sizeof(cl_uint), &cal_in_sz);
                clSetKernelArg(gpu_kernel, 4, sizeof(cl_uint), &blk_cl);
                clSetKernelArg(gpu_kernel, 5, sizeof(cl_uint), &worst_blk_cl);
                clSetKernelArg(gpu_kernel, 6, sizeof(cl_mem), &ws->d_dict);

                if (is_999) {
                    /* 999 kernel: 10 args — swd_pool, swd_pool_count, try_lazy, max_chain */
                    cl_uint swd_pool_count_cl = pool_size_cl;
                    cl_uint try_lazy_cl = 2U;
                    cl_uint max_chain_cl = 4096U;
                    clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &swd_pool_count_cl);
                    clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &try_lazy_cl);
                    clSetKernelArg(gpu_kernel, 9, sizeof(cl_uint), &max_chain_cl);
                } else {
                    uint32_t epoch_base = 1;
                    clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &pool_size_cl);
                    clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &epoch_base);

                    cl_uint krn_num_args = 0;
                    clGetKernelInfo(gpu_kernel, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL);
                    if (krn_num_args >= 11U) {
                        cl_uint dbg_flag = 0U;
                        clSetKernelArg(gpu_kernel, 9, sizeof(cl_mem), &ws->d_len);
                        clSetKernelArg(gpu_kernel, 10, sizeof(cl_uint), &dbg_flag);
                    }
                }

                {
                    size_t local_size = (params->local_size > 0) ? (size_t)params->local_size : 1;
                    size_t global_size = cal_nblk;
                    if (local_size > global_size) local_size = 1;
                    global_size = ((global_size + local_size - 1) / local_size) * local_size;
                    if (global_size == 0) global_size = 1;

                    {
                        uint64_t ge0 = lzo_read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:1/energy_uj");
                        t0 = hybrid_now_ns();
                        err = clEnqueueNDRangeKernel(queue, gpu_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
                        if (err == CL_SUCCESS) {
                            clFinish(queue);
                            t1 = hybrid_now_ns();
                            g_lzo_dev_profile.gpu_throughput = (double)cal_size / ((double)(t1 - t0) * 1e-9);
                            g_lzo_dev_profile.gpu_overhead_s = 0.001;

                            {
                                uint64_t ge1 = lzo_read_rapl_energy_uj("/sys/class/powercap/intel-rapl:0:1/energy_uj");
                                if (ge0 > 0 && ge1 > ge0) {
                                    double energy_j = (double)(ge1 - ge0) * 1e-6;
                                    g_lzo_dev_profile.gpu_energy_per_byte = energy_j / (double)cal_size;
                                }
                            }
                        }
                    }
                }

                if (!is_999) {
                    ws->comp_epoch_base = (uint32_t)cal_nblk + 2;
                }
            }

            if (g_lzo_dev_profile.gpu_throughput <= 0.0) {
                g_lzo_dev_profile.gpu_throughput = 1500e6;
                g_lzo_dev_profile.gpu_overhead_s = 0.001;
            }
        } else {
            g_lzo_dev_profile.gpu_throughput = 1500e6;
            g_lzo_dev_profile.gpu_overhead_s = 0.001;
        }

        free(cal_buf);
    }

    if (g_lzo_dev_profile.cpu_throughput <= 0.0) g_lzo_dev_profile.cpu_throughput = 300e6;
    if (g_lzo_dev_profile.gpu_throughput <= 0.0) g_lzo_dev_profile.gpu_throughput = 1500e6;
    g_lzo_dev_profile.valid = 1;
}

static void ensure_lzo_init(void) {
    if (!g_lzo_initialized) {
        if (lzo_init() != LZO_E_OK) {
            fprintf(stderr, "lzo_init() failed\n");
        }
        g_lzo_initialized = 1;
    }
}

/* ---- workspace management ---- */

void hybrid_workspace_init(hybrid_workspace_t* ws) {
    memset(ws, 0, sizeof(*ws));
    ws->comp_epoch_base = 1;
}

void hybrid_workspace_free(hybrid_workspace_t* ws) {
    if (ws->d_in) clReleaseMemObject(ws->d_in);
    if (ws->d_out) clReleaseMemObject(ws->d_out);
    if (ws->d_len) clReleaseMemObject(ws->d_len);
    if (ws->d_packed_out) clReleaseMemObject(ws->d_packed_out);
    if (ws->d_packed_off) clReleaseMemObject(ws->d_packed_off);
    if (ws->d_dict) clReleaseMemObject(ws->d_dict);
    if (ws->d_comp) clReleaseMemObject(ws->d_comp);
    if (ws->d_off) clReleaseMemObject(ws->d_off);
    if (ws->d_comp_lens) clReleaseMemObject(ws->d_comp_lens);
    if (ws->d_decomp_out) clReleaseMemObject(ws->d_decomp_out);
    if (ws->d_out_lens) clReleaseMemObject(ws->d_out_lens);
    memset(ws, 0, sizeof(*ws));
}

static cl_mem grow_buffer(cl_context ctx, cl_mem* buf, size_t* cap,
                          size_t needed, cl_mem_flags flags, cl_int* err) {
    if (*buf && *cap >= needed) {
        *err = CL_SUCCESS;
        return *buf;
    }
    if (*buf) clReleaseMemObject(*buf);
    *buf = clCreateBuffer(ctx, flags, needed, NULL, err);
    *cap = (*err == CL_SUCCESS && *buf) ? needed : 0;
    return *buf;
}

static int zero_buffer(cl_command_queue q, cl_mem buf, size_t sz) {
    cl_uchar zero = 0;
    cl_int err = clEnqueueFillBuffer(q, buf, &zero, 1, 0, sz, 0, NULL, NULL);
    if (err != CL_SUCCESS) return -1;
    clFinish(q);
    return 0;
}

static int hybrid_env_flag_value(const char* name, int* is_set) {
    const char* env = getenv(name);
    if (is_set) *is_set = 0;
    if (!env || !*env) return 0;
    if (is_set) *is_set = 1;
    if (strcmp(env, "1") == 0 || strcasecmp(env, "true") == 0 || strcasecmp(env, "yes") == 0 || strcasecmp(env, "on") == 0) return 1;
    if (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "no") == 0 || strcasecmp(env, "off") == 0) return 0;
    return atoi(env) != 0;
}

static unsigned hybrid_env_unsigned_value(const char* name, unsigned defv) {
    const char* env = getenv(name);
    char* end = NULL;
    unsigned long parsed;
    if (!env || !*env) return defv;
    parsed = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || parsed > UINT32_MAX) return defv;
    return (unsigned)parsed;
}

static int hybrid_should_use_device_compaction(size_t packed_bytes,
                                               size_t sparse_bytes,
                                               size_t num_blocks,
                                               cl_kernel pack_kernel) {
    int force_set = 0;
    int force_value = hybrid_env_flag_value("LZO_HYBRID_FORCE_COMPACTION", &force_set);
    unsigned min_blocks;
    unsigned min_gain_pct;

    if (!force_set) {
        force_value = hybrid_env_flag_value("LZO_GPU_FORCE_COMPACTION", &force_set);
    }
    if (!pack_kernel) return 0;
    if (force_set) return force_value;
    if (packed_bytes == 0 || sparse_bytes == 0 || packed_bytes >= sparse_bytes) return 0;

    min_blocks = hybrid_env_unsigned_value(
        "LZO_HYBRID_COMPACTION_MIN_BLOCKS",
        hybrid_env_unsigned_value("LZO_GPU_COMPACTION_MIN_BLOCKS", 8U)
    );
    min_gain_pct = hybrid_env_unsigned_value(
        "LZO_HYBRID_COMPACTION_MIN_GAIN_PCT",
        hybrid_env_unsigned_value("LZO_GPU_COMPACTION_MIN_GAIN_PCT", 5U)
    );
    if (num_blocks < (size_t)min_blocks) return 0;

    return (sparse_bytes - packed_bytes) * 100U >= sparse_bytes * (size_t)min_gain_pct;
}

/* ---- CPU worker thread data ---- */

typedef struct {
    const unsigned char* input;
    unsigned char* output;
    uint32_t* lengths;
    size_t block_size;
    size_t in_size;
    size_t nblk;
    size_t worst_blk;
    const size_t* block_indices;
    size_t num_assigned;
    int alg_id;
    int rc;
} cpu_compress_job_t;

typedef struct {
    /* shared across all workers */
    cpu_compress_job_t* job;
    _Atomic size_t next_idx;
    _Atomic uint64_t max_worker_us;
} cpu_compress_pool_t;

static void* cpu_compress_worker(void* arg) {
    cpu_compress_pool_t* pool = (cpu_compress_pool_t*)arg;
    cpu_compress_job_t* job = pool->job;
    uint64_t t0 = hybrid_now_ns();

    /* Per-thread work memory for LZO */
    void* wrkmem = NULL;
    size_t wrkmem_sz = (job->alg_id == 1) ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
    if (lzo_aligned_alloc_portable(&wrkmem, 64, wrkmem_sz) != 0) {
        wrkmem = malloc(wrkmem_sz);
    }
    if (!wrkmem) {
        job->rc = -1;
        return NULL;
    }
    memset(wrkmem, 0, wrkmem_sz);

    for (;;) {
        size_t wi = atomic_fetch_add(&pool->next_idx, 1);
        if (wi >= job->num_assigned) break;

        size_t blk_idx = job->block_indices[wi];
        size_t offset = blk_idx * job->block_size;
        size_t this_blk = job->block_size;
        if (offset + this_blk > job->in_size) this_blk = job->in_size - offset;

        const unsigned char* src = job->input + offset;
        unsigned char* dst = job->output + blk_idx * job->worst_blk;
        lzo_uint dst_len = (lzo_uint)job->worst_blk;

        int rc;
        if (job->alg_id == 1) {
            rc = lzo1y_1_compress(src, (lzo_uint)this_blk, dst, &dst_len, wrkmem);
        } else {
            rc = lzo1x_1_compress(src, (lzo_uint)this_blk, dst, &dst_len, wrkmem);
        }

        if (rc != LZO_E_OK) {
            job->rc = -1;
        }
        job->lengths[blk_idx] = (uint32_t)dst_len;
    }

    lzo_aligned_free_portable(wrkmem);

    {
        uint64_t elapsed_us = (hybrid_now_ns() - t0) / 1000ULL;
        uint64_t old = atomic_load(&pool->max_worker_us);
        while (elapsed_us > old && !atomic_compare_exchange_weak(&pool->max_worker_us, &old, elapsed_us)) {
            /* retry */
        }
    }
    return NULL;
}

/* ---- CPU decompression worker ---- */

typedef struct {
    const unsigned char* comp_data;
    const uint32_t* offsets;
    const uint32_t* comp_lengths;
    unsigned char* output;
    uint32_t* out_lengths;
    size_t block_size;
    size_t orig_size;
    size_t nblk;
    const size_t* block_indices;
    size_t num_assigned;
    int alg_id;
    int rc;
} cpu_decompress_job_t;

typedef struct {
    cpu_decompress_job_t* job;
    _Atomic size_t next_idx;
    _Atomic uint64_t max_worker_us;
} cpu_decompress_pool_t;

static void* cpu_decompress_worker(void* arg) {
    cpu_decompress_pool_t* pool = (cpu_decompress_pool_t*)arg;
    cpu_decompress_job_t* job = pool->job;
    uint64_t t0 = hybrid_now_ns();

    for (;;) {
        size_t wi = atomic_fetch_add(&pool->next_idx, 1);
        if (wi >= job->num_assigned) break;

        size_t blk_idx = job->block_indices[wi];
        size_t out_offset = blk_idx * job->block_size;
        size_t this_blk = job->block_size;
        if (out_offset + this_blk > job->orig_size) this_blk = job->orig_size - out_offset;

        uint32_t comp_off = job->offsets[blk_idx];
        uint32_t comp_len = job->comp_lengths[blk_idx];

        const unsigned char* src = job->comp_data + comp_off;
        unsigned char* dst = job->output + out_offset;
        lzo_uint dst_len = (lzo_uint)this_blk;

        int rc;
        if (job->alg_id == 1) {
            rc = lzo1y_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
        } else {
            rc = lzo1x_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
        }

        if (rc != LZO_E_OK || dst_len != (lzo_uint)this_blk) {
            job->rc = -1;
        }
        job->out_lengths[blk_idx] = (uint32_t)dst_len;
    }

    {
        uint64_t elapsed_us = (hybrid_now_ns() - t0) / 1000ULL;
        uint64_t old = atomic_load(&pool->max_worker_us);
        while (elapsed_us > old && !atomic_compare_exchange_weak(&pool->max_worker_us, &old, elapsed_us)) {
            /* retry */
        }
    }

    return NULL;
}

/* ---- block index partitioning ---- */

static void partition_blocks(size_t nblk, double gpu_ratio,
                             size_t** gpu_indices, size_t* gpu_count,
                             size_t** cpu_indices, size_t* cpu_count) {
    size_t gn = (size_t)(nblk * gpu_ratio + 0.5);
    if (gn > nblk) gn = nblk;
    size_t cn = nblk - gn;

    *gpu_count = gn;
    *cpu_count = cn;

    *gpu_indices = (size_t*)malloc(gn * sizeof(size_t));
    *cpu_indices = (size_t*)malloc(cn * sizeof(size_t));
    if ((gn > 0 && !*gpu_indices) || (cn > 0 && !*cpu_indices)) {
        free(*gpu_indices);
        free(*cpu_indices);
        *gpu_indices = NULL;
        *cpu_indices = NULL;
        *gpu_count = 0;
        *cpu_count = 0;
        return;
    }

    {
        size_t gpu_written = 0;
        size_t cpu_written = 0;
        for (size_t idx = 0; idx < nblk; ++idx) {
            if ((((idx + 1) * gn) / nblk) != ((idx * gn) / nblk)) {
                (*gpu_indices)[gpu_written++] = idx;
            } else {
                (*cpu_indices)[cpu_written++] = idx;
            }
        }
        *gpu_count = gpu_written;
        *cpu_count = cpu_written;
    }
}

static double choose_adaptive_gpu_ratio(cl_context ctx, cl_command_queue queue,
                                        cl_device_id device, cl_kernel gpu_kernel,
                                        const unsigned char* input,
                                        size_t total_input_sz,
                                        size_t blk,
                                        size_t nblk,
                                        const hybrid_params_t* params,
                                        hybrid_workspace_t* ws) {
    /*
     * Device-aware adaptive scheduler using makespan minimization.
     *
     * Model:
     *   Pc_eff = Pc0_per_thread * gC(data) * sC(runtime) * thread_count
     *   Pg_eff = Pg0 * gG(data) * sG(runtime)
     *   r* = Pg_eff / (Pc_eff + Pg_eff) - t0 * Pc_eff * Pg_eff / (B * (Pc_eff + Pg_eff))
     *   gpu_ratio = clamp(r*, 0, 1)
     *
     * LZO uses entropy for data characterization:
     *   gC: CPU throughput varies with data compressibility (entropy-based scaling)
     *   gG: GPU throughput is memory-bandwidth-bound; entropy maps to compression ratio
     */
    double Pc0, Pg0, t0;
    double gC, gG;
    double sC, sG;
    double Pc_eff, Pg_eff;
    double r_star;
    long thread_count;
    long total_cores;
    double B = (double)total_input_sz;

    size_t sample_blocks;
    double mean_entropy = 0.0;
    size_t sampled = 0;
    double sample_cpu_throughput = 0.0;

    if (!gpu_kernel) return 0.0;

    if (!params || nblk == 0 || blk == 0 || total_input_sz == 0)
        return 0.5;

    /* --- 1. Device capability profile (cached) --- */
    lzo_calibrate_device_profile(ctx, queue, device, gpu_kernel, params, ws);
    Pc0 = g_lzo_dev_profile.cpu_throughput;
    Pg0 = g_lzo_dev_profile.gpu_throughput;
    t0  = g_lzo_dev_profile.gpu_overhead_s;

    total_cores = get_online_cpu_count();
    if (total_cores <= 0) total_cores = 4;
    if (params->cpu_threads > 0)
        thread_count = params->cpu_threads;
    else
        thread_count = total_cores;

    /* --- 2. Data characteristics from entropy sampling + CPU timing --- */
    gC = 1.0;
    gG = 1.0;
    sample_blocks = params->adaptive_sample_blocks ? params->adaptive_sample_blocks : 8;
    if (sample_blocks > nblk) sample_blocks = nblk;

    if (input && sample_blocks > 0) {
        size_t prev_block = SIZE_MAX;
        size_t sample_bytes = 0;

        ensure_lzo_init();
        {
            void* wrkmem = NULL;
            size_t wrkmem_sz = (params->alg_id == 1) ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
            unsigned char* tmp_out = (unsigned char*)malloc(lzo_worst_size(blk));
            wrkmem = malloc(wrkmem_sz);

            if (tmp_out && wrkmem) {
                memset(wrkmem, 0, wrkmem_sz);
                uint64_t sample_t0 = hybrid_now_ns();

                for (size_t i = 0; i < sample_blocks; ++i) {
                    const size_t blk_idx = sampled_block_index(i, sample_blocks, nblk);
                    const size_t blk_sz = block_input_size(total_input_sz, blk, blk_idx);
                    if (blk_idx == prev_block || blk_sz == 0) continue;

                    double entropy = lzo_calc_entropy(input + blk_idx * blk, blk_sz);
                    mean_entropy += entropy;
                    sampled++;
                    prev_block = blk_idx;
                    sample_bytes += blk_sz;

                    /* Time CPU compress for throughput measurement */
                    {
                        lzo_uint out_len = (lzo_uint)lzo_worst_size(blk);
                        if (params->alg_id == 1)
                            lzo1y_1_compress(input + blk_idx * blk, (lzo_uint)blk_sz, tmp_out, &out_len, wrkmem);
                        else
                            lzo1x_1_compress(input + blk_idx * blk, (lzo_uint)blk_sz, tmp_out, &out_len, wrkmem);
                    }
                }

                {
                    uint64_t sample_t1 = hybrid_now_ns();
                    if (sample_bytes > 0 && sample_t1 > sample_t0)
                        sample_cpu_throughput = (double)sample_bytes / ((double)(sample_t1 - sample_t0) * 1e-9);
                }
            }
            free(tmp_out);
            free(wrkmem);
        }

        if (sampled > 0) mean_entropy /= (double)sampled;

        /* gC: ratio of sample CPU throughput to calibration throughput */
        if (sample_cpu_throughput > 0.0 && Pc0 > 0.0) {
            gC = sample_cpu_throughput / Pc0;
            if (gC < 0.3) gC = 0.3;
            if (gC > 3.0) gC = 3.0;
        }

        /*
         * gG: GPU data-dependent factor from entropy.
         * High entropy → high compression ratio → less output bandwidth → slightly faster.
         * Low entropy → high compressibility → more varied output → gG stays near 1.0.
         * Map: entropy 0→8 bits corresponds roughly to ratio 0.05→0.95.
         * gG = (1 + m + Rref) / (1 + m + R_est)
         */
        {
            const double m = 2.0;
            const double Rref = 0.50;
            double R_est = mean_entropy / 8.0;
            if (R_est < 0.05) R_est = 0.05;
            if (R_est > 0.95) R_est = 0.95;
            gG = (1.0 + m + Rref) / (1.0 + m + R_est);
            if (gG < 0.5) gG = 0.5;
            if (gG > 2.0) gG = 2.0;
        }
    }

    /* --- 3. Runtime state --- */
    sC = lzo_read_cpu_availability();
    sG = lzo_read_gpu_availability();

    if (params->cpu_threads > 0 && params->cpu_threads < total_cores) {
        double scale = (double)total_cores / (double)params->cpu_threads;
        sC = sC * scale;
        if (sC > 1.0) sC = 1.0;
    }

    Pc_eff = Pc0 * gC * sC * (double)thread_count;
    Pg_eff = Pg0 * gG * sG;

    if (B <= t0 * Pg_eff && t0 > 0.0) {
        if (params->debug) {
            fprintf(stderr, "Adaptive: input too small (%.0f B <= overhead %.6f s * %.0f B/s), CPU-only\n",
                    B, t0, Pg_eff);
        }
        return 0.0;
    }

    if (Pc_eff + Pg_eff <= 0.0) return 0.5;
    r_star = Pg_eff / (Pc_eff + Pg_eff);
    if (B > 0.0 && t0 > 0.0) {
        r_star -= (t0 * Pc_eff * Pg_eff) / (B * (Pc_eff + Pg_eff));
    }

    /* --- 4. Energy-aware correction --- */
    {
        double eC = g_lzo_dev_profile.cpu_energy_per_byte;
        double eG = g_lzo_dev_profile.gpu_energy_per_byte;
        if (eC > 0.0 && eG > 0.0) {
            double r_energy = (Pg_eff * eC) / (Pc_eff * eG + Pg_eff * eC);
            r_star = 0.7 * r_star + 0.3 * r_energy;
        }
    }

    if (r_star < 0.0) r_star = 0.0;
    if (r_star > 1.0) r_star = 1.0;

    if (params->debug) {
        fprintf(stderr,
                "Adaptive: Pc0=%.0f gC=%.2f sC=%.2f threads=%ld Pc_eff=%.0f | "
                "Pg0=%.0f gG=%.2f sG=%.2f Pg_eff=%.0f | "
                "t0=%.6f B=%.0f entropy=%.2f eC=%.2e eG=%.2e r*=%.4f\n",
                Pc0, gC, sC, thread_count, Pc_eff,
                Pg0, gG, sG, Pg_eff,
                t0, B, mean_entropy,
                g_lzo_dev_profile.cpu_energy_per_byte,
                g_lzo_dev_profile.gpu_energy_per_byte,
                r_star);
    }

    return r_star;
}

/* ---- internal buffer-based compress (no file I/O) ---- */

static int hybrid_compress_buf(
    cl_context ctx, cl_command_queue queue, cl_device_id device,
    cl_kernel gpu_kernel, cl_kernel pack_kernel,
    const unsigned char* input_buf, size_t in_sz,
    unsigned char* out_buf, uint32_t* lengths,
    size_t blk, size_t nblk, size_t worst_blk,
    const hybrid_params_t* params, hybrid_workspace_t* ws,
    int skip_upload,
    hybrid_timing_t* timing_out
) {
    cl_int err;
    hybrid_timing_t timing = {0};
    uint64_t t_start = hybrid_now_ns();

    timing.in_size = in_sz;
    timing.blk_size = blk;
    timing.nblk = nblk;

    /* Partition blocks */
    size_t* gpu_idx = NULL;
    size_t* cpu_idx = NULL;
    size_t gpu_count = 0, cpu_count = 0;
    double gpu_ratio = params->gpu_ratio;
    int n_cpu_threads = 0;
    pthread_t* cpu_tids = NULL;
    int force_cpu_only = 0;
    int force_gpu_only = 0;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    if (params->split_mode == HYBRID_SPLIT_ADAPTIVE) {
        gpu_ratio = choose_adaptive_gpu_ratio(ctx, queue, device, gpu_kernel, input_buf, in_sz, blk, nblk, params, ws);
    }

    if (gpu_kernel == NULL) gpu_ratio = 0.0;
    if (gpu_ratio <= 0.0) {
        force_cpu_only = 1;
    } else if (gpu_ratio >= 1.0 || params->cpu_threads <= 0) {
        force_gpu_only = 1;
    }

    if (force_cpu_only) {
        gpu_count = 0;
        cpu_count = nblk;
        if (cpu_count > 0) {
            cpu_idx = (size_t*)malloc(cpu_count * sizeof(size_t));
            if (!cpu_idx) goto fail;
            for (size_t i = 0; i < cpu_count; ++i) cpu_idx[i] = i;
        }
    } else if (force_gpu_only) {
        gpu_count = nblk;
        cpu_count = 0;
        if (gpu_count > 0) {
            gpu_idx = (size_t*)malloc(gpu_count * sizeof(size_t));
            if (!gpu_idx) goto fail;
            for (size_t i = 0; i < gpu_count; ++i) gpu_idx[i] = i;
        }
    } else {
        partition_blocks(nblk, gpu_ratio, &gpu_idx, &gpu_count, &cpu_idx, &cpu_count);
    }

    if (gpu_count > 0 && (gpu_kernel == NULL || ctx == NULL || queue == NULL || ws == NULL)) {
        goto fail;
    }
    timing.gpu_blocks = gpu_count;
    timing.cpu_blocks = cpu_count;

    /* Launch CPU workers */
    cpu_compress_job_t cpu_job = {
        .input = input_buf,
        .output = out_buf,
        .lengths = lengths,
        .block_size = blk,
        .in_size = in_sz,
        .nblk = nblk,
        .worst_blk = worst_blk,
        .block_indices = cpu_idx,
        .num_assigned = cpu_count,
        .alg_id = params->alg_id,
        .rc = 0,
    };
    cpu_compress_pool_t cpu_pool = {
        .job = &cpu_job,
        .next_idx = 0,
        .max_worker_us = 0,
    };

    n_cpu_threads = (cpu_count > 0) ? params->cpu_threads : 0;
    if (n_cpu_threads > (int)cpu_count) n_cpu_threads = (int)cpu_count;
    uint64_t t_cpu_start = hybrid_now_ns();
    if (n_cpu_threads > 0) {
        if (n_cpu_threads == 1) {
            cpu_compress_worker(&cpu_pool);
            timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
        } else {
            cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
            for (int i = 0; i < n_cpu_threads; i++)
                pthread_create(&cpu_tids[i], NULL, cpu_compress_worker, &cpu_pool);
        }
    }

    /* GPU compression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;
    unsigned long pack_kernel_us = 0;

    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();
        const unsigned char* gpu_upload_ptr;
        size_t gpu_input_sz;
        int can_skip_upload = (skip_upload && force_gpu_only && cpu_count == 0);

        if (cpu_count == 0) {
            gpu_upload_ptr = input_buf;
            gpu_input_sz = in_sz;
        } else {
            unsigned char* gpu_input = (unsigned char*)calloc(gpu_count, blk);
            if (!gpu_input) goto fail;
            gpu_input_sz = gpu_count * blk;
            for (size_t i = 0; i < gpu_count; ++i) {
                size_t blk_idx = gpu_idx[i];
                size_t this_blk = block_input_size(in_sz, blk, blk_idx);
                memcpy(gpu_input + i * blk, input_buf + blk_idx * blk, this_blk);
                if (blk_idx == nblk - 1) gpu_input_sz = i * blk + this_blk;
            }
            gpu_upload_ptr = gpu_input;
        }

        if (!can_skip_upload) {
            grow_buffer(ctx, &ws->d_in, &ws->in_cap, gpu_input_sz, CL_MEM_READ_ONLY, &err);
            if (err != CL_SUCCESS) { if (cpu_count != 0) free((void*)(uintptr_t)gpu_upload_ptr); goto fail; }
            err = clEnqueueWriteBuffer(queue, ws->d_in, CL_TRUE, 0, gpu_input_sz, gpu_upload_ptr, 0, NULL, NULL);
        } else {
            err = CL_SUCCESS;
        }
        if (cpu_count != 0) free((void*)(uintptr_t)gpu_upload_ptr);
        if (err != CL_SUCCESS) goto fail;

        size_t out_needed = gpu_count * worst_blk;
        grow_buffer(ctx, &ws->d_out, &ws->out_cap, out_needed, CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) goto fail;

        size_t len_needed = gpu_count * sizeof(cl_uint);
        grow_buffer(ctx, &ws->d_len, &ws->len_cap, len_needed, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) goto fail;

        /* Dictionary / SWD pool */
        int is_999 = (params->comp_level == 999);
        size_t dict_per_block;
        if (is_999) {
            dict_per_block = 458752ULL;
        } else {
            dict_per_block = (1ULL << params->comp_level) * sizeof(uint32_t);
        }
        size_t pool_size = gpu_count;
        size_t total_dict = pool_size * dict_per_block;
        size_t prev_dict = ws->dict_cap;
        grow_buffer(ctx, &ws->d_dict, &ws->dict_cap, total_dict, CL_MEM_READ_WRITE, &err);
        if (err != CL_SUCCESS) goto fail;
        if (is_999 || ws->dict_cap != prev_dict) {
            zero_buffer(queue, ws->d_dict, ws->dict_cap);
        }

        uint32_t epoch_base = 0;
        if (!is_999) {
            if (ws->comp_epoch_base == 0) ws->comp_epoch_base = 1;
            if ((uint32_t)gpu_count + 2U >= 4095U ||
                ws->comp_epoch_base + (uint32_t)gpu_count + 1U > 4095U) {
                zero_buffer(queue, ws->d_dict, ws->dict_cap);
                ws->comp_epoch_base = 1;
            }
            epoch_base = ws->comp_epoch_base;
            ws->comp_epoch_base += (uint32_t)gpu_count + 1U;
        }

        cl_uint gpu_in_sz = (cl_uint)gpu_input_sz;
        cl_uint blk_cl = (cl_uint)blk;
        cl_uint worst_blk_cl = (cl_uint)worst_blk;
        cl_uint pool_size_cl = (cl_uint)pool_size;

        if (!can_skip_upload) {
            CHECK_CL(clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &ws->d_in));
            CHECK_CL(clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &ws->d_out));
            CHECK_CL(clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &ws->d_len));
            CHECK_CL(clSetKernelArg(gpu_kernel, 3, sizeof(cl_uint), &gpu_in_sz));
            CHECK_CL(clSetKernelArg(gpu_kernel, 4, sizeof(cl_uint), &blk_cl));
            CHECK_CL(clSetKernelArg(gpu_kernel, 5, sizeof(cl_uint), &worst_blk_cl));
            CHECK_CL(clSetKernelArg(gpu_kernel, 6, sizeof(cl_mem), &ws->d_dict));

            if (is_999) {
                cl_uint swd_pool_count_cl = pool_size_cl;
                cl_uint try_lazy_cl = 2U;
                cl_uint max_chain_cl = 4096U;
                CHECK_CL(clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &swd_pool_count_cl));
                CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &try_lazy_cl));
                CHECK_CL(clSetKernelArg(gpu_kernel, 9, sizeof(cl_uint), &max_chain_cl));
            } else {
                CHECK_CL(clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &pool_size_cl));
                CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &epoch_base));

                {
                    cl_uint krn_num_args = 0;
                    clGetKernelInfo(gpu_kernel, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL);
                    if (krn_num_args >= 11U) {
                        cl_uint dbg_flag = 0U;
                        CHECK_CL(clSetKernelArg(gpu_kernel, 9, sizeof(cl_mem), &ws->d_len));
                        CHECK_CL(clSetKernelArg(gpu_kernel, 10, sizeof(cl_uint), &dbg_flag));
                    }
                }
            }
        } else if (!is_999) {
            CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_uint), &epoch_base));
        }

        uint64_t t_up1 = hybrid_now_ns();
        timing.upload_us = (unsigned long)((t_up1 - t_up0) / 1000);

        size_t local_size = (params->local_size > 0) ? (size_t)params->local_size : 1;
        size_t global_size = gpu_count;
        if (local_size > global_size) local_size = 1;
        global_size = ((global_size + local_size - 1) / local_size) * local_size;
        if (global_size == 0) global_size = 1;

        t_gpu_k0 = hybrid_now_ns();
        err = clEnqueueNDRangeKernel(queue, gpu_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto fail;
        clFinish(queue);
        t_gpu_k1 = hybrid_now_ns();

        /* Read GPU lengths */
        uint64_t t_dl0 = hybrid_now_ns();
        void* mapped_len = clEnqueueMapBuffer(queue, ws->d_len, CL_TRUE, CL_MAP_READ,
                                               0, gpu_count * sizeof(cl_uint), 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) goto fail;
        for (size_t i = 0; i < gpu_count; i++) {
            lengths[gpu_idx[i]] = ((cl_uint*)mapped_len)[i];
        }
        clEnqueueUnmapMemObject(queue, ws->d_len, mapped_len, 0, NULL, NULL);

        {
            size_t packed_total = 0;
            int use_compaction;
            for (size_t i = 0; i < gpu_count; i++) packed_total += (size_t)lengths[gpu_idx[i]];

            use_compaction = hybrid_should_use_device_compaction(
                packed_total,
                gpu_count * worst_blk,
                gpu_count,
                pack_kernel
            );

            if (use_compaction && packed_total > 0) {
                {
                    uint64_t t_pack_up0 = hybrid_now_ns();

                    grow_buffer(ctx, &ws->d_packed_off, &ws->packed_off_cap,
                                gpu_count * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
                    if (err != CL_SUCCESS) goto fail;
                    grow_buffer(ctx, &ws->d_packed_out, &ws->packed_out_cap,
                                packed_total, CL_MEM_WRITE_ONLY, &err);
                    if (err != CL_SUCCESS) goto fail;

                    {
                        void* mapped_off = clEnqueueMapBuffer(queue, ws->d_packed_off, CL_TRUE, CL_MAP_WRITE,
                                                              0, gpu_count * sizeof(cl_uint), 0, NULL, NULL, &err);
                        if (err != CL_SUCCESS) goto fail;
                        size_t packed_offset = 0;
                        for (size_t i = 0; i < gpu_count; i++) {
                            ((cl_uint*)mapped_off)[i] = (cl_uint)packed_offset;
                            packed_offset += (size_t)lengths[gpu_idx[i]];
                        }
                        err = clEnqueueUnmapMemObject(queue, ws->d_packed_off, mapped_off, 0, NULL, NULL);
                        if (err != CL_SUCCESS) goto fail;
                    }
                    timing.upload_us += (unsigned long)((hybrid_now_ns() - t_pack_up0) / 1000);
                }

                {
                    cl_uint worst_blk_cl = (cl_uint)worst_blk;
                    cl_uint gpu_count_cl = (cl_uint)gpu_count;
                    size_t pack_global = gpu_count;
                    uint64_t t_pack_k0 = hybrid_now_ns();
                    err  = clSetKernelArg(pack_kernel, 0, sizeof(cl_mem), &ws->d_out);
                    err |= clSetKernelArg(pack_kernel, 1, sizeof(cl_mem), &ws->d_len);
                    err |= clSetKernelArg(pack_kernel, 2, sizeof(cl_mem), &ws->d_packed_out);
                    err |= clSetKernelArg(pack_kernel, 3, sizeof(cl_mem), &ws->d_packed_off);
                    err |= clSetKernelArg(pack_kernel, 4, sizeof(cl_uint), &worst_blk_cl);
                    err |= clSetKernelArg(pack_kernel, 5, sizeof(cl_uint), &gpu_count_cl);
                    if (err != CL_SUCCESS) goto fail;
                    err = clEnqueueNDRangeKernel(queue, pack_kernel, 1, NULL, &pack_global, NULL, 0, NULL, NULL);
                    if (err != CL_SUCCESS) goto fail;
                    clFinish(queue);
                    pack_kernel_us = (unsigned long)((hybrid_now_ns() - t_pack_k0) / 1000);
                }

                {
                    void* mapped_packed = clEnqueueMapBuffer(queue, ws->d_packed_out, CL_TRUE, CL_MAP_READ,
                                                             0, packed_total, 0, NULL, NULL, &err);
                    if (err != CL_SUCCESS) goto fail;
                    size_t packed_offset = 0;
                    for (size_t i = 0; i < gpu_count; i++) {
                        size_t blk_idx = gpu_idx[i];
                        memcpy(out_buf + blk_idx * worst_blk,
                               (unsigned char*)mapped_packed + packed_offset,
                               (size_t)lengths[blk_idx]);
                        packed_offset += (size_t)lengths[blk_idx];
                    }
                    clEnqueueUnmapMemObject(queue, ws->d_packed_out, mapped_packed, 0, NULL, NULL);
                }
            } else {
                void* mapped_out = clEnqueueMapBuffer(queue, ws->d_out, CL_TRUE, CL_MAP_READ,
                                                       0, gpu_count * worst_blk, 0, NULL, NULL, &err);
                if (err != CL_SUCCESS) goto fail;
                for (size_t i = 0; i < gpu_count; i++) {
                    size_t blk_idx = gpu_idx[i];
                    memcpy(out_buf + blk_idx * worst_blk,
                           (unsigned char*)mapped_out + i * worst_blk,
                           (size_t)lengths[blk_idx]);
                }
                clEnqueueUnmapMemObject(queue, ws->d_out, mapped_out, 0, NULL, NULL);
            }
        }
        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU workers */
    uint64_t t_cpu_end = t_cpu_start;
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++)
            pthread_join(cpu_tids[i], NULL);
        t_cpu_end = hybrid_now_ns();
        free(cpu_tids);
        cpu_tids = NULL;
        timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
    }
    timing.gpu_kernel_us = (t_gpu_k1 > t_gpu_k0) ? (unsigned long)((t_gpu_k1 - t_gpu_k0) / 1000) + pack_kernel_us : pack_kernel_us;

    if (cpu_job.rc != 0) goto fail;

    /* Compute total compressed size */
    size_t comp_total = 0;
    for (size_t i = 0; i < nblk; i++) comp_total += lengths[i];
    timing.out_size = comp_total;

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    free(gpu_idx); free(cpu_idx);
    return 0;

fail:
    free(gpu_idx); free(cpu_idx);
    if (cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        free(cpu_tids);
    }
    return -1;
}

/* ---- internal buffer-based decompress (no file I/O) ---- */

static int hybrid_decompress_buf(
    cl_context ctx, cl_command_queue queue,
    cl_kernel gpu_kernel,
    const unsigned char* packed, const uint32_t* offsets, const uint32_t* comp_lengths,
    size_t packed_sz, uint32_t orig_sz, uint32_t blk_sz, uint32_t nblk, int alg_id,
    unsigned char* output_buf, uint32_t* out_lengths,
    const hybrid_params_t* params, hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    cl_int err;
    hybrid_timing_t timing = {0};
    uint64_t t_start = hybrid_now_ns();

    timing.in_size = orig_sz;
    timing.blk_size = blk_sz;
    timing.nblk = nblk;

    /* Partition */
    size_t* gpu_idx = NULL;
    size_t* cpu_idx = NULL;
    size_t gpu_count = 0, cpu_count = 0;
    double gpu_ratio = params->gpu_ratio;
    int n_cpu_threads = 0;
    pthread_t* cpu_tids = NULL;
    int force_cpu_only = 0;
    int force_gpu_only = 0;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    if (params->split_mode == HYBRID_SPLIT_ADAPTIVE) {
        gpu_ratio = choose_adaptive_gpu_ratio(ctx, queue, 0, gpu_kernel, NULL, orig_sz, blk_sz, nblk, params, ws);
    }

    if (gpu_kernel == NULL) gpu_ratio = 0.0;
    if (gpu_ratio <= 0.0) {
        force_cpu_only = 1;
    } else if (gpu_ratio >= 1.0 || params->cpu_threads <= 0) {
        force_gpu_only = 1;
    }

    if (force_cpu_only) {
        gpu_count = 0;
        cpu_count = nblk;
        if (cpu_count > 0) {
            cpu_idx = (size_t*)malloc(cpu_count * sizeof(size_t));
            if (!cpu_idx) goto dfail;
            for (size_t i = 0; i < cpu_count; ++i) cpu_idx[i] = i;
        }
    } else if (force_gpu_only) {
        gpu_count = nblk;
        cpu_count = 0;
        if (gpu_count > 0) {
            gpu_idx = (size_t*)malloc(gpu_count * sizeof(size_t));
            if (!gpu_idx) goto dfail;
            for (size_t i = 0; i < gpu_count; ++i) gpu_idx[i] = i;
        }
    } else {
        partition_blocks(nblk, gpu_ratio, &gpu_idx, &gpu_count, &cpu_idx, &cpu_count);
    }

    if (gpu_count > 0 && (gpu_kernel == NULL || ctx == NULL || queue == NULL || ws == NULL)) {
        goto dfail;
    }
    timing.gpu_blocks = gpu_count;
    timing.cpu_blocks = cpu_count;

    memset(out_lengths, 0, nblk * sizeof(uint32_t));

    /* Launch CPU decompression */
    cpu_decompress_job_t cpu_job = {
        .comp_data = packed,
        .offsets = offsets,
        .comp_lengths = comp_lengths,
        .output = output_buf,
        .out_lengths = out_lengths,
        .block_size = blk_sz,
        .orig_size = orig_sz,
        .nblk = nblk,
        .block_indices = cpu_idx,
        .num_assigned = cpu_count,
        .alg_id = alg_id,
        .rc = 0,
    };
    cpu_decompress_pool_t cpu_pool = {
        .job = &cpu_job,
        .next_idx = 0,
        .max_worker_us = 0,
    };

    n_cpu_threads = (cpu_count > 0) ? params->cpu_threads : 0;
    if (n_cpu_threads > (int)cpu_count) n_cpu_threads = (int)cpu_count;
    uint64_t t_cpu_start = hybrid_now_ns();
    if (n_cpu_threads > 0) {
        if (n_cpu_threads == 1) {
            cpu_decompress_worker(&cpu_pool);
            timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
        } else {
            cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
            for (int i = 0; i < n_cpu_threads; i++)
                pthread_create(&cpu_tids[i], NULL, cpu_decompress_worker, &cpu_pool);
        }
    }

    /* GPU decompression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;
    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();
        uint32_t* gpu_offsets = NULL;
        uint32_t* gpu_comp_lens = NULL;
        size_t gpu_orig_sz = gpu_count * blk_sz;
        for (size_t i = 0; i < gpu_count; ++i) {
            size_t blk_idx = gpu_idx[i];
            if (blk_idx == nblk - 1) gpu_orig_sz = i * blk_sz + block_input_size(orig_sz, blk_sz, blk_idx);
        }
        gpu_offsets = (uint32_t*)malloc(gpu_count * sizeof(uint32_t));
        gpu_comp_lens = (uint32_t*)malloc(gpu_count * sizeof(uint32_t));
        if (!gpu_offsets || !gpu_comp_lens) {
            free(gpu_offsets);
            free(gpu_comp_lens);
            goto dfail;
        }
        for (size_t i = 0; i < gpu_count; ++i) {
            size_t blk_idx = gpu_idx[i];
            gpu_offsets[i] = offsets[blk_idx];
            gpu_comp_lens[i] = comp_lengths[blk_idx];
        }

        grow_buffer(ctx, &ws->d_comp, &ws->comp_cap, packed_sz ? packed_sz : 1, CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_off, &ws->off_cap, gpu_count * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_comp_lens, &ws->comp_lens_cap, gpu_count * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_cap, gpu_orig_sz ? gpu_orig_sz : 1, CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_out_lens, &ws->out_lens_cap, gpu_count * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }

        err = clEnqueueWriteBuffer(queue, ws->d_comp, CL_TRUE, 0, packed_sz ? packed_sz : 1, packed, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, ws->d_off, CL_TRUE, 0, gpu_count * sizeof(cl_uint), gpu_offsets, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, ws->d_comp_lens, CL_TRUE, 0, gpu_count * sizeof(cl_uint), gpu_comp_lens, 0, NULL, NULL);
        free(gpu_offsets);
        free(gpu_comp_lens);
        if (err != CL_SUCCESS) goto dfail;
        uint64_t t_up1 = hybrid_now_ns();
        timing.upload_us = (unsigned long)((t_up1 - t_up0) / 1000);

        cl_uint blk_sz_cl = blk_sz;
        cl_uint orig_sz_cl = (cl_uint)gpu_orig_sz;
        cl_uint nblk_cl = (cl_uint)gpu_count;
        CHECK_CL(clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &ws->d_comp));
        CHECK_CL(clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &ws->d_off));
        CHECK_CL(clSetKernelArg(gpu_kernel, 2, sizeof(cl_mem), &ws->d_comp_lens));
        CHECK_CL(clSetKernelArg(gpu_kernel, 3, sizeof(cl_mem), &ws->d_decomp_out));
        CHECK_CL(clSetKernelArg(gpu_kernel, 4, sizeof(cl_mem), &ws->d_out_lens));
        CHECK_CL(clSetKernelArg(gpu_kernel, 5, sizeof(cl_uint), &blk_sz_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 6, sizeof(cl_uint), &orig_sz_cl));
        CHECK_CL(clSetKernelArg(gpu_kernel, 7, sizeof(cl_uint), &nblk_cl));

        cl_uint krn_num_args = 0;
        clGetKernelInfo(gpu_kernel, CL_KERNEL_NUM_ARGS, sizeof(krn_num_args), &krn_num_args, NULL);
        if (krn_num_args >= 10U) {
            cl_uint dbg_flag = 0U;
            CHECK_CL(clSetKernelArg(gpu_kernel, 8, sizeof(cl_mem), &ws->d_out_lens));
            CHECK_CL(clSetKernelArg(gpu_kernel, 9, sizeof(cl_uint), &dbg_flag));
        }

        size_t local_size = (params->local_size > 0) ? (size_t)params->local_size : 1;
        size_t global_size = gpu_count;
        if (local_size > global_size) local_size = 1;
        global_size = ((global_size + local_size - 1) / local_size) * local_size;

        t_gpu_k0 = hybrid_now_ns();
        err = clEnqueueNDRangeKernel(queue, gpu_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) goto dfail;
        clFinish(queue);
        t_gpu_k1 = hybrid_now_ns();

        /* Read GPU output for GPU-assigned blocks only */
        uint64_t t_dl0 = hybrid_now_ns();
        size_t gpu_out_sz = gpu_orig_sz;
        void* mapped = clEnqueueMapBuffer(queue, ws->d_decomp_out, CL_TRUE, CL_MAP_READ,
                                            0, gpu_out_sz, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS) goto dfail;
        for (size_t i = 0; i < gpu_count; i++) {
            size_t bi = gpu_idx[i];
            size_t off = bi * blk_sz;
            size_t this_blk = blk_sz;
            if (off + this_blk > orig_sz) this_blk = orig_sz - off;
            memcpy(output_buf + off, (unsigned char*)mapped + i * blk_sz, this_blk);
            out_lengths[bi] = (uint32_t)this_blk;
        }
        clEnqueueUnmapMemObject(queue, ws->d_decomp_out, mapped, 0, NULL, NULL);
        clFinish(queue);
        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU */
    uint64_t t_cpu_end = t_cpu_start;
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        t_cpu_end = hybrid_now_ns();
        free(cpu_tids);
        cpu_tids = NULL;
        timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
    }
    timing.gpu_kernel_us = (t_gpu_k1 > t_gpu_k0) ? (unsigned long)((t_gpu_k1 - t_gpu_k0) / 1000) : 0;

    if (cpu_job.rc != 0) goto dfail;

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    free(gpu_idx); free(cpu_idx);
    return 0;

dfail:
    if (cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        free(cpu_tids);
    }
    free(gpu_idx); free(cpu_idx);
    return -1;
}

/* ---- public hybrid compress (file-based) ---- */

int hybrid_compress(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel gpu_kernel,
    cl_kernel pack_kernel,
    const char* input_path,
    const char* output_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    ensure_lzo_init();
    uint64_t t_start = hybrid_now_ns();

    struct stat st;
    if (stat(input_path, &st) != 0 || st.st_size <= 0) return -1;
    size_t in_sz = (size_t)st.st_size;

    /* Read input */
    uint64_t t_read0 = hybrid_now_ns();
    unsigned char* input_buf = NULL;
    if (lzo_aligned_alloc_portable((void**)&input_buf, 4096, in_sz) != 0) {
        input_buf = (unsigned char*)malloc(in_sz);
        if (!input_buf) return -1;
    }
    unsigned long read_us = 0;
    if (lzo_read_file_to_buf(input_path, input_buf, in_sz, &read_us) != 0) {
        lzo_aligned_free_portable(input_buf);
        return -1;
    }
    uint64_t t_read1 = hybrid_now_ns();

    /* Block calculation */
    size_t blk = 0, nblk = 0;
    size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
    hybrid_choose_blocking(input_buf, in_sz, device, gpu_kernel, blk_bytes, params->debug, &blk, &nblk);
    size_t worst_blk = lzo_worst_size(blk);

    /* Allocate output arrays */
    uint32_t* lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    unsigned char* out_buf = (unsigned char*)malloc(nblk * worst_blk);
    if (!lengths || !out_buf) {
        lzo_aligned_free_portable(input_buf); free(lengths); free(out_buf);
        return -1;
    }

    /* Compress using buffer-based function */
    hybrid_timing_t timing = {0};
    int rc = hybrid_compress_buf(ctx, queue, device, gpu_kernel, pack_kernel,
                                  input_buf, in_sz, out_buf, lengths,
                                  blk, nblk, worst_blk, params, ws, 0, &timing);
    if (rc != 0) {
        lzo_aligned_free_portable(input_buf); free(lengths); free(out_buf);
        return -1;
    }

    timing.file_read_us = (unsigned long)((t_read1 - t_read0) / 1000);

    /* Write output file */
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        uint64_t t_w0 = hybrid_now_ns();
        int wr = lzo_write_compressed_file(output_path, in_sz, blk, nblk,
                                           (unsigned int*)lengths, out_buf,
                                           worst_blk, params->alg_id, params->debug);
        uint64_t t_w1 = hybrid_now_ns();
        timing.file_write_us = (unsigned long)((t_w1 - t_w0) / 1000);
        if (wr != 0) { lzo_aligned_free_portable(input_buf); free(lengths); free(out_buf); return -1; }
    }

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    if (timing_out) *timing_out = timing;

    lzo_aligned_free_portable(input_buf); free(lengths); free(out_buf);
    return 0;
}

/* ---- public hybrid decompress (file-based) ---- */

int hybrid_decompress(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel gpu_kernel,
    const char* input_path,
    const char* output_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    hybrid_timing_t* timing_out
) {
    ensure_lzo_init();
    uint64_t t_start = hybrid_now_ns();
    hybrid_timing_t timing = {0};

    /* Read .lzo header */
    FILE* f = fopen(input_path, "rb");
    if (!f) return -1;

    uint16_t magic;
    fread(&magic, 2, 1, f);
    if (magic != MAGIC) { fclose(f); return -1; }

    uint32_t orig_sz, blk_sz, nblk, alg_id;
    fread(&orig_sz, 4, 1, f);
    fread(&blk_sz, 4, 1, f);
    fread(&nblk, 4, 1, f);
    fread(&alg_id, 4, 1, f);

    uint32_t* comp_lengths = (uint32_t*)malloc(nblk * sizeof(uint32_t));
    if (!comp_lengths) { fclose(f); return -1; }
    fread(comp_lengths, sizeof(uint32_t), nblk, f);

    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    size_t comp_data_sz = (size_t)(ftell(f) - data_start);
    fseek(f, data_start, SEEK_SET);

    unsigned char* comp_data = (unsigned char*)malloc(comp_data_sz);
    if (!comp_data) { free(comp_lengths); fclose(f); return -1; }
    fread(comp_data, 1, comp_data_sz, f);
    fclose(f);

    /* Build per-block offsets into packed compressed data */
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    offsets[0] = 0;
    for (uint32_t i = 0; i < nblk; i++) offsets[i + 1] = offsets[i] + comp_lengths[i];

    unsigned char* output_buf = (unsigned char*)malloc(orig_sz);
    uint32_t* out_lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    if (!output_buf || !out_lengths) goto dec_fail;

    /* Decompress using buffer-based function */
    int rc = hybrid_decompress_buf(ctx, queue, gpu_kernel,
                                    comp_data, offsets, comp_lengths,
                                    comp_data_sz, orig_sz, blk_sz, nblk, (int)alg_id,
                                    output_buf, out_lengths,
                                    params, ws, &timing);
    if (rc != 0) goto dec_fail;

    /* Write output */
    if (output_path && strcmp(output_path, "/dev/null") != 0) {
        uint64_t t_w0 = hybrid_now_ns();
        FILE* fout = fopen(output_path, "wb");
        if (!fout) goto dec_fail;
        fwrite(output_buf, 1, orig_sz, fout);
        fclose(fout);
        uint64_t t_w1 = hybrid_now_ns();
        timing.file_write_us = (unsigned long)((t_w1 - t_w0) / 1000);
    }

    uint64_t t_end = hybrid_now_ns();
    timing.total_us = (unsigned long)((t_end - t_start) / 1000);
    timing.out_size = comp_data_sz;
    if (timing_out) *timing_out = timing;

    free(comp_data); free(comp_lengths); free(offsets);
    free(output_buf); free(out_lengths);
    return 0;

dec_fail:
    free(comp_data); free(comp_lengths); free(offsets);
    free(output_buf); free(out_lengths);
    return -1;
}

/* ---- Bench mode (in-memory, no file I/O in hot loop) ---- */

static int cmp_double_asc(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static double median_double(const double* vals, size_t n) {
    if (!vals || n == 0) return 0.0;
    double* tmp = (double*)malloc(n * sizeof(double));
    memcpy(tmp, vals, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double_asc);
    double out = (n % 2 == 0) ? (tmp[n/2 - 1] + tmp[n/2]) / 2.0 : tmp[n/2];
    free(tmp);
    return out;
}

static double elapsed_sec(const struct timespec* s, const struct timespec* e) {
    return (double)(e->tv_sec - s->tv_sec) + (double)(e->tv_nsec - s->tv_nsec) / 1e9;
}

static int create_temp_path(char* path_buf, size_t path_buf_size, const char* templ) {
    int fd;
    if (!path_buf || path_buf_size == 0 || !templ) return -1;
    if (strlen(templ) + 1 > path_buf_size) return -1;
    memcpy(path_buf, templ, strlen(templ) + 1);
    fd = mkstemp(path_buf);
    if (fd < 0) return -1;
    close(fd);
    remove(path_buf);
    return 0;
}

int hybrid_bench(
    cl_context ctx,
    cl_command_queue queue,
    cl_device_id device,
    cl_kernel comp_kernel,
    cl_kernel pack_kernel,
    cl_kernel dec_kernel,
    const char* input_path,
    const hybrid_params_t* params,
    hybrid_workspace_t* ws,
    double bench_seconds,
    int include_file_io
) {
    ensure_lzo_init();
    if (bench_seconds <= 0.0) bench_seconds = 3.0;

    if (!comp_kernel) {
        pack_kernel = NULL;
    }
    if (params->cpu_threads <= 0 && !comp_kernel) {
        fprintf(stderr, "bench error: both CPU and GPU compression paths are unavailable\n");
        return 1;
    }
    if (params->cpu_threads <= 0 && !dec_kernel) {
        fprintf(stderr, "bench error: both CPU and GPU decompression paths are unavailable\n");
        return 1;
    }

    struct stat st;
    if (stat(input_path, &st) != 0 || st.st_size <= 0) return 1;
    size_t in_size = (size_t)st.st_size;

    /* Read input once */
    size_t ref_sz = 0;
    unsigned char* input_ref = (unsigned char*)lzo_read_file(input_path, &ref_sz);
    if (!input_ref || ref_sz != in_size) { free(input_ref); return 1; }

    /* Block calculation (done once) */
    size_t blk = 0, nblk = 0;
    size_t blk_bytes = (params->block_size > 0) ? params->block_size : 0;
    hybrid_choose_blocking(input_ref, in_size, device, comp_kernel, blk_bytes, params->debug, &blk, &nblk);
    size_t worst_blk = lzo_worst_size(blk);

    /* Pre-allocate reusable buffers */
    uint32_t* lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    unsigned char* out_buf = (unsigned char*)malloc(nblk * worst_blk);
    unsigned char* packed = NULL;
    size_t packed_cap = 0;
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    unsigned char* dec_buf = (unsigned char*)malloc(in_size);
    uint32_t* out_lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    if (!lengths || !out_buf || !offsets || !dec_buf || !out_lengths) {
        free(lengths); free(out_buf); free(packed); free(offsets);
        free(dec_buf); free(out_lengths); free(input_ref);
        return 1;
    }

    size_t cap = 16, n = 0;
    double* comp_ktp = (double*)malloc(cap * sizeof(double));
    double* dec_ktp = (double*)malloc(cap * sizeof(double));
    double* comp_ttp = (double*)malloc(cap * sizeof(double));
    double* dec_ttp = (double*)malloc(cap * sizeof(double));
    double* ratio_pct = (double*)malloc(cap * sizeof(double));
    int verify_ok = 1;
    char tmp_comp_path[4096] = {0};
    char tmp_dec_path[4096] = {0};

    if (include_file_io) {
        if (create_temp_path(tmp_comp_path, sizeof(tmp_comp_path), "/tmp/lzo_hybrid_bench_io_comp_XXXXXX") != 0 ||
            create_temp_path(tmp_dec_path, sizeof(tmp_dec_path), "/tmp/lzo_hybrid_bench_io_dec_XXXXXX") != 0) {
            free(comp_ktp); free(dec_ktp); free(comp_ttp); free(dec_ttp); free(ratio_pct);
            free(input_ref); free(lengths); free(out_buf);
            free(packed); free(offsets); free(dec_buf); free(out_lengths);
            return 1;
        }
    }

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    while (verify_ok) {
        hybrid_timing_t tc = {0};
        hybrid_timing_t td = {0};
        int rc = 0;
        size_t comp_total = 0;
        unsigned long comp_kernel_us = 0;
        unsigned long dec_kernel_us = 0;
        double dec_total_us = 0.0;

        if (include_file_io) {
            size_t dec_ref_sz = 0;
            unsigned char* dec_ref = NULL;

            rc = hybrid_compress(ctx, queue, device, comp_kernel,
                                 pack_kernel,
                                 input_path, tmp_comp_path, params, ws, &tc);
            if (rc != 0) { verify_ok = 0; break; }
            comp_total = tc.out_size;
            comp_kernel_us = tc.gpu_kernel_us;
            if (tc.cpu_kernel_us > comp_kernel_us) comp_kernel_us = tc.cpu_kernel_us;

            rc = hybrid_decompress(ctx, queue, device, dec_kernel,
                                   tmp_comp_path, tmp_dec_path, params, ws, &td);
            if (rc != 0) {
                remove(tmp_comp_path);
                remove(tmp_dec_path);
                verify_ok = 0;
                break;
            }
            dec_kernel_us = td.gpu_kernel_us;
            if (td.cpu_kernel_us > dec_kernel_us) dec_kernel_us = td.cpu_kernel_us;
            dec_total_us = (double)td.total_us;

            dec_ref = (unsigned char*)lzo_read_file(tmp_dec_path, &dec_ref_sz);
            if (!dec_ref || dec_ref_sz != in_size || memcmp(dec_ref, input_ref, in_size) != 0) {
                free(dec_ref);
                remove(tmp_comp_path);
                remove(tmp_dec_path);
                verify_ok = 0;
                break;
            }
            free(dec_ref);
            remove(tmp_comp_path);
            remove(tmp_dec_path);
        } else {
            /* ---- COMPRESS (in-memory) ---- */
            int skip_upload = (params->gpu_ratio >= 1.0 && n > 0) ? 1 : 0;
            memset(lengths, 0, nblk * sizeof(uint32_t));
            rc = hybrid_compress_buf(ctx, queue, device, comp_kernel, pack_kernel,
                                     input_ref, in_size, out_buf, lengths,
                                     blk, nblk, worst_blk, params, ws, skip_upload, &tc);
            if (rc != 0) { verify_ok = 0; break; }

            comp_total = tc.out_size;
            if (comp_total == 0 || nblk == 0) { verify_ok = 0; break; }

            comp_kernel_us = tc.gpu_kernel_us;
            if (tc.cpu_kernel_us > comp_kernel_us) comp_kernel_us = tc.cpu_kernel_us;

            /* Pack compressed data (strip worst_blk padding) */
            if (comp_total > packed_cap) {
                free(packed);
                packed = (unsigned char*)malloc(comp_total);
                packed_cap = packed ? comp_total : 0;
                if (!packed) { verify_ok = 0; break; }
            }
            offsets[0] = 0;
            size_t co = 0;
            for (size_t i = 0; i < nblk; i++) {
                size_t csz = (size_t)lengths[i];
                memcpy(packed + co, out_buf + i * worst_blk, csz);
                co += csz;
                offsets[i + 1] = (uint32_t)co;
            }

            /* ---- DECOMPRESS (in-memory) ---- */
            uint64_t dec_total_start = hybrid_now_ns();
            rc = hybrid_decompress_buf(ctx, queue, dec_kernel,
                                       packed, offsets, lengths,
                                       comp_total, (uint32_t)in_size, (uint32_t)blk, (uint32_t)nblk,
                                       params->alg_id,
                                       dec_buf, out_lengths,
                                       params, ws, &td);
            uint64_t dec_total_end = hybrid_now_ns();

            if (rc != 0) { verify_ok = 0; break; }

            if (memcmp(dec_buf, input_ref, in_size) != 0) {
                verify_ok = 0;
                break;
            }

            dec_kernel_us = td.gpu_kernel_us;
            if (td.cpu_kernel_us > dec_kernel_us) dec_kernel_us = td.cpu_kernel_us;
            dec_total_us = (double)(dec_total_end - dec_total_start) / 1000.0;
        }

        /* Record */
        if (n == cap) {
            cap *= 2;
            comp_ktp = (double*)realloc(comp_ktp, cap * sizeof(double));
            dec_ktp = (double*)realloc(dec_ktp, cap * sizeof(double));
            comp_ttp = (double*)realloc(comp_ttp, cap * sizeof(double));
            dec_ttp = (double*)realloc(dec_ttp, cap * sizeof(double));
            ratio_pct = (double*)realloc(ratio_pct, cap * sizeof(double));
        }

        double in_mb = (double)in_size / (1024.0 * 1024.0);
        comp_ktp[n] = (comp_kernel_us > 0) ? (in_mb * 1e6 / (double)comp_kernel_us) : 0.0;
        dec_ktp[n] = (dec_kernel_us > 0) ? (in_mb * 1e6 / (double)dec_kernel_us) : 0.0;
        comp_ttp[n] = (tc.total_us > 0) ? (in_mb * 1e6 / (double)tc.total_us) : 0.0;
        dec_ttp[n] = (dec_total_us > 0.0) ? (in_mb * 1e6 / dec_total_us) : 0.0;
        ratio_pct[n] = (in_size > 0) ? (100.0 * (double)comp_total / (double)in_size) : 0.0;
        n++;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double sec = elapsed_sec(&ts0, &ts1);

    if (n > 0) {
        printf("Bench Compress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s ratio=%.2f%%\n",
               median_double(comp_ktp, n), median_double(comp_ttp, n), median_double(ratio_pct, n));
        printf("Bench Decompress : kernel_tp=%.2f MB/s total_tp=%.2f MB/s verify=%s\n",
               median_double(dec_ktp, n), median_double(dec_ttp, n), verify_ok ? "OK" : "FAIL");
        printf("Bench Summary : iterations=%zu seconds=%.2f\n", n, sec);
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_ktp); free(dec_ktp); free(comp_ttp); free(dec_ttp); free(ratio_pct);
    free(input_ref); free(lengths); free(out_buf);
    free(packed); free(offsets); free(dec_buf); free(out_lengths);
    if (include_file_io) {
        remove(tmp_comp_path);
        remove(tmp_dec_path);
    }
    return verify_ok ? 0 : 1;
}
