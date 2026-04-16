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
#include <inttypes.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <math.h>
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

#if defined(__GNUC__)
#define HYBRID_CORE_UNUSED __attribute__((unused))
#else
#define HYBRID_CORE_UNUSED
#endif

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

static size_t HYBRID_CORE_UNUSED hybrid_parse_size_bytes_env(const char* s, size_t defv) {
    char* end = NULL;
    unsigned long long v;
    if (!s || !*s) return defv;
    v = strtoull(s, &end, 10);
    if (!end || end == s) return defv;
    if (*end == 'k' || *end == 'K') v *= 1024ULL;
    else if (*end == 'm' || *end == 'M') v *= 1024ULL * 1024ULL;
    else if (*end == 'g' || *end == 'G') v *= 1024ULL * 1024ULL * 1024ULL;
    if (v == 0ULL) return defv;
    if (v > (unsigned long long)SIZE_MAX) return defv;
    return (size_t)v;
}

static size_t adaptive_min_block_size_bytes(const hybrid_params_t* params) {
    size_t v;
    if (!params) return 0;
    if (params->block_size > 0) return 0;
    if (params->cpu_threads <= 0) return 0;

    v = 240U * 1024U;
    if (v < LZO_MIN_BLOCK_BYTES_DEFAULT) v = LZO_MIN_BLOCK_BYTES_DEFAULT;
    if (v > LZO_MAX_BLOCK_BYTES_DEFAULT) v = LZO_MAX_BLOCK_BYTES_DEFAULT;
    return v;
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
static void lzo_hybrid_metrics_reset(void);
static void lzo_hybrid_metrics_record_adaptive(double ratio);

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
    if (f) { if (fscanf(f, "%" SCNu64, &val) != 1) val = 0; fclose(f); }
    return val;
}

static double lzo_read_env_double_or_neg(const char* name) {
    const char* v;
    char* endp = NULL;
    double out;
    if (!name) return -1.0;
    v = getenv(name);
    if (!v || !*v) return -1.0;
    out = strtod(v, &endp);
    if (!endp || endp == v) return -1.0;
    return out;
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
    if (sscanf(buf,
               "cpu %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64
               " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
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

static double lzo_read_cpu_freq_scale(void) {
    double max_f = lzo_read_sysfs_double("/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq");
    double cur = lzo_read_sysfs_double("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq");
    double target_mhz = -1.0;
    double target_pct = -1.0;
    double s;

    target_mhz = lzo_read_env_double_or_neg("LZO_HYBRID_CPU_FREQ_TARGET_MHZ");
    if (target_mhz <= 0.0) target_mhz = lzo_read_env_double_or_neg("HYBRID_CPU_FREQ_TARGET_MHZ");
    target_pct = lzo_read_env_double_or_neg("LZO_HYBRID_CPU_FREQ_TARGET_PCT");
    if (target_pct <= 0.0) target_pct = lzo_read_env_double_or_neg("HYBRID_CPU_FREQ_TARGET_PCT");

    if (target_mhz > 0.0 && max_f > 0.0) {
        s = (target_mhz * 1000.0) / max_f;
    } else if (target_pct > 0.0) {
        s = target_pct / 100.0;
    } else if (cur > 0.0 && max_f > 0.0) {
        s = cur / max_f;
    } else {
        s = 1.0;
    }

    {
        if (s < 0.30) s = 0.30;
        if (s > 1.20) s = 1.20;
        return s;
    }
}

static double lzo_read_gpu_freq_scale(void) {
    double cur = lzo_read_sysfs_double("/sys/class/drm/card0/gt_cur_freq_mhz");
    double max_f = lzo_read_sysfs_double("/sys/class/drm/card0/gt_max_freq_mhz");
    double target_mhz = -1.0;
    double target_pct = -1.0;
    double s;
    if (cur <= 0.0 || max_f <= 0.0) {
        cur = lzo_read_sysfs_double("/sys/class/drm/card1/gt_cur_freq_mhz");
        max_f = lzo_read_sysfs_double("/sys/class/drm/card1/gt_max_freq_mhz");
    }

    target_mhz = lzo_read_env_double_or_neg("LZO_HYBRID_GPU_FREQ_TARGET_MHZ");
    if (target_mhz <= 0.0) target_mhz = lzo_read_env_double_or_neg("HYBRID_GPU_FREQ_TARGET_MHZ");
    target_pct = lzo_read_env_double_or_neg("LZO_HYBRID_GPU_FREQ_TARGET_PCT");
    if (target_pct <= 0.0) target_pct = lzo_read_env_double_or_neg("HYBRID_GPU_FREQ_TARGET_PCT");

    if (target_mhz > 0.0 && max_f > 0.0) {
        s = target_mhz / max_f;
    } else if (target_pct > 0.0) {
        s = target_pct / 100.0;
    } else if (cur > 0.0 && max_f > 0.0) {
        s = cur / max_f;
    } else {
        s = 1.0;
    }

    {
        if (s < 0.30) s = 0.30;
        if (s > 1.20) s = 1.20;
        return s;
    }
}

static void lzo_adaptive_choose_objective_weights(double input_mb,
                                                  double sample_entropy,
                                                  double cpu_avail,
                                                  double gpu_avail,
                                                  double thread_util,
                                                  double cpu_freq_scale,
                                                  double gpu_freq_scale,
                                                  double* perf_weight_pct,
                                                  double* energy_weight_pct,
                                                  double* ratio_weight_pct,
                                                  double* dec_host_penalty_pct) {
    double perf = 66.0;
    double energy = 22.0;
    double ratio = 12.0;
    double dec_penalty = 3.5;

    if (input_mb <= 8.0) {
        perf += 8.0;
        energy -= 4.0;
        ratio -= 4.0;
    } else if (input_mb >= 256.0) {
        perf -= 2.0;
        energy += 4.0;
        ratio += 4.0;
    }

    if (sample_entropy > 6.5) {
        perf += 3.0;
        ratio -= 2.0;
    } else if (sample_entropy > 0.0 && sample_entropy < 4.8) {
        ratio += 5.0;
        perf -= 1.5;
    }

    if (thread_util >= 0.75) {
        energy += 6.0;
        perf -= 2.0;
    } else if (thread_util <= 0.35) {
        perf += 4.0;
        energy -= 2.0;
    }

    if (cpu_freq_scale < 0.80 && gpu_freq_scale > cpu_freq_scale) {
        perf += 4.0;
        energy += 3.0;
        ratio -= 2.0;
        dec_penalty -= 1.5;
    }
    if (gpu_freq_scale < 0.75 && cpu_freq_scale >= gpu_freq_scale) {
        perf -= 3.0;
        energy += 2.0;
        ratio += 2.0;
        dec_penalty += 2.0;
    }

    if (cpu_avail < 0.35) dec_penalty += 2.0;
    if (gpu_avail < 0.35) dec_penalty -= 1.0;

    if (ratio < 2.0) ratio = 2.0;
    if (energy < 10.0) energy = 10.0;
    if (perf < 20.0) perf = 20.0;
    if (dec_penalty < 0.0) dec_penalty = 0.0;
    if (dec_penalty > 14.0) dec_penalty = 14.0;

    {
        double sum = perf + energy + ratio;
        if (sum <= 0.0) {
            perf = 66.0;
            energy = 22.0;
            ratio = 12.0;
            sum = 100.0;
        }
        perf = perf * 100.0 / sum;
        energy = energy * 100.0 / sum;
        ratio = ratio * 100.0 / sum;
    }

    if (perf_weight_pct) *perf_weight_pct = perf;
    if (energy_weight_pct) *energy_weight_pct = energy;
    if (ratio_weight_pct) *ratio_weight_pct = ratio;
    if (dec_host_penalty_pct) *dec_host_penalty_pct = dec_penalty;
}

static double HYBRID_CORE_UNUSED lzo_adaptive_ratio_guard_cap(double mean_entropy,
                                                               long thread_count,
                                                               double thread_util,
                                                               double cpu_freq_scale,
                                                               double gpu_freq_scale,
                                                               size_t total_input_sz,
                                                               size_t nblk) {
    double cap;
    if (mean_entropy <= 0.0) return 0.85;

    if (mean_entropy < 4.2) cap = 0.55;
    else if (mean_entropy < 5.4) cap = 0.68;
    else if (mean_entropy < 6.4) cap = 0.78;
    else cap = 0.90;

    if (thread_util >= 0.75) cap += 0.08;
    else if (thread_util <= 0.35) cap += 0.03;

    if (cpu_freq_scale < 0.80 && gpu_freq_scale > cpu_freq_scale) cap += 0.10;
    if (gpu_freq_scale < 0.70 && cpu_freq_scale >= gpu_freq_scale) cap -= 0.10;

    if (thread_count >= 2) cap -= 0.02;
    if (thread_count >= 4) cap -= 0.05;
    if (thread_count >= 8) cap -= 0.03;

    if (thread_count >= 2 && nblk >= 512U) cap -= 0.03;
    if (thread_count >= 4 && nblk >= 1024U) cap -= 0.05;
    if (thread_count >= 2 && total_input_sz >= (128U * 1024U * 1024U)) cap -= 0.03;

    if (total_input_sz < (16U * 1024U * 1024U) || nblk < 64U) cap += 0.05;

    if (cap < 0.42) cap = 0.42;
    if (cap > 0.96) cap = 0.96;
    return cap;
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
    ws->adaptive_ratio_cache_comp_valid = 0;
    ws->adaptive_ratio_cache_comp = 0.0;
    ws->adaptive_ratio_cache_dec_valid = 0;
    ws->adaptive_ratio_cache_dec = 0.0;
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

static int zero_buffer_range(cl_command_queue q, cl_mem buf, size_t offset, size_t sz) {
    cl_uchar zero = 0;
    cl_int err;
    if (!buf || sz == 0) return 0;
    err = clEnqueueFillBuffer(q, buf, &zero, 1, offset, sz, 0, NULL, NULL);
    if (err == CL_SUCCESS) {
        clFinish(q);
        return 0;
    }
    return -1;
}

static size_t hybrid_adaptive_adjust_gpu_blocks(size_t nblk, double gpu_ratio, int* collapsed_out) {
    size_t gpu_blocks;
    const unsigned min_mixed = 8U;
    const unsigned quantum = 4U;
    const int collapse_small = 1;

    if (collapsed_out) *collapsed_out = 0;
    if (nblk == 0) return 0;
    if (gpu_ratio <= 0.0) return 0;
    if (gpu_ratio >= 1.0) return nblk;

    gpu_blocks = (size_t)((double)nblk * gpu_ratio + 0.5);
    if (gpu_blocks > nblk) gpu_blocks = nblk;

    if (min_mixed > 0) {
        if (nblk <= (size_t)min_mixed * 2U) {
            if (collapse_small && gpu_blocks > 0 && gpu_blocks < nblk) {
                gpu_blocks = (gpu_blocks * 2 >= nblk) ? nblk : 0;
                if (collapsed_out) *collapsed_out = 1;
            }
        } else {
            if (gpu_blocks > 0 && gpu_blocks < (size_t)min_mixed) gpu_blocks = (size_t)min_mixed;
            if (gpu_blocks < nblk && (nblk - gpu_blocks) < (size_t)min_mixed) {
                gpu_blocks = nblk - (size_t)min_mixed;
            }
        }
    }

    if (quantum > 1 && gpu_blocks > 0 && gpu_blocks < nblk) {
        size_t rounded = ((gpu_blocks + (size_t)quantum / 2U) / (size_t)quantum) * (size_t)quantum;
        if (min_mixed > 0 && rounded < (size_t)min_mixed) rounded = (size_t)min_mixed;
        if (min_mixed > 0 && rounded > nblk - (size_t)min_mixed) rounded = nblk - (size_t)min_mixed;
        if (rounded == 0) rounded = 1;
        if (rounded >= nblk) rounded = nblk - 1;
        gpu_blocks = rounded;
    }

    return gpu_blocks;
}

static int hybrid_should_use_device_compaction(size_t packed_bytes,
                                               size_t sparse_bytes,
                                               size_t num_blocks,
                                               cl_kernel pack_kernel) {
    const unsigned min_blocks = 8U;
    const unsigned min_gain_pct = 5U;

    if (!pack_kernel) return 0;
    if (packed_bytes == 0 || sparse_bytes == 0 || packed_bytes >= sparse_bytes) return 0;
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
    size_t block_index_base;
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

typedef struct {
    cpu_compress_pool_t* pool;
    void* wrkmem;
} cpu_compress_worker_ctx_t;

static void* cpu_compress_worker(void* arg) {
    cpu_compress_worker_ctx_t* ctx = (cpu_compress_worker_ctx_t*)arg;
    cpu_compress_pool_t* pool = ctx->pool;
    cpu_compress_job_t* job = pool->job;
    uint64_t t0 = hybrid_now_ns();

    /* Per-thread work memory comes from top-level pool allocation */
    void* wrkmem = ctx->wrkmem;
    if (!wrkmem) {
        job->rc = -1;
        return NULL;
    }

    if (job->block_indices) {
        for (;;) {
            size_t wi = atomic_fetch_add(&pool->next_idx, 1);
            size_t blk_idx;
            size_t offset;
            size_t this_blk;
            const unsigned char* src;
            unsigned char* dst;
            lzo_uint dst_len;
            int rc;

            if (wi >= job->num_assigned) break;

            blk_idx = job->block_indices[wi];
            offset = blk_idx * job->block_size;
            this_blk = job->block_size;
            if (offset + this_blk > job->in_size) this_blk = job->in_size - offset;

            src = job->input + offset;
            dst = job->output + blk_idx * job->worst_blk;
            dst_len = (lzo_uint)job->worst_blk;

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
    } else {
        const size_t base = job->block_index_base;
        for (;;) {
            size_t wi = atomic_fetch_add(&pool->next_idx, 1);
            size_t blk_idx;
            size_t offset;
            size_t this_blk;
            const unsigned char* src;
            unsigned char* dst;
            lzo_uint dst_len;
            int rc;

            if (wi >= job->num_assigned) break;

            blk_idx = base + wi;
            offset = blk_idx * job->block_size;
            this_blk = job->block_size;
            if (offset + this_blk > job->in_size) this_blk = job->in_size - offset;

            src = job->input + offset;
            dst = job->output + blk_idx * job->worst_blk;
            dst_len = (lzo_uint)job->worst_blk;

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

/* ---- CPU decompression worker ---- */

typedef struct {
    const unsigned char* comp_data;
    const uint32_t* offsets;
    const uint32_t* comp_lengths;
    unsigned char* output;
    size_t block_size;
    size_t orig_size;
    size_t nblk;
    const size_t* block_indices;
    size_t block_index_base;
    size_t num_assigned;
    int alg_id;
    int rc;
} cpu_decompress_job_t;

typedef struct {
    cpu_decompress_job_t* job;
    _Atomic size_t next_idx;
    _Atomic uint64_t max_worker_us;
} cpu_decompress_pool_t;

typedef struct {
    cpu_decompress_pool_t* pool;
    size_t thread_idx;
    size_t thread_count;
} cpu_decompress_worker_ctx_t;

static void* cpu_decompress_worker(void* arg) {
    cpu_decompress_worker_ctx_t* ctx = (cpu_decompress_worker_ctx_t*)arg;
    cpu_decompress_pool_t* pool = ctx->pool;
    cpu_decompress_job_t* job = pool->job;
    uint64_t t0 = hybrid_now_ns();
    const size_t begin = (ctx->thread_idx * job->num_assigned) / ctx->thread_count;
    const size_t end = ((ctx->thread_idx + 1) * job->num_assigned) / ctx->thread_count;

    if (job->block_indices) {
        for (size_t wi = begin; wi < end; ++wi) {
            size_t blk_idx;
            size_t out_offset;
            size_t this_blk;
            uint32_t comp_off;
            uint32_t comp_len;
            const unsigned char* src;
            unsigned char* dst;
            lzo_uint dst_len;
            int rc;

            blk_idx = job->block_indices[wi];
            out_offset = blk_idx * job->block_size;
            this_blk = job->block_size;
            if (out_offset + this_blk > job->orig_size) this_blk = job->orig_size - out_offset;

            comp_off = job->offsets[blk_idx];
            comp_len = job->comp_lengths[blk_idx];

            src = job->comp_data + comp_off;
            dst = job->output + out_offset;
            dst_len = (lzo_uint)this_blk;

            if (job->alg_id == 1) {
                rc = lzo1y_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
            } else {
                rc = lzo1x_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
            }

            if (rc != LZO_E_OK || dst_len != (lzo_uint)this_blk) {
                job->rc = -1;
            }
        }
    } else {
        const size_t base = job->block_index_base;
        for (size_t wi = begin; wi < end; ++wi) {
            size_t blk_idx;
            size_t out_offset;
            size_t this_blk;
            uint32_t comp_off;
            uint32_t comp_len;
            const unsigned char* src;
            unsigned char* dst;
            lzo_uint dst_len;
            int rc;

            blk_idx = base + wi;
            out_offset = blk_idx * job->block_size;
            this_blk = job->block_size;
            if (out_offset + this_blk > job->orig_size) this_blk = job->orig_size - out_offset;

            comp_off = job->offsets[blk_idx];
            comp_len = job->comp_lengths[blk_idx];

            src = job->comp_data + comp_off;
            dst = job->output + out_offset;
            dst_len = (lzo_uint)this_blk;

            if (job->alg_id == 1) {
                rc = lzo1y_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
            } else {
                rc = lzo1x_decompress_safe(src, (lzo_uint)comp_len, dst, &dst_len, NULL);
            }

            if (rc != LZO_E_OK || dst_len != (lzo_uint)this_blk) {
                job->rc = -1;
            }
        }
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

static void partition_blocks_prefix(size_t nblk, double gpu_ratio,
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

    for (size_t i = 0; i < gn; ++i) (*gpu_indices)[i] = i;
    for (size_t i = 0; i < cn; ++i) (*cpu_indices)[i] = gn + i;
}

static void partition_blocks(size_t nblk, double gpu_ratio,
                             size_t** gpu_indices, size_t* gpu_count,
                             size_t** cpu_indices, size_t* cpu_count) {
    partition_blocks_prefix(nblk, gpu_ratio, gpu_indices, gpu_count, cpu_indices, cpu_count);
}

static double lzo_clamp_double(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static double lzo_norm_from_bytes(double bytes, double ref_mb, double span) {
    double ref_bytes;
    double v;

    if (bytes <= 0.0) return 0.0;
    if (ref_mb <= 0.0) ref_mb = 1.0;
    if (span <= 0.0) span = 1.0;

    ref_bytes = ref_mb * 1024.0 * 1024.0;
    if (ref_bytes <= 0.0) ref_bytes = 1024.0 * 1024.0;

    v = log2(bytes / ref_bytes + 1.0) / span;
    return lzo_clamp_double(v, 0.0, 1.0);
}

static double lzo_cpu_thread_scale(long thread_count, int alg_id, int is_unified_memory) {
    double gain;
    double scale;
    double cap;

    if (thread_count <= 1) return 1.0;
    gain = (alg_id == 1) ? 0.46 : 0.56;
    if (!is_unified_memory) gain += 0.05;

    scale = 1.0 + (double)(thread_count - 1) * gain;
    cap = 1.0 + ((alg_id == 1) ? 1.35 : 1.55) * log2((double)thread_count + 1.0);
    if (scale > cap) scale = cap;
    if (scale < 1.0) scale = 1.0;
    return scale;
}

static double lzo_refine_ratio_candidate(const hybrid_params_t* params,
                                         size_t total_input_sz,
                                         size_t nblk,
                                         double Pc_eff,
                                         double Pg_eff,
                                         double t0,
                                         double cpu_freq_scale,
                                         double gpu_freq_scale,
                                         double thread_util,
                                         double mean_entropy,
                                         double seed_ratio,
                                         double min_ratio,
                                         double max_ratio) {
    static const double cands[] = {0.0, 0.04, 0.08, 0.12, 0.16, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.76, 0.82, 0.88, 0.94, 1.0};
    const size_t nc = sizeof(cands) / sizeof(cands[0]);
    const int is_1y = (params && params->alg_id == 1);
    double best_r = seed_ratio;
    double best_obj = 1e300;
    double B = (double)total_input_sz;
    double pc = (Pc_eff > 1.0) ? Pc_eff : 1.0;
    double pg = (Pg_eff > 1.0) ? Pg_eff : 1.0;
    double block_bytes;
    double size_norm;
    double block_norm;
    double ratio_norm;
    double compressibility;
    double device_adv;

    if (!params || total_input_sz == 0 || nblk == 0) return seed_ratio;
    seed_ratio = lzo_clamp_double(seed_ratio, 0.0, 1.0);
    min_ratio = lzo_clamp_double(min_ratio, 0.0, 1.0);
    max_ratio = lzo_clamp_double(max_ratio, 0.0, 1.0);
    if (max_ratio < min_ratio) max_ratio = min_ratio;

    block_bytes = B / (double)nblk;
    if (block_bytes < 4096.0) block_bytes = 4096.0;
    size_norm = lzo_norm_from_bytes(B, 1.0, 8.0);
    block_norm = lzo_norm_from_bytes(block_bytes, 0.0625, 2.5);
    ratio_norm = lzo_clamp_double(mean_entropy / 8.0, 0.0, 1.0);
    compressibility = 1.0 - ratio_norm;
    device_adv = (pg - pc) / (pg + pc);

    for (size_t i = 0; i < nc; ++i) {
        int collapsed = 0;
        size_t gpu_blocks = hybrid_adaptive_adjust_gpu_blocks(nblk, cands[i], &collapsed);
        size_t cpu_blocks = nblk - gpu_blocks;
        double r = (double)gpu_blocks / (double)nblk;
        double bytes_gpu;
        double bytes_cpu;
        double split_mix;
        double gpu_overhead_scale;
        double gpu_comp_overhead_s = 0.0;
        double gpu_dec_overhead_s = 0.0;
        double mix_host_s = 0.0;
        double comp_cpu_s;
        double comp_gpu_s;
        double comp_s;
        double dec_cpu_tp;
        double dec_gpu_tp;
        double dec_cpu_s;
        double dec_gpu_s;
        double dec_s;
        double load_risk_s = 0.0;
        double ratio_pen_s = 0.0;
        double smooth_pen;
        double direction_bonus;
        double aggressiveness_bonus = 0.0;
        double obj;

        if (r < min_ratio || r > max_ratio) {
            continue;
        }

        bytes_gpu = B * r;
        bytes_cpu = B - bytes_gpu;
        split_mix = 4.0 * r * (1.0 - r);

        gpu_overhead_scale = 0.90 + (1.0 - size_norm) * (1.05 - 0.30 * block_norm);
        if (is_1y) gpu_overhead_scale += 0.06;
        gpu_overhead_scale = lzo_clamp_double(gpu_overhead_scale, 0.45, 2.10);

        if (gpu_blocks > 0) {
            gpu_comp_overhead_s = t0 * gpu_overhead_scale;
            gpu_dec_overhead_s = 0.52 * t0 * gpu_overhead_scale;
        }

        comp_cpu_s = bytes_cpu / pc;
        comp_gpu_s = (gpu_blocks > 0) ? (gpu_comp_overhead_s + bytes_gpu / pg) : 0.0;
        comp_s = (gpu_blocks > 0 && cpu_blocks > 0)
            ? ((comp_cpu_s > comp_gpu_s) ? comp_cpu_s : comp_gpu_s)
            : (comp_cpu_s + comp_gpu_s);

        dec_cpu_tp = pc * ((is_1y ? 0.98 : 1.08) + 0.06 * (1.0 - size_norm));
        dec_gpu_tp = pg * ((is_1y ? 0.92 : 1.06) + 0.10 * gpu_freq_scale);
        if (dec_cpu_tp < 1.0) dec_cpu_tp = 1.0;
        if (dec_gpu_tp < 1.0) dec_gpu_tp = 1.0;
        dec_cpu_s = bytes_cpu / dec_cpu_tp;
        dec_gpu_s = (gpu_blocks > 0) ? (gpu_dec_overhead_s + bytes_gpu / dec_gpu_tp) : 0.0;
        dec_s = (gpu_blocks > 0 && cpu_blocks > 0)
            ? ((dec_cpu_s > dec_gpu_s) ? dec_cpu_s : dec_gpu_s)
            : (dec_cpu_s + dec_gpu_s);

        mix_host_s = split_mix * ((is_1y ? 0.00003 : 0.000025) +
                                  (1.0 - size_norm) * 0.00006 +
                                  (1.0 - block_norm) * 0.00004);

        if (thread_util < 0.45) {
            load_risk_s += split_mix * (0.45 - thread_util) * 0.00008;
        }
        if (cpu_freq_scale < 0.75 && cpu_blocks > 0) {
            load_risk_s += (0.75 - cpu_freq_scale) * (1.0 - r) * (is_1y ? 0.00008 : 0.00007);
        }
        if (gpu_freq_scale < 0.72 && gpu_blocks > 0) {
            load_risk_s += (0.72 - gpu_freq_scale) * r * 0.00006;
        }

        ratio_pen_s = split_mix * ((is_1y ? 0.00008 : 0.00006) * compressibility * r +
                                   0.00003 * ratio_norm * (1.0 - r));

        if (thread_util < 0.35 && r > 0.60) {
            aggressiveness_bonus = (0.35 - thread_util) * (r - 0.60) * (is_1y ? 0.00090 : 0.00110);
        }

        smooth_pen = fabs(r - seed_ratio) * 0.0025;
        direction_bonus = (is_1y ? 0.00016 : 0.00020) * device_adv * (r - 0.5);
        obj = 0.62 * comp_s + 0.38 * dec_s + mix_host_s + load_risk_s + ratio_pen_s + smooth_pen
            - direction_bonus - aggressiveness_bonus;
        if (collapsed) {
            obj += 0.00005;
        }

        if (obj < best_obj) {
            best_obj = obj;
            best_r = r;
        }
    }

    if (best_obj >= 1e299) {
        best_r = seed_ratio;
    }

    if (best_r < min_ratio) best_r = min_ratio;
    if (best_r > max_ratio) best_r = max_ratio;

    return best_r;
}

static double choose_adaptive_gpu_ratio(cl_context ctx, cl_command_queue queue,
                                        cl_device_id device, cl_kernel gpu_kernel,
                                        const unsigned char* input,
                                        size_t total_input_sz,
                                        size_t blk,
                                        size_t nblk,
                                        const hybrid_params_t* params,
                                        hybrid_workspace_t* ws) {
    double Pc0, Pg0, t0;
    double gC, gG;
    double sC, sG;
    double Pc_eff, Pg_eff;
    double r_star;
    double r_perf = 0.0;
    double r_energy = 0.0;
    double perf_weight_pct = 70.0;
    double energy_weight_pct = 30.0;
    double ratio_weight_pct = 0.0;
    double dec_host_penalty_pct = 0.0;
    double eC_eff = 0.0;
    double eG_eff = 0.0;
    double freq_mode_bias = 0.0;
    double split_mix_penalty = 0.0;
    double cpu_freq_scale = 1.0;
    double gpu_freq_scale = 1.0;
    double thread_util = 1.0;
    double cpu_thread_scale = 1.0;
    double cpu_thread_penalty = 1.0;
    double ratio_cap = 1.0;
    double min_ratio = 0.0;
    double max_ratio = 1.0;
    double block_bytes;
    double size_norm;
    double block_norm;
    double parallel_norm;
    double pg_parallel_scale;
    double low_thread_gpu_boost = 0.0;
    double mean_entropy = 0.0;
    double ratio_norm = 0.5;
    double compressibility = 0.5;
    double gpu_overhead_scale;
    double t0_eff;
    double device_adv;
    double center;
    double width;
    long thread_count;
    long total_cores;
    double B = (double)total_input_sz;
    size_t sample_blocks;
    size_t sampled = 0;
    double sample_cpu_throughput = 0.0;
    double cached_ratio;
    const int is_1y = (params && params->alg_id == 1);
    int trace_adaptive = 0;

    {
        const char* env_trace = getenv("LZO_HYBRID_TRACE_ADAPTIVE");
        if (env_trace && *env_trace && strcmp(env_trace, "0") != 0) {
            trace_adaptive = 1;
        }
    }

    if (!gpu_kernel) return 0.0;
    if (!params || nblk == 0 || blk == 0 || total_input_sz == 0) return 0.5;

    if (ws) {
        if (input && ws->adaptive_ratio_cache_comp_valid) {
            cached_ratio = lzo_clamp_double(ws->adaptive_ratio_cache_comp, 0.0, 1.0);
            if (params->debug) {
                fprintf(stderr, "Adaptive: reuse cached comp ratio r*=%.4f\n", cached_ratio);
            }
            lzo_hybrid_metrics_record_adaptive(cached_ratio);
            return cached_ratio;
        }
        if (!input && ws->adaptive_ratio_cache_dec_valid) {
            cached_ratio = lzo_clamp_double(ws->adaptive_ratio_cache_dec, 0.0, 1.0);
            if (params->debug) {
                fprintf(stderr, "Adaptive: reuse cached dec ratio r*=%.4f\n", cached_ratio);
            }
            return cached_ratio;
        }
    }

    lzo_calibrate_device_profile(ctx, queue, device, gpu_kernel, params, ws);
    Pc0 = g_lzo_dev_profile.cpu_throughput;
    Pg0 = g_lzo_dev_profile.gpu_throughput;
    t0 = g_lzo_dev_profile.gpu_overhead_s;

    if (Pc0 > 0.0 && Pg0 > 0.0) {
        double rel = Pc0 / Pg0;
        if (rel > 3.2) Pc0 = Pg0 * 3.2;
        if (rel < 0.22) Pc0 = Pg0 * 0.22;
    }

    total_cores = get_online_cpu_count();
    if (total_cores <= 0) total_cores = 4;
    thread_count = (params->cpu_threads > 0) ? params->cpu_threads : total_cores;
    cpu_thread_scale = lzo_cpu_thread_scale(thread_count,
                                            params->alg_id,
                                            g_lzo_dev_profile.is_unified_memory);

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
            size_t wrkmem_sz = is_1y ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
            unsigned char* tmp_out = (unsigned char*)malloc(lzo_worst_size(blk));
            wrkmem = malloc(wrkmem_sz);

            if (tmp_out && wrkmem) {
                uint64_t sample_t0;
                memset(wrkmem, 0, wrkmem_sz);
                sample_t0 = hybrid_now_ns();

                for (size_t i = 0; i < sample_blocks; ++i) {
                    size_t blk_idx = sampled_block_index(i, sample_blocks, nblk);
                    size_t blk_sz = block_input_size(total_input_sz, blk, blk_idx);
                    if (blk_idx == prev_block || blk_sz == 0) continue;

                    mean_entropy += lzo_calc_entropy(input + blk_idx * blk, blk_sz);
                    sampled++;
                    prev_block = blk_idx;

                    {
                        lzo_uint out_len = (lzo_uint)lzo_worst_size(blk);
                        if (is_1y)
                            lzo1y_1_compress(input + blk_idx * blk, (lzo_uint)blk_sz, tmp_out, &out_len, wrkmem);
                        else
                            lzo1x_1_compress(input + blk_idx * blk, (lzo_uint)blk_sz, tmp_out, &out_len, wrkmem);
                        sample_bytes += blk_sz;
                    }
                }

                if (sample_bytes > 0) {
                    uint64_t sample_t1 = hybrid_now_ns();
                    if (sample_t1 > sample_t0)
                        sample_cpu_throughput = (double)sample_bytes / ((double)(sample_t1 - sample_t0) * 1e-9);
                }
            }
            free(tmp_out);
            free(wrkmem);
        }

        if (sampled > 0) {
            mean_entropy /= (double)sampled;
            ratio_norm = lzo_clamp_double(mean_entropy / 8.0, 0.0, 1.0);
            compressibility = 1.0 - ratio_norm;
        }

        if (sample_cpu_throughput > 0.0 && Pc0 > 0.0) {
            gC = lzo_clamp_double(sample_cpu_throughput / Pc0, 0.65, 1.45);
        }
        gG = lzo_clamp_double(1.0 + 0.22 * (0.5 - ratio_norm), 0.68, 1.30);
        if (is_1y) gG *= 0.96;
        gG = lzo_clamp_double(gG, 0.62, 1.28);
    }

    sC = lzo_read_cpu_availability();
    sG = lzo_read_gpu_availability();
    if (params->cpu_threads > 0 && params->cpu_threads < total_cores) {
        double scale = (double)total_cores / (double)params->cpu_threads;
        sC = sC * scale;
        if (sC > 1.0) sC = 1.0;
    }

    thread_util = (total_cores > 0) ? ((double)thread_count / (double)total_cores) : 1.0;
    thread_util = lzo_clamp_double(thread_util, 0.05, 1.25);

    cpu_freq_scale = lzo_read_cpu_freq_scale();
    gpu_freq_scale = lzo_read_gpu_freq_scale();

    block_bytes = (blk > 0) ? (double)blk : (B / (double)nblk);
    if (block_bytes < 4096.0) block_bytes = 4096.0;
    size_norm = lzo_norm_from_bytes(B, 1.0, 8.0);
    block_norm = lzo_norm_from_bytes(block_bytes, 0.0625, 2.5);
    parallel_norm = 1.0 - exp(-(double)nblk / 192.0);
    pg_parallel_scale = 0.55 + 2.45 * parallel_norm;

    if (thread_count <= 2) {
        cpu_thread_penalty -= (0.26 + (is_1y ? 0.10 : 0.08) * (1.0 - size_norm)) * (1.0 - thread_util);
    }
    if (thread_count == 1) {
        cpu_thread_penalty -= (0.10 + 0.10 * size_norm) * parallel_norm;
    }
    cpu_thread_penalty = lzo_clamp_double(cpu_thread_penalty, 0.35, 1.08);

    gpu_overhead_scale = 0.90 + (1.0 - size_norm) * (1.05 - 0.30 * block_norm);
    if (is_1y) gpu_overhead_scale += 0.06;
    gpu_overhead_scale = lzo_clamp_double(gpu_overhead_scale, 0.45, 2.10);
    t0_eff = t0 * gpu_overhead_scale;

    Pc_eff = Pc0 * gC * sC * cpu_thread_scale * cpu_freq_scale * cpu_thread_penalty;
    Pg_eff = Pg0 * gG * sG * gpu_freq_scale * (0.74 + 0.24 * size_norm + 0.14 * block_norm) * pg_parallel_scale;
    if (is_1y) Pg_eff *= 0.95;
    if (Pc_eff < 1.0) Pc_eff = 1.0;
    if (Pg_eff < 1.0) Pg_eff = 1.0;

    if (Pc_eff + Pg_eff <= 0.0) return 0.5;
    r_star = Pg_eff / (Pc_eff + Pg_eff);
    if (B > 0.0 && t0_eff > 0.0) {
        r_star -= (t0_eff * Pc_eff * Pg_eff) / (B * (Pc_eff + Pg_eff));
    }

    low_thread_gpu_boost = (1.0 - thread_util) * parallel_norm * (is_1y ? 0.28 : 0.36) * (0.62 + 0.38 * size_norm);
    r_star += low_thread_gpu_boost;
    r_perf = r_star;

    {
        double input_mb = (double)total_input_sz / (1024.0 * 1024.0);
        lzo_adaptive_choose_objective_weights(input_mb,
                                              mean_entropy,
                                              sC,
                                              sG,
                                              thread_util,
                                              cpu_freq_scale,
                                              gpu_freq_scale,
                                              &perf_weight_pct,
                                              &energy_weight_pct,
                                              &ratio_weight_pct,
                                              &dec_host_penalty_pct);
    }

    {
        double eC = g_lzo_dev_profile.cpu_energy_per_byte;
        double eG = g_lzo_dev_profile.gpu_energy_per_byte;
        if (eC > 0.0 && eG > 0.0) {
            const double cpu_dyn_power = 0.75 + 0.60 * thread_util +
                0.55 * (cpu_freq_scale * cpu_freq_scale * cpu_freq_scale);
            const double gpu_dyn_power = 0.85 + 0.50 * gpu_freq_scale + 0.15 * (1.0 - sG);
            double sum_w;

            eC_eff = eC * cpu_dyn_power / (cpu_freq_scale > 0.30 ? cpu_freq_scale : 0.30);
            eG_eff = eG * gpu_dyn_power / (gpu_freq_scale > 0.30 ? gpu_freq_scale : 0.30);

            {
                double denom = Pc_eff * eG_eff + Pg_eff * eC_eff;
                r_energy = (denom > 0.0) ? ((Pg_eff * eC_eff) / denom) : r_perf;
            }

            sum_w = perf_weight_pct + energy_weight_pct;
            if (sum_w <= 0.0) {
                perf_weight_pct = 62.0;
                energy_weight_pct = 28.0;
                sum_w = perf_weight_pct + energy_weight_pct;
            }
            r_star = (perf_weight_pct * r_perf + energy_weight_pct * r_energy) / sum_w;
        }
    }

    freq_mode_bias = 0.07 * (gpu_freq_scale - cpu_freq_scale);
    if (thread_util >= 0.80) freq_mode_bias += 0.02;
    r_star += freq_mode_bias;

    if (sampled > 0 && ratio_weight_pct > 0.0) {
        double ratio_shift = (ratio_weight_pct / 100.0) *
                             ((is_1y ? 0.16 : 0.13) * compressibility - 0.05 * ratio_norm);
        r_star -= ratio_shift * (0.30 + 0.70 * size_norm);
    }

    if (!input && dec_host_penalty_pct > 0.0) {
        double host_pen = (dec_host_penalty_pct / 100.0) * (0.55 + 0.45 * (1.0 - size_norm));
        host_pen = lzo_clamp_double(host_pen, 0.0, 0.18);
        r_star *= (1.0 - host_pen);
    }

    device_adv = (Pg_eff - Pc_eff) / (Pg_eff + Pc_eff);
    center = 0.50 + 0.45 * device_adv + 0.05 * (gpu_freq_scale - cpu_freq_scale);
    center += 0.55 * low_thread_gpu_boost;
    if (is_1y) center += 0.02;
    if (sampled > 0) center += 0.04 * (0.5 - ratio_norm);
    center = lzo_clamp_double(center, 0.02, 0.98);

    width = 0.15 + 0.25 * size_norm + 0.11 * (1.0 - fabs(device_adv));
    width += 0.10 * low_thread_gpu_boost;
    if (is_1y) width -= 0.02;
    width -= 0.05 * (1.0 - block_norm);
    width = lzo_clamp_double(width, 0.10, 0.44);

    min_ratio = lzo_clamp_double(center - width, 0.0, 0.95);
    max_ratio = lzo_clamp_double(center + width, 0.05, 1.0);
    if (max_ratio < min_ratio) max_ratio = min_ratio;
    ratio_cap = max_ratio;

    r_star = lzo_clamp_double(r_star, min_ratio, max_ratio);

    if (input) {
        split_mix_penalty = 4.0 * r_star * (1.0 - r_star);
        split_mix_penalty *= (0.02 + 0.04 * (1.0 - size_norm));
        split_mix_penalty = lzo_clamp_double(split_mix_penalty, 0.0, 0.08);
        r_star *= (1.0 - split_mix_penalty * 0.20);
    }

    {
        double r_before_refine = r_star;
        double r_refined = lzo_refine_ratio_candidate(params,
                                                      total_input_sz,
                                                      nblk,
                                                      Pc_eff,
                                                      Pg_eff,
                                                      t0_eff,
                                                      cpu_freq_scale,
                                                      gpu_freq_scale,
                                                      thread_util,
                                                      mean_entropy,
                                                      r_before_refine,
                                                      min_ratio,
                                                      max_ratio);
        r_star = 0.70 * r_before_refine + 0.30 * r_refined;
        if (input && thread_util < 0.35 && r_before_refine > r_star) {
            r_star += 0.35 * (r_before_refine - r_star);
        }
    }

    r_star = lzo_clamp_double(r_star, min_ratio, max_ratio);
    r_star = lzo_clamp_double(r_star, 0.0, 1.0);

    if (params->debug || trace_adaptive) {
        fprintf(stderr,
                "Adaptive(E2E): Pc0=%.0f gC=%.2f sC=%.2f cpuF=%.2f threads=%ld cpuScale=%.2f cpuPen=%.2f util=%.2f Pc_eff=%.0f | "
                "Pg0=%.0f gG=%.2f sG=%.2f gpuF=%.2f Pg_eff=%.0f | "
                "t0=%.6f t0eff=%.6f B=%.0f entropy=%.2f sizeN=%.2f blockN=%.2f parN=%.2f pScale=%.2f ratioN=%.2f "
                "perfW=%.1f energyW=%.1f ratioW=%.1f decHostPen=%.1f "
                "rPerf=%.4f rEnergy=%.4f lowBoost=%.4f freqBias=%.4f splitPen=%.4f ratioCap=%.4f eCeff=%.2e eGeff=%.2e "
                "center=%.4f width=%.4f min=%.4f max=%.4f r*=%.4f\n",
                Pc0, gC, sC, cpu_freq_scale, thread_count, cpu_thread_scale, cpu_thread_penalty, thread_util, Pc_eff,
                Pg0, gG, sG, gpu_freq_scale, Pg_eff,
                t0, t0_eff, B, mean_entropy, size_norm, block_norm, parallel_norm, pg_parallel_scale, ratio_norm,
                perf_weight_pct, energy_weight_pct, ratio_weight_pct, dec_host_penalty_pct,
                r_perf, r_energy, low_thread_gpu_boost, freq_mode_bias, split_mix_penalty, ratio_cap, eC_eff, eG_eff,
                center, width, min_ratio, max_ratio, r_star);
    }

    if (input) {
        lzo_hybrid_metrics_record_adaptive(r_star);
    }

    if (ws) {
        if (input) {
            ws->adaptive_ratio_cache_comp = r_star;
            ws->adaptive_ratio_cache_comp_valid = 1;
        } else {
            ws->adaptive_ratio_cache_dec = r_star;
            ws->adaptive_ratio_cache_dec_valid = 1;
        }
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
    pthread_t cpu_tids_stack[64];
    int cpu_tids_heap = 0;
    cpu_compress_worker_ctx_t* cpu_ctx = NULL;
    cpu_compress_worker_ctx_t cpu_ctx_stack[64];
    int cpu_ctx_heap = 0;
    void* cpu_wrkmem_blob = NULL;
    size_t cpu_wrkmem_sz = (params->alg_id == 1) ? LZO1Y_MEM_COMPRESS : LZO1X_1_MEM_COMPRESS;
    int force_cpu_only = 0;
    int force_gpu_only = 0;
    int adaptive_collapsed_split = 0;
    int use_prefix_split = 1;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    if (params->split_mode == HYBRID_SPLIT_ADAPTIVE) {
        gpu_ratio = choose_adaptive_gpu_ratio(ctx, queue, device, gpu_kernel, input_buf, in_sz, blk, nblk, params, ws);
        {
            size_t adjusted_gpu_blocks = hybrid_adaptive_adjust_gpu_blocks(nblk, gpu_ratio, &adaptive_collapsed_split);
            gpu_ratio = (nblk > 0) ? ((double)adjusted_gpu_blocks / (double)nblk) : 0.0;
        }
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
    if (params->debug && params->split_mode == HYBRID_SPLIT_ADAPTIVE && adaptive_collapsed_split) {
        fprintf(stderr,
                "Adaptive split: mixed block count too small, collapsed to %s-only\n",
                gpu_count == 0 ? "CPU" : "GPU");
    }

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
        .block_index_base = 0,
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
    if (n_cpu_threads > 0) {
        cpu_wrkmem_blob = calloc((size_t)n_cpu_threads, cpu_wrkmem_sz);
        if (!cpu_wrkmem_blob) goto fail;

        if (n_cpu_threads <= (int)(sizeof(cpu_ctx_stack) / sizeof(cpu_ctx_stack[0]))) {
            cpu_ctx = cpu_ctx_stack;
        } else {
            cpu_ctx = (cpu_compress_worker_ctx_t*)malloc((size_t)n_cpu_threads * sizeof(cpu_compress_worker_ctx_t));
            cpu_ctx_heap = 1;
        }
        if (!cpu_ctx) goto fail;

        if (n_cpu_threads == 1) {
            cpu_ctx[0].pool = &cpu_pool;
            cpu_ctx[0].wrkmem = (unsigned char*)cpu_wrkmem_blob;
            cpu_compress_worker(&cpu_ctx[0]);
            timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
        } else {
            if (n_cpu_threads <= (int)(sizeof(cpu_tids_stack) / sizeof(cpu_tids_stack[0]))) {
                cpu_tids = cpu_tids_stack;
            } else {
                cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
                cpu_tids_heap = 1;
            }
            if (!cpu_tids) goto fail;
            for (int i = 0; i < n_cpu_threads; i++) {
                cpu_ctx[i].pool = &cpu_pool;
                cpu_ctx[i].wrkmem = (unsigned char*)cpu_wrkmem_blob + (size_t)i * cpu_wrkmem_sz;
                pthread_create(&cpu_tids[i], NULL, cpu_compress_worker, &cpu_ctx[i]);
            }
        }
    }

    /* GPU compression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;
    unsigned long pack_kernel_us = 0;

    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();
        const unsigned char* gpu_upload_ptr;
        size_t gpu_input_sz;
        int gpu_upload_owned = 0;
        int can_skip_upload = (skip_upload && force_gpu_only && cpu_count == 0);
        int gpu_prefix_mapping = (use_prefix_split && gpu_idx == NULL);

        if (cpu_count == 0) {
            gpu_upload_ptr = input_buf;
            gpu_input_sz = in_sz;
            gpu_upload_owned = 0;
        } else if (gpu_prefix_mapping) {
            gpu_upload_ptr = input_buf;
            gpu_input_sz = gpu_count * blk;
            gpu_upload_owned = 0;
        } else {
            unsigned char* gpu_input = (unsigned char*)calloc(gpu_count, blk);
            if (!gpu_input) goto fail;
            gpu_input_sz = gpu_count * blk;
            gpu_upload_owned = 1;
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
            if (err != CL_SUCCESS) { if (gpu_upload_owned) free((void*)(uintptr_t)gpu_upload_ptr); goto fail; }
            err = clEnqueueWriteBuffer(queue, ws->d_in, CL_TRUE, 0, gpu_input_sz, gpu_upload_ptr, 0, NULL, NULL);
        } else {
            err = CL_SUCCESS;
        }
        if (gpu_upload_owned) free((void*)(uintptr_t)gpu_upload_ptr);
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
        if (is_999) {
            zero_buffer(queue, ws->d_dict, ws->dict_cap);
        } else if (ws->dict_cap > prev_dict) {
            zero_buffer_range(queue, ws->d_dict, prev_dict, ws->dict_cap - prev_dict);
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
        if (gpu_idx) {
            for (size_t i = 0; i < gpu_count; i++) {
                size_t blk_idx = gpu_idx[i];
                lengths[blk_idx] = ((cl_uint*)mapped_len)[i];
            }
        } else {
            memcpy(lengths, mapped_len, gpu_count * sizeof(cl_uint));
        }
        clEnqueueUnmapMemObject(queue, ws->d_len, mapped_len, 0, NULL, NULL);

        {
            size_t packed_total = 0;
            int use_compaction;
            const int pack_zerocopy_prefix = 1;
            if (gpu_idx) {
                for (size_t i = 0; i < gpu_count; i++) {
                    size_t blk_idx = gpu_idx[i];
                    packed_total += (size_t)lengths[blk_idx];
                }
            } else {
                for (size_t i = 0; i < gpu_count; i++) {
                    packed_total += (size_t)lengths[i];
                }
            }

            use_compaction = hybrid_should_use_device_compaction(
                packed_total,
                gpu_count * worst_blk,
                gpu_count,
                pack_kernel
            );
            if (pack_zerocopy_prefix && gpu_prefix_mapping) {
                use_compaction = 0;
            }

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
                        if (gpu_idx) {
                            for (size_t i = 0; i < gpu_count; i++) {
                                size_t blk_idx = gpu_idx[i];
                                ((cl_uint*)mapped_off)[i] = (cl_uint)packed_offset;
                                packed_offset += (size_t)lengths[blk_idx];
                            }
                        } else {
                            for (size_t i = 0; i < gpu_count; i++) {
                                ((cl_uint*)mapped_off)[i] = (cl_uint)packed_offset;
                                packed_offset += (size_t)lengths[i];
                            }
                        }
                        err = clEnqueueUnmapMemObject(queue, ws->d_packed_off, mapped_off, 0, NULL, NULL);
                        if (err != CL_SUCCESS) goto fail;
                    }
                    timing.upload_us += (unsigned long)((hybrid_now_ns() - t_pack_up0) / 1000);
                }

                {
                    cl_uint worst_blk_cl = (cl_uint)worst_blk;
                    cl_uint gpu_count_cl = (cl_uint)gpu_count;
                    size_t pack_local = 64U;
                    size_t pack_global;
                    if (pack_local == 0) pack_local = 1;
                    if (pack_local > 256) pack_local = 256;
                    if (pack_local > gpu_count && gpu_count > 0) {
                        size_t p2 = 1;
                        while ((p2 << 1) <= gpu_count) p2 <<= 1;
                        pack_local = p2;
                    }
                    pack_global = gpu_count ? (gpu_count * pack_local) : pack_local;
                    uint64_t t_pack_k0 = hybrid_now_ns();
                    err  = clSetKernelArg(pack_kernel, 0, sizeof(cl_mem), &ws->d_out);
                    err |= clSetKernelArg(pack_kernel, 1, sizeof(cl_mem), &ws->d_len);
                    err |= clSetKernelArg(pack_kernel, 2, sizeof(cl_mem), &ws->d_packed_out);
                    err |= clSetKernelArg(pack_kernel, 3, sizeof(cl_mem), &ws->d_packed_off);
                    err |= clSetKernelArg(pack_kernel, 4, sizeof(cl_uint), &worst_blk_cl);
                    err |= clSetKernelArg(pack_kernel, 5, sizeof(cl_uint), &gpu_count_cl);
                    if (err != CL_SUCCESS) goto fail;
                    err = clEnqueueNDRangeKernel(queue, pack_kernel, 1, NULL, &pack_global, &pack_local, 0, NULL, NULL);
                    if (err != CL_SUCCESS) goto fail;
                    clFinish(queue);
                    pack_kernel_us = (unsigned long)((hybrid_now_ns() - t_pack_k0) / 1000);
                }

                {
                    void* mapped_packed = clEnqueueMapBuffer(queue, ws->d_packed_out, CL_TRUE, CL_MAP_READ,
                                                             0, packed_total, 0, NULL, NULL, &err);
                    if (err != CL_SUCCESS) goto fail;
                    size_t packed_offset = 0;
                    if (gpu_idx) {
                        for (size_t i = 0; i < gpu_count; i++) {
                            size_t blk_idx = gpu_idx[i];
                            memcpy(out_buf + blk_idx * worst_blk,
                                   (unsigned char*)mapped_packed + packed_offset,
                                   (size_t)lengths[blk_idx]);
                            packed_offset += (size_t)lengths[blk_idx];
                        }
                    } else {
                        memcpy(out_buf, mapped_packed, packed_total);
                    }
                    clEnqueueUnmapMemObject(queue, ws->d_packed_out, mapped_packed, 0, NULL, NULL);
                }
            } else {
                if (pack_zerocopy_prefix && gpu_prefix_mapping) {
                    err = clEnqueueReadBuffer(queue,
                                              ws->d_out,
                                              CL_TRUE,
                                              0,
                                              gpu_count * worst_blk,
                                              out_buf,
                                              0,
                                              NULL,
                                              NULL);
                    if (err != CL_SUCCESS) goto fail;
                } else {
                    void* mapped_out = clEnqueueMapBuffer(queue, ws->d_out, CL_TRUE, CL_MAP_READ,
                                                           0, gpu_count * worst_blk, 0, NULL, NULL, &err);
                    if (err != CL_SUCCESS) goto fail;
                    if (gpu_idx) {
                        for (size_t i = 0; i < gpu_count; i++) {
                            size_t blk_idx = gpu_idx[i];
                            memcpy(out_buf + blk_idx * worst_blk,
                                   (unsigned char*)mapped_out + i * worst_blk,
                                   (size_t)lengths[blk_idx]);
                        }
                    } else {
                        memcpy(out_buf, mapped_out, gpu_count * worst_blk);
                    }
                    clEnqueueUnmapMemObject(queue, ws->d_out, mapped_out, 0, NULL, NULL);
                }
            }
        }
        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU workers */
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++)
            pthread_join(cpu_tids[i], NULL);
        if (cpu_tids_heap) free(cpu_tids);
        cpu_tids = NULL;
        timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
    }

    free(cpu_wrkmem_blob);
    if (cpu_ctx_heap) free(cpu_ctx);
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
        if (cpu_tids_heap) free(cpu_tids);
    }
    free(cpu_wrkmem_blob);
    if (cpu_ctx_heap) free(cpu_ctx);
    return -1;
}

/* ---- internal buffer-based decompress (no file I/O) ---- */

static int hybrid_decompress_buf(
    cl_context ctx, cl_command_queue queue,
    cl_kernel gpu_kernel,
    const unsigned char* packed, const uint32_t* offsets, const uint32_t* comp_lengths,
    size_t packed_sz, uint32_t orig_sz, uint32_t blk_sz, uint32_t nblk, int alg_id,
    unsigned char* output_buf,
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
    pthread_t cpu_tids_stack[64];
    int cpu_tids_heap = 0;
    cpu_decompress_worker_ctx_t* cpu_ctx = NULL;
    cpu_decompress_worker_ctx_t cpu_ctx_stack[64];
    int cpu_ctx_heap = 0;
    int force_cpu_only = 0;
    int force_gpu_only = 0;
    int adaptive_collapsed_split = 0;
    int use_prefix_split = 1;
    if (params->cpu_threads <= 0) gpu_ratio = 1.0;
    if (params->split_mode == HYBRID_SPLIT_ADAPTIVE) {
        gpu_ratio = choose_adaptive_gpu_ratio(ctx, queue, 0, gpu_kernel, NULL, orig_sz, blk_sz, nblk, params, ws);
        {
            size_t adjusted_gpu_blocks = hybrid_adaptive_adjust_gpu_blocks(nblk, gpu_ratio, &adaptive_collapsed_split);
            gpu_ratio = (nblk > 0) ? ((double)adjusted_gpu_blocks / (double)nblk) : 0.0;
        }
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
    if (params->debug && params->split_mode == HYBRID_SPLIT_ADAPTIVE && adaptive_collapsed_split) {
        fprintf(stderr,
                "Adaptive split(decompress): mixed block count too small, collapsed to %s-only\n",
                gpu_count == 0 ? "CPU" : "GPU");
    }

    /* Launch CPU decompression */
    cpu_decompress_job_t cpu_job = {
        .comp_data = packed,
        .offsets = offsets,
        .comp_lengths = comp_lengths,
        .output = output_buf,
        .block_size = blk_sz,
        .orig_size = orig_sz,
        .nblk = nblk,
        .block_indices = cpu_idx,
        .block_index_base = 0,
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
    if (n_cpu_threads > 0) {
        if (n_cpu_threads <= (int)(sizeof(cpu_ctx_stack) / sizeof(cpu_ctx_stack[0]))) {
            cpu_ctx = cpu_ctx_stack;
        } else {
            cpu_ctx = (cpu_decompress_worker_ctx_t*)malloc((size_t)n_cpu_threads * sizeof(cpu_decompress_worker_ctx_t));
            cpu_ctx_heap = 1;
        }
        if (!cpu_ctx) goto dfail;

        if (n_cpu_threads == 1) {
            cpu_ctx[0].pool = &cpu_pool;
            cpu_ctx[0].thread_idx = 0;
            cpu_ctx[0].thread_count = 1;
            cpu_decompress_worker(&cpu_ctx[0]);
            timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
        } else {
            if (n_cpu_threads <= (int)(sizeof(cpu_tids_stack) / sizeof(cpu_tids_stack[0]))) {
                cpu_tids = cpu_tids_stack;
            } else {
                cpu_tids = (pthread_t*)malloc(n_cpu_threads * sizeof(pthread_t));
                cpu_tids_heap = 1;
            }
            if (!cpu_tids) goto dfail;
            for (int i = 0; i < n_cpu_threads; i++) {
                cpu_ctx[i].pool = &cpu_pool;
                cpu_ctx[i].thread_idx = (size_t)i;
                cpu_ctx[i].thread_count = (size_t)n_cpu_threads;
                pthread_create(&cpu_tids[i], NULL, cpu_decompress_worker, &cpu_ctx[i]);
            }
        }
    }

    /* GPU decompression */
    uint64_t t_gpu_k0 = 0, t_gpu_k1 = 0;
    if (gpu_count > 0) {
        uint64_t t_up0 = hybrid_now_ns();
        uint32_t* gpu_offsets = NULL;
        uint32_t* gpu_comp_lens = NULL;
        const uint32_t* gpu_offsets_src = NULL;
        const uint32_t* gpu_comp_lens_src = NULL;
        const unsigned char* gpu_comp_src = packed;
        size_t gpu_packed_sz = packed_sz;
        size_t gpu_orig_sz = gpu_count * blk_sz;
        if (gpu_idx) {
            for (size_t i = 0; i < gpu_count; ++i) {
                size_t blk_idx = gpu_idx[i];
                if (blk_idx == nblk - 1) gpu_orig_sz = i * blk_sz + block_input_size(orig_sz, blk_sz, blk_idx);
            }
        } else if (gpu_count == (size_t)nblk && nblk > 0) {
            gpu_orig_sz = (size_t)(nblk - 1) * blk_sz + block_input_size(orig_sz, blk_sz, nblk - 1);
        }
        if (use_prefix_split) {
            gpu_offsets_src = offsets;
            gpu_comp_lens_src = comp_lengths;
            if (gpu_count < nblk) {
                gpu_packed_sz = (size_t)offsets[gpu_count];
            }
        } else {
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
            gpu_offsets_src = gpu_offsets;
            gpu_comp_lens_src = gpu_comp_lens;
        }

        grow_buffer(ctx, &ws->d_comp, &ws->comp_cap, gpu_packed_sz ? gpu_packed_sz : 1, CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_off, &ws->off_cap, gpu_count * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_comp_lens, &ws->comp_lens_cap, gpu_count * sizeof(cl_uint), CL_MEM_READ_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_decomp_out, &ws->decomp_out_cap, gpu_orig_sz ? gpu_orig_sz : 1, CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }
        grow_buffer(ctx, &ws->d_out_lens, &ws->out_lens_cap, gpu_count * sizeof(cl_uint), CL_MEM_WRITE_ONLY, &err);
        if (err != CL_SUCCESS) { free(gpu_offsets); free(gpu_comp_lens); goto dfail; }

        err = clEnqueueWriteBuffer(queue, ws->d_comp, CL_TRUE, 0, gpu_packed_sz ? gpu_packed_sz : 1, gpu_comp_src, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, ws->d_off, CL_TRUE, 0, gpu_count * sizeof(cl_uint), gpu_offsets_src, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, ws->d_comp_lens, CL_TRUE, 0, gpu_count * sizeof(cl_uint), gpu_comp_lens_src, 0, NULL, NULL);
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
        if (use_prefix_split) {
            err = clEnqueueReadBuffer(queue,
                                      ws->d_decomp_out,
                                      CL_TRUE,
                                      0,
                                      gpu_out_sz,
                                      output_buf,
                                      0,
                                      NULL,
                                      NULL);
            if (err != CL_SUCCESS) goto dfail;

            for (size_t i = 0; i < gpu_count; i++) {
                size_t off = i * blk_sz;
                size_t this_blk = blk_sz;
                if (off + this_blk > orig_sz) this_blk = orig_sz - off;
            }
        } else {
            void* mapped_out = NULL;
            err = CL_SUCCESS;
            if (gpu_out_sz > 0) {
                mapped_out = clEnqueueMapBuffer(queue,
                                                ws->d_decomp_out,
                                                CL_TRUE,
                                                CL_MAP_READ,
                                                0,
                                                gpu_out_sz,
                                                0,
                                                NULL,
                                                NULL,
                                                &err);
                if (err != CL_SUCCESS || mapped_out == NULL) {
                    mapped_out = NULL;
                }
            }

            if (mapped_out) {
                if (gpu_idx) {
                    for (size_t i = 0; i < gpu_count; i++) {
                        size_t bi = gpu_idx[i];
                        size_t off = bi * blk_sz;
                        size_t this_blk = blk_sz;
                        if (off + this_blk > orig_sz) this_blk = orig_sz - off;
                        memcpy(output_buf + off, (const unsigned char*)mapped_out + i * blk_sz, this_blk);
                    }
                } else {
                    memcpy(output_buf, mapped_out, gpu_out_sz);
                    for (size_t i = 0; i < gpu_count; i++) {
                        size_t off = i * blk_sz;
                        size_t this_blk = blk_sz;
                        if (off + this_blk > orig_sz) this_blk = orig_sz - off;
                    }
                }
                err = clEnqueueUnmapMemObject(queue, ws->d_decomp_out, mapped_out, 0, NULL, NULL);
                if (err != CL_SUCCESS) goto dfail;
                clFinish(queue);
            } else {
                unsigned char* gpu_out_tmp = NULL;
                if (gpu_out_sz > 0) {
                    gpu_out_tmp = (unsigned char*)malloc(gpu_out_sz);
                    if (!gpu_out_tmp) goto dfail;
                    err = clEnqueueReadBuffer(queue,
                                              ws->d_decomp_out,
                                              CL_TRUE,
                                              0,
                                              gpu_out_sz,
                                              gpu_out_tmp,
                                              0,
                                              NULL,
                                              NULL);
                    if (err != CL_SUCCESS) {
                        free(gpu_out_tmp);
                        goto dfail;
                    }
                }

                if (gpu_idx) {
                    for (size_t i = 0; i < gpu_count; i++) {
                        size_t bi = gpu_idx[i];
                        size_t off = bi * blk_sz;
                        size_t this_blk = blk_sz;
                        if (off + this_blk > orig_sz) this_blk = orig_sz - off;
                        memcpy(output_buf + off, gpu_out_tmp + i * blk_sz, this_blk);
                    }
                } else {
                    memcpy(output_buf, gpu_out_tmp, gpu_out_sz);
                    for (size_t i = 0; i < gpu_count; i++) {
                        size_t off = i * blk_sz;
                        size_t this_blk = blk_sz;
                        if (off + this_blk > orig_sz) this_blk = orig_sz - off;
                    }
                }

                free(gpu_out_tmp);
            }
        }

        uint64_t t_dl1 = hybrid_now_ns();
        timing.download_us = (unsigned long)((t_dl1 - t_dl0) / 1000);
    }

    /* Wait for CPU */
    if (n_cpu_threads > 0 && cpu_tids) {
        for (int i = 0; i < n_cpu_threads; i++) pthread_join(cpu_tids[i], NULL);
        if (cpu_tids_heap) free(cpu_tids);
        cpu_tids = NULL;
        timing.cpu_kernel_us = (unsigned long)atomic_load(&cpu_pool.max_worker_us);
    }
    if (cpu_ctx_heap) {
        free(cpu_ctx);
        cpu_ctx = NULL;
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
        if (cpu_tids_heap) free(cpu_tids);
    }
    if (cpu_ctx_heap) free(cpu_ctx);
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
    {
        size_t min_blk = adaptive_min_block_size_bytes(params);
        if (min_blk > 0 && blk < min_blk) {
            blk = min_blk;
            nblk = (in_sz + blk - 1) / blk;
        }
    }
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
    uint64_t t_read0 = hybrid_now_ns();
    unsigned long file_read_us = 0;

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
    file_read_us = (unsigned long)((hybrid_now_ns() - t_read0) / 1000);

    /* Build per-block offsets into packed compressed data */
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    offsets[0] = 0;
    for (uint32_t i = 0; i < nblk; i++) offsets[i + 1] = offsets[i] + comp_lengths[i];

    unsigned char* output_buf = (unsigned char*)malloc(orig_sz);
    if (!output_buf) goto dec_fail;

    /* Decompress using buffer-based function */
    int rc = hybrid_decompress_buf(ctx, queue, gpu_kernel,
                                    comp_data, offsets, comp_lengths,
                                    comp_data_sz, orig_sz, blk_sz, nblk, (int)alg_id,
                                    output_buf,
                                    params, ws, &timing);
    if (rc != 0) goto dec_fail;

    timing.file_read_us = file_read_us;

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
    free(output_buf);
    return 0;

dec_fail:
    free(comp_data); free(comp_lengths); free(offsets);
    free(output_buf);
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

typedef struct {
    double adaptive_gpu_ratio_sum;
    size_t adaptive_gpu_ratio_count;
    double adaptive_gpu_ratio_min;
    double adaptive_gpu_ratio_max;
} lzo_hybrid_metrics_t;

static lzo_hybrid_metrics_t g_lzo_hybrid_metrics = {0};

static void lzo_hybrid_metrics_reset(void) {
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_sum = 0.0;
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_count = 0;
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_min = 1.0;
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_max = 0.0;
}

static void lzo_hybrid_metrics_record_adaptive(double ratio) {
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_sum += ratio;
    g_lzo_hybrid_metrics.adaptive_gpu_ratio_count += 1;
    if (ratio < g_lzo_hybrid_metrics.adaptive_gpu_ratio_min) {
        g_lzo_hybrid_metrics.adaptive_gpu_ratio_min = ratio;
    }
    if (ratio > g_lzo_hybrid_metrics.adaptive_gpu_ratio_max) {
        g_lzo_hybrid_metrics.adaptive_gpu_ratio_max = ratio;
    }
}

static int HYBRID_CORE_UNUSED create_temp_path(char* path_buf, size_t path_buf_size, const char* templ) {
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
    double bench_seconds
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
    {
        size_t min_blk = adaptive_min_block_size_bytes(params);
        if (min_blk > 0 && blk < min_blk) {
            blk = min_blk;
            nblk = (in_size + blk - 1) / blk;
        }
    }
    size_t worst_blk = lzo_worst_size(blk);

    /* Pre-allocate reusable buffers */
    uint32_t* lengths = (uint32_t*)calloc(nblk, sizeof(uint32_t));
    unsigned char* out_buf = (unsigned char*)malloc(nblk * worst_blk);
    unsigned char* packed = NULL;
    size_t packed_cap = 0;
    uint32_t* offsets = (uint32_t*)malloc((nblk + 1) * sizeof(uint32_t));
    unsigned char* dec_buf = (unsigned char*)malloc(in_size);
    if (!lengths || !out_buf || !offsets || !dec_buf) {
        free(lengths); free(out_buf); free(packed); free(offsets);
        free(dec_buf); free(input_ref);
        return 1;
    }

    size_t cap = 16, n = 0;
    const size_t bench_drop_iterations = 1;
    size_t total_successful_iterations = 0;
    double* comp_ktp = (double*)malloc(cap * sizeof(double));
    double* dec_ktp = (double*)malloc(cap * sizeof(double));
    double* ratio_pct = (double*)malloc(cap * sizeof(double));
    int verify_ok = 1;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    lzo_hybrid_metrics_reset();

    while (verify_ok) {
        hybrid_timing_t tc = {0};
        hybrid_timing_t td = {0};
        int rc = 0;
        size_t comp_total = 0;
        unsigned long comp_kernel_us = 0;
        unsigned long dec_kernel_us = 0;

        {
            /* ---- COMPRESS (in-memory) ---- */
            int skip_upload = (params->gpu_ratio >= 1.0 && total_successful_iterations > 0) ? 1 : 0;
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
            rc = hybrid_decompress_buf(ctx, queue, dec_kernel,
                                       packed, offsets, lengths,
                                       comp_total, (uint32_t)in_size, (uint32_t)blk, (uint32_t)nblk,
                                       params->alg_id,
                                       dec_buf,
                                       params, ws, &td);

            if (rc != 0) { verify_ok = 0; break; }

            if (memcmp(dec_buf, input_ref, in_size) != 0) {
                verify_ok = 0;
                break;
            }

            dec_kernel_us = td.gpu_kernel_us;
            if (td.cpu_kernel_us > dec_kernel_us) dec_kernel_us = td.cpu_kernel_us;
        }

        total_successful_iterations += 1;
        if (total_successful_iterations <= bench_drop_iterations) {
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            continue;
        }

        /* Record */
        if (n == cap) {
            cap *= 2;
            comp_ktp = (double*)realloc(comp_ktp, cap * sizeof(double));
            dec_ktp = (double*)realloc(dec_ktp, cap * sizeof(double));
            ratio_pct = (double*)realloc(ratio_pct, cap * sizeof(double));
        }

        double in_mb = (double)in_size / (1024.0 * 1024.0);
        comp_ktp[n] = (comp_kernel_us > 0) ? (in_mb * 1e6 / (double)comp_kernel_us) : 0.0;
        dec_ktp[n] = (dec_kernel_us > 0) ? (in_mb * 1e6 / (double)dec_kernel_us) : 0.0;
        ratio_pct[n] = (in_size > 0) ? (100.0 * (double)comp_total / (double)in_size) : 0.0;
        n++;

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        if (elapsed_sec(&ts0, &ts1) >= bench_seconds && n > 0) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);

    if (n > 0) {
         printf("Bench Compress : kernel_tp=%.2f MB/s ratio=%.2f%%\n",
             median_double(comp_ktp, n), median_double(ratio_pct, n));
         printf("Bench Decompress : kernel_tp=%.2f MB/s verify=%s\n",
             median_double(dec_ktp, n), verify_ok ? "OK" : "FAIL");
        if (params->split_mode == HYBRID_SPLIT_ADAPTIVE &&
            g_lzo_hybrid_metrics.adaptive_gpu_ratio_count > 0) {
            double ratio_mean = g_lzo_hybrid_metrics.adaptive_gpu_ratio_sum /
                (double)g_lzo_hybrid_metrics.adaptive_gpu_ratio_count;
            printf("Bench Adaptive : gpu_ratio_mean=%.4f min=%.4f max=%.4f samples=%zu\n",
                   ratio_mean,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_min,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_max,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_count);
            printf("Bench Adaptive : gpu_ratio=%.4f objective=perf_energy_ratio min=%.4f max=%.4f samples=%zu\n",
                   ratio_mean,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_min,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_max,
                   g_lzo_hybrid_metrics.adaptive_gpu_ratio_count);
        }
    } else {
        fprintf(stderr, "bench error: no successful iteration\n");
        verify_ok = 0;
    }

    free(comp_ktp); free(dec_ktp); free(ratio_pct);
    free(input_ref); free(lengths); free(out_buf);
    free(packed); free(offsets); free(dec_buf);
    return verify_ok ? 0 : 1;
}
