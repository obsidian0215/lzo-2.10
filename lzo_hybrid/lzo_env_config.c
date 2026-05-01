#include "lzo_env_config.h"
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32) || defined(_WIN64)
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif

static lzo_env_config_t g_cfg;
static int g_cfg_loaded = 0;

int lzo_env_truthy_string(const char* value)
{
    if (!value || !*value) return 0;
    return strcmp(value, "1") == 0 ||
           strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0;
}

long lzo_env_parse_long(const char* value, long defv)
{
    char* endp = NULL;
    long parsed;
    if (!value || !*value) return defv;
    parsed = strtol(value, &endp, 10);
    if (endp == value) return defv;
    return parsed;
}

const lzo_env_config_t* lzo_env_config(void)
{
    const char* standard_copy;
    if (g_cfg_loaded) return &g_cfg;
    memset(&g_cfg, 0, sizeof(g_cfg));

    g_cfg.opencl_device_type = getenv("LZO_OPENCL_DEVICE_TYPE");
    if (!g_cfg.opencl_device_type || !*g_cfg.opencl_device_type) g_cfg.opencl_device_type = "GPU";
    g_cfg.opencl_strict_device = lzo_env_truthy_string(getenv("LZO_OPENCL_STRICT_DEVICE"));

    standard_copy = getenv("LZO_STANDARD_COPY");
    if (standard_copy && *standard_copy) {
        g_cfg.standard_copy_set = 1;
        g_cfg.standard_copy = lzo_env_truthy_string(standard_copy) || atoi(standard_copy) != 0;
    }

    g_cfg.bench_rounds = lzo_env_parse_long(getenv("LZO_BENCH_ROUNDS"), 0);

    g_cfg.block_min_kb = lzo_env_parse_long(getenv("LZO_BLOCK_MIN_KB"), 48);
    g_cfg.block_max_kb = lzo_env_parse_long(getenv("LZO_BLOCK_MAX_KB"), 64);
    g_cfg.block_target_slots = lzo_env_parse_long(getenv("LZO_BLOCK_TARGET_SLOTS"), 0);
    g_cfg.block_slots_per_cu = lzo_env_parse_long(getenv("LZO_BLOCK_SLOTS_PER_CU"), 0);

    g_cfg.gpu_comp_slots = lzo_env_parse_long(getenv("LZO_GPU_COMP_SLOTS"), 0);
    g_cfg.gpu_decomp_chunked = getenv("LZO_GPU_DECOMP_CHUNKED");
    g_cfg.gpu_decomp_chunk_kb = lzo_env_parse_long(getenv("LZO_GPU_DECOMP_CHUNK_KB"), 8192);

    g_cfg_loaded = 1;
    return &g_cfg;
}

size_t lzo_env_effective_gpu_comp_slots(size_t block_count, size_t computed_slots)
{
    long slots = lzo_env_config()->gpu_comp_slots;
    size_t out = computed_slots;
    if (slots > 0) out = (size_t)slots;
    if (block_count > 0 && out > block_count) out = block_count;
    if (out == 0) out = 1;
    return out;
}

int lzo_env_decomp_chunked_mode(void)
{
    const char* value = lzo_env_config()->gpu_decomp_chunked;
    if (!value || !*value || strcasecmp(value, "auto") == 0) return -1;
    if (strcasecmp(value, "on") == 0 || lzo_env_truthy_string(value)) return 1;
    if (strcasecmp(value, "off") == 0 || strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0) return 0;
    return -1;
}

size_t lzo_env_decomp_chunk_kb(void)
{
    long v = lzo_env_config()->gpu_decomp_chunk_kb;
    if (v < 64) v = 64;
    if (v > 1048576) v = 1048576;
    return (size_t)v;
}
