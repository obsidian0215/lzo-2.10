#ifndef LZO_ENV_CONFIG_H
#define LZO_ENV_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* opencl_device_type;
    int opencl_strict_device;
    int standard_copy_set;
    int standard_copy;
    long bench_rounds;
    long block_min_kb;
    long block_max_kb;
    long block_target_slots;
    long block_slots_per_cu;
    long gpu_comp_slots;
    const char* gpu_decomp_chunked;
    long gpu_decomp_chunk_kb;
} lzo_env_config_t;

const lzo_env_config_t* lzo_env_config(void);
int lzo_env_truthy_string(const char* value);
long lzo_env_parse_long(const char* value, long defv);
size_t lzo_env_effective_gpu_comp_slots(size_t block_count, size_t computed_slots);
int lzo_env_decomp_chunked_mode(void); /* -1=auto, 0=off, 1=on */
size_t lzo_env_decomp_chunk_kb(void);

#ifdef __cplusplus
}
#endif

#endif
