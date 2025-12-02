/* timing.h - unified timing struct used by daemon, client, compress/decompress
 * Microsecond-resolution timing for all major stages so caller can receive
 * a compact, future-proof struct instead of many separate output parameters.
 */
#ifndef LZO_GPU_TIMING_H
#define LZO_GPU_TIMING_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    unsigned long file_read_us;
    unsigned long ocl_init_us;
    unsigned long kernel_load_us;

    unsigned long blocking_calc_us;

    unsigned long buffer_alloc_in_us;
    unsigned long data_upload_us;
    unsigned long buffer_alloc_out_us;
    unsigned long buffer_alloc_len_us;

    unsigned long setup_args_us;
    unsigned long kernel_setup_us;
    unsigned long kernel_exec_us;

    unsigned long download_len_us;
    unsigned long download_bulk_us;
    unsigned long download_total_us;

    unsigned long file_write_us;
    unsigned long cleanup_us;
} timing_t;

#endif /* LZO_GPU_TIMING_H */
