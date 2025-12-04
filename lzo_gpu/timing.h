/* timing.h - unified timing struct used by daemon, client, compress/decompress
 * Microsecond-resolution timing for all major stages so caller can receive
 * a compact, future-proof struct instead of many separate output parameters.
 */
#ifndef LZO_GPU_TIMING_H
#define LZO_GPU_TIMING_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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

/* Helper: print a timing value given in microseconds, choosing a human-friendly
 * unit. For values >= 1000 us, print ms with fractional parts. For values < 1000
 * us, print as integer microseconds. If a value is 0, print "N/A" for clarity.
 */
static inline void print_us_tag(FILE *f, const char *tag, unsigned long us) {
    if (!f) return;
    if (us == 0) {
        fprintf(f, "%-22s : %8s\n", tag, "N/A");
    } else if (us >= 1000ul) {
        fprintf(f, "%-22s : %8.3f ms\n", tag, us / 1000.0);
    } else {
        fprintf(f, "%-22s : %8lu us\n", tag, us);
    }
}

#endif /* LZO_GPU_TIMING_H */
