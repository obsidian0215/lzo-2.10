/* lzo_levels.h - declarations for level-specific lzo1x compress functions
 * This header exposes the various lzo1x compression entry points that
 * are compiled from the ../src/ lzo1x_1*.c files. The signatures match
 * the miniLZO public API so they can be called directly.
 */
#ifndef LZO_LEVELS_H
#define LZO_LEVELS_H

#include <lzo/lzo1x.h>
#include <lzo/lzo1y.h>

#ifdef __cplusplus
extern "C" {
#endif

/* level 1: D_BITS=11 */
LZO_EXTERN(int) lzo1x_1_11_compress(const lzo_bytep src, lzo_uint src_len,
                                    lzo_bytep dst, lzo_uintp dst_len,
                                    lzo_voidp wrkmem);

/* level 2: D_BITS=12 */
LZO_EXTERN(int) lzo1x_1_12_compress(const lzo_bytep src, lzo_uint src_len,
                                    lzo_bytep dst, lzo_uintp dst_len,
                                    lzo_voidp wrkmem);

/* level 3: D_BITS=14 (standard) */
LZO_EXTERN(int) lzo1x_1_compress(const lzo_bytep src, lzo_uint src_len,
                                 lzo_bytep dst, lzo_uintp dst_len,
                                 lzo_voidp wrkmem);

/* level 4: D_BITS=15 */
LZO_EXTERN(int) lzo1x_1_15_compress(const lzo_bytep src, lzo_uint src_len,
                                    lzo_bytep dst, lzo_uintp dst_len,
                                    lzo_voidp wrkmem);

/* lzo1y */
LZO_EXTERN(int) lzo1y_1_compress(const lzo_bytep src, lzo_uint src_len,
                                 lzo_bytep dst, lzo_uintp dst_len,
                                 lzo_voidp wrkmem);

LZO_EXTERN(int) lzo1x_999_compress(const lzo_bytep src, lzo_uint src_len,
                                   lzo_bytep dst, lzo_uintp dst_len,
                                   lzo_voidp wrkmem);
LZO_EXTERN(int) lzo1x_999_compress_level(const lzo_bytep src, lzo_uint src_len,
                                         lzo_bytep dst, lzo_uintp dst_len,
                                         lzo_voidp wrkmem,
                                         const lzo_bytep dict, lzo_uint dict_len,
                                         lzo_callback_p cb,
                                         int compression_level);
LZO_EXTERN(int) lzo1y_999_compress(const lzo_bytep src, lzo_uint src_len,
                                   lzo_bytep dst, lzo_uintp dst_len,
                                   lzo_voidp wrkmem);
LZO_EXTERN(int) lzo1y_999_compress_level(const lzo_bytep src, lzo_uint src_len,
                                         lzo_bytep dst, lzo_uintp dst_len,
                                         lzo_voidp wrkmem,
                                         const lzo_bytep dict, lzo_uint dict_len,
                                         lzo_callback_p cb,
                                         int compression_level);


#ifdef __cplusplus
}
#endif

#endif /* LZO_LEVELS_H */
