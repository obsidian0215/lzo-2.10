/* trimmed lzo_dict.h for local build: rely on types from ../include/lzo */
#ifndef __LZO_DICT_H
#define __LZO_DICT_H 1

#include "lzo_conf.h"

#if !defined(D_BITS)
# error "D_BITS is not defined"
#endif

#if (D_BITS < 16)
#  define D_SIZE        (1u << (D_BITS))
#  define D_MASK        ((1u << (D_BITS)) - 1)
#else
#  define D_SIZE        ((unsigned)(1u << (D_BITS)))
#  define D_MASK        ((unsigned)(1u << (D_BITS)) - 1)
#endif

#define D_HIGH          ((D_MASK >> 1) + 1)

/* Use lzo's standard integer typedefs from lzodefs.h */
typedef lzo_uint16_t lzo_dict_t;
typedef lzo_dict_t * lzo_dict_p;

#endif /* __LZO_DICT_H */
