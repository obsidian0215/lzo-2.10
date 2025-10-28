/* lzo1x_1.c -- LZO1X-1 compression (local copy for lzo_cpu) */
#include "lzo_conf.h"
#if 1 && defined(UA_GET_LE32)
#undef  LZO_DICT_USE_PTR
#define LZO_DICT_USE_PTR 0
#undef  lzo_dict_t
#define lzo_dict_t lzo_uint16_t
#endif

#define LZO_NEED_DICT_H 1
#ifndef D_BITS
#define D_BITS          14
#endif
#define D_INDEX1(d,p)       d = DM(DMUL(0x21,DX3(p,5,5,6)) >> 5)
#define D_INDEX2(d,p)       d = (d & (D_MASK & 0x7ff)) ^ (D_HIGH | 0x1f)
#if 1
#define DINDEX(dv,p)        DM(((DMUL(0x1824429d,dv)) >> (32-D_BITS)))
#else
#define DINDEX(dv,p)        DM((dv) + ((dv) >> (32-D_BITS)))
#endif
#include "config1x.h"
#define LZO_DETERMINISTIC !(LZO_DICT_USE_PTR)

#ifndef DO_COMPRESS
#define DO_COMPRESS     lzo1x_1_compress
#endif

#include "lzo1x_c.ch"

/* vim:set ts=4 sw=4 et: */
