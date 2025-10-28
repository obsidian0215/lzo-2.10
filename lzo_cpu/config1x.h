/* copied from src/config1x.h - trimmed for lzo_cpu local build */
#ifndef __LZO_CONFIG1X_H
#define __LZO_CONFIG1X_H 1

#ifndef LZO1X
#define LZO1X 1
#endif

#include "lzo_conf.h"
#include <lzo/lzo1x.h>

#ifndef LZO_EOF_CODE
#define LZO_EOF_CODE 1
#endif

#define M1_MAX_OFFSET   0x0400
#ifndef M2_MAX_OFFSET
#define M2_MAX_OFFSET   0x0800
#endif
#define M3_MAX_OFFSET   0x4000
#define M4_MAX_OFFSET   0xBFFF

#define M1_MIN_LEN      2
#define M1_MAX_LEN      2
#define M2_MIN_LEN      3
#ifndef M2_MAX_LEN
#define M2_MAX_LEN      8
#endif
#define M3_MIN_LEN      3
#define M3_MAX_LEN      33
#define M4_MIN_LEN      3
#define M4_MAX_LEN      9

#define M1_MARKER       0
#define M2_MARKER       64
#define M3_MARKER       32
#define M4_MARKER       16

#ifndef MIN_LOOKAHEAD
#define MIN_LOOKAHEAD       (M2_MAX_LEN + 1)
#endif

#if defined(LZO_NEED_DICT_H)
#include "lzo_dict.h"
#endif

#endif /* __LZO_CONFIG1X_H */
