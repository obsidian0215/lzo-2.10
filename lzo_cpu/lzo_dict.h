/* Wrapper for the internal src/lzo_dict.h so lzo_cpu gets the full
   dictionary macros (DM, GINDEX, UPDATE_I, etc.) without duplicating
   their implementation. */

#include "lzo_conf.h"

/* Prefer including the full internal implementation. Rely on the
   original header's include guards so we don't redefine them here. */
#ifndef __LZO_DICT_H
#include "../src/lzo_dict.h"
#endif
