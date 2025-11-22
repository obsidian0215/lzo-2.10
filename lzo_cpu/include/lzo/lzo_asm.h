/*
 * Reduced lzo_asm.h -- stubbed/portable header
 *
 * To simplify the `lzo_cpu` subtree for portable builds we remove
 * architecture-specific assembler prototypes from this header.  The
 * original file declared many platform-specific assembler helpers.
 * Those are intentionally omitted here so that the build uses the
 * pure-C implementations (from `src/` and `minilzo/`) instead.
 */
#ifndef __LZO_ASM_H_INCLUDED
#define __LZO_ASM_H_INCLUDED 1

#ifndef __LZOCONF_H_INCLUDED
#include <lzo/lzoconf.h>
#endif
#ifdef __cplusplus
/*
 * Reduced lzo_asm.h -- stubbed/portable header
 *
 * To simplify the `lzo_cpu` subtree for portable builds we remove
 * architecture-specific assembler prototypes from this header.  The
 * original file declared many platform-specific assembler helpers.
 * Those are intentionally omitted here so that the build uses the
 * pure-C implementations (from `src/` and `minilzo/`) instead.
 */

#ifndef __LZO_ASM_H_INCLUDED
#define __LZO_ASM_H_INCLUDED 1

#ifndef __LZOCONF_H_INCLUDED
#include <lzo/lzoconf.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* No assembler prototypes are declared in this simplified header. */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* already included */

/* vim:set ts=4 sw=4 et: */
   If not, write to the Free Software Foundation, Inc.,
