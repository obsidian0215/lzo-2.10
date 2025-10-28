#include "minilzo.h"

#define LZO_BYTE(x)       ((unsigned char) (x))

#define LZO_MAX(a,b)        ((a) >= (b) ? (a) : (b))
#define LZO_MIN(a,b)        ((a) <= (b) ? (a) : (b))
#define LZO_MAX3(a,b,c)     ((a) >= (b) ? LZO_MAX(a,c) : LZO_MAX(b,c))
#define LZO_MIN3(a,b,c)     ((a) <= (b) ? LZO_MIN(a,c) : LZO_MIN(b,c))

#define lzo_sizeof(type)    ((lzo_uint) (sizeof(type)))

#define LZO_HIGH(array)     ((lzo_uint) (sizeof(array)/sizeof(*(array))))

#define LZO_SIZE(bits)      (1u << (bits))
#define LZO_MASK(bits)      (LZO_SIZE(bits) - 1)

#define LZO_USIZE(bits)     ((lzo_uint) 1 << (bits))
#define LZO_UMASK(bits)     (LZO_USIZE(bits) - 1)

#define DMUL(a,b) ((lzo_xint) ((a) * (b)))

#define lzo_memops_TU0p void *

#define lzo_memops_TU1p unsigned char *

struct lzo_memops_TU2_struct { unsigned char a[2]; };
typedef struct lzo_memops_TU2_struct lzo_memops_TU2;
#define lzo_memops_TU2p lzo_memops_TU2 *

struct lzo_memops_TU4_struct { unsigned char a[4]; };
typedef struct lzo_memops_TU4_struct lzo_memops_TU4;
#define lzo_memops_TU4p lzo_memops_TU4 *

struct lzo_memops_TU8_struct { unsigned char a[8]; };
typedef struct lzo_memops_TU8_struct lzo_memops_TU8;
#define lzo_memops_TU8p lzo_memops_TU8 *

#define lzo_memops_set_TU1p     volatile lzo_memops_TU1p
#define lzo_memops_move_TU1p    lzo_memops_TU1p

#define LZO_MEMOPS_SET1(dd,cc) \
    LZO_BLOCK_BEGIN \
    lzo_memops_set_TU1p d__1 = (lzo_memops_set_TU1p) (lzo_memops_TU0p) (dd); \
    d__1[0] = LZO_BYTE(cc); \
    LZO_BLOCK_END
#define LZO_MEMOPS_SET2(dd,cc) \
    LZO_BLOCK_BEGIN \
    lzo_memops_set_TU1p d__2 = (lzo_memops_set_TU1p) (lzo_memops_TU0p) (dd); \
    d__2[0] = LZO_BYTE(cc); d__2[1] = LZO_BYTE(cc); \
    LZO_BLOCK_END
#define LZO_MEMOPS_SET3(dd,cc) \
    LZO_BLOCK_BEGIN \
    lzo_memops_set_TU1p d__3 = (lzo_memops_set_TU1p) (lzo_memops_TU0p) (dd); \
    d__3[0] = LZO_BYTE(cc); d__3[1] = LZO_BYTE(cc); d__3[2] = LZO_BYTE(cc); \
    LZO_BLOCK_END
#define LZO_MEMOPS_SET4(dd,cc) \
    LZO_BLOCK_BEGIN \
    lzo_memops_set_TU1p d__4 = (lzo_memops_set_TU1p) (lzo_memops_TU0p) (dd); \
    d__4[0] = LZO_BYTE(cc); d__4[1] = LZO_BYTE(cc); d__4[2] = LZO_BYTE(cc); d__4[3] = LZO_BYTE(cc); \
    LZO_BLOCK_END
#define LZO_MEMOPS_MOVE1(dd,ss) \
    LZO_BLOCK_BEGIN \
    lzo_memops_move_TU1p d__1 = (lzo_memops_move_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_move_TU1p s__1 = (const lzo_memops_move_TU1p) (const lzo_memops_TU0p) (ss); \
    d__1[0] = s__1[0]; \
    LZO_BLOCK_END
#define LZO_MEMOPS_MOVE2(dd,ss) \
    LZO_BLOCK_BEGIN \
    lzo_memops_move_TU1p d__2 = (lzo_memops_move_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_move_TU1p s__2 = (const lzo_memops_move_TU1p) (const lzo_memops_TU0p) (ss); \
    d__2[0] = s__2[0]; d__2[1] = s__2[1]; \
    LZO_BLOCK_END
#define LZO_MEMOPS_MOVE3(dd,ss) \
    LZO_BLOCK_BEGIN \
    lzo_memops_move_TU1p d__3 = (lzo_memops_move_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_move_TU1p s__3 = (const lzo_memops_move_TU1p) (const lzo_memops_TU0p) (ss); \
    d__3[0] = s__3[0]; d__3[1] = s__3[1]; d__3[2] = s__3[2]; \
    LZO_BLOCK_END
#define LZO_MEMOPS_MOVE4(dd,ss) \
    LZO_BLOCK_BEGIN \
    lzo_memops_move_TU1p d__4 = (lzo_memops_move_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_move_TU1p s__4 = (const lzo_memops_move_TU1p) (const lzo_memops_TU0p) (ss); \
    d__4[0] = s__4[0]; d__4[1] = s__4[1]; d__4[2] = s__4[2]; d__4[3] = s__4[3]; \
    LZO_BLOCK_END
#define LZO_MEMOPS_MOVE8(dd,ss) \
    LZO_BLOCK_BEGIN \
    lzo_memops_move_TU1p d__8 = (lzo_memops_move_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_move_TU1p s__8 = (const lzo_memops_move_TU1p) (const lzo_memops_TU0p) (ss); \
    d__8[0] = s__8[0]; d__8[1] = s__8[1]; d__8[2] = s__8[2]; d__8[3] = s__8[3]; \
    d__8[4] = s__8[4]; d__8[5] = s__8[5]; d__8[6] = s__8[6]; d__8[7] = s__8[7]; \
    LZO_BLOCK_END

#define LZO_MEMOPS_COPY1(dd,ss) LZO_MEMOPS_MOVE1(dd,ss)
#define LZO_MEMOPS_COPY2(dd,ss) LZO_MEMOPS_MOVE2(dd,ss)
#define LZO_MEMOPS_COPY4(dd,ss) LZO_MEMOPS_MOVE4(dd,ss)
#define LZO_MEMOPS_COPY8(dd,ss) LZO_MEMOPS_MOVE8(dd,ss)

#define LZO_MEMOPS_COPYN(dd,ss,nn) \
    LZO_BLOCK_BEGIN \
    lzo_memops_TU1p d__n = (lzo_memops_TU1p) (lzo_memops_TU0p) (dd); \
    const lzo_memops_TU1p s__n = (const lzo_memops_TU1p) (const lzo_memops_TU0p) (ss); \
    lzo_uint n__n = (nn); \
    while ((void)0, n__n >= 8) { LZO_MEMOPS_COPY8(d__n, s__n); d__n += 8; s__n += 8; n__n -= 8; } \
    if ((void)0, n__n >= 4) { LZO_MEMOPS_COPY4(d__n, s__n); d__n += 4; s__n += 4; n__n -= 4; } \
    if ((void)0, n__n > 0) do { *d__n++ = *s__n++; } while (--n__n > 0); \
    LZO_BLOCK_END

static inline lzo_uint16_t lzo_memops_get_le16(const lzo_voidp ss)
{
    lzo_uint16_t v;
    LZO_MEMOPS_COPY2(&v, ss);
    return v;
}
#define LZO_MEMOPS_GET_LE16(ss)    lzo_memops_get_le16(ss)

static inline lzo_uint32_t lzo_memops_get_le32(const lzo_voidp ss)
{
    lzo_uint32_t v;
    LZO_MEMOPS_COPY4(&v, ss);
    return v;
}
#define LZO_MEMOPS_GET_LE32(ss)    lzo_memops_get_le32(ss)

static inline lzo_uint16_t lzo_memops_get_ne16(const lzo_voidp ss)
{
    lzo_uint16_t v;
    LZO_MEMOPS_COPY2(&v, ss);
    return v;
}
#define LZO_MEMOPS_GET_NE16(ss)    lzo_memops_get_ne16(ss)

static inline lzo_uint32_t lzo_memops_get_ne32(const lzo_voidp ss)
{
    lzo_uint32_t v;
    LZO_MEMOPS_COPY4(&v, ss);
    return v;
}
#define LZO_MEMOPS_GET_NE32(ss)    lzo_memops_get_ne32(ss)

static inline void lzo_memops_put_le16(lzo_voidp dd, lzo_uint16_t vv)
{
    LZO_MEMOPS_COPY2(dd, &vv);
}
#define LZO_MEMOPS_PUT_LE16(dd,vv) lzo_memops_put_le16(dd,vv)

static inline void lzo_memops_put_le32(lzo_voidp dd, lzo_uint32_t vv)
{
    LZO_MEMOPS_COPY4(dd, &vv);
}
#define LZO_MEMOPS_PUT_LE32(dd,vv) lzo_memops_put_le32(dd,vv)

static inline void lzo_memops_put_ne16(lzo_voidp dd, lzo_uint16_t vv)
{
    LZO_MEMOPS_COPY2(dd, &vv);
}
#define LZO_MEMOPS_PUT_NE16(dd,vv) lzo_memops_put_ne16(dd,vv)

static inline void lzo_memops_put_ne32(lzo_voidp dd, lzo_uint32_t vv)
{
    LZO_MEMOPS_COPY4(dd, &vv);
}
#define LZO_MEMOPS_PUT_NE32(dd,vv) lzo_memops_put_ne32(dd,vv)

#define UA_SET1             LZO_MEMOPS_SET1
#define UA_SET2             LZO_MEMOPS_SET2
#define UA_SET3             LZO_MEMOPS_SET3
#define UA_SET4             LZO_MEMOPS_SET4

#define UA_MOVE1            LZO_MEMOPS_MOVE1
#define UA_MOVE2            LZO_MEMOPS_MOVE2
#define UA_MOVE3            LZO_MEMOPS_MOVE3
#define UA_MOVE4            LZO_MEMOPS_MOVE4
#define UA_MOVE8            LZO_MEMOPS_MOVE8

#define UA_COPY1            LZO_MEMOPS_COPY1
#define UA_COPY2            LZO_MEMOPS_COPY2
#define UA_COPY3            LZO_MEMOPS_COPY3
#define UA_COPY4            LZO_MEMOPS_COPY4
#define UA_COPY8            LZO_MEMOPS_COPY8
#define UA_COPYN            LZO_MEMOPS_COPYN
#define UA_COPYN_X          LZO_MEMOPS_COPYN

#define UA_GET_LE16         LZO_MEMOPS_GET_LE16
#define UA_GET_LE32         LZO_MEMOPS_GET_LE32
#define UA_GET_NE16         LZO_MEMOPS_GET_NE16
#define UA_GET_NE32         LZO_MEMOPS_GET_NE32

#define UA_PUT_LE16         LZO_MEMOPS_PUT_LE16
#define UA_PUT_LE32         LZO_MEMOPS_PUT_LE32
#define UA_PUT_NE16         LZO_MEMOPS_PUT_NE16
#define UA_PUT_NE32         LZO_MEMOPS_PUT_NE32

#define PTR(a)              ((lzo_uintptr_t) (a))
#define PTR_LINEAR(a)       PTR(a)
#define PTR_ALIGNED_4(a)    ((PTR_LINEAR(a) & 3) == 0)
#define PTR_ALIGNED_8(a)    ((PTR_LINEAR(a) & 7) == 0)
#define PTR_ALIGNED2_4(a,b) (((PTR_LINEAR(a) | PTR_LINEAR(b)) & 3) == 0)
#define PTR_ALIGNED2_8(a,b) (((PTR_LINEAR(a) | PTR_LINEAR(b)) & 7) == 0)

#define PTR_LT(a,b)         (PTR(a) < PTR(b))
#define PTR_GE(a,b)         (PTR(a) >= PTR(b))
#define PTR_DIFF(a,b)       (PTR(a) - PTR(b))
#define pd(a,b)             ((lzo_uint) ((a)-(b)))

#define lzo_dict_t    const lzo_bytep
#define lzo_dict_p    lzo_dict_t *

#define LZO_BASE 65521u
#define LZO_NMAX 5552

#define lzo_hsize_t             lzo_uint
#define lzo_hvoid_p             lzo_voidp
#define lzo_hbyte_p             lzo_bytep
#define LZOLIB_PUBLIC(r,f)      LZO_PUBLIC(r) f

LZOLIB_PUBLIC(int, lzo_memcmp) (const lzo_hvoid_p s1, const lzo_hvoid_p s2, lzo_hsize_t len)
{
    const lzo_hbyte_p p1 = LZO_STATIC_CAST(const lzo_hbyte_p, s1);
    const lzo_hbyte_p p2 = LZO_STATIC_CAST(const lzo_hbyte_p, s2);
    if (len > 0) do
    {
        int d = *p1 - *p2;
        if (d != 0)
            return d;
        p1++; p2++;
    } while(--len > 0);
    return 0;
}

LZOLIB_PUBLIC(lzo_hvoid_p, lzo_memcpy) (lzo_hvoid_p dest, const lzo_hvoid_p src, lzo_hsize_t len)
{
    lzo_hbyte_p p1 = LZO_STATIC_CAST(lzo_hbyte_p, dest);
    const lzo_hbyte_p p2 = LZO_STATIC_CAST(const lzo_hbyte_p, src);
    if (!(len > 0) || p1 == p2)
        return dest;
    do
        *p1++ = *p2++;
    while(--len > 0);
    return dest;
}

LZOLIB_PUBLIC(lzo_hvoid_p, lzo_memmove) (lzo_hvoid_p dest, const lzo_hvoid_p src, lzo_hsize_t len)
{
    lzo_hbyte_p p1 = LZO_STATIC_CAST(lzo_hbyte_p, dest);
    const lzo_hbyte_p p2 = LZO_STATIC_CAST(const lzo_hbyte_p, src);
    if (!(len > 0) || p1 == p2)
        return dest;
    if (p1 < p2)
    {
        do
            *p1++ = *p2++;
        while(--len > 0);
    }
    else
    {
        p1 += len;
        p2 += len;
        do
            *--p1 = *--p2;
        while(--len > 0);
    }
    return dest;
}

LZOLIB_PUBLIC(lzo_hvoid_p, lzo_memset) (lzo_hvoid_p s, int cc, lzo_hsize_t len)
{
    lzo_hbyte_p p = LZO_STATIC_CAST(lzo_hbyte_p, s);
    unsigned char c = LZO_ITRUNC(unsigned char, cc);
    if (len > 0) do
        *p++ = c;
    while(--len > 0);
    return s;
}
#undef LZOLIB_PUBLIC

#define LZO1X           1
#define LZO_EOF_CODE    1

#undef  lzo_dict_t
#define lzo_dict_t lzo_uint16_t

//start
__lzo_static_forceinline unsigned lzo_bitops_ctlz32_func(lzo_uint32_t v)
{
#if (LZO_BITOPS_USE_MSC_BITSCAN) && (LZO_ARCH_AMD64 || LZO_ARCH_I386)
    unsigned long r; (void)_BitScanReverse(&r, v); return (unsigned)r ^ 31;
#define lzo_bitops_ctlz32(v)    lzo_bitops_ctlz32_func(v)
#elif (LZO_BITOPS_USE_ASM_BITSCAN) && (LZO_ARCH_AMD64 || LZO_ARCH_I386) && (LZO_ASM_SYNTAX_GNUC)
    lzo_uint32_t r;
    __asm__("bsr %1,%0" : "=r" (r) : "rm" (v)__LZO_ASM_CLOBBER_LIST_CC);
    return (unsigned)r ^ 31;
#define lzo_bitops_ctlz32(v)    lzo_bitops_ctlz32_func(v)
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_INT == 4)
    unsigned r; r = (unsigned)__builtin_clz(v); return r;
#define lzo_bitops_ctlz32(v)    ((unsigned) __builtin_clz(v))
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_LONG == 8) && (LZO_WORDSIZE >= 8)
    unsigned r; r = (unsigned)__builtin_clzl(v); return r ^ 32;
#define lzo_bitops_ctlz32(v)    (((unsigned) __builtin_clzl(v)) ^ 32)
#else
    LZO_UNUSED(v); return 0;
#endif
}

#if defined(lzo_uint64_t)
__lzo_static_forceinline unsigned lzo_bitops_ctlz64_func(lzo_uint64_t v)
{
#if (LZO_BITOPS_USE_MSC_BITSCAN) && (LZO_ARCH_AMD64)
    unsigned long r; (void)_BitScanReverse64(&r, v); return (unsigned)r ^ 63;
#define lzo_bitops_ctlz64(v)    lzo_bitops_ctlz64_func(v)
#elif (LZO_BITOPS_USE_ASM_BITSCAN) && (LZO_ARCH_AMD64) && (LZO_ASM_SYNTAX_GNUC)
    lzo_uint64_t r;
    __asm__("bsr %1,%0" : "=r" (r) : "rm" (v)__LZO_ASM_CLOBBER_LIST_CC);
    return (unsigned)r ^ 63;
#define lzo_bitops_ctlz64(v)    lzo_bitops_ctlz64_func(v)
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_LONG == 8) && (LZO_WORDSIZE >= 8)
    unsigned r; r = (unsigned)__builtin_clzl(v); return r;
#define lzo_bitops_ctlz64(v)    ((unsigned) __builtin_clzl(v))
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_LONG_LONG == 8) && (LZO_WORDSIZE >= 8)
    unsigned r; r = (unsigned)__builtin_clzll(v); return r;
#define lzo_bitops_ctlz64(v)    ((unsigned) __builtin_clzll(v))
#else
    LZO_UNUSED(v); return 0;
#endif
}
#endif

__lzo_static_forceinline unsigned lzo_bitops_cttz32_func(lzo_uint32_t v)
{
#if (LZO_BITOPS_USE_MSC_BITSCAN) && (LZO_ARCH_AMD64 || LZO_ARCH_I386)
    unsigned long r; (void)_BitScanForward(&r, v); return (unsigned)r;
#define lzo_bitops_cttz32(v)    lzo_bitops_cttz32_func(v)
#elif (LZO_BITOPS_USE_ASM_BITSCAN) && (LZO_ARCH_AMD64 || LZO_ARCH_I386) && (LZO_ASM_SYNTAX_GNUC)
    lzo_uint32_t r;
    __asm__("bsf %1,%0" : "=r" (r) : "rm" (v)__LZO_ASM_CLOBBER_LIST_CC);
    return (unsigned)r;
#define lzo_bitops_cttz32(v)    lzo_bitops_cttz32_func(v)
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_INT >= 4)
    unsigned r; r = (unsigned)__builtin_ctz(v); return r;
#define lzo_bitops_cttz32(v)    ((unsigned) __builtin_ctz(v))
#else
    LZO_UNUSED(v); return 0;
#endif
}

#if defined(lzo_uint64_t)
__lzo_static_forceinline unsigned lzo_bitops_cttz64_func(lzo_uint64_t v)
{
#if (LZO_BITOPS_USE_MSC_BITSCAN) && (LZO_ARCH_AMD64)
    unsigned long r; (void)_BitScanForward64(&r, v); return (unsigned)r;
#define lzo_bitops_cttz64(v)    lzo_bitops_cttz64_func(v)
#elif (LZO_BITOPS_USE_ASM_BITSCAN) && (LZO_ARCH_AMD64) && (LZO_ASM_SYNTAX_GNUC)
    lzo_uint64_t r;
    __asm__("bsf %1,%0" : "=r" (r) : "rm" (v)__LZO_ASM_CLOBBER_LIST_CC);
    return (unsigned)r;
#define lzo_bitops_cttz64(v)    lzo_bitops_cttz64_func(v)
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_LONG >= 8) && (LZO_WORDSIZE >= 8)
    unsigned r; r = (unsigned)__builtin_ctzl(v); return r;
#define lzo_bitops_cttz64(v)    ((unsigned) __builtin_ctzl(v))
#elif (LZO_BITOPS_USE_GNUC_BITSCAN) && (LZO_SIZEOF_LONG_LONG >= 8) && (LZO_WORDSIZE >= 8)
    unsigned r; r = (unsigned)__builtin_ctzll(v); return r;
#define lzo_bitops_cttz64(v)    ((unsigned) __builtin_ctzll(v))
#else
    LZO_UNUSED(v); return 0;
#endif
}
#endif

lzo_unused_funcs_impl(void, lzo_bitops_unused_funcs)(void)
{
    LZO_UNUSED_FUNC(lzo_bitops_unused_funcs);
    LZO_UNUSED_FUNC(lzo_bitops_ctlz32_func);
    LZO_UNUSED_FUNC(lzo_bitops_cttz32_func);
#if defined(lzo_uint64_t)
    LZO_UNUSED_FUNC(lzo_bitops_ctlz64_func);
    LZO_UNUSED_FUNC(lzo_bitops_cttz64_func);
#endif
}

union lzo_config_check_union {
    lzo_uint a[2];
    unsigned char b[2 * LZO_MAX(8, sizeof(lzo_uint))];
#if defined(lzo_uint64_t)
    lzo_uint64_t c[2];
#endif
};

static __lzo_noinline lzo_voidp u2p(lzo_voidp ptr, lzo_uint off)
{
    return (lzo_voidp)((lzo_bytep)ptr + off);
}

LZO_PUBLIC(int)
_lzo_config_check(void)
{
#if (LZO_CC_CLANG && (LZO_CC_CLANG >= 0x030100ul && LZO_CC_CLANG < 0x030300ul))
# if 0
    volatile
# endif
#endif
        union lzo_config_check_union u;
    lzo_voidp p;
    unsigned r = 1;

    u.a[0] = u.a[1] = 0;
    p = u2p(&u, 0);
    r &= ((*(lzo_bytep)p) == 0);
#if !(LZO_CFG_NO_CONFIG_CHECK)
#if (LZO_ABI_BIG_ENDIAN)
    u.a[0] = u.a[1] = 0; u.b[sizeof(lzo_uint) - 1] = 128;
    p = u2p(&u, 0);
    r &= ((*(lzo_uintp)p) == 128);
#endif
#if (LZO_ABI_LITTLE_ENDIAN)
    u.a[0] = u.a[1] = 0; u.b[0] = 128;
    p = u2p(&u, 0);
    r &= ((*(lzo_uintp)p) == 128);
#endif
    u.a[0] = u.a[1] = 0;
    u.b[0] = 1; u.b[3] = 2;
    p = u2p(&u, 1);
    r &= UA_GET_NE16(p) == 0;
    r &= UA_GET_LE16(p) == 0;
    u.b[1] = 128;
    r &= UA_GET_LE16(p) == 128;
    u.b[2] = 129;
    r &= UA_GET_LE16(p) == LZO_UINT16_C(0x8180);
#if (LZO_ABI_BIG_ENDIAN)
    r &= UA_GET_NE16(p) == LZO_UINT16_C(0x8081);
#endif
#if (LZO_ABI_LITTLE_ENDIAN)
    r &= UA_GET_NE16(p) == LZO_UINT16_C(0x8180);
#endif
    u.a[0] = u.a[1] = 0;
    u.b[0] = 3; u.b[5] = 4;
    p = u2p(&u, 1);
    r &= UA_GET_NE32(p) == 0;
    r &= UA_GET_LE32(p) == 0;
    u.b[1] = 128;
    r &= UA_GET_LE32(p) == 128;
    u.b[2] = 129; u.b[3] = 130; u.b[4] = 131;
    r &= UA_GET_LE32(p) == LZO_UINT32_C(0x83828180);
#if (LZO_ABI_BIG_ENDIAN)
    r &= UA_GET_NE32(p) == LZO_UINT32_C(0x80818283);
#endif
#if (LZO_ABI_LITTLE_ENDIAN)
    r &= UA_GET_NE32(p) == LZO_UINT32_C(0x83828180);
#endif
#if defined(UA_GET_NE64)
    u.c[0] = u.c[1] = 0;
    u.b[0] = 5; u.b[9] = 6;
    p = u2p(&u, 1);
    u.c[0] = u.c[1] = 0;
    r &= UA_GET_NE64(p) == 0;
#if defined(UA_GET_LE64)
    r &= UA_GET_LE64(p) == 0;
    u.b[1] = 128;
    r &= UA_GET_LE64(p) == 128;
#endif
#endif
#if defined(lzo_bitops_ctlz32)
    {
        unsigned i = 0; lzo_uint32_t v;
        for (v = 1; v != 0 && r == 1; v <<= 1, i++) {
            r &= lzo_bitops_ctlz32(v) == 31 - i;
            r &= lzo_bitops_ctlz32_func(v) == 31 - i;
        }
    }
#endif
#if defined(lzo_bitops_ctlz64)
    {
        unsigned i = 0; lzo_uint64_t v;
        for (v = 1; v != 0 && r == 1; v <<= 1, i++) {
            r &= lzo_bitops_ctlz64(v) == 63 - i;
            r &= lzo_bitops_ctlz64_func(v) == 63 - i;
        }
    }
#endif
#if defined(lzo_bitops_cttz32)
    {
        unsigned i = 0; lzo_uint32_t v;
        for (v = 1; v != 0 && r == 1; v <<= 1, i++) {
            r &= lzo_bitops_cttz32(v) == i;
            r &= lzo_bitops_cttz32_func(v) == i;
        }
    }
#endif
#if defined(lzo_bitops_cttz64)
    {
        unsigned i = 0; lzo_uint64_t v;
        for (v = 1; v != 0 && r == 1; v <<= 1, i++) {
            r &= lzo_bitops_cttz64(v) == i;
            r &= lzo_bitops_cttz64_func(v) == i;
        }
    }
#endif
#endif
    LZO_UNUSED_FUNC(lzo_bitops_unused_funcs);

    return r == 1 ? LZO_E_OK : LZO_E_ERROR;
}

LZO_PUBLIC(int)
__lzo_init_v2(unsigned v, int s1, int s2, int s3, int s4, int s5,
    int s6, int s7, int s8, int s9)
{
    int r;

#if defined(__LZO_IN_MINILZO)
#elif (LZO_CC_MSC && ((_MSC_VER) < 700))
#else
#define LZO_WANT_ACC_CHK_CH 1
#undef LZOCHK_ASSERT
#define LZOCHK_ASSERT(expr)  LZO_COMPILE_TIME_ASSERT(expr)
#endif
#undef LZOCHK_ASSERT

    if (v == 0)
        return LZO_E_ERROR;

    r = (s1 == -1 || s1 == (int)sizeof(short)) &&
        (s2 == -1 || s2 == (int)sizeof(int)) &&
        (s3 == -1 || s3 == (int)sizeof(long)) &&
        (s4 == -1 || s4 == (int)sizeof(lzo_uint32_t)) &&
        (s5 == -1 || s5 == (int)sizeof(lzo_uint)) &&
        (s6 == -1 || s6 == (int)lzo_sizeof_dict_t) &&
        (s7 == -1 || s7 == (int)sizeof(char*)) &&
        (s8 == -1 || s8 == (int)sizeof(lzo_voidp)) &&
        (s9 == -1 || s9 == (int)sizeof(lzo_callback_t));
    if (!r)
        return LZO_E_ERROR;

    r = _lzo_config_check();
    if (r != LZO_E_OK)
        return r;

    return r;
}
//end

#ifndef D_BITS
#define D_BITS          11
#endif
// #define D_INDEX1(d,p)       d = DM(DMUL(0x21,DX3(p,5,5,6)) >> 5)
#define D_INDEX1(d,p)       d = DM(DMUL(0x21,DX2(p,3,5)) >> 5)
#define D_INDEX2(d,p)       d = (d & (D_MASK & 0x7ff)) ^ (D_HIGH | 0x1f)
#define DINDEX(dv,p)        DM(((DMUL(0x1824429d,dv)) >> (32-D_BITS)))

#define M1_MAX_OFFSET   0x0400
#define M2_MAX_OFFSET   0x0800
#define M3_MAX_OFFSET   0x4000
#define M4_MAX_OFFSET   0xbfff

#define MX_MAX_OFFSET   (M1_MAX_OFFSET + M2_MAX_OFFSET)

#define M1_MIN_LEN      2
#define M1_MAX_LEN      2
#define M2_MIN_LEN      3
#define M2_MAX_LEN      8
#define M3_MIN_LEN      3
#define M3_MAX_LEN      33
#define M4_MIN_LEN      3
#define M4_MAX_LEN      9

#define M1_MARKER       0
#define M2_MARKER       64
#define M3_MARKER       32
#define M4_MARKER       16

#define MIN_LOOKAHEAD       (M2_MAX_LEN + 1)

#define DL_MIN_LEN          M2_MIN_LEN

#define D_SIZE        LZO_SIZE(D_BITS)
#define D_MASK        LZO_MASK(D_BITS)

#define D_HIGH          ((D_MASK >> 1) + 1)

#define DD_BITS       0
#define DD_SIZE         LZO_SIZE(DD_BITS)
#define DD_MASK         LZO_MASK(DD_BITS)

#define DL_BITS       (D_BITS - DD_BITS)
#define DL_SIZE       LZO_SIZE(DL_BITS)
#define DL_MASK       LZO_MASK(DL_BITS)

#if (D_BITS != DL_BITS + DD_BITS)
#  error "D_BITS does not match"
#endif
#if (D_BITS < 6 || D_BITS > 18)
#  error "invalid D_BITS"
#endif
#if (DL_BITS < 6 || DL_BITS > 20)
#  error "invalid DL_BITS"
#endif
#if (DD_BITS < 0 || DD_BITS > 6)
#  error "invalid DD_BITS"
#endif

#define DL_SHIFT      ((DL_BITS + (DL_MIN_LEN - 1)) / DL_MIN_LEN)

#undef DM
#undef DX

#define _DV2_A(p,shift1,shift2) \
        (((( (lzo_xint)((p)[0]) << shift1) ^ (p)[1]) << shift2) ^ (p)[2])
#define _DV2_B(p,shift1,shift2) \
        (((( (lzo_xint)((p)[2]) << shift1) ^ (p)[1]) << shift2) ^ (p)[0])
#define _DV3_B(p,shift1,shift2,shift3) \
        ((_DV2_B((p)+1,shift1,shift2) << (shift3)) ^ (p)[0])

#define _DV_A(p,shift)      _DV2_A(p,shift,shift)
#define _DV_B(p,shift)      _DV2_B(p,shift,shift)
#define DA2(p,s1,s2) \
        (((((lzo_xint)((p)[2]) << (s2)) + (p)[1]) << (s1)) + (p)[0])
#define DS2(p,s1,s2) \
        (((((lzo_xint)((p)[2]) << (s2)) - (p)[1]) << (s1)) - (p)[0])
#define DX2(p,s1,s2) \
        (((((lzo_xint)((p)[2]) << (s2)) ^ (p)[1]) << (s1)) ^ (p)[0])
#define DA3(p,s1,s2,s3) ((DA2((p)+1,s2,s3) << (s1)) + (p)[0])
#define DS3(p,s1,s2,s3) ((DS2((p)+1,s2,s3) << (s1)) - (p)[0])
#define DX3(p,s1,s2,s3) ((DX2((p)+1,s2,s3) << (s1)) ^ (p)[0])
#define DMS(v,s)        ((lzo_uint) (((v) & (D_MASK >> (s))) << (s)))
#define DM(v)           DMS(v,0)


// LZO_HASH == LZO_HASH_LZO_INCREMENTAL_B
#define DVAL_FIRST(dv,p)  dv = _DV_B((p),5)
#define DVAL_NEXT(dv,p) \
                dv ^= p[-1]; dv = (((dv) >> 5) ^ ((lzo_xint)(p[2]) << (2*5)))
#define _DINDEX(dv,p)     ((DMUL(0x9f5f,dv)) >> 5)
#define DVAL_LOOKAHEAD    DL_MIN_LEN

#define DINDEX1             D_INDEX1
#define DINDEX2             D_INDEX2

#define DENTRY(p,in)                          ((lzo_dict_t) pd(p, in))
#define GINDEX(m_pos,m_off,dict,dindex,in)    m_off = dict[dindex]

// DD_BITS == 0
#define UPDATE_D(dict,drun,dv,p,in)       dict[ DINDEX(dv,p) ] = DENTRY(p,in)
#define UPDATE_I(dict,drun,index,p,in)    dict[index] = DENTRY(p,in)
#define UPDATE_P(ptr,drun,p,in)           (ptr)[0] = DENTRY(p,in)

#define LZO_CHECK_MPOS_DET(m_pos,m_off,in,ip,max_offset) \
        (m_off == 0 || \
         ((m_off = pd(ip, in) - m_off) > max_offset) || \
         (m_pos = (ip) - (m_off), 0) )

#define LZO_CHECK_MPOS_NON_DET(m_pos,m_off,in,ip,max_offset) \
        (pd(ip, in) <= m_off || \
         ((m_off = pd(ip, in) - m_off) > max_offset) || \
         (m_pos = (ip) - (m_off), 0) )

#define LZO_CHECK_MPOS    LZO_CHECK_MPOS_NON_DET

#define DO_COMPRESS     lzo1x_1_compress

#define do_compress       LZO_PP_ECONCAT2(DO_COMPRESS,_core)

static lzo_uint
do_compress ( const lzo_bytep in , lzo_uint  in_len,
                    lzo_bytep out, lzo_uintp out_len,
                    lzo_uint  ti,  lzo_voidp wrkmem)
{
    const lzo_bytep ip;
    lzo_bytep op;
    const lzo_bytep const in_end = in + in_len;
    const lzo_bytep const ip_end = in + in_len - 20;
    const lzo_bytep ii;
    lzo_dict_p const dict = (lzo_dict_p) wrkmem;

    op = out;
    ip = in;
    ii = ip;

    ip += ti < 4 ? 4 - ti : 0;
    for (;;)
    {
        const lzo_bytep m_pos;

        lzo_uint m_off;
        lzo_uint m_len;
        {
        lzo_uint32_t dv;
        lzo_uint dindex;
literal:
        ip += 1 + ((ip - ii) >> 5);
next:
        if (ip >= ip_end)
            break;
        dv = UA_GET_LE32(ip);
        dindex = DINDEX(dv,ip);
        GINDEX(m_off,m_pos,in+dict,dindex,in);
        UPDATE_I(dict,0,dindex,ip,in);
        if (dv != UA_GET_LE32(m_pos))
            goto literal;
        }

        ii -= ti; ti = 0;
        lzo_uint t = pd(ip,ii);
        if (t != 0)
        {
            if (t <= 3)
            {
                op[-2] = LZO_BYTE(op[-2] | t);
                { do *op++ = *ii++; while (--t > 0); }
            }
            else
            {
                if (t <= 18)
                    *op++ = LZO_BYTE(t - 3);
                else
                {
                    lzo_uint tt = t - 18;
                    *op++ = 0;
                    while (tt > 255)
                    {
                        tt -= 255;
                        UA_SET1(op, 0);
                        op++;
                    }
                    // assert(tt > 0);
                    *op++ = LZO_BYTE(tt);
                }
                { do *op++ = *ii++; while (--t > 0); }
            }
        }
        m_len = 4;
        if (ip[m_len] == m_pos[m_len]) {
            do {
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len])
                    break;
                m_len += 1;
                if (ip + m_len >= ip_end)
                    goto m_len_done;
            } while (ip[m_len] == m_pos[m_len]);
        }
m_len_done:
        m_off = pd(ip,m_pos);
        ip += m_len;
        ii = ip;
        if (m_len <= M2_MAX_LEN && m_off <= M2_MAX_OFFSET)
        {
            m_off -= 1;

            *op++ = LZO_BYTE(((m_len - 1) << 5) | ((m_off & 7) << 2));
            *op++ = LZO_BYTE(m_off >> 3);
        }
        else if (m_off <= M3_MAX_OFFSET)
        {
            m_off -= 1;
            if (m_len <= M3_MAX_LEN)
                *op++ = LZO_BYTE(M3_MARKER | (m_len - 2));
            else
            {
                m_len -= M3_MAX_LEN;
                *op++ = M3_MARKER | 0;
                while(m_len > 255)
                {
                    m_len -= 255;
                    UA_SET1(op, 0);
                    op++;
                }
                *op++ = LZO_BYTE(m_len);
            }
            *op++ = LZO_BYTE(m_off << 2);
            *op++ = LZO_BYTE(m_off >> 6);
        }
        else
        {
            m_off -= 0x4000;
            if (m_len <= M4_MAX_LEN)
                *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8) | (m_len - 2));
            else
            {
                m_len -= M4_MAX_LEN;
                *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8));
                while(m_len > 255)
                {
                    m_len -= 255;
                    UA_SET1(op, 0);
                    op++;
                }
                *op++ = LZO_BYTE(m_len);
            }
            *op++ = LZO_BYTE(m_off << 2);
            *op++ = LZO_BYTE(m_off >> 6);
        }
        goto next;
    }

    *out_len = pd(op, out);
    return pd(in_end,ii-ti);
}

LZO_PUBLIC(int)
DO_COMPRESS      ( const lzo_bytep in , lzo_uint  in_len,
                         lzo_bytep out, lzo_uintp out_len,
                         lzo_voidp wrkmem )
{
    const lzo_bytep ip = in;
    lzo_bytep op = out;
    lzo_uint l = in_len;
    lzo_uint t = 0;

    while (l > 20)
    {
        lzo_uint ll = l;
        lzo_uintptr_t ll_end;
        ll = LZO_MIN(ll, 49152);
        ll_end = (lzo_uintptr_t)ip + ll;
        if ((ll_end + ((t + ll) >> 5)) <= ll_end || (const lzo_bytep)(ll_end + ((t + ll) >> 5)) <= ip + ll)
            break;
        lzo_memset(wrkmem, 0, ((lzo_uint)1 << D_BITS) * sizeof(lzo_dict_t));
        t = do_compress(ip,ll,op,out_len,t,wrkmem);
        ip += ll;
        op += *out_len;
        l  -= ll;
    }
    t += l;

    if (t > 0)
    {
        const lzo_bytep ii = in + in_len - t;

        if (op == out && t <= 238)
            *op++ = LZO_BYTE(17 + t);
        else if (t <= 3)
            op[-2] = LZO_BYTE(op[-2] | t);
        else if (t <= 18)
            *op++ = LZO_BYTE(t - 3);
        else
        {
            lzo_uint tt = t - 18;

            *op++ = 0;
            while (tt > 255)
            {
                tt -= 255;
                UA_SET1(op, 0);
                op++;
            }
            // assert(tt > 0);
            *op++ = LZO_BYTE(tt);
        }
        UA_COPYN(op, ii, t);
        op += t;
    }

    *op++ = M4_MARKER | 1;
    *op++ = 0;
    *op++ = 0;

    *out_len = pd(op, out);
    return LZO_E_OK;
}


#define TEST_IP               1
#define TEST_OP               1
#define TEST_IP_AND_TEST_OP   1
// #define TEST_LB(m_pos)        ((void) 0)
// #define TEST_LBO(m_pos,o)     ((void) 0)
// #define NEED_IP(x)            ((void) 0)
// #define TEST_IV(x)            ((void) 0)
// #define NEED_OP(x)            ((void) 0)
// #define TEST_OV(x)            ((void) 0)

LZO_PUBLIC(int)
lzo1x_decompress(const lzo_bytep in, lzo_uint  in_len,
    lzo_bytep out, lzo_uintp out_len,
    lzo_voidp wrkmem)
{
    lzo_bytep op;
    const lzo_bytep ip;
    lzo_uint t;
    const lzo_bytep m_pos;

    const lzo_bytep const ip_end = in + in_len;
    LZO_UNUSED(wrkmem);

    * out_len = 0;

    op = out;
    ip = in;

    // NEED_IP(1);
    if (*ip > 17)
    {
        t = *ip++ - 17;
        if (t < 4)
            goto match_next;
        // assert(t > 0); NEED_OP(t); NEED_IP(t + 3);
        do *op++ = *ip++; while (--t > 0);
        goto first_literal_run;
    }

    for (;;)
    {
        // NEED_IP(3);
        t = *ip++;
        if (t >= 16)
            goto match;
        if (t == 0)
        {
            while (*ip == 0)
            {
                t += 255;
                ip++;
                // TEST_IV(t); NEED_IP(1);
            }
            t += 15 + *ip++;
        }
        // assert(t > 0); NEED_OP(t + 3); NEED_IP(t + 6);
        {
            *op++ = *ip++; *op++ = *ip++; *op++ = *ip++;
            do *op++ = *ip++; while (--t > 0);
        }
    first_literal_run:

        t = *ip++;
        if (t >= 16)
            goto match;

        m_pos = op - (1 + M2_MAX_OFFSET);
        m_pos -= t >> 2;
        m_pos -= *ip++ << 2;

        // TEST_LB(m_pos); NEED_OP(3);
        *op++ = *m_pos++; *op++ = *m_pos++; *op++ = *m_pos;

        goto match_done;

        for (;;) {
        match:
            if (t >= 64)
            {
                m_pos = op - 1;
                m_pos -= (t >> 2) & 7;
                m_pos -= *ip++ << 3;
                t = (t >> 5) - 1;

                // TEST_LB(m_pos); assert(t > 0); NEED_OP(t + 3 - 1);
                goto copy_match;
            }
            else if (t >= 32)
            {
                t &= 31;
                if (t == 0)
                {
                    while (*ip == 0)
                    {
                        t += 255;
                        ip++;
                        // TEST_OV(t);
                        // NEED_IP(1);
                    }
                    t += 31 + *ip++;
                    // NEED_IP(2);
                }
                m_pos = op - 1;
                m_pos -= (ip[0] >> 2) + (ip[1] << 6);

                ip += 2;
            }
            else if (t >= 16)
            {
                m_pos = op;
                m_pos -= (t & 8) << 11;
                t &= 7;
                if (t == 0)
                {
                    while (*ip == 0)
                    {
                        t += 255;
                        ip++;
                        // TEST_OV(t);
                        // NEED_IP(1);
                    }
                    t += 7 + *ip++;
                    // NEED_IP(2);
                }
                m_pos -= (ip[0] >> 2) + (ip[1] << 6);

                ip += 2;
                if (m_pos == op)
                    goto eof_found;
                m_pos -= 0x4000;
            }
            else
            {
                m_pos = op - 1;
                m_pos -= t >> 2;
                m_pos -= *ip++ << 2;
                // TEST_LB(m_pos); NEED_OP(2);
                *op++ = *m_pos++; *op++ = *m_pos;
                goto match_done;
            }

            // TEST_LB(m_pos); assert(t > 0); NEED_OP(t + 3 - 1);
            {
            copy_match:
                *op++ = *m_pos++; *op++ = *m_pos++;
                do *op++ = *m_pos++; while (--t > 0);
            }

        match_done:
            t = ip[-2] & 3;
            if (t == 0)
                break;

        match_next:
            // assert(t > 0); assert(t < 4); NEED_OP(t); NEED_IP(t + 3);
            *op++ = *ip++;
            if (t > 1) { *op++ = *ip++; if (t > 2) { *op++ = *ip++; } }
            t = *ip++;
            }
        }

eof_found:
    *out_len = pd(op, out);
    return (ip == ip_end ? LZO_E_OK :
        (ip < ip_end ? LZO_E_INPUT_NOT_CONSUMED : LZO_E_INPUT_OVERRUN));

}

#define LZO_TEST_OVERRUN 1

#define LZO_TEST_OVERRUN_INPUT       2
#define LZO_TEST_OVERRUN_OUTPUT      2
#define LZO_TEST_OVERRUN_LOOKBEHIND  1

#undef TEST_IP
#undef TEST_OP
#undef TEST_IP_AND_TEST_OP
#undef TEST_LB
#undef TEST_LBO
#undef NEED_IP
#undef NEED_OP
#undef TEST_IV
#undef TEST_OV
#undef HAVE_TEST_IP
#undef HAVE_TEST_OP
#undef HAVE_NEED_IP
#undef HAVE_NEED_OP
#undef HAVE_ANY_IP
#undef HAVE_ANY_OP

#define TEST_IP             (ip < ip_end)
#define NEED_IP(x) \
            if ((lzo_uint)(ip_end - ip) < (lzo_uint)(x))  goto input_overrun
#define TEST_IV(x)          if ((x) >  (lzo_uint)0 - (511)) goto input_overrun

#define NEED_OP(x) \
            if ((lzo_uint)(op_end - op) < (lzo_uint)(x))  goto output_overrun
#define TEST_OV(x)          if ((x) >  (lzo_uint)0 - (511)) goto output_overrun


#  define TEST_LB(m_pos)        if (PTR_LT(m_pos,out) || PTR_GE(m_pos,op)) goto lookbehind_overrun
#  define TEST_LBO(m_pos,o)     if (PTR_LT(m_pos,out) || PTR_GE(m_pos,op-(o))) goto lookbehind_overrun

#define HAVE_TEST_IP 1
#define HAVE_TEST_OP 1
#define TEST_IP_AND_TEST_OP   (TEST_IP && TEST_OP)
#define HAVE_NEED_IP 1
#define HAVE_NEED_OP 1
#define HAVE_ANY_IP 1
#define HAVE_ANY_OP 1

LZO_PUBLIC(int)
lzo1x_decompress_safe(const lzo_bytep in, lzo_uint  in_len,
    lzo_bytep out, lzo_uintp out_len,
    lzo_voidp wrkmem)
{
    lzo_bytep op;
    const lzo_bytep ip;
    lzo_uint t;
    const lzo_bytep m_pos;

    const lzo_bytep const ip_end = in + in_len;
#if defined(HAVE_ANY_OP)
    lzo_bytep const op_end = out + *out_len;
#endif
#if defined(LZO1Z)
    lzo_uint last_m_off = 0;
#endif

    LZO_UNUSED(wrkmem);

    * out_len = 0;

    op = out;
    ip = in;

    NEED_IP(1);
    if (*ip > 17)
    {
        t = *ip++ - 17;
        if (t < 4)
            goto match_next;
        // assert(t > 0);
        NEED_OP(t); NEED_IP(t + 3);
        do *op++ = *ip++; while (--t > 0);
        goto first_literal_run;
    }

    for (;;)
    {
        NEED_IP(3);
        t = *ip++;
        if (t >= 16)
            goto match;
        if (t == 0)
        {
            while (*ip == 0)
            {
                t += 255;
                ip++;
                TEST_IV(t);
                NEED_IP(1);
            }
            t += 15 + *ip++;
        }
        // assert(t > 0);
        NEED_OP(t + 3); NEED_IP(t + 6);

    first_literal_run:

        t = *ip++;
        if (t >= 16)
            goto match;
        m_pos = op - (1 + M2_MAX_OFFSET);
        m_pos -= t >> 2;
        m_pos -= *ip++ << 2;

        TEST_LB(m_pos); NEED_OP(3);
        *op++ = *m_pos++; *op++ = *m_pos++; *op++ = *m_pos;
        goto match_done;

        for (;;) {
        match:
            if (t >= 64)
            {
                m_pos = op - 1;
                m_pos -= (t >> 2) & 7;
                m_pos -= *ip++ << 3;
                t = (t >> 5) - 1;
                TEST_LB(m_pos); // assert(t > 0);
                NEED_OP(t + 3 - 1);
                goto copy_match;
            }
            else if (t >= 32)
            {
                t &= 31;
                if (t == 0)
                {
                    while (*ip == 0)
                    {
                        t += 255;
                        ip++;
                        TEST_OV(t);
                        NEED_IP(1);
                    }
                    t += 31 + *ip++;
                    NEED_IP(2);
                }
                m_pos = op - 1;
                m_pos -= (ip[0] >> 2) + (ip[1] << 6);
                ip += 2;
            }
            else if (t >= 16)
            {
                m_pos = op;
                m_pos -= (t & 8) << 11;
                t &= 7;
                if (t == 0)
                {
                    while (*ip == 0)
                    {
                        t += 255;
                        ip++;
                        TEST_OV(t);
                        NEED_IP(1);
                    }
                    t += 7 + *ip++;
                    NEED_IP(2);
                }

                m_pos -= (ip[0] >> 2) + (ip[1] << 6);
                ip += 2;
                if (m_pos == op)
                    goto eof_found;
                m_pos -= 0x4000;
            }
            else
            {
                m_pos = op - 1;
                m_pos -= t >> 2;
                m_pos -= *ip++ << 2;
                TEST_LB(m_pos); NEED_OP(2);
                *op++ = *m_pos++; *op++ = *m_pos;
                goto match_done;
            }

            TEST_LB(m_pos);
            // assert(t > 0);
            NEED_OP(t + 3 - 1);
            {
            copy_match:
                *op++ = *m_pos++; *op++ = *m_pos++;
                do *op++ = *m_pos++; while (--t > 0);
            }

        match_done:
            t = ip[-2] & 3;
            if (t == 0)
                break;

        match_next:
            // assert(t > 0); assert(t < 4);
            NEED_OP(t); NEED_IP(t + 3);

            * op++ = *ip++;
            if (t > 1) { *op++ = *ip++; if (t > 2) { *op++ = *ip++; } }
            t = *ip++;
            }
        }

eof_found:
    *out_len = pd(op, out);
    return (ip == ip_end ? LZO_E_OK :
        (ip < ip_end ? LZO_E_INPUT_NOT_CONSUMED : LZO_E_INPUT_OVERRUN));

    input_overrun:
    *out_len = pd(op, out);
    return LZO_E_INPUT_OVERRUN;

    output_overrun:
    *out_len = pd(op, out);
    return LZO_E_OUTPUT_OVERRUN;

    lookbehind_overrun:
    *out_len = pd(op, out);
    return LZO_E_LOOKBEHIND_OVERRUN;
}

