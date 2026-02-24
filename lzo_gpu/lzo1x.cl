#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifndef __generic
#define __generic
#endif

#include "lzo_gpu.h"

/* LZO1X Unified Kernel
 * Supports variable dictionary sizes via D_BITS macro.
 * Default D_BITS = 14 if not defined.
 */

#ifndef D_BITS
#define D_BITS 14
#endif

#ifndef LZO_USE_UNALIGNED
#define LZO_USE_UNALIGNED 1
#endif

#ifndef LZO_USE_UNROLL2
#define LZO_USE_UNROLL2 0
#endif

/* Debug instrumentation removed in production build. */

/* Standard macros */
#define LZO_BYTE(x)       ((unsigned char) (x))

#define LZO_MAX(a,b)        ((a) >= (b) ? (a) : (b))
#define LZO_MIN(a,b)        ((a) <= (b) ? (a) : (b))

#define lzo_sizeof(type)    ((lzo_uint) (sizeof(type)))

#define DMUL(a,b) ((lzo_xint) ((a) * (b)))

/* Use generic address-space pointers */
#define lzo_memops_TU0p __generic void *
#define lzo_memops_TU1p __generic unsigned char *

#define lzo_memops_set_TU1p     volatile lzo_memops_TU1p
#define lzo_memops_move_TU1p    lzo_memops_TU1p

/* Pointer alignment test */
#define lzo_ptr_aligned(p, align_pow2) ((((lzo_uintptr_t)(p)) & ((align_pow2) - 1)) == 0)

#undef lzo_dict_p
#define lzo_dict_p __global lzo_dict_t *

#define LZO_MEMOPS_SET1(dd,cc) \
    LZO_BLOCK_BEGIN \
    *(lzo_memops_set_TU1p) (lzo_memops_TU0p) (dd) = LZO_BYTE(cc); \
    LZO_BLOCK_END

#define LZO_MEMOPS_COPY1(dd,ss)   *((__generic uchar *)(dd)) = *((__generic const uchar *)(ss))
#define LZO_MEMOPS_COPY2(dd,ss)   *((__generic ushort*)(dd)) = *((__generic const ushort*)(ss))

static inline void LZO_MEMOPS_COPY4(__generic void *dd, const __generic void *ss)
{
    if (lzo_ptr_aligned(dd,4) && lzo_ptr_aligned(ss,4))
        *((__generic uint*)dd) =  *((__generic const uint*)ss);
    else {
        uchar4 v = vload4(0, (__generic const uchar*)ss);
        vstore4(v,0,(__generic uchar*)dd);
    }
}

static inline void LZO_MEMOPS_COPY8(__generic void *dd, const __generic void *ss)
{
    if (lzo_ptr_aligned(dd,8) && lzo_ptr_aligned(ss,8))
        *((__generic ulong*)dd) = *((__generic const ulong*)ss);
    else {
        uchar8 v = vload8(0, (__generic const uchar*)ss);
        vstore8(v,0,(__generic uchar*)dd);
    }
}

/* Vectorized copy for unaligned access.
 * Default is 16-byte vectorization.
 * Define LZO_GPU_VECTOR_8 to force 8-byte maximum vectorization.
 */
static inline void LZO_MEMOPS_COPYN_FAST(__generic void *dd, const __generic void *ss, uint nn)
{
    __generic uchar *d = (__generic uchar*)dd;
    __generic const uchar *s = (__generic const uchar*)ss;
#ifndef LZO_GPU_VECTOR_8
    /* 32-byte chunks */
    while (nn >= 32) {
        uchar16 c0 = vload16(0, s);
        uchar16 c1 = vload16(1, s);
        vstore16(c0, 0, d);
        vstore16(c1, 0, d + 16);
        d += 32; s += 32; nn -= 32;
    }
    /* 16-byte chunks */
    while (nn >= 16) {
        uchar16 chunk = vload16(0, s);
        vstore16(chunk, 0, d);
        d += 16; s += 16; nn -= 16;
    }
#endif
    /* 8-byte chunks */
    while (nn >= 8) {
        uchar8 chunk = vload8(0, s);
        vstore8(chunk, 0, d);
        d += 8; s += 8; nn -= 8;
    }
    /* 4-byte chunks */
    while (nn >= 4) {
        uchar4 chunk = vload4(0, s);
        vstore4(chunk, 0, d);
        d += 4; s += 4; nn -= 4;
    }
    /* Tail bytes */
    for (; nn; --nn) *d++ = *s++;
}

#define LZO_MEMOPS_COPYN LZO_MEMOPS_COPYN_FAST  /* EXP alias */

static inline uint lzo_memops_get_le32(const __global void *pp)
{
#if LZO_USE_UNALIGNED
    return as_uint(vload4(0, (const __global uchar*)pp));
#else
    const __global uchar *p = (const __global uchar*)pp;
    if (lzo_ptr_aligned(p,4))
        return as_uint(*((const __global uint*)p));
    return  (uint)p[0] | ((uint)p[1] <<  8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
#endif
}

static inline ulong lzo_memops_get_le64(const __global void *pp)
{
#if LZO_USE_UNALIGNED
    return as_ulong(vload8(0, (const __global uchar*)pp));
#else
    const __global uchar *p = (const __global uchar*)pp;
    if (lzo_ptr_aligned(p,8))
        return as_ulong(*((const __global ulong*)p));
    return (ulong)p[0] | ((ulong)p[1] << 8) | ((ulong)p[2] << 16) | ((ulong)p[3] << 24) |
           ((ulong)p[4] << 32) | ((ulong)p[5] << 40) | ((ulong)p[6] << 48) | ((ulong)p[7] << 56);
#endif
}

#define UA_GET_LE32         lzo_memops_get_le32
#define UA_GET_LE64         lzo_memops_get_le64
#define UA_SET1             LZO_MEMOPS_SET1
#define UA_COPYN            LZO_MEMOPS_COPYN

#define pd(a,b)             ((lzo_uint) ((a)-(b)))

#define M1_MAX_OFFSET   0x0400
#define M2_MAX_OFFSET   0x0800
#define M3_MAX_OFFSET   0x4000
#define M4_MAX_OFFSET   0xbfff

#define M2_MAX_LEN      8
#define M3_MAX_LEN      33
#define M4_MAX_LEN      9

#define M3_MARKER       32
#define M4_MARKER       16

#define D_SIZE          (1u << D_BITS)
#define D_MASK          (D_SIZE - 1)
#define D_HIGH          ((D_MASK >> 1) + 1)

#define DX2(p,s1,s2) \
        (((((lzo_xint)((p)[2]) << (s2)) ^ (p)[1]) << (s1)) ^ (p)[0])
#define DX3(p,s1,s2,s3) ((DX2((p)+1,s2,s3) << (s1)) ^ (p)[0])
#define DMS(v,s)        ((lzo_uint) (((v) & (D_MASK >> (s))) << (s)))
#define DM(v)           DMS(v,0)

/* ctz: count trailing zeros for ulong */
#ifndef ctz
#define ctz(x) (63 - clz((ulong)((x) & -(long)(x))))
#endif

/* LZO1X Hash Function - Plan 2: stronger mix for single-way dictionary (from baseline) */
static inline uint lzo1x_hash32(uint dv)
{
    dv ^= dv >> 7;
    dv ^= dv >> 3;
    dv *= 0x9e3779b1u;
    dv ^= dv >> 16;
    return dv;
}

#define DINDEX(dv,p)        ((lzo1x_hash32(dv)) >> (32 - D_BITS))

/* Fingerprinting: 12 bits high for data, 20 bits low for offset (up to 1MB) */
#define DENTRY(p,in,dv)                       ((lzo_dict_t)(((lzo_uint)pd(p, in) & 0xFFFFF) | ((dv) & 0xFFF00000)))
#define DENTRY_OFF(off,dv)                    ((lzo_dict_t)(((lzo_uint)(off) & 0xFFFFF) | ((dv) & 0xFFF00000)))
#define UPDATE_I(dict,index,p,in,dv)          dict[index] = DENTRY(p,in,dv)

static inline void dict_store_packed(__global ulong* dict, uint idx, lzo_dict_t entry, uint epoch)
{
    dict[idx] = (((ulong)epoch) << 32) | (ulong)entry;
}

static inline lzo_dict_t dict_load_packed(__global const ulong* dict, uint idx, uint epoch, uint* valid)
{
    ulong packed = dict[idx];
    *valid = ((uint)(packed >> 32) == epoch);
    return (lzo_dict_t)packed;
}


static lzo_uint
lzo1x_compress_core(LZO_ADDR_GLOBAL const lzo_bytep in , lzo_uint  in_len,
                   LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
                    lzo_uint ti, __global ulong *dict, uint epoch)
{
    LZO_ADDR_GLOBAL const lzo_bytep ip;
    LZO_ADDR_GLOBAL lzo_bytep op;
    const LZO_ADDR_GLOBAL lzo_bytep in_end = in + in_len;
    const LZO_ADDR_GLOBAL lzo_bytep ip_end = in + in_len - 20;
    LZO_ADDR_GLOBAL const lzo_bytep ii;
    op = out;
    ip = in;
    ii = ip;

    ip += ti < 4 ? 4 - ti : 0;
    for (;;)
    {
        LZO_ADDR_GLOBAL const lzo_bytep m_pos;
        lzo_uint m_off;
        lzo_uint m_len;
        lzo_uint32_t dv = 0;
        lzo_uint dindex = 0;
        lzo_uint saved_dindex = 0;
        {
    literal:
            ip += 1 + ((ip - ii) >> 5);
    next:
            if (ip + 8 < ip_end)
            {
                ulong v8a = UA_GET_LE64(ip);
                ulong v8b = UA_GET_LE64(ip + 4);
                uint4 dvs_a = (uint4)((uint)v8a, (uint)(v8a >> 8), (uint)(v8a >> 16), (uint)(v8a >> 24));
                uint4 dvs_b = (uint4)((uint)v8b, (uint)(v8b >> 8), (uint)(v8b >> 16), (uint)(v8b >> 24));
                lzo_uint ip_off = pd(ip, in);

                // Vectorized hashes
                uint4 h_a = dvs_a ^ (dvs_a >> 7); h_a ^= (h_a >> 3); h_a *= 0x9e3779b1u; h_a ^= (h_a >> 16);
                uint4 h_b = dvs_b ^ (dvs_b >> 7); h_b ^= (h_b >> 3); h_b *= 0x9e3779b1u; h_b ^= (h_b >> 16);
                uint4 idx_a = (h_a >> (32 - D_BITS));
                uint4 idx_b = (h_b >> (32 - D_BITS));

                // Iter 1-8 with Fingerprinting
                lzo_dict_t ent;
                uint valid;
#define CHECK_MATCH(idx, current_dv) \
                ent = dict_load_packed(dict, idx, epoch, &valid); \
                if (valid && ent != 0 && (ent & 0xFFF00000) == (current_dv & 0xFFF00000)) { \
                    m_off = ent & 0xFFFFF; \
                    if (ip_off > m_off && (ip_off - m_off) <= M4_MAX_OFFSET) { \
                        m_pos = in + m_off; \
                        if (current_dv == UA_GET_LE32(m_pos)) { \
                            dv = current_dv; saved_dindex = idx; goto match_found; \
                        } \
                    } \
                } \
                dict_store_packed(dict, idx, DENTRY_OFF(ip_off, current_dv), epoch);

                CHECK_MATCH(idx_a.s0, dvs_a.s0);
                ip++; ip_off++; CHECK_MATCH(idx_a.s1, dvs_a.s1);
                ip++; ip_off++; CHECK_MATCH(idx_a.s2, dvs_a.s2);
                ip++; ip_off++; CHECK_MATCH(idx_a.s3, dvs_a.s3);
                ip++; ip_off++; CHECK_MATCH(idx_b.s0, dvs_b.s0);
                ip++; ip_off++; CHECK_MATCH(idx_b.s1, dvs_b.s1);
                ip++; ip_off++; CHECK_MATCH(idx_b.s2, dvs_b.s2);
                ip++; ip_off++; CHECK_MATCH(idx_b.s3, dvs_b.s3);
#undef CHECK_MATCH

                ip++;
                goto literal;
            }
            ip++;
    next_slow:
            if (ip >= ip_end)
                break;
            dv = UA_GET_LE32(ip);
            lzo_uint ip_off = pd(ip, in);
            dindex = DINDEX(dv,ip);

            uint valid = 0;
            lzo_dict_t ent = dict_load_packed(dict, dindex, epoch, &valid);
            if (valid && ent != 0 && (ent & 0xFFF00000) == (dv & 0xFFF00000)) {
                m_off = ent & 0xFFFFF;
                if (ip_off > m_off && (ip_off - m_off) <= M4_MAX_OFFSET) {
                    m_pos = in + m_off;
                    if (dv == UA_GET_LE32(m_pos)) {
                        saved_dindex = dindex;
                        goto match_found;
                    }
                }
            }

            dict_store_packed(dict, dindex, DENTRY_OFF(ip_off,dv), epoch);

            goto literal;
        }

        match_found:
        /* Match found, continue with match encoding */
        ii -= ti; ti = 0;
        lzo_uint t = pd(ip,ii);
        if (t != 0)
        {
            if (t <= 3)
            {
                op[-2] = LZO_BYTE(op[-2] | t);
                /* Byte-by-byte copy for small literals */
                do *op++ = *ii++; while (--t > 0);
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
                    *op++ = LZO_BYTE(tt);
                }
                /* Use vectorized copy for larger literals */
                LZO_MEMOPS_COPYN_FAST(op, ii, t);
                op += t; ii += t;
            }
        }

        m_len = 4;
        /* Vectorized match check (unrolled if requested) */
#ifdef LZO_USE_UNROLL2
        while (ip + m_len + 16 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);

            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += (ctz(diff) >> 3);
                goto m_len_done;
            }
            ip_val = UA_GET_LE64(ip + m_len + 8);
            mp_val = UA_GET_LE64(m_pos + m_len + 8);
            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += 8 + (ctz(diff) >> 3);
                goto m_len_done;
            }
            m_len += 16;
        }
        while (ip + m_len + 8 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);

            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += (ctz(diff) >> 3);
                goto m_len_done;
            }
            m_len += 8;
        }
#else
        while (ip + m_len + 8 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);

            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += (ctz(diff) >> 3);
                goto m_len_done;
            }
            m_len += 8;
        }
#endif

        while (ip + m_len + 4 <= ip_end) {
            uint ip32 = UA_GET_LE32(ip + m_len);
            uint mp32 = UA_GET_LE32(m_pos + m_len);
            if (ip32 != mp32) {
                m_len += (ctz((ulong)(ip32 ^ mp32)) >> 3);
                goto m_len_done;
            }
            m_len += 4;
        }
        while (ip + m_len < ip_end && ip[m_len] == m_pos[m_len]) {
            m_len += 1;
        }
m_len_done:
        /* Update dictionary entry with current position */
        dict_store_packed(dict, saved_dindex, DENTRY(ip,in,dv), epoch);

        m_off = pd(ip,m_pos);
        ip += m_len;
        ii = ip;

        uint is_m2 = (m_len <= M2_MAX_LEN) & (m_off <= M2_MAX_OFFSET);
        uint is_m3 = (m_off <= M3_MAX_OFFSET) & !is_m2;

        if (is_m2) {
            m_off -= 1;
            *op++ = LZO_BYTE(((m_len - 1) << 5) | ((m_off & 7) << 2));
            *op++ = LZO_BYTE(m_off >> 3);
        }
        else if (is_m3) {
            m_off -= 1;
            if (m_len <= M3_MAX_LEN)
                *op++ = LZO_BYTE(M3_MARKER | (m_len - 2));
            else {
                m_len -= M3_MAX_LEN;
                *op++ = M3_MARKER | 0;
                while(m_len > 255) {
                    m_len -= 255;
                    UA_SET1(op, 0);
                    op++;
                }
                *op++ = LZO_BYTE(m_len);
            }
            *op++ = LZO_BYTE(m_off << 2);
            *op++ = LZO_BYTE(m_off >> 6);
        }
        else {
            m_off -= 0x4000;
            if (m_len <= M4_MAX_LEN)
                *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8) | (m_len - 2));
            else {
                m_len -= M4_MAX_LEN;
                *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8));
                while(m_len > 255) {
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

static __global uchar* lzo1x_compress_terminate(__global const uchar* ip, uint in_len_dummy, __global uchar* op, lzo_uint t)
{
    if (t > 0)
    {
        __global const uchar* ii = ip - t;
        if (t <= 3)
            op[-2] = LZO_BYTE(op[-2] | t);
        else if (t <= 18)
            *op++ = LZO_BYTE(t - 3);
        else
        {
            lzo_uint tt = t - 18;
            *op++ = 0;
            while (tt > 255) { tt -= 255; *op++ = 0; }
            *op++ = LZO_BYTE(tt);
        }
        LZO_MEMOPS_COPYN_FAST(op, ii, t);
        op += t;
    }
    *op++ = M4_MARKER | 1;
    *op++ = 0; *op++ = 0;
    return op;
}

static void do_compress(__global const uchar* in, uint in_len, __global uchar* out, lzo_uintp out_len, lzo_uint ti, __global ulong *dict, uint epoch)
{
    lzo_uint t = ti;
    __global uchar* op = out;

    lzo_uint olen = 0;
    t = lzo1x_compress_core(in, in_len, op, &olen, t, dict, epoch);
    op += olen;
    // Terminate the block
    op = lzo1x_compress_terminate(in + in_len, 0, op, t);

    *out_len = (lzo_uint)(op - out);
}

__kernel void lzo1x_block_compress(__global const uchar *in ,
                                   __global       uchar *out,
                                   __global       uint  *out_len,
                                   const uint  in_sz,
                                   const uint  blk_size,
                                   const uint  worst_blk,
                                   __global ulong *dict_pool,
                                   const uint  dict_pool_size,
                                   const uint  epoch_base)
{
    const uint wi = get_global_id(0);
    const uint total_wi = get_global_size(0);
    const uint total_blocks = (in_sz + blk_size - 1) / blk_size;
    const uint dict_elems = (1u << D_BITS);

    if (wi >= dict_pool_size) return;

    __global ulong *dict = dict_pool + ((size_t)wi * dict_elems);

    for (uint b = wi; b < total_blocks; b += total_wi) {
        uint epoch = epoch_base + b + 1u;
        uint in_off = b * blk_size;
        __global const uchar* ip = in + in_off;
        __global uchar* op = out + b * worst_blk;
        uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);

        lzo_uint olen;
        do_compress(ip, in_len, op, &olen, 0, dict, epoch);
        out_len[b] = (uint)olen;
    }
}
