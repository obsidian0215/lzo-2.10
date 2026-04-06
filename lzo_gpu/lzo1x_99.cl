#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifndef __generic
#define __generic
#endif

#include "lzo_gpu.h"

/*
 * LZO1X Enhanced Greedy Kernel (Level 99 / 方案B)
 *
 * Enhancements over standard greedy (lzo1x.cl):
 *   1. 4-way set-associative dictionary — each hash bucket stores 4
 *      candidate positions, best match among them is selected.
 *   2. Lazy matching — after finding a match at ip, probe ip+1;
 *      if ip+1 yields a longer match (by >=2) or same length with
 *      shorter offset, emit ip as literal and use the ip+1 match.
 *
 * D_BITS is fixed at 14 (16384 buckets × 4 ways = 65536 uint32 entries).
 * Dict layout: dict_pool[wi * DICT_TOTAL + bucket * 4 + way]
 */

#define D_BITS 14

#ifndef LZO_USE_UNALIGNED
#define LZO_USE_UNALIGNED 1
#endif

#define LZO_BYTE(x)       ((unsigned char) (x))
#define LZO_MAX(a,b)        ((a) >= (b) ? (a) : (b))
#define LZO_MIN(a,b)        ((a) <= (b) ? (a) : (b))
#define lzo_sizeof(type)    ((lzo_uint) (sizeof(type)))
#define DMUL(a,b) ((lzo_xint) ((a) * (b)))

#define lzo_memops_TU0p __generic void *
#define lzo_memops_TU1p __generic unsigned char *
#define lzo_memops_set_TU1p     volatile lzo_memops_TU1p
#define lzo_memops_move_TU1p    lzo_memops_TU1p

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

static inline void LZO_MEMOPS_COPYN_FAST(__generic void *dd, const __generic void *ss, uint nn)
{
    __generic uchar *d = (__generic uchar*)dd;
    __generic const uchar *s = (__generic const uchar*)ss;
#ifndef LZO_GPU_VECTOR_8
    while (nn >= 32) {
        uchar16 c0 = vload16(0, s);
        uchar16 c1 = vload16(1, s);
        vstore16(c0, 0, d);
        vstore16(c1, 0, d + 16);
        d += 32; s += 32; nn -= 32;
    }
    while (nn >= 16) {
        uchar16 chunk = vload16(0, s);
        vstore16(chunk, 0, d);
        d += 16; s += 16; nn -= 16;
    }
#endif
    while (nn >= 8) {
        uchar8 chunk = vload8(0, s);
        vstore8(chunk, 0, d);
        d += 8; s += 8; nn -= 8;
    }
    while (nn >= 4) {
        uchar4 chunk = vload4(0, s);
        vstore4(chunk, 0, d);
        d += 4; s += 4; nn -= 4;
    }
    for (; nn; --nn) *d++ = *s++;
}

#define LZO_MEMOPS_COPYN LZO_MEMOPS_COPYN_FAST

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

#define WAYS            4
#define DICT_TOTAL      (D_SIZE * WAYS)

#ifndef ctz
#define ctz(x) (63 - clz((ulong)((x) & -(long)(x))))
#endif

static inline uint lzo1x_hash32(uint dv)
{
    dv ^= dv >> 7;
    dv ^= dv >> 3;
    dv *= 0x9e3779b1u;
    dv ^= dv >> 16;
    return dv;
}

#define HASH_BUCKET(dv)  ((lzo1x_hash32(dv)) >> (32 - D_BITS))

/* 4-way dictionary: epoch:12 | offset:20, stored as 4 consecutive uint32 per bucket */
#define DICT_EPOCH_SHIFT 20
#define DICT_OFF_MASK    0x000FFFFFu

static inline void dict4_store(__global uint *dict, uint bucket, uint way,
                               uint offset, uint epoch)
{
    dict[bucket * WAYS + way] = ((epoch & 0xFFFu) << DICT_EPOCH_SHIFT) | (offset & DICT_OFF_MASK);
}

static inline uint dict4_load(__global const uint *dict, uint bucket, uint way,
                              uint epoch, uint *valid)
{
    uint entry = dict[bucket * WAYS + way];
    *valid = (((entry >> DICT_EPOCH_SHIFT) & 0xFFFu) == (epoch & 0xFFFu));
    return entry & DICT_OFF_MASK;
}

static inline void dict4_insert(__global uint *dict, uint bucket,
                                uint offset, uint epoch)
{
    /* shift ways 2→3, 1→2, 0→1, insert new at way 0 */
    dict[bucket * WAYS + 3] = dict[bucket * WAYS + 2];
    dict[bucket * WAYS + 2] = dict[bucket * WAYS + 1];
    dict[bucket * WAYS + 1] = dict[bucket * WAYS + 0];
    dict[bucket * WAYS + 0] = ((epoch & 0xFFFu) << DICT_EPOCH_SHIFT) | (offset & DICT_OFF_MASK);
}

/* compute match length starting from byte 0 (caller already verified first 4 bytes) */
static inline lzo_uint compute_match_len(LZO_ADDR_GLOBAL const lzo_bytep ip,
                                         LZO_ADDR_GLOBAL const lzo_bytep m_pos,
                                         const LZO_ADDR_GLOBAL lzo_bytep ip_end)
{
    lzo_uint m_len = 4;
    while (ip + m_len + 16 <= ip_end) {
        ulong ip_val = UA_GET_LE64(ip + m_len);
        ulong mp_val = UA_GET_LE64(m_pos + m_len);
        if (ip_val != mp_val) {
            m_len += (ctz(ip_val ^ mp_val) >> 3);
            return m_len;
        }
        ip_val = UA_GET_LE64(ip + m_len + 8);
        mp_val = UA_GET_LE64(m_pos + m_len + 8);
        if (ip_val != mp_val) {
            m_len += 8 + (ctz(ip_val ^ mp_val) >> 3);
            return m_len;
        }
        m_len += 16;
    }
    while (ip + m_len + 8 <= ip_end) {
        ulong ip_val = UA_GET_LE64(ip + m_len);
        ulong mp_val = UA_GET_LE64(m_pos + m_len);
        if (ip_val != mp_val) {
            m_len += (ctz(ip_val ^ mp_val) >> 3);
            return m_len;
        }
        m_len += 8;
    }
    while (ip + m_len + 4 <= ip_end) {
        uint ip32 = UA_GET_LE32(ip + m_len);
        uint mp32 = UA_GET_LE32(m_pos + m_len);
        if (ip32 != mp32) {
            m_len += (ctz((ulong)(ip32 ^ mp32)) >> 3);
            return m_len;
        }
        m_len += 4;
    }
    while (ip + m_len < ip_end && ip[m_len] == m_pos[m_len])
        m_len += 1;
    return m_len;
}

/* find best match among 4 ways; returns match length (0 if none) */
static inline lzo_uint find_best_match_4way(
    __global uint *dict, uint bucket, uint epoch,
    LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint ip_off,
    const LZO_ADDR_GLOBAL lzo_bytep ip_end,
    LZO_ADDR_GLOBAL const lzo_bytep *best_m_pos)
{
    LZO_ADDR_GLOBAL const lzo_bytep ip = in + ip_off;
    uint dv = UA_GET_LE32(ip);
    lzo_uint best_len = 0;
    lzo_uint best_off = 0;

    for (uint w = 0; w < WAYS; w++) {
        uint valid;
        uint off = dict4_load(dict, bucket, w, epoch, &valid);
        if (!valid || off == 0 || ip_off <= off)
            continue;
        lzo_uint m_off = ip_off - off;
        if (m_off > M4_MAX_OFFSET)
            continue;
        LZO_ADDR_GLOBAL const lzo_bytep m_pos = in + off;
        if (dv != UA_GET_LE32(m_pos))
            continue;
        lzo_uint m_len = compute_match_len(ip, m_pos, ip_end);
        /* prefer longer match, or same length with shorter offset */
        if (m_len > best_len || (m_len == best_len && m_off < best_off)) {
            best_len = m_len;
            best_off = m_off;
            *best_m_pos = m_pos;
        }
    }
    return best_len;
}

static inline __global uchar* emit_literals(
    __global uchar *op,
    LZO_ADDR_GLOBAL const lzo_bytep ii, lzo_uint t)
{
    if (t <= 3) {
        op[-2] = LZO_BYTE(op[-2] | t);
        do *op++ = *ii++; while (--t > 0);
    } else {
        if (t <= 18)
            *op++ = LZO_BYTE(t - 3);
        else {
            lzo_uint tt = t - 18;
            *op++ = 0;
            while (tt > 255) { tt -= 255; UA_SET1(op, 0); op++; }
            *op++ = LZO_BYTE(tt);
        }
        LZO_MEMOPS_COPYN_FAST(op, ii, t);
        op += t;
    }
    return op;
}

static inline __global uchar* emit_match(
    __global uchar *op, lzo_uint m_len, lzo_uint m_off)
{
    uint is_m2 = (m_len <= M2_MAX_LEN) & (m_off <= M2_MAX_OFFSET);
    uint is_m3 = (m_off <= M3_MAX_OFFSET) & !is_m2;

    if (is_m2) {
        m_off -= 1;
        *op++ = LZO_BYTE(((m_len - 1) << 5) | ((m_off & 7) << 2));
        *op++ = LZO_BYTE(m_off >> 3);
    } else if (is_m3) {
        m_off -= 1;
        if (m_len <= M3_MAX_LEN)
            *op++ = LZO_BYTE(M3_MARKER | (m_len - 2));
        else {
            lzo_uint ml = m_len - M3_MAX_LEN;
            *op++ = M3_MARKER | 0;
            while (ml > 255) { ml -= 255; UA_SET1(op, 0); op++; }
            *op++ = LZO_BYTE(ml);
        }
        *op++ = LZO_BYTE(m_off << 2);
        *op++ = LZO_BYTE(m_off >> 6);
    } else {
        m_off -= 0x4000;
        if (m_len <= M4_MAX_LEN)
            *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8) | (m_len - 2));
        else {
            lzo_uint ml = m_len - M4_MAX_LEN;
            *op++ = LZO_BYTE(M4_MARKER | ((m_off >> 11) & 8));
            while (ml > 255) { ml -= 255; UA_SET1(op, 0); op++; }
            *op++ = LZO_BYTE(ml);
        }
        *op++ = LZO_BYTE(m_off << 2);
        *op++ = LZO_BYTE(m_off >> 6);
    }
    return op;
}

static lzo_uint
lzo1x_99_compress_core(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint in_len,
                       LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
                       lzo_uint ti, __global uint *dict, uint epoch)
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
        LZO_ADDR_GLOBAL const lzo_bytep m_pos = 0;
        lzo_uint m_len = 0;
        lzo_uint m_off = 0;

    literal:
        ip += 1 + ((ip - ii) >> 5);

    next:
        if (ip >= ip_end)
            break;

        {
            uint dv = UA_GET_LE32(ip);
            uint bucket = HASH_BUCKET(dv);
            lzo_uint ip_off = pd(ip, in);

            m_len = find_best_match_4way(dict, bucket, epoch, in, ip_off, ip_end, &m_pos);
            dict4_insert(dict, bucket, (uint)ip_off, epoch);

            if (m_len < 4)
                goto literal;

            m_off = pd(ip, m_pos);

            /* Lazy matching: check ip+1 for a potentially better match */
            if (ip + 1 < ip_end) {
                uint dv1 = UA_GET_LE32(ip + 1);
                uint bucket1 = HASH_BUCKET(dv1);
                lzo_uint ip1_off = ip_off + 1;
                LZO_ADDR_GLOBAL const lzo_bytep lazy_m_pos = 0;

                lzo_uint lazy_len = find_best_match_4way(dict, bucket1, epoch, in, ip1_off, ip_end, &lazy_m_pos);

                /* use lazy match if it's significantly better */
                if (lazy_len >= m_len + 2 ||
                    (lazy_len >= m_len && pd(ip + 1, lazy_m_pos) < m_off)) {
                    /* skip ip as literal, use ip+1's match */
                    dict4_insert(dict, bucket1, (uint)ip1_off, epoch);
                    ip++;
                    m_len = lazy_len;
                    m_pos = lazy_m_pos;
                    m_off = pd(ip, m_pos);
                }
            }
        }

        /* emit pending literals */
        ii -= ti; ti = 0;
        {
            lzo_uint t = pd(ip, ii);
            if (t != 0)
                op = emit_literals(op, ii, t);
        }

        /* emit the match */
        op = emit_match(op, m_len, m_off);

        ip += m_len;
        ii = ip;
        goto next;
    }

    *out_len = pd(op, out);
    return pd(in_end, ii - ti);
}

static __global uchar* lzo1x_99_compress_terminate(
    __global const uchar* ip, __global uchar* op, lzo_uint t)
{
    if (t > 0) {
        __global const uchar* ii = ip - t;
        if (t <= 3)
            op[-2] = LZO_BYTE(op[-2] | t);
        else if (t <= 18)
            *op++ = LZO_BYTE(t - 3);
        else {
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

static void do_compress_99(__global const uchar* in, uint in_len,
                           __global uchar* out, lzo_uintp out_len,
                           lzo_uint ti, __global uint *dict, uint epoch)
{
    lzo_uint t = ti;
    __global uchar* op = out;
    lzo_uint olen = 0;

    t = lzo1x_99_compress_core(in, in_len, op, &olen, t, dict, epoch);
    op += olen;
    op = lzo1x_99_compress_terminate(in + in_len, op, t);

    *out_len = (lzo_uint)(op - out);
}

__kernel void lzo1x_block_compress_99(
    __global const uchar *in,
    __global       uchar *out,
    __global       uint  *out_len,
    const uint  in_sz,
    const uint  blk_size,
    const uint  worst_blk,
    __global uint *dict_pool,
    const uint  dict_pool_size,
    const uint  epoch_base)
{
    const uint wi = get_global_id(0);
    const uint total_wi = get_global_size(0);
    const uint total_blocks = (in_sz + blk_size - 1) / blk_size;

    if (wi >= dict_pool_size) return;

    __global uint *dict = dict_pool + ((size_t)wi * DICT_TOTAL);

    for (uint b = wi; b < total_blocks; b += total_wi) {
        uint epoch = epoch_base + b + 1u;
        uint in_off = b * blk_size;
        __global const uchar* ip = in + in_off;
        __global uchar* op = out + b * worst_blk;
        uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);

        lzo_uint olen = 0;
        do_compress_99(ip, in_len, op, &olen, 0, dict, epoch);
        out_len[b] = (uint)olen;
    }
}

/* -------- lzo1x decompress (identical to standard kernel) -------- */

#define M2_MAX_OFFSET_D 0x0800

static inline void COPY_MATCH(__generic uchar *op, __generic const uchar *m_pos, uint len)
{
    uint offset = op - m_pos;
    if (offset >= len) {
        UA_COPYN(op, m_pos, len);
        return;
    }
    if (offset <= 4) {
        if (offset == 1) {
            uchar c = *m_pos; uchar16 v16 = (uchar16)c;
            while (len >= 64) {
                vstore16(v16, 0, op); vstore16(v16, 1, op);
                vstore16(v16, 2, op); vstore16(v16, 3, op);
                op += 64; len -= 64;
            }
            while (len >= 16) { vstore16(v16, 0, op); op += 16; len -= 16; }
            if (len >= 8) { vstore8(v16.lo, 0, op); op += 8; len -= 8; }
            if (len >= 4) { vstore4(v16.s0123, 0, op); op += 4; len -= 4; }
            if (len > 0) { *op++ = c; if (len > 1) { *op++ = c; if (len > 2) *op++ = c; } }
            return;
        }
        if (offset == 2) {
            uchar p0 = m_pos[0], p1 = m_pos[1];
            uchar16 v16 = (uchar16)(p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1, p0, p1);
            while (len >= 16) { vstore16(v16, 0, op); op += 16; len -= 16; }
            if (len >= 8) { vstore8(v16.lo, 0, op); op += 8; len -= 8; }
            if (len >= 4) { vstore4(v16.s0123, 0, op); op += 4; len -= 4; }
            while (len >= 2) { *op++ = p0; *op++ = p1; len -= 2; }
            if (len) *op++ = p0;
            return;
        }
        if (offset == 3) {
            uchar p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2];
            while (len >= 3) { *op++ = p0; *op++ = p1; *op++ = p2; len -= 3; }
            if (len == 2) { *op++ = p0; *op++ = p1; } else if (len == 1) { *op++ = p0; }
            return;
        }
        if (offset == 4) {
            uchar p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2], p3 = m_pos[3];
            uchar16 v16 = (uchar16)(p0, p1, p2, p3, p0, p1, p2, p3, p0, p1, p2, p3, p0, p1, p2, p3);
            while (len >= 16) { vstore16(v16, 0, op); op += 16; len -= 16; }
            if (len >= 8) { vstore8(v16.lo, 0, op); op += 8; len -= 8; }
            if (len >= 4) { vstore4(v16.s0123, 0, op); op += 4; len -= 4; }
            while (len--) { *op = *m_pos; op++; m_pos++; }
            return;
        }
    }
    if (offset >= 64) {
        while (len >= 64) {
            uchar16 v0 = vload16(0, m_pos); uchar16 v1 = vload16(1, m_pos);
            uchar16 v2 = vload16(2, m_pos); uchar16 v3 = vload16(3, m_pos);
            vstore16(v0, 0, op); vstore16(v1, 1, op);
            vstore16(v2, 2, op); vstore16(v3, 3, op);
            op += 64; m_pos += 64; len -= 64;
        }
    }
    if (offset >= 32) {
        while (len >= 32) {
            uchar16 v0 = vload16(0, m_pos); uchar16 v1 = vload16(1, m_pos);
            vstore16(v0, 0, op); vstore16(v1, 0, op + 16);
            op += 32; m_pos += 32; len -= 32;
        }
    }
    if (offset >= 16) { while (len >= 16) { vstore16(vload16(0, m_pos), 0, op); op += 16; m_pos += 16; len -= 16; } }
    if (offset >= 8) { while (len >= 8) { vstore8(vload8(0, m_pos), 0, op); op += 8; m_pos += 8; len -= 8; } }
    if (offset >= 4) { if (len >= 4) { vstore4(vload4(0, m_pos), 0, op); op += 4; m_pos += 4; len -= 4; } }
    while (len > 0) { *op++ = *m_pos++; len--; }
}

static lzo_uint
lzo1x_decompress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint in_len,
    LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
    lzo_voidp wrkmem)
{
    LZO_ADDR_GLOBAL lzo_bytep op = out;
    LZO_ADDR_GLOBAL const lzo_bytep ip = in;
    lzo_uint t;
    LZO_ADDR_GLOBAL const lzo_bytep m_pos;
    *out_len = 0;

    if (*ip > 17) {
        t = *ip++ - 17;
        if (t < 4) goto match_next;
        UA_COPYN(op, ip, (uint)t);
        op += t; ip += t;
        goto first_literal_run;
    }

    for (;;) {
        t = *ip++;
        if (t >= 16) goto match;
        if (t == 0) {
            while (*ip == 0) { t += 255; ip++; }
            t += 15 + *ip++;
        }
        {
            uint copy_len = (uint)(3 + t);
            UA_COPYN(op, ip, copy_len);
            op += copy_len; ip += copy_len;
        }
    first_literal_run:
        t = *ip++;
        if (t >= 16) goto match;
        m_pos = op - (1 + M2_MAX_OFFSET_D);
        m_pos -= t >> 2;
        m_pos -= *ip++ << 2;
        *op++ = *m_pos++; *op++ = *m_pos++; *op++ = *m_pos;
        goto match_done;

        for (;;) {
        match:
            if (t >= 64) {
                m_pos = op - 1 - ((t >> 2) & 7) - (*ip++ << 3);
                t = (t >> 5) - 1;
                goto copy_match;
            } else if (t >= 32) {
                t &= 31;
                if (t == 0) {
                    while (*ip == 0) { t += 255; ip++; }
                    t += 31 + *ip++;
                }
                m_pos = op - 1 - (ip[0] >> 2) - (ip[1] << 6);
                ip += 2;
            } else if (t >= 16) {
                m_pos = op - ((t & 8) << 11);
                t &= 7;
                if (t == 0) {
                    while (*ip == 0) { t += 255; ip++; }
                    t += 7 + *ip++;
                }
                m_pos -= (ip[0] >> 2) + (ip[1] << 6);
                ip += 2;
                if (m_pos == op) goto eof_found;
                m_pos -= 0x4000;
            } else {
                m_pos = op - 1 - (t >> 2) - (*ip++ << 2);
                *op++ = *m_pos++; *op++ = *m_pos;
                goto match_done;
            }
        copy_match:
            {
                uint mlen = t + 2;
                uint moff = (uint)(op - m_pos);
                if (moff >= mlen) UA_COPYN(op, m_pos, mlen);
                else COPY_MATCH(op, m_pos, mlen);
                op += mlen;
            }
        match_done:
            t = ip[-2] & 3;
            if (t == 0) break;
        match_next:
            *op++ = *ip++;
            if (t > 1) { *op++ = *ip++; if (t > 2) *op++ = *ip++; }
            t = *ip++;
        }
    }
eof_found:
    *out_len = pd(op, out);
    return LZO_E_OK;
}

__kernel void lzo1x_block_decompress(
    __global const uchar* in_buf, __global const uint* off_arr,
    __global const uint* comp_lens,
    __global       uchar* out_buf, __global uint* out_lens,
    uint blk_sz, uint orig_size, uint nblk)
{
    uint gid = get_global_id(0);
    if (gid >= nblk) return;
    uint in_off = off_arr[gid];
    uint in_len = comp_lens[gid];
    uint out_off = gid * blk_sz;
    uint out_len = (out_off + blk_sz <= orig_size) ? blk_sz : (orig_size - out_off);
    lzo1x_decompress(in_buf + in_off, in_len, out_buf + out_off, &out_len, NULL);
    out_lens[gid] = out_len;
}

__kernel void lzo_pack_compressed_blocks(__global const uchar* sparse_out,
                                          __global const uint* block_lens,
                                          __global uchar* packed_out,
                                          __global const uint* packed_offsets,
                                          uint worst_blk,
                                          uint total_blocks)
{
    uint blk = get_group_id(0);
    uint lane = get_local_id(0);
    uint lanes = get_local_size(0);

    if (blk >= total_blocks) return;

    {
        uint len = block_lens[blk];
        __global const uchar* src = sparse_out + blk * worst_blk;
        __global uchar* dst = packed_out + packed_offsets[blk];

        if (len == 0) return;

        if (len <= 32u) {
            if (lane == 0) {
                uint pos = 0;
                if (len >= 16u) {
                    uchar16 c16 = vload16(0, src);
                    vstore16(c16, 0, dst);
                    pos = 16u;
                }
                if (len - pos >= 8u) {
                    uchar8 c8 = vload8(0, src + pos);
                    vstore8(c8, 0, dst + pos);
                    pos += 8u;
                }
                for (; pos < len; ++pos) dst[pos] = src[pos];
            }
            return;
        }

        if (len <= 128u) {
            uint vec16_end = len & ~15u;
            for (uint pos = lane * 16u; pos < vec16_end; pos += lanes * 16u) {
                uchar16 c = vload16(0, src + pos);
                vstore16(c, 0, dst + pos);
            }
            for (uint pos = vec16_end + lane; pos < len; pos += lanes) {
                dst[pos] = src[pos];
            }
            return;
        }

        uint vec32_end = len & ~31u;
        for (uint pos = lane * 32u; pos < vec32_end; pos += lanes * 32u) {
            uchar16 c0 = vload16(0, src + pos);
            uchar16 c1 = vload16(0, src + pos + 16u);
            vstore16(c0, 0, dst + pos);
            vstore16(c1, 0, dst + pos + 16u);
        }

        uint vec16_end = len & ~15u;
        for (uint pos = vec32_end + lane * 16u; pos < vec16_end; pos += lanes * 16u) {
            uchar16 c = vload16(0, src + pos);
            vstore16(c, 0, dst + pos);
        }

        for (uint pos = vec16_end + lane; pos < len; pos += lanes) {
            dst[pos] = src[pos];
        }
    }
}
