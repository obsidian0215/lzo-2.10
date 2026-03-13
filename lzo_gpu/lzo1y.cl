#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifndef __generic
#define __generic
#endif

typedef __generic unsigned char * lzo_bytep;
typedef __generic void *          lzo_voidp;
typedef __generic unsigned int *  lzo_uintp;

#include "lzo_gpu.h"

#undef lzo_bytep
#undef lzo_voidp
#undef lzo_uintp
#define lzo_bytep __generic unsigned char *
#define lzo_voidp __generic void *
#define lzo_uintp __generic unsigned int *


/* LZO1Y Global Dictionary Kernel
 * Supports variable dictionary sizes via D_BITS macro.
 */

#ifndef D_BITS
#define D_BITS 14
#endif

#ifndef LZO_USE_UNALIGNED
#define LZO_USE_UNALIGNED 1
#endif

#ifndef LZO_USE_UNROLL2
#define LZO_USE_UNROLL2 1
#endif

/* Standard macros */
#define LZO_BYTE(x)       ((unsigned char) (x))

#define LZO_MAX(a,b)        ((a) >= (b) ? (a) : (b))
#define LZO_MIN(a,b)        ((a) <= (b) ? (a) : (b))

#define lzo_sizeof(type)    ((lzo_uint) (sizeof(type)))
#define DMUL(a,b)           ((lzo_xint) ((a) * (b)))

/* OpenCL 2.0 Generic Memory Access */
#define lzo_memops_TU0p __generic void *
#define lzo_memops_TU1p __generic unsigned char *

#define lzo_memops_set_TU1p     volatile lzo_memops_TU1p
#define lzo_memops_move_TU1p    lzo_memops_TU1p

#define lzo_ptr_aligned(p, align_pow2) ((((lzo_uintptr_t)(p)) & ((align_pow2) - 1)) == 0)

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

static inline uint lzo_memops_get_le32(const __generic void *pp)
{
#if LZO_USE_UNALIGNED
    return as_uint(vload4(0, (const __generic uchar*)pp));
#else
    const __generic uchar *p = (const __generic uchar*)pp;
    return  (uint)p[0] | ((uint)p[1] <<  8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
#endif
}

static inline ulong lzo_memops_get_le64(const __generic void *pp)
{
#if LZO_USE_UNALIGNED
    return as_ulong(vload8(0, (const __generic uchar*)pp));
#else
    const __generic uchar *p = (const __generic uchar*)pp;
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
#define M2_MAX_OFFSET   0x0400
#define M3_MAX_OFFSET   0x4000
#define M4_MAX_OFFSET   0xbfff

#define M2_MAX_LEN      14
#define M3_MAX_LEN      33
#define M4_MAX_LEN      9

#define M3_MARKER       32
#define M4_MARKER       16

#define D_SIZE          (1u << D_BITS)
#define D_MASK          (D_SIZE - 1)

#ifndef ctz
#define ctz(x) (63 - clz((ulong)((x) & -(long)(x))))
#endif

static inline uint lzo1y_hash32(uint dv)
{
    dv ^= dv >> 7;
    dv ^= dv >> 3;
    dv *= 0x9e3779b1u;
    dv ^= dv >> 16;
    return dv;
}

#define DINDEX(dv,p)        ((lzo1y_hash32(dv)) >> (32 - D_BITS))
/* 32-bit packed dictionary: epoch_12 (bits 31:20) | offset_20 (bits 19:0) */
#define DICT_EPOCH_SHIFT 20
#define DICT_OFF_MASK    0x000FFFFFu
#define DICT_EPOCH_MASK  0xFFF00000u

static inline void dict_store32(__global uint* dict, uint idx, uint offset, uint epoch)
{
    dict[idx] = ((epoch & 0xFFFu) << DICT_EPOCH_SHIFT) | (offset & DICT_OFF_MASK);
}

static inline uint dict_load32(__global const uint* dict, uint idx, uint epoch, uint* valid)
{
    uint entry = dict[idx];
    *valid = (((entry >> DICT_EPOCH_SHIFT) & 0xFFFu) == (epoch & 0xFFFu));
    return entry & DICT_OFF_MASK;
}

static lzo_uint
lzo1y_compress_core(__generic const lzo_bytep in , lzo_uint  in_len,
                   __generic lzo_bytep out, lzo_uintp out_len,
                    lzo_uint ti, __global uint *dict, uint epoch)
{
    __generic const lzo_bytep ip;
    __generic lzo_bytep op;
    const __generic lzo_bytep in_end = in + in_len;
    const __generic lzo_bytep ip_end = in + in_len - 20;
    __generic const lzo_bytep ii;

    op = out;
    ip = in;
    ii = ip;

    ip += ti < 4 ? 4 - ti : 0;
    for (;;)
    {
        __generic const lzo_bytep m_pos;
        lzo_uint m_off;
        lzo_uint m_len;
        uint dv = 0;
        uint dindex = 0;
        lzo_uint saved_dindex = 0;
        {
    literal:
            ip += 1 + ((ip - ii) >> 5);
    next:
            if (ip + 8 < ip_end)
            {
                ulong v8a = UA_GET_LE64(ip);
                uint4 dvs_a = (uint4)((uint)v8a, (uint)(v8a >> 8), (uint)(v8a >> 16), (uint)(v8a >> 24));
                lzo_uint ip_off = pd(ip, in);

                uint4 h_a = dvs_a ^ (dvs_a >> 7); h_a ^= (h_a >> 3); h_a *= 0x9e3779b1u; h_a ^= (h_a >> 16);
                uint4 idx_a = (h_a >> (32 - D_BITS));
                // Batch load all 4 dict entries (reads), then check, then batch store (writes)
                // Separating reads from writes improves memory controller scheduling
                uint valid0, valid1, valid2, valid3;
                uint off0 = dict_load32(dict, idx_a.s0, epoch, &valid0);
                uint off1 = dict_load32(dict, idx_a.s1, epoch, &valid1);
                uint off2 = dict_load32(dict, idx_a.s2, epoch, &valid2);
                uint off3 = dict_load32(dict, idx_a.s3, epoch, &valid3);

                // Check position 0
                if (valid0 && off0 != 0 && ip_off > off0 && (ip_off - off0) <= M4_MAX_OFFSET) {
                    m_pos = in + off0;
                    if (dvs_a.s0 == UA_GET_LE32(m_pos)) {
                        dv = dvs_a.s0; saved_dindex = idx_a.s0;
                        dict_store32(dict, idx_a.s0, (uint)ip_off, epoch);
                        goto match_found;
                    }
                }
                // Check position 1
                ip++; ip_off++;
                if (valid1 && off1 != 0 && ip_off > off1 && (ip_off - off1) <= M4_MAX_OFFSET) {
                    m_pos = in + off1;
                    if (dvs_a.s1 == UA_GET_LE32(m_pos)) {
                        dv = dvs_a.s1; saved_dindex = idx_a.s1;
                        dict_store32(dict, idx_a.s0, (uint)(ip_off - 1), epoch);
                        dict_store32(dict, idx_a.s1, (uint)ip_off, epoch);
                        goto match_found;
                    }
                }
                // Check position 2
                ip++; ip_off++;
                if (valid2 && off2 != 0 && ip_off > off2 && (ip_off - off2) <= M4_MAX_OFFSET) {
                    m_pos = in + off2;
                    if (dvs_a.s2 == UA_GET_LE32(m_pos)) {
                        dv = dvs_a.s2; saved_dindex = idx_a.s2;
                        dict_store32(dict, idx_a.s0, (uint)(ip_off - 2), epoch);
                        dict_store32(dict, idx_a.s1, (uint)(ip_off - 1), epoch);
                        dict_store32(dict, idx_a.s2, (uint)ip_off, epoch);
                        goto match_found;
                    }
                }
                // Check position 3
                ip++; ip_off++;
                if (valid3 && off3 != 0 && ip_off > off3 && (ip_off - off3) <= M4_MAX_OFFSET) {
                    m_pos = in + off3;
                    if (dvs_a.s3 == UA_GET_LE32(m_pos)) {
                        dv = dvs_a.s3; saved_dindex = idx_a.s3;
                        dict_store32(dict, idx_a.s0, (uint)(ip_off - 3), epoch);
                        dict_store32(dict, idx_a.s1, (uint)(ip_off - 2), epoch);
                        dict_store32(dict, idx_a.s2, (uint)(ip_off - 1), epoch);
                        dict_store32(dict, idx_a.s3, (uint)ip_off, epoch);
                        goto match_found;
                    }
                }
                // No match found: batch store all 4 positions
                dict_store32(dict, idx_a.s0, (uint)(ip_off - 3), epoch);
                dict_store32(dict, idx_a.s1, (uint)(ip_off - 2), epoch);
                dict_store32(dict, idx_a.s2, (uint)(ip_off - 1), epoch);
                dict_store32(dict, idx_a.s3, (uint)ip_off, epoch);

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
            m_off = dict_load32(dict, dindex, epoch, &valid);
            if (valid && m_off != 0) {
                if (ip_off > m_off && (ip_off - m_off) <= M4_MAX_OFFSET) {
                    m_pos = in + m_off;
                    if (dv == UA_GET_LE32(m_pos)) {
                        saved_dindex = dindex;
                        goto match_found;
                    }
                }
            }
            dict_store32(dict, dindex, (uint)ip_off, epoch);
            goto literal;
        }

        match_found:
        ii -= ti; ti = 0;
        lzo_uint t = pd(ip,ii);
        if (t != 0)
        {
            if (t <= 3)
            {
                op[-2] = LZO_BYTE(op[-2] | t);
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
                    while (tt > 255) { tt -= 255; UA_SET1(op, 0); op++; }
                    *op++ = LZO_BYTE(tt);
                }
                UA_COPYN(op, ii, t);
                op += t; ii += t;
            }
        }

        m_len = 4;
#ifdef LZO_USE_UNROLL2
        while (ip + m_len + 16 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);
            if (ip_val != mp_val) {
                m_len += (ctz(ip_val ^ mp_val) >> 3);
                goto m_len_done;
            }
            ip_val = UA_GET_LE64(ip + m_len + 8);
            mp_val = UA_GET_LE64(m_pos + m_len + 8);
            if (ip_val != mp_val) {
                m_len += 8 + (ctz(ip_val ^ mp_val) >> 3);
                goto m_len_done;
            }
            m_len += 16;
        }
#endif
        while (ip + m_len + 8 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);
            if (ip_val != mp_val) {
                m_len += (ctz(ip_val ^ mp_val) >> 3);
                goto m_len_done;
            }
            m_len += 8;
        }
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
    dict_store32(dict, saved_dindex, (uint)pd(ip, in), epoch);
        m_off = pd(ip,m_pos);
        ip += m_len;
        ii = ip;

        uint is_m2 = (m_len <= M2_MAX_LEN) & (m_off <= M2_MAX_OFFSET);
        uint is_m3 = (m_off <= M3_MAX_OFFSET) & !is_m2;

        if (is_m2) {
            m_off -= 1;
            *op++ = LZO_BYTE(((m_len + 1) << 4) | ((m_off & 3) << 2));
            *op++ = LZO_BYTE(m_off >> 2);
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

static __generic uchar* lzo1y_compress_terminate(__generic const uchar* ip, uint in_len_dummy, __generic uchar* op, lzo_uint t, __generic uchar* out_origin)
{
    if (t > 0)
    {
        __generic const uchar* ii = ip - t;
        if (op == out_origin && t <= 238)
            *op++ = LZO_BYTE(17 + t);
        else if (t <= 3)
            op[-2] = LZO_BYTE(op[-2] | t);
        else if (t <= 18)
            *op++ = LZO_BYTE(t - 3);
        else
        {
            lzo_uint tt = t - 18;
            *op++ = 0;
            while (tt > 255) { tt -= 255; UA_SET1(op, 0); op++; }
            *op++ = LZO_BYTE(tt);
        }
        UA_COPYN(op, ii, (uint)t);
        op += t;
    }
    *op++ = M4_MARKER | 1;
    *op++ = 0; *op++ = 0;
    return op;
}

static void do_compress(__global const uchar* in, uint in_len, __global uchar* out, lzo_uintp out_len, lzo_uint ti, __global uint* dict, uint epoch
                        )
{
    lzo_uint t = ti;
    __generic uchar* op = (__generic uchar*)out;

    lzo_uint olen = 0;
    t = lzo1y_compress_core(in, in_len, op, &olen, t, dict, epoch);
    op += olen;
    op = lzo1y_compress_terminate(in + in_len, 0, op, t, (__generic uchar*)out);
    *out_len = (lzo_uint)(op - (__generic uchar*)out);
}

__kernel void lzo1y_block_compress(__global const uchar *in ,
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
    const uint dict_elems = (1u << D_BITS);

    if (wi >= dict_pool_size) return;

    __global uint *dict = dict_pool + ((size_t)wi * dict_elems);

    uint nblk = (in_sz + blk_size - 1) / blk_size;

    for (uint bidx = wi; bidx < nblk; bidx += total_wi) {
        uint epoch = epoch_base + bidx + 1u;
        uint in_off = bidx * blk_size;
        uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);

        __global const uchar* ip = in + in_off;
        __global uchar* op = out + bidx * worst_blk;

        lzo_uint olen = 0;
        do_compress(ip, in_len, op, &olen, 0, dict, epoch);
        out_len[bidx] = (uint)olen;
    }
}

/* ---------------- lzo1y decompress ---------------- */

#define M2_MAX_OFFSET 0x0400

static inline void COPY_MATCH(__generic uchar *op, __generic const uchar *m_pos, uint len)
{
    uint offset = op - m_pos;
    if (offset >= len) {
        UA_COPYN(op, m_pos, len);
        return;
    }
    if (offset <= 4) {
        if (offset == 1) {
            uchar c = *m_pos;
            uchar16 v16 = (uchar16)c;
            while (len >= 64) {
                vstore16(v16, 0, op);
                vstore16(v16, 1, op);
                vstore16(v16, 2, op);
                vstore16(v16, 3, op);
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
            uchar16 v0 = vload16(0, m_pos);
            uchar16 v1 = vload16(1, m_pos);
            uchar16 v2 = vload16(2, m_pos);
            uchar16 v3 = vload16(3, m_pos);
            vstore16(v0, 0, op); vstore16(v1, 1, op);
            vstore16(v2, 2, op); vstore16(v3, 3, op);
            op += 64; m_pos += 64; len -= 64;
        }
    }
    if (offset >= 32) {
        while (len >= 32) {
            uchar16 v0 = vload16(0, m_pos);
            uchar16 v1 = vload16(1, m_pos);
            vstore16(v0, 0, op);
            vstore16(v1, 0, op + 16);
            op += 32; m_pos += 32; len -= 32;
        }
    }
    if (offset >= 16) {
        while (len >= 16) {
            vstore16(vload16(0, m_pos), 0, op);
            op += 16; m_pos += 16; len -= 16;
        }
    }
    if (offset >= 8) {
        while (len >= 8) {
            vstore8(vload8(0, m_pos), 0, op);
            op += 8; m_pos += 8; len -= 8;
        }
    }
    if (offset >= 4) {
        if (len >= 4) {
            vstore4(vload4(0, m_pos), 0, op);
            op += 4; m_pos += 4; len -= 4;
        }
    }
    while (len > 0) { *op++ = *m_pos++; len--; }
}

static lzo_uint
lzo1y_decompress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint in_len,
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
            while (*ip == 0) {
                t += 255; ip++;
            }
            t += 15 + *ip++;
        }
        {
            uint copy_len = (uint)(3 + t);
            UA_COPYN(op, ip, copy_len);
            op += copy_len;
            ip += copy_len;
        }
    first_literal_run:
        t = *ip++;
        if (t >= 16) goto match;
        m_pos = op - (1 + M2_MAX_OFFSET);
        m_pos -= t >> 2;
        m_pos -= *ip++ << 2;
        *op++ = *m_pos++;
        *op++ = *m_pos++;
        *op++ = *m_pos;
        goto match_done;

        for (;;) {
        match:
            if (t >= 64) {
                m_pos = op - 1;
                m_pos -= (t >> 2) & 3;
                m_pos -= *ip++ << 2;
                t = (t >> 4) - 3;
                goto copy_match;
            } else if (t >= 32) {
                t &= 31;
                if (t == 0) {
                    while (*ip == 0) {
                        t += 255;
                        ip++;
                    }
                    t += 31 + *ip++;
                }
                m_pos = op - 1 - (ip[0] >> 2) - (ip[1] << 6); ip += 2;
            } else if (t >= 16) {
                m_pos = op - ((t & 8) << 11);
                t &= 7;
                if (t == 0) {
                    while (*ip == 0) {
                        t += 255;
                        ip++;
                    }
                    t += 7 + *ip++;
                }
                m_pos -= (ip[0] >> 2) + (ip[1] << 6);
                ip += 2;
                if (m_pos == op)
                    goto eof_found;
                m_pos -= 0x4000;
            } else {
                m_pos = op - 1 - (t >> 2) - (*ip++ << 2);
                *op++ = *m_pos++;
                *op++ = *m_pos;
                goto match_done;
            }
        copy_match:
            {
                uint mlen = t + 2;
                uint moff = (uint)(op - m_pos);
                if (moff >= mlen) {
                    UA_COPYN(op, m_pos, mlen);
                } else {
                    COPY_MATCH(op, m_pos, mlen);
                }
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

__kernel void lzo1y_block_decompress(
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
    lzo1y_decompress(in_buf + in_off, in_len, out_buf + out_off, &out_len, NULL);
    out_lens[gid] = out_len;
}

__kernel void lzo_pack_compressed_blocks(__global const uchar* sparse_out,
                                         __global const uint* block_lens,
                                         __global uchar* packed_out,
                                         __global const uint* packed_offsets,
                                         uint worst_blk,
                                         uint total_blocks)
{
    uint gid = get_global_id(0);
    uint gsz = get_global_size(0);
    for (uint idx = gid; idx < total_blocks; idx += gsz) {
        uint len = block_lens[idx];
        if (len == 0) continue;
        uchar const* src = sparse_out + idx * worst_blk;
        uchar* dst = packed_out + packed_offsets[idx];
        for (uint i = 0; i < len; ++i) dst[i] = src[i];
    }
}
