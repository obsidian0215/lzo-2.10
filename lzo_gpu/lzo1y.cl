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

/* Debug instrumentation removed in production build. */

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
    const __generic uchar *p = (const __generic uchar*)pp;
    return  (uint)p[0] | ((uint)p[1] <<  8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

static inline ulong lzo_memops_get_le64(const __generic void *pp)
{
    const __generic uchar *p = (const __generic uchar*)pp;
    return (ulong)p[0] | ((ulong)p[1] << 8) | ((ulong)p[2] << 16) | ((ulong)p[3] << 24) |
           ((ulong)p[4] << 32) | ((ulong)p[5] << 40) | ((ulong)p[6] << 48) | ((ulong)p[7] << 56);
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
/* Fingerprinting: 12 bits high for data, 20 bits low for offset (up to 1MB) */
#define DENTRY(p,in,dv)     ((lzo_dict_t)(((lzo_uint)pd(p, in) & 0xFFFFF) | ((dv) & 0xFFF00000)))

static lzo_uint
lzo1y_compress_core(__generic const lzo_bytep in , lzo_uint  in_len,
                   __generic lzo_bytep out, lzo_uintp out_len,
                    lzo_uint ti, lzo_dict_p dict)
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
                ulong v8b = UA_GET_LE64(ip + 4);
                uint4 dvs_a = (uint4)((uint)v8a, (uint)(v8a >> 8), (uint)(v8a >> 16), (uint)(v8a >> 24));
                uint4 dvs_b = (uint4)((uint)v8b, (uint)(v8b >> 8), (uint)(v8b >> 16), (uint)(v8b >> 24));

                uint4 h_a = dvs_a ^ (dvs_a >> 7); h_a ^= (h_a >> 3); h_a *= 0x9e3779b1u; h_a ^= (h_a >> 16);
                uint4 h_b = dvs_b ^ (dvs_b >> 7); h_b ^= (h_b >> 3); h_b *= 0x9e3779b1u; h_b ^= (h_b >> 16);
                uint4 idx_a = (h_a >> (32 - D_BITS));
                uint4 idx_b = (h_b >> (32 - D_BITS));
                lzo_dict_t m_val;

                // Iter 1-4
                m_val = dict[idx_a.s0]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_a.s0 & 0xFFF00000) && dvs_a.s0 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_a.s0; saved_dindex = idx_a.s0; goto match_found;
                }
                dict[idx_a.s0] = DENTRY(ip, in, dvs_a.s0);

                ip++; m_val = dict[idx_a.s1]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_a.s1 & 0xFFF00000) && dvs_a.s1 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_a.s1; saved_dindex = idx_a.s1; goto match_found;
                }
                dict[idx_a.s1] = DENTRY(ip, in, dvs_a.s1);

                ip++; m_val = dict[idx_a.s2]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_a.s2 & 0xFFF00000) && dvs_a.s2 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_a.s2; saved_dindex = idx_a.s2; goto match_found;
                }
                dict[idx_a.s2] = DENTRY(ip, in, dvs_a.s2);

                ip++; m_val = dict[idx_a.s3]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_a.s3 & 0xFFF00000) && dvs_a.s3 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_a.s3; saved_dindex = idx_a.s3; goto match_found;
                }
                dict[idx_a.s3] = DENTRY(ip, in, dvs_a.s3);

                // Iter 5-8
                ip++; m_val = dict[idx_b.s0]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_b.s0 & 0xFFF00000) && dvs_b.s0 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_b.s0; saved_dindex = idx_b.s0; goto match_found;
                }
                dict[idx_b.s0] = DENTRY(ip, in, dvs_b.s0);

                ip++; m_val = dict[idx_b.s1]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_b.s1 & 0xFFF00000) && dvs_b.s1 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_b.s1; saved_dindex = idx_b.s1; goto match_found;
                }
                dict[idx_b.s1] = DENTRY(ip, in, dvs_b.s1);

                ip++; m_val = dict[idx_b.s2]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_b.s2 & 0xFFF00000) && dvs_b.s2 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_b.s2; saved_dindex = idx_b.s2; goto match_found;
                }
                dict[idx_b.s2] = DENTRY(ip, in, dvs_b.s2);

                ip++; m_val = dict[idx_b.s3]; m_off = m_val & 0xFFFFF; m_pos = in + m_off;
                if (m_off != 0 && (m_val & 0xFFF00000) == (dvs_b.s3 & 0xFFF00000) && dvs_b.s3 == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                    dv = dvs_b.s3; saved_dindex = idx_b.s3; goto match_found;
                }
                dict[idx_b.s3] = DENTRY(ip, in, dvs_b.s3);

                ip++;
                goto literal;
            }
            ip++;
    next_slow:
            if (ip >= ip_end)
                break;
            dv = UA_GET_LE32(ip);
            dindex = DINDEX(dv,ip);
            lzo_dict_t m_val = dict[dindex];
            m_off = m_val & 0xFFFFF;
            m_pos = in + m_off;

            if (m_off != 0 && (m_val & 0xFFF00000) == (dv & 0xFFF00000) && dv == UA_GET_LE32(m_pos) && pd(ip, m_pos) <= M4_MAX_OFFSET) {
                saved_dindex = dindex;
                goto match_found;
            }
            dict[dindex] = DENTRY(ip,in,dv);
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
        while (ip + m_len + 8 <= ip_end) {
            ulong ip_val = UA_GET_LE64(ip + m_len);
            ulong mp_val = UA_GET_LE64(m_pos + m_len);
            if (ip_val != mp_val) {
                m_len += (ctz(ip_val ^ mp_val) >> 3);
                goto m_len_done;
            }
            m_len += 8;
        }
        while (ip + m_len < ip_end && ip[m_len] == m_pos[m_len]) m_len++;

m_len_done:
        dict[saved_dindex] = DENTRY(ip,in,dv);
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
        for (uint i = 0; i < t; i++) *op++ = *ii++;
    }
    *op++ = M4_MARKER | 1;
    *op++ = 0; *op++ = 0;
    return op;
}

static inline void dict_clear_global(__global lzo_dict_t* dict) {
    __global uint16 *p = (__global uint16 *) dict;
    uint tid = get_local_id(0);
    uint sz = get_local_size(0);
    // entries = (1 << D_BITS). Each uint16 is 16 * 4 bytes = 64 bytes = 16 entries.
    uint n = (1 << D_BITS) >> 4;
    for (uint i = tid; i < n; i += sz) {
        p[i] = (uint16)0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

static void do_compress(__global const uchar* in, uint in_len, __global uchar* out, lzo_uintp out_len, lzo_uint ti, __global lzo_dict_t* dict)
{
    lzo_uint t = ti;
    __generic uchar* op = (__generic uchar*)out;

    dict_clear_global(dict);

    if (get_local_id(0) == 0) {
        lzo_uint olen = 0;
        t = lzo1y_compress_core(in, in_len, op, &olen, t, dict);
        op += olen;
        op = lzo1y_compress_terminate(in + in_len, 0, op, t, (__generic uchar*)out);
        *out_len = (lzo_uint)(op - (__generic uchar*)out);
    }
}

__kernel void lzo1y_block_compress(__global const uchar *in ,
                                   __global       uchar *out,
                                   __global       uint  *out_len,
                                   const uint  in_sz,
                                   const uint  blk_size,
                                   const uint  worst_blk,
                                   __global lzo_dict_t *dict_pool,
                                   const uint  dict_pool_size)
{
    const uint gid = get_group_id(0);
    const uint num_groups = get_num_groups(0);
    __global lzo_dict_t *dict = dict_pool + ((size_t)gid << D_BITS);

    uint nblk = (in_sz + blk_size - 1) / blk_size;

    for (uint bidx = gid; bidx < nblk; bidx += num_groups) {
        uint in_off = bidx * blk_size;
        uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);

        __global const uchar* ip = in + in_off;
        __global uchar* op = out + bidx * worst_blk;

        lzo_uint olen;
        do_compress(ip, in_len, op, &olen, 0, dict);
        if (get_local_id(0) == 0) {
            out_len[bidx] = (uint)olen;
        }
    }
}
