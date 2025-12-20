#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "lzo_gpu.h"

/* LZO1Y Decompression Kernel
 * Based on lzo1x_decomp.cl but with M2_MAX_OFFSET = 0x0400
 */

/* --- minimal required macros and helpers --- */
#define LZO_BYTE(x)       ((unsigned char) (x))

#define lzo_sizeof(type)    ((lzo_uint) (sizeof(type)))

#define DMUL(a,b) ((lzo_xint) ((a) * (b)))

#define lzo_memops_TU0p __generic void *
#define lzo_memops_TU1p __generic unsigned char *

#define lzo_memops_set_TU1p     volatile lzo_memops_TU1p
#define lzo_memops_move_TU1p    lzo_memops_TU1p

static inline bool lzo_ptr_aligned(const void *p, uint align_pow2)
{   return (((ulong) p) & (align_pow2 - 1)) == 0; }

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

static inline void LZO_MEMOPS_COPY16(__generic void *dd, const __generic void *ss)
{
    if (lzo_ptr_aligned(dd,16) && lzo_ptr_aligned(ss,16)) {
        if (sizeof(unsigned long) >= 8) {
            *((__generic ulong*)dd) = *((__generic const ulong*)ss);
            *((__generic ulong*)((__generic uchar*)dd + 8)) = *((__generic const ulong*)((__generic const uchar*)ss + 8));
        } else {
            uchar8 a = vload8(0, (__generic const uchar*)ss);
            uchar8 b = vload8(8, (__generic const uchar*)ss);
            vstore8(a,0,(__generic uchar*)dd);
            vstore8(b,8,(__generic uchar*)dd);
        }
    } else {
        uchar16 v = vload16(0, (__generic const uchar*)ss);
        vstore16(v,0,(__generic uchar*)dd);
    }
}

static inline void LZO_MEMOPS_COPYN(__generic void *dd, const __generic void *ss, uint nn)
{
    __generic uchar *d = (__generic uchar*)dd;
    __generic const uchar *s = (__generic const uchar*)ss;

    if (nn == 0) return;

    if (s > d && s < d + nn) {
        __generic uchar *de = d + nn;
        __generic const uchar *se = s + nn;
        while (nn >= 16 && lzo_ptr_aligned((void*)(de - 16),16) && lzo_ptr_aligned((void*)(se - 16),16)) {
            de -= 16; se -= 16; LZO_MEMOPS_COPY16(de, se); nn -= 16; }
        while (nn >= 8 && lzo_ptr_aligned((void*)(de - 8),8) && lzo_ptr_aligned((void*)(se - 8),8)) {
            de -= 8; se -= 8; LZO_MEMOPS_COPY8(de, se); nn -= 8; }
        while (nn >= 4 && lzo_ptr_aligned((void*)(de - 4),4) && lzo_ptr_aligned((void*)(se - 4),4)) {
            de -= 4; se -= 4; LZO_MEMOPS_COPY4(de, se); nn -= 4; }
        while (nn) { de--; se--; *de = *se; nn--; }
    } else {
        while (nn >= 16 && lzo_ptr_aligned(d,16) && lzo_ptr_aligned(s,16))
        {   LZO_MEMOPS_COPY16(d,s); d+=16; s+=16; nn-=16; }
        while (nn >= 8 && lzo_ptr_aligned(d,8) && lzo_ptr_aligned(s,8))
        {   LZO_MEMOPS_COPY8(d,s); d+=8; s+=8; nn-=8; }
        while (nn >= 4 && lzo_ptr_aligned(d,4) && lzo_ptr_aligned(s,4))
        {   LZO_MEMOPS_COPY4(d,s); d+=4; s+=4; nn-=4; }
        for (; nn; --nn) *d++ = *s++;
    }
}

static inline uint lzo_memops_get_le32(const __generic void *pp)
{
    const __generic uchar *p = (__generic const uchar*)pp;

    if (lzo_ptr_aligned(p,4))
        return as_uint(*(__generic const uint*)p);

    return  (uint)p[0]        |
           ((uint)p[1] <<  8) |
           ((uint)p[2] << 16) |
           ((uint)p[3] << 24);
}

#define LZO_MEMOPS_GET_LE32(ss)    lzo_memops_get_le32(ss)

#define UA_SET1             LZO_MEMOPS_SET1
#define UA_COPY1            LZO_MEMOPS_COPY1
#define UA_COPY2            LZO_MEMOPS_COPY2
#define UA_COPY4            LZO_MEMOPS_COPY4
#define UA_COPY8            LZO_MEMOPS_COPY8
#define UA_COPYN            LZO_MEMOPS_COPYN
#define UA_GET_LE32         LZO_MEMOPS_GET_LE32

static inline void COPY_MATCH(__generic uchar *op, __generic const uchar *m_pos, uint len)
{
    while (len--) {
        *op++ = *m_pos++;
    }
}

#define pd(a,b)             ((lzo_uint) ((a)-(b)))

/* LZO1Y Specific Constants */
#define M2_MAX_OFFSET   0x0400
#define M4_MARKER       16

static lzo_uint
lzo1y_decompress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint  in_len,
    LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
    lzo_voidp wrkmem)
{
    LZO_ADDR_GLOBAL lzo_bytep op;
    LZO_ADDR_GLOBAL const lzo_bytep ip;
    lzo_uint t;
    LZO_ADDR_GLOBAL const lzo_bytep m_pos;

    const LZO_ADDR_GLOBAL lzo_bytep const ip_end = in + in_len;
    LZO_UNUSED(wrkmem);

    *out_len = 0;

    op = out;
    ip = in;

    if (*ip > 17)
    {
        t = *ip++ - 17;
        if (t < 4)
            goto match_next;
        UA_COPYN(op, ip, (uint)t);
        op += t; ip += t;
        goto first_literal_run;
    }

    for (;;)
    {
        t = *ip++;
        if (t >= 16)
            goto match;
        if (t == 0)
        {
            while (*ip == 0)
            {
                t += 255;
                ip++;
            }
            t += 15 + *ip++;
        }
        {
            uint copy_len = (uint)(3 + t);
            UA_COPYN(op, ip, copy_len);
            op += copy_len; ip += copy_len;
        }
    first_literal_run:
        t = *ip++;
        if (t >= 16)
            goto match;

        m_pos = op - (1 + M2_MAX_OFFSET);
        m_pos -= t >> 2;
        m_pos -= *ip++ << 2;

        *op++ = *m_pos++; *op++ = *m_pos++; *op++ = *m_pos;
        goto match_done;

        for (;;) {
        match:
            if (t >= 64)
            {
                /* LZO1Y M2 match: mask and shifts differ from LZO1X */
                m_pos = op - 1;
                m_pos -= (t >> 2) & 3;
                m_pos -= *ip++ << 2;
                t = (t >> 4) - 3;
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
                    }
                    t += 31 + *ip++;
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
                    }
                    t += 7 + *ip++;
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
                *op++ = *m_pos++; *op++ = *m_pos;
                goto match_done;
            }
        copy_match:
            {
                uint copy_len = (uint)(2 + t);
                COPY_MATCH(op, m_pos, copy_len);
                op += copy_len; m_pos += copy_len;
            }

        match_done:
            t = ip[-2] & 3;
            if (t == 0)
                break;

        match_next:
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

__kernel void lzo1y_block_decompress(
    __global const uchar* in_buf,
    __global const uint* off_arr,
    __global       uchar* out_buf,
    __global       uint* out_lens,
    uint blk_sz,
    uint orig_size,
    uint nblk)
{
    uint gid = get_global_id(0);
    if (gid >= nblk) return;

    uint in_off = off_arr[gid];
    uint in_len = off_arr[gid + 1] - in_off;

    uint out_off = gid * blk_sz;
    uint out_len = (out_off + blk_sz <= orig_size) ?
        blk_sz : (orig_size - out_off);

    __global const uchar* src = in_buf + in_off;
    __global       uchar* dst = out_buf + out_off;

    lzo1y_decompress(src, in_len, dst, &out_len, NULL);

    out_lens[gid] = out_len;
}
