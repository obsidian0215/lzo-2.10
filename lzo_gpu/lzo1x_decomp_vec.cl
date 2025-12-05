#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "minilzo.h"

/* Vectorized LZO decompressor variant (renamed to lzo1x_decomp_vec.cl)
 * This file is a copy of `lzo1x_decomp.cl` with modified
 * memory copy helpers to prefer 16-byte (uchar16) vector loads/stores
 * when source and destination are suitably aligned.
 */

/* --- minimal required macros and helpers (copied from lzo1x_1.cl) --- */
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

/* new: 16-byte copy using uchar16 */
static inline void LZO_MEMOPS_COPY16(__generic void *dd, const __generic void *ss)
{
    if (lzo_ptr_aligned(dd,16) && lzo_ptr_aligned(ss,16)) {
        /* copy in two 8-byte stores if compiler doesn't support 16-byte atomics */
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
        /* use vector ops when possible */
        uchar16 v = vload16(0, (__generic const uchar*)ss);
        vstore16(v,0,(__generic uchar*)dd);
    }
}

static inline void LZO_MEMOPS_COPYN(__generic void *dd, const __generic void *ss, uint nn)
{
    __generic uchar *d = (__generic uchar*)dd;
    __generic const uchar *s = (__generic const uchar*)ss;

    if (nn == 0) return;
    /* memmove semantics: if src in (d, d+nn) -> backward copy */
    if (s > d && s < d + nn) {
        __generic uchar *de = d + nn;
        __generic const uchar *se = s + nn;
        while (nn >= 16 && lzo_ptr_aligned((void*)(de - 16),16) && lzo_ptr_aligned((void*)(se - 16),16)) {
            de -= 16; se -= 16; LZO_MEMOPS_COPY16(de,se); nn -= 16; }
        while (nn >= 8 && lzo_ptr_aligned((void*)(de - 8),8) && lzo_ptr_aligned((void*)(se - 8),8)) {
            de -= 8; se -= 8; LZO_MEMOPS_COPY8(de,se); nn -= 8; }
        while (nn >= 4 && lzo_ptr_aligned((void*)(de - 4),4) && lzo_ptr_aligned((void*)(se - 4),4)) {
            de -= 4; se -= 4; LZO_MEMOPS_COPY4(de,se); nn -= 4; }
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

#define UA_SET1             LZO_MEMOPS_SET1
#define UA_COPY1            LZO_MEMOPS_COPY1
#define UA_COPY2            LZO_MEMOPS_COPY2
#define UA_COPY4            LZO_MEMOPS_COPY4
#define UA_COPY8            LZO_MEMOPS_COPY8
#define UA_COPYN            LZO_MEMOPS_COPYN
#define UA_GET_LE32         LZO_MEMOPS_GET_LE32


/* 优化的匹配拷贝函数 - 用于解压中的COPY指令 (向量化优化) */
static inline void COPY_MATCH(__generic uchar *op, __generic const uchar *m_pos, uint len)
{
    /* 向量化匹配拷贝 (ROI: ⭐⭐⭐⭐)
     * 优化策略:
     * 1. 使用uchar16/uchar8向量拷贝长匹配
     * 2. 处理重叠拷贝 (offset < 16)
     * 3. RLE模式检测和优化
     * 4. 尾部使用标量拷贝
     */

    uint offset = op - m_pos;
    __generic const uchar *end_src = m_pos + len;
    int overlap = (m_pos < op && end_src > op);

    /* 处理 overlap 场景，保留 memmove 语义：只为安全的小 pattern 使用 fast-path，
     * 其余情况退回到 scalar copy 以避免破坏源数据。
     */
    if (overlap) {
        if (offset == 1) {
            uchar c = *m_pos;
            uchar16 fill16 = (uchar16)(c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c);
            uchar8 fill8 = (uchar8)(c,c,c,c,c,c,c,c);
            while (len >= 16) { vstore16(fill16, 0, op); op += 16; len -= 16; }
            while (len >= 8) { vstore8(fill8, 0, op); op += 8; len -= 8; }
            while (len--) *op++ = c;
            return;
        }
        if (offset == 2 && len >= 8) {
            uchar p0 = m_pos[0], p1 = m_pos[1];
            uchar8 pat8 = (uchar8)(p0,p1,p0,p1,p0,p1,p0,p1);
            while (len >= 8) { vstore8(pat8, 0, op); op += 8; m_pos += 8; len -= 8; }
            while (len >= 2) { *op++ = p0; *op++ = p1; len -= 2; }
            if (len) *op++ = p0; return;
        }
        if (offset == 3 && len >= 3) {
            /* overlapping 3-byte repeating pattern: use memmove-like loop to preserve semantics */
            while (len >= 3) {
                *op++ = *m_pos++;
                *op++ = *m_pos++;
                *op++ = *m_pos++;
                len -= 3;
            }
            if (len == 2) { *op++ = *m_pos++; *op++ = *m_pos++; }
            else if (len == 1) { *op++ = *m_pos++; }
            return;
        }
        if (offset == 4 && len >= 8) {
            uchar p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2], p3 = m_pos[3];
            uchar16 pat16 = (uchar16)(p0,p1,p2,p3,p0,p1,p2,p3,p0,p1,p2,p3,p0,p1,p2,p3);
            uchar8 pat8 = (uchar8)(p0,p1,p2,p3,p0,p1,p2,p3);
            while (len >= 16) { vstore16(pat16, 0, op); op += 16; len -= 16; }
            while (len >= 8) { vstore8(pat8, 0, op); op += 8; len -= 8; }
            while (len >= 4) { *op++ = p0; *op++ = p1; *op++ = p2; *op++ = p3; len -= 4; }
            /* 尾部处理：剩余 0-3 字节 */
            if (len >= 1) *op++ = p0;
            if (len >= 2) *op++ = p1;
            if (len >= 3) *op++ = p2;
            return;
        }
        if (offset == 6 && len >= 8) {
            /* For overlapping offset==6, vectorized 8-byte copies are unsafe because
             * 8 is not a multiple of 6; fall back to memmove-safe generic copy.
             */
            UA_COPYN(op, m_pos, len);
            return;
        }
        if (offset == 8 && len >= 8) {
            /* overlapping 8-byte repeating pattern: 8-byte chunk copies are safe */
            while (len >= 8) {
                uchar8 v = vload8(0, m_pos);
                vstore8(v, 0, op);
                op += 8; m_pos += 8; len -= 8;
            }
            while (len--) *op++ = *m_pos++;
            return;
        }
        if (offset == 16 && len >= 16) {
            /* overlapping 16-byte repeating pattern: 16-byte chunk copies are safe */
            while (len >= 16) {
                uchar16 v = vload16(0, m_pos);
                vstore16(v, 0, op);
                op += 16; m_pos += 16; len -= 16;
            }
            if (len >= 8) { uchar8 v = vload8(0, m_pos); vstore8(v, 0, op); op += 8; m_pos += 8; len -= 8; }
            if (len >= 4) { uchar4 v = vload4(0, m_pos); vstore4(v, 0, op); op += 4; m_pos += 4; len -= 4; }
            while (len--) *op++ = *m_pos++;
            return;
        }
        /* Other overlapping cases: fallback to scalar UA_COPYN (memmove-safe) */
        UA_COPYN(op, m_pos, len);
        return;
    }

    /* 长距离匹配：可以安全使用向量拷贝 */
    if (offset >= 16) {
        /* 循环展开 + ILP 优化：交错 load/store 减少依赖链 */
        while (len >= 64) {
            uchar16 v0 = vload16(0, m_pos);
            uchar16 v1 = vload16(0, m_pos + 16);
            vstore16(v0, 0, op);          // 立即使用 v0，释放寄存器
            uchar16 v2 = vload16(0, m_pos + 32);
            vstore16(v1, 0, op + 16);     // 立即使用 v1
            uchar16 v3 = vload16(0, m_pos + 48);
            vstore16(v2, 0, op + 32);     // 立即使用 v2
            vstore16(v3, 0, op + 48);     // 使用 v3
            op += 64; m_pos += 64; len -= 64;
        }

        /* 16字节向量拷贝 */
        while (len >= 16) {
            uchar16 v = vload16(0, m_pos);
            vstore16(v, 0, op);
            op += 16; m_pos += 16; len -= 16;
        }

        /* 8字节向量拷贝 */
        if (len >= 8) {
            uchar8 v = vload8(0, m_pos);
            vstore8(v, 0, op);
            op += 8; m_pos += 8; len -= 8;
        }

        /* 4字节向量拷贝 */
        if (len >= 4) {
            uchar4 v = vload4(0, m_pos);
            vstore4(v, 0, op);
            op += 4; m_pos += 4; len -= 4;
        }
    }
    /* 中距离匹配：可以使用8字节向量 */
    else if (offset >= 8) {
        /* 8字节向量拷贝 */
        while (len >= 8) {
            uchar8 v = vload8(0, m_pos);
            vstore8(v, 0, op);
            op += 8; m_pos += 8; len -= 8;
        }

        /* 4字节向量拷贝 */
        if (len >= 4) {
            uchar4 v = vload4(0, m_pos);
            vstore4(v, 0, op);
            op += 4; m_pos += 4; len -= 4;
        }
    }
    /* 短距离匹配：不会 overlap（本分支在合并前已判断过 overlap） */
    else if (offset > 0) {
        /* RLE模式检测：offset=1时是字节重复 */
        if (offset == 1) {
            uchar c = *m_pos;
            /* 向量化填充 */
            uchar16 fill16 = (uchar16)(c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c);
            uchar8 fill8 = (uchar8)(c,c,c,c,c,c,c,c);

            while (len >= 16) {
                vstore16(fill16, 0, op);
                op += 16; len -= 16;
            }
            while (len >= 8) {
                vstore8(fill8, 0, op);
                op += 8; len -= 8;
            }
            while (len--) *op++ = c;
            return;
        }

        /* offset=2: 重复2字节模式 */
        if (offset == 2 && len >= 8) {
            uchar p0 = m_pos[0]; uchar p1 = m_pos[1];
            uchar8 pat8 = (uchar8)(p0, p1, p0, p1, p0, p1, p0, p1);
            while (len >= 8) {
                vstore8(pat8, 0, op);
                op += 8; m_pos += 8; len -= 8;
            }
            /* 尾部处理 */
            while (len >= 2) { *op++ = p0; *op++ = p1; len -= 2; }
            if (len >= 1) *op++ = p0;
            return;
        }
        /* offset=3: 重复3字节模式 */
        else if (offset == 3 && len >= 6) {
            uchar c0 = m_pos[0], c1 = m_pos[1], c2 = m_pos[2];
            while (len >= 3) {
                op[0] = c0; op[1] = c1; op[2] = c2;
                op += 3; len -= 3;
            }
            /* 尾部处理 */
            if (len >= 1) *op++ = c0;
            if (len >= 2) *op++ = c1;
            return;
        }
        /* offset=4: 重复4字节模式 */
        else if (offset == 4 && len >= 8) {
            uchar p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2], p3 = m_pos[3];
            uchar16 pat16 = (uchar16)(p0,p1,p2,p3,p0,p1,p2,p3,p0,p1,p2,p3,p0,p1,p2,p3);
            uchar8 pat8 = (uchar8)(p0, p1, p2, p3, p0, p1, p2, p3);
            while (len >= 16) { vstore16(pat16, 0, op); op += 16; len -= 16; }
            while (len >= 8) {
                vstore8(pat8, 0, op);
                op += 8; len -= 8;
            }
            while (len >= 4) { *op++ = p0; *op++ = p1; *op++ = p2; *op++ = p3; len -= 4; }
            /* 尾部处理：剩余 0-3 字节 */
            if (len >= 1) *op++ = p0;
            if (len >= 2) *op++ = p1;
            if (len >= 3) *op++ = p2;
            return;
        }
        /* offset=6: repeat 6-byte pattern - use 8-byte vector loads/stores
         * (reading 8 bytes from m_pos for repeated pattern is safe and faster)
         */
        else if (offset == 6 && len >= 8) {
            uchar p0 = m_pos[0], p1 = m_pos[1], p2 = m_pos[2], p3 = m_pos[3], p4 = m_pos[4], p5 = m_pos[5];
            uchar8 pat8 = (uchar8)(p0, p1, p2, p3, p4, p5, p0, p1);
            while (len >= 8) {
                vstore8(pat8, 0, op);
                op += 8; m_pos += 8; len -= 8;
            }
            /* 尾部处理 */
            while (len >= 6) {
                op[0] = p0; op[1] = p1; op[2] = p2; op[3] = p3; op[4] = p4; op[5] = p5;
                op += 6; len -= 6;
            }
            if (len >= 1) *op++ = p0;
            if (len >= 2) *op++ = p1;
            if (len >= 3) *op++ = p2;
            if (len >= 4) *op++ = p3;
            if (len >= 5) *op++ = p4;
            return;
        }
        /* offset=5,7: 无法安全使用向量拷贝，使用标量拷贝 */
        else {
            while (len--) *op++ = *m_pos++;
            return;
        }
    }

    /* 尾部逐字节拷贝 */
    while (len--) *op++ = *m_pos++;
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

/* common helper macros used by decompressor */
#define pd(a,b)             ((lzo_uint) ((a)-(b)))

/* markers and offsets used by decompressor (same across levels) */
#define M2_MAX_OFFSET   0x0800
#define M4_MARKER       16

/* Original LZO decompressor (same as used in per-level files) */
static lzo_uint
lzo1x_decompress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint  in_len,
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
        /* copy 3 initial bytes + t more bytes */
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
                m_pos = op - 1;
                m_pos -= (t >> 2) & 7;
                m_pos -= *ip++ << 3;
                t = (t >> 5) - 1;
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
            /* 使用优化的向量化拷贝 */
            COPY_MATCH(op, m_pos, t + 2);
            op += t + 2;

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

/* Device kernel for decompression */
__kernel void lzo1x_block_decompress(
    __global const uchar* in_buf,
    __global const uint* off_arr,
    __global       uchar* out_buf,
    __global       uint* out_lens,
    uint blk_sz,
    uint orig_size,
    uint nblk)
{
    uint gid = get_global_id(0);
    /* guard against extra work-items when global size was rounded up */
    if (gid >= nblk) return;

    uint in_off = off_arr[gid];
    uint in_len = off_arr[gid + 1] - in_off;

    uint out_off = gid * blk_sz;
    uint out_len = (out_off + blk_sz <= orig_size) ?
        blk_sz : (orig_size - out_off);

    __global const uchar* src = in_buf + in_off;
    __global       uchar* dst = out_buf + out_off;

    lzo1x_decompress(src, in_len, dst, &out_len, NULL);

    out_lens[gid] = out_len;
}
