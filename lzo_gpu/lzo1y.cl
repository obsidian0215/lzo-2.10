#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "lzo_gpu.h"

/* LZO1Y Unified Kernel
 * Supports variable dictionary sizes via D_BITS macro.
 * Default D_BITS = 14 if not defined.
 */

#ifndef D_BITS
#define D_BITS 14
#endif

/* Debug instrumentation: enable via -D LZO_GPU_DEBUG.
 * When enabled, debug emits extended fields (IN, OUT, FLAG, LOOKUPS, HITS, MATCH_BYTES, UPDATES):
 * 7 fields per block by default. To override, define DBG_FIELDS externally.
 */
#ifdef LZO_GPU_DEBUG
#ifndef DBG_FIELDS
#define DBG_FIELDS 7
#endif
#endif

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

static inline void LZO_MEMOPS_COPYN(__generic void *dd, const __generic void *ss, uint nn)
{
    __generic uchar *d = (__generic uchar*)dd;
    __generic const uchar *s = (__generic const uchar*)ss;

    while (nn >= 8 && lzo_ptr_aligned(d,8) && lzo_ptr_aligned(s,8))
    {   LZO_MEMOPS_COPY8(d,s); d+=8; s+=8; nn-=8; }

    while (nn >= 4 && lzo_ptr_aligned(d,4) && lzo_ptr_aligned(s,4))
    {   LZO_MEMOPS_COPY4(d,s); d+=4; s+=4; nn-=4; }

    for (; nn; --nn) *d++ = *s++;
}

static inline uint lzo_memops_get_le32(const __global void *pp)
{
    const __global uchar *p = (const __global uchar*)pp;
    if (lzo_ptr_aligned(p,4))
        return as_uint(*((const __global uint*)p));
    return  (uint)p[0] | ((uint)p[1] <<  8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

static inline ulong lzo_memops_get_le64(const __global void *pp)
{
    const __global uchar *p = (const __global uchar*)pp;
    if (lzo_ptr_aligned(p,8))
        return as_ulong(*((const __global ulong*)p));
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
#define D_HIGH          ((D_MASK >> 1) + 1)

#define DX2(p,s1,s2) \
        (((((lzo_xint)((p)[2]) << (s2)) ^ (p)[1]) << (s1)) ^ (p)[0])
#define DX3(p,s1,s2,s3) ((DX2((p)+1,s2,s3) << (s1)) ^ (p)[0])
#define DMS(v,s)        ((lzo_uint) (((v) & (D_MASK >> (s))) << (s)))
#define DM(v)           DMS(v,0)

/* LZO1Y Hash Function */
#define DINDEX(dv,p)        DM(((DMUL(0x1824429d,dv)) >> (32-D_BITS)))

#define DENTRY(p,in)                          ((lzo_dict_t) pd(p, in))
#define GINDEX(m_off,m_pos,dict,dindex,in)    m_off = dict[dindex]; m_pos = (in) + (m_off)

#define UPDATE_I(dict,drun,index,p,in)    dict[index] = DENTRY(p,in)

#ifdef LZO_GPU_DEBUG
static lzo_uint
lzo1y_compress_core(LZO_ADDR_GLOBAL const lzo_bytep in , lzo_uint  in_len,
                   LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
                    lzo_uint ti, lzo_voidp wrkmem, __global uint *dbg_out)
#else
static lzo_uint
lzo1y_compress_core(LZO_ADDR_GLOBAL const lzo_bytep in , lzo_uint  in_len,
                   LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
                    lzo_uint ti, lzo_voidp wrkmem)
#endif
{
    LZO_ADDR_GLOBAL const lzo_bytep ip;
    LZO_ADDR_GLOBAL lzo_bytep op;
    const LZO_ADDR_GLOBAL lzo_bytep in_end = in + in_len;
    const LZO_ADDR_GLOBAL lzo_bytep ip_end = in + in_len - 20;
    LZO_ADDR_GLOBAL const lzo_bytep ii;
    lzo_dict_p const dict = (lzo_dict_p) wrkmem;

    op = out;
    ip = in;
    ii = ip;

#ifdef LZO_GPU_DEBUG
    uint dbg_lookups = 0, dbg_hits = 0, dbg_updates = 0, dbg_matched_bytes = 0, dbg_matches = 0;
#endif

    ip += ti < 4 ? 4 - ti : 0;
    for (;;)
    {
        LZO_ADDR_GLOBAL const lzo_bytep m_pos;
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
#ifdef LZO_GPU_DEBUG
            dbg_lookups++;
#endif
            GINDEX(m_off,m_pos,dict,dindex,in);

            /* Prefetch next */
            if (ip + 4 < ip_end) {
                uint next_dv = UA_GET_LE32(ip + 1);
                uint next_idx = DINDEX(next_dv, ip + 1);
                prefetch(in + dict[next_idx], 4);
            }

            UPDATE_I(dict,0,dindex,ip,in);
#ifdef LZO_GPU_DEBUG
            dbg_updates++;
#endif
            if (dv != UA_GET_LE32(m_pos))
                goto literal;
#ifdef LZO_GPU_DEBUG
            dbg_hits++;
#endif
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
                    *op++ = LZO_BYTE(tt);
                }
                { do *op++ = *ii++; while (--t > 0); }
            }
        }

        m_len = 4;
        /* Vectorized match check (unrolled if requested) */
#ifdef LZO_USE_UNROLL4
        while (ip + m_len + 32 <= ip_end) {
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
            ip_val = UA_GET_LE64(ip + m_len + 16);
            mp_val = UA_GET_LE64(m_pos + m_len + 16);
            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += 16 + (ctz(diff) >> 3);
                goto m_len_done;
            }
            ip_val = UA_GET_LE64(ip + m_len + 24);
            mp_val = UA_GET_LE64(m_pos + m_len + 24);
            if (ip_val != mp_val) {
                ulong diff = ip_val ^ mp_val;
                m_len += 24 + (ctz(diff) >> 3);
                goto m_len_done;
            }
            m_len += 32;
        }
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
#elif defined(LZO_USE_UNROLL2)
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

        if (ip[m_len] == m_pos[m_len]) {
            do {
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip[m_len] != m_pos[m_len]) break;
                m_len += 1;
                if (ip + m_len >= ip_end) goto m_len_done;
            } while (ip[m_len] == m_pos[m_len]);
        }
m_len_done:
#ifdef LZO_GPU_DEBUG
        dbg_matched_bytes += m_len;
        dbg_matches++;
#endif
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
#ifdef LZO_GPU_DEBUG
    if (dbg_out) {
        dbg_out[3] = dbg_lookups;
        dbg_out[4] = dbg_hits;
        dbg_out[5] = dbg_matched_bytes;
        dbg_out[6] = dbg_updates;
    }
#endif
    return pd(in_end,ii-ti);
}

static inline void dict_clear(lzo_dict_t* d) {
#pragma unroll
    for (uint i = 0; i < D_SIZE; ++i) d[i] = 0;
}

#ifdef LZO_GPU_DEBUG
static inline int do_compress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint  in_len,
    LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
    lzo_uint  ti, lzo_voidp wrkmem, __global uint* dbg_out)
#else
static inline int do_compress(LZO_ADDR_GLOBAL const lzo_bytep in, lzo_uint  in_len,
    LZO_ADDR_GLOBAL lzo_bytep out, lzo_uintp out_len,
    lzo_uint  ti, lzo_voidp wrkmem)
#endif
{
    __global const uchar* ip = in;
    __global uchar* op = out;
    lzo_uint l = in_len;
    lzo_uint t = 0;

    while (l > 20)
    {
        lzo_uint ll = LZO_MIN(l, 49152);
        lzo_uintptr_t ll_end = (lzo_uintptr_t)ip + ll;
        if ((ll_end + ((t + ll) >> 5)) <= ll_end ||
            (__global uchar*)(ll_end + ((t + ll) >> 5)) <= ip + ll)
            break;

        dict_clear(wrkmem);
#ifdef LZO_GPU_DEBUG
        t = lzo1y_compress_core(ip, ll, op, out_len, t, wrkmem, dbg_out);
#else
        t = lzo1y_compress_core(ip, ll, op, out_len, t, wrkmem);
#endif
        ip += ll;
        op += *out_len;
        l -= ll;
    }
    t += l;

    if (t > 0)
    {
        __global const uchar *ii = in + in_len - t;
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
            while (tt > 255) { tt -= 255; UA_SET1(op, 0); op++; }
            *op++ = LZO_BYTE(tt);
        }
        UA_COPYN(op, ii, t);
        op += t;
    }

    *op++ = M4_MARKER | 1;
    *op++ = 0; *op++ = 0;

    *out_len = pd(op, out);
    return 0;
}

#ifndef LZO_GPU_DEBUG
__kernel void lzo1y_block_compress(__global const uchar *in ,
                                   __global       uchar *out,
                                   __global       uint  *out_len,
                                   const uint  in_sz,
                                   const uint  blk_size,
                                   const uint  worst_blk)
{
    uint in_len, in_off;
    const uint gid = get_global_id(0);
    in_off = gid * blk_size;
    __global const uchar* ip = in + in_off;
    __global uchar* op = out + gid * worst_blk;

    __local lzo_dict_t dict[1<<D_BITS];

    in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);
    if (in_len == 0) {
        out_len[gid] = 0;
        return;
    }

    lzo_uint olen;
    do_compress(ip, in_len, op, &olen, 0, dict);
    out_len[gid] = olen;
}
#else
__kernel void lzo1y_block_compress_debug(__global const uchar *in ,
                                        __global       uchar *out,
                                        __global       uint  *out_len,
                                        const uint  in_sz,
                                        const uint  blk_size,
                                        const uint  worst_blk,
                                        __global uint *dbg)
{
    const uint gid = get_global_id(0);
    const uint in_off = gid * blk_size;
    __global const uchar* ip = in + in_off;
    __global uchar* op = out + gid * worst_blk;

    __local lzo_dict_t dict[1<<D_BITS];

    uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);
    if (in_len == 0) {
        out_len[gid] = 0;
        for (int f = 0; f < DBG_FIELDS; ++f) dbg[gid*DBG_FIELDS + f] = 0;
        dbg[gid*DBG_FIELDS + 2] = 1;
        return;
    }

    lzo_uint olen = 0;
    do_compress(ip, in_len, op, &olen, 0, dict, dbg + gid * DBG_FIELDS);
    out_len[gid] = olen;

    dbg[gid*DBG_FIELDS + 0] = in_len;
    dbg[gid*DBG_FIELDS + 1] = olen;
    dbg[gid*DBG_FIELDS + 2] = (olen == 0) ? 1 : 0;
}
#endif
