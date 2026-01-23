#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "lzo_gpu.h"

#define UA_GET_LE32(pp) as_uint(vload4(0, (__generic const uchar*)(pp)))

static inline void UA_COPYN(__generic uchar *d, const __generic uchar *s, uint nn)
{
    if (nn >= 32) {
        while (nn >= 32) {
            vstore16(vload16(0, s), 0, d);
            vstore16(vload16(1, s), 0, d + 16);
            d += 32; s += 32; nn -= 32;
        }
    }
    if (nn >= 16) {
        vstore16(vload16(0, s), 0, d);
        d += 16;
        s += 16;
        nn -= 16;
    }
    if (nn >= 8) {
        vstore8(vload8(0, s), 0, d);
        d += 8;
        s += 8;
        nn -= 8;
    }
    if (nn >= 4) {
        vstore4(vload4(0, s), 0, d);
        d += 4;
        s += 4;
        nn -= 4;
    }
    if (nn > 0) {
        *d++ = *s++;
        if (nn > 1) {
            *d++ = *s++;
            if (nn > 2)
                *d++ = *s++;
        }
    }
}

#define pd(a,b) ((lzo_uint) ((a)-(b)))
#define M2_MAX_OFFSET 0x0400

static inline void COPY_MATCH(__generic uchar *op, __generic const uchar *m_pos, uint len)
{
    uint offset = op - m_pos;
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

/* Debug instrumentation removed in production build. */

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

            COPY_MATCH(op, m_pos, t + 2);
            op += t + 2;
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
    __global       uchar* out_buf, __global uint* out_lens,
    uint blk_sz, uint orig_size, uint nblk)
{
    uint gid = get_global_id(0);
    if (gid >= nblk) return;
    uint in_off = off_arr[gid];
    uint in_len = off_arr[gid + 1] - in_off;
    uint out_off = gid * blk_sz;
    uint out_len = (out_off + blk_sz <= orig_size) ? blk_sz : (orig_size - out_off);
    lzo1y_decompress(in_buf + in_off, in_len, out_buf + out_off, &out_len, NULL);
    out_lens[gid] = out_len;
}


