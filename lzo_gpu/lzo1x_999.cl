/* lzo1x_999.cl -- GPU port of the LZO1X-999 SWD compression algorithm
 *
 * True port of upstream lzo1x_9x.c Sliding Window Dictionary (SWD)
 * hash-chain algorithm to OpenCL. Produces byte-identical LZO1X output
 * that can be decompressed by the standard lzo1x_decompress kernel.
 *
 * Design:
 *   - 1 work-item per block (sequential SWD processing)
 *   - All SWD arrays in global memory (~458 KB per work-item)
 *   - No dict/epoch mechanism — SWD is initialized fresh per block
 *   - Lazy matching with configurable try_lazy / max_chain parameters
 */

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#ifndef __generic
#define __generic
#endif

#include "lzo_gpu.h"

/* Standard macros */
#define LZO_BYTE(x)       ((unsigned char) (x))
#define LZO_MAX(a,b)      ((a) >= (b) ? (a) : (b))
#define LZO_MIN(a,b)      ((a) <= (b) ? (a) : (b))
#define LZO_MAX3(a,b,c)   LZO_MAX(LZO_MAX(a,b),c)

#define DMUL(a,b)          ((lzo_xint) ((a) * (b)))

#define pd(a,b)            ((lzo_uint) ((a)-(b)))

/* ---- LZO1X encoding constants ---- */
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

/* ---- SWD constants ---- */
#define SWD_N           M4_MAX_OFFSET   /* ring buffer size = 0xbfff = 49151 */
#define SWD_F           2048            /* max match length */
#define SWD_THRESHOLD   1               /* min match length for hash insertion */
#define SWD_HSIZE       16384           /* 3-byte hash table size */
#define SWD_MAX_CHAIN   2048            /* default max chain depth */

#define SWD_BEST_OFF    (LZO_MAX3(M2_MAX_LEN, M3_MAX_LEN, M4_MAX_LEN) + 1)  /* 34 */

/* SWD array types — use 16-bit indices since SWD_N + SWD_F < 65535 */
typedef ushort swd_uint;
#define SWD_UINT_MAX    0xffffu
#define SWD_UINT(x)     ((swd_uint)(x))

/* ---- SWD memory layout in global pool ----
 * All arrays packed sequentially per work-item:
 *
 *   b[SWD_N + 2*SWD_F]    = 53247 bytes
 *   head3[SWD_HSIZE]       = 32768 bytes (16384 * 2)
 *   succ3[SWD_N + SWD_F]   = 102398 bytes (51199 * 2)
 *   best3[SWD_N + SWD_F]   = 102398 bytes (51199 * 2)
 *   llen3[SWD_HSIZE]       = 32768 bytes (16384 * 2)
 *   head2[65536]            = 131072 bytes (65536 * 2)
 *   Total = ~454652 bytes (aligned), packed into SWD_POOL_STRIDE
 */
#define SWD_B_SIZE      (SWD_N + SWD_F + SWD_F)   /* 53247 */
#define SWD_SF_SIZE     (SWD_N + SWD_F)            /* 51199 */
#define SWD_HEAD2_SIZE  65536

/* Align offset to 2-byte boundary for swd_uint (ushort) arrays */
#define SWD_ALIGN2(x)   (((x) + 1u) & ~1u)

/* Byte offsets within per-WI pool — all swd_uint arrays must be 2-byte aligned */
#define SWD_OFF_B       0
#define SWD_OFF_HEAD3   SWD_ALIGN2(SWD_OFF_B + SWD_B_SIZE)
#define SWD_OFF_SUCC3   (SWD_OFF_HEAD3 + SWD_HSIZE * sizeof(swd_uint))
#define SWD_OFF_BEST3   (SWD_OFF_SUCC3 + SWD_SF_SIZE * sizeof(swd_uint))
#define SWD_OFF_LLEN3   (SWD_OFF_BEST3 + SWD_SF_SIZE * sizeof(swd_uint))
#define SWD_OFF_HEAD2   (SWD_OFF_LLEN3 + SWD_HSIZE * sizeof(swd_uint))
#define SWD_POOL_STRIDE SWD_ALIGN2(SWD_OFF_HEAD2 + SWD_HEAD2_SIZE * sizeof(swd_uint))
/* SWD_POOL_STRIDE ≈ 454,652 bytes (aligned) */

/* ---- SWD accessor macros for global memory ---- */
#define s_b(base)       ((__global uchar*)(base + SWD_OFF_B))
#define s_head3(base)   ((__global swd_uint*)(base + SWD_OFF_HEAD3))
#define s_succ3(base)   ((__global swd_uint*)(base + SWD_OFF_SUCC3))
#define s_best3(base)   ((__global swd_uint*)(base + SWD_OFF_BEST3))
#define s_llen3(base)   ((__global swd_uint*)(base + SWD_OFF_LLEN3))
#define s_head2(base)   ((__global swd_uint*)(base + SWD_OFF_HEAD2))

/* Safe head3 access — returns SWD_UINT_MAX if chain is empty */
#define s_get_head3(base, key) \
    ((swd_uint)((s_llen3(base)[key] == 0) ? SWD_UINT_MAX : s_head3(base)[key]))

/* ---- SWD state (kept in private registers where possible) ---- */
typedef struct {
    uint ip;        /* input pointer (lookahead) into ring buffer */
    uint bp;        /* buffer pointer (current position) */
    uint rp;        /* remove pointer */
    uint b_size;    /* SWD_N + SWD_F */
    uint look;      /* bytes in lookahead */
    uint m_len;     /* best match length */
    uint m_off;     /* best match offset */
    uint m_pos;     /* best match position in ring buffer */
    int  b_char;    /* current byte at bp */
    uint node_count;
    uint first_rp;
    uint max_chain;
    uint nice_length;
    uint use_best_off;
    uint best_off[SWD_BEST_OFF];
    uint best_pos[SWD_BEST_OFF];
} swd_state_t;

/* Compress state (LZO_COMPRESS_T equivalent) */
typedef struct {
    uint r1_lit;
    uint r1_m_len;
    uint last_m_len;
    uint last_m_off;
    /* Counters for encoding decisions — not needed for output but for correct
     * lazy match cost model. We skip m1a_m etc. stats, only track what's needed. */
} compress_state_t;

/* ---- Hash functions ---- */

/* 3-byte hash (matches upstream HEAD3) */
static inline uint swd_head3(__global const uchar* b, uint p)
{
    return ((DMUL(0x9f5f, (((((lzo_xint)b[p] << 5) ^ b[p+1]) << 5) ^ b[p+2])) >> 5) & (SWD_HSIZE - 1));
}

/* 2-byte hash (matches upstream HEAD2) */
static inline uint swd_head2(__global const uchar* b, uint p)
{
    return (uint)b[p] ^ ((uint)b[p+1] << 8);
}

/* ---- pos2off: convert ring buffer position to offset ---- */
static inline uint swd_pos2off(uint bp, uint pos, uint b_size)
{
    return (bp > pos) ? (bp - pos) : (b_size - (pos - bp));
}

/* ---- SWD core functions ---- */

static void swd_getbyte(swd_state_t* s, __global uchar* base,
                         __global const uchar* src, uint src_len, uint* src_pos)
{
    __global uchar* b = s_b(base);

    if (*src_pos < src_len) {
        uchar c = src[*src_pos];
        (*src_pos)++;
        b[s->ip] = c;
        if (s->ip < SWD_F)
            b[s->b_size + s->ip] = c;  /* b_wrap mirror */
    } else {
        if (s->look > 0)
            s->look--;
        b[s->ip] = 0;
        if (s->ip < SWD_F)
            b[s->b_size + s->ip] = 0;
    }

    s->ip++;
    if (s->ip == s->b_size) s->ip = 0;
    s->bp++;
    if (s->bp == s->b_size) s->bp = 0;
    s->rp++;
    if (s->rp == s->b_size) s->rp = 0;
}

static void swd_remove_node(swd_state_t* s, __global uchar* base, uint node)
{
    if (s->node_count == 0) {
        uint key = swd_head3(s_b(base), node);
        s_llen3(base)[key]--;

        /* Remove from head2 if it points to this node */
        uint key2 = swd_head2(s_b(base), node);
        if ((uint)s_head2(base)[key2] == node)
            s_head2(base)[key2] = SWD_UINT_MAX;  /* NIL2 */
    } else {
        s->node_count--;
    }
}

static void swd_accept(swd_state_t* s, __global uchar* base,
                        __global const uchar* src, uint src_len, uint* src_pos,
                        uint n)
{
    while (n > 0) {
        uint key;
        swd_remove_node(s, base, s->rp);

        /* Add bp into HEAD3 */
        key = swd_head3(s_b(base), s->bp);
        s_succ3(base)[s->bp] = s_get_head3(base, key);
        s_head3(base)[key] = SWD_UINT(s->bp);
        s_best3(base)[s->bp] = SWD_UINT(SWD_F + 1);
        s_llen3(base)[key]++;

        /* Add bp into HEAD2 */
        {
            uint key2 = swd_head2(s_b(base), s->bp);
            s_head2(base)[key2] = SWD_UINT(s->bp);
        }

        swd_getbyte(s, base, src, src_len, src_pos);
        n--;
    }
}

static void swd_search(swd_state_t* s, __global uchar* base, uint node, uint cnt)
{
    __global const uchar* b = s_b(base);
    __global const uchar* bp_ptr = b + s->bp;
    uint bx_lim = s->bp + s->look;
    uint m_len = s->m_len;
    uchar scan_end1 = bp_ptr[m_len - 1];

    for (; cnt > 0; cnt--, node = s_succ3(base)[node]) {
        __global const uchar* p2 = b + node;

        if (p2[m_len - 1] != scan_end1) continue;
        if (p2[m_len] != bp_ptr[m_len]) continue;
        if (p2[0] != bp_ptr[0]) continue;
        if (p2[1] != bp_ptr[1]) continue;

        /* Full comparison */
        uint i = 2;
        while (i + 1 < s->look && bp_ptr[i + 1] == p2[i + 1]) i++;
        i++;  /* i now = match length */

        if (i < SWD_BEST_OFF) {
            if (s->best_pos[i] == 0)
                s->best_pos[i] = node + 1;
        }

        if (i > m_len) {
            s->m_len = m_len = i;
            s->m_pos = node;
            if (m_len == s->look) return;
            if (m_len >= s->nice_length) return;
            if (m_len > (uint)s_best3(base)[node]) return;
            scan_end1 = bp_ptr[m_len - 1];
        }
    }
}

static int swd_search2(swd_state_t* s, __global uchar* base)
{
    uint key = s_head2(base)[swd_head2(s_b(base), s->bp)];
    if (key == SWD_UINT_MAX) return 0;  /* NIL2 */

    if (s->best_pos[2] == 0)
        s->best_pos[2] = key + 1;

    if (s->m_len < 2) {
        s->m_len = 2;
        s->m_pos = key;
    }
    return 1;
}

static void swd_findbest(swd_state_t* s, __global uchar* base)
{
    uint key = swd_head3(s_b(base), s->bp);
    uint node = s_succ3(base)[s->bp] = s_get_head3(base, key);
    uint cnt = s_llen3(base)[key];
    s_llen3(base)[key]++;
    if (cnt > s->max_chain && s->max_chain > 0)
        cnt = s->max_chain;
    s_head3(base)[key] = SWD_UINT(s->bp);

    s->b_char = s_b(base)[s->bp];
    uint len = s->m_len;

    if (s->m_len >= s->look) {
        if (s->look == 0) s->b_char = -1;
        s->m_off = 0;
        s_best3(base)[s->bp] = SWD_UINT(SWD_F + 1);
    } else {
        /* HEAD2 is always defined — only search chain if swd_search2 found a 2-byte match */
        if (swd_search2(s, base) && s->look >= 3)
            swd_search(s, base, node, cnt);

        if (s->m_len > len)
            s->m_off = swd_pos2off(s->bp, s->m_pos, s->b_size);
        s_best3(base)[s->bp] = SWD_UINT(s->m_len);

        if (s->use_best_off) {
            for (uint i = 2; i < SWD_BEST_OFF; i++) {
                if (s->best_pos[i] > 0)
                    s->best_off[i] = swd_pos2off(s->bp, s->best_pos[i] - 1, s->b_size);
                else
                    s->best_off[i] = 0;
            }
        }
    }

    swd_remove_node(s, base, s->rp);

    /* Add bp into HEAD2 */
    {
        uint key2 = swd_head2(s_b(base), s->bp);
        s_head2(base)[key2] = SWD_UINT(s->bp);
    }
}

/* ---- SWD init ---- */
static void swd_init(swd_state_t* s, __global uchar* base,
                      __global const uchar* src, uint src_len, uint* src_pos)
{
    __global uchar* b = s_b(base);

    s->m_len = 0;
    s->m_off = 0;
    for (uint i = 0; i < SWD_BEST_OFF; i++) {
        s->best_off[i] = 0;
        s->best_pos[i] = 0;
    }

    s->b_size = SWD_N + SWD_F;
    s->max_chain = SWD_MAX_CHAIN;
    s->nice_length = SWD_F;
    s->use_best_off = 0;
    s->node_count = SWD_N;
    s->ip = 0;
    s->bp = 0;
    s->first_rp = 0;

    /* Zero llen3 */
    for (uint i = 0; i < SWD_HSIZE; i++)
        s_llen3(base)[i] = 0;

    /* Set head2 to NIL2 */
    for (uint i = 0; i < SWD_HEAD2_SIZE; i++)
        s_head2(base)[i] = SWD_UINT_MAX;

    /* Fill initial lookahead */
    s->look = 0;
    while (s->look < SWD_F && *src_pos < src_len) {
        b[s->ip] = src[*src_pos];
        (*src_pos)++;
        s->ip++;
        s->look++;
    }
    if (s->ip == s->b_size)
        s->ip = 0;

    s->rp = s->first_rp;
    if (s->rp >= s->node_count)
        s->rp -= s->node_count;
    else
        s->rp += s->b_size - s->node_count;

    /* Initialize lookahead padding */
    if (s->look < 3) {
        b[s->bp + s->look] = 0;
        b[s->bp + s->look + 1] = 0;
        b[s->bp + s->look + 2] = 0;
    }
}

/* ---- find_match (wraps findbest + getbyte + state update) ---- */
typedef struct {
    uint look;   /* updated lookahead */
    uint m_len;  /* best match length */
    uint m_off;  /* best match offset */
    int  b_char; /* byte at current position */
} find_match_result_t;

static void find_match(swd_state_t* s, __global uchar* base,
                        __global const uchar* src, uint src_len, uint* src_pos,
                        uint this_len, uint skip,
                        find_match_result_t* out)
{
    if (skip > 0) {
        swd_accept(s, base, src, src_len, src_pos, this_len - skip);
    }

    s->m_len = SWD_THRESHOLD;
    s->m_off = 0;
    if (s->use_best_off) {
        for (uint i = 0; i < SWD_BEST_OFF; i++)
            s->best_pos[i] = 0;
    }

    swd_findbest(s, base);
    out->m_len = s->m_len;
    out->m_off = s->m_off;

    swd_getbyte(s, base, src, src_len, src_pos);

    if (s->b_char < 0) {
        out->look = 0;
        out->m_len = 0;
    } else {
        out->look = s->look + 1;
    }
    out->b_char = s->b_char;
}

/* ---- Encoding functions ---- */

static __global uchar*
code_match_1x(__global uchar* op, __global uchar* out_start,
              compress_state_t* c, uint m_len, uint m_off)
{
    c->last_m_len = m_len;
    c->last_m_off = m_off;

    if (m_len == 2) {
        /* M1 match */
        m_off -= 1;
        *op++ = LZO_BYTE(M1_MARKER | ((m_off & 3) << 2));
        *op++ = LZO_BYTE(m_off >> 2);
    }
    else if (m_len <= M2_MAX_LEN && m_off <= M2_MAX_OFFSET) {
        /* M2 match */
        m_off -= 1;
        *op++ = LZO_BYTE(((m_len - 1) << 5) | ((m_off & 7) << 2));
        *op++ = LZO_BYTE(m_off >> 3);
    }
    else if (m_len == M2_MIN_LEN && m_off <= MX_MAX_OFFSET && c->r1_lit >= 4) {
        /* M1b match */
        m_off -= 1 + M2_MAX_OFFSET;
        *op++ = LZO_BYTE(M1_MARKER | ((m_off & 3) << 2));
        *op++ = LZO_BYTE(m_off >> 2);
    }
    else if (m_off <= M3_MAX_OFFSET) {
        /* M3 match */
        m_off -= 1;
        if (m_len <= M3_MAX_LEN)
            *op++ = LZO_BYTE(M3_MARKER | (m_len - 2));
        else {
            m_len -= M3_MAX_LEN;
            *op++ = M3_MARKER | 0;
            while (m_len > 255) {
                m_len -= 255;
                *op++ = 0;
            }
            *op++ = LZO_BYTE(m_len);
        }
        *op++ = LZO_BYTE(m_off << 2);
        *op++ = LZO_BYTE(m_off >> 6);
    }
    else {
        /* M4 match */
        uint k;
        m_off -= 0x4000;
        k = (m_off & 0x4000) >> 11;
        if (m_len <= M4_MAX_LEN)
            *op++ = LZO_BYTE(M4_MARKER | k | (m_len - 2));
        else {
            m_len -= M4_MAX_LEN;
            *op++ = LZO_BYTE(M4_MARKER | k | 0);
            while (m_len > 255) {
                m_len -= 255;
                *op++ = 0;
            }
            *op++ = LZO_BYTE(m_len);
        }
        *op++ = LZO_BYTE(m_off << 2);
        *op++ = LZO_BYTE(m_off >> 6);
    }

    return op;
}

static __global uchar*
store_run_1x(__global uchar* op, __global uchar* out_start,
             __global const uchar* ii, uint t)
{
    if (op == out_start && t <= 238) {
        *op++ = LZO_BYTE(17 + t);
    }
    else if (t <= 3) {
        op[-2] = LZO_BYTE(op[-2] | t);
    }
    else if (t <= 18) {
        *op++ = LZO_BYTE(t - 3);
    }
    else {
        uint tt = t - 18;
        *op++ = 0;
        while (tt > 255) {
            tt -= 255;
            *op++ = 0;
        }
        *op++ = LZO_BYTE(tt);
    }
    /* Copy literal bytes */
    for (uint i = 0; i < t; i++)
        *op++ = ii[i];

    return op;
}

static __global uchar*
code_run_1x(__global uchar* op, __global uchar* out_start,
            compress_state_t* c,
            __global const uchar* ii, uint lit, uint m_len)
{
    if (lit > 0) {
        op = store_run_1x(op, out_start, ii, lit);
        c->r1_m_len = m_len;
        c->r1_lit = lit;
    } else {
        c->r1_m_len = 0;
        c->r1_lit = 0;
    }
    return op;
}

/* ---- Cost model functions ---- */

static uint len_of_coded_match_1x(uint m_len, uint m_off, uint lit)
{
    uint n = 4;

    if (m_len < 2) return 0;
    if (m_len == 2)
        return (m_off <= M1_MAX_OFFSET && lit > 0 && lit < 4) ? 2 : 0;
    if (m_len <= M2_MAX_LEN && m_off <= M2_MAX_OFFSET)
        return 2;
    if (m_len == M2_MIN_LEN && m_off <= MX_MAX_OFFSET && lit >= 4)
        return 2;
    if (m_off <= M3_MAX_OFFSET) {
        if (m_len <= M3_MAX_LEN) return 3;
        m_len -= M3_MAX_LEN;
        while (m_len > 255) { m_len -= 255; n++; }
        return n;
    }
    if (m_off <= M4_MAX_OFFSET) {
        if (m_len <= M4_MAX_LEN) return 3;
        m_len -= M4_MAX_LEN;
        while (m_len > 255) { m_len -= 255; n++; }
        return n;
    }
    return 0;
}

static uint min_gain_1x(uint ahead, uint lit1, uint lit2,
                         uint l1, uint l2, uint l3)
{
    uint lazy_match_min_gain = ahead;

    if (lit1 <= 3)
        lazy_match_min_gain += (lit2 <= 3) ? 0 : 2;
    else if (lit1 <= 18)
        lazy_match_min_gain += (lit2 <= 18) ? 0 : 1;

    lazy_match_min_gain += (l2 - l1) * 2;
    if (l3) {
        if (ahead > l3)
            lazy_match_min_gain -= (ahead - l3) * 2;
        else
            lazy_match_min_gain -= 0;  /* avoid underflow */
    }

    /* Clamp to 0 (equivalent to checking (lzo_int) < 0 in original) */
    if (lazy_match_min_gain > 0x7FFFFFFFu)
        lazy_match_min_gain = 0;

    return lazy_match_min_gain;
}

/* ---- better_match: prefer shorter encoding when possible ---- */
static void better_match_1x(swd_state_t* s, uint* p_m_len, uint* p_m_off)
{
    uint m_len = *p_m_len;
    uint m_off = *p_m_off;

    if (m_len <= M2_MIN_LEN) return;
    if (m_off <= M2_MAX_OFFSET) return;

    /* M3/M4 -> M2: if we can lose 1 byte and fit in M2 */
    if (m_off > M2_MAX_OFFSET &&
        m_len >= M2_MIN_LEN + 1 && m_len <= M2_MAX_LEN + 1 &&
        s->best_off[m_len - 1] && s->best_off[m_len - 1] <= M2_MAX_OFFSET)
    {
        *p_m_len = m_len - 1;
        *p_m_off = s->best_off[m_len - 1];
        return;
    }

    /* M4 -> M2: lose 2 bytes */
    if (m_off > M3_MAX_OFFSET &&
        m_len >= M4_MAX_LEN + 1 && m_len <= M2_MAX_LEN + 2 &&
        s->best_off[m_len - 2] && s->best_off[m_len - 2] <= M2_MAX_OFFSET)
    {
        *p_m_len = m_len - 2;
        *p_m_off = s->best_off[m_len - 2];
        return;
    }

    /* M4 -> M3: lose 1 byte */
    if (m_off > M3_MAX_OFFSET &&
        m_len >= M4_MAX_LEN + 1 && m_len <= M3_MAX_LEN + 1 &&
        s->best_off[m_len - 1] && s->best_off[m_len - 1] <= M3_MAX_OFFSET)
    {
        *p_m_len = m_len - 1;
        *p_m_off = s->best_off[m_len - 1];
    }
}


/* ================================================================
 * Main compression function — faithful port of lzo1x_999_compress_internal
 * ================================================================ */

static void
lzo1x_999_compress_block(__global const uchar* in, uint in_len,
                          __global uchar* out, __global uint* out_len_ptr,
                          __global uchar* swd_base,
                          uint try_lazy_parm, uint max_chain_parm)
{
    swd_state_t swd;
    compress_state_t cc;
    find_match_result_t fm;

    __global uchar* op = out;
    __global uchar* out_start = out;
    uint lit = 0;
    uint m_len, m_off;
    uint try_lazy;
    uint good_length, max_lazy, max_chain;
    uint src_pos = 0;

    /* Apply level 9 defaults (highest quality) if not specified */
    try_lazy = try_lazy_parm;
    good_length = SWD_F;
    max_lazy = SWD_F;
    max_chain = max_chain_parm;
    if (max_chain == 0) max_chain = 4096;

    cc.r1_lit = 0;
    cc.r1_m_len = 0;
    cc.last_m_len = 0;
    cc.last_m_off = 0;

    /* Initialize SWD */
    swd_init(&swd, swd_base, in, in_len, &src_pos);
    if (max_chain > 0)
        swd.max_chain = max_chain;
    swd.nice_length = SWD_F;
    swd.use_best_off = 1;

    uint ii_abs = 0;

    /* First find_match */
    find_match(&swd, swd_base, in, in_len, &src_pos, 0, 0, &fm);
    if (in_len == 0) {
        *out_len_ptr = 0;
        return;
    }
    uint look = fm.look;
    m_len = fm.m_len;
    m_off = fm.m_off;

    while (look > 0) {
        uint ahead;
        uint max_ahead;
        uint l1, l2, l3;

        if (lit == 0)
            ii_abs = src_pos - look;

        /* Check if match is usable */
        int reject = 0;
        if (m_len < 2) reject = 1;
        else if (m_len == 2) {
            if (m_off > M1_MAX_OFFSET || lit == 0 || lit >= 4) reject = 1;
            if (op == out_start) reject = 1;
        }
        if (op == out_start && lit == 0) reject = 1;

        if (m_len == M2_MIN_LEN && m_off > MX_MAX_OFFSET && lit >= 4)
            reject = 1;

        if (reject) {
            /* Literal */
            m_len = 0;
            lit++;
            swd.max_chain = max_chain;
            find_match(&swd, swd_base, in, in_len, &src_pos, 1, 0, &fm);
            look = fm.look;
            m_len = fm.m_len;
            m_off = fm.m_off;
            continue;
        }

        /* Apply better_match optimization */
        if (swd.use_best_off)
            better_match_1x(&swd, &m_len, &m_off);

        /* Lazy match evaluation */
        ahead = 0;
        if (try_lazy == 0 || m_len >= max_lazy) {
            l1 = 0;
            max_ahead = 0;
        } else {
            l1 = len_of_coded_match_1x(m_len, m_off, lit);
            max_ahead = LZO_MIN(try_lazy, l1 > 0 ? l1 - 1 : 0);
        }

        while (ahead < max_ahead && look > m_len) {
            uint lazy_match_min_gain;

            if (m_len >= good_length)
                swd.max_chain = max_chain >> 2;
            else
                swd.max_chain = max_chain;

            find_match(&swd, swd_base, in, in_len, &src_pos, 1, 0, &fm);
            ahead++;
            look = fm.look;

            if (fm.m_len < m_len) continue;
            if (fm.m_len == m_len && fm.m_off >= m_off) continue;

            /* Apply better_match to the new candidate */
            uint new_m_len = fm.m_len;
            uint new_m_off = fm.m_off;
            if (swd.use_best_off)
                better_match_1x(&swd, &new_m_len, &new_m_off);

            l2 = len_of_coded_match_1x(new_m_len, new_m_off, lit + ahead);
            if (l2 == 0) continue;

            l3 = (op == out_start) ? 0 : len_of_coded_match_1x(ahead, m_off, lit);

            lazy_match_min_gain = min_gain_1x(ahead, lit, lit + ahead, l1, l2, l3);
            if (new_m_len >= m_len + lazy_match_min_gain) {
                /* Accept lazy match */
                if (l3) {
                    /* Code previous run + shortened match */
                    op = code_run_1x(op, out_start, &cc, in + ii_abs, lit, ahead);
                    lit = 0;
                    op = code_match_1x(op, out_start, &cc, ahead, m_off);
                } else {
                    lit += ahead;
                }
                /* Continue with the new (better) match */
                m_len = new_m_len;
                m_off = new_m_off;
                goto lazy_match_done;
            }
        }

        /* No lazy match won — code current match */
        /* ii_abs points to start of literal run */
        op = code_run_1x(op, out_start, &cc, in + ii_abs, lit, m_len);
        lit = 0;

        op = code_match_1x(op, out_start, &cc, m_len, m_off);
        swd.max_chain = max_chain;
        find_match(&swd, swd_base, in, in_len, &src_pos, m_len, 1 + ahead, &fm);
        look = fm.look;
        m_len = fm.m_len;
        m_off = fm.m_off;

    lazy_match_done: ;
        /* When we arrive here via goto:
         *   - The lazy loop already coded the shortened match (if l3) or added to lit
         *   - m_len/m_off already hold the lazy candidate's (better) match
         *   - look is already updated from the find_match inside the lazy loop
         *   - The while loop re-evaluates m_len/m_off as the "current" match
         *
         * When we fall through (no lazy):
         *   - We already coded the current match and called find_match above
         *   - m_len/m_off/look are set from that find_match
         *
         * Either way, we just continue the while loop.
         */
    }

    /* Store final literal run */
    if (lit > 0) {
        op = store_run_1x(op, out_start, in + ii_abs, lit);
    }

    /* EOF marker */
    *op++ = M4_MARKER | 1;
    *op++ = 0;
    *op++ = 0;

    *out_len_ptr = (uint)(op - out);
}


/* ================================================================
 * Kernel entry point
 * ================================================================ */

__kernel void lzo1x_block_compress_999(
    __global const uchar* in,
    __global       uchar* out,
    __global       uint*  out_len,
    const uint  in_sz,
    const uint  blk_size,
    const uint  worst_blk,
    __global uchar* swd_pool,       /* Pre-allocated SWD arrays, one per WI */
    const uint  swd_pool_count,     /* Number of SWD instances in pool */
    const uint  try_lazy,           /* Lazy match tries (0, 1, or 2) */
    const uint  max_chain           /* Max hash chain depth (e.g., 4096) */
)
{
    const uint wi = get_global_id(0);
    const uint total_wi = get_global_size(0);
    const uint total_blocks = (in_sz + blk_size - 1) / blk_size;

    if (wi >= swd_pool_count) return;

    /* Each work-item gets its own SWD pool region in global memory */
    __global uchar* my_swd = swd_pool + ((size_t)wi * SWD_POOL_STRIDE);

    for (uint b = wi; b < total_blocks; b += total_wi) {
        uint in_off = b * blk_size;
        __global const uchar* ip = in + in_off;
        __global uchar* op = out + (size_t)b * worst_blk;
        uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);

        lzo1x_999_compress_block(ip, in_len, op, &out_len[b],
                                  my_swd, try_lazy, max_chain);
    }
}
