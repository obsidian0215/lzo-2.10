#ifndef LZO_GPU_DEBUG_H
#define LZO_GPU_DEBUG_H

#ifndef LZO_GPU_DEBUG_COUNTERS_RUNTIME
#define LZO_GPU_DEBUG_COUNTERS_RUNTIME 0
#endif

#if LZO_GPU_DEBUG_COUNTERS_RUNTIME
#define LZO_DBG_COMP_SEARCH_ITERS          0
#define LZO_DBG_COMP_MATCH_FOUND           1
#define LZO_DBG_COMP_LITERAL_BYTES         2
#define LZO_DBG_COMP_MATCH_BYTES           3
#define LZO_DBG_COMP_INPUT_BYTES           4
#define LZO_DBG_COMP_OUTPUT_BYTES          5
#define LZO_DBG_COMP_DICT_LOOKUPS          6
#define LZO_DBG_COMP_DICT_STORES           7
#define LZO_DBG_COMP_EPOCH_VALID_HITS      8
#define LZO_DBG_COMP_EPOCH_MISMATCH_MISS   9
#define LZO_DBG_COMP_LIVE_SLOT_OVERWRITES  10
#define LZO_DBG_COMP_STALE_SLOT_OVERWRITES 11
#define LZO_DBG_COMP_LITERAL_OPS           12
#define LZO_DBG_COMP_MATCH_OPS             13
#define LZO_DBG_COMP_M2_MATCHES            14
#define LZO_DBG_COMP_M3_MATCHES            15
#define LZO_DBG_COMP_M4_MATCHES            16
#define LZO_DBG_COMP_TAIL_LITERAL_BYTES    17
#define LZO_DBG_COMP_MATCH_MISS_AFTER_VALID_HIT 18
#define LZO_DBG_COMP_MATCH_MISS_AFTER_EPOCH_MISMATCH 19
#define LZO_DBG_COMP_SHARED_OWNER_BLOCKS   20
#define LZO_DBG_COMP_NOSHARE_FASTPATH_BLOCKS 21
#define LZO_DBG_COMP_SLOT_OVERWRITE_SAME_OWNER 22
#define LZO_DBG_COMP_SLOT_OVERWRITE_CROSS_OWNER 23
#define LZO_DBG_COMP_SHARED_TABLE_PROBE_COUNT 24
#define LZO_DBG_COMP_SHARED_TABLE_WRITE_COUNT 25
#define LZO_DBG_COMP_N                     26

#define LZO_DBG_DEC_TOKENS            0
#define LZO_DBG_DEC_LITERAL_BYTES     1
#define LZO_DBG_DEC_MATCH_BYTES       2
#define LZO_DBG_DEC_SMALL_OFFSETS     3
#define LZO_DBG_DEC_OUTPUT_ERROR      4
#define LZO_DBG_DEC_LITERAL_OPS       5
#define LZO_DBG_DEC_MATCH_OPS         6
#define LZO_DBG_DEC_OVERLAP_MATCHES   7
#define LZO_DBG_DEC_M2_MATCHES        8
#define LZO_DBG_DEC_M3_MATCHES        9
#define LZO_DBG_DEC_M4_MATCHES        10
#define LZO_DBG_DEC_FIRST_LITERAL_RUN_BYTES 11
#define LZO_DBG_DEC_FIRST_LITERAL_RUN_OPS   12
#define LZO_DBG_DEC_POST_MATCH_LITERAL_BYTES 13
#define LZO_DBG_DEC_POST_MATCH_LITERAL_OPS   14
#define LZO_DBG_DEC_EOF_MARKERS       15
#define LZO_DBG_DEC_N                 16

static inline void lzo_dbg_add(__global uint* dbg, uint base, uint idx, uint val)
{
    if (dbg) dbg[base + idx] += val;
}

#define LZO_DBG_COMP_ADD(dbg, base, idx, val) lzo_dbg_add((dbg), (base), (idx), (uint)(val))
#define LZO_DBG_DEC_ADD(dbg, base, idx, val)  lzo_dbg_add((dbg), (base), (idx), (uint)(val))
#define LZO_DBG_COMP_NOTE_STORE(dbg, base, valid, off) \
    do { \
        LZO_DBG_COMP_ADD((dbg), (base), LZO_DBG_COMP_DICT_STORES, 1u); \
        if ((off) != 0u) { \
            if (valid) LZO_DBG_COMP_ADD((dbg), (base), LZO_DBG_COMP_LIVE_SLOT_OVERWRITES, 1u); \
            else LZO_DBG_COMP_ADD((dbg), (base), LZO_DBG_COMP_STALE_SLOT_OVERWRITES, 1u); \
            LZO_DBG_COMP_ADD((dbg), (base), LZO_DBG_COMP_SLOT_OVERWRITE_SAME_OWNER, 1u); \
        } \
    } while (0)

#define LZO_COMP_DBG_ARGS , __global uint* dbg_stats, uint dbg_base
#define LZO_COMP_DBG_PASS , dbg_stats, dbg_base
#define LZO_DEC_DBG_ARGS  , __global uint* dbg_stats, uint dbg_base
#define LZO_DEC_DBG_PASS  , dbg_stats, dbg_base
#else
#define LZO_DBG_COMP_ADD(dbg, base, idx, val) ((void)0)
#define LZO_DBG_DEC_ADD(dbg, base, idx, val)  ((void)0)
#define LZO_DBG_COMP_NOTE_STORE(dbg, base, valid, off) ((void)0)
#define LZO_COMP_DBG_ARGS
#define LZO_COMP_DBG_PASS
#define LZO_DEC_DBG_ARGS
#define LZO_DEC_DBG_PASS
#endif

#endif
