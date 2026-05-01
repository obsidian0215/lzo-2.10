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
#define LZO_DBG_COMP_PREFIX_BYTES_CONSUMED 26
#define LZO_DBG_COMP_PREFIX_LITERAL_BYTES  27
#define LZO_DBG_COMP_PREFIX_MATCH_BYTES    28
#define LZO_DBG_COMP_PREFIX_LITERAL_OPS    29
#define LZO_DBG_COMP_PREFIX_MATCH_OPS      30
#define LZO_DBG_COMP_PREFIX_PREWARM_STORES 31
#define LZO_DBG_COMP_SUFFIX_INPUT_BYTES    32
#define LZO_DBG_COMP_SUFFIX_LITERAL_BYTES  33
#define LZO_DBG_COMP_SUFFIX_MATCH_BYTES    34
#define LZO_DBG_COMP_SUFFIX_LITERAL_OPS    35
#define LZO_DBG_COMP_SUFFIX_MATCH_OPS      36
#define LZO_DBG_COMP_CORE_VECTOR_BATCHES   37
#define LZO_DBG_COMP_CORE_SLOW_STEPS       38
#define LZO_DBG_COMP_CORE_EXTEND8_ITERS    39
#define LZO_DBG_COMP_CORE_EXTEND4_ITERS    40
#define LZO_DBG_COMP_CORE_EXTEND1_ITERS    41
#define LZO_DBG_COMP_CORE_SKIP_ADVANCE     42
#define LZO_DBG_COMP_SUFFIX_CORE_CALLS     43
#define LZO_DBG_COMP_FAST_BATCH_ITERS      44
#define LZO_DBG_COMP_SLOW_SINGLE_ITERS     45
#define LZO_DBG_COMP_VALID_DISTANCE_FAIL   46
#define LZO_DBG_COMP_VALID_4B_MISMATCH     47
#define LZO_DBG_COMP_VALID_4B_MATCH        48
#define LZO_DBG_COMP_CORE_EXTEND32_ITERS   49
#define LZO_DBG_COMP_CORE_EXTEND_EXIT_32   50
#define LZO_DBG_COMP_CORE_EXTEND_EXIT_8    51
#define LZO_DBG_COMP_CORE_EXTEND_EXIT_4    52
#define LZO_DBG_COMP_CORE_EXTEND_EXIT_1    53
#define LZO_DBG_COMP_LITERAL_SMALL_OPS     54
#define LZO_DBG_COMP_LITERAL_SHORT_OPS     55
#define LZO_DBG_COMP_LITERAL_LONG_OPS      56
#define LZO_DBG_COMP_LITERAL_LONG_ZERO_RUNS 57
#define LZO_DBG_COMP_M3_LONG_OPS           58
#define LZO_DBG_COMP_M4_LONG_OPS           59
#define LZO_DBG_COMP_M3_LEN_ZERO_RUNS      60
#define LZO_DBG_COMP_M4_LEN_ZERO_RUNS      61
#define LZO_DBG_COMP_MATCH_DIST_LE_64      62
#define LZO_DBG_COMP_MATCH_DIST_LE_256     63
#define LZO_DBG_COMP_MATCH_DIST_LE_1024    64
#define LZO_DBG_COMP_MATCH_DIST_LE_4096    65
#define LZO_DBG_COMP_MATCH_DIST_LE_16384   66
#define LZO_DBG_COMP_MATCH_DIST_GT_16384   67
#define LZO_DBG_COMP_MATCH_LANE0           68
#define LZO_DBG_COMP_MATCH_LANE1           69
#define LZO_DBG_COMP_MATCH_LANE2           70
#define LZO_DBG_COMP_MATCH_LANE3           71
#define LZO_DBG_COMP_MATCH_SLOW_LANE       72
#define LZO_DBG_COMP_FAST_BATCH_FULL_MISS  73
#define LZO_DBG_COMP_SKIP_ADVANCE_BYTES    74
#define LZO_DBG_COMP_SKIP_ADVANCE_OPS      75
#define LZO_DBG_COMP_N                     76

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
#define LZO_DBG_DEC_TOKEN_GE64        16
#define LZO_DBG_DEC_TOKEN_32_63       17
#define LZO_DBG_DEC_TOKEN_16_31       18
#define LZO_DBG_DEC_TOKEN_LT16_MATCH  19
#define LZO_DBG_DEC_M2_AFTER_LITERAL  20
#define LZO_DBG_DEC_M2_SHORT2         21
#define LZO_DBG_DEC_M3_LONG_LEN       22
#define LZO_DBG_DEC_M4_LONG_LEN       23
#define LZO_DBG_DEC_LITERAL_LONG_LEN  24
#define LZO_DBG_DEC_POST_LIT0         25
#define LZO_DBG_DEC_POST_LIT1         26
#define LZO_DBG_DEC_POST_LIT2         27
#define LZO_DBG_DEC_POST_LIT3         28
#define LZO_DBG_DEC_DIRECT_COPY       29
#define LZO_DBG_DEC_GENERIC_COPY      30
#define LZO_DBG_DEC_MATCH_LEN_LE4     31
#define LZO_DBG_DEC_MATCH_LEN_LE8     32
#define LZO_DBG_DEC_MATCH_LEN_LE16    33
#define LZO_DBG_DEC_MATCH_LEN_GT16    34
#define LZO_DBG_DEC_N                 35

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
#define LZO_DBG_COMP_NOTE_MATCH_DIST(dbg, base, dist) lzo_dbg_note_match_dist((dbg), (base), (uint)(dist))

#define LZO_COMP_DBG_ARGS , __global uint* dbg_stats, uint dbg_base
#define LZO_COMP_DBG_PASS , dbg_stats, dbg_base
#define LZO_DEC_DBG_ARGS  , __global uint* dbg_stats, uint dbg_base
#define LZO_DEC_DBG_PASS  , dbg_stats, dbg_base
#else
#define LZO_DBG_COMP_ADD(dbg, base, idx, val) ((void)0)
#define LZO_DBG_DEC_ADD(dbg, base, idx, val)  ((void)0)
#define LZO_DBG_COMP_NOTE_STORE(dbg, base, valid, off) ((void)0)
#define LZO_DBG_COMP_NOTE_MATCH_DIST(dbg, base, dist) ((void)0)
#define LZO_COMP_DBG_ARGS
#define LZO_COMP_DBG_PASS
#define LZO_DEC_DBG_ARGS
#define LZO_DEC_DBG_PASS
#endif

#endif
