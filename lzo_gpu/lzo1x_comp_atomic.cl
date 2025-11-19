#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

/* Atomic-publish comp frontend: include shared core and publish with atomics */
#define LZO_NO_DEFAULT_KERNEL
#define D_BITS 11
#include "lzo1x_1.cl"

__kernel void lzo1x_block_compress(__global const uchar *in ,
                                       __global       uchar *out,
                                       __global       uint  *out_len,
                                       const uint  in_sz,
                                       const uint  blk_size,
                                       const uint  worst_blk)
{
    const uint gid = get_global_id(0);
    uint in_off = gid * blk_size;
    __global const uchar* ip = in + in_off;
    __global uchar* op = out + gid * worst_blk;

    uint in_len = (in_off + blk_size <= in_sz) ? blk_size : (in_sz - in_off);
    if (in_len == 0) { out_len[gid] = 0; return; }

    lzo_dict_t dict[D_SIZE]; lzo_uint olen = 0;
#if 1
    do_compress(ip, in_len, op, &olen, 0, dict);
#else
    /* keep behavior identical but allow easy toggling during debugging */
    do_compress(ip, in_len, op, &olen, 0, dict);
#endif
#if defined(cl_khr_global_int32_base_atomics) || defined(cl_khr_global_int32_extended_atomics)
    atomic_xchg((volatile __global int *)&out_len[gid], (int)olen);
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#else
    out_len[gid] = olen;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif
}

