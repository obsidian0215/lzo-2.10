#!/usr/bin/env python3
"""
extract_block.py - extract a single block from LZO container and write a new LZO container

Usage:
  ./extract_block.py <full.lzo> <block_index> <out_block.lzo>

This assumes the LZO container format as written by lzo_gpu/lzo_host:
  [u16 magic][u32 orig_size][u32 block_size][u32 nblk][u32 len[nblk]]<compressed data>

It writes a new container that contains a single block (nblk=1) whose compressed data is
copied from the given block index. The uncompressed size of the single-block container
is computed as min(block_size, orig_size - block_idx * block_size).
"""

import sys
import struct

if len(sys.argv) != 4:
    print("Usage: extract_block.py <full.lzo> <block_index> <out_block.lzo>")
    sys.exit(2)

in_path = sys.argv[1]
block_idx = int(sys.argv[2])
out_path = sys.argv[3]

with open(in_path, 'rb') as f:
    data = f.read()

# Offsets / header parsing
# magic: u16
# orig_size: u32
# block_size: u32
# nblk: u32
# len array: nblk x u32

if len(data) < 2 + 4 + 4 + 4 + 4:
    print("File too short or invalid format")
    sys.exit(1)

offset = 0
magic = struct.unpack_from('<H', data, offset)[0]; offset += 2
if magic != 0x4C5A:  # 'LZ'
    print(f"Invalid magic: 0x{magic:04x}")
    sys.exit(1)

orig_sz = struct.unpack_from('<I', data, offset)[0]; offset += 4
blk_sz = struct.unpack_from('<I', data, offset)[0]; offset += 4
nblk = struct.unpack_from('<I', data, offset)[0]; offset += 4
alg_id = struct.unpack_from('<I', data, offset)[0]; offset += 4

if block_idx < 0 or block_idx >= nblk:
    print(f"Invalid block index {block_idx}, nblk={nblk}")
    sys.exit(1)

if len(data) < offset + 4 * nblk:
    print("Invalid container: not enough length header data")
    sys.exit(1)

lens = list(struct.unpack_from('<' + 'I'*nblk, data, offset)); offset += 4*nblk

# compute start and length
start = offset + sum(lens[:block_idx])
length = lens[block_idx]

if start + length > len(data):
    print("Invalid block: extends past EOF")
    sys.exit(1)

block_data = data[start:start+length]

# compute uncompressed size for this block
block_uncomp = min(blk_sz, orig_sz - block_idx * blk_sz)

# create new container: header with nblk=1, lengths=[length], followed by block_data
with open(out_path, 'wb') as fo:
    fo.write(struct.pack('<H', magic))
    fo.write(struct.pack('<I', block_uncomp))
    fo.write(struct.pack('<I', blk_sz))
    fo.write(struct.pack('<I', 1))  # single block
    fo.write(struct.pack('<I', alg_id))
    fo.write(struct.pack('<I', length))
    fo.write(block_data)

print(f"Wrote single-block container to {out_path} (block_index={block_idx}, comp_len={length}, uncomp_len={block_uncomp})")
