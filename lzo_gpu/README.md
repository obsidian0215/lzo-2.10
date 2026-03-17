# lzo_gpu

`lzo_gpu` is the OpenCL GPU implementation of LZO in this repository.

It provides:

- standalone compression/decompression
- benchmark mode with kernel and total throughput reporting
- optional daemon/client mode on Linux
- support for `lzo1x` and `lzo1y`

## Build

### Linux

```bash
make
```

### Windows (MSYS2 / MinGW-w64)

Supported Windows build target is **MSYS2 MinGW-w64**.

Example with NVIDIA CUDA-provided OpenCL headers/libs:

```bash
make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

If needed, also override:

```bash
OPENCL_LIB_NAME=OpenCL
```

Notes:

- CUDA on Windows includes OpenCL development files; the runtime comes from the installed GPU driver.
- Current Windows support targets standalone and benchmark flows.
- Linux-only daemon/client features are compiled out on Windows.

## Usage

```bash
./lzo_gpu input.bin -o out.lzo
./lzo_gpu -d out.lzo -o restored.bin
./lzo_gpu --bench 3 -B 64K input.bin
```

### Key options

- `-c`: compress mode (default)
- `-d`, `--decompress`: decompress mode
- `-o`, `--output FILE`: output file (use '-' for stdout)
- `-a`, `--alg ALG`: algorithm (`lzo1x`, `lzo1y`) (default: `lzo1x`)
- `-L`, `--level LEVEL`: compression level: 11-15 = D_BITS (default: 12), 99 = enhanced greedy, 999 = optimal (SWD)
- `-B`, `--block-size N`: fixed block size (B/KB/MB) (default: 16KB)
- `-v`, `--verbose`: enable performance statistics
- `--local N`: OpenCL local work-group size (default: 1)
- `--bench [SECONDS]`: warmed benchmark loop (compress+decompress+verify)

### Device selection

```bash
FORCE_OPENCL_DEVICE=GPU ./lzo_gpu --bench 3 -B 64K file
FORCE_OPENCL_DEVICE=CPU ./lzo_gpu --bench 3 -B 64K file
```

## Current implementation notes

- `lzo_gpu` supports three compression levels:
  - Standard greedy: 11-15 D_BITS (default: 12).
  - Level 99: enhanced greedy kernel with 4-way dictionary probe and lazy matching, ~5% better compression ratio than standard greedy at ~600-940 MB/s.
  - Level 999: full Sliding Window Dictionary (SWD) port to GPU, matches CPU 999 compression ratio.
- GPU decompression is also implemented.
- It shows weaker block-size sensitivity than the current LZ4 GPU path on the same Intel iGPU platform.
- The project relies on a reusable OpenCL runtime plus precompiled `.clbin` support for repeated benchmark runs.
(End of file)
