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

- `-a`, `--alg`: `lzo1x` or `lzo1y`
- `-L`, `--level`: dictionary bits / kernel level
- `-B`, `--block-size`: fixed block size
- `--local`: OpenCL local work-group size
- `--bench [SECONDS]`: warmed benchmark loop

### Device selection

```bash
FORCE_OPENCL_DEVICE=GPU ./lzo_gpu --bench 3 -B 64K file
FORCE_OPENCL_DEVICE=CPU ./lzo_gpu --bench 3 -B 64K file
```

## Current implementation notes

- `lzo_gpu` remains the default strongest engine in the verified LZO family results.
- It shows weaker block-size sensitivity than the current LZ4 GPU path on the same Intel iGPU platform.
- The project relies on a reusable OpenCL runtime plus precompiled `.clbin` support for repeated benchmark runs.

## 2026-03 stitched full-corpus results

The current LZO GPU baseline should be read from the stitched 83-file artifact:

- `../exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- provenance: `../exp_results/runs/20260309_merged_full_83/merge_manifest.json`

Current matched-corpus best-per-file medians:

- `GPU lzo1x`: `1334.63 MB/s` compress total, `808.86 MB/s` decompress total
- `GPU lzo1y`: `1330.73 MB/s` compress total, `808.09 MB/s` decompress total

These figures come from one matched 83-file corpus shared with the refreshed LZ4 analysis. They supersede older subset and 36-file summaries.
