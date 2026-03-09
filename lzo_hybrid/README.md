# lzo_hybrid

`lzo_hybrid` is the CPU+GPU collaborative LZO execution path in this repo. It uses:

- CPU compression/decompression from `lzo_cpu`
- GPU compression/decompression from `lzo_gpu`
- fixed or adaptive CPU/GPU split policies
- in-memory and file-backed warmed benchmark modes

## Build

### Linux

```bash
make -C ../lzo_gpu
make
```

Dependencies:

- C compiler (`gcc` recommended)
- OpenCL development headers and loader
- pthread support

### Windows (MSYS2 / MinGW-w64)

Supported Windows build target is **MSYS2 MinGW-w64**.

```bash
make -C ../lzo_gpu \
  OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"

make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

Notes:

- CUDA-installed OpenCL is a valid provider on Windows as long as the system driver also supplies the OpenCL runtime DLL.
- If your MinGW environment expects a different import library name, override `OPENCL_LIB_NAME`.
- Current Windows support is focused on standalone and benchmark use; Unix daemon/client mode remains Linux-only.

## Usage

```bash
./lzo_hybrid input.bin -o out.lzo
./lzo_hybrid -d out.lzo -o restored.bin
```

### Common options

- `-a`, `--alg`: `lzo1x` or `lzo1y`
- `-L`, `--level`: dictionary bits / compression level
- `-B`, `--block-size`: fixed block size
- `--cpu-threads`: CPU worker threads
- `--gpu-ratio`: fixed GPU fraction
- `--adaptive`: adaptive split mode
- `--sample-blocks`: adaptive sample count
- `--local`: OpenCL local size

### Benchmarks

```bash
./lzo_hybrid --bench 3 -B 64K --cpu-threads 2 --gpu-ratio 0.7 /path/to/file
./lzo_hybrid --bench 3 --bench-io -B 64K --cpu-threads 2 --adaptive --sample-blocks 8 /path/to/file
```

## Current implementation notes

- Recent scheduler work changed the hybrid split from a simple GPU-prefix policy to distributed block assignment with explicit gather/scatter around GPU kernels.
- Bench path temp-file handling was updated to avoid fixed-path collisions.
- The current verified subset results still rank `lzo_gpu` as the default fastest engine, while `lzo_hybrid` remains a workload-specific compromise.

## 2026-03 stitched full-corpus results

The current full-corpus hybrid analysis is derived from the stitched 83-file artifact:

- `../exp_results/runs/20260309_merged_full_83/lzo_param_sweep_merged.csv`
- analysis bundle: `/root/analysis/20260309_full_refresh/`

Current matched-corpus best-per-file medians:

- `lzo1x fixed`: `925.46 / 821.24 MB/s`
- `lzo1x adaptive`: `927.49 / 897.38 MB/s`
- `lzo1y fixed`: `926.01 / 812.25 MB/s`
- `lzo1y adaptive`: `942.86 / 900.74 MB/s`

Updated interpretation: on the refreshed 83-file corpus, adaptive hybrid is no longer weaker than fixed. GPU remains the default compression leader, while adaptive hybrid is currently the stronger decompression-oriented cooperative mode.
