# lzo_hybrid

`lzo_hybrid` is the primary CPU+GPU collaborative LZO execution path. It is the default mode for comprehensive benchmarking, replacing separate GPU and CPU runs. It features:

- CPU compression/decompression from `lzo_cpu`
- GPU compression/decompression from `lzo_gpu`
- Adaptive and fixed split policies for workload partitioning
- In-memory and file-backed warmed benchmark modes
- Pure CPU (R=0.0) and Pure GPU (R=1.0) optimized execution paths

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
- `-L`, `--level`: Dictionary bits (11-16, 99=enhanced greedy, 999=optimal SWD)
- `-B`, `--block-size`: Fixed block size
- `--cpu-threads`: CPU worker threads (default: auto = all cores via `sysconf`)
- `--gpu-ratio`: Fixed GPU fraction (0.0 for pure CPU, 1.0 for pure GPU)
- `--adaptive`: Enable adaptive per-file CPU/GPU split
- `--sample-blocks`: Adaptive sample count
- `--local`: OpenCL work-group size
- `--bench`: Benchmark mode (e.g. `--bench 3`)

### Benchmarks

```bash
./lzo_hybrid --bench 3 -B 64K --cpu-threads 2 --gpu-ratio 0.7 /path/to/file
./lzo_hybrid --bench 3 --bench-io -B 64K --cpu-threads 2 --adaptive --sample-blocks 8 /path/to/file
```

## Current implementation notes

- Hybrid mode partitions input blocks between CPU and GPU workers using either fixed or adaptive split policies.
- R=0.0 optimization: When `--gpu-ratio` is 0.0, OpenCL initialization is skipped entirely for maximum CPU performance.
- R=1.0 path: Pure GPU execution bypasses host-side gather overhead when no CPU workers are active.
- Compression levels 99 (enhanced greedy) and 999 (optimal SWD) are supported across both CPU and GPU execution paths.
- Benchmarking includes per-block distribution logic with explicit gather/scatter around GPU kernels.
- Bench loop optimization: GPU kernel args are cached across iterations; input buffer upload and file re-read are skipped on repeated iterations (skip_input_upload).
- Thread auto-detection: defaults to all available cores, overridable via `--cpu-threads`.

## Adaptive scheduling model

The adaptive split model (`--adaptive`) is energy-aware, load-aware, and compute-resource-aware.

- **Throughput model**: Calibrates per-byte CPU throughput (Pc0) and GPU throughput (Pg0) at startup.
- **Compression-ratio gain** (gC/gG): Adjusts for actual vs reference compression ratio.
- **Load awareness**: Reads `/proc/stat` for CPU idle fraction, scales by thread count vs total cores.
- **GPU availability**: Monitors GPU utilization for scheduling.
- **Compute-resource awareness**: CPU capacity is measured as `Pc0 * threads * cpu_availability`, treating the thread count parameter as the CPU capacity bound.
- **Energy-aware correction**: During calibration, measures per-byte energy consumption for CPU (RAPL core domain `intel-rapl:0:0`) and GPU (RAPL uncore domain `intel-rapl:0:1`). The final ratio uses a 70% throughput-optimal + 30% energy-optimal blend.
- **Small-input guard**: If total input is smaller than GPU overhead (t0), routes everything to CPU.
- **Degenerate fallback**: Returns 0.5 when both effective throughputs are zero (decoupled from user-specified gpu_ratio).
