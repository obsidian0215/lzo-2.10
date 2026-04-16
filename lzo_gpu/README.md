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

### Host-memory copy mode

```bash
LZO_STANDARD_COPY=0 ./lzo_gpu --bench 3 -B 64K file
LZO_STANDARD_COPY=1 ./lzo_gpu --bench 3 -B 64K file
```

- `0`: map/zero-copy 优先（统一内存设备常用）
- `1`: standard host->device copy

### Pipeline 与 pipeline-overlap（重点）

- `LZO_PIPELINE_ENABLE=1`：开启 **chunked pipeline 压缩路径**（按 chunk 分段推进）。
- `LZO_PIPELINE_OVERLAP_ENABLE=1`：在 pipeline 路径上尝试 upload/compute overlap。
- 生效前提：`LZO_PIPELINE_ENABLE=1` **且** `LZO_STANDARD_COPY=1`。
  - 如果 `LZO_STANDARD_COPY=0`（map/zero-copy），overlap 开关会退化为无效。

### Environment variables（完整速查）

| 变量 | 取值 / 默认 | 作用 |
| --- | --- | --- |
| `FORCE_OPENCL_DEVICE` | `GPU`(默认) / `CPU` / `DEFAULT` / `ALL` | 指定 OpenCL 设备优先级 |
| `LZO_STANDARD_COPY` | `auto`(默认) / `0` / `1` | host 与 device 之间的数据读写方式 |
| `LZO_GPU_ENABLE_COMPACTION` | `0/1`（默认 `0`） | 启用 device-side compaction/pack 路径 |
| `LZO_GPU_FORCE_COMPACTION` | `0/1`（默认空） | 强制关闭/开启 compaction，覆盖自适应门控 |
| `LZO_PIPELINE_ENABLE` | `0/1`（默认 `0`） | 启用 chunked pipeline 压缩路径 |
| `LZO_PIPELINE_OVERLAP_ENABLE` | `0/1`（默认 `0`） | 启用 pipeline 的 upload/compute overlap（仅 `LZO_PIPELINE_ENABLE=1` 且 standard-copy 生效） |
| `LZO_PIPELINE_THRESHOLD_MB` | 整数 MB（默认 `64`） | pipeline 启动的最小输入大小 |
| `LZO_PIPELINE_CHUNK_BLOCKS` | 正整数（默认 `512`） | 每个 pipeline chunk 的 block 数 |
| `LZO_PIPELINE_ENTROPY_ENABLE` | `0/1`（默认 `0`） | 启用 entropy gate，按样本熵决定是否走 pipeline |
| `LZO_PIPELINE_ENTROPY_SAMPLE_KB` | 正整数 KB（默认 `256`） | 熵采样窗口大小 |
| `LZO_PIPELINE_ENTROPY_MAX` | 浮点（默认 `7.60`） | 熵门限，超过则回退非 pipeline |

建议：实验报告必须记录完整环境变量快照（尤其是 `LZO_PIPELINE_ENABLE` 与 `LZO_PIPELINE_OVERLAP_ENABLE`）。

## Current implementation notes

- `lzo_gpu` supports three compression levels:
  - Standard greedy: 11-15 D_BITS (default: 12).
  - Level 99: enhanced greedy kernel with 4-way dictionary probe and lazy matching, ~5% better compression ratio than standard greedy at ~600-940 MB/s.
  - Level 999: full Sliding Window Dictionary (SWD) port to GPU, matches CPU 999 compression ratio.
- GPU decompression is also implemented.
- It shows weaker block-size sensitivity than the current LZ4 GPU path on the same Intel iGPU platform.
- The project relies on a reusable OpenCL runtime plus precompiled `.clbin` support for repeated benchmark runs.
(End of file)
