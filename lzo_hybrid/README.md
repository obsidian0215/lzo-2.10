# lzo_hybrid

`lzo_hybrid` 是本仓库中的 LZO OpenCL CPU/GPU 协同实现。它基于 `lzo_gpu_v2` 当前源码整理而来，用于承载 OpenCL GPU、OpenCL CPU、以及 CPU+GPU mixed 调度路径。

当前主线保留以下路径：

- standalone 压缩/解压
- benchmark（压缩 + 解压 + roundtrip 校验）
- Linux 下的 daemon/client 模式
- OpenCL GPU-only / CPU-only / CPU+GPU mixed
- `lzo1x` / `lzo1y`
- `D_BITS=11..15`

## Build

### Linux

```bash
make
```

### Windows (MSYS2 / MinGW-w64)

支持 Windows standalone / bench 路径。

示例：

```bash
make OS=Windows_NT CC=gcc   OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include"   OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

如有需要可额外指定：

```bash
OPENCL_LIB_NAME=OpenCL
```

## Usage

```bash
./lzo_hybrid input.bin -o out.lzo
./lzo_hybrid -d out.lzo -o restored.bin
./lzo_hybrid --bench 3 -B 64K input.bin
```

### Key options

- `-c`：压缩（默认）
- `-d`, `--decompress`：解压
- `-o`, `--output FILE`：输出文件（`-` 表示 stdout）
- `-a`, `--alg ALG`：`lzo1x` / `lzo1y`
- `-L`, `--level LEVEL`：`11..15`（默认 `14`）
- `-B`, `--block-size N`：固定 block size，支持 `B/KB/MB`
- `--local N`：OpenCL local work-group size（默认 `1`）
- `-v`, `--verbose`：打印详细时序统计
- `--bench [SECONDS]`：稳定 bench（压缩 + 解压 + verify）
- `--cpu-threads N`：OpenCL CPU 路径使用的 slots/工作项限制
- `--gpu-ratio F`：GPU 处理 block 的比例，`0.0` 为 CPU-only，`1.0` 为 GPU-only
- `--adaptive`：预留自适应调度入口，当前不改变固定比例调度

### Environment variables

| 变量 | 作用 |
| --- | --- |
| `LZO_OPENCL_DEVICE_TYPE` | 设备优先级：`GPU` / `CPU` / `DEFAULT` / `ALL` |
| `LZO_OPENCL_STRICT_DEVICE` | `1`=严格使用指定设备类型，找不到则失败 |
| `LZO_GPU_COMP_SLOTS` | GPU 压缩 range kernel 的 slots 上限覆盖 |
| `LZO_STANDARD_COPY` | `0`=mapped/zero-copy 优先，`1`=显式 host-device copy |
| `LZO_GPU_DECOMP_CHUNKED` | 解压 chunked readback 策略：`auto` / `on` / `off` |
| `LZO_GPU_DECOMP_CHUNK_KB` | 解压 chunked readback 的 chunk 大小 |
| `LZO_GPU_DEBUG` | 启用 debug-enabled kernel 与 counters |

## Current implementation notes

- `lzo_hybrid` 使用 `lzo_hybrid*` 主机源码和 `lzo_hybrid`/`lzo_hybrid.exe` 二进制名；
- OpenCL kernel 仍为 `lzo1x.cl` / `lzo1y.cl`，压缩/解压格式与 `lzo_gpu` 兼容；
- mixed 路径通过 `--gpu-ratio` 与 `--cpu-threads` 在同一进程内划分 OpenCL CPU/GPU block；
- CPU-only 与 GPU-only 是 mixed 调度的两个边界配置；
- range kernel 用于让 work-item/slots 与 block 数解耦；
- 运行时会复用 OpenCL context、kernel 与 workspace buffer；
- `.clbin` 只作为显式可选行为，默认从 OpenCL 源码构建 kernel。
