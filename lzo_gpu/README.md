# lzo_gpu

`lzo_gpu` 是本仓库中的 LZO OpenCL GPU 实现。

当前主线只保留标准压缩/解压路径：

- standalone 压缩/解压
- benchmark（压缩 + 解压 + roundtrip 校验）
- Linux 下的 daemon/client 模式
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
make OS=Windows_NT CC=gcc \
  OPENCL_INCLUDE_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" \
  OPENCL_LIB_DIR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/lib/x64"
```

如有需要可额外指定：

```bash
OPENCL_LIB_NAME=OpenCL
```

## Usage

```bash
./lzo_gpu input.bin -o out.lzo
./lzo_gpu -d out.lzo -o restored.bin
./lzo_gpu --bench 3 -B 64K input.bin
```

### Key options

- `-c`：压缩（默认）
- `-d`, `--decompress`：解压
- `-o`, `--output FILE`：输出文件（`-` 表示 stdout）
- `-a`, `--alg ALG`：`lzo1x` / `lzo1y`
- `-L`, `--level LEVEL`：`11..15`（默认 `14`）
- `-B`, `--block-size N`：固定 block size，支持 `B/KB/MB`（默认 `64KB`）
- `--local N`：OpenCL local work-group size（默认 `1`）
- `-v`, `--verbose`：打印详细时序统计
- `--bench [SECONDS]`：稳定 bench（压缩 + 解压 + verify）

### Environment variables

| 变量 | 作用 |
| --- | --- |
| `FORCE_OPENCL_DEVICE` | 设备优先级：`GPU` / `CPU` / `DEFAULT` / `ALL` |
| `LZO_STANDARD_COPY` | `0`=mapped/zero-copy 优先，`1`=显式 host->device copy |
| `LZO_GPU_DEBUG` | 启用 debug-enabled kernel 与 counters |

## Current implementation notes

- 压缩主线为标准 `D_BITS=11..20` 路径；
- 解压主线已实现并参与 bench roundtrip 校验；
- 运行时会复用 OpenCL context、kernel 与 workspace buffer；
- 支持 `.clbin` 以减少重复 bench 的内核编译成本。
