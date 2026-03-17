# lzo_cpu

`lzo_cpu` is the native CPU baseline used by the GPU and hybrid comparisons in this tree.

It builds a standalone benchmarkable CLI from the local `lzo_cpu` sources and does not depend on OpenCL.

## Build

### Linux

```bash
make gcc
```

### Windows (MSYS2 / MinGW-w64)

```bash
make win32-mingw
```

The Makefile already includes a MinGW target. On Windows this is the recommended path for consistency with the GPU and hybrid tooling.

## Usage

```bash
./lzo_cpu input.bin -o out.lzo
./lzo_cpu -d out.lzo -o restored.bin
```

This binary is also used by `tools/bench_lzo.py` as the CPU reference engine.

## Notes

- Thread count defaults to auto (0), resolved at runtime via `sysconf(_SC_NPROCESSORS_ONLN)`.
- `--cpu-threads N` overrides this (0 = auto, N > 0 = fixed).
- Fallback to 4 if sysconf is unavailable.
- `lzo_cpu` is the preferred CPU implementation on the current Intel platform.
- CPU OpenCL was explored separately through the GPU backend, but the native CPU path remains the recommended CPU baseline.
