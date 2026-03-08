# tools

This directory contains the benchmark runners and analysis helpers used by the LZO CPU/GPU/hybrid experiments.

## Main entry points

- `bench_lzo.py` — current CPU / GPU / hybrid benchmark driver
- `generate-test-data.py` — synthetic corpus generator
- legacy shell scripts and plotting helpers — retained for older experiment flows

## Recommended current workflow

### Build the binaries first

```bash
make -C ../lzo_cpu gcc
make -C ../lzo_gpu
make -C ../lzo_hybrid
```

### Run targeted subset benchmarks

```bash
python3 bench_lzo.py --samples-dir /root/samples_subset --bench-seconds 1
python3 bench_lzo.py --samples-dir /root/samples_subset --hybrid-only --bench-seconds 1
```

### Run a larger sweep

```bash
python3 bench_lzo.py --samples-dir /root/samples --bench-seconds 3
```

## Binary assumptions

By default `bench_lzo.py` expects:

- CPU: `/root/lzo-2.10/lzo_cpu/lzo_cpu`
- GPU: `/root/lzo-2.10/lzo_gpu/lzo_gpu`
- Hybrid: `/root/lzo-2.10/lzo_hybrid/lzo_hybrid`

You can override the CPU binary with `LZO_CPU_BIN`.

## Windows notes

The benchmark scripts are Python-based and can still be used on Windows, but the supported build path is currently **MSYS2 / MinGW-w64** for the underlying binaries.

If you build the binaries on Windows, make sure the paths passed into the script match your local checkout instead of the default `/root/...` paths used on this Linux lab machine.
