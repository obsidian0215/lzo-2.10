# lzo_gpu — Usage & I/O modes

This document explains the three ways to run the GPU-enabled LZO compressor and the I/O and tuning options related to zero-copy / standard-copy and multi-threaded I/O.

## Binaries
- `./lzo_gpu` — Standalone, single-command compressor that initializes OpenCL resources and runs compression locally.
- `./lzo_gpu_daemon` — Long-running daemon that initializes OpenCL once and handles requests from clients.
- `./lzo_gpu_client` — Client program that sends compression/decompression requests to a running daemon over a unix socket.

## Quick command-line usage
The README below documents per-binary CLI flags and which environment variables they honor.

### ./lzo_gpu (standalone)
- Basic usage: ./lzo_gpu [--debug|-v] [--verify|-c] [-L <level>] [-B <blocksize>] [--local <N>] [-o <out.lzo>] <input_file>
  - -L|--level <1|1k|1l|1o|1-9> : compression level / kernel variant (default: 1l for GPU-optimized)
  - -o|--output <path>         : output archive path (default: input + .lzo)
  - --verify|-c                : (compress) in-memory roundtrip verify
  - -d|--decompress             : switch to decompress mode (see help output)
  - -B <blocksize>|--block-size <blocksize> : fix block size (units accepted: B/KB/MB). When provided adaptive selection is disabled.
  - --local <N>                 : set local work-group size for kernels (1,8,64). Compression kernels require local=1 and will be forced to 1.
  - -h|--help                   : show help (detailed usage and environment variables)

### ./lzo_gpu_daemon (daemon)
- Basic usage: ./lzo_gpu_daemon [--help]
  - -h|--help : print daemon usage and environment variables / per-request options

### ./lzo_gpu_client (client)
- Basic usage: ./lzo_gpu_client [--help] [-d] [-l <1|1k|1l|1o|1-9>] [-B blocksize] [--local N] <input_file> <output_file>
  - -l|--level   : compression level (used for compression requests)
  - -B|--block-size : fixed block size for compression (sent as per-request option to daemon)
  - --local N : local workgroup size for compressors/decompressors (sent as per-request option to daemon)
  - -d|--decompress : run in decompress mode (client will ask daemon to decompress)
  - -h|--help    : show help and the environment variables the client will include in each request


## Primary modes & differences

### Zero-copy (default)
- Behavior: map a pinned device buffer (CL_MEM_ALLOC_HOST_PTR) and `fread` / `pread` directly into the mapped pointer. The device can read immediately — avoids explicit host→device transfers.
- Best for: integrated GPUs (iGPU) where mapped host pages are directly accessible by the device.
- Controlled by: `LZO_STANDARD_COPY=0` (default).

### Standard-copy
- Behavior: allocate a host buffer (aligned), read file into it, then explicitly upload via `clEnqueueWriteBuffer` into a device buffer.
- Best for: discrete GPUs (dGPU) or drivers where explicit upload path behaves better.
- Enabled by: `LZO_STANDARD_COPY=1`.

### Multi-threaded I/O (pread + parallel uploads)
- Behavior: split the input file into sub-ranges and use `pread` in worker threads to parallelize file reads (or parallel `clEnqueueWriteBuffer` uploads). Helpful when file read latency is bottleneck (fast NVMe, high read concurrency).
- Controlled by:
  - `LZO_MT_IO=1` — enable multi-threaded I/O
  - `LZO_MT_IO_THREADS=N` — number of worker threads (1..32). Defaults to 4 when `LZO_MT_IO` is enabled.
- Notes:
  - Threads with zero-length subranges are skipped — no useless threads are spawned.
  - If thread creation fails, code gracefully falls back to single-threaded `fread`.

## CLI options (summary)
Common flags available to the binaries:
- `-L|--level <1|1k|1l|1o|1-9>` : compression level / kernel variant (default: 1l).
- `-B|--block-size <size>` : fixed per-block size (accepts units: B/KB/MB). When provided, adaptive block selection is disabled.
- `--local <N>` : local work-group size for kernels (1,8,64).
- `-d|--decompress` : decompress mode.
- `-o|--output <file>` : specify output file.
- `--help|-h` : display help.

## Key environment variables (summary)
Grouped and explained concisely.

I/O mode
- `LZO_STANDARD_COPY=0|1` — 0 = zero-copy (map & read), 1 = standard copy (read into host -> upload). Default: 0.

MT I/O
- `LZO_MT_IO=0|1` — enable multi-threaded pread / parallel upload.
- `LZO_MT_IO_THREADS=N` — 1..32 (default 4 when mt_io enabled).

OpenCL & misc
- `LZO_OPENCL_DEVICE=CPU|GPU` — device preference (may be ignored by daemon)

### Full environment variable table (name / allowed values / default / supported)

| Name | Values | Default | Supported by | Notes |
|-----:|:-------|:--------|:-------------|:------|
| LZO_STANDARD_COPY | 0 / 1 | 0 | standalone, client->daemon request | 0 = zero-copy (map & fread), 1 = standard (host->device upload). Daemon honors per-request option but may be configured to ignore. |
| LZO_MT_IO | 0 / 1 | 0 | standalone, client->daemon request | Enables multi-threaded pread + parallel uploads. Worker thread fallback to single-threaded fread on error. |
| LZO_MT_IO_THREADS | integer (1-32) | 4 when LZO_MT_IO=1; otherwise N/A | standalone, client->daemon request | Number of I/O worker threads. Ignored unless LZO_MT_IO=1. |
| LZO_OPENCL_DEVICE | CPU / GPU | GPU | standalone, daemon | Device preference — daemon may ignore depending on its configuration and available devices. |

### Asynchronous uploads (LZO_ASYNC_UPLOAD)

Note: Asynchronous uploads (LZO_ASYNC_UPLOAD) have been removed from the codebase and are no longer supported. Any historical references in the performance notes remain for archival purposes; refer to `PERFORMANCE_SUMMARY.md` for past measurements.




### Environment defaults & dependency rules
- Default device selection is GPU unless `LZO_OPENCL_DEVICE=CPU`.
- `LZO_MT_IO_THREADS` is meaningful only when `LZO_MT_IO=1`; default 4 threads in that case (bounded to 1..32).


## Client -> Daemon behavior
- `lzo_gpu_client` reads the above environment variables and sends them as per-request options to the daemon.
- The daemon may accept or ignore specific request flags (device choice, etc.) depending on its configuration.

## Example usage
- Standalone zero-copy (default):

  LZO_MT_IO=1 LZO_MT_IO_THREADS=8 ./lzo_gpu input.bin -o out.lzo

- Standalone standard-copy + MT uploads:

  LZO_STANDARD_COPY=1 LZO_MT_IO=1 LZO_MT_IO_THREADS=8 ./lzo_gpu input.bin -o out.lzo

- Client -> daemon (request-level options):

  export LZO_STANDARD_COPY=1
  export LZO_MT_IO=1
  export LZO_MT_IO_THREADS=8
  ./lzo_gpu_client input.bin out.lzo

## Safety & fallback behavior
- If multi-threaded reads or thread-creation fail, the implementation falls back to single-threaded `fread` to ensure correctness.
- All MT I/O worker threads check for EINTR and short-read conditions and report errors properly.

## Notes for maintainers
- The standalone and daemon implementations share the same design: use pinned host buffers when possible, allow zero-copy for iGPUs, allow standard-copy uploads for dGPUs, and optionally parallelize file I/O.
- Check `lzo_gpu_standalone.c` and `lzo_gpu_core.c` for the latest implementation details and test coverage. The previous `daemon_compress.c`/`daemon_decompress.c` implementations have been consolidated into `lzo_gpu_core.c` and are archived as `.bak` files.

---
The repository includes a consolidated Python runner `tools/bench.py` which re-runs the `/tmp/sample_*` benchmarks across the standard modes and prints a concise comparison table. Run it from the project root:

## More detailed examples & recommended settings

These examples show concrete `env` + command-line combinations and what they are intended to exercise. Use them as a starting point for tuning in your environment.


### 1) iGPU (integrated — zero-copy preferred)
Best for Intel/AMD APUs or integrated NV hardware where host mapped pages are directly accessible by the device.

```bash
# Prefer the default zero-copy + multi-threaded file reads to minimize explicit upload cost
export LZO_STANDARD_COPY=0
export LZO_MT_IO=1
export LZO_MT_IO_THREADS=4
./lzo_gpu input.bin -o out.lzo
```

Why: zero-copy avoids DMA stage and multi-threaded pread reduces read latency on NVMe.


### 2) Discrete GPU (dGPU) — standard-copy often safer
Some drivers and PCIe stacks perform better with explicit host→device copies.

```bash
export LZO_STANDARD_COPY=1
export LZO_MT_IO=1
export LZO_MT_IO_THREADS=8
./lzo_gpu input.bin -o out.lzo
```

Why: explicit copy + parallel uploads often yields more predictable throughput on PCIe dGPUs.


### 3) Very small files (desktop / script friendly)
For many small files, multi-threaded I/O is overhead; use default zero-copy without MT_IO.

```bash
unset LZO_MT_IO
./lzo_gpu smallfile.bin -o smallfile.lzo
```


### 4) Client → daemon example (per-request control)
Send per-request options from client using environment variables. The daemon receives these per-request and will apply them where it can.

```bash
export LZO_STANDARD_COPY=1
export LZO_MT_IO=1
export LZO_MT_IO_THREADS=8
./lzo_gpu_client large.dat out.lzo
```


### 5) For testing block-size choices or reproducing behavior
Force a fixed block size to reproduce or explore block-splitting impacts (KB):

```bash
./lzo_gpu -B 64KB input.bin -o out.lzo

```

## Tuning advice / troubleshooting

- Start with the following baseline to evaluate your system using a large file (>= 100MB):

  - iGPU baseline: LZO_MT_IO=1 LZO_MT_IO_THREADS=4 (zero-copy default)
  - dGPU baseline: LZO_STANDARD_COPY=1 LZO_MT_IO=1 LZO_MT_IO_THREADS=8

- If you see low upload times but high kernel times, try changing the kernel variant (-L flag) — smaller block sizes (1k/1l) often improve throughput.
- If disk read time dominates and you're on NVMe, increase LZO_MT_IO_THREADS (8–16) to reduce read bottleneck. Avoid excessive threads (>32) as it causes high context switching.

### Common diagnostics

- Enable debug traces to inspect timings: use the standalone or client `--debug` flag — this prints time breakdowns for each stage.
- If daemon is not applying preferences, confirm `lzo_gpu_client` is sending options (check client help) and daemon's logs for accepted/ignored options.

## Next steps / further optimizations (ideas)

1. The `python3 tools/bench.py run` command automatically runs the common modes (zero, zero+mt, std, std+mt) across sample files and produces a comparison CSV (`lzo_gpu/benchmark_mt_io_results.csv`).
2. Add CI job to run the benchmark on representative hardware or a mocked I/O environment to prevent regressions.
3. Add small runtime self-test (client mode) that verifies a round-trip for small files exercising each combination of flags.

The `tools/bench.py` runner and `tools/analysis.py` post-processing utilities are available — see `python3 tools/bench.py run --help` and `python3 tools/analysis.py --help` for usage and examples.

## Test data generator (for benchmarks)

The repository includes a small helper script to generate test data used by benchmarks and experiments:

  python3 ../tools/generate-test-data.py SIZE [--pattern zero|random|repeat|structured|mixed] [--output PATH]

  Or from repo root:

  python3 tools/generate-test-data.py SIZE [--pattern zero|random|repeat|structured|mixed] [--output PATH]

Defaults and useful options:
- `--pattern-size N` — controls the repeating pattern size (default 1 for repeat pattern)
- `--density` — portion of file which is structured/repeat (defaults depend on pattern)
- `--symbols` — number of unique symbols used for `repeat` patterns (default 1)
- `--noise` — relative noise injected into `repeat` patterns (default 0.0001)

This tool uses streaming writers to avoid excessive memory use and prints a sample-based Shannon entropy for the resulting file.