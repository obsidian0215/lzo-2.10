# lzo_gpu — Usage & I/O modes

This document explains the three ways to run the GPU-enabled LZO compressor and the I/O and tuning options related to zero-copy / standard-copy and multi-threaded I/O.

## Unified Tool: lzo_gpu
The project is unified into a single binary `./lzo_gpu`.

### 1. Standalone mode
Basic usage: `./lzo_gpu [options] <input_file>`
Initializes OpenCL resources once and runs compression locally.

### 2. Daemon mode
Basic usage: `./lzo_gpu --daemon [options]`
Starts a long-running daemon that initializes OpenCL once and handles requests. Use `./lzo_gpu --stop-daemon` to shutdown the daemon cleanly.

### 3. Client mode (using daemon)
Basic usage: `./lzo_gpu --use-daemon [options] <input_file>`
Sends compression/decompression requests to a running daemon over a unix socket.

## Command-line Usage
The consolidated tool honors different flags depending on the mode.


## Primary modes & differences

### Zero-copy (default)
- Behavior: map a pinned device buffer (CL_MEM_ALLOC_HOST_PTR) and `fread` / `pread` directly into the mapped pointer. The device can read immediately — avoids explicit host→device transfers.
- Best for: integrated GPUs (iGPU) where mapped host pages are directly accessible by the device.
- Controlled by: `LZO_STANDARD_COPY=0` (default).

### Standard-copy
- Behavior: allocate a host buffer (aligned), read file into it, then explicitly upload via `clEnqueueWriteBuffer` into a device buffer.
- Best for: discrete GPUs (dGPU) or drivers where explicit upload path behaves better.
- Enabled by: `LZO_STANDARD_COPY=1`.

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

OpenCL & misc
- (No environment-based device selection enabled currently)

### Full environment variable table (name / allowed values / default / supported)

| Name | Values | Default | Supported by | Notes |
|-----:|:-------|:--------|:-------------|:------|
| LZO_STANDARD_COPY | 0 / 1 | 0 | standalone, client->daemon request | 0 = zero-copy (map & fread), 1 = standard (host->device upload). Daemon honors per-request option but may be configured to ignore. |

### Asynchronous uploads (LZO_ASYNC_UPLOAD)

Note: Asynchronous uploads (LZO_ASYNC_UPLOAD) have been removed from the codebase and are no longer supported. Any historical references in the performance notes remain for archival purposes; refer to `PERFORMANCE_SUMMARY.md` for past measurements.




### Environment defaults & dependency rules
- Default device selection is GPU.


## Client -> Daemon behavior
- `lzo_gpu_client` reads the above environment variables and sends them as per-request options to the daemon.
- The daemon may accept or ignore specific request flags (device choice, etc.) depending on its configuration.

## Example usage
- Standalone zero-copy (default):
  ./lzo_gpu input.bin -o out.lzo

- Standalone standard-copy:
  LZO_STANDARD_COPY=1 ./lzo_gpu input.bin -o out.lzo

- Client -> daemon (request-level options):

  export LZO_STANDARD_COPY=1
  ./lzo_gpu_client input.bin out.lzo

## Safety & fallback behavior
- The implementation uses standard POSIX I/O to ensure correctness.
- All operations check for EINTR and short-read conditions and report errors properly.

## Notes for maintainers
- The standalone and daemon implementations share the same design: use pinned host buffers when possible, allow zero-copy for iGPUs, and allow standard-copy uploads for dGPUs.
- Check `lzo_gpu_core.c` for the latest implementation details and test coverage. The previous `daemon_compress.c`/`daemon_decompress.c` implementations have been consolidated into `lzo_gpu_core.c` and are archived as `.bak` files.

---
The repository includes a consolidated Python runner `tools/bench.py` which re-runs the `/tmp/sample_*` benchmarks across the standard modes and prints a concise comparison table. Run it from the project root:

## More detailed examples & recommended settings

These examples show concrete `env` + command-line combinations and what they are intended to exercise. Use them as a starting point for tuning in your environment.


### 1) iGPU (integrated — zero-copy preferred)
Best for Intel/AMD APUs or integrated NV hardware where host mapped pages are directly accessible by the device.

```bash
# Prefer default zero-copy
export LZO_STANDARD_COPY=0
./lzo_gpu input.bin -o out.lzo
```

Why: zero-copy avoids DMA stage and uses host-mapped memory.


### 2) Discrete GPU (dGPU) — standard-copy often safer
Some drivers and PCIe stacks perform better with explicit host→device copies.

```bash
export LZO_STANDARD_COPY=1
./lzo_gpu input.bin -o out.lzo
```

Why: explicit copy yields more predictable throughput on PCIe dGPUs.


### 3) Very small files (desktop / script friendly)
For many small files, use default zero-copy.

```bash
./lzo_gpu smallfile.bin -o smallfile.lzo
```


### 4) Client → daemon example (per-request control)
Send per-request options from client using environment variables.

```bash
export LZO_STANDARD_COPY=1
./lzo_gpu_client large.dat out.lzo
```


### 5) For testing block-size choices or reproducing behavior
Force a fixed block size to reproduce or explore block-splitting impacts (KB):

```bash
./lzo_gpu -B 64KB input.bin -o out.lzo

```

## Tuning advice / troubleshooting

- Start with the following baseline to evaluate your system using a large file (>= 100MB):

  - iGPU baseline: LZO_STANDARD_COPY=0 (zero-copy default)
  - dGPU baseline: LZO_STANDARD_COPY=1 (standard copy)

- If you see low upload times but high kernel times, try changing the kernel variant (-L flag) — smaller block sizes (1k/1l) often improve throughput.

## Automated Benchmarking and Analysis

To perform a comprehensive scan of algorithm variants, block sizes, and thread counts:

1. Use the `param_scan.sh` script in the `tools/` directory. It automatically tests CPU and GPU (Standalone & Daemon) modes across all sample files.
   ```bash
   cd tools
   SAMPLES_DIR=/path/to/samples ./param_scan.sh --full
   ```
2. The results are saved in `exp_results/param_scan/` as CSV files, and a comprehensive summary report (ranking best configurations) is automatically generated.
3. You can manually re-run the analysis on a CSV file using:
   ```bash
   python3 tools/analyze_results.py exp_results/param_scan/your_results.csv
   ```

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