lzo_gpu
=======

This folder contains the GPU-backed LZO utilities and precompiled OpenCL kernels.

Usage (host `lzo_gpu`):

- Compress:
  - `./lzo_gpu [--debug|-v] [--verify|-c] [-L level] [-o out.lzo] input_file`
    - `-L|--level LEVEL` selects the compression kernel variant. Default is `1`.
    - Supported LEVEL values:
      - `1`  : default LZO1X-1 compressor (kernel: `lzo1x_1`)
      - `1k` : LZO1X-1K variant (kernel: `lzo1x_1k`)
      - `1l` : LZO1X-1L variant (kernel: `lzo1x_1l`)
      - `1o` : LZO1X-1O variant (kernel: `lzo1x_1o`)
    - `--verify` / `-c` in compress mode performs an in-memory roundtrip verify after compression.

- Decompress:
  - `./lzo_gpu -d [-v] [--verify|-c ORIG] [-o out_file] input.lzo`
    - `--verify/-c ORIG` verifies that the decompressed output equals `ORIG` before writing (safe default).
    - Use `-o -` to write raw decompressed bytes to stdout (pipeline friendly; non-data output is suppressed).

Notes
-----
- The Makefile auto-discovers any `*.cl` kernels in this directory and will precompile them during `make`.
- New compression `-L` levels map directly to kernel base names (for convenient precompilation and testing).

Regression script
-----------------
A helper script `run_samples_levels.sh` is provided to run multi-level regression tests over `../samples/`.
It compresses each sample for each supported level and writes compressed `.lzo` outputs to `test_outputs_levels/<level>/`.

Important: when `--verify` is used for decompression, the program will not write the decompressed output unless an explicit `-o` is given. The regression script and experiment runners use `--verify` without `-o` by default to avoid creating large decompressed files on disk; they rely on the verify exit status to determine correctness.

Example
-------
  cd lzo_gpu
  make
  ./lzo_gpu -L 1k ../samples/test_16kb_mixed.dat -o out.lzo
  # Verify only, do not write decompressed output:
  ./lzo_gpu -d --verify ../samples/test_16kb_mixed.dat out.lzo
  # If you do need the decompressed file, pass -o explicitly:
  ./lzo_gpu -d --verify ../samples/test_16kb_mixed.dat out.lzo -o out.dec

