Experiment scripts consolidation

This project has multiple helper scripts for running experiments, parsing logs, and generating rankings. To reduce clutter and provide a single entrypoint, use `tools/experiment_manager.sh` which exposes common operations.

Canonical scripts (kept):

- `tools/run_param_scan_full.sh`
  - Conservative, serial runner that performs the full parameter scan and writes per-run logs under `lzo_gpu/experiments/logs/param_scans/lzo1x_1l_vec/`.
  - It already deletes generated compressed files after decompress+verify to avoid filling disk.

- `tools/parse_profile_logs.py`
  - Parser that takes `--input-dir` and `--out` options and writes per-combo per-run CSVs and a combined summary CSV.

- `tools/generate_separate_rankings.py`
  - Generates `ranking_comp.csv` and `ranking_decomp.csv` from the summary and per-combo CSVs.

- `tools/experiment_manager.sh`
  - High-level wrapper to start scans, parse, rank, monitor, and show summaries. Uses `conda run -n dirtytrack` by default; set `USE_CONDA=0` to disable.

Legacy helpers (not recommended for new runs):
- `tools/param_scan_lzo1x_1l_vec.sh` (template)
- `tools/run_profiler_for_candidates.sh` (used earlier for profiling "practical" combos)

If you want, I can move legacy helpers to `tools/legacy/` to declutter the top-level `tools/` directory. I will not remove or modify them without your approval.

Conda and environment notes
- Default manager assumes a conda env named `dirtytrack`.
- To run manager without conda, export `USE_CONDA=0`.

Examples

Start full scan in background (uses conda env `dirtytrack` by default):

  bash tools/experiment_manager.sh run-param-scan

Parse results (explicit directories):

  bash tools/experiment_manager.sh parse lzo_gpu/experiments/logs/param_scans/lzo1x_1l_vec lzo_gpu/experiments/logs/param_scans/lzo1x_1l_vec/summary_profiles.csv

Generate rank CSVs:

  bash tools/experiment_manager.sh rank

Monitor running scan:

  bash tools/experiment_manager.sh monitor

Next steps
- If you approve, I can archive legacy scripts into `tools/legacy/` and keep only the canonical set above.
- I can also add an optional flag to persist compressed outputs for debugging (`--keep-out`) or to run scans in parallel with bounded concurrency.
