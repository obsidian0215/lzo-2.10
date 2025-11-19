#!/usr/bin/env bash
# Runner for lzo_gpu experiments
# This script is the entrypoint for running and analyzing lzo_gpu experiments.
# NOTE: lzo_cpu comparisons should use a separate runner in future (e.g. run_lzo_cpu.sh).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
ARCHIVE="$TOOLS/archived"
LZO_BIN="$ROOT/lzo_gpu/lzo_gpu"
OUT_BASE="$ROOT/exp_results"
LZO_GPU_LOGS="$OUT_BASE/lzo_gpu/logs"
TOOLS_LOGS="$OUT_BASE/tools/logs"

usage() {
  cat <<EOF
Usage: $0 <command> [options]
Commands:
  precompile            Build precompiled kernels (runs: make -C lzo_gpu precompile-combos)
  param-scan [opts]     Run full parameter scan (wraps tools/param_scan.sh)
  parse <indir> <out>   Parse profile logs (uses tools/parse_profile_logs.py)
  analyze               Run basic analysis (aggregates summary.csv and analysis CSVs)
  summarize             Aggregate experiment markdowns into a single summary (lzo_gpu only)
  plots <summary.csv>   Generate plots from the provided summary CSV (uses tools/generate_plots.py)
  archived-list         List archived scripts moved from tools/
  help                  Show this help

Examples:
  $0 precompile
  $0 param-scan -s /root/samples
  $0 parse /path/to/param_scans /tmp/summary_profiles.csv
  $0 analyze
  $0 summarize
  $0 plots exp_results/lzo_gpu/logs/summary.csv

Notes:
- This runner is specific to lzo_gpu experiments. If you later add lzo_cpu tests,
  create a separate runner (e.g. tools/run_lzo_cpu.sh) to avoid mixing outputs.
- Many legacy scripts were consolidated into tools/archived/ for clarity; archived
  scripts are still available in $ARCHIVE if you need to run them directly.
EOF
}

if [ "$#" -lt 1 ]; then usage; exit 1; fi
cmd=$1; shift

case "$cmd" in
  precompile)
    echo "Running precompile-combos (may take a while)..."
    (cd "$ROOT/lzo_gpu" && make precompile-combos)
    ;;

  param-scan)
    if [ ! -x "$TOOLS/param_scan.sh" ]; then
      echo "Param-scan runner not found or executable: $TOOLS/param_scan.sh" >&2
      exit 1
    fi
    echo "Starting lzo_gpu param scan using samples dir ${SAMPLES_DIR:-/root/samples}"
    bash "$TOOLS/param_scan.sh" "$@"
    ;;

  parse)
    IN=${1:-}
    OUT=${2:-}
    if [ -z "$IN" ]; then echo "parse requires input dir"; exit 1; fi
    OUT=${OUT:-"$TOOLS_LOGS/summary_profiles.csv"}
    mkdir -p "$(dirname "$OUT")"
    python3 "$TOOLS/parse_profile_logs.py" -i "$IN" -o "$OUT"
    echo "Wrote $OUT"
    ;;

  analyze)
    echo "Aggregating param-scan results into $LZO_GPU_LOGS/summary.csv"
    mkdir -p "$LZO_GPU_LOGS/param_scans"
    python3 "$TOOLS/analyze.py" -i "$LZO_GPU_LOGS/param_scans" -o "$LZO_GPU_LOGS/summary.csv"
    echo "Analysis complete; outputs under $LZO_GPU_LOGS"
    ;;

  summarize)
    OUT=${1:-"$LZO_GPU_LOGS/EXPERIMENT_SUMMARY_ALL.md"}
    if [ ! -f "$TOOLS/aggregate_experiment_markdowns.py" ]; then
      echo "Aggregate markdown script missing: $TOOLS/aggregate_experiment_markdowns.py" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$OUT")"
    python3 "$TOOLS/aggregate_experiment_markdowns.py" "$OUT"
    ;;

  plots)
    CSV=${1:-"$LZO_GPU_LOGS/summary.csv"}
    if [ ! -f "$CSV" ]; then echo "Summary CSV not found: $CSV" >&2; echo "Run '$0 analyze' first to produce $CSV"; exit 1; fi
    python3 "$TOOLS/generate_plots.py" "$CSV"
    ;;

  archived-list)
    ls -1 "$ARCHIVE" || true
    ;;

  help|--help|-h)
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
