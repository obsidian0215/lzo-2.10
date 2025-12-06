#!/usr/bin/env bash
# Simple script to run the analysis with plotting enabled under the 'dirtytrack' conda environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="$REPO_ROOT/exp_results/param_scan/param_scan.csv"
OUT_DIR="$REPO_ROOT/exp_results/param_scan_analysis"
OUT_JSON="$OUT_DIR/param_scan_analysis.json"
mkdir -p "$OUT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install Conda and ensure 'dirtytrack' environment exists." >&2
    exit 1
fi

echo "Activating conda environment: dirtytrack"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dirtytrack

echo "Running analysis (with plotting) -> outputs in $OUT_DIR"
python3 "$REPO_ROOT/tools/analyze_param_scan.py" --csv "$CSV_PATH" --out "$OUT_JSON"
echo "Running microbench analysis (top 12)"
python3 "$REPO_ROOT/tools/microbench_analysis.py" --sample-agg "$OUT_DIR/param_scan_sample_config_agg.csv" --topN 12 --outdir "$OUT_DIR"
echo "Running heuristics recommendation"
python3 "$REPO_ROOT/tools/recommend_heuristics.py" --sample-agg "$OUT_DIR/param_scan_sample_config_agg.csv" --outdir "$OUT_DIR"

echo "Analysis complete. See PNGs and CSVs under $REPO_ROOT/exp_results/param_scan_analysis"
