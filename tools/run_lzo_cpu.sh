#!/usr/bin/env bash
# Placeholder runner for lzo_cpu experiments
# This is intentionally minimal: it documents expected behavior and creates
# the default output directory for CPU experiments. Implement actual tests
# here when you add lzo_cpu binaries and workloads.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$ROOT/exp_results"
LZO_CPU_LOGS="$OUT_BASE/lzo_cpu/logs"

usage() {
  cat <<EOF
Usage: $0 <command>
Commands:
  prepare    Create output directories for lzo_cpu experiments
  help       Show this help

Notes:
- This is a placeholder. Implement a dedicated CPU runner (e.g. run_lzo_cpu.sh)
  to run CPU-side experiments and store outputs under exp_results/lzo_cpu/logs.
EOF
}

if [ "$#" -lt 1 ]; then usage; exit 1; fi
cmd=$1; shift

case "$cmd" in
  prepare)
    mkdir -p "$LZO_CPU_LOGS/param_scans"
    echo "Created placeholder lzo_cpu logs dir: $LZO_CPU_LOGS"
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
