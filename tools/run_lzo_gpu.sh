#!/usr/bin/env bash
# Unified LZO GPU benchmark runner
# Integrates param-scan functionality and analysis workflows

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/tools"
ARCHIVE="$TOOLS/archived"
LZO_BIN="$ROOT/lzo_gpu/lzo_gpu"
OUT_BASE="$ROOT/exp_results"
LZO_GPU_LOGS="$OUT_BASE/lzo_gpu/logs"
TOOLS_LOGS="$OUT_BASE/tools/logs"
WRAPDIR="$ROOT/lzo_gpu"

# Param-scan defaults
SAMPLES_DIR_DEFAULT="/root/samples"
SAMPLES_DIR="${SAMPLES_DIR:-$SAMPLES_DIR_DEFAULT}"
DRY_RUN=0
REPEATS=${REPEATS:-5}
PARAM_SCAN_OUT_DIR="${OUT_DIR:-$LZO_GPU_LOGS/param_scans}"

usage() {
  cat <<EOF
Usage: $0 <command> [options]

Commands:
  precompile            Build precompiled kernels (runs: make -C lzo_gpu precompile-combos)

  param-scan [opts]     Run full parameter scan with configurable parameters
    Options:
      -s, --samples DIR   Directory containing sample files (default: $SAMPLES_DIR_DEFAULT)
      -n, --dry-run       Print commands and record planned actions, do not execute
      -r, --repeats N     Number of repeats per configuration (default: 5)
      -o, --output DIR    Output directory for scan results (default: exp_results/lzo_gpu/logs/param_scans)
      -h, --help          Show param-scan specific help

    Environment Variables:
      LZO_COMP_LEVELS     Comma-separated compression levels (default: 1,1k,1l,1o)
      LZO_DEBUG           Set to 1 to enable debug output

  parse <indir> <out>   Parse profile logs (uses tools/parse_profile_logs.py)
  analyze               Run basic analysis (aggregates summary.csv and analysis CSVs)
  summarize [out]       Aggregate experiment markdowns into a single summary (lzo_gpu only)
  plots <summary.csv>   Generate plots from the provided summary CSV (uses tools/generate_plots.py)
  archived-list         List archived scripts moved from tools/
  help                  Show this help

Examples:
  # Precompile kernels
  $0 precompile

  # Run parameter scan with default settings
  $0 param-scan

  # Run parameter scan with custom samples directory and 3 repeats
  $0 param-scan -s /path/to/samples -r 3

  # Dry run to preview parameter combinations
  $0 param-scan --dry-run

  # Run with custom compression levels
  LZO_COMP_LEVELS=1k,1l $0 param-scan

  # Parse profile logs
  $0 parse /path/to/param_scans /tmp/summary_profiles.csv

  # Analyze results
  $0 analyze

  # Generate summary report
  $0 summarize

  # Generate plots
  $0 plots exp_results/lzo_gpu/logs/summary.csv

Notes:
- This unified runner handles all lzo_gpu experiment workflows
- For lzo_cpu experiments, use run_lzo_cpu.sh instead
- Legacy scripts were consolidated into tools/archived/ for clarity
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
    # Parse param-scan specific options
    PARAM_SCAN_HELP=0
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -s|--samples)
          shift; SAMPLES_DIR="${1:-}"; shift || true ;;
        -n|--dry-run)
          DRY_RUN=1; shift || true ;;
        -r|--repeats)
          shift; REPEATS="${1:-5}"; shift || true ;;
        -o|--output)
          shift; PARAM_SCAN_OUT_DIR="${1:-}"; shift || true ;;
        -h|--help)
          PARAM_SCAN_HELP=1; shift || true ;;
        *) shift ;;
      esac
    done

    if [ "$PARAM_SCAN_HELP" = "1" ]; then
      cat <<EOF
Param-scan mode: Comprehensive parameter exploration for lzo_gpu

Options:
  -s, --samples DIR   Directory containing sample files (default: $SAMPLES_DIR_DEFAULT)
  -n, --dry-run       Print commands and record planned actions, do not execute
  -r, --repeats N     Number of repeats per configuration (default: 5)
  -o, --output DIR    Output directory for scan results (default: exp_results/lzo_gpu/logs/param_scans)
  -h, --help          Show this help

Environment Variables (comma-separated):
  LZO_COMP_LEVELS     Compression levels to test (default: 1,1k,1l,1o)
  LZO_DEBUG           Enable debug output (set to 1)

Examples:
  # Basic scan with defaults
  $0 param-scan

  # Custom samples and repeats
  $0 param-scan -s /my/samples -r 10

  # Limited parameter space
  LZO_COMP_LEVELS=1k $0 param-scan

  # Dry run to preview combinations
  $0 param-scan --dry-run
EOF
      exit 0
    fi

    mkdir -p "$PARAM_SCAN_OUT_DIR"

    if [ ! -d "$SAMPLES_DIR" ]; then
      echo "Samples directory not found: $SAMPLES_DIR" >&2
      exit 1
    fi

    # Parse environment variables for parameter enumerations
    if [ -n "${LZO_COMP_LEVELS:-}" ]; then
      read -r -a COMP_LEVELS <<< "$(echo "$LZO_COMP_LEVELS" | tr ',' ' ')"
    else
      COMP_LEVELS=(1 1k 1l 1o)
    fi

    # Gather samples
    SAMPLES=()
    while IFS= read -r -d '' f; do
      if [ -f "$f" ]; then
        SAMPLES+=("$f")
      fi
    done < <(find "$SAMPLES_DIR" \( -type f -o -type l \) -print0)

    if [ ${#SAMPLES[@]} -eq 0 ]; then
      echo "No sample files found in $SAMPLES_DIR" >&2
      exit 1
    fi

    # Optional debug flag
    LZO_DEBUG_FLAG=""
    if [ "${LZO_DEBUG:-0}" = "1" ]; then
      LZO_DEBUG_FLAG="--debug"
    fi

    total_runs=0

    echo "Starting param-scan:"
    echo "  Compression levels: ${COMP_LEVELS[*]}"
    echo "  Samples: ${#SAMPLES[@]}"
    echo "  Repeats: $REPEATS"
    echo "  Output: $PARAM_SCAN_OUT_DIR"
    echo "  Dry-run: $DRY_RUN"
    echo ""

    for comp_level in "${COMP_LEVELS[@]}"; do
      for sample in "${SAMPLES[@]}"; do
        relpath="${sample#${SAMPLES_DIR}/}"
        if [ "$relpath" = "$sample" ]; then relpath="$(basename "$sample")"; fi
        rel_sanitized=$(printf "%s" "$relpath" | sed 's/[^A-Za-z0-9._-]/_/g')
        sample_hash=$(printf "%s" "$sample" | sha1sum 2>/dev/null | awk '{print $1}' | cut -c1-8 || echo unknown)
        sname="${rel_sanitized}_${sample_hash}"

        for r in $(seq 1 "$REPEATS"); do
          total_runs=$((total_runs+1))

          cfg_dir="$PARAM_SCAN_OUT_DIR/comp_${comp_level}"
          mkdir -p "$cfg_dir"

          if type lzo_mktemp_dir >/dev/null 2>&1; then
            lzo_mktemp_dir tmp_run_dir || tmp_run_dir=$(mktemp -d /tmp/lzo_gpu_tmp.XXXXXX)
          else
            tmp_run_dir=$(mktemp -d /tmp/lzo_gpu_tmp.XXXXXX)
          fi

          out_lzo="$tmp_run_dir/lzo_out_${sname}_run${r}.lzo"
          logf="$cfg_dir/${sname}_run${r}.log"

          echo "[Run $total_runs] COMP=$comp_level SAMPLE=$sname R=$r -> $logf"
          echo "# COMP=$comp_level SAMPLE=$sname R=$r" > "$logf"
          echo "Compressing: $sample -> $out_lzo" >> "$logf"

          COMP_CMD=("$LZO_BIN" $LZO_DEBUG_FLAG -L "$comp_level" "$sample" -o "$out_lzo")
          DECMD=("$LZO_BIN" -d --verify "$sample" "$out_lzo")

                  if [ "$DRY_RUN" = "1" ]; then
                    printf "# DRY-RUN CMD: (cd \"%s\" && %s)\n" "$WRAPDIR" "${COMP_CMD[*]}" | tee -a "$logf"
                    compress_status=0
                  else
                    (
                      cd "$WRAPDIR"
                      "${COMP_CMD[@]}"
                    ) >> "$logf" 2>&1
                    compress_status=$?
                    timestamp=$(date --iso-8601=seconds 2>/dev/null || date)
                    if [ "$compress_status" -ne 0 ]; then
                      if [ "$compress_status" -gt 128 ]; then
                        sig=$((compress_status-128))
                        signame=$(kill -l "$sig" 2>/dev/null || echo "SIG$sig")
                        printf "[Run %d] COMP failed: signal %d (%s) at %s (rc=%d)\n" "$total_runs" "$sig" "$signame" "$timestamp" "$compress_status" | tee -a "$logf"
                      else
                        printf "[Run %d] COMP failed: exit %d at %s\n" "$total_runs" "$compress_status" "$timestamp" | tee -a "$logf"
                      fi
                    else
                      printf "[Run %d] COMP exit 0 at %s\n" "$total_runs" "$timestamp" >> "$logf"
                    fi
                  fi

                  echo "Decompress+verify" >> "$logf"
                  if [ "$DRY_RUN" = "1" ]; then
                    printf "# DRY-RUN CMD: (cd \"%s\" && %s)\n" "$WRAPDIR" "${DECMD[*]}" | tee -a "$logf"
                    verify_status=0
                  else
                    (
                      cd "$WRAPDIR"
                      "${DECMD[@]}"
                    ) >> "$logf" 2>&1
                    verify_status=$?
                    timestamp=$(date --iso-8601=seconds 2>/dev/null || date)
                    if [ "$verify_status" -ne 0 ]; then
                      if [ "$verify_status" -gt 128 ]; then
                        sig=$((verify_status-128))
                        signame=$(kill -l "$sig" 2>/dev/null || echo "SIG$sig")
                        printf "[Run %d] VERIFY failed: signal %d (%s) at %s (rc=%d)\n" "$total_runs" "$sig" "$signame" "$timestamp" "$verify_status" | tee -a "$logf"
                      else
                        printf "[Run %d] VERIFY failed: exit %d at %s\n" "$total_runs" "$verify_status" "$timestamp" | tee -a "$logf"
                      fi
                    else
                      printf "[Run %d] VERIFY exit 0 at %s\n" "$total_runs" "$timestamp" >> "$logf"
                    fi
                  fi

          if [ $verify_status -eq 0 ]; then
            rm -f "$out_lzo" || true
          else
            mkdir -p "$cfg_dir/artifacts"
            mv "$out_lzo" "$cfg_dir/artifacts/lzo_out_${sname}_run${r}.lzo" 2>/dev/null || true
          fi

          rm -rf "$tmp_run_dir" 2>/dev/null || true
        done
      done
    done

    echo "Full param scan finished. Total runs: $total_runs. Logs under $PARAM_SCAN_OUT_DIR"
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
    echo "Summary written to $OUT"
    ;;

  plots)
    CSV=${1:-"$LZO_GPU_LOGS/summary.csv"}
    if [ ! -f "$CSV" ]; then
      echo "Summary CSV not found: $CSV" >&2
      echo "Run '$0 analyze' first to produce $CSV"
      exit 1
    fi
    python3 "$TOOLS/generate_plots.py" "$CSV"
    ;;

  archived-list)
    ls -1 "$ARCHIVE" 2>/dev/null || echo "No archived scripts found"
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
