#!/usr/bin/env bash
# Unified LZO CPU benchmark runner
# Supports three modes:
#   single - Run single configuration (original run_lzo_cpu.sh)
#   batch  - Run batch tests with iterations (original run_full_test.sh)  
#   sweep  - Run parameter sweep experiments (original run_full_experiments.sh)

set -euo pipefail
shopt -s nullglob

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$WORKDIR/lzo_cpu/lzo_cpu"
SAMPLES_DIR="/root/samples"
OUT_DIR="$WORKDIR/exp_results/lzo_cpu"
TMP_DIR="/tmp"

# Default mode
MODE="single"

# Common defaults
ALGS=(1 1k 1l 1o)
THREADS=(1 2 4)
ITER=1
SAMPLES_LIMIT=0

# Batch mode defaults
BATCH_ITERATIONS=3
BATCH_CPU_CORES="0-7"

# Sweep mode defaults
MONITOR_INTERVAL=10
MAX_COMBINATIONS_DEFAULT=64
MAX_COMBINATIONS=${MAX_COMBINATIONS:-$MAX_COMBINATIONS_DEFAULT}
GOVS=${GOVS:-"powersave"}
NO_TURBOS=${NO_TURBOS:-"1"}
MAX_KHZS=${MAX_KHZS:-"0,80%,60%,40%,20%"}
PINS=${PINS:-""}
DRY_RUN=0
NO_ARCHIVE=${NO_ARCHIVE:-1}
GOVS_PROVIDED_ENV=0
GOVS_PROVIDED_OPT=0

# Check if GOVS was provided in environment
if [ "${GOVS+set}" = "set" ]; then
  GOVS_PROVIDED_ENV=1
fi

usage() {
  cat <<EOF
Usage: $0 [mode] [options]

Modes:
  single    Run single benchmark configuration (default)
  batch     Run batch tests with multiple iterations
  sweep     Run full parameter sweep experiments

Common Options:
  -s DIR    samples directory (default: /root/samples)
  -a LIST   comma-separated alg list (default: 1,1k,1l,1o)
  -t LIST   comma-separated thread counts (default: 1,2,4)
  -n N      limit to first N samples (default: all)
  -o DIR    output results directory (default: exp_results/lzo_cpu)
  -h        show this help

Single Mode Options:
  -i N      iterations per config (default: 1)
  -c LIST   cpu cores to pin (taskset list, e.g. 0-3)
  -g GOV    try to set cpu scaling governor (performance|powersave)
  -T 0|1    whether turbo/no_turbo should be set (1 means set no_turbo=1)
  -M KHz    target scaling_max_freq in kHz

Batch Mode Options:
  -I N      iterations per config (default: 3)
  -C LIST   cpu cores to pin (default: 0-7)

Sweep Mode Options:
  -G <govs>        Comma-separated governors to test when max_khz != 0 (default: ${GOVS})
  -N <no_turbos>   Comma-separated no_turbo values (0 or 1) (default: ${NO_TURBOS})
  -K <max_khz>     Comma-separated max_khz list (0, absolute kHz, or percent) (default: ${MAX_KHZS})
  -P <pins>        Comma-separated CPU pin/affinity strings (default: ${PINS})
  -L <limit>       Maximum allowed combinations (default: ${MAX_COMBINATIONS})
  --dry-run        Print planned combinations and exit (no system changes)
  --no-archive     Do not create per-run tar.gz archives (default: no archive)

Examples:
  # Single mode: test one configuration
  $0 single -a 1k -t 4 -i 3

  # Batch mode: comprehensive test
  $0 batch -I 5 -a "1,1k,1l,1o" -t "1,2,4,8"

  # Sweep mode: parameter exploration
  $0 sweep -G "performance,powersave" -K "0,80%,60%" -N "0,1"

  # Dry run to preview sweep combinations
  $0 sweep --dry-run -K "0,75%,50%"
EOF
}

# Parse mode
if [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; then
  MODE="$1"
  shift
fi

# Validate mode
case "$MODE" in
  single|batch|sweep) ;;
  *) echo "Error: Invalid mode '$MODE'. Use: single, batch, or sweep" >&2; usage; exit 1 ;;
esac

# Support long flags
_args=()
for _a in "$@"; do
  case "$_a" in
    --dry-run) DRY_RUN=1 ;;
    --no-archive) NO_ARCHIVE=1 ;;
    *) _args+=("$_a") ;;
  esac
done
set -- "${_args[@]:-}"

# Parse options based on mode
while getopts "s:a:t:n:i:o:c:g:T:M:I:C:G:N:K:P:L:h" opt; do
  case "$opt" in
    # Common options
    s) SAMPLES_DIR="$OPTARG" ;;
    a) IFS=',' read -r -a ALGS <<< "$OPTARG" ;;
    t) IFS=',' read -r -a THREADS <<< "$OPTARG" ;;
    n) SAMPLES_LIMIT="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    # Single mode options
    i) ITER="$OPTARG" ;;
    c) CPU_CORES="$OPTARG" ;;
    g) GOV="$OPTARG" ;;
    T) NO_TURBO_FLAG="$OPTARG" ;;
    M) MAX_KHZ="$OPTARG" ;;
    # Batch mode options
    I) BATCH_ITERATIONS="$OPTARG" ;;
    C) BATCH_CPU_CORES="$OPTARG" ;;
    # Sweep mode options
    G) GOVS="$OPTARG"; GOVS_PROVIDED_OPT=1 ;;
    N) NO_TURBOS="$OPTARG" ;;
    K) MAX_KHZS="$OPTARG" ;;
    P) PINS="$OPTARG" ;;
    L) MAX_COMBINATIONS="$OPTARG" ;;
  esac
done

# Initialize output directory
mkdir -p "$OUT_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)

#############################################################################
# SINGLE MODE - Original run_lzo_cpu.sh functionality
#############################################################################
run_single_mode() {
  CSV="$OUT_DIR/summary.csv"
  LOG_DIR="$OUT_DIR/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/run_$TS.log"
  LOG_ROOT_PARAM="$LOG_DIR/param_scans"
  mkdir -p "$LOG_ROOT_PARAM"

  # Detect flock for atomic CSV writes
  if command -v flock >/dev/null 2>&1; then
    HAVE_FLOCK=1
  else
    HAVE_FLOCK=0
  fi

  set_governor() {
    if [ -z "${GOV:-}" ]; then return 0; fi
    if [ $(id -u) -ne 0 ]; then
      echo "skipping governor set; need root" | tee -a "$LOG_FILE"
      return 0
    fi
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
      govf="$cpu/cpufreq/scaling_governor"
      if [ -w "$govf" ]; then
        echo "$GOV" > "$govf" || true
      fi
    done
  }

  set_governor

  echo "run started: $(date -u)" | tee -a "$LOG_FILE"
  echo "EFFECTIVE_ALGS=${ALGS[*]}" | tee -a "$LOG_FILE"
  echo "EFFECTIVE_THREADS=${THREADS[*]}" | tee -a "$LOG_FILE"
  if [ -n "${GOV:-}" ]; then
    echo "EFFECTIVE_GOV=${GOV}" | tee -a "$LOG_FILE"
  fi
  echo "EFFECTIVE_NO_TURBO=${NO_TURBO_FLAG:-}" | tee -a "$LOG_FILE"
  echo "EFFECTIVE_MAX_KHZ=${MAX_KHZ:-}" | tee -a "$LOG_FILE"
  echo "EFFECTIVE_CPU_CORES=${CPU_CORES:-}" | tee -a "$LOG_FILE"

  # Prepare CSV header
  if [ ! -f "$CSV" ]; then
    if [ -n "${GOV:-}" ]; then
      header='sample,alg,threads,iter,cpu_gov,cpu_no_turbo,cpu_max_khz,cpu_pin,comp_bytes,ratio_pct,comp_ms,comp_MB_s,decomp_ms,decomp_MB_s'
    else
      header='sample,alg,threads,iter,cpu_no_turbo,cpu_max_khz,cpu_pin,comp_bytes,ratio_pct,comp_ms,comp_MB_s,decomp_ms,decomp_MB_s'
    fi
    if [ "$HAVE_FLOCK" -eq 1 ]; then
      (
        flock 9
        if [ ! -s "$CSV" ]; then
          printf "%s\n" "$header" >&9
        fi
      ) 9>>"$CSV"
    else
      printf "%s\n" "$header" > "$CSV"
    fi
  fi

  pin_cmd() {
    if [ -n "${CPU_CORES:-}" ]; then
      echo "taskset -c $CPU_CORES"
    else
      echo ""
    fi
  }

  SAMPLES=("$SAMPLES_DIR"/*)
  if [ ${#SAMPLES[@]} -eq 0 ]; then
    echo "no samples found in $SAMPLES_DIR" | tee -a "$LOG_FILE"
    exit 1
  fi
  if [ "$SAMPLES_LIMIT" -gt 0 ]; then
    SAMPLES=("${SAMPLES[@]:0:$SAMPLES_LIMIT}")
  fi

  for sample in "${SAMPLES[@]}"; do
    sample_base=$(basename "$sample")
    for alg in "${ALGS[@]}"; do
      for threads in "${THREADS[@]}"; do
        for ((it=1; it<=ITER; it++)); do
          outtmp="$TMP_DIR/${sample_base}.${alg}.t${threads}.lzo"
          RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
          run_dir="$LOG_ROOT_PARAM/$sample_base/$alg/threads_${threads}/iter_${it}_$RUN_TS"
          mkdir -p "$run_dir"
          comp_out_file="$run_dir/compress.log"
          verify_out_file="$run_dir/verify.log"
          cmd_prefix=$(pin_cmd)
          echo "[$(date -u)] sample=$sample_base alg=$alg threads=$threads iter=$it" | tee -a "$LOG_FILE"
          
          # Compress
          if [ -n "$cmd_prefix" ]; then
            eval "$cmd_prefix \"$BIN\" \"$sample\" \"$outtmp\" -L $alg -t $threads" > "$comp_out_file" 2>&1 || true
          else
            "$BIN" "$sample" "$outtmp" -L "$alg" -t $threads > "$comp_out_file" 2>&1 || true
          fi
          comp_out=$(cat "$comp_out_file")
          comp_bytes=$(grep -oE -- '-> [0-9]+ bytes' "$comp_out_file" | head -n1 | grep -oE '[0-9]+' || echo 0)
          comp_ratio=$(grep -oE -- '\([0-9]+(\.[0-9]+)?%\)' "$comp_out_file" | head -n1 | tr -d '()% ' || echo 0)
          comp_ms=$(grep -oE -- 'time=[0-9]+\.[0-9]+' "$comp_out_file" | head -n1 | sed 's/time=//' || echo 0)
          comp_MB_s=$(grep -oE -- '[0-9]+\.[0-9]+ MB/s' "$comp_out_file" | head -n1 | awk '{print $1}' || echo 0)

          # Decompress verify
          if [ -n "$cmd_prefix" ]; then
            eval "$cmd_prefix \"$BIN\" -d --verify \"$outtmp\" -t $threads" > "$verify_out_file" 2>&1 || true
          else
            "$BIN" -d --verify "$outtmp" -t $threads > "$verify_out_file" 2>&1 || true
          fi
          decomp_out=$(cat "$verify_out_file")
          decomp_ms=$(grep -oE -- 'time=[0-9]+\.[0-9]+' "$verify_out_file" | tail -n1 | sed 's/time=//' || echo 0)
          decomp_MB_s=$(grep -oE -- '[0-9]+\.[0-9]+ MB/s' "$verify_out_file" | tail -n1 | awk '{print $1}' || echo 0)

          # Determine status
          if echo "$decomp_out" | grep -q "Verify OK" || echo "$decomp_out" | grep -q "Verify decompress OK"; then
            status=OK
            rm -f "$outtmp" || true
          else
            status=FAIL
          fi

          # Write CSV
          if [ -n "${GOV:-}" ]; then
            csv_line=$(printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
              "$sample_base" "$alg" "$threads" "$it" "${GOV}" "${NO_TURBO_FLAG:-}" "${MAX_KHZ:-}" "${CPU_CORES:-}" \
              "$comp_bytes" "$comp_ratio" "$comp_ms" "$comp_MB_s" "$decomp_ms" "$decomp_MB_s")
          else
            csv_line=$(printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
              "$sample_base" "$alg" "$threads" "$it" "${NO_TURBO_FLAG:-}" "${MAX_KHZ:-}" "${CPU_CORES:-}" \
              "$comp_bytes" "$comp_ratio" "$comp_ms" "$comp_MB_s" "$decomp_ms" "$decomp_MB_s")
          fi
          if [ "$HAVE_FLOCK" -eq 1 ]; then
            ( flock 9; printf "%s\n" "$csv_line" >&9 ) 9>>"$CSV"
          else
            printf "%s\n" "$csv_line" >> "$CSV"
          fi

          sleep 0.1
        done
      done
    done
  done

  echo "run finished: $(date -u)" | tee -a "$LOG_FILE"
  echo "results: $CSV"
}

#############################################################################
# BATCH MODE - Original run_full_test.sh functionality
#############################################################################
run_batch_mode() {
  OUT_DIR="$OUT_DIR/full_test"
  mkdir -p "$OUT_DIR"
  CSV_FILE="$OUT_DIR/full_test_results.csv"
  
  # Check binary exists
  if [[ ! -x "$BIN" ]]; then
    echo "Error: Binary not found or not executable: $BIN" >&2
    exit 1
  fi

  # CSV header
  if [[ ! -f "$CSV_FILE" ]]; then
    echo "sample,alg,threads,iter,orig_bytes,comp_bytes,ratio_pct,comp_ms,comp_MB_s,decomp_ms,decomp_MB_s" > "$CSV_FILE"
  fi

  # Get list of samples
  mapfile -t SAMPLES < <(ls -1 "$SAMPLES_DIR"/*.img 2>/dev/null || true)

  if [[ ${#SAMPLES[@]} -eq 0 ]]; then
    echo "Error: No .img files found in $SAMPLES_DIR" >&2
    exit 1
  fi

  if [ "$SAMPLES_LIMIT" -gt 0 ]; then
    SAMPLES=("${SAMPLES[@]:0:$SAMPLES_LIMIT}")
  fi

  echo "=== LZO CPU Full Test ==="
  echo "Samples: ${#SAMPLES[@]}"
  echo "Algorithms: ${ALGS[*]}"
  echo "Threads: ${THREADS[*]}"
  echo "Iterations: $BATCH_ITERATIONS"
  echo "CPU cores: $BATCH_CPU_CORES"
  echo "Output: $CSV_FILE"
  echo ""

  total_tests=$((${#SAMPLES[@]} * ${#ALGS[@]} * ${#THREADS[@]} * BATCH_ITERATIONS))
  current_test=0
  start_time=$(date +%s)

  for sample_path in "${SAMPLES[@]}"; do
    sample_name=$(basename "$sample_path")
    
    for alg in "${ALGS[@]}"; do
      for threads in "${THREADS[@]}"; do
        for iter in $(seq 1 $BATCH_ITERATIONS); do
          current_test=$((current_test + 1))
          progress=$((current_test * 100 / total_tests))
          
          echo -ne "\r[Progress: $progress% ($current_test/$total_tests)] Testing: $sample_name alg=$alg threads=$threads iter=$iter"
          
          # Run compression test with taskset
          output=$(taskset -c "$BATCH_CPU_CORES" "$BIN" -L "$alg" -t "$threads" --benchmark "$sample_path" 2>&1 || true)
          
          # Parse output
          orig_bytes=$(echo "$output" | grep -oP 'Compressed \K[0-9]+' | head -1 || echo "0")
          comp_bytes=$(echo "$output" | grep -oP 'Compressed [0-9]+ bytes -> \K[0-9]+' | head -1 || echo "0")
          ratio=$(echo "$output" | grep -oP '\(\K[0-9.]+(?=%\))' | head -1 || echo "0")
          
          # Multi Compress/Decompress lines
          comp_time=$(echo "$output" | grep 'Multi.*Compress' | grep -oP '\K[0-9.]+(?= ms)' || echo "0")
          comp_mbps=$(echo "$output" | grep 'Multi.*Compress' | grep -oP '\(.*\K[0-9.]+(?= MB/s)' || echo "0")
          decomp_time=$(echo "$output" | grep 'Multi.*Decompress' | grep -oP '\K[0-9.]+(?= ms)' || echo "0")
          decomp_mbps=$(echo "$output" | grep 'Multi.*Decompress' | grep -oP '\K[0-9.]+(?= MB/s)' || echo "0")
          
          # Write to CSV (atomic write with flock if available)
          if command -v flock &> /dev/null; then
            (
              flock -x 200
              echo "$sample_name,$alg,$threads,$iter,$orig_bytes,$comp_bytes,$ratio,$comp_time,$comp_mbps,$decomp_time,$decomp_mbps" >> "$CSV_FILE"
            ) 200>"$CSV_FILE.lock"
          else
            echo "$sample_name,$alg,$threads,$iter,$orig_bytes,$comp_bytes,$ratio,$comp_time,$comp_mbps,$decomp_time,$decomp_mbps" >> "$CSV_FILE"
          fi
        done
      done
    done
  done

  end_time=$(date +%s)
  elapsed=$((end_time - start_time))

  echo ""
  echo ""
  echo "=== Test Complete ==="
  echo "Total tests: $total_tests"
  echo "Elapsed time: ${elapsed}s"
  echo "Results: $CSV_FILE"
  echo ""
}

#############################################################################
# SWEEP MODE - Original run_full_experiments.sh functionality
#############################################################################
run_sweep_mode() {
  PIDFILE="$OUT_DIR/run_full_experiments.pid"
  LOG="$OUT_DIR/run_full_experiments_$TS.log"

  # Parse sweep-specific parameters
  IFS=',' read -r -a GOV_LIST <<< "$GOVS"
  IFS=',' read -r -a NT_LIST  <<< "$NO_TURBOS"
  IFS=',' read -r -a MK_LIST_RAW  <<< "$MAX_KHZS"
  IFS=',' read -r -a PIN_LIST <<< "$PINS"

  if [ ${#PIN_LIST[@]} -eq 0 ]; then
    PIN_LIST=("")
  fi

  # Convert percent entries to absolute kHz
  cpuinfo_max_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
  base_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/base_frequency 2>/dev/null || echo 0)

  convert_to_khz() {
    local raw=$1
    local no_turbo=$2
    if [ "$raw" = "0" ]; then
      echo 0
      return
    fi
    if [[ "$raw" =~ %$ ]]; then
      pct=${raw%%%}
      if [ "$no_turbo" -eq 1 ]; then
        if [ "$base_khz" -gt 0 ]; then
          echo $(( base_khz * pct / 100 ))
        else
          echo $(( cpuinfo_max_khz * pct / 100 ))
        fi
      else
        echo $(( cpuinfo_max_khz * pct / 100 ))
      fi
    else
      echo "$raw"
    fi
  }

  # Count combinations
  combos=0
  pin_count=${#PIN_LIST[@]}
  gov_count=${#GOV_LIST[@]}
  nt_count=${#NT_LIST[@]}
  for _m in "${MK_LIST_RAW[@]}"; do
    if [ "$_m" = "0" ]; then
      combos=$((combos + pin_count))
    else
      combos=$((combos + gov_count * nt_count * pin_count))
    fi
  done

  if [ "$combos" -gt "$MAX_COMBINATIONS" ] && [ -z "${FORCE_COMBINATIONS:-}" ]; then
    echo "ERROR: configuration combos=$combos exceeds limit=$MAX_COMBINATIONS. Set FORCE_COMBINATIONS=1 to force or increase -L." | tee -a "$LOG"
    exit 1
  fi

  # Dry run mode
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "Planned combos: $combos"
    echo "--- Listing planned combinations ---"
    for max_khz_raw in "${MK_LIST_RAW[@]}"; do
      if [ "$max_khz_raw" = "0" ]; then
        gov_default="performance"
        nt_default=0
        for pin in "${PIN_LIST[@]}"; do
          printf "CFG: gov=%s no_turbo=%s max_khz=%s pin=%s -> extra_args: -g %s%s\n" \
            "$gov_default" "$nt_default" "0" "$pin" "$gov_default" "${pin:+ -c $pin}"
        done
      else
        for gov in "${GOV_LIST[@]}"; do
          for no_turbo in "${NT_LIST[@]}"; do
            for pin in "${PIN_LIST[@]}"; do
              numeric_max_khz=$(convert_to_khz "$max_khz_raw" "$no_turbo")
              include_gov=0
              if [ "$gov" != "powersave" ] || [ "$GOVS_PROVIDED_ENV" -eq 1 ] || [ "$GOVS_PROVIDED_OPT" -eq 1 ]; then
                include_gov=1
              fi

              if [ "$include_gov" -eq 1 ]; then
                out="CFG: gov=$gov no_turbo=$no_turbo max_khz=$numeric_max_khz pin=$pin -> extra_args: -g $gov"
              else
                out="CFG: no_turbo=$no_turbo max_khz=$numeric_max_khz pin=$pin -> extra_args:"
              fi
              [ -n "$pin" ] && out+=" -c $pin"
              out+=" -T $no_turbo -M $numeric_max_khz"
              echo "$out"
            done
          done
        done
      fi
    done
    exit 0
  fi

  # Single instance guard with flock
  LOCKFD=9
  LOCKFILE="$PIDFILE.lock"
  LOCKDIR="$PIDFILE.lockdir"
  if command -v flock >/dev/null 2>&1; then
    exec {LOCKFD}>>"$LOCKFILE" || { echo "cannot open lockfile $LOCKFILE" | tee -a "$LOG"; exit 1; }
    if ! flock -n "$LOCKFD"; then
      curpid=$(cat "$PIDFILE" 2>/dev/null || echo "")
      echo "run_full_experiments already running (pid=${curpid:-unknown}). Exiting." | tee -a "$LOG"
      exit 1
    fi
    echo $$ > "$PIDFILE"
  else
    if mkdir "$LOCKDIR" 2>/dev/null; then
      echo $$ > "$PIDFILE"
    else
      curpid=$(cat "$PIDFILE" 2>/dev/null || echo "")
      echo "run_full_experiments already running (pid=${curpid:-unknown}). Exiting." | tee -a "$LOG"
      exit 1
    fi
  fi

  cleanup_and_exit() {
    cleanup_monitors 2>/dev/null || true
    restore_defaults 2>/dev/null || true
    rm -f "$PIDFILE" || true
    if [ -n "${LOCKFD:-}" ] && [ "$LOCKFD" -ge 0 ] 2>/dev/null; then
      eval "exec ${LOCKFD}>&-" || true
    fi
    [ -d "$LOCKDIR" ] && rmdir "$LOCKDIR" 2>/dev/null || true
  }
  trap 'cleanup_and_exit' EXIT

  log() { echo "[$(date -u)] $*" | tee -a "$LOG"; }

  cleanup_monitors() {
    for f in "$OUT_DIR"/*/monitor.pid; do
      [ -f "$f" ] || continue
      pid=$(cat "$f" 2>/dev/null || echo "")
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "cleanup: stopping monitor pid=$pid (from $f)"
        kill "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
      fi
      rm -f "$f" || true
    done
  }

  restore_defaults() {
    log "restoring defaults: turbo=on, governor=performance, scaling_max=cpuinfo_max"
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo || true
    elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then echo 0 > /sys/devices/system/cpu/cpufreq/boost || true; fi
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
      govf="$cpu/cpufreq/scaling_governor"; maxf_f="$cpu/cpufreq/scaling_max_freq"; info_max="$cpu/cpufreq/cpuinfo_max_freq"
      if [ -f "$info_max" ] && [ -w "$maxf_f" ]; then vmax=$(cat "$info_max" 2>/dev/null || echo ""); [ -n "$vmax" ] && echo "$vmax" > "$maxf_f" || true; fi
      [ -w "$govf" ] && echo performance > "$govf" || true
    done
  }

  set_governor_all() { local gov=$1; for cpu in /sys/devices/system/cpu/cpu[0-9]*; do govf="$cpu/cpufreq/scaling_governor"; [ -w "$govf" ] && echo "$gov" > "$govf" || true; done }
  set_scaling_bounds() { local maxf=$1; local minf=${2:-$maxf}; for cpu in /sys/devices/system/cpu/cpu[0-9]*; do maxf_f="$cpu/cpufreq/scaling_max_freq"; minf_f="$cpu/cpufreq/scaling_min_freq"; [ -w "$maxf_f" ] && echo "$maxf" > "$maxf_f" || true; [ -w "$minf_f" ] && echo "$minf" > "$minf_f" || true; done }
  set_no_turbo() { local v=$1; if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then echo "$v" > /sys/devices/system/cpu/intel_pstate/no_turbo || true; elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then echo "$v" > /sys/devices/system/cpu/cpufreq/boost || true; fi }

  start_monitor() {
    local d=$1
    mkdir -p "$d"
    logfile="$d/cpu_state.json"
    (
      while :; do
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        turbo="N/A"
        if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
          turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo N/A)
        elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
          turbo=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo N/A)
        fi

        printf '{"timestamp":"%s","turbo":"%s","cpus":[' "$ts" "$turbo" > "$logfile"
        first=1
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
          id=$(basename "$cpu")
          govf="$cpu/cpufreq/scaling_governor"
          curf="$cpu/cpufreq/scaling_cur_freq"
          maxf="$cpu/cpufreq/scaling_max_freq"
          minf="$cpu/cpufreq/scaling_min_freq"
          gov=$( [ -f "$govf" ] && cat "$govf" 2>/dev/null || echo N/A )
          cur=$( [ -f "$curf" ] && cat "$curf" 2>/dev/null || echo N/A )
          maxv=$( [ -f "$maxf" ] && cat "$maxf" 2>/dev/null || echo N/A )
          minv=$( [ -f "$minf" ] && cat "$minf" 2>/dev/null || echo N/A )
          if [ $first -eq 0 ]; then
            printf ',' >> "$logfile"
          else
            first=0
          fi
          printf '{"id":"%s","gov":"%s","cur_khz":"%s","max_khz":"%s","min_khz":"%s"}' "$id" "$gov" "$cur" "$maxv" "$minv" >> "$logfile"
        done
        printf ']}' >> "$logfile"
        sleep "$MONITOR_INTERVAL"
      done
    ) &
    echo $! > "$d/monitor.pid"
  }

  stop_monitor() { local d=$1; [ -f "$d/monitor.pid" ] && { pid=$(cat "$d/monitor.pid"); kill "$pid" 2>/dev/null || true; rm -f "$d/monitor.pid"; } }

  run_phase() {
    local name=$1; shift
    local outdir="$OUT_DIR/$name"
    mkdir -p "$outdir"
    
    # Build runner command - call this script in single mode
    cmd=( "$0" "single" -o "$outdir" "$@" )
    printf '%q ' "${cmd[@]}" > "$outdir/run_cmd.txt"
    log "CMD: $(printf '%q ' "${cmd[@]}")"
    start_monitor "$outdir"
    "${cmd[@]}"
    stop_monitor "$outdir"
    if [ "${NO_ARCHIVE:-0}" -eq 0 ]; then
      tar -C "$OUT_DIR" -czf "$OUT_DIR/${name}_results_$TS.tar.gz" "${name}"
    else
      log "skipping archive for ${name} (NO_ARCHIVE=1)"
    fi
  }

  log "full experiment started (combos=$combos)"
  combos_run=0
  for max_khz_raw in "${MK_LIST_RAW[@]}"; do
    if [ "$max_khz_raw" = "0" ]; then
      gov_default="performance"
      nt_default=0
      for pin in "${PIN_LIST[@]}"; do
        log "Applying config (no-limit): gov=$gov_default no_turbo=$nt_default max_khz=$max_khz_raw pin=$pin"
        set_no_turbo "$nt_default" || true
        set_governor_all "$gov_default" || true
        maxf=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
        [ "$maxf" -gt 0 ] && set_scaling_bounds "$maxf" "$maxf" || true
        outname="gov_${gov_default}_nt${nt_default}_max${max_khz_raw}_pin${pin// /_}"
        extra_args=( -g "$gov_default" )
        [ -n "$pin" ] && extra_args+=( -c "$pin" )
        extra_args+=( -T "$nt_default" )
        extra_args+=( -M "0" )
        run_phase "$outname" "${extra_args[@]}"
        combos_run=$((combos_run+1))
      done
    else
      for gov in "${GOV_LIST[@]}"; do
        for no_turbo in "${NT_LIST[@]}"; do
          for pin in "${PIN_LIST[@]}"; do
            max_khz=$(convert_to_khz "$max_khz_raw" "$no_turbo")
            log "Applying config (limited): gov=$gov no_turbo=$no_turbo max_khz=$max_khz pin=$pin"
            if [ "$no_turbo" = "1" ]; then
              set_no_turbo 1 || true
            else
              set_no_turbo 0 || true
            fi
            set_governor_all "$gov" || true
            max_khz=$(convert_to_khz "$max_khz_raw" "$no_turbo")
            set_scaling_bounds "$max_khz" "$max_khz" || true
            include_gov=0
            if [ "$gov" != "powersave" ] || [ "$GOVS_PROVIDED_ENV" -eq 1 ] || [ "$GOVS_PROVIDED_OPT" -eq 1 ]; then
              include_gov=1
            fi

            if [ "$include_gov" -eq 1 ]; then
              outname="gov_${gov}_nt${no_turbo}_max${max_khz}_pin${pin// /_}"
              extra_args=( -g "$gov" )
            else
              outname="nt${no_turbo}_max${max_khz}_pin${pin// /_}"
              extra_args=()
            fi
            [ -n "$pin" ] && extra_args+=( -c "$pin" )
            [ -n "$no_turbo" ] && extra_args+=( -T "$no_turbo" )
            extra_args+=( -M "$max_khz" )
            run_phase "$outname" "${extra_args[@]}"
            combos_run=$((combos_run+1))
          done
        done
      done
    fi
  done
  restore_defaults || true
  cleanup_monitors || true
  log "full experiment finished (ran $combos_run combinations)"
}

#############################################################################
# MAIN ENTRY POINT
#############################################################################
case "$MODE" in
  single) run_single_mode ;;
  batch)  run_batch_mode ;;
  sweep)  run_sweep_mode ;;
esac
