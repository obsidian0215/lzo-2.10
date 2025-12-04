#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARAM_SCAN="$ROOT/tools/param_scan.sh"
# Default OUT_DIR is inside lzo_gpu/exp_results to keep all logs and results
# under the lzo_gpu tree (as requested). Allow overriding by environment var.
OUT_DIR="${OUT_DIR:-$ROOT/lzo_gpu/exp_results/logs}"
SAMPLES_DIR="/root/samples"

if [ ! -x "$PARAM_SCAN" ]; then
  echo "param_scan not found or not executable: $PARAM_SCAN" >&2
  exit 1
fi

echo "Cleaning previous analysis outputs and plots under $OUT_DIR"
if [ -d "$OUT_DIR" ]; then
  mv "$OUT_DIR" "${OUT_DIR}.bak.$(date +%s)" || true
fi
mkdir -p "$OUT_DIR"
export OUT_DIR

REPEATS=3
COMP_LEVELS="1,1k,1l,1o"
WG_SIZES="1,8,auto"
BLOCK_SIZES="32,64,256"
MT_THREADS=2

export REPEATS
export LZO_COMP_LEVELS="$COMP_LEVELS"
export LZO_WG_SIZE="$WG_SIZES"
export LZO_FIXED_BLOCK_SIZES="$BLOCK_SIZES"
export LZO_MT_IO_THREADS="$MT_THREADS"

echo "Starting daemon to support daemon-mode tests (if present)"
if [ -x "$ROOT/lzo_gpu/lzo_gpu_daemon" ]; then
  if ! pgrep -f lzo_gpu_daemon >/dev/null 2>&1; then
    # Redirect daemon stdout to the OUT_DIR so logs remain inside lzo_gpu/exp_results
    (cd "$ROOT/lzo_gpu" && nohup ./lzo_gpu_daemon > "$OUT_DIR/lzo_gpu_daemon.stdout.log" 2>&1 &) || true
    sleep 1
  else
    echo "Daemon already running"
  fi
else
  echo "Daemon binary missing - daemon tests will be skipped unless lzo_gpu_daemon is present"
fi

echo "Running zero+mt (zero copy + multi-thread IO) across runners"
export LZO_STANDARD_COPY_MODES="0"
export LZO_MT_IO_MODES="1"
unset LZO_ASYNC_UPLOAD
export LZO_RUNNERS="standalone,daemon"
LOGFILE="$OUT_DIR/param_scan_zero_mt_$(date +%s).log"
echo "Logging param_scan output to $LOGFILE"
# Run in foreground so analysis runs after param_scan completes; redirect all output to OUT_DIR
bash "$PARAM_SCAN" -s "$SAMPLES_DIR" >> "$LOGFILE" 2>&1 || true

echo "Running std_async (standard copy + async upload) across runners"
export LZO_STANDARD_COPY_MODES="1"
export LZO_MT_IO_MODES="0"
export LZO_ASYNC_UPLOAD="1"
export LZO_RUNNERS="standalone,daemon"
LOGFILE="$OUT_DIR/param_scan_std_async_$(date +%s).log"
echo "Logging param_scan output to $LOGFILE"
bash "$PARAM_SCAN" -s "$SAMPLES_DIR" >> "$LOGFILE" 2>&1 || true

echo "Completed runs for requested modes. Now run analysis & plotting"
python3 "$ROOT/tools/analysis.py" analyze -i "$OUT_DIR/param_scans" -o "$OUT_DIR/summary.csv" || true
python3 "$ROOT/tools/plot_gpu_analysis.py" -i "$OUT_DIR/analysis_summary.csv" || true

echo "Custom scan finished. Logs under $OUT_DIR/param_scans, analysis under $OUT_DIR"
