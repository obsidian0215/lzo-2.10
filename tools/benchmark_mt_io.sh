#!/usr/bin/env bash
# tools/benchmark_mt_io.sh
# Run lzo_gpu benchmark across modes (zero, zero+MT, std, std+MT)
# Supports passing files and/or directories (optionally recursive).
# Outputs CSV summary with columns: file,size_bytes,mode,run,ms,output_size_bytes,rc,decomp_ms,decomp_rc,decomp_ok

set -euo pipefail

# defaults
OUT_CSV=benchmark_mt_io_results.csv
MT_THREADS_DEFAULT=4
RUNS_PER=3
RECURSIVE=0
MIN_SIZE=0
LZO_BIN=./lzo_gpu
TMPDIR=$(mktemp -d /tmp/lzo-bench.XXXX)
KEEP_DIR=""
cleanup() {
  if [ -n "$KEEP_DIR" ]; then
    mkdir -p "$KEEP_DIR"
    # copy all useful artifacts into KEEP_DIR
    cp -a "$TMPDIR"/* "$KEEP_DIR"/ 2>/dev/null || true
  fi
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

usage() {
  cat <<EOF
Usage: $0 [options] <file|dir>...

Options:
  -o FILE      output CSV file (default: ${OUT_CSV})
  -t THREADS   mt threads when MT enabled (default: ${MT_THREADS_DEFAULT})
  -n RUNS      runs per mode (default: ${RUNS_PER})
  -r           recursive when directories passed
  -m MIN_BYTES minimum file size to include (default: ${MIN_SIZE})
  -b BIN       path to lzo_gpu binary (default: ${LZO_BIN})
  -k DIR       keep run logs and artifacts in DIR (persist)
  -K           keep run logs and artifacts in ./lzo-bench-logs.TIMESTAMP
  -h           show this help

Modes tested (per file):
  zero          : LZO_STANDARD_COPY=0 LZO_MT_IO=0
  zero+mt       : LZO_STANDARD_COPY=0 LZO_MT_IO=1
  std           : LZO_STANDARD_COPY=1 LZO_MT_IO=0
  std+mt        : LZO_STANDARD_COPY=1 LZO_MT_IO=1

CSV columns: file,size_bytes,mode,run,ms,output_size_bytes,rc,decomp_ms,decomp_rc,decomp_ok

EOF
}

if ! command -v "$LZO_BIN" >/dev/null 2>&1; then
  # Maybe compiled binaries are at the repo root
  if [ -x "$(pwd)/lzo_gpu" ]; then
    LZO_BIN="$(pwd)/lzo_gpu"
  else
    echo "ERROR: lzo_gpu binary not found at ${LZO_BIN} or ./lzo_gpu" >&2
    exit 1
  fi
fi

while getopts ":o:t:n:rm:b:hk:hK" opt; do
  case ${opt} in
    o) OUT_CSV=${OPTARG} ;;
    t) MT_THREADS_DEFAULT=${OPTARG} ;;
    n) RUNS_PER=${OPTARG} ;;
    r) RECURSIVE=1 ;;
    m) MIN_SIZE=${OPTARG} ;;
    b) LZO_BIN=${OPTARG} ;;
    k) KEEP_DIR=${OPTARG} ;;   # enable keep logs to specified dir
    K) KEEP_DIR="./lzo-bench-logs.$(date +%Y%m%d-%H%M%S)" ;; # auto dir
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

if [ $# -lt 1 ]; then
  echo "ERROR: supply one or more files or directories to test" >&2
  usage
  exit 1
fi

# Collect files
FILES=()
for P in "$@"; do
  if [ -f "$P" ]; then
    FILES+=("$P")
  elif [ -d "$P" ]; then
    if [ "$RECURSIVE" -eq 1 ]; then
      while IFS= read -r -d '' f; do
        FILES+=("$f")
      done < <(find "$P" -type f -size +${MIN_SIZE}c -print0)
    else
      while IFS= read -r -d '' f; do
        FILES+=("$f")
      done < <(find "$P" -maxdepth 1 -type f -size +${MIN_SIZE}c -print0)
    fi
  else
    echo "Warning: $P is not a file or directory — skipping" >&2
  fi
done

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files found to test (check min-size and inputs)" >&2
  exit 1
fi

# write CSV header (added decompression measurement + verification columns)
# NOTE: times are milliseconds (ms)
# Columns: file,size_bytes,mode,run,ms,read_ms,upload_ms,io_ms,output_size_bytes,rc,decomp_ms,decomp_read_ms,decomp_upload_ms,decomp_io_ms,decomp_rc,decomp_ok
printf 'file,size_bytes,mode,run,ms,read_ms,upload_ms,io_ms,output_size_bytes,rc,decomp_ms,decomp_read_ms,decomp_upload_ms,decomp_io_ms,decomp_rc,decomp_ok\n' > "$OUT_CSV"

modes=("zero" "zero+mt" "std" "std+mt")

for f in "${FILES[@]}"; do
  size=$(stat -c %s "$f")
  # unique id per input file path to prevent basename collisions across different directories
  file_id=$(printf '%s' "$f" | sha256sum | awk '{print $1}')
  safe_name=$(basename "$f" | tr '/\\' '_')
  echo "\n=== File: $f  size=${size} bytes ==="
  for mode in "${modes[@]}"; do
    echo "  mode: $mode"
    for run in $(seq 1 $RUNS_PER); do
      # create a tmp output file to avoid clashing
      out=$(mktemp "$TMPDIR/out.XXXX.lzo")
      envvars=()
      envvars+=("LC_ALL=C")
      case "$mode" in
        zero)
          envvars+=("LZO_STANDARD_COPY=0")
          envvars+=("LZO_MT_IO=0")
          ;;
        "zero+mt")
          envvars+=("LZO_STANDARD_COPY=0")
          envvars+=("LZO_MT_IO=1")
          envvars+=("LZO_MT_IO_THREADS=${MT_THREADS_DEFAULT}")
          ;;
        std)
          envvars+=("LZO_STANDARD_COPY=1")
          envvars+=("LZO_MT_IO=0")
          ;;
        "std+mt")
          envvars+=("LZO_STANDARD_COPY=1")
          envvars+=("LZO_MT_IO=1")
          envvars+=("LZO_MT_IO_THREADS=${MT_THREADS_DEFAULT}")
          ;;
      esac

      # run and measure wall clock using /usr/bin/time
      # use -f to only print elapsed seconds
      # run with properly prefixed env vars and /usr/bin/time to capture wall-clock
      echo "    run #$run: env ${envvars[*]} $LZO_BIN '$f' -o '$out'"
      run_log="$TMPDIR/run.${mode}.run${run}.log"
      # we use env to set environment variables for the command and /usr/bin/time to measure
      # capture stdout+stderr to run_log for debugging
      ( env "${envvars[@]}" /usr/bin/time -f %e -o "$TMPDIR/time.out" "$LZO_BIN" "$f" -o "$out" ) >"$run_log" 2>&1 || true
      rc=$?
      # parse run_log for IO split (File Read / Data Upload) if present
      read_ms=""
      upload_ms=""
      io_ms=""
      if [ -f "$TMPDIR/time.out" ]; then
        sec=$(cat "$TMPDIR/time.out" | tr -d '\n')
              # extract breakdown values from the run_log if printed (e.g. '1. File Read' and '6. Data Upload')
              if [ -f "$run_log" ]; then
                # try to extract the 'File Read' and 'Data Upload' fields (numbers are in ms)
                read_val=$(grep -E "File Read|read input" "$run_log" | tail -n1 | awk -F":" '{print $2}' | sed -E 's/[^0-9.]*([0-9.]+(\.[0-9]+)?).*/\1/') || true
                upload_val=$(grep -E "Data Upload|create\+upload|Data Upload" "$run_log" | tail -n1 | awk -F":" '{print $2}' | sed -E 's/[^0-9.]*([0-9.]+(\.[0-9]+)?).*/\1/') || true
                if [ -n "$read_val" ]; then read_ms=$(printf "%.3f" "$read_val") ; fi
                if [ -n "$upload_val" ]; then upload_ms=$(printf "%.3f" "$upload_val") ; fi
                if [ -n "$read_ms" ] && [ -n "$upload_ms" ]; then io_ms=$(awk -v a="$read_ms" -v b="$upload_ms" 'BEGIN{printf "%.3f", a + b}') ; fi
              fi
        # convert seconds to milliseconds (float 3 decimals)
        if [ -n "$sec" ]; then
          seconds=$(awk -v s="$sec" 'BEGIN{printf "%.3f", s*1000}')
        else
          seconds=""
        fi
      else
        seconds=""
      fi
      if [ -f "$out" ]; then
        outsz=$(stat -c %s "$out")
      else
        outsz=0
      fi

      # We'll append decompression info later to the same CSV row. Start by printing compression results
      printf '%s,%s,%s,%d,%s,%s,%s,%s,%s,%d' "${f}" "$size" "$mode" "$run" "$seconds" "$read_ms" "$upload_ms" "$io_ms" "$outsz" "$rc" >> "$OUT_CSV"

      # If this is the first run for the mode keep a copy for later cross-mode comparison
      if [ "$run" -eq 1 ] && [ "$rc" -eq 0 ] && [ -f "$out" ]; then
        saved_out="$TMPDIR/${safe_name}.${file_id}.${mode}.lzo"
        cp -f "$out" "$saved_out" || true
      fi

      # Decompress the compressed output using the same environment and measure wall-clock time
      decomp_out="$TMPDIR/decomp.${mode}.run${run}.tmp"
      decomp_seconds=""
      decomp_read=""
      decomp_upload=""
      decomp_io=""
      decomp_rc=0
      decomp_ok=0
      if [ -f "$out" ]; then
        echo "      decompress run #$run (mode=$mode): $LZO_BIN -d -o $decomp_out"
        decomp_log="$TMPDIR/run.${mode}.run${run}.decomp.log"
        ( env "${envvars[@]}" /usr/bin/time -f %e -o "$TMPDIR/decomp.time" "$LZO_BIN" -d "$out" -o "$decomp_out" ) >"$decomp_log" 2>&1 || true
        decomp_rc=$?
        if [ -f "$TMPDIR/decomp.time" ]; then
          dsec=$(cat "$TMPDIR/decomp.time" | tr -d '\n')
          if [ -n "$dsec" ]; then
            decomp_seconds=$(awk -v s="$dsec" 'BEGIN{printf "%.3f", s*1000}')
          else
            decomp_seconds=""
          fi
        fi

        # parse decomp_log for File Read / Data Upload
        if [ -f "$decomp_log" ]; then
          d_read=$(grep -E "File Read|read input" "$decomp_log" | tail -n1 | awk -F":" '{print $2}' | sed -E 's/[^0-9.]*([0-9.]+(\.[0-9]+)?).*/\1/') || true
          d_upload=$(grep -E "Data Upload|create\+upload|Data Upload" "$decomp_log" | tail -n1 | awk -F":" '{print $2}' | sed -E 's/[^0-9.]*([0-9.]+(\.[0-9]+)?).*/\1/') || true
          if [ -n "$d_read" ]; then decomp_read=$(printf "%.3f" "$d_read"); fi
          if [ -n "$d_upload" ]; then decomp_upload=$(printf "%.3f" "$d_upload"); fi
          if [ -n "$decomp_read" ] && [ -n "$decomp_upload" ]; then decomp_io=$(awk -v a="$decomp_read" -v b="$decomp_upload" 'BEGIN{printf "%.3f", a + b}'); fi
        fi

        if [ -f "$decomp_out" ]; then
          if cmp -s "$f" "$decomp_out"; then
            decomp_ok=1
          else
            decomp_ok=0
          fi
        else
          decomp_ok=0
        fi
      else
        decomp_rc=127
        decomp_ok=0
      fi

      # If compression or decompression failed, print the small log snippets for debugging
      if [ "$rc" -ne 0 ]; then
        echo "    ERROR: compression failed -> mode=$mode run=$run rc=$rc log=$run_log"
        echo "    --- compression log (last 80 lines) ---"
        tail -n 80 "$run_log" || true
      fi
      if [ "$decomp_rc" -ne 0 ]; then
        echo "    ERROR: decompression failed -> mode=$mode run=$run rc=$decomp_rc log=$decomp_log"
        echo "    --- decompression log (last 80 lines) ---"
        tail -n 80 "$decomp_log" || true
      fi
      if [ "$decomp_ok" -ne 1 ] && [ "$decomp_rc" -eq 0 ]; then
        echo "    ERROR: decompressed content mismatch -> mode=$mode run=$run (compressed=$out)"
        echo "    You can inspect logs: $run_log and $decomp_log"
      fi

      # append decompression fields and newline (ms units) — include read/upload/io breakdown if available
      printf ',%s,%s,%s,%s,%d,%d\n' "$decomp_seconds" "$decomp_read" "$decomp_upload" "$decomp_io" "$decomp_rc" "$decomp_ok" >> "$OUT_CSV"

      # persist artifacts if requested
      if [ -n "$KEEP_DIR" ]; then
        mkdir -p "$KEEP_DIR"
        if [ -f "$out" ]; then
          cp -f "$out" "$KEEP_DIR/$(basename "$f").${mode}.run${run}.lzo" || true
        fi
        if [ -f "$decomp_out" ]; then
          cp -f "$decomp_out" "$KEEP_DIR/$(basename "$f").${mode}.run${run}.dec" || true
        fi
        # copy run logs too
        cp -f "$run_log" "$KEEP_DIR/$(basename "$f").${mode}.run${run}.log" 2>/dev/null || true
        cp -f "$decomp_log" "$KEEP_DIR/$(basename "$f").${mode}.run${run}.decomp.log" 2>/dev/null || true
      fi

      # remove temporary files to keep disk small (but keep mode sample kept_outs for later compare)
      rm -f "$out"
      rm -f "$decomp_out" "$TMPDIR/decomp.time"
    done
  done

  # Compare per-mode first-run compressed samples (if present)
  # Use a unique id based on full file path to avoid collisions when different files share the same basename
  file_id=$(printf '%s' "$f" | sha256sum | awk '{print $1}')
  safe_name=$(basename "$f" | tr '/\\' '_')
  declare -A mode_hash
  for mode in "${modes[@]}"; do
    samp="$TMPDIR/${safe_name}.${file_id}.${mode}.lzo"
    if [ -f "$samp" ]; then
      h=$(sha256sum "$samp" | awk '{print $1}')
      mode_hash["$mode"]="$h"
    else
      mode_hash["$mode"]=""
    fi
  done

  # Build human-friendly per-mode hashes & sizes and detect uniqueness
  declare -A mode_size
  declare -A uniq
  for mode in "${modes[@]}"; do
    samp="$TMPDIR/${safe_name}.${file_id}.${mode}.lzo"
    if [ -f "$samp" ]; then
      mode_hash["$mode"]=$(sha256sum "$samp" | awk '{print $1}')
      mode_size["$mode"]=$(stat -c %s "$samp")
      uniq["${mode_hash[$mode]}"]=1
    else
      mode_hash["$mode"]=""
      mode_size["$mode"]=0
    fi
  done

  # determine if all non-empty mode hashes are identical
  first_hash=""
  for mode in "${modes[@]}"; do
    if [ -n "${mode_hash[$mode]}" ]; then
      first_hash="${mode_hash[$mode]}"
      break
    fi
  done

  different=0
  for mode in "${modes[@]}"; do
    h="${mode_hash[$mode]}"
    if [ -n "$h" ] && [ "$h" != "$first_hash" ]; then
      different=1
      break
    fi
  done

  if [ $different -eq 0 ]; then
    echo "  [COMPARE] Compressed outputs for $f: ALL IDENTICAL (or only one mode produced output)"
  else
    echo "  [COMPARE] Compressed outputs for $f: DIFFERENT across modes (per-mode sample hash & size)"
    for mode in "${modes[@]}"; do
      h=${mode_hash[$mode]}
      s=${mode_size[$mode]}
      if [ -z "$h" ]; then
        echo "    $mode: MISSING"
      else
        echo "    $mode: $h  size=$s"
      fi
    done
  fi

  # clean up saved mode sample files for this input
  rm -f "$TMPDIR/${safe_name}.${file_id}."*.lzo || true
done

echo "Results saved to: $OUT_CSV"

exit 0
