#!/usr/bin/env bash
# Lightweight, single-file parameter-scan runner for lzo_gpu

# Enable strict mode. Support both bash and zsh (zsh uses `setopt PIPE_FAIL`).
# In most cases the script runs under `bash` due to the shebang; but make the
# settings defensive so sourcing under zsh or other shells doesn't break.
set -e
set -u
if [ -n "${BASH_VERSION:-}" ]; then
  # bash supports the combined form
  set -o pipefail
elif [ -n "${ZSH_VERSION:-}" ]; then
  # zsh uses setopt to enable pipefail
  setopt PIPE_FAIL || true
else
  # try the POSIX-ish form, ignore if not supported
  set -o pipefail 2>/dev/null || true
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPDIR="$ROOT/lzo_gpu"
OUT_DIR="${OUT_DIR:-$ROOT/exp_results/lzo_gpu/logs/param_scans}"
mkdir -p "$OUT_DIR"
TMP_BASE="$OUT_DIR/tmp"
mkdir -p "$TMP_BASE"

# Defaults
SAMPLES_DIR_DEFAULT="/root/samples"
SAMPLES_DIR="${SAMPLES_DIR:-$SAMPLES_DIR_DEFAULT}"
DRY_RUN=0
REPEATS=${REPEATS:-5}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
Options:
  -s, --samples DIR   Directory containing sample files (default: $SAMPLES_DIR_DEFAULT)
  -n, --dry-run       Print commands and record planned actions, do not execute
  -h, --help          Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -s|--samples)
      shift; SAMPLES_DIR="${1:-}"; shift || true; ;;
    -n|--dry-run)
      DRY_RUN=1; shift || true; ;;
    -h|--help)
      usage; exit 0; ;;
    *) shift ;;
  esac
done

if [ ! -d "$SAMPLES_DIR" ]; then
  echo "Samples directory not found: $SAMPLES_DIR" >&2
  exit 1
fi

# Enumerations (env override supported)
if [ -n "${LZO_COMP_LEVELS:-}" ]; then
  read -r -a COMP_LEVELS <<< "$(echo "$LZO_COMP_LEVELS" | tr ',' ' ')"
else
  COMP_LEVELS=(1 1k 1l 1o)
fi

if [ -n "${LZO_STRATEGIES:-}" ]; then
  read -r -a STRATEGIES <<< "$(echo "$LZO_STRATEGIES" | tr ',' ' ')"
else
  STRATEGIES=(none)
fi

if [ -n "${LZO_WG_SIZE:-}" ]; then
  read -r -a WG_SIZES <<< "$(echo "$LZO_WG_SIZE" | tr ',' ' ')"
else
  WG_SIZES=(32 64 128 256)
fi

# Block sizes (KB) to sweep; default to the adaptive-friendly range
if [ -n "${LZO_FIXED_BLOCK_SIZES:-}" ]; then
  read -r -a BLOCK_SIZES <<< "$(echo "$LZO_FIXED_BLOCK_SIZES" | tr ',' ' ')"
else
  BLOCK_SIZES=(64 128 256 512)
fi

# Multi-thread IO modes to consider (0: no mt, 1: mt)
if [ -n "${LZO_MT_IO_MODES:-}" ]; then
  read -r -a MT_IO_MODES <<< "$(echo "$LZO_MT_IO_MODES" | tr ',' ' ')"
else
  MT_IO_MODES=(0 1)
fi

# Runners: standalone,daemon - can be overridden by env LZO_RUNNERS="standalone,daemon"
if [ -n "${LZO_RUNNERS:-}" ]; then
  read -r -a RUNNERS <<< "$(echo "$LZO_RUNNERS" | tr ',' ' ')"
else
  RUNNERS=(standalone daemon)
fi

# Optional: LZO_ASYNC_UPLOAD and LZO_MT_IO_THREADS can be set to include these envs in the invocation.
LZO_ASYNC_UPLOAD=${LZO_ASYNC_UPLOAD:-}
LZO_MT_IO_THREADS=${LZO_MT_IO_THREADS:-2}

# Decompression vector modes: if LZO_DECOMP_VEC_MODES supplied, iterate them.
# If not supplied, use empty string in DECOMP_VEC_MODES meaning "do not set LZO_DECOMP_VEC, use binary default".
if [ -n "${LZO_DECOMP_VEC_MODES:-}" ]; then
  read -r -a DECOMP_VEC_MODES <<< "$(echo "$LZO_DECOMP_VEC_MODES" | tr ',' ' ')"
else
  DECOMP_VEC_MODES=("")
fi



# Mapping mode: do not include a mapping dimension by default. To explicitly
# include mapping choices in the scan, set LZO_STANDARD_COPY_MODES to a
# comma-separated list of 0/1 (0=zero-copy, 1=standard-copy). For backward
# compatibility, LZO_MAP_MODES is accepted as an alias.
if [ -n "${LZO_STANDARD_COPY_MODES:-}" ]; then
  read -r -a STD_COPY_MODES <<< "$(echo "$LZO_STANDARD_COPY_MODES" | tr ',' ' ')"
elif [ -n "${LZO_MAP_MODES:-}" ]; then
  # Deprecated alias: translate into new variable name
  read -r -a STD_COPY_MODES <<< "$(echo "$LZO_MAP_MODES" | tr ',' ' ')"
  echo "Warning: LZO_MAP_MODES is deprecated; use LZO_STANDARD_COPY_MODES instead" >&2
else
  STD_COPY_MODES=()
fi
# Validate allowed values: only 0 or 1 are permitted
for mm in "${STD_COPY_MODES[@]}"; do
  if [ "$mm" != "0" ] && [ "$mm" != "1" ]; then
    echo "Invalid STD_COPY_MODE value: $mm. Allowed values are 0 or 1." >&2
    exit 2
  fi
done

# normalize STD_COPY_MODES for iteration (empty means no stdcopy dimension)
if [ ${#STD_COPY_MODES[@]} -eq 0 ]; then
  STD_COPY_LOOP=("")
else
  STD_COPY_LOOP=("${STD_COPY_MODES[@]}")
fi

# gather samples
# Include regular files and symlinks that point to regular files.
# Use find to list both file entries and symlinks, then filter with
# shell test -f to ensure broken symlinks are ignored.
SAMPLES=()
while IFS= read -r -d '' f; do
  if [ -f "$f" ]; then
    SAMPLES+=("$f")
  fi
done < <(find "$SAMPLES_DIR" \( -type f -o -type l \) -print0)
if [ ${#SAMPLES[@]} -eq 0 ]; then
  echo "No sample files found in $SAMPLES_DIR" >&2; exit 1
fi

LZO_BIN="$WRAPDIR/lzo_gpu"
total_runs=0

# Optional debug flag controlled by env var LZO_DEBUG
LZO_DEBUG_FLAG=""
if [ "${LZO_DEBUG:-0}" = "1" ]; then
  LZO_DEBUG_FLAG="--debug"
fi

for comp_level in "${COMP_LEVELS[@]}"; do
  for runner in "${RUNNERS[@]}"; do
  for wg in "${WG_SIZES[@]}"; do
    for block_kb in "${BLOCK_SIZES[@]}"; do
      for mt_io in "${MT_IO_MODES[@]}"; do
          for stdcopy in "${STD_COPY_LOOP[@]}"; do
          for sample in "${SAMPLES[@]}"; do
        relpath="${sample#${SAMPLES_DIR}/}"
        if [ "$relpath" = "$sample" ]; then relpath="$(basename "$sample")"; fi
        rel_sanitized=$(printf "%s" "$relpath" | sed 's/[^A-Za-z0-9._-]/_/g')
        sample_hash=$(printf "%s" "$sample" | sha1sum 2>/dev/null | awk '{print $1}' | cut -c1-8 || echo unknown)
        sname="${rel_sanitized}_${sample_hash}"

        for devec in "${DECOMP_VEC_MODES[@]}"; do
          for r in $(seq 1 "$REPEATS"); do
            # default: no explicit LZO_STANDARD_COPY mode unless user configured it
            stdcopy=""
            if [ ${#STD_COPY_MODES[@]} -gt 0 ]; then
              # choose the first (and typically only) configured standard-copy mode
              stdcopy="${STD_COPY_MODES[0]}"
            fi
            # No decomp-mode dimension (always base); drop the extra directory level
            devec_val="$devec"
            total_runs=$((total_runs+1))

            if [ -n "$stdcopy" ]; then
              cfg_dir_mode="$OUT_DIR/comp_${comp_level}/stdcopy_${stdcopy}/wg_${wg}/block_${block_kb}kb/mt_${mt_io}"
            else
              cfg_dir_mode="$OUT_DIR/runner_${runner}/comp_${comp_level}/wg_${wg}/block_${block_kb}kb/mt_${mt_io}"
            fi
            mkdir -p "$cfg_dir_mode"

              if type lzo_mktemp_dir >/dev/null 2>&1; then
                lzo_mktemp_dir tmp_run_dir || tmp_run_dir=$(mktemp -d "$TMP_BASE/lzo_gpu_tmp.XXXXXX")
              else
                tmp_run_dir=$(mktemp -d "$TMP_BASE/lzo_gpu_tmp.XXXXXX")
              fi

              out_lzo="$tmp_run_dir/lzo_out_${sname}_run${r}.lzo"
              logf="$cfg_dir_mode/${sname}_run${r}.log"

              # Make log header explicit: include standard-copy mode, block size and mt io if configured
              BLK_SZ=${block_kb}
              MT_IO=${mt_io}
              if [ -n "$stdcopy" ]; then
                echo "[Run $total_runs] COMP=$comp_level STD_COPY=$stdcopy WG=$wg BLOCK=${BLK_SZ}KB MT=${MT_IO} SAMPLE=$sname R=$r -> $logf"
                echo "# COMP=$comp_level STD_COPY=$stdcopy WG=$wg BLOCK=${BLK_SZ}KB MT=${MT_IO} SAMPLE=$sname R=$r" > "$logf"
              else
                echo "[Run $total_runs] COMP=$comp_level WG=$wg BLOCK=${BLK_SZ}KB MT=${MT_IO} SAMPLE=$sname R=$r -> $logf"
                echo "# COMP=$comp_level WG=$wg BLOCK=${BLK_SZ}KB MT=${MT_IO} SAMPLE=$sname R=$r" > "$logf"
              fi
              echo "Compressing: $sample -> $out_lzo" >> "$logf"
              # no strategy argument (strategy dimension removed)
              strategy_arg=()
              # Construct env vars for invocation; include optional STD_COPY if configured.
              # Build COMP_ENV with optional entries
              COMP_ENV=()
              # Only set LZO_DECOMP_VEC if explicit value provided (empty string means use binary default)
              if [ -n "$devec_val" ]; then
                COMP_ENV+=(LZO_DECOMP_VEC=$devec_val)
              fi
              if [ "$wg" != "auto" ]; then
                COMP_ENV+=(LZO_WG_SIZE=$wg)
              fi
              COMP_ENV+=(LZO_FIXED_BLOCK_SIZE=${BLK_SZ:-0})
              COMP_ENV+=(LZO_MT_IO=${MT_IO:-0})
              # Pass LZO_MT_IO_THREADS if set
              if [ -n "${LZO_MT_IO_THREADS:-}" ]; then
                COMP_ENV+=(LZO_MT_IO_THREADS=${LZO_MT_IO_THREADS})
              fi
              # If standard copy mode is set, include it
              if [ -n "$stdcopy" ]; then
                COMP_ENV+=(LZO_STANDARD_COPY=${stdcopy})
              fi
              # Pass async upload if configured in environment
              if [ -n "${LZO_ASYNC_UPLOAD:-}" ]; then
                COMP_ENV+=(LZO_ASYNC_UPLOAD=${LZO_ASYNC_UPLOAD})
              fi
              if [ -n "$stdcopy" ]; then
                COMP_ENV+=(LZO_STANDARD_COPY=$stdcopy)
              fi
              # LZO_VLEN is metadata only (host does not use it), so do not export it to the child process
              # Choose binary based on runner
              if [ "$runner" = "daemon" ]; then
                LZO_BIN_RUN="$WRAPDIR/lzo_gpu_client"
                # Attempt to start daemon if not running
                if ! pgrep -f lzo_gpu_daemon >/dev/null 2>&1; then
                  echo "Starting lzo_gpu_daemon for runner=daemon"
                  (cd "$WRAPDIR" && nohup ./lzo_gpu_daemon > "$OUT_DIR/lzo_gpu_daemon.stdout.log" 2>&1 &) || true
                  sleep 0.5
                fi
              else
                LZO_BIN_RUN="$WRAPDIR/lzo_gpu"
              fi
              COMP_CMD=(env "${COMP_ENV[@]}" "$LZO_BIN_RUN" $LZO_DEBUG_FLAG -L "$comp_level" "${strategy_arg[@]}" "$sample" -o "$out_lzo")
              # DECMD: ensure decompress/verify uses same envs as COMP
              DEC_ENV=()
              if [ -n "$devec_val" ]; then DEC_ENV+=(LZO_DECOMP_VEC=$devec_val); fi
              if [ "$wg" != "auto" ]; then DEC_ENV+=(LZO_WG_SIZE=$wg); fi
              DEC_ENV+=(LZO_FIXED_BLOCK_SIZE=${BLK_SZ:-0})
              DEC_ENV+=(LZO_MT_IO=${MT_IO:-0})
              if [ -n "${LZO_MT_IO_THREADS:-}" ]; then DEC_ENV+=(LZO_MT_IO_THREADS=${LZO_MT_IO_THREADS}); fi
              if [ -n "$stdcopy" ]; then DEC_ENV+=(LZO_STANDARD_COPY=${stdcopy}); fi
              if [ -n "${LZO_ASYNC_UPLOAD:-}" ]; then DEC_ENV+=(LZO_ASYNC_UPLOAD=${LZO_ASYNC_UPLOAD}); fi
              DECMD=(env "${DEC_ENV[@]}" "$LZO_BIN_RUN" -d --verify "$sample" "$out_lzo")

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
                mkdir -p "$cfg_dir_mode/artifacts"
                mv "$out_lzo" "$cfg_dir_mode/artifacts/lzo_out_${sname}_run${r}.lzo" 2>/dev/null || true
              fi

              rm -rf "$tmp_run_dir" 2>/dev/null || true
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Full param scan finished. Total runs: $total_runs. Logs under $OUT_DIR"
