#!/usr/bin/env bash
# Lightweight, single-file parameter-scan runner for lzo_gpu

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WRAPDIR="$ROOT/lzo_gpu"
OUT_DIR="${OUT_DIR:-$ROOT/exp_results/lzo_gpu/logs/param_scans}"
mkdir -p "$OUT_DIR"

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
  STRATEGIES=(none atomic usehost)
fi

if [ -n "${LZO_WG_SIZE:-}" ]; then
  read -r -a WG_SIZES <<< "$(echo "$LZO_WG_SIZE" | tr ',' ' ')"
else
  WG_SIZES=(32 64 128 256)
fi

if [ -n "${LZO_VLEN:-}" ]; then
  read -r -a VLEN <<< "$(echo "$LZO_VLEN" | tr ',' ' ')"
else
  VLEN=(1 2 4 8)
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
  for wg in "${WG_SIZES[@]}"; do
    for v in "${VLEN[@]}"; do
      for sample in "${SAMPLES[@]}"; do
        relpath="${sample#${SAMPLES_DIR}/}"
        if [ "$relpath" = "$sample" ]; then relpath="$(basename "$sample")"; fi
        rel_sanitized=$(printf "%s" "$relpath" | sed 's/[^A-Za-z0-9._-]/_/g')
        sample_hash=$(printf "%s" "$sample" | sha1sum 2>/dev/null | awk '{print $1}' | cut -c1-8 || echo unknown)
        sname="${rel_sanitized}_${sample_hash}"

        for r in $(seq 1 "$REPEATS"); do
          for strategy in "${STRATEGIES[@]}"; do
            for mode in base vec; do
              devec_val=0
              if [ "$mode" = "vec" ]; then devec_val=1; fi
              total_runs=$((total_runs+1))

              cfg_dir_mode="$OUT_DIR/comp_${comp_level}/strategy_${strategy}/decomp_${mode}/wg_${wg}_v_${v}"
              mkdir -p "$cfg_dir_mode"

              if type lzo_mktemp_dir >/dev/null 2>&1; then
                lzo_mktemp_dir tmp_run_dir || tmp_run_dir=$(mktemp -d /tmp/lzo_gpu_tmp.XXXXXX)
              else
                tmp_run_dir=$(mktemp -d /tmp/lzo_gpu_tmp.XXXXXX)
              fi

              out_lzo="$tmp_run_dir/lzo_out_${sname}_run${r}.lzo"
              logf="$cfg_dir_mode/${sname}_run${r}.log"

              echo "[Run $total_runs] COMP=$comp_level MODE=$mode WG=$wg VLEN=$v SAMPLE=$sname R=$r -> $logf"
              echo "# COMP=$comp_level MODE=$mode WG=$wg VLEN=$v SAMPLE=$sname R=$r" > "$logf"
              echo "Compressing: $sample -> $out_lzo" >> "$logf"

              # construct strategy argument: omit when 'none'
              strategy_arg=()
              if [ "$strategy" != "none" ]; then
                strategy_arg=(--strategy "$strategy")
              fi

              COMP_CMD=(env LZO_DECOMP_VEC=$devec_val LZO_WG_SIZE=$wg LZO_VLEN=$v "$LZO_BIN" $LZO_DEBUG_FLAG -L "$comp_level" "${strategy_arg[@]}" "$sample" -o "$out_lzo")
              DECMD=(env LZO_DECOMP_VEC=$devec_val "$LZO_BIN" -d --verify "$sample" "$out_lzo")

              if [ "$DRY_RUN" = "1" ]; then
                printf "# DRY-RUN CMD: (cd \"%s\" && %s)\n" "$WRAPDIR" "${COMP_CMD[*]}" | tee -a "$logf"
                compress_status=0
              else
                (cd "$WRAPDIR" && "${COMP_CMD[@]}") >> "$logf" 2>&1 || true
                compress_status=$?
              fi

              echo "Decompress+verify" >> "$logf"
              if [ "$DRY_RUN" = "1" ]; then
                printf "# DRY-RUN CMD: (cd \"%s\" && %s)\n" "$WRAPDIR" "${DECMD[*]}" | tee -a "$logf"
                verify_status=0
              else
                (cd "$WRAPDIR" && "${DECMD[@]}") >> "$logf" 2>&1 || true
                verify_status=$?
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

echo "Full param scan finished. Total runs: $total_runs. Logs under $OUT_DIR"
