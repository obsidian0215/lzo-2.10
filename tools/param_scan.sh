#!/bin/bash
# LZO GPU/CPU 完整参数扫描测试脚本
# 测试各种 blocksize, localsize, 算法, IO优化选项等

# set -e removed to allow continuing after failures

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LZO_CPU="$REPO_DIR/lzo_cpu/lzo_cpu"
LZO_GPU="$REPO_DIR/lzo_gpu/lzo_gpu"
LZO_GPU_DIR="$REPO_DIR/lzo_gpu"
LZO_GPU_DAEMON="$REPO_DIR/lzo_gpu/lzo_gpu_daemon"
LZO_GPU_CLIENT="$REPO_DIR/lzo_gpu/lzo_gpu_client"
SAMPLES_DIR="${SAMPLES_DIR:-/root/samples}"

# 结果文件
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_DIR/exp_results/param_scan"
CSV_FILE="${CSV_FILE:-$RESULTS_DIR/param_scan_${TIMESTAMP}.csv}"
REPORT_FILE="${REPORT_FILE:-$RESULTS_DIR/param_scan_report_${TIMESTAMP}.txt}"

# 测试配置
ALGORITHMS=("1x" "1y")
LEVELS=(10 11 12 14)
BLOCK_SIZES=(8 16 32 64)  # KB
LOCAL_SIZES=(1)
CPU_THREADS=(1 2 4)
COPY_MODES=(0)  # 0=zero-copy, 1=standard copy
STDIO_BUF_OPTIONS=(4)  # stdio buffer sizes in MB
MT_IO_OPTIONS=(0 1)  # multi-threaded IO
MT_IO_THREADS_OPTIONS=(2) # MT_IO thread counts
COALESCE_OPTIONS=(1)  # enable/disable output coalescing

# Helper: start a daemon for param_scan runs
start_daemon() {
    local daemon_env_ext="$1"
    local env_vars="${daemon_env_ext:-}"
    # Extract socket path in the requested daemon env (default: /tmp/lzo_gpu_daemon.sock)
    local sock
    sock=$(echo "$env_vars" | grep -oP 'LZO_DAEMON_SOCKET=\K[^ ]+' || true)
    if [ -z "$sock" ]; then
        sock="/tmp/lzo_gpu_daemon.sock"
    fi

    # Create unique pid/log paths for this socket
    local base
    base=$(basename "$sock" .sock)
    local pidfile="/tmp/${base}.$$.pid"
    local logfile="/tmp/${base}.$$.log"

    # Kill any existing daemons before starting test-specific daemon
    stop_daemon 2>/dev/null || true

    # Start a new daemon in the background and capture its PID
    local cwd
    cwd=$(pwd)
    cd "$LZO_GPU_DIR"
    if [ -n "$env_vars" ]; then
        # Use eval / env to set environment variables for child process
        # shellcheck disable=SC2086
        eval env $env_vars LZO_DAEMON_PID="$pidfile" LZO_DAEMON_LOG="$logfile" "$LZO_GPU_DAEMON" > "$logfile" 2>&1 &
    else
        "$LZO_GPU_DAEMON" > "$logfile" 2>&1 &
    fi
    local pid=$!
    cd "$cwd"
    # Save the pid and aux info for stop_daemon and for clients
    DAEMON_PID=$pid
    SCAN_DAEMON_PID=$pid
    SCAN_DAEMON_PIDFILE=$pidfile
    SCAN_DAEMON_LOGFILE=$logfile
    SCAN_DAEMON_SOCKET=$sock
    SCAN_DAEMON_OWNED=1
    echo "Started new daemon (pid=$pid, socket=$SCAN_DAEMON_SOCKET)"

    # Wait for the process to start and socket to be created AND accept a connection
    for i in {1..30}; do
        if [ -S "$sock" ]; then
            # Try to actually connect to the unix socket to verify the server is listening
            # Use python3 to attempt a simple AF_UNIX connect (safe & small)
            if python3 - <<PYTEST "$sock" >/dev/null 2>&1
import socket,sys
try:
  s=socket.socket(socket.AF_UNIX)
  s.settimeout(0.5)
  s.connect(sys.argv[1])
  s.close()
  sys.exit(0)
except Exception:
  sys.exit(1)
PYTEST
            then
                # short pause to let daemon stabilize
                sleep 0.2
                return 0
            fi
        fi
        # If process died, break early
        if ! ps -p "$pid" >/dev/null 2>&1; then
            break
        fi
        sleep 0.2
    done
    # If we get here, starting daemon likely failed
    # If daemon failed to create socket, dump daemon log for debugging and fail
    echo "Failed to start daemon (socket=${sock})" >&2
    if [ -f "$logfile" ]; then
        echo "--- Daemon log tail ($logfile) ---" >&2
        tail -n 200 "$logfile" >&2 || true
        echo "--- End Daemon log tail ---" >&2
    fi
    # If a log file exists, show last 200 lines to help debugging
    return 1
}

# Helper: stop a daemon started by start_daemon
stop_daemon() {
    # If we have a SCAN_DAEMON_PID, kill it
    local pid="${SCAN_DAEMON_PID:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        local t=0
        while kill -0 "$pid" 2>/dev/null && [ $t -lt 50 ]; do
            sleep 0.05
            t=$((t + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi

    # Also attempt to kill any lzo_gpu_daemon process to ensure clean state
    pkill -f "lzo_gpu_daemon" 2>/dev/null || true

    # Cleanup artifacts
    if [ -n "${SCAN_DAEMON_PIDFILE:-}" ]; then rm -f "${SCAN_DAEMON_PIDFILE}" >/dev/null 2>&1 || true; fi
    if [ -n "${SCAN_DAEMON_LOGFILE:-}" ]; then rm -f "${SCAN_DAEMON_LOGFILE}" >/dev/null 2>&1 || true; fi
    local sock="${SCAN_DAEMON_SOCKET:-/tmp/lzo_gpu_daemon.sock}"
    if [ -S "$sock" ]; then
        rm -f "$sock" >/dev/null 2>&1 || true
    fi
    unset SCAN_DAEMON_PID
    unset SCAN_DAEMON_PIDFILE
    unset SCAN_DAEMON_LOGFILE
    unset SCAN_DAEMON_SOCKET
    unset SCAN_DAEMON_OWNED
}


test_gpu_daemon() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sample_size_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)

    echo -e "\n${CYAN}=== GPU Daemon Tests: $sample_name ===${NC}"

    for alg in "${ALGORITHMS[@]}"; do
        for lvl in "${LEVELS[@]}"; do
            for blk_kb in "${BLOCK_SIZES[@]}"; do
                    for lsz in "${LOCAL_SIZES[@]}"; do
                        for mt_io in "${MT_IO_OPTIONS[@]}"; do
                            for copy_mode in "${COPY_MODES[@]}"; do
                                for mt_threads in "${MT_IO_THREADS_OPTIONS[@]}"; do
                                    local config="daemon_${alg}_${lvl}_${blk_kb}k"
                                    local tmp_lzo="/tmp/pscan_${sample_name}.lzo"
                                    local tmp_out="/tmp/pscan_${sample_name}.out"

                                    # start daemon with minimal daemon-level env. Keep per-request overrides as client env vars
                                    local uniq_sock
                                    # Use default daemon socket unless overridden by environment.
                                    uniq_sock="${LZO_DAEMON_SOCKET:-/tmp/lzo_gpu_daemon.sock}"
                                    local daemon_env="LZO_DAEMON_SOCKET=${uniq_sock}"
                                    if [ "${blk_kb}" -lt 64 ]; then
                                        daemon_env="$daemon_env LZO_MIN_BLOCK_SIZE=${blk_kb}KB"
                                    fi

                                    for coalesce in "${COALESCE_OPTIONS[@]}"; do
                                        for stdio_buf in "${STDIO_BUF_OPTIONS[@]}"; do
                                            local daemon_env_ext="$daemon_env"
                                            local config_ext="${config}_mt${mt_threads}_co${coalesce}_sbuf${stdio_buf}"
                                            if ! start_daemon "$daemon_env_ext"; then
                                                echo -e "  ${RED}Daemon start failed for $config (coalesce=$coalesce,sbuf=$stdio_buf)${NC}"
                                                continue
                                            fi

                                            # client per-request env vars - use the actual socket used by the daemon (SCAN_DAEMON_SOCKET)
                                            local socket_used="${SCAN_DAEMON_SOCKET:-${uniq_sock}}"
                                            local env_vars="LZO_DAEMON_SOCKET=${socket_used} LZO_MT_IO=${mt_io} LZO_STANDARD_COPY=${copy_mode} LZO_COALESCE_OUTPUT=${coalesce} LZO_STDIO_BUF_MB=${stdio_buf}"
                                            if [ "${mt_io}" -eq 1 ]; then
                                                env_vars="$env_vars LZO_MT_IO_THREADS=${mt_threads}"
                                            else
                                                mt_threads="NA"
                                            fi
                                            if [ "${blk_kb}" -lt 64 ]; then
                                                env_vars="$env_vars LZO_MIN_BLOCK_SIZE=${blk_kb}KB"
                                            fi

                                            local cli_args_comp="-a ${alg} -l ${lvl} -B ${blk_kb}KB --local 1"
                                            local cli_args_decomp="-a ${alg} -l ${lvl} -B ${blk_kb}KB --local ${lsz}"

                                            # run compression via client
                                            local comp_out
                                            comp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU_CLIENT" ${cli_args_comp} "$sample" -o "$tmp_lzo" 2>&1)
                                            local comp_perf
                                            comp_perf=$(extract_perf "$comp_out")

                                            # Sanity-check: ensure perf extractor produced expected number of comma-separated fields
                                            local expected_fields=20
                                            local comp_fields
                                            comp_fields=$(awk -F, '{print NF}' <<< "$comp_perf")
                                            if [ "$comp_fields" -ne "$expected_fields" ]; then
                                                echo "  ${YELLOW}[WARN] Unexpected comp_perf field count: $comp_fields != $expected_fields (${config_ext})${NC}" >&2
                                                mkdir -p "$RESULTS_DIR/debug" >/dev/null 2>&1 || true
                                                printf "=== COMP OUT (%s) ===\n%s\n\n=== PERF ===\n%s\n" "$config_ext" "$comp_out" "$comp_perf" > "$RESULTS_DIR/debug/${sample_name}_${config_ext}_comp_raw.txt"
                                            fi

                                            # run decompression via client
                                            local decomp_out
                                            decomp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU_CLIENT" ${cli_args_decomp} -d "$tmp_lzo" -o "$tmp_out" 2>&1)
                                            local decomp_perf
                                            decomp_perf=$(extract_perf "$decomp_out")

                                            # Sanity-check decomp perf fields as well
                                            local decomp_fields
                                            decomp_fields=$(awk -F, '{print NF}' <<< "$decomp_perf")
                                            if [ "$decomp_fields" -ne "$expected_fields" ]; then
                                                echo "  ${YELLOW}[WARN] Unexpected decomp_perf field count: $decomp_fields != $expected_fields (${config_ext})${NC}" >&2
                                                mkdir -p "$RESULTS_DIR/debug" >/dev/null 2>&1 || true
                                                printf "=== DECOMP OUT (%s) ===\n%s\n\n=== PERF ===\n%s\n" "$config_ext" "$decomp_out" "$decomp_perf" > "$RESULTS_DIR/debug/${sample_name}_${config_ext}_decomp_raw.txt"
                                            fi

                                            local verified
                                            verified=$(verify_output "$sample" "$tmp_out")

                                            stop_daemon

                                            # parse and write CSV
                                            IFS=',' read -r c_tp c_ktp c_tt c_kt c_rt c_wt c_ut c_dt c_ratio_frac c_ratio_pct c_kernel_name c_global c_local c_buf_in c_buf_out c_buf_len c_block_calc c_dl_len c_dl_bulk c_dl_total <<< "$comp_perf"
                                            echo "$sample_name,$sample_size_mb,Daemon,$config_ext,compress,$alg,$lvl,$blk_kb,NA,$mt_threads,$mt_io,$copy_mode,$coalesce,$stdio_buf,$c_ratio_frac,$c_ratio_pct,$c_tt,$c_kt,$c_rt,$c_wt,$c_ut,$c_dt,${c_kernel_name:-},${c_global:-},${c_local:-},${c_buf_in:-},${c_buf_out:-},${c_buf_len:-},${c_block_calc:-},${c_dl_len:-},${c_dl_bulk:-},${c_dl_total:-},$c_tp,$c_ktp,$verified" >> "$CSV_FILE"

                                            IFS=',' read -r d_tp d_ktp d_tt d_kt d_rt d_wt d_ut d_dt d_ratio_frac d_ratio_pct d_kernel_name d_global d_local d_buf_in d_buf_out d_buf_len d_block_calc d_dl_len d_dl_bulk d_dl_total <<< "$decomp_perf"
                                            echo "$sample_name,$sample_size_mb,Daemon,$config_ext,decompress,$alg,$lvl,$blk_kb,NA,$mt_threads,$mt_io,$copy_mode,$coalesce,$stdio_buf,$d_ratio_frac,$d_ratio_pct,$d_tt,$d_kt,$d_rt,$d_wt,$d_ut,$d_dt,${d_kernel_name:-},${d_global:-},${d_local:-},${d_buf_in:-},${d_buf_out:-},${d_buf_len:-},${d_block_calc:-},${d_dl_len:-},${d_dl_bulk:-},${d_dl_total:-},$d_tp,$d_ktp,$verified" >> "$CSV_FILE"

                                            local status="✓"
                                            local status_color="$GREEN"
                                            if [ "$verified" != "YES" ]; then
                                                status="✗"
                                                status_color="$RED"
                                                local fail_dir="$RESULTS_DIR/failures/${sample_name}/${config_ext}"
                                                mkdir -p "$fail_dir"
                                                [ -f "$tmp_lzo" ] && mv "$tmp_lzo" "$fail_dir/failed.lzo"
                                                [ -f "$tmp_out" ] && mv "$tmp_out" "$fail_dir/failed.out"
                                            else
                                                rm -f "$tmp_lzo" "$tmp_out"
                                            fi
                                            printf "  %-35s " "$config_ext"
                                            echo -en "${status_color}${status}${NC} "
                                            printf "COMP:%6.1f (k:%7.1f) [R: %s (%s%%)] | DECOMP:%6.1f (k:%7.1f)\n" "${c_tp:-0}" "${c_ktp:-0}" "${c_ratio_frac:-NA}" "${c_ratio_pct:-NA}" "${d_tp:-0}" "${d_ktp:-0}"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
}
cleanup() {
    stop_daemon
    rm -f /tmp/pscan_*.lzo /tmp/pscan_*.out
}
trap cleanup EXIT

# 提取性能数据
extract_perf() {
    local output="$1"
    local total_tp=$(echo "$output" | grep -oP 'Throughput\s*:\s*\K[\d.]+' | head -1)
    local kernel_tp=$(echo "$output" | grep -oP 'kernel\s*:\s*\K[\d.]+' | head -1)
    local total_time=$(echo "$output" | grep -oP 'TOTAL\s*:\s*\K[\d.]+' | head -1)
    local kernel_time=$(echo "$output" | grep -oP 'Kernel Exec\s*:\s*\K[\d.]+' | head -1)

    # Read & write times: standalone uses 'File Read', daemon may not. Use fallback.
    local read_time=$(echo "$output" | grep -oP '1\.[ ]*File Read\s*:\s*\K[\d.]+' | head -1)
    if [ -z "$read_time" ]; then
        read_time=$(echo "$output" | grep -oP '1\.[ ]*Buffer Alloc\s*:\s*\K[\d.]+' | head -1)
    fi
    local write_time=$(echo "$output" | grep -oP '[0-9]+\.[ ]*File Write\s*:\s*\K[\d.]+' | head -1)

    # Upload/download times: look for several variants
    local upload_time=$(echo "$output" | grep -oP 'Data Upload\s*:\s*\K[\d.]+' | head -1)
    if [ -z "$upload_time" ]; then
        upload_time=$(echo "$output" | grep -oP 'Data Upload.*\(total\)\s*:\s*\K[\d.]+' | head -1)
    fi
    # Download time: try several numeric patterns in order. Avoid alternation that can
    # accidentally match literal labels like 'Download (len)' which would break CSV layout.
    local download_time=""
    download_time=$(echo "$output" | grep -oP 'Data Download\s*:\s*\K[0-9.]+' | head -1)
    if [ -z "$download_time" ]; then
        download_time=$(echo "$output" | grep -oP 'Download Total\s*:\s*\K[0-9.]+' | head -1)
    fi
    if [ -z "$download_time" ]; then
        download_time=$(echo "$output" | grep -oP 'Download \(bulk\)\s*:\s*\K[0-9.]+' | head -1)
    fi
    if [ -z "$download_time" ]; then
        download_time=$(echo "$output" | grep -oP 'Download \(len\)\s*:\s*\K[0-9.]+' | head -1)
    fi
    if [ -z "$download_time" ]; then
        download_time=$(echo "$output" | grep -oP 'Download\s*:\s*\K[0-9.]+' | head -1)
    fi

    # compression ratio - parse both fraction (e.g., 4.70:1 or '4.70 : 1') and percent (e.g., 21.28%)
    local ratio_frac=""
    local ratio_pct=""
    # Grab the first Compression ratio line (case-insensitive) if present
    local ratio_line=$(echo "$output" | grep -m1 -i 'Compression ratio' || true)
    if [ -n "$ratio_line" ]; then
        # parse fraction like '4.70:1' or '4.70 : 1' and normalize (remove spaces)
        ratio_frac=$(echo "$ratio_line" | grep -oP '[0-9.]+\s*:\s*[0-9.]+' | head -1 | tr -d ' ' || true)
        # parse percent anywhere on the line (e.g., '21.28%')
        ratio_pct=$(echo "$ratio_line" | grep -oP '[0-9.]+(?=%)' | head -1 || true)
    fi
    # fallback: try to capture percent from the 'Compressed ... (xx%)' style line if not present
    if [ -z "$ratio_pct" ]; then
        ratio_pct=$(echo "$output" | grep -oP 'Compressed .*\(\s*\K[0-9.]+(?=%)' | head -1 || true)
    fi
    # if we couldn't find a fraction but have percent, derive fraction as 100/percent
    if [ -z "$ratio_frac" ] && [ -n "$ratio_pct" ]; then
        if awk "BEGIN {exit !($ratio_pct+0 > 0)}"; then
            local computed_frac=$(awk "BEGIN {printf \"%.2f\", 100.0 / $ratio_pct}")
            ratio_frac="${computed_frac}:1"
        fi
    fi
    # final fallback: try a simpler numeric fallback on the "Compression ratio" line
    if [ -z "$ratio_frac" ]; then
        ratio_frac=$(echo "$output" | grep -oP 'Compression ratio\s*:\s*\K[0-9.]+' | head -1 || true)
        if [ -n "$ratio_frac" ]; then
            ratio_frac="${ratio_frac}:1"
        fi
    fi
    # expansion ratio fallback for percent if still empty
    if [ -z "$ratio_pct" ]; then
        ratio_pct=$(echo "$output" | grep -oP 'Expansion ratio.*:\s*\K[0-9.]+' | head -1 || true)
    fi

    # kernel metadata: kernel name, workgroups
    local kernel_name=$(echo "$output" | grep -oP '^\s*Kernel\s*:\s*\K[^\s]+' | head -1)
    if [ -z "$kernel_name" ]; then
        kernel_name=$(echo "$output" | grep -oP '^Kernel\s*:\s*\K[^\s]+' | head -1)
    fi
    # Work groups
    local global_size=$(echo "$output" | grep -oP 'Work groups\s*:\s*global=\K[\d]+' | head -1)
    local local_size=$(echo "$output" | grep -oP 'Work groups\s*:\s*.*local=\K[\d]+' | head -1)

    # buffer allocs: in/out/len
    local buffer_in=$(echo "$output" | grep -oP 'Buffer Alloc \(in\)\s*:\s*\K[0-9.]+' | head -1)
    local buffer_out=$(echo "$output" | grep -oP 'Buffer Alloc \(out\)\s*:\s*\K[0-9.]+' | head -1)
    local buffer_len=$(echo "$output" | grep -oP 'Buffer Alloc \(len\)\s*:\s*\K[0-9.]+' | head -1)
    # fallback: single 'Buffer Alloc' generic timing
    if [ -z "$buffer_in" ]; then buffer_in=$(echo "$output" | grep -oP 'Buffer Alloc\s*:\s*\K[0-9.]+' | head -1); fi
    if [ -z "$buffer_out" ]; then buffer_out=$buffer_in; fi
    if [ -z "$buffer_len" ]; then buffer_len=$buffer_in; fi

    # blocking calc & download len/bulk
    local block_calc=$(echo "$output" | grep -oP 'Blocking Calc\s*:\s*\K[0-9.]+' | head -1)
    local dl_len=$(echo "$output" | grep -oP 'Download \(len\)\s*:\s*\K[0-9.]+' | head -1)
    local dl_bulk=$(echo "$output" | grep -oP 'Download \(bulk\)\s*:\s*\K[0-9.]+' | head -1)
    local dl_total=$(echo "$output" | grep -oP 'Download Total\s*:\s*\K[0-9.]+' | head -1)

    # defaults
    total_tp=${total_tp:-0}; kernel_tp=${kernel_tp:-0}; total_time=${total_time:-0}; kernel_time=${kernel_time:-0}
    read_time=${read_time:-0}; write_time=${write_time:-0}; upload_time=${upload_time:-0}; download_time=${download_time:-0}; ratio_frac=${ratio_frac:-NA}; ratio_pct=${ratio_pct:-0}
    kernel_name=${kernel_name:-""}; global_size=${global_size:-0}; local_size=${local_size:-0}
    buffer_in=${buffer_in:-0}; buffer_out=${buffer_out:-0}; buffer_len=${buffer_len:-0}; block_calc=${block_calc:-0}
    dl_len=${dl_len:-0}; dl_bulk=${dl_bulk:-0}; dl_total=${dl_total:-0}

    # Output: all fields comma-separated
    echo "${total_tp},${kernel_tp},${total_time},${kernel_time},${read_time},${write_time},${upload_time},${download_time},${ratio_frac},${ratio_pct},${kernel_name},${global_size},${local_size},${buffer_in},${buffer_out},${buffer_len},${block_calc},${dl_len},${dl_bulk},${dl_total}"
}

verify_output() {
    local orig="$1"
    local out="$2"
    if [ -f "$out" ]; then
        local orig_md5=$(md5sum "$orig" | awk '{print $1}')
        local out_md5=$(md5sum "$out" | awk '{print $1}')
        [ "$orig_md5" = "$out_md5" ] && echo "YES" || echo "NO"
    else
        echo "NO"
    fi
}

# ========================================
# 初始化 CSV
# ========================================
init_csv() {
    mkdir -p "$RESULTS_DIR"
    echo "sample,size_mb,tool,config,mode,algorithm,block_kb,threads,mt_threads,mt_io,copy_mode,coalesce,stdio_buf_mb,ratio_frac,ratio_pct,total_time_ms,kernel_time_ms,read_time_ms,write_time_ms,upload_time_ms,download_time_ms,kernel_name,global_size,local_size,buffer_alloc_in_ms,buffer_alloc_out_ms,buffer_alloc_len_ms,blocking_calc_ms,download_len_ms,download_bulk_ms,download_total_ms,total_throughput_mbps,kernel_throughput_mbps,verified" > "$CSV_FILE"
}

# ========================================
# CPU 测试
# ========================================
test_cpu() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sample_size_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)

    echo -e "\n${CYAN}=== CPU Tests: $sample_name ===${NC}"

    for threads in "${CPU_THREADS[@]}"; do
        for alg in "${ALGORITHMS[@]}"; do
            local config="cpu_t${threads}_${alg}"
            local tmp_lzo="/tmp/pscan_${sample_name}.lzo"
            local tmp_out="/tmp/pscan_${sample_name}.out"

            # 压缩
            local comp_out=$("$LZO_CPU" -t "$threads" -a "$alg" "$sample" -o "$tmp_lzo" 2>&1)
            local comp_perf=$(extract_perf "$comp_out")

            # 解压
            local decomp_out=$("$LZO_CPU" -t "$threads" -d "$tmp_lzo" -o "$tmp_out" 2>&1)
            local decomp_perf=$(extract_perf "$decomp_out")

            local verified=$(verify_output "$sample" "$tmp_out")

            # 解析压缩性能
            IFS=',' read -r c_tp c_ktp c_tt c_kt c_rt c_wt c_ut c_dt c_ratio_frac c_ratio_pct c_kernel_name c_global c_local c_buf_in c_buf_out c_buf_len c_block_calc c_dl_len c_dl_bulk c_dl_total <<< "$comp_perf"
            # Ensure numeric fields are explicit to avoid empty fields shifting CSV columns
            echo "$sample_name,$sample_size_mb,CPU,$config,compress,$alg,NA,$threads,NA,NA,NA,NA,NA,${c_ratio_frac:-NA},${c_ratio_pct:-0},${c_tt:-0},${c_kt:-0},${c_rt:-0},${c_wt:-0},${c_ut:-0},${c_dt:-0},${c_kernel_name:-},${c_global:-0},${c_local:-0},${c_buf_in:-0},${c_buf_out:-0},${c_buf_len:-0},${c_block_calc:-0},${c_dl_len:-0},${c_dl_bulk:-0},${c_dl_total:-0},${c_tp:-0},${c_ktp:-0},${verified:-NO}" >> "$CSV_FILE"

            # 解析解压性能
            IFS=',' read -r d_tp d_ktp d_tt d_kt d_rt d_wt d_ut d_dt d_ratio_frac d_ratio_pct d_kernel_name d_global d_local d_buf_in d_buf_out d_buf_len d_block_calc d_dl_len d_dl_bulk d_dl_total <<< "$decomp_perf"
            echo "$sample_name,$sample_size_mb,CPU,$config,decompress,$alg,NA,$threads,NA,NA,NA,NA,NA,${d_ratio_frac:-NA},${d_ratio_pct:-0},${d_tt:-0},${d_kt:-0},${d_rt:-0},${d_wt:-0},${d_ut:-0},${d_dt:-0},${d_kernel_name:-},${d_global:-0},${d_local:-0},${d_buf_in:-0},${d_buf_out:-0},${d_buf_len:-0},${d_block_calc:-0},${d_dl_len:-0},${d_dl_bulk:-0},${d_dl_total:-0},${d_tp:-0},${d_ktp:-0},${verified:-NO}" >> "$CSV_FILE"

            local status="✓"
            local status_color="$GREEN"
            if [ "$verified" != "YES" ]; then
                status="✗"
                status_color="$RED"
                local fail_dir="$RESULTS_DIR/failures/${sample_name}/${config}"
                mkdir -p "$fail_dir"
                [ -f "$tmp_lzo" ] && mv "$tmp_lzo" "$fail_dir/failed.lzo"
                [ -f "$tmp_out" ] && mv "$tmp_out" "$fail_dir/failed.out"
            else
                rm -f "$tmp_lzo" "$tmp_out"
            fi
            printf "  %-20s " "$config"
            echo -en "${status_color}${status}${NC} "
            printf "COMP: %7.1f MB/s (k:%7.1f) [R: %s (%s%%)] | DECOMP: %7.1f MB/s (k:%7.1f)\n" "${c_tp:-0}" "${c_ktp:-0}" "${c_ratio_frac:-NA}" "${c_ratio_pct:-0}" "${d_tp:-0}" "${d_ktp:-0}"
        done
    done
}

# ========================================
# GPU Standalone 测试
# ========================================
test_gpu_standalone() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sample_size_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)

    echo -e "\n${CYAN}=== GPU Standalone Tests: $sample_name ===${NC}"

    for alg in "${ALGORITHMS[@]}"; do
        for lvl in "${LEVELS[@]}"; do
            for blk_kb in "${BLOCK_SIZES[@]}"; do
                for lsz in "${LOCAL_SIZES[@]}"; do
                    for mt_io in "${MT_IO_OPTIONS[@]}"; do
                        for copy_mode in "${COPY_MODES[@]}"; do
                            for mt_threads in "${MT_IO_THREADS_OPTIONS[@]}"; do
local io_mode="zerocopy"; [ "$copy_mode" = "1" ] && io_mode="stdcopy"
                local mt_tag="single"; [ "$mt_io" = "1" ] && mt_tag="mt"
                local config="gpu_${alg}_lvl${lvl}_blk${blk_kb}k_local${lsz}_${io_mode}_${mt_tag}"

                                local tmp_lzo="/tmp/pscan_${sample_name}.lzo"
                                local tmp_out="/tmp/pscan_${sample_name}.out"

                                # Base env for this test - mt_io may be 0/1
                                local env_vars="LZO_MT_IO=${mt_io} LZO_STANDARD_COPY=${copy_mode}"
                                if [ "${mt_io}" -eq 1 ]; then
                                    env_vars="$env_vars LZO_MT_IO_THREADS=${mt_threads}"
                                else
                                    # For reporting purposes, record NA when mt_io is disabled
                                    mt_threads="NA"
                                fi

                                for coalesce in "${COALESCE_OPTIONS[@]}"; do
                                    for stdio_buf in "${STDIO_BUF_OPTIONS[@]}"; do
                                        local env_vars_ext="$env_vars LZO_COALESCE_OUTPUT=${coalesce} LZO_STDIO_BUF_MB=${stdio_buf}"
                                        local config_ext="${config}_mt${mt_threads}_co${coalesce}_sbuf${stdio_buf}"

                                            # If min block size smaller that default 64KB requested, allow via env override
                                            if [ "${blk_kb}" -lt 64 ]; then
                                                env_vars_ext="$env_vars_ext LZO_MIN_BLOCK_SIZE=${blk_kb}KB"
                                            fi

                                            # For compression, local must be 1; for decompression, use requested local
                                            local cli_args_comp="-a ${alg} -L ${lvl} -B ${blk_kb}KB --local 1"
                                            local cli_args_decomp="-a ${alg} -L ${lvl} -B ${blk_kb}KB --local ${lsz}"

                                            # 压缩
                                            local comp_out
                                            comp_out=$(cd "$LZO_GPU_DIR" && env $env_vars_ext "$LZO_GPU" ${cli_args_comp} "$sample" -o "$tmp_lzo" 2>&1)
                                            local comp_perf
                                            comp_perf=$(extract_perf "$comp_out")

                                            # parse comp_perf into array and then assign
                                            IFS=',' read -ra CPERF_ARR <<< "$comp_perf"
                                            local c_tp="${CPERF_ARR[0]}"
                                            local c_ktp="${CPERF_ARR[1]}"
                                            local c_tt="${CPERF_ARR[2]}"
                                            local c_kt="${CPERF_ARR[3]}"
                                            local c_rt="${CPERF_ARR[4]}"
                                            local c_wt="${CPERF_ARR[5]}"
                                            local c_ut="${CPERF_ARR[6]}"
                                            local c_dt="${CPERF_ARR[7]}"
                                            local c_ratio_frac="${CPERF_ARR[8]}"
                                            local c_ratio_pct="${CPERF_ARR[9]}"
                                            local c_kernel_name="${CPERF_ARR[10]}"
                                            local c_global="${CPERF_ARR[11]}"
                                            local c_local="${CPERF_ARR[12]}"
                                            local c_buf_in="${CPERF_ARR[13]}"
                                            local c_buf_out="${CPERF_ARR[14]}"
                                            local c_buf_len="${CPERF_ARR[15]}"
                                            local c_block_calc="${CPERF_ARR[16]}"
                                            local c_dl_len="${CPERF_ARR[17]}"
                                            local c_dl_bulk="${CPERF_ARR[18]}"
                                            local c_dl_total="${CPERF_ARR[19]}"

                                            # 解压
                                            local decomp_out
                                            decomp_out=$(cd "$LZO_GPU_DIR" && env $env_vars_ext "$LZO_GPU" ${cli_args_decomp} -d "$tmp_lzo" -o "$tmp_out" 2>&1)
                                            local decomp_perf
                                            decomp_perf=$(extract_perf "$decomp_out")
                                            IFS=',' read -ra DPERF_ARR <<< "$decomp_perf"
                                            local d_tp="${DPERF_ARR[0]}"
                                            local d_ktp="${DPERF_ARR[1]}"
                                            local d_tt="${DPERF_ARR[2]}"
                                            local d_kt="${DPERF_ARR[3]}"
                                            local d_rt="${DPERF_ARR[4]}"
                                            local d_wt="${DPERF_ARR[5]}"
                                            local d_ut="${DPERF_ARR[6]}"
                                            local d_dt="${DPERF_ARR[7]}"
                                            local d_ratio_frac="${DPERF_ARR[8]}"
                                            local d_ratio_pct="${DPERF_ARR[9]}"
                                            local d_kernel_name="${DPERF_ARR[10]}"
                                            local d_global="${DPERF_ARR[11]}"
                                            local d_local="${DPERF_ARR[12]}"
                                            local d_buf_in="${DPERF_ARR[13]}"
                                            local d_buf_out="${DPERF_ARR[14]}"
                                            local d_buf_len="${DPERF_ARR[15]}"
                                            local d_block_calc="${DPERF_ARR[16]}"
                                            local d_dl_len="${DPERF_ARR[17]}"
                                            local d_dl_bulk="${DPERF_ARR[18]}"
                                            local d_dl_total="${DPERF_ARR[19]}"

                                            local verified
                                            verified=$(verify_output "$sample" "$tmp_out")

                                            # Write CSV lines including mt_threads
                                            echo "$sample_name,$sample_size_mb,GPU,$config_ext,compress,$alg,$blk_kb,NA,$mt_threads,$mt_io,$copy_mode,$coalesce,$stdio_buf,$c_ratio_frac,$c_ratio_pct,$c_tt,$c_kt,$c_rt,$c_wt,$c_ut,$c_dt,${c_kernel_name:-},${c_global:-},${c_local:-},${c_buf_in:-},${c_buf_out:-},${c_buf_len:-},${c_block_calc:-},${c_dl_len:-},${c_dl_bulk:-},${c_dl_total:-},$c_tp,$c_ktp,$verified" >> "$CSV_FILE"

                                            echo "$sample_name,$sample_size_mb,GPU,$config_ext,decompress,$alg,$blk_kb,NA,$mt_threads,$mt_io,$copy_mode,$coalesce,$stdio_buf,$d_ratio_frac,$d_ratio_pct,$d_tt,$d_kt,$d_rt,$d_wt,$d_ut,$d_dt,${d_kernel_name:-},${d_global:-},${d_local:-},${d_buf_in:-},${d_buf_out:-},${d_buf_len:-},${d_block_calc:-},${d_dl_len:-},${d_dl_bulk:-},${d_dl_total:-},$d_tp,$d_ktp,$verified" >> "$CSV_FILE"

                                            local status="✓"
                                            local status_color="$GREEN"
                                            if [ "$verified" != "YES" ]; then
                                                status="✗"
                                                status_color="$RED"
                                                local fail_dir="$RESULTS_DIR/failures/${sample_name}/${config_ext}"
                                                mkdir -p "$fail_dir"
                                                [ -f "$tmp_lzo" ] && mv "$tmp_lzo" "$fail_dir/failed.lzo"
                                                [ -f "$tmp_out" ] && mv "$tmp_out" "$fail_dir/failed.out"
                                            else
                                                rm -f "$tmp_lzo" "$tmp_out"
                                            fi
                                            printf "  %-45s " "$config_ext"
                                            echo -en "${status_color}${status}${NC} "
                                            printf "C:%6.1f K:%7.1f [R: %s (%s%%)] | D:%6.1f K:%7.1f\n" "${c_tp:-0}" "${c_ktp:-0}" "${c_ratio_frac:-NA}" "${c_ratio_pct:-NA}" "${d_tp:-0}" "${d_ktp:-0}"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
}

# ========================================
# 生成报告
# ========================================
generate_report() {
    {
        echo "========================================"
        echo "   LZO Parameter Scan Report"
        echo "   $(date)"
        echo "========================================"
        echo ""

        echo "=== Configuration Summary ==="
        echo "Algorithms: ${ALGORITHMS[*]}"
        echo "Block sizes: ${BLOCK_SIZES[*]} KB"
        echo "CPU threads: ${CPU_THREADS[*]}"
        echo "MT IO: ${MT_IO_OPTIONS[*]} (0=off, 1=on)"
        echo "Copy modes: ${COPY_MODES[*]} (0=zerocopy, 1=stdcopy)"
        echo ""

        # Map CSV header names to column indices so AWK can use the right fields even when rows vary
        IDX_TP=$(head -n1 "$CSV_FILE" | awk -F, '{for(i=1;i<=NF;i++) if ($i=="total_throughput_mbps") print i}')
        IDX_KTP=$(head -n1 "$CSV_FILE" | awk -F, '{for(i=1;i<=NF;i++) if ($i=="kernel_throughput_mbps") print i}')
        IDX_VER=$(head -n1 "$CSV_FILE" | awk -F, '{for(i=1;i<=NF;i++) if ($i=="verified") print i}')

        echo "=== Best Compression Configurations ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "compress" {
            sample=$1; tool=$3; config=$4; tp=$(NF-2); ktp=$(NF-1)
            key=sample"|"tool
            if (tp+0 > best_tp[key]+0 || best_tp[key] == "") {
                best_tp[key] = tp
                best_ktp[key] = ktp
                best_config[key] = config
            }
        }
        END {
            printf "%-40s | %-8s | %-45s | %12s | %12s\n", "Sample", "Tool", "Best Config", "Total MB/s", "Kernel MB/s"
            printf "%-40s-+-%-8s-+-%-45s-+-%12s-+-%12s\n", "----------------------------------------", "--------", "---------------------------------------------", "------------", "------------"
            for (key in best_tp) {
                split(key, parts, "|")
                printf "%-40s | %-8s | %-45s | %12.2f | %12.2f\n", parts[1], parts[2], best_config[key], best_tp[key], best_ktp[key]
            }
        }'
        echo ""

        echo "=== Best Decompression Configurations ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "decompress" {
            sample=$1; tool=$3; config=$4; tp=$(NF-2); ktp=$(NF-1)
            key=sample"|"tool
            if (tp+0 > best_tp[key]+0 || best_tp[key] == "") {
                best_tp[key] = tp
                best_ktp[key] = ktp
                best_config[key] = config
            }
        }
        END {
            printf "%-40s | %-8s | %-45s | %12s | %12s\n", "Sample", "Tool", "Best Config", "Total MB/s", "Kernel MB/s"
            printf "%-40s-+-%-8s-+-%-45s-+-%12s-+-%12s\n", "----------------------------------------", "--------", "---------------------------------------------", "------------", "------------"
            for (key in best_tp) {
                split(key, parts, "|")
                printf "%-40s | %-8s | %-45s | %12.2f | %12.2f\n", parts[1], parts[2], best_config[key], best_tp[key], best_ktp[key]
            }
        }'
        echo ""

        echo "=== Block Size Analysis (GPU Compression) ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $3 == "GPU" && $5 == "compress" && $7 != "NA" {
            blk=$7; tp=$(NF-2); ktp=$(NF-1)
            count[blk]++
            sum_tp[blk] += tp
            sum_ktp[blk] += ktp
        }
        END {
            printf "%-10s | %15s | %15s | %8s\n", "Block KB", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-10s-+-%15s-+-%15s-+-%8s\n", "----------", "---------------", "---------------", "--------"
            for (blk in count) {
                printf "%-10s | %15.2f | %15.2f | %8d\n", blk, sum_tp[blk]/count[blk], sum_ktp[blk]/count[blk], count[blk]
            }
        }'
        echo ""

        echo "=== Algorithm Analysis ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "compress" {
            alg=$6; tool=$3; tp=$(NF-2); ktp=$(NF-1)
            key=alg"|"tool
            count[key]++
            sum_tp[key] += tp
            sum_ktp[key] += ktp
        }
        END {
            printf "%-6s | %-8s | %15s | %15s | %8s\n", "Alg", "Tool", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-6s-+-%-8s-+-%15s-+-%15s-+-%8s\n", "------", "--------", "---------------", "---------------", "--------"
            for (key in count) {
                split(key, parts, "|")
                printf "%-6s | %-8s | %15.2f | %15.2f | %8d\n", parts[1], parts[2], sum_tp[key]/count[key], sum_ktp[key]/count[key], count[key]
            }
        }'
        echo ""

        echo "=== IO Optimization Analysis (GPU) ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $3 == "GPU" && $5 == "compress" {
            mt=$10; copy=$11
            if (mt == "NA") mt = "0"
            if (copy == "NA") copy = "0"
            key=mt"|"copy
            count[key]++
            sum_tp[key] += $(NF-2)
            sum_ktp[key] += $(NF-1)
        }
        END {
            printf "%-6s | %-8s | %15s | %15s | %8s\n", "MT_IO", "CopyMode", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-6s-+-%-8s-+-%15s-+-%15s-+-%8s\n", "------", "--------", "---------------", "---------------", "--------"
            for (k in count) {
                split(k, parts, "|")
                mt = parts[1] == "0" ? "single" : "mt"
                copy = parts[2] == "0" ? "zerocopy" : "stdcopy"
                printf "%-6s | %-8s | %15.2f | %15.2f | %8d\n", mt, copy, sum_tp[k]/count[k], sum_ktp[k]/count[k], count[k]
            }
        }'
        echo ""

        echo "===  Analysis (Decompression) ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "decompress" && $9 != "NA" {
            tool=$3; tp=$(NF-2); ktp=$(NF-1)
            key=tool
            count[key]++
            sum_tp[key] += tp
            sum_ktp[key] += ktp
        }
        END {
            printf "%-8s | %15s | %15s | %8s\n", "Tool", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-8s-+-%-8s-+-%15s-+-%15s-+-%8s\n", "--------", "--------", "---------------", "---------------", "--------"
            for (key in count) {
                split(key, parts, "|")
                printf " %-8s | %15.2f | %15.2f | %8d\n", parts[2], sum_tp[key]/count[key], sum_ktp[key]/count[key], count[key]
            }
        }'
        echo ""

        echo "=== CPU Thread Scaling ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $3 == "CPU" && $5 == "compress" && $8 != "NA" {
            threads=$8; tp=$(NF-2); ktp=$(NF-1)
            count[threads]++
            sum_tp[threads] += tp
            sum_ktp[threads] += ktp
        }
        END {
            printf "%-8s | %15s | %15s | %8s\n", "Threads", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-8s-+-%15s-+-%15s-+-%8s\n", "--------", "---------------", "---------------", "--------"
            for (t in count) {
                printf "%-8d | %15.2f | %15.2f | %8d\n", t, sum_tp[t]/count[t], sum_ktp[t]/count[t], count[t]
            }
        }'
        echo ""

        echo "=== Verification Summary ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        { verified[$NF]++ }
        END {
            total = verified["YES"] + verified["NO"]
            if (total > 0)
                printf "Passed: %d / %d (%.1f%%)\n", verified["YES"], total, 100.0 * verified["YES"] / total
            if (verified["NO"] > 0)
                printf "FAILED: %d tests\n", verified["NO"]
        }'
        echo ""

        echo "=== Overall Performance Summary ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "compress" {
            tool=$3; tp=$(NF-2); ktp=$(NF-1)
            count[tool]++
            sum_tp[tool] += tp
            sum_ktp[tool] += ktp
            if (tp+0 > max_tp[tool]+0) { max_tp[tool] = tp; max_config[tool] = $4 }
        }
        END {
            printf "\n%-10s | %12s | %12s | %12s | %s\n", "Tool", "Avg MB/s", "Avg Kernel", "Max MB/s", "Best Config"
            printf "%-10s-+-%12s-+-%12s-+-%12s-+-%s\n", "----------", "------------", "------------", "------------", "--------------------------------------------"
            for (t in count) {
                printf "%-10s | %12.2f | %12.2f | %12.2f | %s\n", t, sum_tp[t]/count[t], sum_ktp[t]/count[t], max_tp[t], max_config[t]
            }
        }'

    } | tee "$REPORT_FILE"

    echo ""
    echo "Results saved to: $CSV_FILE"
    echo "Report saved to: $REPORT_FILE"
}

# ========================================
# 快速扫描模式
# ========================================
quick_scan() {
    echo "=== Quick Parameter Scan Mode ==="
    init_csv

    # 减少配置组合
    ALGO_TYPES=("1x" "1y")
    LEVELS=(10 11 12 14)
    BLOCK_SIZES=(8 16 64)  # KB
    LOCAL_SIZES=(1)
    CPU_THREADS=(1 2)
    COPY_MODES=(0)  # 0=zero-copy, 1=standard copy
    STDIO_BUF_OPTIONS=(4)  # stdio buffer sizes in MB
    MT_IO_OPTIONS=(1)  # multi-threaded IO
    MT_IO_THREADS_OPTIONS=(2) # MT_IO thread counts
    COALESCE_OPTIONS=(1)  # enable/disable output coalescing

    local samples=()
    # collect all files (regular files + symlinks) in SAMPLES_DIR
    while IFS= read -r -d '' f; do
        [ -f "$f" -o -L "$f" ] && samples+=("$f")
    done < <(find "$SAMPLES_DIR" \( -type f -o -type l \) -print0)

    for sample in "${samples[@]}"; do
        test_cpu "$sample"
        test_gpu_standalone "$sample"
        test_gpu_daemon "$sample"
    done

    generate_report
}

# ========================================
# 中等规模扫描模式 - 更全面的配置
# ========================================
medium_scan() {
    echo "=== Medium Parameter Scan Mode ==="
    echo "Testing more configurations..."
    init_csv

    # 配置组合
    ALGO_TYPES=("1x" "1y")
    LEVELS=(10 11 12 14)
    BLOCK_SIZES=(4 8 16 32 64)  # KB
    CPU_THREADS=(1 2)
    LOCAL_SIZES=(1)
    COPY_MODES=(0 1)
    COALESCE_OPTIONS=(0 1)
    STDIO_BUF_OPTIONS=(4 32)
    MT_IO_THREADS_OPTIONS=(1 2)

    local samples=()
    # collect all files (regular files + symlinks) in SAMPLES_DIR
    while IFS= read -r -d '' f; do
        [ -f "$f" -o -L "$f" ] && samples+=("$f")
    done < <(find "$SAMPLES_DIR" \( -type f -o -type l \) -print0)

    local total=${#samples[@]}
    local idx=0

    for sample in "${samples[@]}"; do
        idx=$((idx + 1))
        echo -e "\n${BLUE}[${idx}/${total}] Processing: $(basename "$sample")${NC}"
        test_cpu "$sample"
        test_gpu_standalone "$sample"
        test_gpu_daemon "$sample"
    done

    generate_report
}

# ========================================
# 完整扫描
# ========================================
full_scan() {
    echo "=== Full Parameter Scan ==="
    echo "Warning: This will take a long time!"
    echo ""
    init_csv

    local samples=()
    # collect all files (regular files + symlinks) in SAMPLES_DIR
    while IFS= read -r -d '' f; do
        [ -f "$f" -o -L "$f" ] && samples+=("$f")
    done < <(find "$SAMPLES_DIR" \( -type f -o -type l \) -print0)

    local total=${#samples[@]}
    local idx=0

    for sample in "${samples[@]}"; do
        idx=$((idx + 1))
        echo -e "\n${BLUE}[${idx}/${total}] Processing: $(basename "$sample")${NC}"
        test_cpu "$sample"
        test_gpu_standalone "$sample"
        test_gpu_daemon "$sample"
    done

    generate_report
}

# ========================================
# 主程序
# ========================================
case "${1:-}" in
    -q|--quick)
        quick_scan
        ;;
    -D|--daemon-only)
        daemon_only_scan
        ;;
    -m|--medium)
        medium_scan
        ;;
    -f|--full)
        full_scan
        ;;
    -r|--report-only)
        generate_report
        ;;
    -h|--help)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  -q, --quick    Quick scan with reduced configurations (default)"
        echo "  -m, --medium   Medium scan with more configurations (recommended)"
        echo "  -f, --full     Full parameter scan (takes a long time)"
        echo "  -r, --report-only  Generate a report from an existing CSV (override via CSV_FILE env)"
        echo "  -h, --help     Show this help"
        echo ""
        echo "Environment:"
        echo "  SAMPLES_DIR    Directory containing test samples (default: /root/samples)"
        ;;
    *)
        quick_scan
        ;;
esac
