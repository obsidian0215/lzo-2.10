#!/bin/bash
# LZO GPU/CPU 完整参数扫描测试脚本
# 测试各种 blocksize, localsize, 算法, IO优化选项等

set -e

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
SAMPLES_DIR="${SAMPLES_DIR:-$HOME/samples}"

# 结果文件
RESULTS_DIR="$REPO_DIR/exp_results/param_scan"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/param_scan_${TIMESTAMP}.csv"
REPORT_FILE="$RESULTS_DIR/param_scan_report_${TIMESTAMP}.txt"

# 测试配置
ALGORITHMS=("1" "1k" "1l" "1o")
BLOCK_SIZES=(32 64 128 256 512)  # KB
CPU_THREADS=(1 2 4 8)
VEC_MODES=(0 1)  # 0=scalar, 1=vectorized
MT_IO_OPTIONS=(0 1)  # multi-threaded IO
COPY_MODES=(0 1)  # 0=zero-copy, 1=standard copy

# daemon 相关
DAEMON_PID=""

mkdir -p "$RESULTS_DIR"

# ========================================
# 辅助函数
# ========================================

start_daemon() {
    local env_vars="${1:-}"
    stop_daemon 2>/dev/null || true

    cd "$LZO_GPU_DIR"
    if [ -n "$env_vars" ]; then
        env $env_vars "$LZO_GPU_DAEMON" > /tmp/lzo_daemon_scan.log 2>&1 &
    else
        "$LZO_GPU_DAEMON" > /tmp/lzo_daemon_scan.log 2>&1 &
    fi
    DAEMON_PID=$!

    # 等待 daemon 就绪
    for i in {1..30}; do
        if [ -S "/tmp/lzo_gpu_daemon.sock" ]; then
            sleep 0.2
            return 0
        fi
        sleep 0.2
    done

    echo -e "${RED}Failed to start daemon${NC}"
    return 1
}

stop_daemon() {
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
        DAEMON_PID=""
    fi
    pkill -f "lzo_gpu_daemon" 2>/dev/null || true
    rm -f /tmp/lzo_gpu_daemon.sock 2>/dev/null || true
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
    local kernel_tp=$(echo "$output" | grep -oP 'kernel:\s*\K[\d.]+' | head -1)
    local total_time=$(echo "$output" | grep -oP 'TOTAL\s*:\s*\K[\d.]+' | head -1)
    local kernel_time=$(echo "$output" | grep -oP 'Kernel Exec\s*:\s*\K[\d.]+' | head -1)
    local read_time=$(echo "$output" | grep -oP '1\. File Read\s*:\s*\K[\d.]+' | head -1)
    local write_time=$(echo "$output" | grep -oP '\d+\. File Write\s*:\s*\K[\d.]+' | head -1)
    local upload_time=$(echo "$output" | grep -oP '\d+\. Data Upload\s*:\s*\K[\d.]+' | head -1)
    local download_time=$(echo "$output" | grep -oP '\d+\. Data Download\s*:\s*\K[\d.]+' | head -1)
    local ratio=$(echo "$output" | grep -oP 'Compression ratio.*\(\K[\d.]+(?=%)' | head -1)
    [ -z "$ratio" ] && ratio=$(echo "$output" | grep -oP 'Compression ratio\s*:\s*\K[\d.]+' | head -1)

    echo "${total_tp:-0},${kernel_tp:-0},${total_time:-0},${kernel_time:-0},${read_time:-0},${write_time:-0},${upload_time:-0},${download_time:-0},${ratio:-0}"
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
    echo "sample,size_mb,tool,config,mode,algorithm,block_kb,threads,vec,mt_io,copy_mode,ratio,total_time_ms,kernel_time_ms,read_time_ms,write_time_ms,upload_time_ms,download_time_ms,total_throughput_mbps,kernel_throughput_mbps,verified" > "$CSV_FILE"
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
            local comp_out=$("$LZO_CPU" -t "$threads" -l "$alg" "$sample" -o "$tmp_lzo" 2>&1)
            local comp_perf=$(extract_perf "$comp_out")

            # 解压
            local decomp_out=$("$LZO_CPU" -t "$threads" -d "$tmp_lzo" -o "$tmp_out" 2>&1)
            local decomp_perf=$(extract_perf "$decomp_out")

            local verified=$(verify_output "$sample" "$tmp_out")

            # 解析压缩性能
            IFS=',' read -r c_tp c_ktp c_tt c_kt c_rt c_wt c_ut c_dt c_ratio <<< "$comp_perf"
            echo "$sample_name,$sample_size_mb,CPU,$config,compress,$alg,NA,$threads,NA,NA,NA,$c_ratio,$c_tt,$c_kt,$c_rt,$c_wt,$c_ut,$c_dt,$c_tp,$c_ktp,$verified" >> "$CSV_FILE"

            # 解析解压性能
            IFS=',' read -r d_tp d_ktp d_tt d_kt d_rt d_wt d_ut d_dt d_ratio <<< "$decomp_perf"
            echo "$sample_name,$sample_size_mb,CPU,$config,decompress,$alg,NA,$threads,NA,NA,NA,,$d_tt,$d_kt,$d_rt,$d_wt,$d_ut,$d_dt,$d_tp,$d_ktp,$verified" >> "$CSV_FILE"

            local status="✓"
            local status_color="$GREEN"
            if [ "$verified" != "YES" ]; then
                status="✗"
                status_color="$RED"
            fi
            printf "  %-20s " "$config"
            echo -en "${status_color}${status}${NC} "
            printf "COMP: %7.1f MB/s (k:%7.1f) | DECOMP: %7.1f MB/s (k:%7.1f)\n" "${c_tp:-0}" "${c_ktp:-0}" "${d_tp:-0}" "${d_ktp:-0}"

            rm -f "$tmp_lzo" "$tmp_out"
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
        for blk_kb in "${BLOCK_SIZES[@]}"; do
            for vec in "${VEC_MODES[@]}"; do
                for mt_io in "${MT_IO_OPTIONS[@]}"; do
                    for copy_mode in "${COPY_MODES[@]}"; do
                        local decomp_mode="vec"; [ "$vec" = "0" ] && decomp_mode="scalar"
                        local io_mode="zerocopy"; [ "$copy_mode" = "1" ] && io_mode="stdcopy"
                        local mt_tag="single"; [ "$mt_io" = "1" ] && mt_tag="mt"
                        local config="gpu_${alg}_${decomp_mode}_${blk_kb}k_${io_mode}_${mt_tag}"

                        local tmp_lzo="/tmp/pscan_${sample_name}.lzo"
                        local tmp_out="/tmp/pscan_${sample_name}.out"

                        local env_vars="LZO_FIXED_BLOCK_SIZE=$blk_kb LZO_DECOMP_VEC=$vec LZO_MT_IO=$mt_io LZO_STANDARD_COPY=$copy_mode"

                        # 压缩
                        local comp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU" -L "$alg" "$sample" -o "$tmp_lzo" 2>&1)
                        local comp_perf=$(extract_perf "$comp_out")

                        # 解压
                        local decomp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU" -d "$tmp_lzo" -o "$tmp_out" 2>&1)
                        local decomp_perf=$(extract_perf "$decomp_out")

                        local verified=$(verify_output "$sample" "$tmp_out")

                        # 解析压缩性能
                        IFS=',' read -r c_tp c_ktp c_tt c_kt c_rt c_wt c_ut c_dt c_ratio <<< "$comp_perf"
                        echo "$sample_name,$sample_size_mb,GPU,$config,compress,$alg,$blk_kb,NA,$vec,$mt_io,$copy_mode,$c_ratio,$c_tt,$c_kt,$c_rt,$c_wt,$c_ut,$c_dt,$c_tp,$c_ktp,$verified" >> "$CSV_FILE"

                        # 解析解压性能
                        IFS=',' read -r d_tp d_ktp d_tt d_kt d_rt d_wt d_ut d_dt d_ratio <<< "$decomp_perf"
                        echo "$sample_name,$sample_size_mb,GPU,$config,decompress,$alg,$blk_kb,NA,$vec,$mt_io,$copy_mode,,$d_tt,$d_kt,$d_rt,$d_wt,$d_ut,$d_dt,$d_tp,$d_ktp,$verified" >> "$CSV_FILE"

                        local status="✓"
                        local status_color="$GREEN"
                        if [ "$verified" != "YES" ]; then
                            status="✗"
                            status_color="$RED"
                        fi
                        printf "  %-45s " "$config"
                        echo -en "${status_color}${status}${NC} "
                        printf "C:%6.1f K:%7.1f | D:%6.1f K:%7.1f\n" "${c_tp:-0}" "${c_ktp:-0}" "${d_tp:-0}" "${d_ktp:-0}"

                        rm -f "$tmp_lzo" "$tmp_out"
                    done
                done
            done
        done
    done
}

# ========================================
# GPU Daemon 测试
# ========================================
test_gpu_daemon() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sample_size_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)

    echo -e "\n${CYAN}=== GPU Daemon Tests: $sample_name ===${NC}"

    for alg in "${ALGORITHMS[@]}"; do
        for blk_kb in "${BLOCK_SIZES[@]}"; do
            for vec in "${VEC_MODES[@]}"; do
                local decomp_mode="vec"; [ "$vec" = "0" ] && decomp_mode="scalar"
                local config="daemon_${alg}_${decomp_mode}_${blk_kb}k"

                local tmp_lzo="/tmp/pscan_${sample_name}.lzo"
                local tmp_out="/tmp/pscan_${sample_name}.out"

                # 启动 daemon
                local daemon_env="LZO_DECOMP_VEC=$vec LZO_FIXED_BLOCK_SIZE=$blk_kb"
                if ! start_daemon "$daemon_env"; then
                    echo -e "  ${RED}Daemon start failed for $config${NC}"
                    continue
                fi

                local env_vars="LZO_FIXED_BLOCK_SIZE=$blk_kb LZO_DECOMP_VEC=$vec"

                # 压缩
                local comp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU_CLIENT" -L "$alg" "$sample" -o "$tmp_lzo" 2>&1)
                local comp_perf=$(extract_perf "$comp_out")

                # 解压
                local decomp_out=$(cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU_CLIENT" -d "$tmp_lzo" -o "$tmp_out" 2>&1)
                local decomp_perf=$(extract_perf "$decomp_out")

                local verified=$(verify_output "$sample" "$tmp_out")

                stop_daemon

                # 解析压缩性能
                IFS=',' read -r c_tp c_ktp c_tt c_kt c_rt c_wt c_ut c_dt c_ratio <<< "$comp_perf"
                echo "$sample_name,$sample_size_mb,Daemon,$config,compress,$alg,$blk_kb,NA,$vec,NA,NA,$c_ratio,$c_tt,$c_kt,$c_rt,$c_wt,$c_ut,$c_dt,$c_tp,$c_ktp,$verified" >> "$CSV_FILE"

                # 解析解压性能
                IFS=',' read -r d_tp d_ktp d_tt d_kt d_rt d_wt d_ut d_dt d_ratio <<< "$decomp_perf"
                echo "$sample_name,$sample_size_mb,Daemon,$config,decompress,$alg,$blk_kb,NA,$vec,NA,NA,,$d_tt,$d_kt,$d_rt,$d_wt,$d_ut,$d_dt,$d_tp,$d_ktp,$verified" >> "$CSV_FILE"

                local status="✓"
                local status_color="$GREEN"
                if [ "$verified" != "YES" ]; then
                    status="✗"
                    status_color="$RED"
                fi
                printf "  %-35s " "$config"
                echo -en "${status_color}${status}${NC} "
                printf "COMP:%6.1f (k:%7.1f) | DECOMP:%6.1f (k:%7.1f)\n" "${c_tp:-0}" "${c_ktp:-0}" "${d_tp:-0}" "${d_ktp:-0}"

                rm -f "$tmp_lzo" "$tmp_out"
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
        echo "VEC modes: ${VEC_MODES[*]} (0=scalar, 1=vec)"
        echo "MT IO: ${MT_IO_OPTIONS[*]} (0=off, 1=on)"
        echo "Copy modes: ${COPY_MODES[*]} (0=zerocopy, 1=stdcopy)"
        echo ""

        echo "=== Best Compression Configurations ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "compress" {
            sample=$1; tool=$3; config=$4; tp=$19; ktp=$20
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
            sample=$1; tool=$3; config=$4; tp=$19; ktp=$20
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
            blk=$7; tp=$19; ktp=$20
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
            alg=$6; tool=$3; tp=$19; ktp=$20
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
            sum_tp[key] += $19
            sum_ktp[key] += $20
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

        echo "=== Vectorization Analysis (Decompression) ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $5 == "decompress" && $9 != "NA" {
            vec=$9; tool=$3; tp=$19; ktp=$20
            key=vec"|"tool
            count[key]++
            sum_tp[key] += tp
            sum_ktp[key] += ktp
        }
        END {
            printf "%-8s | %-8s | %15s | %15s | %8s\n", "Vec", "Tool", "Avg Total MB/s", "Avg Kernel MB/s", "Tests"
            printf "%-8s-+-%-8s-+-%15s-+-%15s-+-%8s\n", "--------", "--------", "---------------", "---------------", "--------"
            for (key in count) {
                split(key, parts, "|")
                vec = parts[1] == "0" ? "scalar" : "vector"
                printf "%-8s | %-8s | %15.2f | %15.2f | %8d\n", vec, parts[2], sum_tp[key]/count[key], sum_ktp[key]/count[key], count[key]
            }
        }'
        echo ""

        echo "=== CPU Thread Scaling ==="
        tail -n +2 "$CSV_FILE" | awk -F',' '
        $3 == "CPU" && $5 == "compress" && $8 != "NA" {
            threads=$8; tp=$19; ktp=$20
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
        { verified[$21]++ }
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
            tool=$3; tp=$19; ktp=$20
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
    ALGORITHMS=("1l")
    BLOCK_SIZES=(64 128)
    CPU_THREADS=(1 4)
    VEC_MODES=(1)
    MT_IO_OPTIONS=(0)
    COPY_MODES=(0)

    local samples=()
    for f in "$SAMPLES_DIR"/*_pages*.img; do
        [ -f "$f" ] && samples+=("$f")
        [ ${#samples[@]} -ge 2 ] && break
    done

    if [ ${#samples[@]} -eq 0 ]; then
        # 尝试txt样本
        for f in "$SAMPLES_DIR"/*.txt; do
            [ -f "$f" ] && samples+=("$f")
            [ ${#samples[@]} -ge 2 ] && break
        done
    fi

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

    # 更多配置组合
    ALGORITHMS=("1" "1k" "1l" "1o")
    BLOCK_SIZES=(32 64 128 256)
    CPU_THREADS=(1 2 4)
    VEC_MODES=(0 1)
    MT_IO_OPTIONS=(0)
    COPY_MODES=(0)

    local samples=()
    # 获取3个不同类型的样本
    for f in "$SAMPLES_DIR"/*_pages*.img; do
        [ -f "$f" ] && samples+=("$f")
        [ ${#samples[@]} -ge 2 ] && break
    done
    for f in "$SAMPLES_DIR"/*.txt; do
        [ -f "$f" ] && samples+=("$f")
        [ ${#samples[@]} -ge 3 ] && break
    done

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
    for f in "$SAMPLES_DIR"/*.txt; do
        [ -f "$f" ] && samples+=("$f")
    done
    for f in "$SAMPLES_DIR"/*_pages*.img; do
        [ -f "$f" ] && samples+=("$f")
    done

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
    -m|--medium)
        medium_scan
        ;;
    -f|--full)
        full_scan
        ;;
    -h|--help)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  -q, --quick    Quick scan with reduced configurations (default)"
        echo "  -m, --medium   Medium scan with more configurations (recommended)"
        echo "  -f, --full     Full parameter scan (takes a long time)"
        echo "  -h, --help     Show this help"
        echo ""
        echo "Environment:"
        echo "  SAMPLES_DIR    Directory containing test samples (default: ~/samples)"
        ;;
    *)
        quick_scan
        ;;
esac
