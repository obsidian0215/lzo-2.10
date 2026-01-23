#!/bin/bash
set -e

# ========================================
# 颜色定义
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ========================================
# 配置参数
# ========================================
RESULTS_DIR="/root/lzo-2.10/exp_results/param_scan"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

SAMPLES_DIR="${SAMPLES_DIR:-/root/samples}"
CSV_FILE="${CSV_FILE:-$RESULTS_DIR/param_scan_${TIMESTAMP}.csv}"

# Configuration Matrix (Arrays) - Use string for env-friendly overrides
ALGORITHMS=(${ALGORITHMS:-1x 1y})
LEVEL_NAMES=(${LEVEL_NAMES:-1k 1l 1 1o}) # for lzo_cpu: '1k' = level 11, '1l' = level 12, '1' = level 14, '1o' = level 15
# GPU uses numeric dictionary bit sizes (10-18). Specify GPU_LEVELS if you want to test different -L values.
GPU_LEVELS=(${GPU_LEVELS:-11 12 14 15})
BLOCK_SIZES=(${BLOCK_SIZES:-8 16 32 64}) # in KB
LOCAL_SIZES=(${LOCAL_SIZES:-1 4 8})
CPU_THREADS=(${CPU_THREADS:-1 2 4})

LZO_GPU_DIR="/root/lzo-2.10/lzo_gpu"
LZO_GPU="./lzo_gpu"
LZO_CPU="/root/lzo-2.10/lzo_cpu/lzo_cpu"

# ========================================
# 核心工具函数
# ========================================
extract_perf() {
    local output="$1"
    get_val() { echo "$output" | grep -oP "$1" | head -1 | awk '{print ($1==""?0:$1)}'; }

    local total_tp=$(get_val 'Throughput\s*:\s*\K[0-9.]+')
    local kernel_tp=$(get_val 'kernel\s*:\s*\K[0-9.]+')
    local total_time=$(get_val 'TOTAL\s*:\s*\K[0-9.]+')
    local kernel_time=$(get_val '(Kernel Exec|Compress|Decompress)\s*:\s*\K[0-9.]+')
    local d_total=$(get_val 'daemon\s*:\s*\K[0-9.]+')
    local t_read=$(get_val 'File Read\s*:\s*\K[0-9.]+')
    local t_write=$(get_val 'File Write\s*:\s*\K[0-9.]+')
    # Prefer explicit percentage inside parentheses (e.g., "(21.23% of original)")
    local ratio_pct=$(echo "$output" | grep -i 'Compression ratio' | grep -oP '[0-9]+(?:\\.[0-9]+)?(?=%)' | head -1 | awk '{print ($1==""?0:$1)}')
    if [ -z "$ratio_pct" ] || [ "$ratio_pct" = "0" ]; then
        # Fallback: extract multiplicative ratio (e.g., "4.71" from "4.71:1") and convert to percent
        local ratio_raw=$(echo "$output" | grep -oP '(?<=Compression ratio\s*:\s*)[0-9]+(?:\\.[0-9]+)?' | head -1)
        if [ -n "$ratio_raw" ] && [ "$ratio_raw" != "0" ]; then
            ratio_pct=$(echo "scale=8; 100.0 / $ratio_raw" | bc 2>/dev/null)
        else
            ratio_pct=0
        fi
    fi

    local t_io=$(echo "$t_read + $t_write" | bc 2>/dev/null || echo 0)
    local t_mem=0
    if (( $(echo "$d_total > 0" | bc -l 2>/dev/null || echo 0) )); then
        t_mem=$(echo "$d_total - $kernel_time" | bc 2>/dev/null || echo 0)
    else
        local t_alloc=$(get_val 'Buffer Alloc[^:]*:\s*\K[0-9.]+')
        local t_up=$(get_val 'Upload\s*:\s*\K[0-9.]+')
        local t_dl=$(get_val 'Download\s*:\s*\K[0-9.]+')
        t_mem=$(echo "$t_alloc + $t_up + $t_dl" | bc 2>/dev/null || echo 0)
    fi
    echo "$total_tp,$kernel_tp,$total_time,$kernel_time,$t_io,$t_mem,$ratio_pct"
}

verify_output() {
    local original="$1"
    local decompressed="$2"
    if [ ! -f "$decompressed" ]; then echo "FAIL"; return; fi
    if cmp -s "$original" "$decompressed"; then echo "YES"; else echo "NO"; fi
}

init_csv() {
    echo "sample,size_mb,tool,mode,alg,level,block_kb,lsz,ratio_pct,mbps_total,mbps_kernel,t_total,t_kernel,t_io,t_mem,verified" > "$CSV_FILE"
}

# ========================================
# 启动/停止守护进程
# ========================================
start_daemon() {
    local env_vars="${1:-}"
    local sock=$(echo "$env_vars" | grep -oP 'LZO_DAEMON_SOCKET=\K[^ ]+' || echo "/tmp/lzo_gpu_daemon.sock")
    if pgrep -f "lzo_gpu --daemon" > /dev/null; then pkill -f "lzo_gpu --daemon"; sleep 1; fi
    (cd "$LZO_GPU_DIR" && env $env_vars "$LZO_GPU" --daemon --verbose > /tmp/lzo_daemon.log 2>&1 &)
    for i in {1..10}; do if [ -S "$sock" ]; then return 0; fi; sleep 0.5; done
    return 1
}

# ========================================
# 测试逻辑
# ========================================
test_cpu() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sz_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)
    echo -e "\n${CYAN}=== CPU Tests: $sample_name ===${NC}"
    for blk in "${BLOCK_SIZES[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
    for lvl_n in "${LEVEL_NAMES[@]}"; do
        case "$lvl_n" in "1k") d_lvl="11" ;; "1l") d_lvl="12" ;; "1") d_lvl="14" ;; "1o") d_lvl="15" ;; *) d_lvl="$lvl_n" ;; esac
        for thr in "${CPU_THREADS[@]}"; do
            local cfg="cpu_${alg}_L${d_lvl}_B${blk}k_T${thr}"
            local tmp_lzo="/tmp/pscan.lzo"; local tmp_out="/tmp/pscan.out"
            local cout=$("$LZO_CPU" -a "$alg" -l "$lvl_n" -B "${blk}KB" -t "$thr" "$sample" -o "$tmp_lzo" 2>&1)
            local cp=$(extract_perf "$cout")
            local dout=$("$LZO_CPU" -a "$alg" -d -B "${blk}KB" -t "$thr" "$tmp_lzo" -o "$tmp_out" 2>&1)
            local dp=$(extract_perf "$dout")
            local ver=$(verify_output "$sample" "$tmp_out")
            IFS=',' read -r c_tp c_ktp c_tt c_kt c_tio c_tmem c_ratio <<< "$cp"
            IFS=',' read -r d_tp d_ktp d_tt d_kt d_tio d_tmem d_ratio <<< "$dp"
            echo "$sample_name,$sz_mb,CPU,compress,$alg,$d_lvl,$blk,$thr,$c_ratio,$c_tp,$c_ktp,$c_tt,$c_kt,$c_tio,$c_tmem,$ver" >> "$CSV_FILE"
            echo "$sample_name,$sz_mb,CPU,decompress,$alg,$d_lvl,$blk,$thr,$d_ratio,$d_tp,$d_ktp,$d_tt,$d_kt,$d_tio,$d_tmem,$ver" >> "$CSV_FILE"
            printf "  %-35s %b COMP:%6.1f (k:%7.1f) [R: %s%%] | DECOMP:%6.1f (k:%7.1f)\n" "$cfg" "$GREEN✓$NC" "$c_tp" "$c_ktp" "$c_ratio" "$d_tp" "$d_ktp"
            rm -f "$tmp_lzo" "$tmp_out"
        done
    done
    done
    done
}

test_gpu_standalone() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sz_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)
    echo -e "\n${CYAN}=== GPU Standalone: $sample_name ===${NC}"
    for blk in "${BLOCK_SIZES[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
    for lvl in "${GPU_LEVELS[@]}"; do
        for lsz in "${LOCAL_SIZES[@]}"; do
            local cfg="gpu_${alg}_L${lvl}_B${blk}k_lsz${lsz}"
            local tmp_lzo="/tmp/pscan.lzo"; local tmp_out="/tmp/pscan.out"
            local cout=$(cd "$LZO_GPU_DIR" && "$LZO_GPU" --verbose -a "$alg" -L "$lvl" -B "${blk}KB" --local "$lsz" "$sample" -o "$tmp_lzo" 2>&1)
            local cp=$(extract_perf "$cout")
            local dout=$(cd "$LZO_GPU_DIR" && "$LZO_GPU" --verbose -d -a "$alg" -L "$lvl" -B "${blk}KB" --local "$lsz" "$tmp_lzo" -o "$tmp_out" 2>&1)
            local dp=$(extract_perf "$dout")
            local ver=$(verify_output "$sample" "$tmp_out")
            IFS=',' read -r c_tp c_ktp c_tt c_kt c_tio c_tmem c_ratio <<< "$cp"
            IFS=',' read -r d_tp d_ktp d_tt d_kt d_tio d_tmem d_ratio <<< "$dp"
            echo "$sample_name,$sz_mb,GPU,compress,$alg,$lvl,$blk,$lsz,$c_ratio,$c_tp,$c_ktp,$c_tt,$c_kt,$c_tio,$c_tmem,$ver" >> "$CSV_FILE"
            echo "$sample_name,$sz_mb,GPU,decompress,$alg,$lvl,$blk,$lsz,$d_ratio,$d_tp,$d_ktp,$d_tt,$d_kt,$d_tio,$d_tmem,$ver" >> "$CSV_FILE"
            printf "  %-35s %b COMP:%6.1f (k:%7.1f) | DECOMP:%6.1f (k:%7.1f)\n" "$cfg" "$GREEN✓$NC" "$c_tp" "$c_ktp" "$d_tp" "$d_ktp"
            rm -f "$tmp_lzo" "$tmp_out"
        done
    done
    done
    done
}

test_gpu_daemon() {
    local sample="$1"
    local sample_name=$(basename "$sample")
    local sz_mb=$(echo "scale=2; $(stat -Lc%s "$sample") / 1048576" | bc)
    local sock="/tmp/lzo_daemon_$$.sock"
    start_daemon "LZO_DAEMON_SOCKET=$sock" || return 1
    echo -e "\n${CYAN}=== GPU Daemon: $sample_name ===${NC}"
    for blk in "${BLOCK_SIZES[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
    for lvl in "${GPU_LEVELS[@]}"; do
        for lsz in "${LOCAL_SIZES[@]}"; do
            local cfg="daemon_${alg}_L${lvl}_B${blk}k_lsz${lsz}"
            local tmp_lzo="/tmp/pscan.lzo"; local tmp_out="/tmp/pscan.out"
            local cout=$(cd "$LZO_GPU_DIR" && env LZO_DAEMON_SOCKET=$sock "$LZO_GPU" --use-daemon --verbose -a "$alg" -L "$lvl" -B "${blk}KB" --local "$lsz" "$sample" -o "$tmp_lzo" 2>&1 || true)
            local cp=$(extract_perf "$cout")
            local dout=$(cd "$LZO_GPU_DIR" && env LZO_DAEMON_SOCKET=$sock "$LZO_GPU" --use-daemon --verbose -d -a "$alg" -L "$lvl" -B "${blk}KB" --local "$lsz" "$tmp_lzo" -o "$tmp_out" 2>&1 || true)
            local dp=$(extract_perf "$dout")
            local ver=$(verify_output "$sample" "$tmp_out")
            IFS=',' read -r c_tp c_ktp c_tt c_kt c_tio c_tmem c_ratio <<< "$cp"
            IFS=',' read -r d_tp d_ktp d_tt d_kt d_tio d_tmem d_ratio <<< "$dp"
            echo "$sample_name,$sz_mb,Daemon,compress,$alg,$lvl,$blk,$lsz,$c_ratio,$c_tp,$c_ktp,$c_tt,$c_kt,$c_tio,$c_tmem,$ver" >> "$CSV_FILE"
            echo "$sample_name,$sz_mb,Daemon,decompress,$alg,$lvl,$blk,$lsz,$d_ratio,$d_tp,$d_ktp,$d_tt,$d_kt,$d_tio,$d_tmem,$ver" >> "$CSV_FILE"
            printf "  %-35s %b COMP:%6.1f | DECOMP:%6.1f\n" "$cfg" "$GREEN✓$NC" "$c_tp" "$d_tp"
            rm -f "$tmp_lzo" "$tmp_out"
        done
    done
    done
    done
    pkill -f "$sock" || true; rm -f "$sock"
}

merge_csv() {
    local cpu_csv="$1"
    local gpu_csv="$2"
    local out_csv="${3:-$RESULTS_DIR/param_scan_merged_$(date +%Y%m%d_%H%M%S).csv}"
    if [ ! -f "$cpu_csv" ]; then echo "ERROR: CPU CSV not found: $cpu_csv"; return 1; fi
    if [ ! -f "$gpu_csv" ]; then echo "ERROR: GPU CSV not found: $gpu_csv"; return 1; fi
    head -n 1 "$cpu_csv" > "$out_csv"
    awk -F, 'NR>1 && toupper($3)=="CPU" {print}' "$cpu_csv" >> "$out_csv"
    awk -F, 'NR>1 && toupper($3)!="CPU" {print}' "$gpu_csv" >> "$out_csv"
    echo "Wrote merged CSV -> $out_csv"
}

generate_report() {
    local csv="${1:-$CSV_FILE}"
    python3 - "$csv" <<'PY'
import sys, csv, os, re, statistics, math

# --- helpers ---

def to_float(s):
    try:
        return float(str(s).strip())
    except:
        try:
            s2 = re.sub(r'[^0-9+\-.eE]', '', str(s))
            return float(s2) if s2 else 0.0
        except:
            return 0.0

def to_int(s):
    try:
        return int(float(str(s).strip()))
    except:
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else None

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return xs[n//2]
    return 0.5 * (xs[n//2 - 1] + xs[n//2])

def stddev(xs):
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    try:
        return statistics.stdev(xs)
    except Exception:
        return 0.0

def summarize(xs, positive=False):
    vals = [float(x) for x in xs if x is not None and (x > 0 if positive else True)]
    n = len(vals)
    if n == 0:
        return {'n': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {'n': n, 'mean': statistics.mean(vals), 'median': median(vals), 'std': stddev(vals), 'min': min(vals), 'max': max(vals)}

# --- read CSV ---

def read_csv(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            tool = row.get('tool','').strip().upper()
            mode = row.get('mode','').strip().lower()
            level = row.get('level','').strip()
            level_int = to_int(level)
            bk = to_int(row.get('block_kb',''))
            lsz_raw = row.get('lsz','').strip()
            lsz_int = to_int(lsz_raw)
            rows.append({
                'sample': row.get('sample','').strip(),
                'size_mb': to_float(row.get('size_mb','0')),
                'tool': tool,
                'mode': mode,
                'alg': row.get('alg','').strip(),
                'level': level,
                'level_int': level_int,
                'block_kb': bk,
                'lsz_raw': lsz_raw,
                'lsz_int': lsz_int,
                'ratio_pct': to_float(row.get('ratio_pct','0')),
                'mbps_total': to_float(row.get('mbps_total','0')),
                'mbps_kernel': to_float(row.get('mbps_kernel','0')),
                't_total': to_float(row.get('t_total','0')),
                't_kernel': to_float(row.get('t_kernel','0')),
                't_io': to_float(row.get('t_io','0')),
                't_mem': to_float(row.get('t_mem','0')),
                'verified': row.get('verified','').strip()
            })
    return rows

path = sys.argv[1]
if not os.path.isfile(path):
    print('ERROR: CSV not found:', path); sys.exit(2)
rows = read_csv(path)
results_dir = os.path.dirname(path) or '.'
base = os.path.splitext(os.path.basename(path))[0]

# capture stdout so we can save the full textual report to a file at the end
import io
buf = io.StringIO()
old_stdout = sys.stdout
class Tee:
    def write(self, data):
        buf.write(data)
        old_stdout.write(data)
    def flush(self):
        old_stdout.flush()
sys.stdout = Tee()

# counts
print('='*120)
print(' ' * 40 + 'LZO PERFORMANCE - EXTENDED SENSITIVITY REPORT')
print('='*120)
print(f"Input CSV: {path} | total rows: {len(rows)} | verified rows: {sum(1 for r in rows if r.get('verified','').upper()=='YES')}")
print()

# --- Baselines ---
cpu_all = [d for d in rows if d['tool']=='CPU' and d['mode']=='compress']
gpu_all = [d for d in rows if d['tool']=='GPU' and d['mode']=='compress']
daemon_all = [d for d in rows if d['tool']=='DAEMON' and d['mode']=='compress']

print('[SECTION A] SYSTEM BASELINES (detailed)')
# CPU per-thread stats
cpu_thread_vals = sorted(set(d['lsz_int'] for d in cpu_all if d['lsz_int'] not in (None,0)))
if cpu_thread_vals:
    print('\nHost CPU thread breakdown:')
    print('  T   | rows | total med/mean (std)  | kernel med/mean (std)')
    print(' -----|------|----------------------|----------------------')
    for t in cpu_thread_vals:
        ds = [d for d in cpu_all if d['lsz_int']==t]
        tot = summarize([d['mbps_total'] for d in ds], positive=True)
        ker = summarize([d['mbps_kernel'] for d in ds], positive=True)
        print(f"  T{t:<3} | {tot['n']:4d} | {tot['median']:6.1f}/{tot['mean']:6.1f} ({tot['std']:5.1f}) | {ker['median']:6.1f}/{ker['mean']:6.1f} ({ker['std']:5.1f})")
else:
    print('Host CPU Single-thread baseline not available')

# GPU / Daemon baselines
print('\nAggregated GPU (Standalone) and GPU Daemon baselines:')
for name, ds in [('GPU Standalone', gpu_all), ('GPU Daemon', daemon_all)]:
    tot = summarize([d['mbps_total'] for d in ds], positive=True)
    ker = summarize([d['mbps_kernel'] for d in ds], positive=True)
    print(f"  {name:<16} | rows={tot['n']:4d} | total med={tot['median']:6.1f} mean={tot['mean']:6.1f} | kernel med={ker['median']:6.1f} mean={ker['mean']:6.1f} std={ker['std']:5.1f}")

# Combined kernel across GPU+Daemon
combined_comp = [d for d in rows if d['tool'] in ('GPU','DAEMON') and d['mode']=='compress']
combined_decomp = [d for d in rows if d['tool'] in ('GPU','DAEMON') and d['mode']=='decompress']
comp_k = summarize([d['mbps_kernel'] for d in combined_comp], positive=True)
dec_k = summarize([d['mbps_kernel'] for d in combined_decomp], positive=True)
comp_ratio = summarize([d['ratio_pct'] for d in combined_comp], positive=True)
print('\nCombined Kernel Throughput (GPU + Daemon):')
print(f"  Compress kernel med={comp_k['median']:6.1f} mean={comp_k['mean']:6.1f} std={comp_k['std']:5.1f} rows={comp_k['n']:4d} | Ratio med={comp_ratio['median']:5.2f}%")
print(f"  Decompress kernel med={dec_k['median']:6.1f} mean={dec_k['mean']:6.1f} std={dec_k['std']:5.1f} rows={dec_k['n']:4d}")
print('\nNote: Kernel throughput values are combined across GPU Standalone and Daemon (same kernel). Use per-tool Total MB/s for CPU comparisons.')

# per-alg
algs = sorted(set(d['alg'] for d in rows if d['alg']))
if algs:
    print('\nPer-algorithm baseline (combined kernel):')
    print('alg | rows | kernel med/mean (std) | ratio med')
    print('----|------|----------------------|-----------')
    for a in algs:
        ds = [d for d in combined_comp if d['alg']==a]
        ker = summarize([d['mbps_kernel'] for d in ds], positive=True)
        rat = summarize([d['ratio_pct'] for d in ds], positive=True)
        print(f" {a:<3} | {ker['n']:4d} | {ker['median']:6.1f}/{ker['mean']:6.1f} ({ker['std']:5.1f}) | {rat['median']:7.2f}%")

# SECTION B: factor sensitivity (show combined kernel, ratio and per-tool totals)
print('\n[SECTION B] FACTOR SENSITIVITY (detailed)')

def factor_table(rows, f_key, f_label):
    print(f"\nFactor: {f_label}  (combined kernel metrics; totals by tool)")
    for mode in ('compress','decompress'):
        all_ds = [d for d in rows if d['tool'] in ('GPU','DAEMON') and d['mode']==mode and d.get(f_key) not in (None,'',0)]
        if not all_ds:
            print(f"  No data for mode {mode}")
            continue
        overall_k = summarize([d['mbps_kernel'] for d in all_ds], positive=True)
        overall_r = summarize([d['ratio_pct'] for d in all_ds], positive=True) if mode == 'compress' else {'median':0.0}
        vals = sorted(set(d[f_key] for d in all_ds if d[f_key] is not None), key=lambda x: (int(x) if (isinstance(x,(str,int)) and str(x).isdigit()) else str(x)))
        print(f"\n  Mode: {mode.upper()} -> overall kernel med={overall_k['median']:.1f} MB/s")
        print('  VAL   | rows | k_med | k_mean | k_std | ratio_med% | totGPU_med | totDaemon_med | eff%')
        print('  ------|------|-------|--------|-------|-----------|-----------|---------------|------')
        kernel_meds = []
        ratio_meds = []
        for v in vals:
            ds_val = [d for d in all_ds if d[f_key]==v]
            ds_gpu = [d for d in ds_val if d['tool']=='GPU']
            ds_daemon = [d for d in ds_val if d['tool']=='DAEMON']
            ker = summarize([d['mbps_kernel'] for d in ds_val], positive=True)
            rat = summarize([d['ratio_pct'] for d in ds_val], positive=True) if mode=='compress' else {'median':0.0, 'n':0, 'mean':0.0, 'std':0.0}
            tot_gpu = summarize([d['mbps_total'] for d in ds_gpu], positive=True)
            tot_daemon = summarize([d['mbps_total'] for d in ds_daemon], positive=True)
            eff = (ker['median'] - overall_k['median'])/overall_k['median']*100 if overall_k['median']>0 else 0.0
            kernel_meds.append(ker['median'])
            if mode=='compress':
                ratio_meds.append(rat['median'])
            print(f"  {str(v):<5} | {ker['n']:4d} | {ker['median']:6.1f} | {ker['mean']:6.1f} | {ker['std']:6.1f} | {rat['median'] if mode=='compress' else 0.0:9.2f} | {tot_gpu['median']:9.1f} | {tot_daemon['median']:13.1f} | {eff:>+6.2f}%")
        if kernel_meds:
            rng = max(kernel_meds) - min(kernel_meds)
            rng_pct = rng / overall_k['median'] * 100 if overall_k['median']>0 else 0
            print(f"  Range across {f_label} (mode={mode}): {rng:.1f} MB/s ({rng_pct:.2f}% of overall kernel median)")
        if mode=='compress' and ratio_meds:
            rngr = max(ratio_meds) - min(ratio_meds)
            rngr_pct = rngr / (overall_r['median'] if overall_r['median']>0 else 1.0) * 100
            print(f"  Range across {f_label} (ratio): {rngr:.2f}% ({rngr_pct:.2f}% of overall ratio med)")

factor_table(rows, 'block_kb', 'BlockSize')
factor_table(rows, 'level_int', 'Level')
factor_table(rows, 'lsz_int', 'LocalSize')
factor_table(rows, 'alg', 'Algorithm')

# SECTION C: LocalSize per-sample impact (compress + decompress + ratio)
print('\n[SECTION C] LocalSize impact - increases and decreases (compress & decompress, ratio)')
lsz_vals = sorted(set(d['lsz_int'] for d in rows if d['lsz_int'] not in (None,0)))
samples = sorted(set(d['sample'] for d in rows))
sample_impacts = []
for s in sorted(set(d['sample'] for d in rows)):
    ds_s_c = [d for d in rows if d['sample']==s and d['mode']=='compress' and d['tool'] in ('GPU','DAEMON')]
    ds_s_d = [d for d in rows if d['sample']==s and d['mode']=='decompress' and d['tool'] in ('GPU','DAEMON')]
    if not ds_s_c:
        continue
    per_lsz_c = {}
    per_lsz_d = {}
    per_lsz_r = {}
    for l in lsz_vals:
        vals_c = [d['mbps_kernel'] for d in ds_s_c if d['lsz_int']==l and d['mbps_kernel']>0]
        if vals_c:
            per_lsz_c[l] = median(vals_c)
        vals_d = [d['mbps_kernel'] for d in ds_s_d if d['lsz_int']==l and d['mbps_kernel']>0]
        if vals_d:
            per_lsz_d[l] = median(vals_d)
        vals_r = [d['ratio_pct'] for d in ds_s_c if d['lsz_int']==l and d['ratio_pct']>0]
        if vals_r:
            per_lsz_r[l] = median(vals_r)
    if len(per_lsz_c) >= 2:
        med_all = median(list(per_lsz_c.values()))
        max_med = max(per_lsz_c.values())
        min_med = min(per_lsz_c.values())
        range_pct = (max_med - min_med)/med_all*100 if med_all>0 else 0.0
        change_1_8 = None
        if 1 in per_lsz_c and 8 in per_lsz_c and per_lsz_c[1]>0:
            change_1_8 = (per_lsz_c[8] - per_lsz_c[1]) / per_lsz_c[1] * 100
        sorted_items = sorted(per_lsz_c.items())
        adj_changes = []
        for i in range(len(sorted_items)-1):
            a = sorted_items[i][1]; b = sorted_items[i+1][1]
            if a>0:
                adj_changes.append((b-a)/a*100)
        max_adj = max(adj_changes) if adj_changes else 0.0
        min_adj = min(adj_changes) if adj_changes else 0.0
        d_range_pct = 0.0
        if len(per_lsz_d) >= 2:
            d_med_all = median(list(per_lsz_d.values()))
            d_range_pct = (max(per_lsz_d.values()) - min(per_lsz_d.values()))/d_med_all*100 if d_med_all>0 else 0.0
        r_range_pct = 0.0
        if len(per_lsz_r) >= 2:
            r_med_all = median(list(per_lsz_r.values()))
            r_range_pct = (max(per_lsz_r.values()) - min(per_lsz_r.values()))/r_med_all*100 if r_med_all>0 else 0.0
        sample_impacts.append({
            'sample': s,
            'per_lsz_c': per_lsz_c,
            'per_lsz_d': per_lsz_d,
            'per_lsz_r': per_lsz_r,
            'range_pct_c': range_pct,
            'change_1_8_c': change_1_8 if change_1_8 is not None else 0.0,
            'max_adj_c': max_adj,
            'min_adj_c': min_adj,
            'range_pct_d': d_range_pct,
            'range_pct_r': r_range_pct
        })

# increases/decreases (filter positive/negative)
inc = sorted([x for x in sample_impacts if x['change_1_8_c'] is not None and x['change_1_8_c']>0], key=lambda z: z['change_1_8_c'], reverse=True)
dec = sorted([x for x in sample_impacts if x['change_1_8_c'] is not None and x['change_1_8_c']<0], key=lambda z: z['change_1_8_c'])
print('\nTop samples by LocalSize increase (1->8, compress kernel):')
print(' sample (effect%) | per-lsz medians (lsz:med)')
print('-------------------|--------------------------------')
for x in inc[:10]:
    s = x['sample']; eff = x['change_1_8_c']; med_str = ', '.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_c'].items()))
    print(f" {s:<22} | {eff:7.2f}% | {med_str}")

if not inc:
    print('  (no increasing samples detected for 1->8)')

print('\nTop samples by LocalSize decrease (1->8, compress kernel):')
print(' sample (effect%) | per-lsz medians (lsz:med)')
print('-------------------|--------------------------------')
for x in dec[:10]:
    s = x['sample']; eff = x['change_1_8_c']; med_str = ', '.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_c'].items()))
    print(f" {s:<22} | {eff:7.2f}% | {med_str}")

if not dec:
    print('  (no decreasing samples detected for 1->8)')
# Export current LocalSize 1->8 compress lists to CSV for quick inspection
out_inc_csv = os.path.join(results_dir, f"{base}.localsize.1to8.compress.increases.csv")
with open(out_inc_csv, 'w', newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['sample','change_pct','per_lsz','per_lsz_ratio'])
    for x in inc:
        vals_s = ';'.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_c'].items()))
        rat_s = ';'.join(f"{k}:{v:.2f}" for k,v in sorted(x.get('per_lsz_r', {}).items()))
        w.writerow([x['sample'], f"{x['change_1_8_c']:.2f}", vals_s, rat_s])
print(f"  Wrote CSV -> {out_inc_csv}")

out_dec_csv = os.path.join(results_dir, f"{base}.localsize.1to8.compress.decreases.csv")
with open(out_dec_csv, 'w', newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['sample','change_pct','per_lsz','per_lsz_ratio'])
    for x in dec:
        vals_s = ';'.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_c'].items()))
        rat_s = ';'.join(f"{k}:{v:.2f}" for k,v in sorted(x.get('per_lsz_r', {}).items()))
        w.writerow([x['sample'], f"{x['change_1_8_c']:.2f}", vals_s, rat_s])
print(f"  Wrote CSV -> {out_dec_csv}")

# Build decompress 1->8 lists from sample_impacts if present
inc_decomp = []
dec_decomp = []
for s in sample_impacts:
    per_d = s.get('per_lsz_d', {})
    if per_d and 1 in per_d and 8 in per_d and per_d[1]>0:
        ch_d = (per_d[8] - per_d[1]) / per_d[1] * 100
        if ch_d > 0:
            inc_decomp.append({'sample': s['sample'], 'per_lsz_d': per_d, 'change_1_8_d': ch_d})
        elif ch_d < 0:
            dec_decomp.append({'sample': s['sample'], 'per_lsz_d': per_d, 'change_1_8_d': ch_d})
inc_decomp.sort(key=lambda z: z['change_1_8_d'], reverse=True)
dec_decomp.sort(key=lambda z: z['change_1_8_d'])

print('\nTop samples by LocalSize increase (1->8, DECOMPRESS kernel):')
print(' sample (effect%) | per-lsz medians (lsz:med)')
print('-------------------|--------------------------------')
for x in inc_decomp[:10]:
    s = x['sample']; eff = x['change_1_8_d']; med_str = ', '.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_d'].items()))
    print(f" {s:<22} | {eff:7.2f}% | {med_str}")
if not inc_decomp:
    print('  (no increasing samples detected for 1->8 decompress)')

print('\nTop samples by LocalSize decrease (1->8, DECOMPRESS kernel):')
print(' sample (effect%) | per-lsz medians (lsz:med)')
print('-------------------|--------------------------------')
for x in dec_decomp[:10]:
    s = x['sample']; eff = x['change_1_8_d']; med_str = ', '.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_d'].items()))
    print(f" {s:<22} | {eff:7.2f}% | {med_str}")
if not dec_decomp:
    print('  (no decreasing samples detected for 1->8 decompress)')

# export decompress 1->8 CSVs (ratio not applicable; column added for consistency)
out_inc_d_csv = os.path.join(results_dir, f"{base}.localsize.1to8.decompress.increases.csv")
with open(out_inc_d_csv,'w',newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['sample','change_pct','per_lsz','per_lsz_ratio'])
    for x in inc_decomp:
        w.writerow([x['sample'], f"{x['change_1_8_d']:.2f}", ';'.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_d'].items())), ''])
print(f"  Wrote CSV -> {out_inc_d_csv}")

out_dec_d_csv = os.path.join(results_dir, f"{base}.localsize.1to8.decompress.decreases.csv")
with open(out_dec_d_csv,'w',newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['sample','change_pct','per_lsz','per_lsz_ratio'])
    for x in dec_decomp:
        w.writerow([x['sample'], f"{x['change_1_8_d']:.2f}", ';'.join(f"{k}:{v:.1f}" for k,v in sorted(x['per_lsz_d'].items())), ''])
print(f"  Wrote CSV -> {out_dec_d_csv}")

# Now compute expanded per-factor per-sample impacts and export them
print('\n[SECTION C.2] Per-factor per-sample impacts & CSV exports')

factors = [('lsz_int','LocalSize'), ('block_kb','BlockSize'), ('level_int','Level'), ('alg','Algorithm')]

def per_sample_factor_impact(f_key, mode):
    out = []
    for s in samples:
        ds_s = [d for d in rows if d['sample']==s and d['mode']==mode and d['tool'] in ('GPU','DAEMON')]
        if not ds_s:
            continue
        vals_by = {}
        ratios_by = {}
        for v in sorted(set(d[f_key] for d in ds_s if d[f_key] not in (None,'')), key=lambda x: (int(x) if str(x).isdigit() else str(x))):
            vals = [d['mbps_kernel'] for d in ds_s if d[f_key]==v and d['mbps_kernel']>0]
            if vals:
                vals_by[v] = median(vals)
            if mode == 'compress':
                rats = [d['ratio_pct'] for d in ds_s if d[f_key]==v and d['ratio_pct']>0]
                if rats:
                    ratios_by[v] = median(rats)
        if len(vals_by) >= 2:
            med_all = median(list(vals_by.values()))
            rng = max(vals_by.values()) - min(vals_by.values())
            rng_pct = rng / med_all * 100 if med_all>0 else 0.0
            out.append((rng_pct, s, vals_by, ratios_by))
    out.sort(reverse=True)
    return out

for fk, fl in factors:
    for mode in ('compress','decompress'):
        topn = per_sample_factor_impact(fk, mode)[:20]
        outcsv = os.path.join(results_dir, f"{base}.top_{fl}.{mode}.csv")
        with open(outcsv, 'w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(['sample','range_pct','kernel_values','ratio_values'])
            for row in topn:
                # unpack 3- or 4-tuple to be tolerant
                if len(row) == 4:
                    rng, s, vals, rats = row
                else:
                    rng, s, vals = row; rats = {}
                vals_s = ';'.join(f"{k}:{v:.1f}" for k,v in sorted(vals.items(), key=lambda kv: (int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))))
                rat_s = ''
                if mode == 'compress' and rats:
                    rat_s = ';'.join(f"{k}:{v:.2f}" for k,v in sorted(rats.items(), key=lambda kv: (int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))))
                w.writerow([s, f"{rng:.2f}", vals_s, rat_s])
        print(f"  Wrote per-factor top list -> {outcsv}")
# adjacent decreases
adj_decrease = sorted([x for x in sample_impacts if x['min_adj_c']<0], key=lambda z: z['min_adj_c'])
print('\nTop samples with largest adjacent decrease (compress kernel):')
print(' sample | min_adj% | per-lsz medians')
print('--------|---------|-----------------')
for x in adj_decrease[:10]:
    print(f" {x['sample']:<22} | {x['min_adj_c']:7.2f}% | {', '.join(f'{k}:{v:.1f}' for k,v in sorted(x['per_lsz_c'].items()))}")

# Ratio changes
inc_r = sorted([x for x in sample_impacts if x['range_pct_r']>0], key=lambda z: z['range_pct_r'], reverse=True)
print('\nTop samples by compression ratio range across LocalSize:')
print(' sample | range% | per-lsz ratios')
print('--------|--------|---------------')
for x in inc_r[:10]:
    print(f" {x['sample']:<22} | {x['range_pct_r']:6.2f}% | {', '.join(f'{k}:{v:.2f}' for k,v in sorted(x['per_lsz_r'].items()))}")

# SECTION D: Per-alg LocalSize sensitivity (expanded)
print('\n[SECTION D] Per-alg LocalSize sensitivity (GPU Standalone)')
for a in algs:
    ds_a = [d for d in rows if d['alg']==a and d['tool']=='GPU' and d['mode']=='compress']
    if not ds_a:
        continue
    print(f"\nAlgorithm: {a}")
    print('lsz | rows | kernel med | kernel mean | std | ratio med')
    for l in lsz_vals:
        ds_l = [d for d in ds_a if d['lsz_int']==l]
        ker = summarize([d['mbps_kernel'] for d in ds_l], positive=True)
        rat = summarize([d['ratio_pct'] for d in ds_l], positive=True)
        print(f"{str(l):>3} | {ker['n']:4d} | {ker['median']:10.1f} | {ker['mean']:11.1f} | {ker['std']:6.1f} | {rat['median']:8.2f}")

# SECTION E: High variability configs
print('\n[SECTION E] High-variability configs across samples (compress kernel)')
from collections import defaultdict
cfgs = defaultdict(list)
for d in rows:
    if d['mode']=='compress' and d['tool'] in ('GPU','DAEMON'):
        key = (d['alg'], d['level'], d['block_kb'], d['lsz_int'])
        cfgs[key].append(d['mbps_kernel'])
var_list = []
for k,v in cfgs.items():
    n = len([x for x in v if x>0])
    if n >= 3:
        st = stddev([x for x in v if x>0])
        med = median([x for x in v if x>0])
        var_list.append((st, med, n, k))
var_list.sort(reverse=True)
print('stdev | med | n | config(alg,level,blk,lsz)')
for st,med,n,k in var_list[:15]:
    print(f"{st:5.1f} | {med:6.1f} | {n:2d} | {k}")

# conclusions
print('\n[CONCLUSIONS - ACTIONABLE]')
print('1. Kernel throughput is reported combined for GPU+Daemon (as requested); use per-tool total MB/s when comparing to Host CPU.')
print('2. LocalSize can both increase and decrease throughput; top-increase and top-decrease lists identify targets for focused re-runs with more repeats.')
print('3. Consider adding REPEATS and finer LocalSize sweeps for top samples; use median/std to improve separability.')

# restore stdout and write the captured report to a file
sys.stdout = old_stdout
report_path = os.path.join(results_dir, f"{base}.extended_report.txt")
with open(report_path, 'w') as rf:
    rf.write(buf.getvalue())
print(f"Wrote extended report -> {report_path}")
PY
}

# ========================================
# 主逻辑
# ========================================
case "$1" in
    -c|--cpu-only)
        init_csv
        for s in "$SAMPLES_DIR"/*; do [ -f "$s" ] && test_cpu "$s"; done
        generate_report
        ;;
    -g|--gpu-only)
        # GPU-only full scan; writes GPU rows only to $CSV_FILE
        init_csv
        for s in "$SAMPLES_DIR"/*; do [ -f "$s" ] && (test_gpu_standalone "$s"; test_gpu_daemon "$s") || true; done
        generate_report
        ;;
    -r|--report-only)
        generate_report "$2"
        ;;
    -m|--merge-csv)
        merge_csv "$2" "$3" "$4"
        ;;
    -f|--full)
        # Full CPU + GPU scan
        init_csv
        for s in "$SAMPLES_DIR"/*; do
            [ -f "$s" ] && (test_cpu "$s"; test_gpu_standalone "$s"; test_gpu_daemon "$s")
        done
        generate_report
        ;;
    *)
        init_csv
        for s in "$SAMPLES_DIR"/*; do
            [ -f "$s" ] && (test_cpu "$s"; test_gpu_standalone "$s"; test_gpu_daemon "$s")
        done
        generate_report
        ;;
esac
