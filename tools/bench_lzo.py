#!/usr/bin/env python3
import os
import subprocess
import re
import csv
from pathlib import Path

# Paths
LZO_CPU_BIN = "/root/lzo-2.10/lzo_cpu/lzo_cpu"
LZO_GPU_BIN = "/root/lzo-2.10/lzo_gpu/lzo_gpu"
SAMPLES_DIR = "/root/samples"
RESULTS_DIR = "/root/lzo-2.10/exp_results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "lzo_param_sweep.csv")

# Configuration Space
ALGS = ["lzo1x", "lzo1y"]
BLOCK_SIZES = ["16K", "64K", "256K", "512K"]
CPU_THREADS = [1, 2, 4]
GPU_LEVELS = [11, 12, 13, 14]
GPU_LOCAL_SIZES = [1, 8, 64]

# Helpers
import argparse
import tempfile
import time

def parse_size_to_bytes(s):
    s = str(s).strip()
    if not s:
        return 0
    unit = s[-1].upper()
    if unit == 'K':
        return int(float(s[:-1]) * 1024)
    if unit == 'M':
        return int(float(s[:-1]) * 1024 * 1024)
    if unit == 'G':
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    return int(s)


def parse_gpu_output(output):
    stats = {}
    try:
        ratio_match = re.search(r"\((\d+\.\d+)% ratio\)", output)
        if ratio_match:
            stats['ratio'] = float(ratio_match.group(1))

        kernel_tp_match = re.search(r"Kernel Throughput\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", output)
        if kernel_tp_match:
            stats['kernel_tp'] = float(kernel_tp_match.group(1))

        incl_tp_match = re.search(r"Inclusive Throughput\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", output)
        if incl_tp_match:
            stats['inclusive_tp'] = float(incl_tp_match.group(1))
    except Exception:
        pass
    return stats


def run_lzo_cpu(file_path, alg, bs, threads):
    # Mapping "lzo1x" to "1x" for CPU tool
    alg_short = alg.replace("lzo", "")
    print(f"Bench_CPU: {file_path.name} A={alg_short} T={threads}")
    # Note: lzo_cpu doesn't accept block size flag -B, so omit it for CPU runs
    cmd = [LZO_CPU_BIN, "--benchmark", "-a", alg_short, "-t", str(threads), str(file_path), "-o", "/dev/null"]
    stats = {'ratio': 0, 'comp_kernel_tp': 0, 'dec_kernel_tp': 0}
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (res.stdout or "") + (res.stderr or "")

        ratio_match = re.search(r"Compression ratio\s*:\s*([0-9]+\.?[0-9]*)%", output)
        if ratio_match:
            stats['ratio'] = float(ratio_match.group(1))

        comp_match = re.search(r"Multi\s+Compress\s*:\s*[0-9]+\.?[0-9]*\s*ms\s*\(\s*[0-9]+\s*blocks,\s*([0-9]+\.?[0-9]*)\s*MB/s\s*\)", output)
        if comp_match:
            stats['comp_kernel_tp'] = float(comp_match.group(1))

        single_dec_match = re.search(r"Single\s+Decompress\s*:\s*[0-9]+\.?[0-9]*\s*ms\s*\(\s*([0-9]+\.?[0-9]*)\s*MB/s\s*\)", output)
        multi_dec_match = re.search(r"Multi\s+Decompress\s*:\s*([0-9]+\.?[0-9]*)\s*MB/s", output)

        if single_dec_match and multi_dec_match:
            s_tp = float(single_dec_match.group(1))
            m_tp = float(multi_dec_match.group(1))
            stats['dec_kernel_tp'] = (s_tp + m_tp) / 2.0
    except Exception as e:
        print(f"CPU error: {e}")
    return stats


def run_lzo_gpu(file_path, alg, level, bs, lsz):
    print(f"Bench_GPU: {file_path.name} A={alg} L={level} BS={bs} LSZ={lsz}")
    bs_arg = str(bs).lower()
    cmd_c = [LZO_GPU_BIN, "-v", "-a", alg, "-L", str(level), "-B", bs_arg, "--local", str(lsz), str(file_path), "-o", "/dev/null"]
    stats = {'ratio': 0, 'comp_kernel_tp': 0, 'dec_kernel_tp': 0}
    try:
        res_c = subprocess.run(cmd_c, capture_output=True, text=True, check=False)
        output_c = (res_c.stdout or "") + (res_c.stderr or "")
        stats_c = parse_gpu_output(output_c)
        stats['ratio'] = stats_c.get('ratio', 0)
        stats['comp_kernel_tp'] = stats_c.get('kernel_tp', 0)

        with tempfile.NamedTemporaryFile(prefix="bench_lzo_tmp_", suffix=".lzo", delete=False) as tf:
            tmp_lzo = tf.name
        try:
            subprocess.run([LZO_GPU_BIN, "-a", alg, "-L", str(level), "-B", bs_arg, str(file_path), "-o", tmp_lzo], check=True, capture_output=True)
            cmd_d = [LZO_GPU_BIN, "-v", "-d", tmp_lzo, "-o", "/dev/null", "--local", str(lsz)]
            res_d = subprocess.run(cmd_d, capture_output=True, text=True, check=False)
            output_d = (res_d.stdout or "") + (res_d.stderr or "")
            stats_d = parse_gpu_output(output_d)
            stats['dec_kernel_tp'] = stats_d.get('kernel_tp', 0)
        finally:
            try: os.remove(tmp_lzo)
            except Exception: pass
    except Exception as e:
        print(f"GPU Error: {e}")
        pass
    return stats

def main():
    parser = argparse.ArgumentParser(description='Bench LZO CPU/GPU sweep (supports --limit for quick runs)')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of samples (0 = all)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    samples = sorted([p for p in Path(SAMPLES_DIR).glob("*") if p.is_file()])

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Engine", "Alg", "Level", "BlockSize", "Threads_LSZ", "Ratio%", "CompKernel_MBs", "DecKernel_MBs"])

        for sample in samples:
            # CPU Sweep (block sizes not applicable for lzo_cpu)
            for t in CPU_THREADS:
                cpu_stats = run_lzo_cpu(sample, "lzo1x", None, t)
                writer.writerow([sample.name, f"CPU_T{t}", "lzo1x", "1", "N/A", t, f"{cpu_stats['ratio']:.2f}", f"{cpu_stats['comp_kernel_tp']:.2f}", f"{cpu_stats['dec_kernel_tp']:.2f}"])
                f.flush()

            # GPU Sweep
            for alg in ALGS:
                for level in GPU_LEVELS:
                    for bs in BLOCK_SIZES:
                        for lsz in GPU_LOCAL_SIZES:
                            gpu_stats = run_lzo_gpu(sample, alg, level, bs, lsz)
                            writer.writerow([sample.name, "GPU", alg, level, bs, lsz, f"{gpu_stats['ratio']:.2f}", f"{gpu_stats['comp_kernel_tp']:.2f}", f"{gpu_stats['dec_kernel_tp']:.2f}"])
                            f.flush()

if __name__ == "__main__":
    main()
