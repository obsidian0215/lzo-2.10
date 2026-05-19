#!/usr/bin/env python3
import argparse
import csv
import hashlib
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager


IS_WINDOWS = os.name == "nt"
EXEEXT = ".exe" if IS_WINDOWS else ""
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SAMPLES = REPO_ROOT.parent / "samples"
DEFAULT_RESULTS = REPO_ROOT / "exp_results"
DEFAULT_GPU_BIN = REPO_ROOT / "lzo_gpu" / f"lzo_gpu{EXEEXT}"
DEFAULT_HYBRID_BIN = REPO_ROOT / "lzo_hybrid" / f"lzo_hybrid{EXEEXT}"
DEFAULT_CPU_BIN = REPO_ROOT / "lzo_cpu" / f"lzo_cpu{EXEEXT}"
DEFAULT_LZOP_BIN = REPO_ROOT / "tools" / f"lzop{EXEEXT}"

# Default scan matrix. Edit these lists when you want a new standard bench set.
# DEFAULT_ENGINES = ["gpu", "native_cpu", "hybrid"]
DEFAULT_ENGINES = ["gpu", "native_cpu"]
DEFAULT_ALG = "lzo1x"
DEFAULT_GPU_LEVELS = [15]
DEFAULT_CPU_LEVELS = [14]
DEFAULT_HYBRID_GPU_LEVELS = [15]
DEFAULT_HYBRID_CPU_LEVELS = [14]
DEFAULT_GPU_BLOCKS = ["64KB"]
DEFAULT_CPU_BLOCKS = ["64KB"]
DEFAULT_HYBRID_BLOCKS = ["64KB"]
DEFAULT_LOCAL_SIZES = [1]
DEFAULT_CPU_THREADS = [1]
DEFAULT_GPU_RATIOS = [0, 0.5]
DEFAULT_LZOP_LEVELS = [1, 3, 5]
DEFAULT_BENCH_SECONDS = 5
DEFAULT_MANUAL_ROUNDS = 6


BENCH_COMP_RE = re.compile(
    r"Bench\s+Compress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s.*?ratio=([0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)
BENCH_DEC_RE = re.compile(
    r"Bench\s+Decompress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s.*?verify=(OK|FAIL)",
    re.IGNORECASE,
)
RATIO_RE = re.compile(r"\(([0-9]+(?:\.[0-9]+)?)%\s+ratio\)", re.IGNORECASE)
KERNEL_TP_RE = re.compile(r"Kernel\s+Throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MB/s", re.IGNORECASE)
OCI_RE = re.compile(r"OCI\s+Setup\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
TOTAL_RE = re.compile(r"TOTAL\s+INCLUSIVE\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
CPU_COMP_RE = re.compile(
    r"Compressed\s+([0-9]+)\s+bytes\s+->\s+([0-9]+)\s+bytes\s+\(([0-9]+(?:\.[0-9]+)?)%\)",
    re.IGNORECASE,
)
CPU_TP_RE = re.compile(
    r"Throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+\(kernel:\s*([0-9]+(?:\.[0-9]+)?)\s*MB/s\)",
    re.IGNORECASE,
)
HYBRID_COMP_RE = re.compile(
    r"\[HYBRID\]\[C\].*?:\s*([0-9]+)\s*->\s*([0-9]+).*?"
    r"in\s*([0-9]+(?:\.[0-9]+)?)\s*ms.*?"
    r"span=([0-9]+(?:\.[0-9]+)?)\s*ms.*?"
    r"init_load=([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
HYBRID_DEC_RE = re.compile(
    r"\[HYBRID\]\[D\].*?:\s*([0-9]+)\s*->\s*([0-9]+).*?"
    r"in\s*([0-9]+(?:\.[0-9]+)?)\s*ms.*?"
    r"span=([0-9]+(?:\.[0-9]+)?)\s*ms.*?"
    r"init_load=([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
ACCEL_ROW_RE = re.compile(
    r"^\s*\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,(yes|no),\d+,\d+,"
    r"([0-9]+(?:\.[0-9]+)?),([0-9]+(?:\.[0-9]+)?),"
    r"[0-9]+(?:\.[0-9]+)?,([0-9]+(?:\.[0-9]+)?),"
    r"[0-9]+(?:\.[0-9]+)?,([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)

LZO_GPU_DAEMON_SOCKET = Path("/tmp/lzo_gpu_daemon.sock")
LZO_HYBRID_DAEMON_SOCKET = Path("/tmp/lzo_hybrid_daemon.sock")


RAW_FIELDS = [
    "sample", "engine", "phase", "round",
    "alg", "level", "block", "local_size", "gpu_ratio", "cpu_threads",
    "input_bytes", "compressed_bytes", "ratio_pct", "verify_ok",
    "bench_comp_kernel_mbs", "bench_dec_kernel_mbs", "manual_comp_kernel_mbs", "manual_dec_kernel_mbs",
    "manual_comp_no_ocl_mbs", "manual_dec_no_ocl_mbs",
    "manual_comp_seconds", "manual_dec_seconds",
    "status", "error",
]


def str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def mb(byte_count):
    return float(byte_count or 0) / (1024.0 * 1024.0)


def median(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return statistics.median(vals) if vals else None


def mean(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def stdev(values):
    vals = [float(v) for v in values if v not in (None, "")]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0 if vals else None


def numeric_values(values):
    return [float(v) for v in values if v not in (None, "")]


def sample_stdev(values):
    vals = numeric_values(values)
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None


def ci95_half_width(values):
    vals = numeric_values(values)
    if len(vals) <= 1:
        return 0.0 if vals else None
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)


def cv_pct(values):
    vals = numeric_values(values)
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    if avg == 0.0:
        return 0.0
    return 100.0 * statistics.stdev(vals) / avg if len(vals) > 1 else 0.0


def add_metric_stats(out, prefix, values):
    vals = numeric_values(values)
    out[f"{prefix}_n"] = len(vals)
    out[f"{prefix}_mean"] = mean(vals)
    out[f"{prefix}_median"] = median(vals)
    out[f"{prefix}_stdev"] = sample_stdev(vals)
    out[f"{prefix}_cv_pct"] = cv_pct(vals)
    out[f"{prefix}_ci95_half"] = ci95_half_width(vals)


def choose_metric_values(primary_rows, primary_field, fallback_rows, fallback_field):
    vals = numeric_values([r.get(primary_field, "") for r in primary_rows])
    if vals:
        return vals
    return numeric_values([r.get(fallback_field, "") for r in fallback_rows])


def fmt(value, digits=6):
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def host_id():
    return platform.node() or os.environ.get("COMPUTERNAME") or "unknown"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd, cwd=None, env=None, timeout=None):
    merged_env = os.environ.copy()
    if IS_WINDOWS:
        merged_env["PATH"] = r"C:\msys64\ucrt64\bin;C:\msys64\usr\bin;" + merged_env.get("PATH", "")
    if env:
        merged_env.update(env)
    start = time.perf_counter()
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout or "", elapsed


def parse_bench_output(text):
    comp = BENCH_COMP_RE.search(text or "")
    dec = BENCH_DEC_RE.search(text or "")
    if comp and dec:
        return {
            "ratio_pct": float(comp.group(2)),
            "comp_kernel_mbs": float(comp.group(1)),
            "dec_kernel_mbs": float(dec.group(1)),
            "verify_ok": dec.group(2).upper() == "OK",
        }
    accel_rows = [m for line in (text or "").splitlines() if (m := ACCEL_ROW_RE.match(line))]
    if accel_rows:
        return {
            "ratio_pct": median([m.group(3) for m in accel_rows]),
            "comp_kernel_mbs": median([m.group(4) for m in accel_rows]),
            "dec_kernel_mbs": median([m.group(5) for m in accel_rows]),
            "verify_ok": all(m.group(1).lower() == "yes" for m in accel_rows),
        }
    return None


def parse_total_seconds(text, wall_seconds, subtract_ocl):
    hybrid = HYBRID_COMP_RE.search(text or "") or HYBRID_DEC_RE.search(text or "")
    if hybrid:
        total = float(hybrid.group(3)) / 1000.0
        if subtract_ocl:
            init_s = float(hybrid.group(5)) / 1000.0
            if 0.0 < init_s < total:
                total -= init_s
        return max(0.0, total)
    total = None
    matches = TOTAL_RE.findall(text or "")
    if matches:
        total = float(matches[-1]) / 1000.0
    if not total or total <= 0.0:
        total = float(wall_seconds or 0.0)
    if subtract_ocl:
        oci = OCI_RE.search(text or "")
        if oci:
            oci_s = float(oci.group(1)) / 1000.0
            if 0.0 < oci_s < total:
                total -= oci_s
    return max(0.0, total)


def parse_manual_output(text):
    ratio = None
    compressed_bytes = None
    kernel_mbs = None
    ratio_match = RATIO_RE.search(text or "")
    if ratio_match:
        ratio = float(ratio_match.group(1))
    cpu_comp = CPU_COMP_RE.search(text or "")
    if cpu_comp:
        compressed_bytes = int(cpu_comp.group(2))
        ratio = float(cpu_comp.group(3))
    kernel_match = KERNEL_TP_RE.search(text or "")
    if kernel_match:
        kernel_mbs = float(kernel_match.group(1))
    cpu_tp = CPU_TP_RE.search(text or "")
    if cpu_tp:
        kernel_mbs = float(cpu_tp.group(2))
    hybrid_comp = HYBRID_COMP_RE.search(text or "")
    if hybrid_comp:
        input_bytes = int(hybrid_comp.group(1))
        compressed_bytes = int(hybrid_comp.group(2))
        ratio = 100.0 * compressed_bytes / input_bytes if input_bytes else 0.0
        span_s = float(hybrid_comp.group(4)) / 1000.0
        kernel_mbs = mb(input_bytes) / span_s if span_s > 0 else None
    hybrid_dec = HYBRID_DEC_RE.search(text or "")
    if hybrid_dec:
        output_bytes = int(hybrid_dec.group(2))
        span_s = float(hybrid_dec.group(4)) / 1000.0
        kernel_mbs = mb(output_bytes) / span_s if span_s > 0 else None
    return ratio, compressed_bytes, kernel_mbs


def discover_samples(samples_dir, limit, single_file):
    root = Path(samples_dir)
    if not root.exists():
        raise SystemExit(f"samples dir not found: {root}")
    files = sorted([p for p in root.iterdir() if p.is_file()])
    if not files:
        child_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if len(child_dirs) == 1:
            files = sorted([p for p in child_dirs[0].iterdir() if p.is_file()])
    if single_file:
        direct = Path(single_file)
        if direct.is_file():
            return [direct]
        matches = [p for p in files if p.name == single_file]
        if len(matches) != 1:
            raise SystemExit(f"single file not found or ambiguous: {single_file}")
        return matches
    if limit > 0:
        return files[:limit]
    return files


def base_row(args, sample, engine, phase, round_id, cfg):
    return {
        "sample": sample.name,
        "engine": engine,
        "phase": phase,
        "round": round_id,
        "alg": cfg.get("alg", ""),
        "level": cfg.get("level", ""),
        "block": cfg.get("block", ""),
        "local_size": cfg.get("local_size", ""),
        "gpu_ratio": cfg.get("gpu_ratio", ""),
        "cpu_threads": cfg.get("cpu_threads", ""),
        "input_bytes": sample.stat().st_size,
        "compressed_bytes": "",
        "ratio_pct": "",
        "verify_ok": "",
        "bench_comp_kernel_mbs": "",
        "bench_dec_kernel_mbs": "",
        "manual_comp_kernel_mbs": "",
        "manual_dec_kernel_mbs": "",
        "manual_comp_no_ocl_mbs": "",
        "manual_dec_no_ocl_mbs": "",
        "manual_comp_seconds": "",
        "manual_dec_seconds": "",
        "status": "ok",
        "error": "",
    }


def opencl_env(device_type=None):
    env = {
        "LZO_STANDARD_COPY": "0",
        "LZO_GPU_NO_CLBIN": "1",
    }
    if device_type:
        env["FORCE_OPENCL_DEVICE"] = str(device_type).upper()
    return env


def append_hybrid_args(cmd, cfg):
    gpu_ratio = str(cfg.get("gpu_ratio", "")).strip()
    if gpu_ratio:
        if gpu_ratio.lower() == "adaptive":
            cmd.append("--adaptive")
        else:
            cmd += ["--gpu-ratio", gpu_ratio]
    if cfg.get("cpu_threads") != "":
        cmd += ["--cpu-threads", cfg["cpu_threads"]]
    if cfg.get("gpu_level", "") != "":
        cmd += ["--gpu-level", cfg["gpu_level"]]
    if cfg.get("cpu_level", "") != "":
        cmd += ["--cpu-level", cfg["cpu_level"]]


def append_opencl_cpu_args(cmd, cfg):
    cmd += ["--gpu-ratio", "0"]
    if cfg.get("cpu_threads") != "":
        cmd += ["--cpu-threads", cfg["cpu_threads"]]


def append_daemon_hybrid_args(cmd, cfg):
    gpu_ratio = str(cfg.get("gpu_ratio", "")).strip()
    if gpu_ratio:
        if gpu_ratio.lower() == "adaptive":
            cmd.append("--adaptive")
        else:
            cmd += ["--gpu-ratio", gpu_ratio]
    if cfg.get("cpu_threads") != "":
        cmd += ["--cpu-threads", cfg["cpu_threads"]]


def append_daemon_opencl_cpu_args(cmd, cfg):
    cmd += ["--gpu-ratio", "0"]
    if cfg.get("cpu_threads") != "":
        cmd += ["--cpu-threads", cfg["cpu_threads"]]


def run_lzop_case(args, sample, level, tmp_root):
    exe = Path(args.lzop_bin).resolve()
    cfg = {"lzop_level": level}
    rows = []
    comp_path = tmp_root / f"{sample.name}.lzop.lzo"
    dec_path = tmp_root / f"{sample.name}.lzop.dec"
    original_hash = sha256(sample)
    input_bytes = sample.stat().st_size

    for round_id in range(args.manual_rounds):
        row = base_row(args, sample, f"lzop_{level}", "manual", round_id, cfg)
        try:
            comp_cmd = [exe, f"-{level}", "-f", "-o", comp_path, sample]
            rc_c, out_c, wall_c = run_cmd(comp_cmd, timeout=args.timeout)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"lzop compress failed rc={rc_c}: {out_c[-500:]}")
            compressed_bytes = comp_path.stat().st_size
            dec_cmd = [exe, "-d", "-f", "-o", dec_path, comp_path]
            rc_d, out_d, wall_d = run_cmd(dec_cmd, timeout=args.timeout)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"lzop decompress failed rc={rc_d}: {out_d[-500:]}")
            verify_ok = sha256(dec_path) == original_hash
            row["compressed_bytes"] = compressed_bytes
            row["ratio_pct"] = 100.0 * compressed_bytes / input_bytes if input_bytes else 0.0
            row["verify_ok"] = verify_ok
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / wall_c if wall_c > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / wall_d if wall_d > 0 else ""
            row["manual_comp_seconds"] = wall_c
            row["manual_dec_seconds"] = wall_d
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)
    return rows


def run_native_cpu_case(args, sample, threads, block, level, tmp_root):
    exe = Path(args.cpu_bin).resolve()
    cfg = {"alg": args.alg, "level": level, "block": block, "cpu_threads": threads}
    rows = []
    original_hash = sha256(sample)
    input_bytes = sample.stat().st_size

    bench_cmd = [exe, "--bench", args.bench_seconds, "-a", args.alg.replace("lzo", ""), "-L", level, "-B", block, "-t", threads, sample]
    row = base_row(args, sample, "lzo_cpu", "bench", 0, cfg)
    try:
        rc, out, _ = run_cmd(bench_cmd, timeout=args.timeout)
        bench = parse_bench_output(out)
        if rc != 0 or not bench:
            raise RuntimeError(f"native CPU bench failed rc={rc}: {out[-500:]}")
        row["ratio_pct"] = bench["ratio_pct"]
        row["verify_ok"] = bench["verify_ok"]
        row["bench_comp_kernel_mbs"] = bench["comp_kernel_mbs"]
        row["bench_dec_kernel_mbs"] = bench["dec_kernel_mbs"]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    rows.append(row)

    for round_id in range(args.manual_rounds):
        comp_path = tmp_root / f"{sample.name}.cpu.{round_id}.lzo"
        dec_path = tmp_root / f"{sample.name}.cpu.{round_id}.dec"
        row = base_row(args, sample, "lzo_cpu", "manual", round_id, cfg)
        try:
            comp_cmd = [exe, "-a", args.alg.replace("lzo", ""), "-L", level, "-B", block, "-t", threads, "-o", comp_path, sample]
            rc_c, out_c, wall_c = run_cmd(comp_cmd, timeout=args.timeout)
            ratio, compressed_bytes, comp_kernel = parse_manual_output(out_c)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"native CPU compress failed rc={rc_c}: {out_c[-500:]}")
            dec_cmd = [exe, "-d", "-t", threads, "-o", dec_path, comp_path]
            rc_d, out_d, wall_d = run_cmd(dec_cmd, timeout=args.timeout)
            _, _, dec_kernel = parse_manual_output(out_d)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"native CPU decompress failed rc={rc_d}: {out_d[-500:]}")
            verify_ok = sha256(dec_path) == original_hash
            compressed_bytes = compressed_bytes or comp_path.stat().st_size
            row["compressed_bytes"] = compressed_bytes
            row["ratio_pct"] = ratio if ratio is not None else 100.0 * compressed_bytes / input_bytes
            row["verify_ok"] = verify_ok
            row["manual_comp_kernel_mbs"] = comp_kernel
            row["manual_dec_kernel_mbs"] = dec_kernel
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / wall_c if wall_c > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / wall_d if wall_d > 0 else ""
            row["manual_comp_seconds"] = wall_c
            row["manual_dec_seconds"] = wall_d
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)
    return rows


def run_opencl_case(args, sample, engine, exe, device_type, cfg, tmp_root, extra_arg_fn=None, use_daemon=False):
    exe = Path(exe).resolve()
    rows = []
    original_hash = sha256(sample)
    input_bytes = sample.stat().st_size
    env = opencl_env(device_type)
    cwd = exe.parent
    socket_path = LZO_GPU_DAEMON_SOCKET if engine == "lzo_gpu" else LZO_HYBRID_DAEMON_SOCKET

    daemon_prefix = ["--use-daemon"] if use_daemon else []

    row = base_row(args, sample, engine, "bench", 0, cfg)
    try:
        bench_cmd = [
            exe, "--bench", args.bench_seconds,
            "-a", cfg["alg"], "-L", cfg["level"], "-B", cfg["block"],
            "--local", cfg["local_size"],
        ]
        if extra_arg_fn:
            extra_arg_fn(bench_cmd, cfg)
        bench_cmd.append(sample)
        rc, out, _ = run_cmd(bench_cmd, cwd=cwd, env=env, timeout=args.timeout)
        bench = parse_bench_output(out)
        if rc != 0 or not bench:
            raise RuntimeError(f"{engine} bench failed rc={rc}: {out[-500:]}")
        row["ratio_pct"] = bench["ratio_pct"]
        row["verify_ok"] = bench["verify_ok"]
        row["bench_comp_kernel_mbs"] = bench["comp_kernel_mbs"]
        row["bench_dec_kernel_mbs"] = bench["dec_kernel_mbs"]
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    rows.append(row)

    for round_id in range(args.manual_rounds):
        comp_path = tmp_root / f"{sample.name}.{engine}.{round_id}.lzo"
        dec_path = tmp_root / f"{sample.name}.{engine}.{round_id}.dec"
        row = base_row(args, sample, engine, "manual", round_id, cfg)
        try:
            comp_cmd = [
                exe, "-v", "-a", cfg["alg"], "-L", cfg["level"], "-B", cfg["block"],
                "--local", cfg["local_size"],
            ]
            if daemon_prefix:
                comp_cmd[1:1] = daemon_prefix
            if extra_arg_fn:
                if use_daemon and engine == "lzo_hybrid":
                    append_daemon_hybrid_args(comp_cmd, cfg)
                elif use_daemon and engine == "lzo_opencl_cpu":
                    append_daemon_opencl_cpu_args(comp_cmd, cfg)
                else:
                    extra_arg_fn(comp_cmd, cfg)
            comp_cmd += ["-o", comp_path, sample]
            rc_c, out_c, wall_c = run_cmd_with_daemon_retry(
                comp_cmd,
                cwd=cwd,
                env=env,
                timeout=args.timeout,
                daemon_enabled=use_daemon,
                exe_path=exe,
                socket_path=socket_path,
            )
            ratio, compressed_bytes, comp_kernel = parse_manual_output(out_c)
            if rc_c != 0 or not comp_path.exists():
                raise RuntimeError(f"{engine} compress failed rc={rc_c}: {out_c[-500:]}")
            dec_cmd = [exe, "-v", "-d"]
            if daemon_prefix:
                dec_cmd[1:1] = daemon_prefix
            if extra_arg_fn:
                if use_daemon and engine == "lzo_hybrid":
                    append_daemon_hybrid_args(dec_cmd, cfg)
                elif use_daemon and engine == "lzo_opencl_cpu":
                    append_daemon_opencl_cpu_args(dec_cmd, cfg)
                else:
                    extra_arg_fn(dec_cmd, cfg)
            dec_cmd += ["-o", dec_path, comp_path]
            rc_d, out_d, wall_d = run_cmd_with_daemon_retry(
                dec_cmd,
                cwd=cwd,
                env=env,
                timeout=args.timeout,
                daemon_enabled=use_daemon,
                exe_path=exe,
                socket_path=socket_path,
            )
            _, _, dec_kernel = parse_manual_output(out_d)
            verify_ok = (rc_d == 0 and dec_path.exists() and sha256(dec_path) == original_hash)
            if (rc_d != 0 or not dec_path.exists() or not verify_ok) and use_daemon:
                comp_cmd_sd = _strip_use_daemon_arg(comp_cmd)
                dec_cmd_sd = _strip_use_daemon_arg(dec_cmd)
                rc_c, out_c, wall_c = run_cmd(comp_cmd_sd, cwd=cwd, env=env, timeout=args.timeout)
                ratio, compressed_bytes, comp_kernel = parse_manual_output(out_c)
                rc_d, out_d, wall_d = run_cmd(dec_cmd_sd, cwd=cwd, env=env, timeout=args.timeout)
                _, _, dec_kernel = parse_manual_output(out_d)
                verify_ok = (rc_d == 0 and dec_path.exists() and sha256(dec_path) == original_hash)
            if rc_d != 0 or not dec_path.exists():
                raise RuntimeError(f"{engine} decompress failed rc={rc_d}: {out_d[-500:]}")
            compressed_bytes = compressed_bytes or comp_path.stat().st_size
            comp_s = parse_total_seconds(out_c, wall_c, subtract_ocl=True)
            dec_s = parse_total_seconds(out_d, wall_d, subtract_ocl=True)
            row["compressed_bytes"] = compressed_bytes
            row["ratio_pct"] = ratio if ratio is not None else 100.0 * compressed_bytes / input_bytes
            row["verify_ok"] = verify_ok
            row["manual_comp_kernel_mbs"] = comp_kernel
            row["manual_dec_kernel_mbs"] = dec_kernel
            row["manual_comp_no_ocl_mbs"] = mb(input_bytes) / comp_s if comp_s > 0 else ""
            row["manual_dec_no_ocl_mbs"] = mb(input_bytes) / dec_s if dec_s > 0 else ""
            row["manual_comp_seconds"] = comp_s
            row["manual_dec_seconds"] = dec_s
            if not verify_ok:
                row["status"] = "verify_failed"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        finally:
            safe_unlink(comp_path)
            safe_unlink(dec_path)
        rows.append(row)
    return rows


def safe_unlink(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _strip_use_daemon_arg(cmd):
    return [str(x) for x in cmd if str(x) != "--use-daemon"]


def _daemon_retry_needed(output_text):
    t = (output_text or "").lower()
    hints = (
        "connection refused",
        "daemon is not running",
        "connect daemon",
        "recv response",
        "连接守护进程失败",
        "接收响应失败",
        "守护进程未运行",
    )
    return any(h in t for h in hints)


def run_cmd_with_daemon_retry(cmd, *, cwd=None, env=None, timeout=None, daemon_enabled=False, exe_path=None, socket_path=None):
    rc, out, wall = run_cmd(cmd, cwd=cwd, env=env, timeout=timeout)
    if (not daemon_enabled) or rc == 0 or (not _daemon_retry_needed(out)):
        return rc, out, wall
    if not exe_path or not socket_path:
        return rc, out, wall
    try:
        stop_daemon(exe_path)
        time.sleep(0.1)
        start_daemon(exe_path, socket_path)
    except Exception:
        pass
    rc2, out2, wall2 = run_cmd(cmd, cwd=cwd, env=env, timeout=timeout)
    if rc2 == 0:
        return rc2, out2, wall2

    # Last-resort fallback: run standalone mode for this command.
    standalone_cmd = [str(x) for x in cmd if str(x) != "--use-daemon"]
    rc3, out3, wall3 = run_cmd(standalone_cmd, cwd=cwd, env=env, timeout=timeout)
    if rc3 == 0:
        merged = (out2 or out or "") + "\n[bench_lzo] daemon fallback -> standalone succeeded\n" + (out3 or "")
        return rc3, merged, wall3
    return rc2, out2, wall2


def daemon_is_running(socket_path):
    return Path(socket_path).exists()


def stop_daemon(exe_path):
    exe = Path(exe_path)
    cwd = exe.parent
    try:
        rc, out, _ = run_cmd([exe, "--stop-daemon"], cwd=cwd, timeout=10)
        return rc == 0, out
    except Exception as exc:
        return False, str(exc)


def start_daemon(exe_path, socket_path, timeout_s=15.0):
    exe = Path(exe_path)
    cwd = exe.parent
    proc = subprocess.Popen(
        [str(exe), "--daemon"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    deadline = time.time() + max(1.0, float(timeout_s))
    logs = []
    while time.time() < deadline:
        if daemon_is_running(socket_path):
            return proc, "".join(logs)
        if proc.poll() is not None:
            logs.append(proc.stdout.read() if proc.stdout else "")
            break
        time.sleep(0.1)
    if proc.poll() is None:
        proc.terminate()
    try:
        out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        out = ""
    logs.append(out)
    raise RuntimeError(f"failed to start daemon for {exe}: {''.join(logs)[-500:]}")


@contextmanager
def daemon_session(exe_path, socket_path, enabled):
    if (not enabled) or IS_WINDOWS:
        yield False
        return

    started_here = False
    proc = None
    try:
        if daemon_is_running(socket_path):
            yield True
            return
        proc, _ = start_daemon(exe_path, socket_path)
        started_here = True
        yield True
    finally:
        if started_here:
            stop_daemon(exe_path)
        if proc and proc.poll() is None:
            proc.terminate()


def case_key(row):
    return (
        row["sample"], row["engine"], row["alg"], row["level"], row["block"],
        row["local_size"], row["gpu_ratio"], row["cpu_threads"],
    )


def summarize(raw_rows):
    grouped = {}
    for row in raw_rows:
        grouped.setdefault(case_key(row), []).append(row)

    per_file = []
    for key, rows in sorted(grouped.items()):
        bench_rows = [r for r in rows if r["phase"] == "bench" and r["status"] == "ok"]
        manual_rows = [r for r in rows if r["phase"] == "manual" and r["status"] == "ok"]
        first = rows[0]
        ratio_rows = manual_rows if manual_rows else bench_rows
        out = {
            "sample": first.get("sample", ""),
            "engine": first.get("engine", ""),
            "alg": first.get("alg", ""),
            "level": first.get("level", ""),
            "block": first.get("block", ""),
            "local_size": first.get("local_size", ""),
            "gpu_ratio": first.get("gpu_ratio", ""),
            "cpu_threads": first.get("cpu_threads", ""),
            "input_bytes": first.get("input_bytes", ""),
            "compressed_bytes": first.get("compressed_bytes", ""),
            "bench_rounds": len(bench_rows),
            "manual_rounds": len(manual_rows),
            "ok": all(str(r.get("verify_ok", "")).lower() in ("true", "yes", "1") for r in manual_rows) if manual_rows else False,
        }
        add_metric_stats(out, "ratio_pct", [r["ratio_pct"] for r in ratio_rows])
        add_metric_stats(out, "comp_mbs", choose_metric_values(manual_rows, "manual_comp_kernel_mbs", bench_rows, "bench_comp_kernel_mbs"))
        add_metric_stats(out, "dec_mbs", choose_metric_values(manual_rows, "manual_dec_kernel_mbs", bench_rows, "bench_dec_kernel_mbs"))
        add_metric_stats(out, "e2e_comp_mbs", [r["manual_comp_no_ocl_mbs"] for r in manual_rows])
        add_metric_stats(out, "e2e_dec_mbs", [r["manual_dec_no_ocl_mbs"] for r in manual_rows])
        per_file.append(out)

    aggregate_grouped = {}
    for row in per_file:
        key = (
            row["engine"], row["alg"], row["level"], row["block"],
            row["local_size"], row["gpu_ratio"], row["cpu_threads"],
        )
        aggregate_grouped.setdefault(key, []).append(row)

    aggregate = []
    for key, rows in sorted(aggregate_grouped.items()):
        first = rows[0]
        aggregate.append({
            "engine": first["engine"],
            "alg": first["alg"],
            "level": first["level"],
            "block": first["block"],
            "local_size": first["local_size"],
            "gpu_ratio": first["gpu_ratio"],
            "cpu_threads": first["cpu_threads"],
            "samples": len(rows),
            "verify_all": all(bool(r["ok"]) for r in rows),
            "ratio_pct_median_of_files": median([r["ratio_pct_median"] for r in rows]),
            "ratio_pct_mean_of_files": mean([r["ratio_pct_median"] for r in rows]),
            "comp_mbs_median_of_files": median([r["comp_mbs_median"] for r in rows]),
            "comp_mbs_mean_of_files": mean([r["comp_mbs_median"] for r in rows]),
            "dec_mbs_median_of_files": median([r["dec_mbs_median"] for r in rows]),
            "dec_mbs_mean_of_files": mean([r["dec_mbs_median"] for r in rows]),
            "e2e_comp_mbs_median_of_files": median([r["e2e_comp_mbs_median"] for r in rows]),
            "e2e_comp_mbs_mean_of_files": mean([r["e2e_comp_mbs_median"] for r in rows]),
            "e2e_dec_mbs_median_of_files": median([r["e2e_dec_mbs_median"] for r in rows]),
            "e2e_dec_mbs_mean_of_files": mean([r["e2e_dec_mbs_median"] for r in rows]),
        })
    return per_file, aggregate


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_run_dir(results_dir):
    run_dir = Path(results_dir).resolve() / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_all(args):
    samples = discover_samples(args.samples, args.limit, args.single_file)
    if not samples:
        raise SystemExit("no samples found")

    run_dir = make_run_dir(args.results_dir)
    tmp_root = run_dir / "tmp"
    raw_rows = []
    print(f"[bench_lzo] run_dir={run_dir}")
    print(f"[bench_lzo] samples={len(samples)} bench_seconds={args.bench_seconds} manual_rounds={args.manual_rounds}")
    if args.use_daemon and not IS_WINDOWS:
        print("[bench_lzo] use_daemon=on (check daemon, start if missing, stop only if started by this run)")

    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        with daemon_session(args.gpu_bin, LZO_GPU_DAEMON_SOCKET, args.use_daemon) as gpu_daemon_active, daemon_session(args.hybrid_bin, LZO_HYBRID_DAEMON_SOCKET, args.use_daemon) as hybrid_daemon_active:
            for sample in samples:
                if "lzop" in args.engines:
                    for level in str_list(args.lzop_levels):
                        raw_rows.extend(run_lzop_case(args, sample, level, tmp_root))
                if "native_cpu" in args.engines:
                    for block in str_list(args.cpu_blocks):
                        for level in str_list(args.cpu_levels):
                            for threads in str_list(args.cpu_threads):
                                raw_rows.extend(run_native_cpu_case(args, sample, threads, block, level, tmp_root))
                if "gpu" in args.engines:
                    for block in str_list(args.gpu_blocks):
                        for level in str_list(args.gpu_levels):
                            for local_size in str_list(args.local_sizes):
                                cfg = {
                                    "alg": args.alg, "level": level, "block": block, "local_size": local_size,
                                    "gpu_ratio": "1", "cpu_threads": "", "gpu_level": "", "cpu_level": "",
                                }
                                raw_rows.extend(run_opencl_case(args, sample, "lzo_gpu", Path(args.gpu_bin), "GPU", cfg, tmp_root, use_daemon=gpu_daemon_active))
                if "opencl_cpu" in args.engines:
                    for block in str_list(args.cpu_blocks):
                        for level in str_list(args.gpu_levels):
                            for threads in str_list(args.cpu_threads):
                                cfg = {
                                    "alg": args.alg, "level": level, "block": block, "local_size": "1",
                                    "gpu_ratio": "0", "cpu_threads": threads, "gpu_level": "", "cpu_level": "",
                                }
                                raw_rows.extend(run_opencl_case(args, sample, "lzo_opencl_cpu", Path(args.hybrid_bin), "CPU", cfg, tmp_root, append_opencl_cpu_args, use_daemon=hybrid_daemon_active))
                if "hybrid" in args.engines:
                    for block in str_list(args.hybrid_blocks):
                        for gpu_level in str_list(args.hybrid_gpu_levels):
                            for cpu_level in str_list(args.hybrid_cpu_levels):
                                level = gpu_level
                                for ratio in str_list(args.gpu_ratios):
                                    for threads in str_list(args.cpu_threads):
                                        for local_size in str_list(args.local_sizes):
                                            cfg = {
                                                "alg": args.alg, "level": level, "block": block, "local_size": local_size,
                                                "gpu_ratio": ratio, "cpu_threads": threads,
                                                "gpu_level": gpu_level, "cpu_level": cpu_level,
                                            }
                                            raw_rows.extend(run_opencl_case(args, sample, "lzo_hybrid", Path(args.hybrid_bin), "ALL", cfg, tmp_root, append_hybrid_args, use_daemon=hybrid_daemon_active))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    per_file, aggregate = summarize(raw_rows)
    raw_path = run_dir / "raw.csv"
    per_file_path = run_dir / "per_file_summary.csv"
    aggregate_path = run_dir / "aggregate.csv"
    write_csv(raw_path, raw_rows, RAW_FIELDS)
    per_file_fields = list(per_file[0].keys()) if per_file else []
    aggregate_fields = list(aggregate[0].keys()) if aggregate else []
    if per_file_fields:
        write_csv(per_file_path, per_file, per_file_fields)
    if aggregate_fields:
        write_csv(aggregate_path, aggregate, aggregate_fields)
    with open(run_dir / "run_meta.txt", "w", encoding="utf-8") as handle:
        handle.write(f"argv={' '.join(sys.argv)}\n")
        handle.write(f"platform_id={args.platform_id}\n")
        handle.write(f"bench_seconds={args.bench_seconds}\n")
        handle.write(f"manual_rounds={args.manual_rounds}\n")
        handle.write(f"samples={args.samples}\n")
        handle.write(f"engines={args.engines}\n")
    print(f"[bench_lzo] raw={raw_path}")
    print(f"[bench_lzo] per_file={per_file_path}")
    print(f"[bench_lzo] aggregate={aggregate_path}")
    return run_dir


def parse_args(argv):
    parser = argparse.ArgumentParser(description="LZO benchmark/manual runner; power/frequency scans belong to external wrappers.")
    parser.add_argument("--platform-id", default=platform.node() or "local")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--single-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--bench-seconds", default=str(DEFAULT_BENCH_SECONDS))
    parser.add_argument("--manual-rounds", type=int, default=DEFAULT_MANUAL_ROUNDS)
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES), help="Comma list: gpu,opencl_cpu,native_cpu,hybrid,lzop")
    parser.add_argument("--alg", default=DEFAULT_ALG)
    parser.add_argument("--gpu-levels", default=",".join(str(x) for x in DEFAULT_GPU_LEVELS))
    parser.add_argument("--cpu-levels", default=",".join(str(x) for x in DEFAULT_CPU_LEVELS))
    parser.add_argument("--hybrid-gpu-levels", default=",".join(str(x) for x in DEFAULT_HYBRID_GPU_LEVELS))
    parser.add_argument("--hybrid-cpu-levels", default=",".join(str(x) for x in DEFAULT_HYBRID_CPU_LEVELS))
    parser.add_argument("--gpu-blocks", default=",".join(DEFAULT_GPU_BLOCKS))
    parser.add_argument("--cpu-blocks", default=",".join(DEFAULT_CPU_BLOCKS))
    parser.add_argument("--hybrid-blocks", default=",".join(DEFAULT_HYBRID_BLOCKS))
    parser.add_argument("--local-sizes", default=",".join(str(x) for x in DEFAULT_LOCAL_SIZES))
    parser.add_argument("--cpu-threads", default=",".join(str(x) for x in DEFAULT_CPU_THREADS))
    parser.add_argument("--gpu-ratios", default=",".join(str(x) for x in DEFAULT_GPU_RATIOS))
    parser.add_argument("--lzop-levels", default=",".join(str(x) for x in DEFAULT_LZOP_LEVELS))
    parser.add_argument("--gpu-bin", default=str(DEFAULT_GPU_BIN))
    parser.add_argument("--hybrid-bin", default=str(DEFAULT_HYBRID_BIN))
    parser.add_argument("--cpu-bin", default=str(DEFAULT_CPU_BIN))
    parser.add_argument("--lzop-bin", default=str(DEFAULT_LZOP_BIN))
    parser.add_argument("--use-daemon", action="store_true", help="Linux only: use GPU daemon; check running, start if missing, stop only if started by this run")
    args = parser.parse_args(argv)
    args.engines = set(str_list(args.engines))
    return args


if __name__ == "__main__":
    run_all(parse_args(sys.argv[1:]))
