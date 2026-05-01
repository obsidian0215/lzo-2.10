#!/usr/bin/env python3
import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hw_telemetry import TelemetryProbe, apply_freq_mhz, apply_freq_percent


CPU_CONTROL = SCRIPT_DIR / "cpu_control.sh"
GPU_CONTROL = SCRIPT_DIR / "gpu_control.sh"
BENCH_LZO = SCRIPT_DIR / "bench_lzo.py"


def parse_frequency_points(text):
    if text is None or str(text).strip() == "":
        return [{"kind": "none", "value": None, "label": "NA"}]
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item or item.upper() in ("NA", "NONE", "-"):
            out.append({"kind": "none", "value": None, "label": "NA"})
        elif item.endswith("%"):
            value = int(item[:-1].strip())
            if value < 0 or value > 100:
                raise SystemExit(f"invalid percent frequency point: {item}")
            out.append({"kind": "percent", "value": value, "label": f"{value}pct"})
        else:
            raw = item[:-3] if item.lower().endswith("mhz") else item
            value = int(raw.strip())
            if value < 1:
                raise SystemExit(f"invalid MHz frequency point: {item}")
            out.append({"kind": "mhz", "value": value, "label": f"{value}mhz"})
    return out or [{"kind": "none", "value": None, "label": "NA"}]


def build_points(cpu_points, gpu_points):
    return [(cpu, gpu) for cpu in cpu_points for gpu in gpu_points]


def has_bench_option(args_list, option):
    return any(item == option or item.startswith(option + "=") for item in args_list)


def with_bench_option(args_list, option, value):
    out = []
    skip_next = False
    for item in args_list:
        if skip_next:
            skip_next = False
            continue
        if item == option:
            skip_next = True
            continue
        if item.startswith(option + "="):
            continue
        out.append(item)
    out += [option, str(value)]
    return out


def bench_arg_value(args_list, option):
    for idx, item in enumerate(args_list):
        if item == option and idx + 1 < len(args_list):
            return args_list[idx + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None


def split_csv_text(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def default_bench_value(name):
    code = (
        "import importlib.util; "
        "s=importlib.util.spec_from_file_location('b','tools/bench_lzo.py'); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"print(','.join(str(x) for x in m.{name}))"
    )
    try:
        res = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if res.returncode == 0:
            return (res.stdout or "").strip()
    except Exception:
        pass
    return ""


def effective_engines(args_list):
    text = bench_arg_value(args_list, "--engines") or default_bench_value("DEFAULT_ENGINES")
    return set(split_csv_text(text))


def effective_gpu_ratios(args_list):
    text = bench_arg_value(args_list, "--gpu-ratios") or default_bench_value("DEFAULT_GPU_RATIOS")
    return split_csv_text(text)


def is_cpu_only_ratio(ratio):
    try:
        return float(ratio) <= 0.0
    except Exception:
        return False


def is_gpu_only_ratio(ratio):
    try:
        return float(ratio) >= 1.0
    except Exception:
        return False


def build_work_items(args, cpu_points, gpu_points):
    engines = effective_engines(args.bench_args)
    ratios = effective_gpu_ratios(args.bench_args)
    common = list(args.bench_args)
    items = []
    seen = set()

    def add(label, cpu_point, gpu_point, bench_args):
        key = (label, cpu_point["label"], gpu_point["label"], tuple(bench_args))
        if key in seen:
            return
        seen.add(key)
        items.append({"label": label, "cpu": cpu_point, "gpu": gpu_point, "bench_args": bench_args})

    none_cpu = {"kind": "none", "value": None, "label": "NA"}
    none_gpu = {"kind": "none", "value": None, "label": "NA"}

    gpu_engines = [engine for engine in ("gpu",) if engine in engines]
    cpu_engines = [engine for engine in ("native_cpu", "opencl_cpu", "lzop") if engine in engines]
    if gpu_engines:
        bench_args = with_bench_option(common, "--engines", ",".join(gpu_engines))
        for gpu_point in gpu_points:
            add("gpu", none_cpu, gpu_point, bench_args)
    if cpu_engines:
        bench_args = with_bench_option(common, "--engines", ",".join(cpu_engines))
        for cpu_point in cpu_points:
            add("cpu", cpu_point, none_gpu, bench_args)
    if "hybrid" in engines:
        for ratio in ratios:
            bench_args = with_bench_option(with_bench_option(common, "--engines", "hybrid"), "--gpu-ratios", ratio)
            if str(ratio).strip().lower() == "adaptive":
                for cpu_point in cpu_points:
                    for gpu_point in gpu_points:
                        add(f"hybrid_ratio_{ratio}", cpu_point, gpu_point, bench_args)
            elif is_cpu_only_ratio(ratio):
                for cpu_point in cpu_points:
                    add(f"hybrid_ratio_{ratio}", cpu_point, none_gpu, bench_args)
            elif is_gpu_only_ratio(ratio):
                for gpu_point in gpu_points:
                    add(f"hybrid_ratio_{ratio}", none_cpu, gpu_point, bench_args)
            else:
                for cpu_point in cpu_points:
                    for gpu_point in gpu_points:
                        add(f"hybrid_ratio_{ratio}", cpu_point, gpu_point, bench_args)
    return items


def apply_frequency(script, point):
    kind = point["kind"]
    value = point["value"]
    if kind == "none":
        return run_control_reset(script)
    if kind == "percent":
        return apply_freq_percent(str(script), value)
    if kind == "mhz":
        return apply_freq_mhz(str(script), value)
    return f"invalid_kind:{kind}"


def run_control_reset(script):
    if os.name == "nt":
        return "unsupported_on_windows"
    cmd = [str(script), "reset"]
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        cmd = ["sudo", "-n"] + cmd
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return "ok"
        msg = ((res.stderr or "") + "\n" + (res.stdout or "")).strip().splitlines()
        return f"failed:{res.returncode}:{msg[0][:100] if msg else ''}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def summarize_samples(telemetry, samples):
    if not samples:
        return telemetry.summarize_samples([])
    return telemetry.summarize_samples(samples)


def run_bench(args, point_idx, item):
    cpu_point = item["cpu"]
    gpu_point = item["gpu"]
    safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in item["label"])
    run_dir = Path(args.output_dir) / f"point_{point_idx:03d}_{safe_label}_cpu_{cpu_point['label']}_gpu_{gpu_point['label']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(BENCH_LZO),
        "--platform-id", args.platform_id,
        "--results-dir", str(run_dir),
    ]
    cmd.extend(item["bench_args"])
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc, cmd, run_dir, start


def collect_until_done(proc, telemetry, interval_s):
    samples = []
    lines = []
    stop = threading.Event()

    def sampler():
        while not stop.wait(interval_s):
            try:
                samples.append(telemetry.snapshot())
            except Exception:
                pass
            if proc.poll() is not None:
                break

    thread = threading.Thread(target=sampler, daemon=True)
    try:
        samples.append(telemetry.snapshot())
    except Exception:
        pass
    thread.start()
    for line in proc.stdout:
        lines.append(line)
        print(line, end="")
    rc = proc.wait()
    stop.set()
    thread.join(timeout=max(1.0, interval_s * 2.0))
    try:
        samples.append(telemetry.snapshot())
    except Exception:
        pass
    return rc, "".join(lines), samples


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv):
    parser = argparse.ArgumentParser(
        description="External power/frequency wrapper for bench_lzo.py. Arguments after -- are passed to bench_lzo.py."
    )
    parser.add_argument("--platform-id", default=os.environ.get("LZO_PLATFORM_ID", "local"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "exp_results" / "power_freq_runs" / time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--cpu-frequencies", "--cpu-points", default="NA", help='Comma list: "2100,3400,NA" or "50%,100%,NA"')
    parser.add_argument("--gpu-frequencies", "--gpu-points", default="NA", help='Comma list: "1000,NA" or "70%,NA"')
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--no-reset-before", action="store_true")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.bench_args and args.bench_args[0] == "--":
        args.bench_args = args.bench_args[1:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = TelemetryProbe()
    cpu_points = parse_frequency_points(args.cpu_frequencies)
    gpu_points = parse_frequency_points(args.gpu_frequencies)
    points = build_work_items(args, cpu_points, gpu_points)
    records = []

    print(f"[wrapper] output_dir={output_dir}")
    print(f"[wrapper] telemetry={telemetry.describe_sources()}")
    print(f"[wrapper] points={[(p['label'], p['cpu']['label'], p['gpu']['label'], shlex.join(p['bench_args'])) for p in points]}")

    for idx, item in enumerate(points, start=1):
        cpu_point = item["cpu"]
        gpu_point = item["gpu"]
        cpu_apply = "not_requested"
        gpu_apply = "not_requested"
        cpu_reset_before = "skipped"
        gpu_reset_before = "skipped"
        cpu_reset_after = "not_run"
        gpu_reset_after = "not_run"
        rc = -1
        run_dir = ""
        elapsed = 0.0
        summary = {}
        try:
            if not args.no_reset_before:
                cpu_reset_before = run_control_reset(CPU_CONTROL)
                gpu_reset_before = run_control_reset(GPU_CONTROL)
            cpu_apply = apply_frequency(CPU_CONTROL, cpu_point)
            gpu_apply = apply_frequency(GPU_CONTROL, gpu_point)

            proc, cmd, run_dir_path, start = run_bench(args, idx, item)
            rc, output, samples = collect_until_done(proc, telemetry, max(0.05, args.sample_interval))
            elapsed = max(0.0, time.perf_counter() - start)
            run_dir = str(run_dir_path)
            summary = summarize_samples(telemetry, samples)
            with open(run_dir_path / "wrapper_command.txt", "w", encoding="utf-8") as handle:
                handle.write(" ".join(str(x) for x in cmd) + "\n")
            with open(run_dir_path / "wrapper_output.log", "w", encoding="utf-8") as handle:
                handle.write(output)
        finally:
            cpu_reset_after = run_control_reset(CPU_CONTROL)
            gpu_reset_after = run_control_reset(GPU_CONTROL)

        record = {
            "point_idx": idx,
            "workload": item["label"],
            "cpu_target_kind": cpu_point["kind"],
            "gpu_target_kind": gpu_point["kind"],
            "cpu_target": "" if cpu_point["value"] is None else cpu_point["value"],
            "gpu_target": "" if gpu_point["value"] is None else gpu_point["value"],
            "cpu_apply": cpu_apply,
            "gpu_apply": gpu_apply,
            "cpu_reset_before": cpu_reset_before,
            "gpu_reset_before": gpu_reset_before,
            "cpu_reset_after": cpu_reset_after,
            "gpu_reset_after": gpu_reset_after,
            "returncode": rc,
            "elapsed_s": elapsed,
            "run_dir": run_dir,
            "bench_args": shlex.join(item["bench_args"]),
            "telemetry_sources": telemetry.describe_sources(),
            "cpu_freq_avg_mhz": summary.get("cpu_freq_avg_mhz", 0.0),
            "gpu_freq_avg_mhz": summary.get("gpu_freq_avg_mhz", 0.0),
            "cpu_energy_j": summary.get("cpu_energy_j", 0.0),
            "core_energy_j": summary.get("core_energy_j", 0.0),
            "dram_energy_j": summary.get("dram_energy_j", 0.0),
            "gpu_energy_j": summary.get("gpu_energy_j", 0.0),
            "cpu_pkg_avg_power_w": summary.get("cpu_pkg_avg_power_w", 0.0),
            "cpu_core_avg_power_w": summary.get("cpu_core_avg_power_w", 0.0),
            "dram_avg_power_w": summary.get("dram_avg_power_w", 0.0),
            "gpu_avg_power_w": summary.get("gpu_avg_power_w", 0.0),
            "cpu_pkg_peak_power_w": summary.get("cpu_pkg_peak_power_w", 0.0),
            "cpu_core_peak_power_w": summary.get("cpu_core_peak_power_w", 0.0),
            "dram_peak_power_w": summary.get("dram_peak_power_w", 0.0),
            "gpu_peak_power_w": summary.get("gpu_peak_power_w", 0.0),
        }
        records.append(record)
        write_csv(output_dir / "power_frequency_summary.csv", records, list(record.keys()))
        print(f"[wrapper] point={idx} rc={rc} cpu_reset_after={cpu_reset_after} gpu_reset_after={gpu_reset_after}")

    print(f"[wrapper] summary={output_dir / 'power_frequency_summary.csv'}")
    return 0 if all(int(r["returncode"]) == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
