#!/usr/bin/env python3
"""
tools/bench.py - Consolidated LZO bench runner

Features:
  - run: execute benchmark runs across modes/runners and keep logs
  - parse: parse logs for a specific sample and write stage breakdown JSON/CSV
  - throughput: augment CSV with throughput and write a summary
  - variance: Analyze CV across runs (read JSON breakdowns and report high CV combos)
  - analysis: generate per-sample analysis Markdown using stage breakdowns
  - all: run the entire pipeline for files

This consolidated script replaces the older lzo_gpu/tools/*.sh/.py helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import hashlib
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LZO_GPU = ROOT / 'lzo_gpu' / 'lzo_gpu'
LZO_CLIENT = ROOT / 'lzo_gpu' / 'lzo_gpu_client'
EXP_DIR = ROOT / 'lzo_gpu' / 'exp_results'

def parse_time_to_ms(val_str: str | None) -> float:
    if not val_str:
        return 0.0
    s = str(val_str).strip()
    m = re.match(r"^([0-9.]+)\s*(ms|us|s)?$", s, flags=re.I)
    if not m:
        # fallback to last numeric token
        nums = re.findall(r"[0-9.]+", s)
        if not nums:
            return 0.0
        val = float(nums[-1]); unit = 'ms'
    else:
        val = float(m.group(1)); unit = (m.group(2) or 'ms').lower()
    if unit == 'ms': return val
    if unit == 'us': return val / 1000.0
    if unit == 's': return val * 1000.0
    return val


def ensure_bins():
    if not LZO_GPU.exists() or not LZO_GPU.is_file():
        raise FileNotFoundError(f'lzo_gpu binary not found at {LZO_GPU}')


def compute_sha1_id(path: str) -> str:
    h = hashlib.sha1()
    h.update(path.encode('utf-8'))
    return h.hexdigest()[:8]


def get_free_space_bytes(path: Path | str) -> int:
    try:
        st = os.statvfs(str(path))
        return st.f_bavail * st.f_frsize
    except Exception:
        return 0


def gather_files(paths: list[str], recursive: bool, min_size: int) -> list[Path]:
    files = []
    for p in paths:
        pth = Path(p)
        if pth.is_file():
            if pth.stat().st_size >= min_size:
                files.append(pth)
        elif pth.is_dir():
            if recursive:
                for f in pth.rglob('*'):
                    if f.is_file() and f.stat().st_size >= min_size:
                        files.append(f)
            else:
                for f in pth.glob('*'):
                    if f.is_file() and f.stat().st_size >= min_size:
                        files.append(f)
        else:
            print(f'Warning: {p} not a file or dir')
    return sorted(files)


def create_logs_dir(prefix='lzo-bench-logs') -> Path:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = EXP_DIR / f"{prefix}.{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_bench(files: list[Path], runs: int = 3, mt_threads: int = 4, runners: list[str] = ['standalone'], out_csv: Path | str = 'benchmark_mt_io_results.csv', keep_logs: Path | None = None, modes: list[str] | None = None, keep_artifacts_on_fail: bool = False, resume: bool = False):
    if modes is None:
        modes = ['zero', 'zero+mt', 'std', 'std+mt', 'std_async', 'std+mt_async']

    out_csv = Path(out_csv)
    logs_dir = create_logs_dir()
    TMPDIR = tempfile.TemporaryDirectory(prefix='lzo-bench-')

    # CSV header
    header = ['file','size_bytes','runner','mode','async','mt','run','ms','ms_no_ocl','ocl_init_ms','read_ms','upload_ms','io_ms','file_write_ms','output_size_bytes','rc','decomp_ms','decomp_read_ms','decomp_upload_ms','decomp_io_ms','decomp_rc','decomp_ok']
    completed = set()
    tmp_out_csv_global = None
    if resume and out_csv.exists():
        # Collect completed runs to avoid re-running them
        with open(out_csv, 'r', newline='') as fh:
            r = csv.reader(fh)
            try:
                _ = next(r)
            except StopIteration:
                pass
            for row in r:
                if len(row) > 6:
                    try:
                        run_idx = int(row[6])
                    except Exception:
                        continue
                    # Normalize path to absolute to avoid mismatch between relative vs absolute CSV entries
                    try:
                        p_abs = os.path.abspath(row[0])
                    except Exception:
                        p_abs = row[0]
                    # Consider 'completed' only when rc==0 and decomp_ok==1 (if present)
                    rc_val = row[14] if len(row) > 14 else ''
                    decomp_ok_val = row[20] if len(row) > 20 else ''
                    try:
                        rc_ok = (int(rc_val) == 0)
                    except Exception:
                        rc_ok = False
                    try:
                        decomp_ok = (int(decomp_ok_val) == 1)
                    except Exception:
                        decomp_ok = False
                    if rc_ok and decomp_ok:
                        completed.add((p_abs, row[2], row[3], run_idx))
                print(f'Resume mode: found {len(completed)} completed runs in {out_csv}')
    # Write header if not resuming/CSV not exists
    if not (resume and out_csv.exists()):
        with open(out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header)

    for f in files:
        size = f.stat().st_size
        file_id = compute_sha1_id(str(f))
        file_abs = str(f.resolve())
        print(f'Processing: {f} size={size}')
        for runner in runners:
            CMD_BIN = str(LZO_GPU) if runner == 'standalone' else str(LZO_CLIENT)
            if runner == 'daemon' and not Path('/tmp/lzo_gpu_daemon.sock').exists():
                print('Skipping daemon (socket not found)')
                continue
            # If daemon is requested, ensure that only one daemon process is alive
            if runner == 'daemon':
                try:
                    p = subprocess.run(['pgrep', '-a', 'lzo_gpu_daemon'], capture_output=True, text=True)
                    procs = [l for l in p.stdout.splitlines() if l.strip()]
                    if len(procs) == 0:
                        print('No lzo_gpu_daemon processes found — skipping daemon tests')
                        continue
                    if len(procs) > 1:
                        print('Error: Multiple lzo_gpu_daemon processes found:')
                        for pr in procs: print('  ', pr)
                        print('Please ensure only one daemon is running (use `pkill -TERM -f lzo_gpu_daemon` or stop manually).')
                        print('Skipping daemon tests.')
                        continue
                except Exception:
                    # fall back to naive check
                    pass
            for mode in modes:
                mt_flag = 0; async_flag = 0
                env_base = os.environ.copy()
                if mode == 'zero':
                    env_base['LZO_STANDARD_COPY'] = '0'; env_base['LZO_MT_IO'] = '0'
                elif mode == 'zero+mt':
                    env_base['LZO_STANDARD_COPY'] = '0'; env_base['LZO_MT_IO'] = '1'; env_base['LZO_MT_IO_THREADS'] = str(mt_threads)
                    mt_flag = 1
                elif mode == 'std':
                    env_base['LZO_STANDARD_COPY'] = '1'; env_base['LZO_MT_IO'] = '0'
                elif mode == 'std+mt':
                    env_base['LZO_STANDARD_COPY'] = '1'; env_base['LZO_MT_IO'] = '1'; env_base['LZO_MT_IO_THREADS'] = str(mt_threads); mt_flag = 1
                elif mode == 'std_async':
                    env_base['LZO_STANDARD_COPY'] = '1'; env_base['LZO_MT_IO'] = '0'; env_base['LZO_ASYNC_UPLOAD'] = '1'; async_flag = 1
                elif mode == 'std+mt_async':
                    env_base['LZO_STANDARD_COPY'] = '1'; env_base['LZO_MT_IO'] = '1'; env_base['LZO_ASYNC_UPLOAD'] = '1'; env_base['LZO_MT_IO_THREADS'] = str(mt_threads); mt_flag = 1; async_flag = 1
                else:
                    # unknown mode: skip
                    continue

                for run in range(1, runs+1):
                    # skip runs that are already done when resuming
                    if resume:
                        key = (file_abs, runner, mode, run)
                        if key in completed:
                            # Skip already completed runs when resuming
                            print(f"Skipping {f.name} {runner} {mode} run {run} (already completed)")
                            continue
                    out_tmp = Path(TMPDIR.name) / f"out.{file_id}.{runner}.{mode}.run{run}.lzo"
                    run_log = logs_dir / f"{f.name}.{file_id}.{runner}.{mode}.run{run}.log"
                    if runner == 'standalone':
                        cmd = [CMD_BIN, str(f), '-o', str(out_tmp)]
                    else:
                        # Client expects: input + -o <output>; do not pass positional output argument
                        cmd = [CMD_BIN, str(f), '-o', str(out_tmp)]
                    print(f'Running {runner} {mode} run {run} -> {run_log}')
                    start_ns = time.perf_counter_ns()
                    # Attempt to create the run_log; if not possible (e.g., ENOSPC), fall back to a tmp file
                    tmp_run_log = None
                    try:
                        outfh = open(run_log, 'wb')
                    except OSError as e:
                        print(f"Warning: cannot write run log {run_log} ({e}); falling back to TMP file")
                        tmp_run_log = Path(TMPDIR.name) / f"out.{file_id}.{runner}.{mode}.run{run}.log.tmp"
                        outfh = open(tmp_run_log, 'wb')
                    with outfh:
                        proc = subprocess.run(cmd, env=env_base, cwd=str(LZO_GPU.parent), stdout=outfh, stderr=subprocess.STDOUT)
                    # If we used a tmp_run_log fallback, try to move it to the intended run_log now
                    if tmp_run_log is not None and tmp_run_log.exists():
                        try:
                            shutil.copy2(tmp_run_log, run_log)
                            tmp_run_log.unlink()
                        except OSError as e:
                            print(f"Warning: could not copy tmp run log {tmp_run_log} to {run_log} ({e}) -- skipping copy")
                    end_ns = time.perf_counter_ns()
                    rc = proc.returncode
                    ms = (end_ns - start_ns) / 1e6

                    # parse run_log to extract read/upload breakdowns
                    content = run_log.read_text(encoding='utf-8', errors='ignore') if run_log.exists() else ''
                    read_ms = None
                    upload_ms = None
                    io_ms = None
                    file_write_ms = None
                    ocl_init_ms = None
                    # attempt to find 'File Read' and 'Data Upload' labels
                    m_read = re.search(r"(File Read|read input)\s*[:]?\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
                    if m_read:
                        read_ms = parse_time_to_ms(m_read.group(2) + (' ' + (m_read.group(3) or 'ms')))
                    m_up = re.search(r"(Data Upload|create\+upload|create\+upload)\s*[:]?\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
                    if m_up:
                        upload_ms = parse_time_to_ms(m_up.group(2) + (' ' + (m_up.group(3) or 'ms')))
                    # File write
                    m_write = re.search(r"(File Write|file write)\s*[:]?:?\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
                    if m_write:
                        file_write_ms = parse_time_to_ms(m_write.group(2) + (' ' + (m_write.group(3) or 'ms')))
                    # Set io_ms to the sum of read/upload parts when available. For zero-copy
                    # runs we may have only read_ms available, so treat missing upload_ms as 0.
                    if read_ms is not None and upload_ms is not None:
                        io_ms = read_ms + upload_ms
                    elif read_ms is not None:
                        io_ms = read_ms
                    elif upload_ms is not None:
                        io_ms = upload_ms

                    # OCL init parse; exclude OCL init from ms when possible
                    m_ocl = re.search(r"(2\.\s*OCL Init|OCL Init|OpenCL init)\s*:?\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
                    if not m_ocl:
                        # try an alternative: '2. OCL Init            : 35.000 ms'
                        m_ocl = re.search(r"OCL Init\s*:?\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
                    if m_ocl:
                        try:
                            # patternA: (label) (num) (unit) -> numeric is group 2
                            # patternB (fallback): (num) (unit) -> numeric is group 1
                            if m_ocl.lastindex and m_ocl.lastindex >= 2 and re.match(r'^[0-9.]+$', m_ocl.group(2)):
                                val = m_ocl.group(2)
                                unit = m_ocl.group(3) if m_ocl.lastindex >= 3 else 'ms'
                            else:
                                val = m_ocl.group(1)
                                unit = m_ocl.group(2) if m_ocl.lastindex >= 2 else 'ms'
                            ocl_init_ms = parse_time_to_ms(val + ' ' + (unit or 'ms'))
                        except Exception:
                            ocl_init_ms = ''

                    outsize = out_tmp.stat().st_size if out_tmp.exists() else 0

                    # decompression + verify
                    decomp_out = Path(TMPDIR.name) / f"decomp.{file_id}.{runner}.{mode}.run{run}.out"
                    decomp_log = logs_dir / f"{f.name}.{file_id}.{runner}.{mode}.run{run}.decomp.log"
                    decomp_rc = 0; decomp_seconds = ''
                    decomp_read = None
                    decomp_upload = None
                    decomp_io = None
                    decomp_ok = 0
                    if out_tmp.exists():
                        if runner == 'standalone':
                            cmd_decomp = [CMD_BIN, '-d', str(out_tmp), '-o', str(decomp_out)]
                        else:
                            cmd_decomp = [CMD_BIN, '-d', str(out_tmp), '-o', str(decomp_out)]
                        start_ns2 = time.perf_counter_ns()
                        tmp_decomp_log = None
                        try:
                            outfh = open(decomp_log, 'wb')
                        except OSError as e:
                            print(f"Warning: cannot write decomp log {decomp_log} ({e}); falling back to TMP file")
                            tmp_decomp_log = Path(TMPDIR.name) / f"decomp.{file_id}.{runner}.{mode}.run{run}.log.tmp"
                            outfh = open(tmp_decomp_log, 'wb')
                        with outfh:
                            proc2 = subprocess.run(cmd_decomp, env=env_base, cwd=str(LZO_GPU.parent), stdout=outfh, stderr=subprocess.STDOUT)
                        if tmp_decomp_log is not None and tmp_decomp_log.exists():
                            try:
                                shutil.copy2(tmp_decomp_log, decomp_log)
                                tmp_decomp_log.unlink()
                            except OSError as e:
                                print(f"Warning: could not copy tmp decomp log {tmp_decomp_log} to {decomp_log} ({e}) -- skipping copy")
                        end_ns2 = time.perf_counter_ns()
                        decomp_seconds = (end_ns2 - start_ns2) / 1e6
                        decomp_rc = proc2.returncode
                        dcontent = decomp_log.read_text(encoding='utf-8', errors='ignore') if decomp_log.exists() else ''
                        dm_read = re.search(r"(File Read|read input)\s*[:]?:?\s*([0-9.]+)\s*(ms|us|s)?", dcontent, flags=re.I)
                        if dm_read:
                            decomp_read = parse_time_to_ms(dm_read.group(2) + (' ' + (dm_read.group(3) or 'ms')))
                        dm_up = re.search(r"(Data Upload|create\+upload|create\+upload|Data Download)\s*[:]?:?\s*([0-9.]+)\s*(ms|us|s)?", dcontent, flags=re.I)
                        if dm_up:
                            decomp_upload = parse_time_to_ms(dm_up.group(2) + (' ' + (dm_up.group(3) or 'ms')))
                        # Decomp file write
                        dm_write = re.search(r"(File Write|file write)\s*[:]?:?\s*([0-9.]+)\s*(ms|us|s)?", dcontent, flags=re.I)
                        if dm_write:
                            # currently we don't track per-decomp file_write in CSV; keep as local var
                            decomp_file_write = parse_time_to_ms(dm_write.group(2) + (' ' + (dm_write.group(3) or 'ms')))
                        # Compute decomp_io with available parts
                        if decomp_read is not None and decomp_upload is not None:
                            decomp_io = decomp_read + decomp_upload
                        elif decomp_read is not None:
                            decomp_io = decomp_read
                        elif decomp_upload is not None:
                            decomp_io = decomp_upload
                        if decomp_out.exists():
                            try:
                                if decomp_out.read_bytes() == f.read_bytes():
                                    decomp_ok = 1
                                else:
                                    decomp_ok = 0
                            except Exception:
                                decomp_ok = 0
                    # write csv row
                    # effective ms (exclude any OCL init time) for throughput eval
                    eff_ms = ms
                    try:
                        if isinstance(ocl_init_ms, (int, float)) and ocl_init_ms > 0:
                            if eff_ms > ocl_init_ms:
                                eff_ms = eff_ms - ocl_init_ms
                    except Exception:
                        pass
                    # Write CSV row; if writing to the configured out_csv fails due to disk errors, fall back to a tmp CSV in TMPDIR
                    tmp_out_csv = None
                    try:
                        with open(out_csv, 'a', newline='') as fh:
                            w = csv.writer(fh)
                            w.writerow([file_abs, size, runner, mode, async_flag, mt_flag, run, f"{ms:.3f}", f"{eff_ms:.3f}", f"{ocl_init_ms:.3f}" if ocl_init_ms is not None else '', f"{read_ms:.3f}" if read_ms is not None else '', f"{upload_ms:.3f}" if upload_ms is not None else '', f"{io_ms:.3f}" if io_ms is not None else '', f"{file_write_ms:.3f}" if file_write_ms is not None else '', outsize, rc, f"{decomp_seconds:.3f}" if decomp_seconds != '' else '', f"{decomp_read:.3f}" if decomp_read is not None else '', f"{decomp_upload:.3f}" if decomp_upload is not None else '', f"{decomp_io:.3f}" if decomp_io is not None else '', decomp_rc, decomp_ok])
                    except OSError as e:
                        try:
                            tmp_out_csv = Path(TMPDIR.name) / 'benchmark_fallback.csv'
                            tmp_out_csv_global = tmp_out_csv
                            with open(tmp_out_csv, 'a', newline='') as fh:
                                w = csv.writer(fh)
                                w.writerow([file_abs, size, runner, mode, async_flag, mt_flag, run, f"{ms:.3f}", f"{eff_ms:.3f}", f"{ocl_init_ms:.3f}" if ocl_init_ms is not None else '', f"{read_ms:.3f}" if read_ms is not None else '', f"{upload_ms:.3f}" if upload_ms is not None else '', f"{io_ms:.3f}" if io_ms is not None else '', f"{file_write_ms:.3f}" if file_write_ms is not None else '', outsize, rc, f"{decomp_seconds:.3f}" if decomp_seconds != '' else '', f"{decomp_read:.3f}" if decomp_read is not None else '', f"{decomp_upload:.3f}" if decomp_upload is not None else '', f"{decomp_io:.3f}" if decomp_io is not None else '', decomp_rc, decomp_ok])
                            print(f"Warning: cannot write to out_csv {out_csv} ({e}). Falling back to TMP file: {tmp_out_csv}")
                        except Exception as e2:
                            print(f"Error: unable to write fallback CSV {tmp_out_csv} ({e2}). Skipping CSV write for this run.")

                    # Copy logs to keep logs dir and optionally artifacts on failure
                    if keep_logs:
                        keep_logs.mkdir(parents=True, exist_ok=True)
                        # Check free space on destination; if low, avoid copying logs
                        free = get_free_space_bytes(keep_logs)
                        if free < (20 * 1024 * 1024):
                            print(f"Warning: low disk space ({free} bytes) on {keep_logs}, skipping log copy")
                        else:
                            try:
                                shutil.copy2(run_log, keep_logs / run_log.name)
                            except OSError as e:
                                # Don't crash the run if the filesystem fills up or copy fails
                                print(f"Warning: failed to copy run log to keep_logs ({e}). Continuing without copying this log.")
                            try:
                                if decomp_log.exists(): shutil.copy2(decomp_log, keep_logs / decomp_log.name)
                            except OSError as e:
                                print(f"Warning: failed to copy decomp log to keep_logs ({e}). Continuing without copying this log.")
                    # If requested, copy artifacts (compressed outputs/decompressed outputs) for failed runs to keep_logs
                    if keep_artifacts_on_fail and (rc != 0 or decomp_ok != 1) and keep_logs:
                        try:
                            if out_tmp.exists(): shutil.copy2(out_tmp, keep_logs / out_tmp.name)
                        except OSError as e:
                            print(f"Warning: failed to copy artifact to keep_logs ({e}). Continuing without copying this artifact.")
                        try:
                            if decomp_out.exists(): shutil.copy2(decomp_out, keep_logs / decomp_out.name)
                        except OSError as e:
                            print(f"Warning: failed to copy decomp_out to keep_logs ({e}). Continuing without copying this artifact.")

                    # Remove temporary artifacts (out_tmp & decomp_out) *after* optionally copying them
                    try:
                        if out_tmp.exists():
                            out_tmp.unlink()
                    except OSError as e:
                        print(f"Warning: could not delete tmp artifact {out_tmp}: {e}")
                    try:
                        if decomp_out.exists():
                            decomp_out.unlink()
                    except OSError as e:
                        print(f"Warning: could not delete tmp decomp output {decomp_out}: {e}")

    # If we had to use a fallback TMP CSV, attempt to flush it back to the requested out_csv
    if tmp_out_csv_global and tmp_out_csv_global.exists():
        try:
            free_dest = get_free_space_bytes(out_csv.parent)
            if free_dest > (10 * 1024 * 1024):
                if out_csv.exists() and out_csv.stat().st_size > 0:
                    # Append rows from tmp to out_csv
                    with open(out_csv, 'a', newline='') as dest_fh, open(tmp_out_csv_global, 'r', newline='') as src_fh:
                        for line in src_fh:
                            dest_fh.write(line)
                else:
                    shutil.copy2(tmp_out_csv_global, out_csv)
                print(f"Flushed fallback CSV {tmp_out_csv_global} to {out_csv}")
            else:
                print(f"Warning: insufficient space to flush fallback CSV {tmp_out_csv_global} to {out_csv}; leaving it in TMP at {tmp_out_csv_global}")
        except Exception as e:
            print(f"Warning: Unable to flush fallback CSV {tmp_out_csv_global} to {out_csv} ({e})")
    TMPDIR.cleanup()
    print('Wrote CSV:', out_csv)
    print('Logs:', logs_dir)
    return 0


def cmd_run(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench run')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-n', '--runs', default=3, type=int)
    parser.add_argument('-t', '--mt-threads', default=1, type=int)
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('-m', '--min-size', default=0, type=int)
    parser.add_argument('--runners', default='standalone', help='comma-separated: standalone,daemon')
    parser.add_argument('--modes', default='zero,zero+mt,std,std+mt', help='comma-separated modes to test')
    parser.add_argument('--out-csv', default='benchmark_mt_io_results.csv')
    parser.add_argument('--keep-logs', default=None, help='directory to copy logs into')
    parser.add_argument('--keep-artifacts-on-fail', action='store_true')
    parser.add_argument('--resume', action='store_true', help='resume from existing CSV, skipping completed runs')
    args = parser.parse_args(argv)
    runners = [r.strip() for r in args.runners.split(',') if r.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    files = gather_files(args.paths, args.recursive, args.min_size)
    if not files:
        print('No files found to run on, exiting')
        return 1
    return run_bench(files, runs=args.runs, mt_threads=args.mt_threads, runners=runners, out_csv=args.out_csv, keep_logs=Path(args.keep_logs) if args.keep_logs else None, modes=modes, keep_artifacts_on_fail=args.keep_artifacts_on_fail, resume=args.resume)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog='tools/bench.py')
    sub = parser.add_subparsers(dest='cmd')
    # NOTE: paths are parsed by the cmd_run parser to avoid double-parsing in main
    sub_run = sub.add_parser('run', help='Run benchmarks')
    # keep rest args parsing to cmd_run to reuse logic
    args, rest = parser.parse_known_args(argv)
    if args.cmd == 'run':
        return cmd_run(rest)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
