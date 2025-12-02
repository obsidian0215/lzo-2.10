#!/usr/bin/env python3
"""
tools/analysis.py - Consolidated analysis utilities for LZO benchmarks

Subcommands:
  parse    - Parse logs under exp_results to generate per-sample JSON/CSV breakdowns
  throughput - Compute per-run throughput from CSV and generate summary
  variance - Analyze CV across runs using breakdown JSONs
  analysis - Generate per-sample Markdown analysis reports
  all - Run parse + throughput + variance + analysis in sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'lzo_gpu' / 'exp_results'

def parse_time_to_ms(val_str: str | None) -> float:
    if not val_str:
        return 0.0
    s = str(val_str).strip()
    m = re.match(r"^([0-9.]+)\s*(ms|us|s)?$", s, flags=re.I)
    if not m:
        s2 = s.replace(',', '')
        m = re.match(r"^([0-9.]+)\s*(ms|us|s)?$", s2, flags=re.I)
        if not m:
            return 0.0
    val = float(m.group(1)); unit = (m.group(2) or 'ms').lower()
    if unit == 'ms': return val
    if unit == 'us': return val/1000.0
    if unit == 's': return val*1000.0
    return val


LABEL_TO_KEY = {
    'file read': 'file_read_ms',
    'blocking calc': 'blocking_calc_ms',
    'buffer alloc (in)': 'buffer_alloc_in_ms',
    'buffer alloc (out)': 'buffer_alloc_out_ms',
    'buffer alloc (len)': 'buffer_alloc_len_ms',
    'buffer alloc': 'buffer_alloc_ms',
    'data upload': 'data_upload_ms',
    'setup args': 'setup_args_ms',
    'kernel exec': 'kernel_exec_ms',
    'download (len)': 'download_len_ms',
    'download (bulk)': 'download_bulk_ms',
    'download total': 'download_total_ms',
    'data download': 'data_download_ms',
    'file write': 'file_write_ms',
    'cleanup': 'cleanup_ms',
    'ocl init': 'ocl_init_ms',
    'kernel load': 'kernel_load_ms',
    'total': 'total_ms',
}

def canonicalize_label(lbl: str):
    l = lbl.strip().lower()
    l = re.sub(r"\s+", ' ', l)
    return LABEL_TO_KEY.get(l, None)


def parse_file(path: Path) -> tuple[str, dict]:
    content = path.read_text(encoding='utf-8', errors='ignore')
    t = 'decompression' if '.decomp.' in path.name or path.name.endswith('.decomp.log') else 'compression'
    if re.search(r'解压缩|Decompression', content, flags=re.I):
        t = 'decompression'
    if re.search(r'压缩成功|Compression', content, flags=re.I) and not re.search(r'解压缩|Decompression', content, flags=re.I):
        t = 'compression'

    # parse numbers
    results = {}
    # block breakdown blocks
    block_re = re.compile(r"=== Time Breakdown \((Compression|Decompression)\) ===\s*(.*?)\n\s*TOTAL\s*:?[ \t]*([0-9.]+)[ \t]*(ms|us|s)?", flags=re.S | re.I)
    m = block_re.search(content)
    if m:
        inner = m.group(2)
        for lm in re.finditer(r"^\s*\d+\.\s*([^:]+?)\s*:\s*([0-9.]+)\s*(ms|us|s)?\s*$", inner, flags=re.M | re.I):
            lbl = lm.group(1)
            val = lm.group(2) + (' ' + (lm.group(3) or 'ms'))
            key = canonicalize_label(lbl) or (lbl.strip().lower().replace(' ', '_') + '_ms')
            results[key] = parse_time_to_ms(val)
        total_val = m.group(3) + (' ' + (m.group(4) or 'ms'))
        results['total_ms'] = parse_time_to_ms(total_val)
        return t, results
    # fallback: numbered lines anywhere
    for lm in re.finditer(r"^\s*\d+\.\s*([^:]+?)\s*:\s*([0-9.]+)\s*(ms|us|s)?\s*$", content, flags=re.M | re.I):
        lbl = lm.group(1)
        val = lm.group(2) + (' ' + (lm.group(3) or 'ms'))
        key = canonicalize_label(lbl) or (lbl.strip().lower().replace(' ', '_') + '_ms')
        results[key] = parse_time_to_ms(val)
    tm = re.search(r"^TOTAL\s*:?[ \t]*([0-9.]+)\s*(ms|us|s)?\s*$", content, flags=re.M | re.I)
    if tm:
        results['total_ms'] = parse_time_to_ms(tm.group(1) + (' ' + (tm.group(2) or 'ms')))
    if 'total_ms' not in results and results:
        results['total_ms'] = sum(results.values())

    block_times = []
    for m in re.finditer(r"BLOCK_WRITE\s+(\d+)\s+len=(\d+)\s*:\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I):
        block_times.append(parse_time_to_ms(m.group(3) + (' ' + (m.group(4) or 'ms'))))
    if block_times:
        results['block_write_count'] = float(len(block_times))
        results['block_write_total_ms'] = sum(block_times)
        results['block_write_mean_ms'] = mean(block_times)
        results['block_write_max_ms'] = max(block_times)
        results['block_write_std_ms'] = stdev(block_times) if len(block_times) > 1 else 0.0
    cm = re.search(r"COALESCE_COPY\s*:\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
    if cm:
        results['coalesce_copy_ms'] = parse_time_to_ms(cm.group(1) + (' ' + (cm.group(2) or 'ms')))
    cw = re.search(r"COALESCE_WRITE\s*:\s*([0-9.]+)\s*(ms|us|s)?", content, flags=re.I)
    if cw:
        results['coalesce_write_ms'] = parse_time_to_ms(cw.group(1) + (' ' + (cw.group(2) or 'ms')))
    return t, results


def aggregate(entries: list[dict]):
    keys = set()
    for e in entries:
        keys.update(e.keys())
    out = {}
    for k in sorted(keys):
        vals = [e.get(k, 0.0) for e in entries]
        if all(v == 0.0 for v in vals):
            out[k] = {'n': len(vals), 'mean': 0.0, 'stddev': 0.0}
        else:
            s = stdev(vals) if len(vals) > 1 else 0.0
            out[k] = {'n': len(vals), 'mean': mean(vals), 'stddev': s}
    return out


def find_logs_for_sample(sample_name: str, logs_dir: Path | None = None) -> list[Path]:
    base_dir = logs_dir or EXP_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f'exp_results dir missing: {base_dir}')
    results = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and child.name.startswith('lzo-bench-logs'):
            for f in sorted(child.iterdir()):
                if f.name.startswith(sample_name) and f.name.endswith('.log'):
                    results.append(f)
    for f in sorted(base_dir.iterdir()):
        if f.is_file() and f.name.startswith(sample_name) and f.name.endswith('.log'):
            results.append(f)
    return results


def clean_sample_name(s: str) -> str:
    """Clean sample portion from log name: remove per-run 8-hex hash suffix if present."""
    return re.sub(r"\.[0-9a-fA-F]{8}$", "", s)


def cmd_parse(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench parse')
    parser.add_argument('--file', '-f', dest='sample', default='sample_real_1.5gb.txt')
    parser.add_argument('--logs-dir', dest='logs_dir', default=None)
    args = parser.parse_args(argv)
    logs = find_logs_for_sample(args.sample, Path(args.logs_dir) if args.logs_dir else None)
    if not logs:
        print('No logs found for', args.sample); return 2
    groups = {}
    # Accept logname patterns where the sample portion may include file extensions
    # and also optional per-run hash segments (e.g., sample.json.12345678.standalone.zero.run1.log)
    fname_pattern = re.compile(r"^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$")
    _clean_sample = clean_sample_name
    for p in logs:
        m = fname_pattern.match(p.name)
        if not m:
            print('SKIP (unrecognized name):', p.name); continue
        sample_face = _clean_sample(m.group('sample'))
        # Accept if the requested sample equals the parsed sample or is a dot-prefixed prefix
        if not (sample_face == args.sample or sample_face.startswith(args.sample + '.') or sample_face.startswith(args.sample)):
            print('SKIP (sample mismatch):', p.name, 'parsed sample', sample_face, 'requested', args.sample); continue
        runner = m.group('runner'); mode = m.group('mode'); runnum = int(m.group('run'))
        try:
            typ, parsed = parse_file(p)
        except Exception as e:
            print('Failed to parse', p, '->', e)
            continue
        key = (runner, mode, typ)
        groups.setdefault(key, []).append(parsed)
    # produce JSON and CSV
    # Determine canonical sample name for outputs
    out_json = {}
    # derive canonical sample prefix from first matched log (to prefer full sample with extension)
    canonical_sample = None
    for p in logs:
        m = fname_pattern.match(p.name)
        if m:
            canonical_sample = clean_sample_name(m.group('sample'))
            break
    if not canonical_sample:
        canonical_sample = args.sample
    csv_rows = []
    cols = ['runner','mode','type','n','total_ms','file_read_ms','data_upload_ms','kernel_exec_ms','download_total_ms','file_write_ms','buffer_alloc_in_ms','buffer_alloc_out_ms','buffer_alloc_len_ms','buffer_alloc_ms','ocl_init_ms','kernel_load_ms','blocking_calc_ms','download_len_ms','download_bulk_ms','data_download_ms','setup_args_ms','cleanup_ms']
    for key, entries in sorted(groups.items()):
        runner, mode, typ = key
        numeric_entries = entries
        agg = aggregate(numeric_entries)
        out_json.setdefault(runner, {}).setdefault(mode, {})[typ] = agg
        row = [runner, mode, typ, len(entries)]
        for c in cols[4:]:
            val = agg.get(c, {'mean': 0.0})['mean']
            row.append(val)
        csv_rows.append(row)
    out_dir = Path(args.logs_dir) if args.logs_dir else EXP_DIR
    out_json_path = out_dir / f"{canonical_sample}_breakdown_summary_all.json"
    csv_path = out_dir / f"{canonical_sample}_stage_summary_all.csv"
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in csv_rows:
            fmt = [r[0], r[1], r[2], str(r[3])] + [f"{v:.3f}" for v in r[4:]]
            f.write(','.join(fmt) + '\n')
    print('Wrote:', out_json_path, csv_path)
    return 0


def cmd_parse_all(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench parse-all')
    parser.add_argument('--logs-dir', dest='logs_dir', default=None)
    args = parser.parse_args(argv)
    exp = Path(args.logs_dir) if args.logs_dir else EXP_DIR
    if not exp.exists():
        print('exp dir not found:', exp); return 2
    # discover sample prefixes from log file names
    found = set()
    for child in sorted(exp.iterdir()):
        if child.is_dir() and child.name.startswith('lzo-bench-'):
            for f in sorted(child.iterdir()):
                if f.is_file() and f.name.endswith('.log'):
                    m = re.match(r'^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$', f.name)
                    if m:
                        s = clean_sample_name(m.group('sample'))
                        found.add(s)
    for f in sorted(exp.iterdir()):
        if f.is_file() and f.name.endswith('.log'):
            m = re.match(r'^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$', f.name)
            if m:
                s = clean_sample_name(m.group('sample'))
                found.add(s)
    if not found:
        print('No sample logs discovered in', exp); return 1
    print('Discovered samples:', ', '.join(sorted(found)))
    # call cmd_parse for each sample
    for s in sorted(found):
        print('Parsing sample:', s)
        cmd_parse(['--file', s, '--logs-dir', str(exp)])
    return 0


def cmd_throughput(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench throughput')
    parser.add_argument('--csv', dest='csv', default='benchmark_mt_io_results.csv')
    parser.add_argument('--logs-dir', dest='logs_dir', default=None, help='Optional logs dir to search for per-run breakdowns to augment CSVs')
    parser.add_argument('--threads', dest='threads', default='1,2,3', help='comma-separated list of threads to find latest bench logs for when logs-dir not provided')
    args = parser.parse_args(argv)
    IN = Path(args.csv)
    OUT = IN.parent / (IN.stem + '_with_thr.csv')
    SUMMARY = IN.parent / 'benchmark_throughput_summary.txt'
    if not IN.exists():
        print('Input CSV not found:', IN); return 1
    rows = []
    with IN.open('r', newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    def safe_float(x):
        try: return float(x)
        except: return None
    # Attempt to augment rows with per-run breakdowns (file_write_ms) when logs dir is available
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    # Build a log index for fast lookup: key -> (sample_clean, runner, mode, run) -> path
    def build_log_index(base_dir) -> tuple[dict, dict]:
        run_map = {}
        decomp_map = {}
        if base_dir is None:
            return run_map, decomp_map
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return run_map, decomp_map
        # scan recursively for *.log once; use regex to extract keys
        pattern1 = re.compile(r"^(?P<sample>.+)\.(?P<fileid>[0-9a-fA-F]{8})\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$")
        pattern2 = re.compile(r"^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$")
        for p in base_dir.rglob('*.log'):
            name = p.name
            m = pattern1.match(name) or pattern2.match(name)
            if not m:
                continue
            try:
                sample_raw = m.group('sample')
                runner = m.group('runner')
                mode = m.group('mode')
                rn = int(m.group('run'))
            except Exception:
                continue
            sample_cleaned = clean_sample_name(sample_raw)
            key = (sample_cleaned, runner, mode, rn)
            if name.endswith('.decomp.log'):
                decomp_map[key] = p
            else:
                run_map[key] = p
        return run_map, decomp_map

    def find_log_for_row_using_index(r, run_map, decomp_map):
        sample = Path(r.get('file', '')).name
        if not sample:
            return None
        sample_cleaned = clean_sample_name(sample)
        try:
            key = (sample_cleaned, r.get('runner'), r.get('mode'), int(r.get('run')))
        except Exception:
            return None
        # Prefer main run log; fall back to decomp log
        if key in run_map:
            return run_map[key]
        if key in decomp_map:
            return decomp_map[key]
        # fallback: try matching by sample prefix (some logs include graphic names with dots)
        # look for keys where sample_cleaned is prefix of indexed key
        for k in run_map.keys():
            if k[0].startswith(sample_cleaned) and k[1] == r.get('runner') and k[2] == r.get('mode') and k[3] == int(r.get('run')):
                return run_map[k]
        for k in decomp_map.keys():
            if k[0].startswith(sample_cleaned) and k[1] == r.get('runner') and k[2] == r.get('mode') and k[3] == int(r.get('run')):
                return decomp_map[k]
        return None

    # Build a log index once for spectral lookup. If logs_dir not specified, find latest bench dirs
    if logs_dir:
        try:
            run_map, decomp_map = build_log_index(logs_dir)
        except Exception:
            run_map, decomp_map = ({}, {})
    else:
        # examine threads/exp dir to find latest bench directories and index them
        run_map, decomp_map = ({}, {})
        try:
            threads = [int(x.strip()) for x in args.threads.split(',') if x.strip()]
            logs_base = EXP_DIR
            # find latest for each thread
            for th in threads:
                candidates = []
                for p in sorted(logs_base.iterdir()):
                    if p.is_dir() and p.name.startswith(f'bench_mt{th}_runs'):
                        candidates.append(p)
                if candidates:
                    latest = candidates[-1]
                    # index this bench dir
                    _run_map, _decomp_map = build_log_index(latest)
                    # merge into run_map/decomp_map
                    run_map.update(_run_map)
                    decomp_map.update(_decomp_map)
        except Exception:
            run_map, decomp_map = ({}, {})

    for r in rows:
        # Try to augment file_write_ms from logs if available
        if 'file_write_ms' not in r or not r.get('file_write_ms'):
            p = find_log_for_row_using_index(r, run_map, decomp_map)
            if p:
                try:
                    _, parsed = parse_file(p)
                    val = parsed.get('file_write_ms') if parsed else None
                    if val is not None:
                        r['file_write_ms'] = f"{val:.3f}"
                except Exception:
                    pass
        # Fill missing io_ms from read/upload parts for datasets where zero-copy only writes read_ms
        io_val = safe_float(r.get('io_ms'))
        if io_val is None:
            rd = safe_float(r.get('read_ms'))
            up = safe_float(r.get('upload_ms'))
            if rd is not None or up is not None:
                rd = rd or 0.0
                up = up or 0.0
                r['io_ms'] = f"{(rd + up):.3f}"
        # Prefer effective ms excluding OCL init if available
        ms = safe_float(r.get('ms_no_ocl') or r.get('ms'))
        sizeb = safe_float(r.get('size_bytes'))
        rc = int(r.get('rc', '0') or 0)
        thr = ''
        if rc == 0 and sizeb and ms and ms > 0:
            mb = sizeb / (1024.0*1024.0)
            thr_val = (mb * 1000.0) / ms
            thr = f"{thr_val:.3f}"
        r['throughput_mbps'] = thr
    # write augmented CSV
    # Make sure augmented keys (file_write_ms etc) are included in fieldnames
    fieldnames = list(rows[0].keys()) if rows else []
    if 'file_write_ms' not in fieldnames:
        # try to insert after io_ms if present, else append
        try:
            idx = fieldnames.index('io_ms') + 1
        except ValueError:
            try:
                idx = fieldnames.index('output_size_bytes')
            except ValueError:
                idx = len(fieldnames)
        fieldnames.insert(idx, 'file_write_ms')
    # Ensure all rows have the key
    for r in rows:
        if 'file_write_ms' not in r:
            r['file_write_ms'] = ''
    with OUT.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader();
        for r in rows:
            w.writerow(r)
    # aggregate summary like compute_bench_throughput.py
    groups = defaultdict(list)
    for r in rows:
        key = (r['file'], r['runner'], r['mode'], r['async'], r['mt'])
        thr = safe_float(r.get('throughput_mbps') or '')
        if thr is not None:
            groups[key].append(thr)
    per_file = defaultdict(list); per_combo = defaultdict(list)
    for (file, runner, mode, async_f, mt_f), thr_list in groups.items():
        if not thr_list: continue
        avg = mean(thr_list)
        sd = stdev(thr_list) if len(thr_list) > 1 else 0.0
        entry = {'runner': runner, 'mode': mode, 'async': async_f, 'mt': mt_f, 'mean': avg, 'stddev': sd, 'n': len(thr_list)}
        per_file[file].append(entry)
        per_combo[(runner, mode, async_f, mt_f)].append(avg)
    with SUMMARY.open('w') as fh:
        fh.write('Benchmark throughput summary\n')
        fh.write('Runs processed: %d\n\n' % len(rows))
        fh.write('Top combos per file (mean MB/s)\n')
        for file, entries in sorted(per_file.items()):
            fh.write(f'File: {file}\n')
            entries.sort(key=lambda e: e['mean'], reverse=True)
            for i, e in enumerate(entries[:5], 1):
                fh.write(f"  {i}. runner={e['runner']} mode={e['mode']} async={e['async']} mt={e['mt']} -> mean={e['mean']:.2f} MB/s n={e['n']} stddev={e['stddev']:.2f}\n")
            fh.write('\n')
        fh.write('\nOverall means by combo\n')
        combo_entries = []
        for combo_key, means in per_combo.items():
            combo_entries.append((combo_key, mean(means), stdev(means) if len(means)>1 else 0.0, len(means)))
        combo_entries.sort(key=lambda x: x[1], reverse=True)
        for (runner, mode, async_f, mt_f), mean_v, sd_v, count in combo_entries:
            fh.write(f"runner={runner} mode={mode} async={async_f} mt={mt_f} -> mean={mean_v:.2f} MB/s across {count} files (std={sd_v:.2f})\n")
    print('Wrote:', OUT, SUMMARY)
    return 0


def cmd_variance(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench variance')
    parser.add_argument('--dir', default=None)
    parser.add_argument('--cv-threshold', default=0.25, type=float)
    parser.add_argument('--out-prefix', default='variance_read_write')
    parser.add_argument('--file', default=None)
    args = parser.parse_args(argv)
    exp = Path(args.dir) if args.dir else EXP_DIR
    jsons = sorted(exp.glob('*_breakdown_summary_all.json'))
    if args.file:
        jsons = [p for p in jsons if p.name.startswith(args.file)]
    if not jsons:
        print('No breakdown jsons found'); return 1
    rows = []
    flags = []
    for j in jsons:
        sample_name = j.name.replace('_breakdown_summary_all.json','')
        data = json.load(open(j))
        for runner in sorted(data.keys()):
            for mode in sorted(data[runner].keys()):
                for typ in sorted(data[runner][mode].keys()):
                    d = data[runner][mode][typ]
                    n_file_read = d.get('file_read_ms', {}).get('n', 0)
                    mean_file_read = d.get('file_read_ms', {}).get('mean', 0.0)
                    sd_file_read = d.get('file_read_ms', {}).get('stddev', 0.0)
                    n_file_write = d.get('file_write_ms', {}).get('n', 0)
                    mean_file_write = d.get('file_write_ms', {}).get('mean', 0.0)
                    sd_file_write = d.get('file_write_ms', {}).get('stddev', 0.0)
                    cv_read = (sd_file_read / mean_file_read) if mean_file_read else None
                    cv_write = (sd_file_write / mean_file_write) if mean_file_write else None
                    rows.append({'sample': sample_name, 'runner': runner, 'mode': mode, 'type': typ, 'n_read': n_file_read, 'mean_read_ms': mean_file_read, 'sd_read_ms': sd_file_read, 'cv_read': cv_read, 'n_write': n_file_write, 'mean_write_ms': mean_file_write, 'sd_write_ms': sd_file_write, 'cv_write': cv_write})
                    if (cv_read is not None and cv_read > args.cv_threshold) or (cv_write is not None and cv_write > args.cv_threshold):
                        flags.append({'sample': sample_name, 'runner': runner, 'mode': mode, 'type': typ, 'cv_read': cv_read, 'cv_write': cv_write})
    csv_path = exp / (args.out_prefix + '.csv')
    with open(csv_path, 'w') as fo:
        hdr = 'sample,runner,mode,type,n_read,mean_read_ms,sd_read_ms,cv_read,n_write,mean_write_ms,sd_write_ms,cv_write\n'
        fo.write(hdr)
        for r in rows:
            fo.write(','.join([r['sample'], r['runner'], r['mode'], r['type'], str(r['n_read']), f"{r['mean_read_ms']:.3f}", f"{r['sd_read_ms']:.3f}", f"{r['cv_read']:.5f}" if r['cv_read'] is not None else '', str(r['n_write']), f"{r['mean_write_ms']:.3f}", f"{r['sd_write_ms']:.3f}", f"{r['cv_write']:.5f}" if r['cv_write'] is not None else '']) + '\n')
    md_path = exp / (args.out_prefix + '.md')
    with open(md_path, 'w') as fm:
        fm.write('# Read/Write Variance and CV\n\n')
        fm.write(f'CV threshold: {args.cv_threshold}\n\n')
        fm.write('| sample | runner | mode | type | n_read | mean_read_ms | sd_read_ms | cv_read | n_write | mean_write_ms | sd_write_ms | cv_write |\n')
        fm.write('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            cv_read_s = f"{r['cv_read']:.3f}" if r['cv_read'] is not None else ''
            cv_write_s = f"{r['cv_write']:.3f}" if r['cv_write'] is not None else ''
            fm.write('| {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {} | {} | {:.3f} | {:.3f} | {} |\n'.format(r['sample'], r['runner'], r['mode'], r['type'], r['n_read'], r['mean_read_ms'], r['sd_read_ms'], cv_read_s, r['n_write'], r['mean_write_ms'], r['sd_write_ms'], cv_write_s))
        if flags:
            fm.write('\n## Flagged combos with high CV\n')
            fm.write('| sample | runner | mode | type | cv_read | cv_write |\n')
            fm.write('|---|---|---|---|---:|---:|\n')
            for f in flags:
                cv_r = f['cv_read'] if f['cv_read'] is not None else ''
                cv_w = f['cv_write'] if f['cv_write'] is not None else ''
                fm.write('| {} | {} | {} | {} | {} | {} |\n'.format(f['sample'], f['runner'], f['mode'], f['type'], f"{cv_r:.3f}" if cv_r != '' and cv_r is not None else '', f"{cv_w:.3f}" if cv_w != '' and cv_w is not None else ''))
    print('Wrote CSV:', csv_path)
    print('Wrote MD:', md_path)
    print('Total combos:', len(rows), 'Flagged:', len(flags))
    return 0


def cmd_analysis(argv: list[str]):
    # The per-sample render_report helper lives in lzo_gpu/tools/generate_stage_analysis.py
    # Add that path to sys.path so we can import it cleanly.
    import sys
    gpath = str(ROOT / 'lzo_gpu' / 'tools')
    if gpath not in sys.path:
        sys.path.insert(0, gpath)
    import generate_stage_analysis as gsa
    _render = gsa.render_report
    parser = argparse.ArgumentParser(prog='bench analysis')
    parser.add_argument('--file', '-f', default='sample_real_1.5gb.txt')
    parser.add_argument('--logs-dir', dest='logs_dir', default=None, help='exp_results dir or bench run dir containing the breakdown jsons')
    args = parser.parse_args(argv)
    if args.logs_dir:
        # Allow user to point to a logs dir; update the module's EXP_DIR before invoking render
        gsa.EXP_DIR = args.logs_dir
    return _render(args.file)


def cmd_all(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench all')
    parser.add_argument('--logs-dir', dest='logs_dir', default=None)
    parser.add_argument('--csv', dest='csv', default=None)
    parser.add_argument('--cv-threshold', dest='cv_threshold', default=0.25, type=float)
    args = parser.parse_args(argv)
    exp = Path(args.logs_dir) if args.logs_dir else EXP_DIR
    # 1) parse all
    cmd_parse_all(['--logs-dir', str(exp)])
    # 2) throughput on given csv (if provided)
    if args.csv:
        # Provide logs dir to throughput to allow picking up file_write_ms timings from per-run logs
        logs_dir_arg = ['--logs-dir', str(exp)] if exp else []
        cmd_throughput(['--csv', args.csv] + logs_dir_arg)
    # 3) variance
    cmd_variance(['--dir', str(exp), '--cv-threshold', str(args.cv_threshold)])
    # 4) analysis per-sample
    # discover samples
    found = set()
    for child in sorted(exp.iterdir()):
        if child.is_dir() and child.name.startswith('lzo-bench-'):
            for f in sorted(child.iterdir()):
                if f.is_file() and f.name.endswith('.log'):
                    m = re.match(r'^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$', f.name)
                    if m:
                        s = clean_sample_name(m.group('sample'))
                        found.add(s)
    for f in sorted(exp.iterdir()):
        if f.is_file() and f.name.endswith('.log'):
            m = re.match(r'^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$', f.name)
            if m:
                s = clean_sample_name(m.group('sample'))
                found.add(s)
    if not found:
        print('No sample logs discovered in', exp)
        return 1
    print('Running analysis reports for', len(found), 'samples')
    for s in sorted(found):
        print('Generating analysis for:', s)
        cmd_analysis(['--file', s, '--logs-dir', str(exp)])
    return 0


def cmd_analyze(argv: list[str]):
    """Replicate tools/analyze.py behavior as a `analyze` subcommand in tools/analysis.py
    Aggregates param_scan logs and computes analysis summaries for GPU experiments.
    """
    parser = argparse.ArgumentParser(prog='bench analyze')
    parser.add_argument('-i', '--input-dir', dest='input_dir', default=None, help='param_scan root (default: exp_results/lzo_gpu/logs/param_scans)')
    parser.add_argument('-o', '--out', dest='out', default=None, help='output summary CSV (default: exp_results/lzo_gpu/logs/summary.csv)')
    args = parser.parse_args(argv)
    # Determine default paths
    ROOT_PATH = Path(__file__).resolve().parents[1]
    EXP_ROOT = ROOT_PATH / 'exp_results' / 'lzo_gpu' / 'logs'
    default_in = EXP_ROOT / 'param_scans'
    default_out_csv = EXP_ROOT / 'summary.csv'
    INDIR = Path(args.input_dir) if args.input_dir else default_in
    OUTCSV = Path(args.out) if args.out else default_out_csv

    if not INDIR.exists():
        print(f"Input directory not found: {INDIR}")
        print("No param-scan logs found to analyze. Run the param-scan runner to generate logs under this path.")
        OUTCSV.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty summary CSV so downstream scripts don't fail
        fieldnames = ['core','strategy','sample','run','compress_kernel_ms','compress_total_ms','compress_thrpt_MBps','compress_orig_bytes','compress_comp_bytes','decomp_kernel_ms','decomp_total_ms','decomp_thrpt_MBps','decomp_orig_bytes','decomp_comp_bytes']
        with open(OUTCSV, 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=fieldnames)
            w.writeheader()
        print('Wrote empty summary CSV ->', OUTCSV)
        return 0

    rows = []
    # regex patterns from previous analyze.py
    COMP_RE = re.compile(r"\[COMP\s*\].*orig=([0-9]+)\s+comp=([0-9]+).*kernel=([0-9.]+)\s+ms.*total=([0-9.]+)\s+ms.*thrpt=([0-9.]+)\s+MB/s", re.M)
    DECOMP_RE = re.compile(r"\[DECOMP\].*orig=([0-9]+)\s+comp=([0-9]+).*kernel=([0-9.]+)\s+ms.*total=([0-9.]+)\s+ms.*thrpt=([0-9.]+)\s+MB/s", re.M)
    # Alternative patterns: parse compression/decompression statistics blocks that contain sizes and MB/s metrics
    COMP_STATS_RE = re.compile(r"=== Compression Statistics ===(.*?)(?:\n===|\Z)", re.S | re.I)
    DECOMP_STATS_RE = re.compile(r"=== Decompression Statistics ===(.*?)(?:\n===|\Z)", re.S | re.I)
    HEADER_LINE_RE = re.compile(r"^#\s*COMP=(?P<comp>\S+)(?:.*(?:MAP_MODE|STD_COPY)=(?P<map_mode>\S+))?.*WG=(?P<wg>\S+)(?:.*BLOCK=(?P<block_kb>\d+)KB)?(?:.*MT=(?P<mt>\S+))?.*SAMPLE=(?P<sample>\S+).*R=(?P<run>\d+)")
    LOOSE_HEADER_RE = re.compile(r"#\s*COMP=(?P<comp>\S+)(?:.*(?:MAP_MODE|STD_COPY)=(?P<map_mode>\S+))?.*(?:BLOCK=(?P<block_kb>\d+)KB)?(?:.*MT=(?P<mt>\S+))?.*SAMPLE=(?P<sample>\S+).*R=(?P<run>\d+)")
    STRAT_RE = re.compile(r"^STRATEGY=(?P<strategy>\S+)", flags=re.M)

    # Recursively discover all comp_* directories under INDIR to allow nested scans
    comp_dirs = []
    if INDIR.exists():
        for p in Path(str(INDIR)).rglob('comp_*'):
            if p.is_dir():
                comp_dirs.append(p)
    for comp_path in sorted(comp_dirs, key=lambda p: str(p)):
        core = comp_path.name
        for entry in sorted(os.listdir(str(comp_path))):
            parent_for_cfgs = None
            if entry.startswith('map_') or entry.startswith('stdcopy_'):
                parent_for_cfgs = os.path.join(str(comp_path), entry)
            elif entry.startswith('wg_'):
                parent_for_cfgs = str(comp_path)
            else:
                continue
            for cfg in sorted(os.listdir(parent_for_cfgs)):
                cfg_path = os.path.join(parent_for_cfgs, cfg)
                if not os.path.isdir(cfg_path):
                    continue
                cfg_dir = cfg_path
                # Recursively find log files under this configuration directory (block/mt subdirs)
                for p in sorted(Path(cfg_dir).rglob('*.log')):
                    fpath = str(p)
                    try:
                        with open(fpath, 'r', errors='ignore') as f:
                            data = f.read()
                    except Exception as e:
                        print('Failed to read', fpath, e); continue
                    m = HEADER_LINE_RE.search(data) or LOOSE_HEADER_RE.search(data)
                    sample = ''
                    run = ''
                    block_kb = ''
                    mt = ''
                    wg_val = ''
                    if m:
                        sample = m.groupdict().get('sample','')
                        run = m.groupdict().get('run','')
                        block_kb = m.groupdict().get('block_kb','') or ''
                        mt = m.groupdict().get('mt','') or ''
                        wg_val = m.groupdict().get('wg','') or ''
                    else:
                        m2 = re.match(r'(?P<sample>.+)_run(?P<run>\d+)\.log', Path(fpath).name)
                        if m2:
                            sample = m2.group('sample'); run = m2.group('run')
                    sm = STRAT_RE.search(data)
                    strategy = sm.group('strategy') if sm else 'none'
                    comp_vals = {'orig':None,'comp':None,'kernel':None,'total':None,'thrpt':None}
                    decomp_vals = {'orig':None,'comp':None,'kernel':None,'total':None,'thrpt':None}
                    # First try the compact [COMP]/[DECOMP] form
                    mcomp = COMP_RE.search(data)
                    if mcomp:
                        comp_vals['orig'] = int(mcomp.group(1))
                        comp_vals['comp'] = int(mcomp.group(2))
                        comp_vals['kernel'] = float(mcomp.group(3))
                        comp_vals['total'] = float(mcomp.group(4))
                        comp_vals['thrpt'] = float(mcomp.group(5))
                    else:
                        # Fallback: parse Compression Statistics block (orig/compressed sizes + throughput)
                        mc = COMP_STATS_RE.search(data)
                        if mc:
                            block = mc.group(1)
                            try:
                                m_orig = re.search(r"Input size\s*:\s*([0-9,]+)\s*bytes", block, flags=re.I)
                                if m_orig:
                                    comp_vals['orig'] = int(m_orig.group(1).replace(',', ''))
                                m_comp = re.search(r"Compressed size\s*:\s*([0-9,]+)\s*bytes", block, flags=re.I)
                                if m_comp:
                                    comp_vals['comp'] = int(m_comp.group(1).replace(',', ''))
                                m_thr = re.search(r"Throughput\s*:\s*([0-9.]+)\s*MB/s", block, flags=re.I)
                                if m_thr and comp_vals.get('orig'):
                                    thr = float(m_thr.group(1))
                                    comp_vals['thrpt'] = thr
                                    # compute total_ms from orig and throughput
                                    comp_vals['total'] = float(comp_vals['orig']) / (thr * 1048576.0) * 1000.0
                                # kernel throughput if present
                                m_kthr = re.search(r"\(kernel:\s*([0-9.]+)\s*MB/s\)", block, flags=re.I)
                                if m_kthr and comp_vals.get('orig'):
                                    kthr = float(m_kthr.group(1))
                                    comp_vals['kernel'] = float(comp_vals['orig']) / (kthr * 1048576.0) * 1000.0
                            except Exception:
                                pass
                    mdec = DECOMP_RE.search(data)
                    if mdec:
                        decomp_vals['orig'] = int(mdec.group(1))
                        decomp_vals['comp'] = int(mdec.group(2))
                        decomp_vals['kernel'] = float(mdec.group(3))
                        decomp_vals['total'] = float(mdec.group(4))
                        decomp_vals['thrpt'] = float(mdec.group(5))
                    else:
                        md = DECOMP_STATS_RE.search(data)
                        if md:
                            block = md.group(1)
                            try:
                                # Output size represents the decompressed orig size
                                m_out = re.search(r"Output size\s*:\s*([0-9,]+)\s*bytes", block, flags=re.I)
                                if m_out:
                                    decomp_vals['orig'] = int(m_out.group(1).replace(',', ''))
                                m_comp = re.search(r"Compressed size\s*:\s*([0-9,]+)\s*bytes", block, flags=re.I)
                                if m_comp:
                                    decomp_vals['comp'] = int(m_comp.group(1).replace(',', ''))
                                m_thr = re.search(r"Throughput\s*:\s*([0-9.]+)\s*MB/s", block, flags=re.I)
                                if m_thr and decomp_vals.get('orig'):
                                    thr = float(m_thr.group(1))
                                    decomp_vals['thrpt'] = thr
                                    decomp_vals['total'] = float(decomp_vals['orig']) / (thr * 1048576.0) * 1000.0
                                m_kthr = re.search(r"\(kernel:\s*([0-9.]+)\s*MB/s\)", block, flags=re.I)
                                if m_kthr and decomp_vals.get('orig'):
                                    kthr = float(m_kthr.group(1))
                                    decomp_vals['kernel'] = float(decomp_vals['orig']) / (kthr * 1048576.0) * 1000.0
                            except Exception:
                                pass
                    # Only require at least one of the key metrics to exist (total or kernel for comp/decomp)
                    if not (comp_vals.get('kernel') or comp_vals.get('total') or decomp_vals.get('kernel') or decomp_vals.get('total')):
                        continue
                    row = {
                        'core': core,
                        'strategy': strategy,
                        'sample': sample,
                        'wg': wg_val,
                        'block_kb': block_kb,
                        'mt': mt,
                        'run': run,
                        'compress_kernel_ms': f"{comp_vals['kernel']:.6f}" if comp_vals['kernel'] is not None else '',
                        'compress_total_ms': f"{comp_vals['total']:.6f}" if comp_vals['total'] is not None else '',
                        'compress_thrpt_MBps': f"{comp_vals['thrpt']:.6f}" if comp_vals['thrpt'] is not None else '',
                        'compress_orig_bytes': str(comp_vals['orig']) if comp_vals['orig'] is not None else '',
                        'compress_comp_bytes': str(comp_vals['comp']) if comp_vals['comp'] is not None else '',
                        'decomp_kernel_ms': f"{decomp_vals['kernel']:.6f}" if decomp_vals['kernel'] is not None else '',
                        'decomp_total_ms': f"{decomp_vals['total']:.6f}" if decomp_vals['total'] is not None else '',
                        'decomp_thrpt_MBps': f"{decomp_vals['thrpt']:.6f}" if decomp_vals['thrpt'] is not None else '',
                        'decomp_orig_bytes': str(decomp_vals['orig']) if decomp_vals['orig'] is not None else '',
                        'decomp_comp_bytes': str(decomp_vals['comp']) if decomp_vals['comp'] is not None else '',
                    }
                    rows.append(row)
    # Write summary CSV (default out location)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['core','strategy','sample','wg','block_kb','mt','run','compress_kernel_ms','compress_total_ms','compress_thrpt_MBps','compress_orig_bytes','compress_comp_bytes','decomp_kernel_ms','decomp_total_ms','decomp_thrpt_MBps','decomp_orig_bytes','decomp_comp_bytes']
    with open(OUTCSV, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote summary CSV ->', OUTCSV)
    # Simple analysis and aggregation
    agg = {}
    for rec in rows:
        core = rec.get('core','')
        strat = rec.get('strategy','') or 'none'
        wg_val = rec.get('wg','') or ''
        block_kb = rec.get('block_kb','') or ''
        mt = rec.get('mt','') or ''
        key = (core,strat,wg_val,block_kb,mt)
        agg.setdefault(key, {'comp_ms':[],'decomp_ms':[],'orig':[],'comp_sz':[],'count':0})
        a = agg[key]
        def tofloat(x):
            try: return float(x)
            except: return math.nan
        def toint(x):
            try: return int(float(x))
            except: return None
        ckm = tofloat(rec.get('compress_kernel_ms',''))
        dkm = tofloat(rec.get('decomp_kernel_ms',''))
        orig = toint(rec.get('compress_orig_bytes','') or rec.get('decomp_orig_bytes',''))
        comp_sz = toint(rec.get('compress_comp_bytes','') or rec.get('decomp_comp_bytes',''))
        if not math.isnan(ckm): a['comp_ms'].append(ckm)
        if not math.isnan(dkm): a['decomp_ms'].append(dkm)
        if orig: a['orig'].append(orig)
        if comp_sz: a['comp_sz'].append(comp_sz)
        a['count'] += 1
    metrics = []
    for (core,strat,wg_val,block_kb,mt),v in sorted(agg.items(), key=lambda x:(x[0][0],x[0][1],x[0][2] or '',x[0][3] or '', x[0][4] or '')):
        def mean_or_nan(a): return mean(a) if a else math.nan
        avg_comp = mean_or_nan(v['comp_ms'])
        avg_decomp = mean_or_nan(v['decomp_ms'])
        avg_ratio = math.nan
        if v['comp_sz'] and v['orig']:
            ratios = []
            for o,c in zip(v['orig'], v['comp_sz']):
                if c>0: ratios.append(float(o)/float(c))
            if ratios: avg_ratio = mean_or_nan(ratios)
        avg_comp_mb_s = math.nan
        if avg_comp and v['orig']:
            mean_orig = mean_or_nan(v['orig'])
            if not math.isnan(mean_orig) and avg_comp>0:
                avg_comp_mb_s = (mean_orig/(avg_comp/1000.0))/(1024*1024)
        avg_decomp_mb_s = math.nan
        if avg_decomp and v['orig']:
            mean_orig = mean_or_nan(v['orig'])
            if not math.isnan(mean_orig) and avg_decomp>0:
                avg_decomp_mb_s = (mean_orig/(avg_decomp/1000.0))/(1024*1024)
        metrics.append({'core':core,'strategy':strat,'wg':wg_val,'block_kb':block_kb,'mt':mt,'count':v['count'],'avg_comp_ms':avg_comp,'avg_decomp_ms':avg_decomp,'avg_ratio':avg_ratio,'avg_comp_MBps':avg_comp_mb_s,'avg_decomp_MBps':avg_decomp_mb_s})
    # Write aggregated CSV and vector-vs-base diffs & report to analysis log path
    OUT_DIR = EXP_ROOT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV = OUT_DIR / 'analysis_summary.csv'
    OUT_DIFF = OUT_DIR / 'vec_vs_base.csv'
    OUT_REPORT = OUT_DIR / 'analysis_report.txt'
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['core','strategy','wg','block_kb','mt','rows','avg_comp_ms','avg_decomp_ms','avg_ratio','avg_comp_MBps','avg_decomp_MBps'])
        for m in metrics:
            def fmt(v): return f"{v:.3f}" if (v is not None and not math.isnan(v)) else ''
            w.writerow([m['core'],m['strategy'], m.get('wg', ''), m.get('block_kb', ''), m.get('mt', ''), m['count'], fmt(m['avg_comp_ms']), fmt(m['avg_decomp_ms']), fmt(m['avg_ratio']), fmt(m['avg_comp_MBps']), fmt(m['avg_decomp_MBps'])])

    # vec vs base diffs
    rows_diff = []
    for m in metrics:
        strat = m['strategy']
        if strat.endswith('_vec'):
            base = strat[:-4]
            base_entry = next((x for x in metrics if x['core']==m['core'] and x['strategy']==base), None)
            if base_entry:
                a_de = m['avg_decomp_ms']
                b_de = base_entry['avg_decomp_ms']
                pct_change_de = None
                if a_de and b_de and not math.isnan(a_de) and not math.isnan(b_de) and b_de>0:
                    pct_change_de = (a_de - b_de)/b_de*100.0
                a_co = m['avg_comp_ms']
                b_co = base_entry['avg_comp_ms']
                pct_change_co = None
                if a_co and b_co and not math.isnan(a_co) and not math.isnan(b_co) and b_co>0:
                    pct_change_co = (a_co - b_co)/b_co*100.0
                rows_diff.append({'core':m['core'],'vec_strategy':strat,'base_strategy':base,'vec_avg_decomp_ms':a_de,'base_avg_decomp_ms':b_de,'decomp_pct_change':pct_change_de,'vec_avg_comp_ms':a_co,'base_avg_comp_ms':b_co,'comp_pct_change':pct_change_co})
    with open(OUT_DIFF, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['core','vec_strategy','base_strategy','vec_avg_decomp_ms','base_avg_decomp_ms','decomp_pct_change','vec_avg_comp_ms','base_avg_comp_ms','comp_pct_change'])
        for d in rows_diff:
            w.writerow([d['core'],d['vec_strategy'],d['base_strategy'],f"{d['vec_avg_decomp_ms']:.3f}" if d['vec_avg_decomp_ms'] and not math.isnan(d['vec_avg_decomp_ms']) else '',f"{d['base_avg_decomp_ms']:.3f}" if d['base_avg_decomp_ms'] and not math.isnan(d['base_avg_decomp_ms']) else '',f"{d['decomp_pct_change']:.2f}" if d['decomp_pct_change'] is not None else '',f"{d['vec_avg_comp_ms']:.3f}" if d['vec_avg_comp_ms'] and not math.isnan(d['vec_avg_comp_ms']) else '',f"{d['base_avg_comp_ms']:.3f}" if d['base_avg_comp_ms'] and not math.isnan(d['base_avg_comp_ms']) else '',f"{d['comp_pct_change']:.2f}" if d['comp_pct_change'] is not None else ''])
    # report
    with open(OUT_REPORT, 'w') as f:
        f.write('Analysis of combinations\n')
        f.write('========================\n\n')
        good_decomp = sorted([m for m in metrics if not math.isnan(m['avg_decomp_MBps'])], key=lambda x: -x['avg_decomp_MBps'])
        f.write('Top 10 by decompression throughput (MB/s):\n')
        for m in good_decomp[:10]:
            f.write(f"  {m['core']} + {m['strategy']}: {m['avg_decomp_MBps']:.3f} MB/s (avg_decomp_ms={m['avg_decomp_ms']:.3f})\n")
        f.write('\n')
        good_comp = sorted([m for m in metrics if not math.isnan(m['avg_comp_MBps'])], key=lambda x: -x['avg_comp_MBps'])
        f.write('Top 10 by compression throughput (MB/s):\n')
        for m in good_comp[:10]:
            f.write(f"  {m['core']} + {m['strategy']}: {m['avg_comp_MBps']:.3f} MB/s (avg_comp_ms={m['avg_comp_ms']:.3f})\n")
        f.write('\n')
        good_ratio = sorted([m for m in metrics if not math.isnan(m['avg_ratio'])], key=lambda x: -x['avg_ratio'])
        f.write('Top 10 by compression ratio (orig/comp):\n')
        for m in good_ratio[:10]:
            f.write(f"  {m['core']} + {m['strategy']}: ratio={m['avg_ratio']:.3f}\n")
        f.write('\n')
        f.write('Vectorized vs base comparisons (decomp pct change, positive => vec slower):\n')
        for d in rows_diff:
            pct = d['decomp_pct_change']
            s = 'n/a' if pct is None else f"{pct:.2f}%"
            f.write(f"  {d['core']}: {d['vec_strategy']} vs {d['base_strategy']}: decomp change {s}\n")
        f.write('\nNote: results are averages across samples available in summary.csv.\n')
    print('Wrote', OUT_CSV, OUT_DIFF, OUT_REPORT)
    return 0


def safe_get_local(d, k):
    return d.get(k, {}).get('mean', 0.0)


def collect_metrics_local(sample_json):
    metrics = []
    for runner in sorted(sample_json.keys()):
        for mode in sorted(sample_json[runner].keys()):
            for typ in sorted(sample_json[runner][mode].keys()):
                d = sample_json[runner][mode][typ]
                total = safe_get_local(d, 'total_ms')
                kernel = safe_get_local(d, 'kernel_exec_ms')
                download = safe_get_local(d, 'data_download_ms') if 'data_download_ms' in d else safe_get_local(d, 'download_total_ms')
                upload = safe_get_local(d, 'data_upload_ms')
                file_read = safe_get_local(d, 'file_read_ms')
                file_write = safe_get_local(d, 'file_write_ms')
                buf_in = safe_get_local(d, 'buffer_alloc_in_ms')
                buf_out = safe_get_local(d, 'buffer_alloc_out_ms')
                buf_len = safe_get_local(d, 'buffer_alloc_len_ms')
                buf_tot = safe_get_local(d, 'buffer_alloc_ms')
                alloc_total = sum([x for x in [buf_in, buf_out, buf_len, buf_tot] if x is not None])
                ocl_init = safe_get_local(d, 'ocl_init_ms') + safe_get_local(d, 'kernel_load_ms')
                # Effective total excluding OCL init and kernel load (preload)
                total_eff = max(total - ocl_init, 0.0)
                other = safe_get_local(d, 'setup_args_ms') + safe_get_local(d, 'blocking_calc_ms') + safe_get_local(d, 'cleanup_ms')
                metrics.append({'runner':runner,'mode':mode,'type':typ,'total':total_eff,'kernel':kernel,'file_read':file_read,'file_write':file_write,'upload':upload,'download':download,'alloc_total':alloc_total,'ocl_init':ocl_init,'other':other, 'total_raw': total})
    return metrics


def fmt(x):
    return f"{x:.3f}"


def pct(part, total):
    return (part / total * 100.0) if total else 0.0


def best_and_worst(metrics, typ):
    m = [x for x in metrics if x['type'] == typ]
    if not m:
        return None, None
    best = min(m, key=lambda r: r['total'])
    worst = max(m, key=lambda r: r['total'])
    return best, worst


def render_report(sample_name: str):
    json_path = EXP_DIR / f"{sample_name}_breakdown_summary_all.json"
    if not json_path.exists():
        print('ERR: json summary not found:', json_path); return 2
    data = json.loads(json_path.read_text(encoding='utf-8'))
    metrics = collect_metrics_local(data)
    if not metrics:
        print('no metrics'); return 3
    out = []
    out.append(f"# {sample_name} — per-stage 时间分布分析\n")
    out.append('来源文件: ' + str(json_path) + '\n')
    out.append('注: 所有数字为平均值 (mean)，单位 ms。百分比基于 total/ms。\n')
    for typ in ['compression','decompression']:
        out.append('## ' + ('压缩' if typ=='compression' else '解压缩') + '\n')
        rows = [r for r in metrics if r['type']==typ]
        if not rows:
            out.append('无数据\n'); continue
        out.append('| runner | mode | total (ms) | kernel (ms, %) | file_read (ms, %) | upload (ms, %) | download (ms, %) | alloc_total (ms, %) | file_write (ms, %) |')
        out.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
        for r in sorted(rows, key=lambda x: x['total']):
            total=r['total']; k=r['kernel']; fr=r['file_read']; fw=r['file_write']; up=r['upload']; dl=r['download']; alloc=r['alloc_total']
            out.append('| %s | %s | %s | %s (%.1f%%) | %s (%.1f%%) | %s (%.1f%%) | %s (%.1f%%) | %s (%.1f%%) | %s (%.1f%%) |' % (r['runner'], r['mode'], fmt(total), fmt(k), pct(k,total), fmt(fr), pct(fr,total), fmt(up), pct(up,total), fmt(dl), pct(dl,total), fmt(alloc), pct(alloc,total), fmt(fw), pct(fw,total)))
        best, worst = best_and_worst(metrics, typ)
        if best:
            out.append('\n'); out.append('**最佳（平均总时长最短）组合**: %s + %s = %s ms' % (best['runner'], best['mode'], fmt(best['total']))); out.append('\n')
            out.append('**最差（平均总时长最长）组合**: %s + %s = %s ms' % (worst['runner'], worst['mode'], fmt(worst['total']))); out.append('\n')
        baseline = next((r for r in rows if r['runner']=='standalone' and r['mode']=='std'), None)
        if baseline:
            out.append('\n**相对 baseline (standalone std)**:\n')
            base_total = baseline['total']
            out.append('\n| combo | total(ms) | delta(ms) | improvement(%) | kernel delta(ms) | non-kernel delta(ms) |')
            out.append('|---|---:|---:|---:|---:|---:|')
            for r in sorted(rows, key=lambda x: x['total']):
                dt = r['total'] - base_total
                imp = (base_total - r['total']) / base_total * 100.0
                kd = r['kernel'] - baseline['kernel']
                nk = (r['total'] - r['kernel']) - (baseline['total'] - baseline['kernel'])
                out.append('| %s+%s | %s | %s | %s%% | %s | %s |' % (r['runner'], r['mode'], fmt(r['total']), ('+'+fmt(dt)) if dt>=0 else fmt(dt), ('+'+fmt(imp) if imp>=0 else fmt(imp)), fmt(kd), fmt(nk)))
        out.append('\n---\n')
    out.append('## 快速结论与建议\n')
    out.append('- 文件 I/O（尤其是 file_write）和缓冲分配是主要开销，占比明显；建议优化 file_write 路径和缓冲复用。\n')
    out.append('\n建议优先方向：\n- 优先优化 file_write 路径。\n- 减少 buffer 分配。\n- 在资料族/平台上同时对 zero-copy 和 std-copy 做对比测试以找到最佳路径。')
    out_path = EXP_DIR / f"{sample_name}_analysis.md"
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print('Wrote', out_path)
    return 0


def cmd_coalesce(argv: list[str]):
    parser = argparse.ArgumentParser(prog='bench coalesce')
    parser.add_argument('--sample', '-s', required=True)
    parser.add_argument('--logdir', '-l', required=True)
    parser.add_argument('--out', '-o', default='coalesce_status.csv')
    args = parser.parse_args(argv)
    sample = args.sample; logdir = Path(args.logdir)
    outcsv = Path(args.out)
    if not logdir.exists():
        print('Logdir not exists:', logdir); return 2
    logs = [p for p in sorted(logdir.iterdir()) if p.is_file() and p.name.startswith(sample) and p.name.endswith('.log')]
    if not logs:
        print('No logs found for sample', sample); return 2
    rows = []
    # Accept logname patterns where the sample portion may include file extensions
    fname_re = re.compile(r"^(?P<sample>.+)\.(?P<runner>[^.]+)\.(?P<mode>[^.]+)\.run(?P<run>\d+)(?:\.decomp)?\.log$")
    for p in logs:
        m = fname_re.match(p.name)
        if not m:
            print('SKIP (unrecognized name):', p.name); continue
        sample_face = clean_sample_name(m.group('sample'))
        if not (sample_face == sample or sample_face.startswith(sample + '.') or sample_face.startswith(sample)):
            print('SKIP (sample mismatch):', p.name, 'parsed sample', sample_face, 'requested', sample); continue
        runner = m.group('runner'); mode = m.group('mode'); runnum = int(m.group('run'))
        typ, parsed = parse_file(p)
        status = 'none'
        content = p.read_text(encoding='utf-8', errors='ignore')
        if re.search(r'COALESCE:\s*full contiguous allocation success|COALESCE_COPY', content, flags=re.I):
            status = 'contig'
        elif re.search(r'COALESCE:\s*full contiguous allocation failed|CHUNK_WRITE', content, flags=re.I):
            status = 'chunk'
        elif re.search(r'COALESCE:\s*chunk allocation failed|BLOCK_WRITE', content, flags=re.I):
            status = 'per-block'
        coalesce_copy_ms = parsed.get('coalesce_copy_ms', 0.0)
        coalesce_write_ms = parsed.get('coalesce_write_ms', 0.0)
        block_count = parsed.get('block_write_count', 0)
        block_mean_ms = parsed.get('block_write_mean_ms', 0.0)
        file_write_ms = parsed.get('file_write_ms', 0.0)
        rows.append([sample_face, runner, mode, typ, runnum, status, coalesce_copy_ms, coalesce_write_ms, int(block_count), block_mean_ms, file_write_ms])
    rows.sort(key=lambda r: (r[1], r[2], r[3], r[4]))
    with open(outcsv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample','runner','mode','type','run','coalesce_status','coalesce_copy_ms','coalesce_write_ms','block_count','block_mean_ms','file_write_ms'])
        for r in rows:
            w.writerow(r)
    print('Wrote:', outcsv)
    return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('parse')
    sub.add_parser('parse-all')
    sub.add_parser('throughput')
    sub.add_parser('variance')
    sub.add_parser('coalesce')
    sub.add_parser('analysis')
    sub.add_parser('analyze')
    sub.add_parser('all')
    args, rest = parser.parse_known_args(argv)
    if args.cmd == 'parse':
        return cmd_parse(rest)
    if args.cmd == 'parse-all':
        return cmd_parse_all(rest)
    if args.cmd == 'throughput':
        return cmd_throughput(rest)
    if args.cmd == 'variance':
        return cmd_variance(rest)
    if args.cmd == 'coalesce':
        # reuse parse_coalesce_status.py logic (simple inline variant here)
        return cmd_coalesce(rest)
    if args.cmd == 'analysis':
        return cmd_analysis(rest)
    if args.cmd == 'analyze':
        return cmd_analyze(rest)
    if args.cmd == 'all':
        return cmd_all(rest)
    parser.print_help(); return 0


if __name__ == '__main__':
    raise SystemExit(main())
