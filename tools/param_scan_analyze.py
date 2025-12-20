#!/usr/bin/env python3
"""Analyze param_scan CSV and produce a short report and CSVs (best-per-sample, anomalies).

Usage:
  python3 tools/param_scan_analyze.py /path/to/param_scan.csv -o /path/to/out.txt
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict, Counter, namedtuple

Row = namedtuple('Row', ['sample','size_mb','tool','config','mode','algorithm','block_kb','threads','mt_threads','vec','mt_io','copy_mode','coalesce','stdio_buf_mb','ratio_frac','ratio_pct','total_time_ms','kernel_time_ms','read_time_ms','write_time_ms','upload_time_ms','download_time_ms','kernel_name','global_size','local_size','buffer_alloc_in_ms','buffer_alloc_out_ms','buffer_alloc_len_ms','blocking_calc_ms','download_len_ms','download_bulk_ms','download_total_ms','total_throughput_mbps','kernel_throughput_mbps','verified'])


def parse_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == '' or s.upper() == 'NA':
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_int(s):
    if s is None:
        return None
    s = s.strip()
    if s == '' or s.upper() == 'NA':
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Build Row with direct mapping; some CSV rows (CPU) may be missing trailing fields, so we'll fill them and apply fallbacks below
            row = Row(
                sample = r.get('sample',''),
                size_mb = parse_float(r.get('size_mb')),
                tool = r.get('tool',''),
                config = r.get('config',''),
                mode = r.get('mode',''),
                algorithm = r.get('algorithm',''),
                block_kb = parse_int(r.get('block_kb')),
                threads = parse_int(r.get('threads')),
                mt_threads = (r.get('mt_threads') or '').strip(),
                vec = r.get('vec',''),
                mt_io = r.get('mt_io',''),
                copy_mode = r.get('copy_mode',''),
                coalesce = r.get('coalesce',''),
                stdio_buf_mb = parse_int(r.get('stdio_buf_mb')),
                ratio_frac = r.get('ratio_frac',''),
                ratio_pct = parse_float(r.get('ratio_pct')),
                total_time_ms = parse_float(r.get('total_time_ms')),
                kernel_time_ms = parse_float(r.get('kernel_time_ms')),
                read_time_ms = parse_float(r.get('read_time_ms')),
                write_time_ms = parse_float(r.get('write_time_ms')),
                upload_time_ms = parse_float(r.get('upload_time_ms')),
                download_time_ms = parse_float(r.get('download_time_ms')),
                kernel_name = r.get('kernel_name',''),
                global_size = parse_int(r.get('global_size')),
                local_size = parse_int(r.get('local_size')),
                buffer_alloc_in_ms = parse_float(r.get('buffer_alloc_in_ms')),
                buffer_alloc_out_ms = parse_float(r.get('buffer_alloc_out_ms')),
                buffer_alloc_len_ms = parse_float(r.get('buffer_alloc_len_ms')),
                blocking_calc_ms = parse_float(r.get('blocking_calc_ms')),
                download_len_ms = parse_float(r.get('download_len_ms')),
                download_bulk_ms = parse_float(r.get('download_bulk_ms')),
                download_total_ms = parse_float(r.get('download_total_ms')),
                total_throughput_mbps = parse_float(r.get('total_throughput_mbps')),
                kernel_throughput_mbps = parse_float(r.get('kernel_throughput_mbps')),
                verified = (r.get('verified') or '').strip(),
            )

            # Fallbacks for rows with missing trailing columns (some CPU rows are shorter):
            # If total_throughput_mbps is missing, search the trailing numeric fields and pick the largest as total and 2nd largest as kernel throughput.
            if row.total_throughput_mbps is None:
                # collect numeric candidates from tail of row dict (columns after ratio_pct)
                tail_keys = ['total_time_ms','kernel_time_ms','read_time_ms','write_time_ms','upload_time_ms','download_time_ms','kernel_name','global_size','local_size','buffer_alloc_in_ms','buffer_alloc_out_ms','buffer_alloc_len_ms','blocking_calc_ms','download_len_ms','download_bulk_ms','download_total_ms','total_throughput_mbps','kernel_throughput_mbps']
                cand = []
                for k in tail_keys:
                    v = r.get(k)
                    val = parse_float(v)
                    if val is not None:
                        cand.append((k,val))
                if cand:
                    # pick top two numeric values (largest assumed to be total throughput)
                    cand_sorted = sorted(cand, key=lambda kv: kv[1], reverse=True)
                    if not row.total_throughput_mbps:
                        row = row._replace(total_throughput_mbps=cand_sorted[0][1])
                    if not row.kernel_throughput_mbps and len(cand_sorted) > 1:
                        # choose the next largest as kernel throughput
                        row = row._replace(kernel_throughput_mbps=cand_sorted[1][1])
            # If verified is missing, try to find YES/NO anywhere in the row values
            if not row.verified:
                for v in r.values():
                    if isinstance(v,str) and v.strip().upper() in ('YES','NO'):
                        row = row._replace(verified=v.strip().upper())
                        break
            # append the parsed (and possibly patched) row
            rows.append(row)
    return rows


def mean_and_median(xs):
    xs_nonzero = [x for x in xs if x is not None]
    if not xs_nonzero:
        return (None, None)
    try:
        m = statistics.mean(xs_nonzero)
    except Exception:
        m = None
    try:
        med = statistics.median(xs_nonzero)
    except Exception:
        med = None
    return (m, med)


def summarize(rows, out_path=None, top_n=10):
    total_rows = len(rows)
    samples = sorted(set(r.sample for r in rows))
    sample_count = len(samples)

    tools = Counter(r.tool for r in rows)

    # Per-tool stats
    per_tool = {}
    for tool in tools:
        tool_rows = [r for r in rows if r.tool == tool]
        totals = [r.total_throughput_mbps or 0.0 for r in tool_rows]
        totals_nonzero = [r.total_throughput_mbps for r in tool_rows if r.total_throughput_mbps and r.total_throughput_mbps > 0]
        kernels = [r.kernel_throughput_mbps or 0.0 for r in tool_rows]
        verified_yes = sum(1 for r in tool_rows if r.verified.upper() == 'YES')
        per_tool[tool] = {
            'count': len(tool_rows),
            'nonzero_count': len(totals_nonzero),
            'pct_nonzero': (len(totals_nonzero)/len(tool_rows)*100.0) if tool_rows else 0.0,
            'mean_including_zero': statistics.mean(totals) if totals else None,
            'mean_nonzero': statistics.mean(totals_nonzero) if totals_nonzero else None,
            'median_nonzero': statistics.median(totals_nonzero) if totals_nonzero else None,
            'max_throughput': max(totals) if totals else 0.0,
            'max_row': max(tool_rows, key=lambda r: (r.total_throughput_mbps or 0.0)) if tool_rows else None,
            'verified_yes': verified_yes,
        }

    # Global top configs
    rows_sorted = sorted(rows, key=lambda r: (r.total_throughput_mbps or 0.0), reverse=True)
    top_overall = rows_sorted[:top_n]

    # Best per sample
    best_per_sample = {}
    for r in rows:
        cur = best_per_sample.get(r.sample)
        t = r.total_throughput_mbps or 0.0
        if cur is None or t > (cur.total_throughput_mbps or 0.0):
            best_per_sample[r.sample] = r

    # Daemon-specific anomalies
    daemon_rows = [r for r in rows if r.tool.upper() == 'DAEMON']
    daemon_count = len(daemon_rows)
    daemon_zero = [r for r in daemon_rows if not r.total_throughput_mbps or r.total_throughput_mbps == 0]
    daemon_nonzero = [r for r in daemon_rows if r.total_throughput_mbps and r.total_throughput_mbps > 0]

    # MT IO effect (group by tool)
    mtio_effect = defaultdict(list)
    for r in rows:
        key2 = (r.tool, r.mt_io)
        if r.total_throughput_mbps:
            mtio_effect[key2].append(r.total_throughput_mbps)

    # Block size analysis for GPU compression (mode=compress & tool=GPU)
    block_groups = defaultdict(list)
    for r in rows:
        if r.tool.upper() == 'GPU' and r.mode=='compress':
            if r.block_kb:
                if r.total_throughput_mbps:
                    block_groups[r.block_kb].append(r.total_throughput_mbps)

    # Prepare textual report
    lines = []
    lines.append(f"Param scan CSV analysis: total rows={total_rows}, samples={sample_count}")
    lines.append('')
    lines.append('Entries per tool:')
    for t,c in tools.items():
        pt = per_tool[t]
        lines.append(f"  {t}: {c} rows, verified YES {pt['verified_yes']}, non-zero throughput {pt['nonzero_count']} ({pt['pct_nonzero']:.1f}%)")
        lines.append(f"    mean incl zeros: {pt['mean_including_zero']:.2f} MB/s, mean non-zero: {pt['mean_nonzero'] or 0:.2f} MB/s, median non-zero: {pt['median_nonzero'] or 0:.2f} MB/s, max: {pt['max_throughput']:.2f} MB/s")

    lines.append('')
    lines.append(f"Top {top_n} configs by total_throughput_mbps:")
    for r in top_overall:
        lines.append(f"  {r.total_throughput_mbps or 0:.2f} MB/s | kernel {r.kernel_throughput_mbps or 0:.2f} | {r.tool} | {r.config} | sample={r.sample} | verified={r.verified}")

    lines.append('')
    lines.append('Best config per sample (sample, best-tool, throughput MB/s, verified) — showing up to 20 samples:')
    count=0
    for s,r in sorted(best_per_sample.items(), key=lambda kv: (-(kv[1].total_throughput_mbps or 0.0), kv[0])):
        lines.append(f"  {s}: {r.tool} {r.config} {r.total_throughput_mbps or 0:.2f} MB/s verified={r.verified}")
        count+=1
        if count>=20:
            break

    lines.append('')
    lines.append('Daemon summary:')
    lines.append(f"  total daemon rows: {daemon_count}, zero-throughput rows: {len(daemon_zero)} ({(len(daemon_zero)/daemon_count*100.0) if daemon_count else 0:.1f}%), non-zero rows: {len(daemon_nonzero)}")
    if daemon_nonzero:
        top_d = sorted(daemon_nonzero, key=lambda r: r.total_throughput_mbps, reverse=True)[:10]
        lines.append('  Top daemon non-zero examples:')
        for r in top_d:
            lines.append(f"    {r.total_throughput_mbps:.2f} MB/s | {r.config} | sample={r.sample} | verified={r.verified}")

    lines.append('')
    # (async upload removed)

    lines.append('')
    lines.append('MT IO effect (mean non-zero throughput per (tool,mt_io)):')
    for k,vals in sorted(mtio_effect.items()):
        lines.append(f"  {k}: mean_nonzero={statistics.mean(vals):.2f} MB/s, n={len(vals)}")

    lines.append('')
    lines.append('Block size (GPU compress) analysis (avg non-zero MB/s):')
    for b,vals in sorted(block_groups.items()):
        lines.append(f"  block={b} KB: mean={statistics.mean(vals):.2f} MB/s, n={len(vals)}")

    # Anomalies: daemon rows with total==0 but verified YES, and rows with kernel>0 and total==0
    anomalies = []
    for r in rows:
        if (not r.total_throughput_mbps or r.total_throughput_mbps==0) and r.verified.upper()=='YES':
            if r.tool.upper()=='DAEMON':
                anomalies.append(('daemon_zero_verified', r))
        if r.kernel_throughput_mbps and (not r.total_throughput_mbps or r.total_throughput_mbps==0):
            anomalies.append(('kernel_nonzero_total_zero', r))
    lines.append('')
    lines.append(f"Anomalies found: {len(anomalies)} (showing up to 20):")
    for tag,r in anomalies[:20]:
        lines.append(f"  [{tag}] {r.tool} {r.config} sample={r.sample} total={r.total_throughput_mbps or 0} kernel={r.kernel_throughput_mbps or 0} verified={r.verified}")

    # Save outputs: best_per_sample CSV and anomalies CSV and text report
    if out_path:
        txt = '\n'.join(lines)
        with open(out_path,'w') as f:
            f.write(txt)
        # best per sample CSV
        best_csv = out_path.replace('.txt','_best_per_sample.csv')
        with open(best_csv,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample','best_tool','best_config','throughput_mbps','verified'])
            for s,r in sorted(best_per_sample.items()):
                w.writerow([s,r.tool,r.config,(r.total_throughput_mbps or 0),r.verified])
        # anomalies
        an_csv = out_path.replace('.txt','_anomalies.csv')
        with open(an_csv,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['type','tool','config','sample','total_mbps','kernel_mbps','verified'])
            for tag,r in anomalies:
                w.writerow([tag,r.tool,r.config,r.sample,(r.total_throughput_mbps or 0),(r.kernel_throughput_mbps or 0),r.verified])

    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='param_scan CSV file')
    p.add_argument('-o','--out', help='output text report path', default=None)
    args = p.parse_args()
    rows = read_csv(args.csv)
    out = args.out
    if not out:
        # default to same dir and name
        import os
        out = os.path.splitext(args.csv)[0] + '_analysis.txt'
    lines = summarize(rows, out_path=out, top_n=15)
    print('\n'.join(lines[:200]))
    print('\nReport saved to:', out)

if __name__ == '__main__':
    main()
