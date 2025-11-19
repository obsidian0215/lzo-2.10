#!/usr/bin/env python3
"""
Combined aggregator + analyzer: produces `summary.csv` from per-run logs and
computes simple analysis CSVs and a textual report.
This merges the previous `aggregate_param_scan.py` and `analyze_combinations.py`.

Usage: tools/analyze.py -i <param_scan_root> -o <out_summary_csv>
Defaults: -i lzo_gpu/experiments/logs/param_scans -o lzo_gpu/experiments/logs/summary.csv
"""
import os, re, csv, argparse, math, statistics

parser = argparse.ArgumentParser(description='Aggregate and analyze param-scan logs')
parser.add_argument('-i','--input-dir', default=None)
parser.add_argument('-o','--out', default=None)
args = parser.parse_args()
ROOT = os.path.dirname(os.path.dirname(__file__))
# Default outputs live under the repository root `exp_results/`.
# lzo_gpu-specific outputs are placed under `exp_results/lzo_gpu/logs`.
OUT_BASE = os.path.join(ROOT, 'exp_results')
DEFAULT_IN = os.path.join(OUT_BASE, 'lzo_gpu', 'logs', 'param_scans')
DEFAULT_OUT = os.path.join(OUT_BASE, 'lzo_gpu', 'logs', 'summary.csv')
INDIR = args.input_dir if args.input_dir else DEFAULT_IN
OUTCSV = args.out if args.out else DEFAULT_OUT

COMP_RE = re.compile(r"\[COMP\s*\].*orig=([0-9]+)\s+comp=([0-9]+).*kernel=([0-9.]+)\s+ms.*total=([0-9.]+)\s+ms.*thrpt=([0-9.]+)\s+MB/s", re.M)
DECOMP_RE = re.compile(r"\[DECOMP\].*orig=([0-9]+)\s+comp=([0-9]+).*kernel=([0-9.]+)\s+ms.*total=([0-9.]+)\s+ms.*thrpt=([0-9.]+)\s+MB/s", re.M)
HEADER_LINE_RE = re.compile(r"^#\s*COMP=(?P<comp>\S+).*MODE=(?P<mode>\S+).*WG=(?P<wg>\S+).*VLEN=(?P<vlen>\S+).*SAMPLE=(?P<sample>\S+).*R=(?P<run>\d+)")
# The header format in some logs may differ; try a looser header capture
LOOSE_HEADER_RE = re.compile(r"#\s*COMP=(?P<comp>\S+).*MODE=(?P<mode>\S+).*SAMPLE=(?P<sample>\S+).*R=(?P<run>\d+)")

rows = []
if not os.path.isdir(INDIR):
    # Be tolerant: tools should not assume presence of experiment logs.
    print(f"Input directory not found: {INDIR}\nNo param-scan logs found to analyze. Run the param-scan runner to generate logs under this path.")
    # Ensure output directory exists for downstream commands and exit cleanly.
    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    # Write an empty summary CSV with header so downstream consumers behave predictably.
    fieldnames = ['core','strategy','sample','run','compress_kernel_ms','compress_total_ms','compress_thrpt_MBps','compress_orig_bytes','compress_comp_bytes','decomp_kernel_ms','decomp_total_ms','decomp_thrpt_MBps','decomp_orig_bytes','decomp_comp_bytes']
    with open(OUTCSV, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
    print('Wrote empty summary CSV ->', OUTCSV)
    raise SystemExit(0)
# Directory layout: <INDIR>/comp_<level>/decomp_{base,vec}/wg_<wg>_v_<vlen>/*.log
for comp_dir in sorted(os.listdir(INDIR)):
    if not comp_dir.startswith('comp_'):
        continue
    comp_path = os.path.join(INDIR, comp_dir)
    if not os.path.isdir(comp_path):
        continue
    # comp_dir is like 'comp_1' or 'comp_gpuport'
    core = comp_dir
    for decomp_mode in sorted(os.listdir(comp_path)):
        decomp_path = os.path.join(comp_path, decomp_mode)
        if not os.path.isdir(decomp_path):
            continue
        # strategy mapping: prefer 'base' or 'vec' when present in directory name
        if 'base' in decomp_mode:
            strategy = 'base'
        elif 'vec' in decomp_mode:
            strategy = 'vec'
        else:
            strategy = decomp_mode
        # iterate per-workgroup/vlen configuration directories
        for cfg in sorted(os.listdir(decomp_path)):
            cfg_path = os.path.join(decomp_path, cfg)
            if not os.path.isdir(cfg_path):
                continue
            for fname in sorted(os.listdir(cfg_path)):
                if not fname.endswith('.log'):
                    continue
                fpath = os.path.join(cfg_path, fname)
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    data = f.read()
            except Exception as e:
                print('Failed to read', fpath, e)
                continue
            # header
            m = HEADER_LINE_RE.search(data) or LOOSE_HEADER_RE.search(data)
            sample = ''
            run = ''
            if m:
                sample = m.groupdict().get('sample','')
                run = m.groupdict().get('run','')
            else:
                m2 = re.match(r'(?P<sample>.+)_run(?P<run>\d+)\.log', fname)
                if m2:
                    sample = m2.group('sample'); run = m2.group('run')
            comp_vals = {'orig':None,'comp':None,'kernel':None,'total':None,'thrpt':None}
            decomp_vals = {'orig':None,'comp':None,'kernel':None,'total':None,'thrpt':None}
            mcomp = COMP_RE.search(data)
            if mcomp:
                comp_vals['orig'] = int(mcomp.group(1))
                comp_vals['comp'] = int(mcomp.group(2))
                comp_vals['kernel'] = float(mcomp.group(3))
                comp_vals['total'] = float(mcomp.group(4))
                comp_vals['thrpt'] = float(mcomp.group(5))
            mdec = DECOMP_RE.search(data)
            if mdec:
                decomp_vals['orig'] = int(mdec.group(1))
                decomp_vals['comp'] = int(mdec.group(2))
                decomp_vals['kernel'] = float(mdec.group(3))
                decomp_vals['total'] = float(mdec.group(4))
                decomp_vals['thrpt'] = float(mdec.group(5))
            if comp_vals['kernel'] is None and decomp_vals['kernel'] is None:
                continue
            row = {
                'core': core,
                'strategy': strategy,
                'sample': sample,
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

# write summary CSV
os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
fieldnames = ['core','strategy','sample','run','compress_kernel_ms','compress_total_ms','compress_thrpt_MBps','compress_orig_bytes','compress_comp_bytes','decomp_kernel_ms','decomp_total_ms','decomp_thrpt_MBps','decomp_orig_bytes','decomp_comp_bytes']
with open(OUTCSV, 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('Wrote summary CSV ->', OUTCSV)

# simple analysis: aggregate by (core,strategy)
agg = {}
for rec in rows:
    core = rec.get('core','')
    strat = rec.get('strategy','') or 'none'
    key = (core,strat)
    if key not in agg:
        agg[key] = {'comp_ms':[],'decomp_ms':[],'orig':[],'comp_sz':[],'count':0}
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
for (core,strat),v in sorted(agg.items(), key=lambda x:(x[0][0],x[0][1])):
    def mean_or_nan(a): return statistics.mean(a) if a else math.nan
    def median_or_nan(a): return statistics.median(a) if a else math.nan
    def std_or_nan(a): return statistics.stdev(a) if len(a)>1 else math.nan
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
    metrics.append({'core':core,'strategy':strat,'count':v['count'],'avg_comp_ms':avg_comp,'avg_decomp_ms':avg_decomp,'avg_ratio':avg_ratio,'avg_comp_MBps':avg_comp_mb_s,'avg_decomp_MBps':avg_decomp_mb_s})

OUT_DIR = os.path.join(ROOT, 'lzo_gpu', 'experiments', 'logs')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'analysis_summary.csv')
OUT_DIFF = os.path.join(OUT_DIR, 'vec_vs_base.csv')
OUT_REPORT = os.path.join(OUT_DIR, 'analysis_report.txt')
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['core','strategy','rows','avg_comp_ms','avg_decomp_ms','avg_ratio','avg_comp_MBps','avg_decomp_MBps'])
    for m in metrics:
        def fmt(v): return f"{v:.3f}" if (v is not None and not math.isnan(v)) else ''
        w.writerow([m['core'],m['strategy'],m['count'],fmt(m['avg_comp_ms']),fmt(m['avg_decomp_ms']),fmt(m['avg_ratio']),fmt(m['avg_comp_MBps']),fmt(m['avg_decomp_MBps'])])

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
        if pct is None:
            s = 'n/a'
        else:
            s = f"{pct:.2f}%"
        f.write(f"  {d['core']}: {d['vec_strategy']} vs {d['base_strategy']}: decomp change {s}\n")
    f.write('\nNote: results are averages across samples available in summary.csv.\n')
print('Wrote', OUT_CSV, OUT_DIFF, OUT_REPORT)
