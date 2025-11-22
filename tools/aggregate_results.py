#!/usr/bin/env python3
import csv
import glob
import os
from collections import defaultdict
import statistics

OUTDIR = '/root/lzo-2.10/exp_results/lzo_cpu'
PAT = os.path.join(OUTDIR, '*', 'summary.csv')
files = glob.glob(PAT)
if not files:
    print('No summary.csv files found under', OUTDIR)
    raise SystemExit(1)

# Read all rows into structure: data[config].append(rowdict)
data = {}
headers = None
for path in sorted(files):
    config = os.path.basename(os.path.dirname(path))
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        if headers is None:
            headers = r.fieldnames
        rows = list(r)
    data[config] = rows

# determine baseline config
baseline = 'gov_performance_nt0_max0_pin' if 'gov_performance_nt0_max0_pin' in data else sorted(data.keys())[0]
print('Found configs:', ', '.join(sorted(data.keys())))
print('Baseline config for relative change:', baseline)

# Aggregate: groups by (alg, threads) per config
# metrics: comp_MB_s (float), decomp_MB_s (float), ratio_pct (float)
metrics = ['comp_MB_s', 'decomp_MB_s', 'ratio_pct']
agg = defaultdict(lambda: defaultdict(list))  # agg[(alg,threads)][config] -> list of dicts
for config, rows in data.items():
    for r in rows:
        alg = r.get('alg','')
        threads = r.get('threads','')
        key = (alg, threads)
        # try parse metrics
        try:
            comp = float(r.get('comp_MB_s','') or 0.0)
        except:
            comp = 0.0
        try:
            decomp = float(r.get('decomp_MB_s','') or 0.0)
        except:
            decomp = 0.0
        try:
            ratio = float(r.get('ratio_pct','') or 0.0)
        except:
            ratio = 0.0
        agg[key][config].append({'comp':comp,'decomp':decomp,'ratio':ratio})

# produce summary per key per config (mean)
summary_rows = []
configs = sorted(data.keys())
for key in sorted(agg.keys(), key=lambda x:(x[0], int(x[1]) if x[1].isdigit() else x[1])):
    alg, threads = key
    row = {'alg':alg,'threads':threads}
    # compute baseline values
    base_val = {}
    for m in metrics:
        vals = [d['comp'] for d in []]  # placeholder
    for cfg in configs:
        vals = agg[key].get(cfg, [])
        if vals:
            comp_m = statistics.mean([v['comp'] for v in vals])
            decomp_m = statistics.mean([v['decomp'] for v in vals])
            ratio_m = statistics.mean([v['ratio'] for v in vals])
        else:
            comp_m = decomp_m = ratio_m = 0.0
        row[f'{cfg}_comp_MB_s'] = round(comp_m,3)
        row[f'{cfg}_decomp_MB_s'] = round(decomp_m,3)
        row[f'{cfg}_ratio_pct'] = round(ratio_m,3)
    # compute relative change to baseline for comp/decomp
    base_comp = row.get(f'{baseline}_comp_MB_s', 0.0)
    base_decomp = row.get(f'{baseline}_decomp_MB_s', 0.0)
    for cfg in configs:
        compv = row[f'{cfg}_comp_MB_s']
        decompv = row[f'{cfg}_decomp_MB_s']
        row[f'{cfg}_comp_delta_pct'] = round(((compv - base_comp)/base_comp*100) if base_comp else 0.0,2)
        row[f'{cfg}_decomp_delta_pct'] = round(((decompv - base_decomp)/base_decomp*100) if base_decomp else 0.0,2)
    summary_rows.append(row)

# write aggregate CSV
outcsv = os.path.join(OUTDIR, 'aggregate_results.csv')
allcols = ['alg','threads']
for cfg in configs:
    allcols += [f'{cfg}_comp_MB_s', f'{cfg}_decomp_MB_s', f'{cfg}_ratio_pct', f'{cfg}_comp_delta_pct', f'{cfg}_decomp_delta_pct']
with open(outcsv,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=allcols)
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

print('\nWrote aggregate CSV to', outcsv)

# print a compact human-readable comparison for top N keys
print('\nSample comparison (comp_MB_s):')
for r in summary_rows:
    alg = r['alg']; threads = r['threads']
    comps = [(cfg, r[f'{cfg}_comp_MB_s']) for cfg in configs]
    best = max(comps, key=lambda x:x[1])
    worst = min(comps, key=lambda x:x[1])
    print(f'alg={alg} threads={threads}: best={best[0]} {best[1]} MB/s, worst={worst[0]} {worst[1]} MB/s, baseline={baseline} {r[f"{baseline}_comp_MB_s"]} MB/s')

print('\nDone.')
