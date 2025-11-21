#!/usr/bin/env python3
"""
Generate simple plots from `analysis_summary.csv`:
 - compression ratio histogram
 - comp vs decomp scatter (MB/s)

Saves PNGs to lzo_gpu/experiments/logs/
"""
import os
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
# Use repository-level exp_results; lzo_gpu outputs live under exp_results/lzo_gpu/logs
OUT_BASE = os.path.join(ROOT, 'exp_results')
DEFAULT_LOG_DIR = os.path.join(OUT_BASE, 'lzo_gpu', 'logs')

# Parse command-line args: allow specifying an input CSV
parser = argparse.ArgumentParser(description='Generate plots from analysis_summary.csv')
parser.add_argument('csv', nargs='?', default=None, help='Path to analysis_summary.csv (optional)')
parser.add_argument('-i','--input', dest='csv_in', default=None, help='Path to analysis_summary.csv (optional)')
args = parser.parse_args()

# Determine CSV path and output directory
if args.csv_in:
    ANAL = args.csv_in
elif args.csv:
    ANAL = args.csv
else:
    ANAL = os.path.join(DEFAULT_LOG_DIR, 'analysis_summary.csv')

LOG_DIR = os.path.dirname(ANAL) if os.path.dirname(ANAL) else DEFAULT_LOG_DIR
OUT1 = os.path.join(LOG_DIR, 'comp_ratio_hist.png')
OUT2 = os.path.join(LOG_DIR, 'comp_vs_decomp_scatter.png')

try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    print('matplotlib/numpy not available, skipping plots:', e)
    raise SystemExit(0)

ratios = []
comp_mb = []
decomp_mb = []
labels = []
if os.path.exists(ANAL):
    with open(ANAL,newline='') as f:
        r = csv.DictReader(f)
        for rec in r:
            try:
                rv = float(rec.get('avg_ratio','') or 0)
            except Exception:
                rv = 0
            try:
                cb = float(rec.get('avg_comp_MBps','') or float('nan'))
            except Exception:
                cb = float('nan')
            try:
                db = float(rec.get('avg_decomp_MBps','') or float('nan'))
            except Exception:
                db = float('nan')
            if rv>0:
                ratios.append(rv)
            if not np.isnan(cb) and not np.isnan(db):
                comp_mb.append(cb)
                decomp_mb.append(db)
                labels.append(f"{rec.get('core','')}")

if not os.path.exists(ANAL):
    print('Analysis CSV not found:', ANAL, '\nNo plots generated. Run tools/run_lzo_gpu.sh analyze to produce analysis_summary.csv under exp_results/lzo_gpu/logs')
    raise SystemExit(0)

# histogram of ratios
if ratios:
    plt.figure(figsize=(6,4))
    plt.hist(ratios, bins=30)
    plt.xlabel('Compression ratio (orig/comp)')
    plt.ylabel('count')
    plt.title('Compression ratio distribution')
    plt.tight_layout()
    plt.savefig(OUT1)
    print('Wrote',OUT1)
else:
    print('No ratios to plot')

# scatter comp vs decomp
if comp_mb and decomp_mb:
    plt.figure(figsize=(6,6))
    plt.scatter(comp_mb, decomp_mb, alpha=0.7)
    plt.xlabel('Comp MB/s')
    plt.ylabel('Decomp MB/s')
    plt.title('Comp vs Decomp throughput')
    # annotate top few
    for i,lab in enumerate(labels[:20]):
        plt.annotate(lab, (comp_mb[i], decomp_mb[i]), fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT2)
    print('Wrote',OUT2)
else:
    print('Not enough comp/decomp MB/s to plot scatter')
