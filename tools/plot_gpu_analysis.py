#!/usr/bin/env python3
"""
Unified GPU plotting script - generates all GPU analysis plots from analyze.py output.

Generates:
  1. Compression ratio histogram
  2. Comp vs Decomp scatter plot
  3. Top configurations bar charts
  4. Distribution histograms
  5. Markdown summary report

Input: analysis_summary.csv (from analyze.py)
Output: Multiple PNG plots + REPORT.md

Usage:
  ./plot_gpu_analysis.py                           # Use default path
  ./plot_gpu_analysis.py -i path/to/summary.csv    # Specify input
"""

import os
import sys
import csv
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Generate comprehensive GPU analysis plots')
parser.add_argument('-i', '--input', default=None, help='Path to analysis_summary.csv')
parser.add_argument('-o', '--output-dir', default=None, help='Output directory for plots')
args = parser.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(ROOT, 'exp_results', 'lzo_gpu', 'logs')

# Determine paths
if args.input:
    CSV_PATH = args.input
    DEFAULT_OUT = os.path.dirname(CSV_PATH)
else:
    CSV_PATH = os.path.join(OUT_BASE, 'analysis_summary.csv')
    DEFAULT_OUT = OUT_BASE

OUT_DIR = args.output_dir if args.output_dir else os.path.join(DEFAULT_OUT, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(DEFAULT_OUT, 'REPORT.md')

# Import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('ERROR: matplotlib not available. Install with: pip3 install matplotlib', file=sys.stderr)
    sys.exit(1)

# Check if CSV exists
if not os.path.exists(CSV_PATH):
    print(f'ERROR: Analysis CSV not found: {CSV_PATH}', file=sys.stderr)
    print('Run: tools/analyze.py first to generate analysis_summary.csv', file=sys.stderr)
    sys.exit(1)

# Read data
rows = []
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    
    for row in reader:
        # Convert numeric fields
        try:
            row['avg_ratio'] = float(row.get('avg_ratio', '') or 0)
        except:
            row['avg_ratio'] = 0
        
        for key in ['avg_comp_MBps', 'avg_decomp_MBps', 'comp_median', 'decomp_median', 
                    'comp_stdev', 'decomp_stdev']:
            try:
                row[key] = float(row.get(key, '') or 'nan')
            except:
                row[key] = float('nan')
        
        try:
            row['n_samples'] = int(row.get('n_samples', 0))
        except:
            row['n_samples'] = 0
            
        rows.append(row)

print(f'Loaded {len(rows)} configurations from {CSV_PATH}')

# Extract data for plots
comp_ratios = [r['avg_ratio'] for r in rows if r['avg_ratio'] > 0]
comp_throughput = [r['avg_comp_MBps'] for r in rows if not np.isnan(r['avg_comp_MBps'])]
decomp_throughput = [r['avg_decomp_MBps'] for r in rows if not np.isnan(r['avg_decomp_MBps'])]

# Scatter plot data (only where both are valid)
scatter_comp = []
scatter_decomp = []
scatter_labels = []
for r in rows:
    if not np.isnan(r['avg_comp_MBps']) and not np.isnan(r['avg_decomp_MBps']):
        scatter_comp.append(r['avg_comp_MBps'])
        scatter_decomp.append(r['avg_decomp_MBps'])
        label = r.get('core', '') or f"{r.get('comp','')}_wg{r.get('wg','')}"
        scatter_labels.append(label)

# Top configurations
def top_n_by(key, n=15):
    valid = [r for r in rows if not np.isnan(r[key])]
    sorted_rows = sorted(valid, key=lambda x: x[key], reverse=True)
    return sorted_rows[:n]

top_comp = top_n_by('avg_comp_MBps', 15)
top_decomp = top_n_by('avg_decomp_MBps', 15)

# === PLOT 1: Compression Ratio Histogram ===
if comp_ratios:
    plt.figure(figsize=(8, 5))
    plt.hist(comp_ratios, bins=30, alpha=0.75, edgecolor='black')
    plt.xlabel('Compression Ratio (original/compressed)')
    plt.ylabel('Count')
    plt.title('Compression Ratio Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, 'comp_ratio_hist.png')
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f'  ✓ {out1}')
else:
    out1 = None
    print('  ⚠ No compression ratios to plot')

# === PLOT 2: Comp vs Decomp Scatter ===
if scatter_comp and scatter_decomp:
    plt.figure(figsize=(8, 8))
    plt.scatter(scatter_comp, scatter_decomp, alpha=0.6, s=50)
    plt.xlabel('Compression Throughput (MB/s)')
    plt.ylabel('Decompression Throughput (MB/s)')
    plt.title('Compression vs Decompression Throughput')
    plt.grid(True, alpha=0.3)
    
    # Annotate top 10
    for i in range(min(10, len(scatter_labels))):
        plt.annotate(scatter_labels[i], 
                    (scatter_comp[i], scatter_decomp[i]),
                    fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, 'comp_vs_decomp_scatter.png')
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f'  ✓ {out2}')
else:
    out2 = None
    print('  ⚠ Not enough data for scatter plot')

# === PLOT 3: Top Compression Throughput ===
if top_comp:
    labels = [f"{r.get('core', r.get('comp',''))}\nwg{r.get('wg','')}" for r in top_comp]
    values = [r['avg_comp_MBps'] for r in top_comp]
    errors = [r.get('comp_stdev', 0) for r in top_comp]
    
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, values, yerr=errors, align='center', alpha=0.8, capsize=5)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Compression Throughput (MB/s)')
    plt.title('Top 15 Configurations by Compression Throughput')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, 'top_comp_throughput.png')
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f'  ✓ {out3}')
else:
    out3 = None

# === PLOT 4: Top Decompression Throughput ===
if top_decomp:
    labels = [f"{r.get('core', r.get('comp',''))}\nwg{r.get('wg','')}" for r in top_decomp]
    values = [r['avg_decomp_MBps'] for r in top_decomp]
    errors = [r.get('decomp_stdev', 0) for r in top_decomp]
    
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, values, yerr=errors, align='center', alpha=0.8, 
            color='orange', capsize=5)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Decompression Throughput (MB/s)')
    plt.title('Top 15 Configurations by Decompression Throughput')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out4 = os.path.join(OUT_DIR, 'top_decomp_throughput.png')
    plt.savefig(out4, dpi=150)
    plt.close()
    print(f'  ✓ {out4}')
else:
    out4 = None

# === PLOT 5 & 6: Distribution Histograms ===
out5 = out6 = None

if comp_throughput:
    plt.figure(figsize=(8, 5))
    plt.hist(comp_throughput, bins=50, alpha=0.75, edgecolor='black')
    plt.xlabel('Compression Throughput (MB/s)')
    plt.ylabel('Count')
    plt.title('Compression Throughput Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out5 = os.path.join(OUT_DIR, 'hist_comp_throughput.png')
    plt.savefig(out5, dpi=150)
    plt.close()
    print(f'  ✓ {out5}')

if decomp_throughput:
    plt.figure(figsize=(8, 5))
    plt.hist(decomp_throughput, bins=50, alpha=0.75, 
             color='orange', edgecolor='black')
    plt.xlabel('Decompression Throughput (MB/s)')
    plt.ylabel('Count')
    plt.title('Decompression Throughput Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out6 = os.path.join(OUT_DIR, 'hist_decomp_throughput.png')
    plt.savefig(out6, dpi=150)
    plt.close()
    print(f'  ✓ {out6}')

# === Generate Markdown Report ===
with open(REPORT_PATH, 'w') as f:
    f.write('# LZO GPU Performance Analysis Report\n\n')
    f.write(f'Generated from: `{os.path.basename(CSV_PATH)}`\n\n')
    f.write(f'Total configurations analyzed: **{len(rows)}**\n\n')
    
    # Summary statistics
    if comp_throughput:
        f.write('## Compression Performance\n\n')
        f.write(f'- Mean: {np.mean(comp_throughput):.2f} MB/s\n')
        f.write(f'- Median: {np.median(comp_throughput):.2f} MB/s\n')
        f.write(f'- Max: {np.max(comp_throughput):.2f} MB/s\n')
        f.write(f'- Min: {np.min(comp_throughput):.2f} MB/s\n\n')
    
    if decomp_throughput:
        f.write('## Decompression Performance\n\n')
        f.write(f'- Mean: {np.mean(decomp_throughput):.2f} MB/s\n')
        f.write(f'- Median: {np.median(decomp_throughput):.2f} MB/s\n')
        f.write(f'- Max: {np.max(decomp_throughput):.2f} MB/s\n')
        f.write(f'- Min: {np.min(decomp_throughput):.2f} MB/s\n\n')
    
    # Top compression configs
    if top_comp:
        f.write('## Top 15 Compression Configurations\n\n')
        f.write('| Rank | Config | WG | Throughput (MB/s) | Samples | Std Dev |\n')
        f.write('|-----:|--------|---:|-----------------:|--------:|--------:|\n')
        for i, r in enumerate(top_comp, 1):
            config = r.get('core', r.get('comp', ''))
            wg = r.get('wg', '')
            f.write(f'| {i} | {config} | {wg} | {r["avg_comp_MBps"]:.2f} | '
                   f'{r["n_samples"]} | {r.get("comp_stdev", 0):.2f} |\n')
        f.write('\n')
        if out3:
            rel_path = os.path.relpath(out3, os.path.dirname(REPORT_PATH))
            f.write(f'![Top Compression]({rel_path})\n\n')
    
    # Top decompression configs
    if top_decomp:
        f.write('## Top 15 Decompression Configurations\n\n')
        f.write('| Rank | Config | WG | Throughput (MB/s) | Samples | Std Dev |\n')
        f.write('|-----:|--------|---:|-----------------:|--------:|--------:|\n')
        for i, r in enumerate(top_decomp, 1):
            config = r.get('core', r.get('comp', ''))
            wg = r.get('wg', '')
            f.write(f'| {i} | {config} | {wg} | {r["avg_decomp_MBps"]:.2f} | '
                   f'{r["n_samples"]} | {r.get("decomp_stdev", 0):.2f} |\n')
        f.write('\n')
        if out4:
            rel_path = os.path.relpath(out4, os.path.dirname(REPORT_PATH))
            f.write(f'![Top Decompression]({rel_path})\n\n')
    
    # Additional plots
    f.write('## Distribution Analysis\n\n')
    
    if out1:
        rel_path = os.path.relpath(out1, os.path.dirname(REPORT_PATH))
        f.write(f'### Compression Ratio\n\n![Compression Ratio]({rel_path})\n\n')
    
    if out2:
        rel_path = os.path.relpath(out2, os.path.dirname(REPORT_PATH))
        f.write(f'### Compression vs Decompression\n\n![Scatter]({rel_path})\n\n')
    
    if out5:
        rel_path = os.path.relpath(out5, os.path.dirname(REPORT_PATH))
        f.write(f'### Compression Throughput Distribution\n\n![Comp Dist]({rel_path})\n\n')
    
    if out6:
        rel_path = os.path.relpath(out6, os.path.dirname(REPORT_PATH))
        f.write(f'### Decompression Throughput Distribution\n\n![Decomp Dist]({rel_path})\n\n')
    
    f.write('---\n\n')
    f.write('Generated by `tools/plot_gpu_analysis.py`\n')

print(f'\n✓ Report written to: {REPORT_PATH}')
print(f'✓ Plots saved to: {OUT_DIR}')
