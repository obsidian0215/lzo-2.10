#!/usr/bin/env python3
"""
Microbenchmark-style breakdown and visualization for param_scan results.
Uses aggregated sample-config CSV `param_scan_sample_config_agg.csv` produced by `analyze_param_scan.py`.

Outputs:
 - exp_results/param_scan_analysis/param_scan_microbench_topN.csv
 - exp_results/param_scan_analysis/plot_microbench_topN_stacked_{mode}.png
 - exp_results/param_scan_analysis/microbench_topN_summary.md

This script should be run in the conda environment `dirtytrack` to ensure plotting works.
"""
import argparse
import csv
import os
import math
from collections import defaultdict
import statistics

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns
    sns.set(style='whitegrid')
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


def read_sample_agg(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ['size_mb','total_tp','kernel_tp','read_time_ms','upload_time_ms','write_time_ms','download_time_ms','total_time_ms']:
                try:
                    row[k] = float(row.get(k) or 0.0)
                except Exception:
                    row[k] = 0.0
            rows.append(row)
    return rows


def bucket_label(size_mb):
    if size_mb < 1.0:
        return 'tiny_<1MB'
    if size_mb < 10.0:
        return 'small_1-10MB'
    if size_mb < 100.0:
        return 'medium_10-100MB'
    if size_mb < 500.0:
        return 'large_100-500MB'
    return 'huge_>500MB'


def safe_mean(vals):
    try:
        return statistics.mean(vals) if vals else 0.0
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-agg', default='exp_results/param_scan_analysis/param_scan_sample_config_agg.csv')
    parser.add_argument('--outdir', default='exp_results/param_scan_analysis')
    parser.add_argument('--topN', type=int, default=10, help='Top N largest samples to analyze')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load aggregated rows
    rows = read_sample_agg(args.sample_agg)

    # identify top N samples by size
    sample_sizes = {}
    for r in rows:
        sample_sizes[r['sample']] = float(r.get('size_mb') or 0.0)
    # sort samples by size desc
    samples_sorted = sorted(sample_sizes.items(), key=lambda x: x[1], reverse=True)
    top_samples = [s for s,_ in samples_sorted[:args.topN]]

    # aggregate per sample/mode/tool
    data_map = defaultdict(lambda: defaultdict(dict))  # sample -> mode -> tool -> row
    for r in rows:
        sample = r['sample']
        mode = r['mode']
        tool = r['tool']
        # pick mean row for the config (these are already mean per sample-config)
        # But there may be multiple configs per tool; we'll take the max total_tp one (best config)
        existing = data_map[sample][mode].get(tool)
        if not existing or r['total_tp'] > existing['total_tp']:
            data_map[sample][mode][tool] = r

    outfile = os.path.join(args.outdir, f'param_scan_microbench_top{args.topN}.csv')
    with open(outfile, 'w', newline='') as f:
        w = csv.writer(f)
        headers = ['sample','size_mb','mode','tool','total_tp','total_time_ms','kernel_time_ms','upload_time_ms','download_time_ms','read_time_ms','write_time_ms','kernel_pct','upload_pct','download_pct','read_pct','write_pct']
        w.writerow(headers)
        for sample in top_samples:
            for mode in ('compress','decompress'):
                for tool, row in data_map.get(sample, {}).get(mode, {}).items():
                    total_time = float(row.get('total_time_ms') or 0.0)
                    k = float(row.get('kernel_time_ms') or 0.0)
                    up = float(row.get('upload_time_ms') or 0.0)
                    down = float(row.get('download_time_ms') or 0.0)
                    rd = float(row.get('read_time_ms') or 0.0)
                    wr = float(row.get('write_time_ms') or 0.0)
                    # avoid division by zero
                    def pct(x):
                        return (x / total_time * 100.0) if total_time > 0 else 0.0
                    w.writerow([sample, row.get('size_mb'), mode, tool, row.get('total_tp') or 0.0, total_time, k, up, down, rd, wr, pct(k), pct(up), pct(down), pct(rd), pct(wr)])

    print('Wrote microbench CSV: ', outfile)

    # produce stacked bar plots by mode
    if not PLOTTING_AVAILABLE:
        print('Plotting libs unavailable; skipping PNG generation')
        return

    for mode in ('compress','decompress'):
        samples = top_samples
        # prepare stacked components per sample & per tool
        # We'll produce one plot per tool showing stacked breakdown for top samples
        tools_set = set()
        for s in samples:
            tools_set.update(data_map.get(s, {}).get(mode, {}).keys())
        tools_list = sorted(tools_set)

        for tool in tools_list:
            labels = []
            rd_vals = []
            up_vals = []
            k_vals = []
            down_vals = []
            wr_vals = []
            for s in samples:
                row = data_map.get(s, {}).get(mode, {}).get(tool)
                if not row:
                    # zeros
                    labels.append(os.path.basename(s))
                    rd_vals.append(0.0)
                    up_vals.append(0.0)
                    k_vals.append(0.0)
                    down_vals.append(0.0)
                    wr_vals.append(0.0)
                else:
                    labels.append(os.path.basename(s))
                    total_time = float(row.get('total_time_ms') or 0.0)
                    rd = float(row.get('read_time_ms') or 0.0)
                    up = float(row.get('upload_time_ms') or 0.0)
                    k = float(row.get('kernel_time_ms') or 0.0)
                    down = float(row.get('download_time_ms') or 0.0)
                    wr = float(row.get('write_time_ms') or 0.0)
                    # convert to percent of total to be stackable
                    def pct(x):
                        return (x / total_time * 100.0) if total_time > 0 else 0.0
                    rd_vals.append(pct(rd))
                    up_vals.append(pct(up))
                    k_vals.append(pct(k))
                    down_vals.append(pct(down))
                    wr_vals.append(pct(wr))

            # stacked bar
            fig, ax = plt.subplots(figsize=(10,6))
            x = list(range(len(labels)))
            bottom = [0.0] * len(labels)
            components = [('read', rd_vals), ('upload', up_vals), ('kernel', k_vals), ('download', down_vals), ('write', wr_vals)]
            palette = sns.color_palette('tab10', n_colors=len(components)) if PLOTTING_AVAILABLE else None
            for i, (comp_name, vals) in enumerate(components):
                ax.bar(x, vals, bottom=bottom, label=comp_name, color=palette[i] if palette else None)
                bottom = [bottom[j] + vals[j] for j in range(len(vals))]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Percent of total time (%)')
            ax.set_title(f'Microbench top {args.topN} - {mode} - {tool} (time breakdown %)')
            ax.legend()
            fig.tight_layout()
            outpng = os.path.join(args.outdir, f'plot_microbench_top{args.topN}_{mode}_{tool}_stacked.png')
            fig.savefig(outpng, dpi=150)
            plt.close(fig)
            print('Wrote plot', outpng)

    # produce summary markdown
    md = os.path.join(args.outdir, f'microbench_top{args.topN}_summary.md')
    with open(md, 'w') as mdout:
        mdout.write('# Microbench Top{} Summary\n\n'.format(args.topN))
        mdout.write('Generated with sample aggregated CSV: {}\n\n'.format(args.sample_agg))
        mdout.write('## Top-Samples by Size\n')
        for s, size in samples_sorted[:args.topN]:
            mdout.write(f'- {s}: {size:.2f} MB\n')
        mdout.write('\n')
        mdout.write('Per-mode breakdown plots attached in the directory.\n')
    print('Wrote markdown:', md)


if __name__ == '__main__':
    main()
