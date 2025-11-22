#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

OUTDIR = '/root/lzo-2.10/exp_results/lzo_cpu'
CSV = os.path.join(OUTDIR, 'aggregate_results.csv')
PLOTS = os.path.join(OUTDIR, 'plots')
os.makedirs(PLOTS, exist_ok=True)

if not os.path.exists(CSV):
    print('aggregate_results.csv not found at', CSV)
    sys.exit(2)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print('Missing plotting libraries:', e)
    print('Install matplotlib and seaborn in your conda env `dirtytrack` and re-run:')
    print('  conda run -n dirtytrack python3', __file__)
    sys.exit(3)

sns.set(context='paper', style='whitegrid', font_scale=1.0)

df = pd.read_csv(CSV)
# identify configs from column names
cols = [c for c in df.columns if c not in ('alg','threads')]
configs = sorted(set([c.split('_comp_MB_s')[0] for c in cols if c.endswith('_comp_MB_s')]))

# row index label
df['id'] = df['alg'] + '__threads=' + df['threads'].astype(str)
index = df['id']

# build matrices
comp_mat = df[[f'{cfg}_comp_MB_s' for cfg in configs]].copy()
comp_mat.index = index
comp_mat.columns = configs

delta_mat = df[[f'{cfg}_comp_delta_pct' for cfg in configs]].copy()
delta_mat.index = index
delta_mat.columns = configs

decomp_mat = df[[f'{cfg}_decomp_delta_pct' for cfg in configs]].copy()
decomp_mat.index = index
decomp_mat.columns = configs

# helper to plot heatmap
def plot_heatmap(mat, fname, title, cmap='RdBu_r', center=None, fmt='.1f'):
    plt.figure(figsize=(max(6, len(configs)*0.6), max(6, len(mat.index)*0.25)))
    ax = sns.heatmap(mat, cmap=cmap, center=center, annot=True, fmt=fmt, linewidths=.5, cbar_kws={'label': title})
    ax.set_xlabel('CPU config')
    ax.set_ylabel('alg__threads')
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(PLOTS, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print('Wrote', out)

# Plot absolute comp_MB_s (log scale to compress ranges)
plot_heatmap(np.log1p(comp_mat), 'comp_MB_s_log1p_heatmap.png', 'log1p(comp_MB_s) heatmap', cmap='viridis', center=None, fmt='.2f')
# Plot comp delta %
plot_heatmap(delta_mat, 'comp_delta_pct_heatmap.png', 'comp delta % vs baseline', cmap='RdBu_r', center=0.0, fmt='.1f')
# Plot decomp delta %
plot_heatmap(decomp_mat, 'decomp_delta_pct_heatmap.png', 'decomp delta % vs baseline', cmap='RdBu_r', center=0.0, fmt='.1f')

# Also write a compact CSV of deltas only
deltas_only = pd.concat([df[['alg','threads']], delta_mat.add_prefix('comp_delta_') , decomp_mat.add_prefix('decomp_delta_')], axis=1)
deltas_csv = os.path.join(OUTDIR, 'aggregate_deltas.csv')
deltas_only.to_csv(deltas_csv, index=False)
print('Wrote', deltas_csv)

print('Done plotting.')
