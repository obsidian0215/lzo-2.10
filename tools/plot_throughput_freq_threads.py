#!/usr/bin/env python3
"""Single combined plot: throughput vs frequency across thread counts.

Reads `exp_results/lzo_cpu/thread_limit_sweep_summary.csv` and produces a
single PNG where x=frequency (MHz), y=mean throughput (MB/s), hue=threads,
and linestyle=no_turbo (0/1). Saves to
`exp_results/lzo_cpu/plots/throughput_freq_threads_combined.png`.
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path(__file__).resolve().parents[1] / 'exp_results' / 'lzo_cpu'
PLOTS = OUT_DIR / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)

csv = OUT_DIR / 'thread_limit_sweep_summary.csv'
if not csv.exists():
    print('Missing', csv)
    raise SystemExit(1)

df = pd.read_csv(csv)
# expected columns: dir,no_turbo,max_khz,threads,mean_comp_MB_s
for c in ['max_khz','threads','mean_comp_MB_s','no_turbo']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# drop rows with missing throughput
df = df.dropna(subset=['mean_comp_MB_s','max_khz','threads'])

# convert kHz -> MHz for nicer x-axis
df['freq_MHz'] = df['max_khz'] / 1000.0

# sort for plotting
df = df.sort_values(['no_turbo','threads','freq_MHz'])

sns.set(style='whitegrid')

plt.figure(figsize=(12,7))
ax = sns.lineplot(
    data=df,
    x='freq_MHz',
    y='mean_comp_MB_s',
    hue='threads',
    style='no_turbo',
    markers=True,
    dashes=True,
    palette='tab10',
    ci=None,
)

ax.set_xlabel('CPU limit frequency (MHz)')
ax.set_ylabel('Mean throughput (MB/s)')
ax.set_title('Throughput vs CPU frequency — threads (hue), no_turbo (style)')

# Improve legend: show threads labels and map no_turbo to human text
handles, labels = ax.get_legend_handles_labels()
# Save
out = PLOTS / 'throughput_freq_threads_combined.png'
plt.tight_layout()
plt.savefig(out, dpi=200)
print('Wrote', out)
print('Used CSV:', csv)
