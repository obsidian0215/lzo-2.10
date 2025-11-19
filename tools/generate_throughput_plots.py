#!/usr/bin/env python3
"""Generate throughput plots and a markdown report from throughput_summary.csv
"""
import os,sys,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV='exp_results/lzo_gpu/logs/throughput_summary.csv'
OUT_DIR='exp_results/lzo_gpu/logs/plots'
REPORT='exp_results/lzo_gpu/logs/REPORT.md'
os.makedirs(OUT_DIR,exist_ok=True)

if not os.path.isfile(CSV):
    print('CSV not found:',CSV,file=sys.stderr); sys.exit(1)

rows=[]
with open(CSV,'r',newline='') as fh:
    r=csv.DictReader(fh)
    for row in r:
        # convert numeric fields
        try:
            row['samples']=int(row['samples'])
        except:
            row['samples']=0
        for k in ('comp_mean','comp_median','comp_stdev','decomp_mean','decomp_median','decomp_stdev'):
            try:
                row[k]=float(row[k]) if row[k] != '' else np.nan
            except:
                row[k]=np.nan
        rows.append(row)

def top_n_by(key,n=10):
    arr=[r for r in rows if not np.isnan(r[key])]
    arr=sorted(arr,key=lambda x: x[key],reverse=True)
    return arr[:n]

# top lists
top_decomp=top_n_by('decomp_mean',15)
top_comp=top_n_by('comp_mean',15)

def bar_plot(list_rows,key,title,fname, ylabel):
    labels=[f"{r['comp']}/{r['mode']}/wg{r['wg']}/v{r['vlen']}" for r in list_rows]
    vals=[r[key] for r in list_rows]
    errs=[r[key.replace('mean','stdev')] if (key=='decomp_mean' or key=='comp_mean') else 0 for r in list_rows]
    x=np.arange(len(labels))
    plt.figure(figsize=(10,6))
    plt.bar(x,vals,yerr=errs,align='center',alpha=0.8)
    plt.xticks(x,labels,rotation=45,ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out=os.path.join(OUT_DIR,fname)
    plt.savefig(out)
    plt.close()
    return out

plots=[]
plots.append(bar_plot(top_decomp,'decomp_mean','Top configs by Decompression mean throughput','top_decomp_mean.png','Decompression MB/s'))
plots.append(bar_plot(top_comp,'comp_mean','Top configs by Compression mean throughput','top_comp_mean.png','Compression MB/s'))

# distribution histograms
for key,title,fname in [('decomp_mean','Decompression mean distribution','hist_decomp_mean.png'),('comp_mean','Compression mean distribution','hist_comp_mean.png')]:
    vals=[r[key] for r in rows if not np.isnan(r[key])]
    plt.figure(figsize=(8,5))
    plt.hist(vals,bins=50,alpha=0.8)
    plt.title(title)
    plt.xlabel('MB/s')
    plt.ylabel('Count')
    plt.tight_layout()
    out=os.path.join(OUT_DIR,fname)
    plt.savefig(out)
    plt.close()
    plots.append(out)

# generate REPORT.md
with open(REPORT,'w') as rep:
    rep.write('# LZO GPU Throughput Summary\n\n')
    rep.write('This report summarizes compression and decompression throughput across the parameter scan.\n\n')
    rep.write('## Top configurations by Decompression mean throughput\n\n')
    rep.write('| Rank | COMP/MODE | strategy | decomp_mode | WG | VLEN | samples | decomp_mean (MB/s) | decomp_median | decomp_stdev |\n')
    rep.write('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for i,r in enumerate(top_decomp,1):
        rep.write(f"| {i} | {r['comp']}/{r['mode']} | {r['strategy']} | {r['decomp_mode']} | {r['wg']} | {r['vlen']} | {r['samples']} | {r['decomp_mean']:.2f} | {r['decomp_median']} | {r['decomp_stdev']:.2f} |\n")
    rep.write('\n')
    rep.write('![Top decomp](' + os.path.relpath(plots[0], start=os.path.dirname(REPORT)) + ')\n\n')

    rep.write('## Top configurations by Compression mean throughput\n\n')
    rep.write('| Rank | COMP/MODE | strategy | decomp_mode | WG | VLEN | samples | comp_mean (MB/s) | comp_median | comp_stdev |\n')
    rep.write('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n')
    for i,r in enumerate(top_comp,1):
        rep.write(f"| {i} | {r['comp']}/{r['mode']} | {r['strategy']} | {r['decomp_mode']} | {r['wg']} | {r['vlen']} | {r['samples']} | {r['comp_mean']:.2f} | {r['comp_median']} | {r['comp_stdev']:.2f} |\n")
    rep.write('\n')
    rep.write('![Top comp](' + os.path.relpath(plots[1], start=os.path.dirname(REPORT)) + ')\n\n')

    rep.write('## Distribution plots\n\n')
    rep.write('!['+ 'decomp hist](' + os.path.relpath(plots[2], start=os.path.dirname(REPORT)) + ')\n\n')
    rep.write('!['+ 'comp hist](' + os.path.relpath(plots[3], start=os.path.dirname(REPORT)) + ')\n\n')

    rep.write('Generated with `tools/generate_throughput_plots.py`. See `throughput_summary.csv` for full aggregates.\n')

print('Wrote plots to',OUT_DIR)
print('Wrote report to',REPORT)
