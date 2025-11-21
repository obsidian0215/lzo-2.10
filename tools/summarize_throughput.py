#!/usr/bin/env python3
import re,os,csv,sys,statistics
root='exp_results/lzo_gpu/logs/param_scans'
if not os.path.isdir(root):
    print('logs dir not found:', root, file=sys.stderr); sys.exit(1)
comp_re=re.compile(r"\[COMP\s*\].*thrpt=([0-9.]+)\s*MB/s")
decomp_re=re.compile(r"\[DECOMP\].*thrpt=([0-9.]+)\s*MB/s")
header_re=re.compile(r"^#\s*(.*)")
fields_re=re.compile(r"(\w+)=([^\s]+)")
results={}
map_mode_seen=False
count=0
for dirpath,dirs,files in os.walk(root):
    for f in files:
        if not f.endswith('.log'): continue
        path=os.path.join(dirpath,f)
        try:
            with open(path,'r',errors='ignore') as fh:
                text=fh.read()
        except Exception as e:
            continue
        # header
        header=''
        m=header_re.search(text)
        if m:
            header=m.group(1)
        # parse header fields
        hdr_fields=dict(fields_re.findall(header))
        comp=hdr_fields.get('COMP','')
        # MAP_MODE is optional: only present if user configured it.
        mode=hdr_fields.get('MAP_MODE','')
        if mode:
            map_mode_seen=True
        wg=hdr_fields.get('WG','')
        # fallback: try to extract wg from path (no decomp_mode dimension anymore)
        parts=path.split(os.sep)
        for p in parts:
            if p.startswith('wg_'):
                try:
                    wg_p=re.search(r'wg_(\d+)',p).group(1)
                    if not wg: wg=wg_p
                except:
                    pass
        # find thrpt
        comp_m=comp_re.search(text)
        decomp_m=decomp_re.search(text)
        comp_thrpt=float(comp_m.group(1)) if comp_m else None
        decomp_thrpt=float(decomp_m.group(1)) if decomp_m else None
        key=(comp,mode,wg)
        if key not in results:
            results[key]={'comp':[],'decomp':[],'files':0}
        if comp_thrpt is not None:
            results[key]['comp'].append(comp_thrpt)
        if decomp_thrpt is not None:
            results[key]['decomp'].append(decomp_thrpt)
        results[key]['files']+=1
        count+=1
# write CSV
out='exp_results/lzo_gpu/logs/throughput_summary.csv'
with open(out,'w',newline='') as csvf:
    w=csv.writer(csvf)
    rows=[]
    # Choose header depending on whether MAP_MODE was seen in any log
    if map_mode_seen:
        header = ['comp','mode','wg','samples','comp_mean','comp_median','comp_stdev','decomp_mean','decomp_median','decomp_stdev']
    else:
        header = ['comp','wg','samples','comp_mean','comp_median','comp_stdev','decomp_mean','decomp_median','decomp_stdev']
    w.writerow(header)
    for k,v in results.items():
        comp_list=v['comp']
        decomp_list=v['decomp']
        comp_mean=statistics.mean(comp_list) if comp_list else ''
        comp_median=statistics.median(comp_list) if comp_list else ''
        comp_stdev=statistics.pstdev(comp_list) if len(comp_list)>1 else 0.0 if comp_list else ''
        decomp_mean=statistics.mean(decomp_list) if decomp_list else ''
        decomp_median=statistics.median(decomp_list) if decomp_list else ''
        decomp_stdev=statistics.pstdev(decomp_list) if len(decomp_list)>1 else 0.0 if decomp_list else ''
        if map_mode_seen:
            row=list(k)+[v['files'],comp_mean,comp_median,comp_stdev,decomp_mean,decomp_median,decomp_stdev]
        else:
            # k is (comp,mode,wg) but mode is unused; produce (comp,wg,...)
            comp_val,_,wg_val = k
            row=[comp_val,wg_val,v['files'],comp_mean,comp_median,comp_stdev,decomp_mean,decomp_median,decomp_stdev]
        rows.append((k,row,comp_mean,decomp_mean))
        w.writerow(row)
# print summary rankings (robust to presence/absence of MAP_MODE)
print('Total log files scanned:',count)
print('Unique configurations:',len(results))

print('\nTop 10 by Decompression mean throughput (MB/s):')
rows_with_decomp_info = []
for k,row,comp_mean,decomp_mean in rows:
    if decomp_mean == '' or decomp_mean is None:
        continue
    if map_mode_seen:
        # row: comp,mode,wg,samples,comp_mean,comp_median,comp_stdev,decomp_mean,decomp_median,decomp_stdev
        comp_val, mode_val, wg_val, samples, _, _, _, decomp_mean_val, decomp_med, decomp_std = row
    else:
        # row: comp,wg,samples,comp_mean,comp_median,comp_stdev,decomp_mean,decomp_median,decomp_stdev
        comp_val, wg_val, samples, _, _, _, decomp_mean_val, decomp_med, decomp_std = row
        mode_val = ''
    rows_with_decomp_info.append((float(decomp_mean_val), comp_val, mode_val, wg_val, samples, decomp_med, decomp_std))

rows_with_decomp_sorted = sorted(rows_with_decomp_info, key=lambda t: t[0], reverse=True)
for t in rows_with_decomp_sorted[:10]:
    de_mean, comp_val, mode_val, wg_val, samples, de_med, de_std = t
    print(f"comp={comp_val} mode={mode_val} wg={wg_val} samples={samples} decomp_mean={de_mean:.2f} med={de_med} stdev={de_std}")

print('\nTop 10 by Compression mean throughput (MB/s):')
rows_with_comp_info = []
for k,row,comp_mean,decomp_mean in rows:
    if comp_mean == '' or comp_mean is None:
        continue
    if map_mode_seen:
        comp_val, mode_val, wg_val, samples, comp_mean_val, comp_med, comp_std, _, _, _ = row
    else:
        comp_val, wg_val, samples, comp_mean_val, comp_med, comp_std, _, _, _ = row
        mode_val = ''
    rows_with_comp_info.append((float(comp_mean_val), comp_val, mode_val, wg_val, samples, comp_med, comp_std))

rows_with_comp_sorted = sorted(rows_with_comp_info, key=lambda t: t[0], reverse=True)
for t in rows_with_comp_sorted[:10]:
    c_mean, comp_val, mode_val, wg_val, samples, c_med, c_std = t
    print(f"comp={comp_val} mode={mode_val} wg={wg_val} samples={samples} comp_mean={c_mean:.2f} med={c_med} stdev={c_std}")

print('\nWrote summary CSV to',out)
