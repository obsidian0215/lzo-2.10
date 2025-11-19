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
        mode=hdr_fields.get('MODE','')
        wg=hdr_fields.get('WG','')
        vlen=hdr_fields.get('VLEN','')
        # fallback: try to extract strategy and decomp_mode from path
        parts=path.split(os.sep)
        strategy=''
        decomp_mode=''
        for i,p in enumerate(parts):
            if p.startswith('strategy_'):
                strategy=p.replace('strategy_','')
            if p.startswith('decomp_'):
                decomp_mode=p.replace('decomp_','')
            if p.startswith('wg_') and '_v_' in p:
                # wg_128_v_4 -> extract
                s=p
                try:
                    wg_p=re.search(r'wg_(\d+)',s).group(1); vlen_p=re.search(r'v_(\d+)',s).group(1)
                    if not wg: wg=wg_p
                    if not vlen: vlen=vlen_p
                except:
                    pass
        # find thrpt
        comp_m=comp_re.search(text)
        decomp_m=decomp_re.search(text)
        comp_thrpt=float(comp_m.group(1)) if comp_m else None
        decomp_thrpt=float(decomp_m.group(1)) if decomp_m else None
        key=(comp,mode,strategy,decomp_mode,wg,vlen)
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
    w.writerow(['comp','mode','strategy','decomp_mode','wg','vlen','samples','comp_mean','comp_median','comp_stdev','decomp_mean','decomp_median','decomp_stdev'])
    rows=[]
    for k,v in results.items():
        comp_list=v['comp']
        decomp_list=v['decomp']
        comp_mean=statistics.mean(comp_list) if comp_list else ''
        comp_median=statistics.median(comp_list) if comp_list else ''
        comp_stdev=statistics.pstdev(comp_list) if len(comp_list)>1 else 0.0 if comp_list else ''
        decomp_mean=statistics.mean(decomp_list) if decomp_list else ''
        decomp_median=statistics.median(decomp_list) if decomp_list else ''
        decomp_stdev=statistics.pstdev(decomp_list) if len(decomp_list)>1 else 0.0 if decomp_list else ''
        row=list(k)+[v['files'],comp_mean,comp_median,comp_stdev,decomp_mean,decomp_median,decomp_stdev]
        rows.append((k,row,comp_mean,decomp_mean))
        w.writerow(row)
# print summary rankings
# rank by decomp_mean (desc)
rows_with_decomp=[r for (_,r,_,d) in rows if d!='' and d is not None]
rows_with_decomp_sorted=sorted(rows_with_decomp, key=lambda r: float(r[10]), reverse=True)
print('Total log files scanned:',count)
print('Unique configurations:',len(results))
print('\nTop 10 by Decompression mean throughput (MB/s):')
for r in rows_with_decomp_sorted[:10]:
    print(f"comp={r[0]} mode={r[1]} strategy={r[2]} decomp_mode={r[3]} wg={r[4]} vlen={r[5]} samples={r[6]} decomp_mean={r[10]:.2f} med={r[11]} stdev={r[12]}")
# rank by comp_mean
rows_with_comp=[r for (_,r,c,_) in rows if c!='' and c is not None]
rows_with_comp_sorted=sorted(rows_with_comp, key=lambda r: float(r[7]), reverse=True)
print('\nTop 10 by Compression mean throughput (MB/s):')
for r in rows_with_comp_sorted[:10]:
    print(f"comp={r[0]} mode={r[1]} strategy={r[2]} decomp_mode={r[3]} wg={r[4]} vlen={r[5]} samples={r[6]} comp_mean={r[7]:.2f} med={r[8]} stdev={r[9]}")
print('\nWrote summary CSV to',out)
