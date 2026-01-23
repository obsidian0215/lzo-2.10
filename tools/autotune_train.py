#!/usr/bin/env python3
"""Autotune trainer: create/iterate a small autotune config from sample files.

Writes a small key=value config (default: ~/.lzo_autotune.conf) with keys:
  global_alg=1x|1y
  global_level=11
  global_block_kb=8
  global_lsz=4
  samples_list=comma,separated,basenames

Usage:
  python3 tools/autotune_train.py --sample-dir /path/to/samples [--csv path/to/param_scan.csv] [--out ~/.lzo_autotune.conf] [--existing <file>] [--merge]

Behavior:
 - Matches sample basenames against 'sample' column in the CSV. If matches are found
   the trainer scores candidate configs only for those samples; otherwise falls back
   to computing global best from all CSV rows.
 - If --existing is provided and --merge is set, and the existing file contains
   a samples_list entry, the samples lists are merged and evaluation is run on the
   union.
"""
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--sample-dir', required=True, help='Directory containing sample files to build training set')
parser.add_argument('--csv', default='exp_results/param_scan/param_scan_20260114_004558.csv')
parser.add_argument('--out', default=None, help='Output config file path (default: <dir-of-lzo_gpu>/lzo_gpu.autotune.conf or ~/.lzo_autotune.conf)')
parser.add_argument('--existing', help='Path to existing autotune config to merge with (optional)')
parser.add_argument('--merge', action='store_true', help='Merge samples_list from existing config if present')
parser.add_argument('--iters', type=int, default=1, help='Number of refinement iterations (default 1)')
parser.add_argument('--w_comp', type=float, default=0.4)
parser.add_argument('--w_decomp', type=float, default=0.4)
parser.add_argument('--w_ratio', type=float, default=0.2)
args = parser.parse_args()

CSV = Path(args.csv)
if not CSV.exists():
    print('CSV not found:', CSV)
    sys.exit(1)

print('Loading CSV:', CSV)
df = pd.read_csv(CSV)
# normalize relevant fields
for c in ['mbps_kernel','ratio_pct','block_kb','lsz','level','alg','sample','mode']:
    if c in df.columns:
        df[c] = df[c]

# gather sample basenames from sample-dir
sample_dir = Path(args.sample_dir)
if not sample_dir.exists() or not sample_dir.is_dir():
    print('sample-dir not found or not a directory:', sample_dir)
    sys.exit(1)

samples = [p.name for p in sample_dir.iterdir() if p.is_file()]
if not samples:
    print('No files found in sample-dir:', sample_dir)
    sys.exit(1)

# if existing and merge, try to load samples_list
existing_samples = []
if args.existing and Path(args.existing).exists() and args.merge:
    try:
        with open(args.existing,'r',encoding='utf-8') as fh:
            for line in fh:
                line=line.strip()
                if line.startswith('samples_list='):
                    existing_samples = line.split('=',1)[1].split(',') if '=' in line else []
    except Exception as e:
        print('Warning: failed to read existing config:', e)

all_samples = set(samples) | set(existing_samples)
print(f'Training on {len(all_samples)} samples (merged={bool(existing_samples)})')

# find matching rows in CSV
matched = df[df['sample'].isin(all_samples)] if 'sample' in df.columns else pd.DataFrame()
if matched.empty:
    print('No matching sample names found in CSV; falling back to global evaluation across all rows')
    matched = df

# Build candidate configs set from matched rows
# Candidate key: (alg, level, block_kb, lsz)
cand_groups = {}
for (alg, level, block_kb, lsz), grp in matched.groupby(['alg','level','block_kb','lsz']):
    comp_rows = grp[grp['mode']=='compress'] if 'mode' in grp.columns else grp
    decomp_rows = grp[grp['mode']=='decompress'] if 'mode' in grp.columns else grp
    comp_med = float(comp_rows['mbps_kernel'].median()) if 'mbps_kernel' in comp_rows.columns and not comp_rows['mbps_kernel'].dropna().empty else np.nan
    decomp_med = float(decomp_rows['mbps_kernel'].median()) if 'mbps_kernel' in decomp_rows.columns and not decomp_rows['mbps_kernel'].dropna().empty else np.nan
    ratio_med = float(comp_rows['ratio_pct'].median()) if 'ratio_pct' in comp_rows.columns and not comp_rows['ratio_pct'].dropna().empty else np.nan
    key = (str(alg), int(level), int(block_kb), int(lsz))
    cand_groups[key] = {'alg':str(alg),'level':int(level),'block_kb':int(block_kb),'lsz':int(lsz),'comp_med':comp_med,'decomp_med':decomp_med,'ratio_med':ratio_med}

if not cand_groups:
    print('No configuration groups found in matched data; aborting')
    sys.exit(1)

cdf = pd.DataFrame.from_dict(cand_groups, orient='index')
# normalize
def normalize_series(s):
    s = s.dropna()
    if s.empty:
        return pd.Series([], dtype=float)
    mn = float(s.min()); mx = float(s.max())
    if mx == mn:
        return s.apply(lambda x: 1.0)
    return s.apply(lambda x: (x-mn)/(mx-mn))

comp_norm = normalize_series(cdf['comp_med'])
decomp_norm = normalize_series(cdf['decomp_med'])
ratio_norm = normalize_series(-cdf['ratio_med'])  # lower ratio (smaller compressed size) is better

comp_norm = comp_norm.reindex(cdf.index).fillna(0.0)
decomp_norm = decomp_norm.reindex(cdf.index).fillna(0.0)
ratio_norm = ratio_norm.reindex(cdf.index).fillna(0.0)

scores = args.w_comp * comp_norm + args.w_decomp * decomp_norm + args.w_ratio * ratio_norm
cdf['score'] = scores

best = cdf.sort_values('score', ascending=False).iloc[0]

# Write out config file
# determine default out path: try to write to <dir-of-lzo_gpu>/lzo_gpu.autotune.conf if possible
import shutil, os
if args.out:
    outp = Path(args.out).expanduser()
else:
    exe = shutil.which('lzo_gpu')
    if exe:
        outp = Path(os.path.join(os.path.dirname(exe), 'lzo_gpu.autotune.conf'))
    else:
        outp = Path.home() / '.lzo_autotune.conf'

# if existing not provided, check whether outp exists and use it as existing if present (iterative behavior)
existing_path = None
if args.existing:
    existing_path = Path(args.existing).expanduser()
elif outp.exists():
    existing_path = outp

# if existing_path found and merge requested (or default merge behavior when existing present), include its samples
existing_samples = []
if existing_path and existing_path.exists():
    try:
        with open(existing_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('samples_list='):
                    existing_samples = line.split('=',1)[1].strip().split(',') if '=' in line else []
    except Exception as e:
        print('Warning: failed to read existing config:', e)

if existing_samples:
    print(f'Found existing config at {existing_path}; merged samples: {len(existing_samples)}')
    all_samples = set(all_samples) | set(existing_samples)

# Iterative refinement loop: allow small neighbor expansion around best candidate
iters = max(1, int(args.iters))
prev_best = None
for it in range(iters):
    # Build candidate groups from current matched subset
    cand_groups = {}
    for (alg, level, block_kb, lsz), grp in matched.groupby(['alg','level','block_kb','lsz']):
        comp_rows = grp[grp['mode']=='compress'] if 'mode' in grp.columns else grp
        decomp_rows = grp[grp['mode']=='decompress'] if 'mode' in grp.columns else grp
        comp_med = float(comp_rows['mbps_kernel'].median()) if 'mbps_kernel' in comp_rows.columns and not comp_rows['mbps_kernel'].dropna().empty else np.nan
        decomp_med = float(decomp_rows['mbps_kernel'].median()) if 'mbps_kernel' in decomp_rows.columns and not decomp_rows['mbps_kernel'].dropna().empty else np.nan
        ratio_med = float(comp_rows['ratio_pct'].median()) if 'ratio_pct' in comp_rows.columns and not comp_rows['ratio_pct'].dropna().empty else np.nan
        key = (str(alg), int(level), int(block_kb), int(lsz))
        cand_groups[key] = {'alg':str(alg),'level':int(level),'block_kb':int(block_kb),'lsz':int(lsz),'comp_med':comp_med,'decomp_med':decomp_med,'ratio_med':ratio_med}

    if not cand_groups:
        print('No configuration groups found in matched data; aborting')
        sys.exit(1)

    cdf = pd.DataFrame.from_dict(cand_groups, orient='index')
    comp_norm = normalize_series(cdf['comp_med'])
    decomp_norm = normalize_series(cdf['decomp_med'])
    ratio_norm = normalize_series(-cdf['ratio_med'])  # lower ratio (smaller compressed size) is better
    comp_norm = comp_norm.reindex(cdf.index).fillna(0.0)
    decomp_norm = decomp_norm.reindex(cdf.index).fillna(0.0)
    ratio_norm = ratio_norm.reindex(cdf.index).fillna(0.0)
    scores = args.w_comp * comp_norm + args.w_decomp * decomp_norm + args.w_ratio * ratio_norm
    cdf['score'] = scores

    best = cdf.sort_values('score', ascending=False).iloc[0]

    # if iteration > 1, attempt neighbor expansion
    if it + 1 < iters:
        b_alg = best['alg']; b_level = int(best['level']); b_block = int(best['block_kb']); b_lsz = int(best['lsz'])
        candidates_added = 0
        # nearby levels
        neighbor_levels = set([b_level-1, b_level, b_level+1])
        # nearby block sizes (common ratio 2) and canonical set
        neighbor_blocks = set([b_block])
        neighbor_blocks.update([v for v in [b_block//2, b_block*2, 8,16,32,64] if v>0])
        # nearby lsz
        neighbor_lsz = set([b_lsz, max(1,b_lsz//2), b_lsz*2])
        # add new candidate rows from matched df if present
        for algv in set(matched['alg'].astype(str).unique()):
            for lvl in neighbor_levels:
                if lvl < 10 or lvl > 18: continue
                for bb in neighbor_blocks:
                    for lszv in neighbor_lsz:
                        key = (algv, int(lvl), int(bb), int(lszv))
                        if key not in cand_groups:
                            # check if such rows exist in matched
                            cond = (matched['alg'].astype(str)==algv) & (matched['level']==lvl) & (matched['block_kb']==bb) & (matched['lsz']==lszv)
                            sub = matched[cond]
                            if not sub.empty:
                                comp_med = float(sub[sub['mode']=='compress']['mbps_kernel'].median()) if 'mbps_kernel' in sub.columns and not sub[sub['mode']=='compress']['mbps_kernel'].dropna().empty else np.nan
                                decomp_med = float(sub[sub['mode']=='decompress']['mbps_kernel'].median()) if 'mbps_kernel' in sub.columns and not sub[sub['mode']=='decompress']['mbps_kernel'].dropna().empty else np.nan
                                ratio_med = float(sub[sub['mode']=='compress']['ratio_pct'].median()) if 'ratio_pct' in sub.columns and not sub[sub['mode']=='compress']['ratio_pct'].dropna().empty else np.nan
                                cand_groups[key] = {'alg':algv,'level':lvl,'block_kb':bb,'lsz':lszv,'comp_med':comp_med,'decomp_med':decomp_med,'ratio_med':ratio_med}
                                candidates_added += 1
        if candidates_added == 0:
            print(f'Iteration {it+1}: no neighbor candidates found; stopping early')
            break
        else:
            print(f'Iteration {it+1}: added {candidates_added} neighbor candidates; continuing')
            continue  # next iteration
    else:
        # final iteration, set best and exit loop
        break

# write out config file
lines = []
lines.append(f"# Autotune config generated: {datetime.datetime.utcnow().isoformat()}Z")
lines.append(f"samples_list={','.join(sorted(list(all_samples))) }")
lines.append(f"global_alg={best['alg']}")
lines.append(f"global_level={int(best['level'])}")
lines.append(f"global_block_kb={int(best['block_kb'])}")
lines.append(f"global_lsz={int(best['lsz'])}")
lines.append(f"score={best['score']:.6f}")

outp.parent.mkdir(parents=True, exist_ok=True)
with open(outp, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')

print('Wrote autotune config to', outp)
print('Best config: alg=%s level=%d block_kb=%d lsz=%d score=%.3f' % (best['alg'], int(best['level']), int(best['block_kb']), int(best['lsz']), best['score']))
print('\nTop 5 candidates:')
print(cdf.sort_values('score', ascending=False).head(5)[['alg','level','block_kb','lsz','comp_med','decomp_med','ratio_med','score']])

sys.exit(0)
