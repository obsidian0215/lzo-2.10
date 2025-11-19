#!/usr/bin/env python3
"""
Parse per-run profile logs under `lzo_gpu/experiments/logs/profiles/` and
produce per-combo stats (kernel_ms, total_ms, thrpt_MB) saved to
`lzo_gpu/experiments/logs/profiles/<combo>_perrun_profile.csv` and a
combined `summary_profiles.csv`.
"""
import os, re, csv, math, argparse
from statistics import mean, pstdev

parser = argparse.ArgumentParser(description='Parse per-run profile logs and produce per-combo stats')
parser.add_argument('--input-dir', '-i', default=None, help='Directory containing combo subdirectories with .log files. Defaults to experiments/logs/profiles')
parser.add_argument('--out', '-o', default=None, help='Combined summary CSV output path. Defaults to <input-dir>/summary_profiles.csv')
args = parser.parse_args()

ROOT = os.path.join(os.path.dirname(__file__), '..')
DEFAULT_PROFILES = os.path.join(ROOT, 'lzo_gpu', 'experiments', 'logs', 'profiles')
PROFILES_DIR = args.input_dir if args.input_dir else DEFAULT_PROFILES
OUT_SUM = args.out if args.out else os.path.join(PROFILES_DIR, 'summary_profiles.csv')

COMP_RE = re.compile(r"\[COMP\s*\] .*kernel=([0-9.]+) ms .*total=([0-9.]+) ms .*thrpt=([0-9.]+) MB/s")
DECOMP_RE = re.compile(r"\[DECOMP\] .*kernel=([0-9.]+) ms .*total=([0-9.]+) ms .*thrpt=([0-9.]+) MB/s")

def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    idx = min(int(n * q), n-1)
    return sorted_vals[idx]

summary_rows = []

for entry in sorted(os.listdir(PROFILES_DIR)):
    combo_path = os.path.join(PROFILES_DIR, entry)
    if not os.path.isdir(combo_path):
        continue
    # collect per-log metrics
    comp_kernel = []
    comp_total = []
    comp_thr = []
    decomp_kernel = []
    decomp_total = []
    decomp_thr = []
    for fname in sorted(os.listdir(combo_path)):
        if not fname.endswith('.log'):
            continue
        p = os.path.join(combo_path, fname)
        with open(p, 'r', errors='ignore') as f:
            data = f.read()
        for m in COMP_RE.finditer(data):
            k = float(m.group(1)); t = float(m.group(2)); thr = float(m.group(3))
            comp_kernel.append(k); comp_total.append(t); comp_thr.append(thr)
        for m in DECOMP_RE.finditer(data):
            k = float(m.group(1)); t = float(m.group(2)); thr = float(m.group(3))
            decomp_kernel.append(k); decomp_total.append(t); decomp_thr.append(thr)

    def make_stats(vals):
        if not vals:
            return {'n':0,'mean':None,'std':None,'p50':None,'p90':None,'p99':None}
        s = sorted(vals)
        n = len(s)
        mn = mean(s)
        sd = pstdev(s) if n>1 else 0.0
        return {'n':n,'mean':mn,'std':sd,'p50':percentile(s,0.5),'p90':percentile(s,0.9),'p99':percentile(s,0.99)}

    stats = {}
    stats['combo'] = entry
    ck = make_stats(comp_kernel)
    ct = make_stats(comp_total)
    ch = make_stats(comp_thr)
    dk = make_stats(decomp_kernel)
    dt = make_stats(decomp_total)
    dh = make_stats(decomp_thr)

    # write per-combo CSV
    out_csv = os.path.join(combo_path, f"{entry}_perrun_profile.csv")
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric','n','mean_ms','std_ms','p50_ms','p90_ms','p99_ms'])
        w.writerow(['comp_kernel', ck['n'], ck['mean'] if ck['mean'] is None else f"{ck['mean']:.6f}", ck['std'] if ck['std'] is None else f"{ck['std']:.6f}", ck['p50'] if ck['p50'] is None else f"{ck['p50']:.6f}", ck['p90'] if ck['p90'] is None else f"{ck['p90']:.6f}", ck['p99'] if ck['p99'] is None else f"{ck['p99']:.6f}"])
        w.writerow(['comp_total', ct['n'], ct['mean'] if ct['mean'] is None else f"{ct['mean']:.6f}", ct['std'] if ct['std'] is None else f"{ct['std']:.6f}", ct['p50'] if ct['p50'] is None else f"{ct['p50']:.6f}", ct['p90'] if ct['p90'] is None else f"{ct['p90']:.6f}", ct['p99'] if ct['p99'] is None else f"{ct['p99']:.6f}"])
        w.writerow(['comp_thr_MBps', ch['n'], ch['mean'] if ch['mean'] is None else f"{ch['mean']:.6f}", ch['std'] if ch['std'] is None else f"{ch['std']:.6f}", ch['p50'] if ch['p50'] is None else f"{ch['p50']:.6f}", ch['p90'] if ch['p90'] is None else f"{ch['p90']:.6f}", ch['p99'] if ch['p99'] is None else f"{ch['p99']:.6f}"])
        w.writerow(['decomp_kernel', dk['n'], dk['mean'] if dk['mean'] is None else f"{dk['mean']:.6f}", dk['std'] if dk['std'] is None else f"{dk['std']:.6f}", dk['p50'] if dk['p50'] is None else f"{dk['p50']:.6f}", dk['p90'] if dk['p90'] is None else f"{dk['p90']:.6f}", dk['p99'] if dk['p99'] is None else f"{dk['p99']:.6f}"])
        w.writerow(['decomp_total', dt['n'], dt['mean'] if dt['mean'] is None else f"{dt['mean']:.6f}", dt['std'] if dt['std'] is None else f"{dt['std']:.6f}", dt['p50'] if dt['p50'] is None else f"{dt['p50']:.6f}", dt['p90'] if dt['p90'] is None else f"{dt['p90']:.6f}", dt['p99'] if dt['p99'] is None else f"{dt['p99']:.6f}"])
        w.writerow(['decomp_thr_MBps', dh['n'], dh['mean'] if dh['mean'] is None else f"{dh['mean']:.6f}", dh['std'] if dh['std'] is None else f"{dh['std']:.6f}", dh['p50'] if dh['p50'] is None else f"{dh['p50']:.6f}", dh['p90'] if dh['p90'] is None else f"{dh['p90']:.6f}", dh['p99'] if dh['p99'] is None else f"{dh['p99']:.6f}"])

    summary_rows.append({
        'combo': entry,
        'comp_n': ck['n'], 'comp_mean_ms': ck['mean'], 'comp_std_ms': ck['std'],
        'decomp_n': dk['n'], 'decomp_mean_ms': dk['mean'], 'decomp_std_ms': dk['std'],
        'decomp_mean_MBps': dh['mean']
    })

    print(f"Wrote {out_csv}")

# write combined summary
with open(OUT_SUM, 'w', newline='') as f:
    fieldnames = ['combo','comp_n','comp_mean_ms','comp_std_ms','decomp_n','decomp_mean_ms','decomp_std_ms','decomp_mean_MBps']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

print('Wrote combined summary ->', OUT_SUM)
