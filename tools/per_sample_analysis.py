#!/usr/bin/env python3
"""
Per-sample comparison for param_scan results.
Generates CSVs that compare CPU/GPU/Daemon configs per sample per mode (compress/decompress),
with CPU thread summaries and GPU/Daemon top lists.
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict


def parse_csv(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # normalize numeric fields
            for c in ['size_mb','total_tp','kernel_tp','ratio','total_time_ms','kernel_time_ms','read_time_ms','upload_time_ms','write_time_ms','download_time_ms']:
                try:
                    row[c] = float(row.get(c, '') or 0)
                except Exception:
                    row[c] = 0.0
            rows.append(row)
    return rows


def build_groups(rows):
    groups = defaultdict(lambda: defaultdict(list))  # groups[sample][mode] -> rows
    for r in rows:
        groups[r['sample']][r['mode']].append(r)
    return groups


def compute_best_per_tool(groups):
    best = defaultdict(lambda: defaultdict(dict))  # best[sample][mode][tool] = row
    for sample, mmap in groups.items():
        for mode, rows in mmap.items():
            tools = defaultdict(list)
            for r in rows:
                tools[r['tool']].append(r)
            for t, rl in tools.items():
                best[sample][mode][t] = max(rl, key=lambda x: x['total_tp']) if rl else None
    return best


def parse_config(config):
    """Return a dict of parsed fields from config name."""
    out = {'threads': '', 'vec': '', 'mt_io': '', 'copy_mode': '', 'block_kb': ''}
    if not config:
        return out
    # parse CPU thread count: cpu_t8_1k, cpu_t4_1
    if config.startswith('cpu_'):
        # find 't<num>' pattern
        import re
        m = re.search(r'_t(\d+)_', config)
        if not m:
            # alternative pattern: cpu_t8_1k (no underscore after t8?) but our configs have t8_1k; if not found, search for 'tX'
            m = re.search(r'_t(\d+)', config)
        if m:
            out['threads'] = m.group(1)
        return out
    # parse gpu / daemon config
    # examples: gpu_1k_vec_32k_zerocopy_mt, daemon_1k_scalar_128k
    if config.startswith('gpu_') or config.startswith('daemon_'):
        parts = config.split('_')
        # block_kb pattern as '32k' or '128k'
        for p in parts:
            if p.endswith('k') and p[:-1].isdigit():
                out['block_kb'] = int(p[:-1])
            if p in ('vec', 'vector'):
                out['vec'] = 1
            if p == 'scalar':
                out['vec'] = 0
            if p == 'mt':
                out['mt_io'] = 1
            if p == 'single':
                out['mt_io'] = 0
            if p == 'stdcopy':
                out['copy_mode'] = 1
            if p == 'zerocopy':
                out['copy_mode'] = 0
    return out


def bucket_size_label(size_mb):
    if size_mb < 1.0:
        return 'tiny_<1MB'
    if size_mb < 10.0:
        return 'small_1-10MB'
    if size_mb < 100.0:
        return 'medium_10-100MB'
    if size_mb < 500.0:
        return 'large_100-500MB'
    return 'huge_>500MB'


def write_per_sample_comparison(groups, best_map, out_csv):
    headers = [
        'sample','size_mb','mode','tool','config','threads','vec','mt_io','copy_mode','block_kb',
        'total_tp','kernel_tp','ratio',
        'cpu_best_tp','cpu_best_config','gpu_best_tp','gpu_best_config','daemon_best_tp','daemon_best_config',
        'pct_vs_cpu_best','pct_vs_gpu_best','pct_vs_daemon_best','pct_vs_overall_best'
    ]
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for sample, mmap in groups.items():
            for mode, rows in mmap.items():
                cpu_best = best_map.get(sample, {}).get(mode, {}).get('CPU')
                gpu_best = best_map.get(sample, {}).get(mode, {}).get('GPU')
                daemon_best = best_map.get(sample, {}).get(mode, {}).get('Daemon')
                overall_best_tp = max([x['total_tp'] for x in [cpu_best,gpu_best,daemon_best] if x is not None] + [0])
                for r in rows:
                    # parse fields from config string (threads, vec, block_kb, copy_mode, mt_io)
                    parsed = parse_config(r.get('config','') or '')
                    cpu_best_tp = cpu_best['total_tp'] if cpu_best else 0
                    gpu_best_tp = gpu_best['total_tp'] if gpu_best else 0
                    daemon_best_tp = daemon_best['total_tp'] if daemon_best else 0
                    rt = r['total_tp'] or 0
                    pct_cpu = (rt/cpu_best_tp*100.0) if cpu_best_tp else ''
                    pct_gpu = (rt/gpu_best_tp*100.0) if gpu_best_tp else ''
                    pct_daemon = (rt/daemon_best_tp*100.0) if daemon_best_tp else ''
                    pct_overall = (rt/overall_best_tp*100.0) if overall_best_tp else ''
                    w.writerow([
                        r['sample'], r['size_mb'], mode, r['tool'], r['config'], parsed.get('threads',''), parsed.get('vec',''), parsed.get('mt_io',''), parsed.get('copy_mode',''), parsed.get('block_kb',''),
                        r['total_tp'], r['kernel_tp'], r['ratio'],
                        cpu_best_tp, cpu_best['config'] if cpu_best else '', gpu_best_tp, gpu_best['config'] if gpu_best else '', daemon_best_tp, daemon_best['config'] if daemon_best else '',
                        pct_cpu, pct_gpu, pct_daemon, pct_overall
                    ])


def write_cpu_thread_summary(groups, best_map, out_csv):
    headers = ['sample','size_mb','mode','threads','best_tp','best_config','ratio','pct_of_cpu_best']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for sample, mmap in groups.items():
            for mode, rows in mmap.items():
                # select CPU rows
                cpu_rows = [r for r in rows if r['tool']=='CPU']
                if not cpu_rows:
                    continue
                # group by threads
                thr_map = defaultdict(list)
                for r in cpu_rows:
                    parsed = parse_config(r.get('config','') or '')
                    threads = parsed.get('threads','')
                    thr_map[threads].append(r)
                cpu_best_tp = best_map.get(sample, {}).get(mode, {}).get('CPU', {}).get('total_tp', 0)
                for thr, rl in sorted(thr_map.items(), key=lambda x: (int(x[0]) if x[0].isdigit() else 0)):
                    best = max(rl, key=lambda x: x['total_tp'])
                    pct = (best['total_tp']/cpu_best_tp*100.0) if cpu_best_tp else ''
                    w.writerow([sample, best['size_mb'], mode, thr, best['total_tp'], best['config'], best['ratio'], pct])


def write_gpu_daemon_top(groups, out_csv, n=3):
    headers = ['sample','size_mb','mode']
    for i in range(1, n+1):
        headers += [f'gpu_top{i}_config', f'gpu_top{i}_tp', f'gpu_top{i}_ratio']
    for i in range(1, n+1):
        headers += [f'daemon_top{i}_config', f'daemon_top{i}_tp', f'daemon_top{i}_ratio']

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for sample, mmap in groups.items():
            for mode, rows in mmap.items():
                gpu_rows = [r for r in rows if r['tool']=='GPU']
                daemon_rows = [r for r in rows if r['tool']=='Daemon']
                gpu_rows.sort(key=lambda x: x['total_tp'], reverse=True)
                daemon_rows.sort(key=lambda x: x['total_tp'], reverse=True)
                out = [sample, rows[0]['size_mb'], mode]
                for i in range(n):
                    if i < len(gpu_rows):
                        g = gpu_rows[i]
                        out += [g['config'], g['total_tp'], g['ratio']]
                    else:
                        out += ['','', '']
                for i in range(n):
                    if i < len(daemon_rows):
                        d = daemon_rows[i]
                        out += [d['config'], d['total_tp'], d['ratio']]
                    else:
                        out += ['','', '']
                w.writerow(out)


def write_summary_text(groups, best_map, out_txt):
    total_samples = 0
    best_wins = {'compress':{'CPU':0,'GPU':0,'Daemon':0}, 'decompress':{'CPU':0,'GPU':0,'Daemon':0}}
    with open(out_txt, 'w') as f:
        f.write('Per-sample CPU/GPU/Daemon comparison\n')
        for sample, mmap in groups.items():
            total_samples += 1
            f.write('\n=== Sample: %s ===\n' % sample)
            for mode, rows in mmap.items():
                cpu_best = best_map.get(sample, {}).get(mode, {}).get('CPU')
                gpu_best = best_map.get(sample, {}).get(mode, {}).get('GPU')
                daemon_best = best_map.get(sample, {}).get(mode, {}).get('Daemon')
                # determine winner for this sample/mode
                winners = []
                best_val = 0
                for t in [('CPU', cpu_best), ('GPU', gpu_best), ('Daemon', daemon_best)]:
                    if t[1] and t[1]['total_tp'] > best_val:
                        best_val = t[1]['total_tp']
                        winners = [t[0]]
                    elif t[1] and t[1]['total_tp'] == best_val:
                        winners.append(t[0])
                for w_ in winners:
                    best_wins[mode][w_] += 1
                f.write('\n  %s:\n' % mode.upper())
                if cpu_best:
                    f.write('    CPU best: %s -> %.2f MB/s (ratio %.3f, threads=%s)\n' % (cpu_best['config'], cpu_best['total_tp'], cpu_best['ratio'], cpu_best.get('threads','')))
                else:
                    f.write('    CPU best: -\n')
                if gpu_best:
                    f.write('    GPU best: %s -> %.2f MB/s (ratio %.3f)\n' % (gpu_best['config'], gpu_best['total_tp'], gpu_best['ratio']))
                else:
                    f.write('    GPU best: -\n')
                if daemon_best:
                    f.write('    Daemon best: %s -> %.2f MB/s (ratio %.3f)\n' % (daemon_best['config'], daemon_best['total_tp'], daemon_best['ratio']))
                else:
                    f.write('    Daemon best: -\n')
                # percent differences
                if cpu_best and gpu_best:
                    if cpu_best['total_tp']>0:
                        f.write('    CPU vs GPU: CPU is %.1f%% %s than GPU\n' % ((cpu_best['total_tp']/gpu_best['total_tp']-1.0)*100.0, 'faster' if cpu_best['total_tp']>gpu_best['total_tp'] else 'slower'))
                if cpu_best and daemon_best:
                    if cpu_best['total_tp']>0 and daemon_best['total_tp']>0:
                        f.write('    CPU vs Daemon: CPU is %.1f%% %s than Daemon\n' % ((cpu_best['total_tp']/daemon_best['total_tp']-1.0)*100.0, 'faster' if cpu_best['total_tp']>daemon_best['total_tp'] else 'slower'))

        # global counts
        f.write('\n=== Global wins summary ===\n')
        for mode in ('compress','decompress'):
            f.write('  %s: CPU wins=%d, GPU wins=%d, Daemon wins=%d\n' % (mode, best_wins[mode]['CPU'], best_wins[mode]['GPU'], best_wins[mode]['Daemon']))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-agg', help='path to sample config agg CSV', default='param_scan_sample_config_agg.csv')
    parser.add_argument('--outdir', help='output dir', default='/root/lzo-2.10/exp_results/param_scan_analysis')
    args = parser.parse_args()

    sample_csv = args.sample_agg
    outdir = args.outdir
    if not os.path.exists(sample_csv):
        sample_csv = os.path.join(outdir, sample_csv)
    if not os.path.exists(sample_csv):
        print('Sample agg CSV not found:', sample_csv)
        return 1

    rows = parse_csv(sample_csv)
    groups = build_groups(rows)
    best_map = compute_best_per_tool(groups)

    # write files
    per_sample_out = os.path.join(outdir, 'param_scan_per_sample_comparison.csv')
    write_per_sample_comparison(groups, best_map, per_sample_out)

    cpu_threads_out = os.path.join(outdir, 'param_scan_cpu_thread_summary.csv')
    write_cpu_thread_summary(groups, best_map, cpu_threads_out)

    gpu_daemon_out = os.path.join(outdir, 'param_scan_gpu_daemon_top.csv')
    write_gpu_daemon_top(groups, gpu_daemon_out, n=3)

    text_out = os.path.join(outdir, 'param_scan_per_sample_summary.txt')
    write_summary_text(groups, best_map, text_out)

    # write thread-level comparisons: how GPU and Daemon compare to CPU per-thread performance
    thread_vs_csv, thread_summary_csv, thread_better_csv, bucket_summary_csv, bucket_gpu_block_csv, bucket_daemon_block_csv = compare_gpu_daemon_to_cpu_threads(groups, best_map, outdir)
    print('Wrote thread-level comparison:', thread_vs_csv)
    print('Wrote thread-level summary:', thread_summary_csv)
    print('Wrote thread-level better samples:', thread_better_csv)
    print('Wrote bucket-level summary:', bucket_summary_csv)
    print('Wrote bucket GPU block distribution:', bucket_gpu_block_csv)
    print('Wrote bucket Daemon block distribution:', bucket_daemon_block_csv)

    print('Wrote:', per_sample_out)
    print('Wrote:', cpu_threads_out)
    print('Wrote:', gpu_daemon_out)
    print('Wrote:', text_out)
    return 0


def compute_best_cpu_by_thread(groups):
    # returns dict: (sample,mode) -> { thread: row }
    best_cpu_threads = {}
    for sample, mmap in groups.items():
        for mode, rows in mmap.items():
            key = (sample, mode)
            best_cpu_threads[key] = {}
            for r in rows:
                if r['tool'] != 'CPU':
                    continue
                parsed = parse_config(r.get('config','') or '')
                thr = parsed.get('threads','')
                if thr == '':
                    continue
                # store the best per thread count
                curr = best_cpu_threads[key].get(thr)
                if curr is None or float(r['total_tp']) > float(curr['total_tp']):
                    best_cpu_threads[key][thr] = r
    return best_cpu_threads


def compare_gpu_daemon_to_cpu_threads(groups, best_map, outdir):
    # For each sample & mode & thread, compare GPU/Daemon best vs CPU thread best
    csv_rows = []
    summary = {}
    bucket_summary = {}  # key: (bucket, mode, thr)
    # Tower of counters for block size distributions per bucket/mode/thread
    bucket_gpu_block_dist = defaultdict(lambda: defaultdict(int))
    bucket_daemon_block_dist = defaultdict(lambda: defaultdict(int))
    better_samples = []
    best_cpu_threads = compute_best_cpu_by_thread(groups)
    for sample, mmap in groups.items():
        for mode, rows in mmap.items():
            key = (sample, mode)
            cpu_thr_map = best_cpu_threads.get(key, {})
            gpu_best = best_map.get(sample, {}).get(mode, {}).get('GPU')
            daemon_best = best_map.get(sample, {}).get(mode, {}).get('Daemon')
            for thr, cpu_row in cpu_thr_map.items():
                cpu_tp = float(cpu_row.get('total_tp') or 0.0)
                cpu_cfg = cpu_row.get('config')
                gpu_tp = float(gpu_best.get('total_tp') or 0.0) if gpu_best else 0.0
                gpu_cfg = gpu_best.get('config') if gpu_best else ''
                daemon_tp = float(daemon_best.get('total_tp') or 0.0) if daemon_best else 0.0
                daemon_cfg = daemon_best.get('config') if daemon_best else ''
                gpu_ge = gpu_tp >= cpu_tp if cpu_tp > 0 else False
                daemon_ge = daemon_tp >= cpu_tp if cpu_tp > 0 else False
                gpu_pct = (gpu_tp/cpu_tp-1.0)*100.0 if cpu_tp>0 else ''
                daemon_pct = (daemon_tp/cpu_tp-1.0)*100.0 if cpu_tp>0 else ''
                csv_rows.append({
                    'sample': sample,
                    'size_mb': rows[0].get('size_mb',0),
                    'mode': mode,
                    'thread': thr,
                    'cpu_tp': cpu_tp,
                    'cpu_cfg': cpu_cfg,
                    'gpu_tp': gpu_tp,
                    'gpu_cfg': gpu_cfg,
                    'gpu_ge_cpu': gpu_ge,
                    'gpu_pct_vs_cpu': gpu_pct,
                    'daemon_tp': daemon_tp,
                    'daemon_cfg': daemon_cfg,
                    'daemon_ge_cpu': daemon_ge,
                    'daemon_pct_vs_cpu': daemon_pct,
                })
                # record better samples
                if gpu_ge or daemon_ge:
                    better_samples.append({
                        'sample': sample,
                        'size_mb': rows[0].get('size_mb',0),
                        'mode': mode,
                        'thread': thr,
                        'cpu_tp': cpu_tp,
                        'cpu_cfg': cpu_cfg,
                        'gpu_tp': gpu_tp,
                        'gpu_cfg': gpu_cfg,
                        'gpu_ge_cpu': gpu_ge,
                        'daemon_tp': daemon_tp,
                        'daemon_cfg': daemon_cfg,
                        'daemon_ge_cpu': daemon_ge
                    })
                # bucket analytics
                size = rows[0].get('size_mb', 0)
                bucket = bucket_size_label(size)
                bs_key = (bucket, mode, thr)
                if bs_key not in bucket_summary:
                    bucket_summary[bs_key] = {'count':0, 'gpu_ge':0, 'daemon_ge':0, 'gpu_pcts':[], 'daemon_pcts':[], 'gpu_uploads':[], 'gpu_downloads':[], 'gpu_reads':[], 'gpu_writes':[], 'daemon_uploads':[], 'daemon_downloads':[], 'daemon_reads':[], 'daemon_writes':[]}
                bucket_summary[bs_key]['count'] += 1
                if gpu_ge:
                    bucket_summary[bs_key]['gpu_ge'] += 1
                    # collect IO stats
                    if gpu_best:
                        bucket_summary[bs_key]['gpu_uploads'].append(float(gpu_best.get('upload_time_ms') or 0))
                        bucket_summary[bs_key]['gpu_downloads'].append(float(gpu_best.get('download_time_ms') or 0))
                        bucket_summary[bs_key]['gpu_reads'].append(float(gpu_best.get('read_time_ms') or 0))
                        bucket_summary[bs_key]['gpu_writes'].append(float(gpu_best.get('write_time_ms') or 0))
                        # block distribution
                        parsed_gpu = parse_config(gpu_best.get('config','') or '')
                        block_k = parsed_gpu.get('block_kb','')
                        if block_k:
                            bucket_gpu_block_dist[bs_key][block_k] += 1
                if daemon_ge:
                    bucket_summary[bs_key]['daemon_ge'] += 1
                    if daemon_best:
                        bucket_summary[bs_key]['daemon_uploads'].append(float(daemon_best.get('upload_time_ms') or 0))
                        bucket_summary[bs_key]['daemon_downloads'].append(float(daemon_best.get('download_time_ms') or 0))
                        bucket_summary[bs_key]['daemon_reads'].append(float(daemon_best.get('read_time_ms') or 0))
                        bucket_summary[bs_key]['daemon_writes'].append(float(daemon_best.get('write_time_ms') or 0))
                        parsed_daemon = parse_config(daemon_best.get('config','') or '')
                        block_k = parsed_daemon.get('block_kb','')
                        if block_k:
                            bucket_daemon_block_dist[bs_key][block_k] += 1
                # aggregate for summary
                summ_key = (mode, thr)
                if summ_key not in summary:
                    summary[summ_key] = {'count':0, 'gpu_ge':0, 'daemon_ge':0, 'gpu_pcts':[], 'daemon_pcts':[]}
                summary[summ_key]['count'] += 1
                if gpu_ge:
                    summary[summ_key]['gpu_ge'] += 1
                if daemon_ge:
                    summary[summ_key]['daemon_ge'] += 1
                if gpu_pct != '':
                    summary[summ_key]['gpu_pcts'].append(gpu_pct)
                if daemon_pct != '':
                    summary[summ_key]['daemon_pcts'].append(daemon_pct)

    # write per-sample vs cpu thread csv
    out_csv = os.path.join(outdir, 'param_scan_thread_level_vs_gpu_daemon.csv')
    headers = ['sample','size_mb','mode','thread','cpu_tp','cpu_cfg','gpu_tp','gpu_cfg','gpu_ge_cpu','gpu_pct_vs_cpu','daemon_tp','daemon_cfg','daemon_ge_cpu','daemon_pct_vs_cpu']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in csv_rows:
            w.writerow([r['sample'], r['size_mb'], r['mode'], r['thread'], r['cpu_tp'], r['cpu_cfg'], r['gpu_tp'], r['gpu_cfg'], r['gpu_ge_cpu'], r['gpu_pct_vs_cpu'], r['daemon_tp'], r['daemon_cfg'], r['daemon_ge_cpu'], r['daemon_pct_vs_cpu']])

    # write summary csv
    out_summary = os.path.join(outdir, 'param_scan_thread_level_summary.csv')
    headers = ['mode','thread','samples_with_thread','gpu_ge_count','gpu_ge_pct','daemon_ge_count','daemon_ge_pct','mean_gpu_pct_vs_cpu','median_gpu_pct_vs_cpu','mean_daemon_pct_vs_cpu','median_daemon_pct_vs_cpu']
    with open(out_summary, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for (mode, thr), v in sorted(summary.items()):
            cnt = v['count']
            gpu_ge = v['gpu_ge']
            daemon_ge = v['daemon_ge']
            gpu_pct_mean = statistics.mean(v['gpu_pcts']) if v['gpu_pcts'] else ''
            gpu_pct_med = statistics.median(v['gpu_pcts']) if v['gpu_pcts'] else ''
            daemon_pct_mean = statistics.mean(v['daemon_pcts']) if v['daemon_pcts'] else ''
            daemon_pct_med = statistics.median(v['daemon_pcts']) if v['daemon_pcts'] else ''
            w.writerow([mode, thr, cnt, gpu_ge, (gpu_ge/cnt*100.0 if cnt else 0.0), daemon_ge, (daemon_ge/cnt*100.0 if cnt else 0.0), gpu_pct_mean, gpu_pct_med, daemon_pct_mean, daemon_pct_med])

    # write list of samples where gpu/daemon >= cpu thread
    out_better = os.path.join(outdir, 'param_scan_thread_level_better_samples.csv')
    headers = ['sample','size_mb','mode','thread','cpu_tp','cpu_cfg','gpu_tp','gpu_cfg','gpu_ge_cpu','daemon_tp','daemon_cfg','daemon_ge_cpu']
    with open(out_better, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in better_samples:
            w.writerow([r['sample'], r['size_mb'], r['mode'], r['thread'], r['cpu_tp'], r['cpu_cfg'], r['gpu_tp'], r['gpu_cfg'], r['gpu_ge_cpu'], r['daemon_tp'], r['daemon_cfg'], r['daemon_ge_cpu']])

    # write bucket-level summary and distributions
    bucket_summary_csv, bucket_gpu_block_csv, bucket_daemon_block_csv = write_bucket_thread_summary_csv(bucket_summary, bucket_gpu_block_dist, bucket_daemon_block_dist, outdir)
    return out_csv, out_summary, out_better, bucket_summary_csv, bucket_gpu_block_csv, bucket_daemon_block_csv


def write_bucket_thread_summary_csv(bucket_summary, bucket_gpu_block_dist, bucket_daemon_block_dist, outdir):
    out_summary = os.path.join(outdir, 'param_scan_bucket_thread_summary.csv')
    headers = ['bucket','mode','thread','samples_with_thread','gpu_ge_count','gpu_ge_pct','daemon_ge_count','daemon_ge_pct','median_gpu_pct_vs_cpu','median_daemon_pct_vs_cpu','mean_gpu_upload_ms','mean_gpu_download_ms','mean_gpu_read_ms','mean_gpu_write_ms','mean_daemon_upload_ms','mean_daemon_download_ms','mean_daemon_read_ms','mean_daemon_write_ms']
    with open(out_summary, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for (bucket, mode, thr), v in sorted(bucket_summary.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))):
            cnt = v['count']
            gpu_ge = v['gpu_ge']
            daemon_ge = v['daemon_ge']
            median_gpu_pct = statistics.median(v['gpu_pcts']) if v['gpu_pcts'] else ''
            median_daemon_pct = statistics.median(v['daemon_pcts']) if v['daemon_pcts'] else ''
            mean_gpu_up = statistics.mean(v['gpu_uploads']) if v['gpu_uploads'] else ''
            mean_gpu_down = statistics.mean(v['gpu_downloads']) if v['gpu_downloads'] else ''
            mean_gpu_reads = statistics.mean(v['gpu_reads']) if v['gpu_reads'] else ''
            mean_gpu_writes = statistics.mean(v['gpu_writes']) if v['gpu_writes'] else ''
            mean_daemon_up = statistics.mean(v['daemon_uploads']) if v['daemon_uploads'] else ''
            mean_daemon_down = statistics.mean(v['daemon_downloads']) if v['daemon_downloads'] else ''
            mean_daemon_reads = statistics.mean(v['daemon_reads']) if v['daemon_reads'] else ''
            mean_daemon_writes = statistics.mean(v['daemon_writes']) if v['daemon_writes'] else ''
            w.writerow([bucket, mode, thr, cnt, gpu_ge, (gpu_ge/cnt*100.0 if cnt else 0.0), daemon_ge, (daemon_ge/cnt*100.0 if cnt else 0.0), median_gpu_pct, median_daemon_pct, mean_gpu_up, mean_gpu_down, mean_gpu_reads, mean_gpu_writes, mean_daemon_up, mean_daemon_down, mean_daemon_reads, mean_daemon_writes])

    # write detailed block distributions for GPU and Daemon
    out_gpu_block = os.path.join(outdir, 'param_scan_bucket_gpu_block_distribution.csv')
    with open(out_gpu_block, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','block_kb','count'])
        for (bucket, mode, thr), d in bucket_gpu_block_dist.items():
            for block_k, cnt in sorted(d.items()):
                w.writerow([bucket, mode, thr, block_k, cnt])

    out_daemon_block = os.path.join(outdir, 'param_scan_bucket_daemon_block_distribution.csv')
    with open(out_daemon_block, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','block_kb','count'])
        for (bucket, mode, thr), d in bucket_daemon_block_dist.items():
            for block_k, cnt in sorted(d.items()):
                w.writerow([bucket, mode, thr, block_k, cnt])

    return out_summary, out_gpu_block, out_daemon_block

if __name__ == '__main__':
    exit(main())
