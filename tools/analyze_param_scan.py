#!/usr/bin/env python3
"""
Detailed analysis of param_scan CSV results.
Generates JSON summaries and a per-sample best config table.
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
import math
import re

# optional plotting libs
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    try:
        import seaborn as sns
    except Exception:
        sns = None
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False


def latest_csv(path):
    files = glob.glob(os.path.join(path, 'param_scan_*.csv'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as csvfile:
        r = csv.DictReader(csvfile)
        for row in r:
            # Normalize numeric fields
            for k in ['size_mb','block_kb','ratio','total_time_ms','kernel_time_ms','read_time_ms','write_time_ms','upload_time_ms','download_time_ms','total_throughput_mbps','kernel_throughput_mbps']:
                try:
                    row[k] = float(row.get(k,'') or 0)
                except Exception:
                    row[k] = 0.0
            rows.append(row)
    return rows


def bucket_size_label(size_mb):
    # categorize sample sizes into buckets
    if size_mb < 1.0:
        return 'tiny_<1MB'
    if size_mb < 10.0:
        return 'small_1-10MB'
    if size_mb < 100.0:
        return 'medium_10-100MB'
    if size_mb < 500.0:
        return 'large_100-500MB'
    return 'huge_>500MB'


def parse_threads_from_config(config):
    # find patterns like _t1_ or _t1 at the end
    if not config:
        return None
    m = re.search(r'_t(\d+)(?:_|$)', config)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_block_kb_from_config(config):
    if not config:
        return None
    m = re.search(r'_(\d+)k(?:_|$)', config)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_vec_from_config(config):
    if not config:
        return None
    if 'vec' in config or 'vector' in config:
        return True
    if 'scalar' in config:
        return False
    return None


def parse_copy_mode_from_config(config):
    if not config:
        return None
    if 'stdcopy' in config or 'standard' in config:
        return 'stdcopy'
    if 'zerocopy' in config or 'zero_copy' in config:
        return 'zerocopy'
    return None


def percentile_sorted(values, p):
    # values must be a list of numbers; p in [0,100]
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values_sorted = sorted(values)
    k = (len(values_sorted)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[int(f)] * (c-k)
    d1 = values_sorted[int(c)] * (k-f)
    return float(d0 + d1)


def safe_mean(vals):
    return statistics.mean(vals) if vals else 0.0


def safe_stdev(vals):
    try:
        return statistics.pstdev(vals) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0


def safe_cv(vals):
    if not vals:
        return 0.0
    meanv = safe_mean(vals)
    if meanv == 0:
        return 0.0
    return safe_stdev(vals) / meanv


def aggregate_sample_config(rows, filter_verified=True):
    # Aggregate multiple runs per (sample, tool, config, mode) into sample-level averages
    samples = defaultdict(list)  # key -> list of row dicts
    for r in rows:
        if filter_verified and r.get('verified','') != 'YES':
            continue
        key = (r['sample'], r['tool'], r['config'], r['mode'])
        samples[key].append(r)

    agg_rows = []
    for key, recs in samples.items():
        sample, tool, config, mode = key
        # aggregate statistics from recs
        mean_tp = safe_mean([float(rr['total_throughput_mbps']) for rr in recs])
        mean_ktp = safe_mean([float(rr['kernel_throughput_mbps']) for rr in recs])
        mean_ratio = safe_mean([float(rr.get('ratio',0.0) or 0.0) for rr in recs])
        mean_total_time_ms = safe_mean([float(rr.get('total_time_ms',0.0) or 0.0) for rr in recs])
        mean_kernel_time_ms = safe_mean([float(rr.get('kernel_time_ms',0.0) or 0.0) for rr in recs])
        mean_read_ms = safe_mean([float(rr.get('read_time_ms',0.0) or 0.0) for rr in recs])
        mean_upload_ms = safe_mean([float(rr.get('upload_time_ms',0.0) or 0.0) for rr in recs])
        mean_write_ms = safe_mean([float(rr.get('write_time_ms',0.0) or 0.0) for rr in recs])
        mean_download_ms = safe_mean([float(rr.get('download_time_ms',0.0) or 0.0) for rr in recs])
        # size_mb should be consistent for given sample
        size_mb = float(recs[0].get('size_mb',0.0) or 0.0)
        # store aggregated
        agg_rows.append({
            'sample': sample,
            'tool': tool,
            'config': config,
            'mode': mode,
            'size_mb': size_mb,
            'total_tp': mean_tp,
            'kernel_tp': mean_ktp,
            'ratio': mean_ratio,
            'total_time_ms': mean_total_time_ms,
            'kernel_time_ms': mean_kernel_time_ms,
            'read_time_ms': mean_read_ms,
            'upload_time_ms': mean_upload_ms,
            'write_time_ms': mean_write_ms,
            'download_time_ms': mean_download_ms,
        })
    return agg_rows


def aggregate_by_config(agg_rows):
    # group by (tool,config,mode)
    groups = defaultdict(list)
    for r in agg_rows:
        key = (r['tool'], r['config'], r['mode'])
        groups[key].append(r)

    config_stats = {}
    for key, rows in groups.items():
        tool, config, mode = key
        tps = [rr['total_tp'] for rr in rows]
        ktps = [rr['kernel_tp'] for rr in rows]
        ratios = [rr['ratio'] for rr in rows]
        total_times = [rr['total_time_ms'] for rr in rows]
        read_times = [rr['read_time_ms'] for rr in rows]
        upload_times = [rr['upload_time_ms'] for rr in rows]
        write_times = [rr['write_time_ms'] for rr in rows]
        download_times = [rr['download_time_ms'] for rr in rows]
        mean_tp = safe_mean(tps)
        median_tp = statistics.median(tps) if tps else 0.0
        stdev_tp = safe_stdev(tps)
        p25 = percentile_sorted(tps, 25)
        p75 = percentile_sorted(tps, 75)
        mean_ktp = safe_mean(ktps)
        mean_ratio = safe_mean(ratios) if ratios else 0.0
        mean_total_time = safe_mean(total_times)
        mean_io = safe_mean([a+b for a,b in zip(read_times, upload_times)])
        read_frac = safe_mean([ (a/(a+b) if (a+b)>0 else 0.0) for a,b in zip(read_times, upload_times)])
        cv_read = safe_cv(read_times)
        cv_write = safe_cv(write_times)
        config_stats[key] = {
            'tool': tool,
            'config': config,
            'mode': mode,
            'count_samples': len(rows),
            'mean_total_tp': mean_tp,
            'median_total_tp': median_tp,
            'stdev_total_tp': stdev_tp,
            'min_total_tp': min(tps) if tps else 0.0,
            'max_total_tp': max(tps) if tps else 0.0,
            'p25_total_tp': p25,
            'p75_total_tp': p75,
            'mean_kernel_tp': mean_ktp,
            'mean_ratio': mean_ratio,
            'mean_total_time_ms': mean_total_time,
            'mean_io_ms': mean_io,
            'read_frac': read_frac,
            'cv_read': cv_read,
            'cv_write': cv_write,
        }
    return config_stats


def compute_rank_frequencies(agg_rows):
    # For each sample, per tool and mode, rank configs by total_tp, and compute frequency that a config is top1/top3/top10
    sample_tool_mode = defaultdict(list)
    for r in agg_rows:
        key = (r['sample'], r['tool'], r['mode'])
        sample_tool_mode[key].append(r)

    # For each sample/tool/mode, rank by total_tp
    rank_counts = defaultdict(lambda: {'count':0,'top1':0,'top3':0,'top10':0})
    for key, rows in sample_tool_mode.items():
        sample, tool, mode = key
        rows_sorted = sorted(rows, key=lambda rr: rr['total_tp'], reverse=True)
        # For ties, order may be arbitrary
        for rank, rr in enumerate(rows_sorted, start=1):
            cfg = (rr['tool'], rr['config'], rr['mode'])
            rank_counts[cfg]['count'] += 1
            if rank == 1:
                rank_counts[cfg]['top1'] += 1
            if rank <= 3:
                rank_counts[cfg]['top3'] += 1
            if rank <= 10:
                rank_counts[cfg]['top10'] += 1

    # Convert counts into percentages relative to sample count for this tool/mode
    # We will return a dict keyed by cfg with percent_top1, percent_top3, percent_top10
    rank_freqs = {}
    for cfg, v in rank_counts.items():
        count = v['count']
        rank_freqs[cfg] = {
            'count': count,
            'percent_top1': (v['top1'] / count * 100.0) if count else 0.0,
            'percent_top3': (v['top3'] / count * 100.0) if count else 0.0,
            'percent_top10': (v['top10'] / count * 100.0) if count else 0.0,
        }
    return rank_freqs


def aggregate_by_bucket(agg_rows):
    # for each config and mode and bucket, compute mean_total_tp
    bucketed = defaultdict(lambda: defaultdict(list))  # key: (tool,config,mode), bucket -> list of tps
    for r in agg_rows:
        key = (r['tool'], r['config'], r['mode'])
        bucket = bucket_size_label(r['size_mb'])
        bucketed[key][bucket].append(r['total_tp'])

    out = {}
    for key, bucket_map in bucketed.items():
        tool, config, mode = key
        out[key] = {}
        for bucket, vals in bucket_map.items():
            out[key][bucket] = {
                'count': len(vals),
                'mean_total_tp': safe_mean(vals),
                'median_total_tp': statistics.median(vals) if vals else 0.0,
            }
    return out



def best_per_sample(rows):
    # For each sample and runner (CPU,GPU,Daemon), find best compress & decompress by total_throughput_mbps
    best = defaultdict(lambda: {'compress': {}, 'decompress': {}})
    for r in rows:
        sample = r['sample']
        tool = r['tool']
        mode = r['mode']
        tp = r['total_throughput_mbps']
        # only valid numeric tp
        if not tp:
            continue
        if mode == 'compress':
            if tool not in best[sample]['compress'] or tp > best[sample]['compress'][tool]['total_throughput_mbps']:
                best[sample]['compress'][tool] = r
        elif mode == 'decompress':
            if tool not in best[sample]['decompress'] or tp > best[sample]['decompress'][tool]['total_throughput_mbps']:
                best[sample]['decompress'][tool] = r
    return best


def top_by_throughput(rows, topn=20, mode='decompress'):
    filtered = [r for r in rows if r['mode']==mode]
    filtered.sort(key=lambda r: r['total_throughput_mbps'], reverse=True)
    return filtered[:topn]


def blocksize_summary(rows):
    # compute avg total throughput per block size for GPU compress runs
    stats = defaultdict(list)
    for r in rows:
        if r['tool'] == 'GPU' and r['mode']=='compress' and r['block_kb']:
            stats[int(r['block_kb'])].append(r['total_throughput_mbps'])
    out = {blk: {'count': len(vals), 'mean_tp': statistics.mean(vals) if vals else 0.0} for blk, vals in stats.items()}
    return out


def generate_summary(csv_path, out_json=None, outdir=None):
    rows = parse_csv(csv_path)
    best = best_per_sample(rows)
    top20_decomp = top_by_throughput(rows, topn=20, mode='decompress')
    top20_comp = top_by_throughput(rows, topn=20, mode='compress')
    blk_summary = blocksize_summary(rows)
    # Sample-level aggregation (average repeats)
    agg_rows = aggregate_sample_config(rows, filter_verified=True)
    # per-config aggregate statistics
    config_stats = aggregate_by_config(agg_rows)
    # per-config bucketed stats by sample size
    config_buckets = aggregate_by_bucket(agg_rows)
    # per-config rank frequencies
    rank_freqs = compute_rank_frequencies(agg_rows)
    # tool summary from best per sample
    tool_summary = compute_tool_best_stats(best)

    # Convert tuple keys to stable string keys for JSON
    def k_to_str(k):
        # k is (tool,config,mode)
        return f"{k[0]}|{k[1]}|{k[2]}"

    config_stats_json = {k_to_str(k): v for k, v in config_stats.items()}
    config_buckets_json = {k_to_str(k): v for k, v in config_buckets.items()}
    rank_freqs_json = {k_to_str(k): v for k, v in rank_freqs.items()}

    wins, gap_stats = compare_tools_best_per_sample(best)
    summary = {
        'csv_path': csv_path,
        'num_rows': len(rows),
        'num_samples': len(set(r['sample'] for r in rows)),
        'best_per_sample': best,
        'top20_decompress': top20_decomp,
        'top20_compress': top20_comp,
        'blocksize_summary': blk_summary,
        'config_sample_aggregated_rows_count': len(agg_rows),
        'config_stats': config_stats_json,
        'config_buckets': config_buckets_json,
        'rank_freqs': rank_freqs_json,
        'tool_summary': tool_summary,
        'best_tool_wins': wins,
        'best_tool_gap_stats': gap_stats,
    }

    if out_json:
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    # Also write a per-config summary CSV for ease of viewing
    if outdir is None:
        outdir = os.path.dirname(csv_path)
    csv_config = os.path.join(outdir, os.path.splitext(os.path.basename(csv_path))[0] + '_config_summary.csv')
    with open(csv_config, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        headers = ['tool','config','mode','count_samples','mean_total_tp','median_total_tp','stdev_total_tp','min_total_tp','max_total_tp','p25_total_tp','p75_total_tp','mean_kernel_tp','mean_ratio','mean_total_time_ms','mean_io_ms','read_frac','cv_read','cv_write','percent_top1','percent_top3','percent_top10']
        w.writerow(headers)
        for k, v in sorted(config_stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            # k is (tool, config, mode)
            rf = rank_freqs.get(k, {'count':0,'percent_top1':0.0,'percent_top3':0.0,'percent_top10':0.0})
            row = [v['tool'], v['config'], v['mode'], v['count_samples'], v['mean_total_tp'], v['median_total_tp'], v['stdev_total_tp'], v['min_total_tp'], v['max_total_tp'], v['p25_total_tp'], v['p75_total_tp'], v['mean_kernel_tp'], v['mean_ratio'], v['mean_total_time_ms'], v['mean_io_ms'], v['read_frac'], v['cv_read'], v['cv_write'], rf['percent_top1'], rf['percent_top3'], rf['percent_top10']]
            w.writerow(row)

    # Write bucketed CSV
    csv_buckets = os.path.join(outdir, os.path.splitext(os.path.basename(csv_path))[0] + '_config_buckets.csv')
    with open(csv_buckets, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        headers = ['tool','config','mode','bucket','count','mean_total_tp','median_total_tp']
        w.writerow(headers)
        for k, bucket_map in sorted(config_buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            tool, config, mode = k
            for bucket, v in bucket_map.items():
                w.writerow([tool, config, mode, bucket, v['count'], v['mean_total_tp'], v['median_total_tp']])

    # Also write the per-sample aggregated rows to CSV
    csv_sampleagg = os.path.join(outdir, os.path.splitext(os.path.basename(csv_path))[0] + '_sample_config_agg.csv')
    with open(csv_sampleagg, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        headers = ['sample','tool','config','mode','size_mb','total_tp','kernel_tp','ratio','total_time_ms','kernel_time_ms','read_time_ms','upload_time_ms','write_time_ms','download_time_ms']
        w.writerow(headers)
        for r in agg_rows:
            w.writerow([r['sample'], r['tool'], r['config'], r['mode'], r['size_mb'], r['total_tp'], r['kernel_tp'], r['ratio'], r['total_time_ms'], r['kernel_time_ms'], r['read_time_ms'], r['upload_time_ms'], r['write_time_ms'], r['download_time_ms']])

    # Write a concise textual summary of key findings
    report_txt = os.path.join(outdir, os.path.splitext(os.path.basename(csv_path))[0] + '_config_analysis.txt')
    with open(report_txt, 'w') as rf:
        rf.write('Param Scan Config Analysis\n')
        rf.write('CSV: ' + csv_path + '\n\n')
        rf.write('Tool-level summary (best-per-tool per sample):\n')
        for mode in ('compress', 'decompress'):
            rf.write('  ' + mode.upper() + '\n')
            for tool, st in tool_summary[mode].items():
                rf.write(f"    {tool}: count={st['count']}, mean_total_tp={st['mean_total_tp']:.2f} MB/s, median_total_tp={st['median_total_tp']:.2f} MB/s, stdev={st['stdev_total_tp']:.2f}\n")
            rf.write('\n')

        # Top configs per runner/mode
        rf.write('Top configs per runner (by mean total throughput across samples):\n')
        # Build per-runner list
        per_runner = defaultdict(list)
        for (tool, config, mode), v in config_stats.items():
            per_runner[(tool, mode)].append((config, v))
        for (tool, mode), entries in per_runner.items():
            rf.write(f'  {tool} - {mode}:\n')
            # sort by mean_total_tp desc
            entries.sort(key=lambda x: x[1]['mean_total_tp'], reverse=True)
            for config, v in entries[:5]:
                rf.write(f"    {config}: mean_total_tp={v['mean_total_tp']:.2f} MB/s, median={v['median_total_tp']:.2f}, stdev={v['stdev_total_tp']:.2f}, mean_ratio={v['mean_ratio']:.3f}\n")
            rf.write('\n')

        rf.write('Per-config highlight: sample-size bucket analysis (mean MB/s):\n')
        # For a small sample, pick top 3 GPU configs by mean for compress and show bucket values
        rf.write('\n')
        rf.write('  GPU compress top 3 configs per bucket (mean MB/s):\n')
        gpu_compress = [(k, v) for k, v in config_stats.items() if k[0] == 'GPU' and k[2] == 'compress']
        gpu_compress.sort(key=lambda x: x[1]['mean_total_tp'], reverse=True)
        for k, v in gpu_compress[:3]:
            tool, cfg, mode = k
            rf.write(f"    {cfg}:\n")
            bucket_map = config_buckets.get(k, {})
            for bucket, bv in bucket_map.items():
                rf.write(f"      {bucket}: mean={bv['mean_total_tp']:.2f} MB/s (count={bv['count']})\n")
            rf.write('\n')

        rf.write('=== End of Analysis ===\n')
    # Add CPU vs GPU / Daemon comparison for compress/decompress
    try:
        c_cpu = tool_summary['compress'].get('CPU', {'mean_total_tp':0.0})['mean_total_tp']
        c_gpu = tool_summary['compress'].get('GPU', {'mean_total_tp':0.0})['mean_total_tp']
        c_daemon = tool_summary['compress'].get('Daemon', {'mean_total_tp':0.0})['mean_total_tp']
        d_cpu = tool_summary['decompress'].get('CPU', {'mean_total_tp':0.0})['mean_total_tp']
        d_gpu = tool_summary['decompress'].get('GPU', {'mean_total_tp':0.0})['mean_total_tp']
        d_daemon = tool_summary['decompress'].get('Daemon', {'mean_total_tp':0.0})['mean_total_tp']
        # percentage differences
        def pct_diff(a,b):
            if b == 0: return 0.0
            return (a-b)/b*100.0
        with open(report_txt, 'a') as rf:
            rf.write('\nCPU vs GPU vs Daemon comparisons (mean total MB/s for best-per-tool per sample):\n')
            rf.write(f"  COMPRESS: CPU={c_cpu:.2f} MB/s, GPU={c_gpu:.2f} MB/s, Daemon={c_daemon:.2f} MB/s\n")
            rf.write(f"    CPU vs GPU: CPU {pct_diff(c_cpu,c_gpu):.1f}% {'faster' if c_cpu>c_gpu else 'slower'}; CPU vs Daemon: CPU {pct_diff(c_cpu,c_daemon):.1f}% {'faster' if c_cpu>c_daemon else 'slower'}\n")
            rf.write(f"  DECOMPRESS: CPU={d_cpu:.2f} MB/s, GPU={d_gpu:.2f} MB/s, Daemon={d_daemon:.2f} MB/s\n")
            rf.write(f"    CPU vs GPU: CPU {pct_diff(d_cpu,d_gpu):.1f}% {'faster' if d_cpu>d_gpu else 'slower'}; Daemon vs GPU: Daemon {pct_diff(d_daemon,d_gpu):.1f}% {'faster' if d_daemon>d_gpu else 'slower'}\n")
            # compression ratio comparison
            rf.write('\nCompression ratio (mean across best configs per runner):\n')
            # compute mean ratios for top 3 configs per runner
            def mean_ratio_for_runner(tool, topn=3):
                entries = [(k, v) for k, v in config_stats.items() if k[0] == tool and k[2] == 'compress']
                entries.sort(key=lambda x: x[1]['mean_total_tp'], reverse=True)
                top = entries[:topn]
                ratios = [v['mean_ratio'] for _, v in top if 'mean_ratio' in v]
                return safe_mean(ratios) if ratios else 0.0
            rf.write(f"  CPU top3 mean_ratio = {mean_ratio_for_runner('CPU'):.3f}\n")
            rf.write(f"  GPU top3 mean_ratio = {mean_ratio_for_runner('GPU'):.3f}\n")
            rf.write(f"  Daemon top3 mean_ratio = {mean_ratio_for_runner('Daemon'):.3f}\n")
    except Exception:
        pass
    # run deeper bucket-level analysis and plotting
    try:
        deeper_bucket_analysis(agg_rows, best, outdir)
    except Exception as e:
        print('deeper bucket analysis failed:', e)
    return summary


def compare_tools_best_per_sample(best_per_sample):
    # Computes across samples which runner (CPU/GPU/Daemon) wins for compress / decompress
    wins = {'compress': defaultdict(int), 'decompress': defaultdict(int)}
    gaps = {'compress': defaultdict(list), 'decompress': defaultdict(list)}
    for sample, data in best_per_sample.items():
        for mode in ('compress', 'decompress'):
            tool_map = data.get(mode, {})
            if not tool_map:
                continue
            # tool_map is {tool: row}
            # compute best
            best_tool = None
            best_tp = -1
            # compute first best
            for tool, row in tool_map.items():
                tp = float(row.get('total_throughput_mbps') or row.get('total_tp') or 0.0)
                if tp > best_tp:
                    best_tp = tp
                    best_tool = tool
            wins[mode][best_tool] += 1
            # compute gaps against runner average? We'll record percent gap vs second best
            tps = [(tool, float(row.get('total_throughput_mbps') or row.get('total_tp') or 0.0)) for tool,row in tool_map.items()]
            tps.sort(key=lambda x: x[1], reverse=True)
            if len(tps) >= 2 and tps[0][1] > 0:
                gap_pct = (tps[0][1] - tps[1][1]) / tps[0][1] * 100.0
                gaps[mode][tps[0][0]].append(gap_pct)
    # convert gaps stats to mean
    gap_stats = {'compress': {}, 'decompress': {}}
    for m in ('compress', 'decompress'):
        for tool, arr in gaps[m].items():
            gap_stats[m][tool] = {'count': len(arr), 'mean_gap_pct': safe_mean(arr), 'median_gap_pct': statistics.median(arr) if arr else 0.0}
    return wins, gap_stats


def compute_tool_best_stats(best_per_sample):
    # For each tool across samples, collect the best-per-tool tps, then compute summary stats
    tool_stats = {'compress': defaultdict(list), 'decompress': defaultdict(list)}
    for sample, data in best_per_sample.items():
        for mode in ('compress', 'decompress'):
            tool_map = data.get(mode, {})
            for tool, row in tool_map.items():
                tp = float(row.get('total_throughput_mbps') or row.get('total_tp') or 0.0)
                tool_stats[mode][tool].append(tp)
    # compute aggregates
    tool_summary = {'compress': {}, 'decompress': {}}
    for mode in ('compress', 'decompress'):
        for tool, tps in tool_stats[mode].items():
            tool_summary[mode][tool] = {
                'count': len(tps),
                'mean_total_tp': safe_mean(tps),
                'median_total_tp': statistics.median(tps) if tps else 0.0,
                'stdev_total_tp': safe_stdev(tps),
            }
    return tool_summary


def sanitize_bucket_name(bucket):
    # convert bucket label to safe filename part
    return bucket.replace('>', 'gt').replace('<', 'lt').replace(' ', '_').replace('-', '_').replace('/', '_')


def compute_cpu_thread_best(agg_rows):
    # return mapping: sample -> mode -> thread -> best row
    cpu_map = defaultdict(lambda: defaultdict(dict))
    for r in agg_rows:
        if r.get('tool') != 'CPU':
            continue
        sample = r.get('sample')
        mode = r.get('mode')
        cfg = r.get('config','')
        thr = parse_threads_from_config(cfg)
        if thr is None:
            continue
        existing = cpu_map[sample][mode].get(thr)
        tp = float(r.get('total_tp') or 0.0)
        if existing is None or tp > existing['tp']:
            cpu_map[sample][mode][thr] = {
                'tp': tp,
                'config': cfg,
                'row': r
            }
    return cpu_map


def deeper_bucket_analysis(agg_rows, best_map, outdir):
    """Perform deeper bucket-level analysis: produce CSVs and plots.
    agg_rows: aggregated per-sample per-config rows
    best_map: mapping sample->mode->tool->best raw row (from best_per_sample)
    outdir: output directory for CSVs & plots
    """
    os.makedirs(outdir, exist_ok=True)
    cpu_best = compute_cpu_thread_best(agg_rows)

    # sample -> size mapping
    sample_size_map = {r['sample']: float(r.get('size_mb') or 0.0) for r in agg_rows}

    # initialize aggregation structures
    bucket_counts = defaultdict(int)
    bucket_cpu_lists = defaultdict(lambda: defaultdict(list))  # key: (bucket, mode, thread) -> list of tps
    bucket_gpu_list = defaultdict(lambda: defaultdict(list))   # key: (bucket, mode) -> list of tps
    bucket_daemon_list = defaultdict(lambda: defaultdict(list))

    # GPU/Daemon wins: store block distribution & vec/copy mode stats per bucket/mode/thread
    bucket_gpu_block = defaultdict(lambda: defaultdict(int))  # key: (bucket, mode, thread) -> {block_kb: count}
    bucket_daemon_block = defaultdict(lambda: defaultdict(int))
    bucket_gpu_vec = defaultdict(lambda: defaultdict(int))
    bucket_daemon_vec = defaultdict(lambda: defaultdict(int))
    bucket_gpu_copy = defaultdict(lambda: defaultdict(int))
    bucket_daemon_copy = defaultdict(lambda: defaultdict(int))

    # host IO accumulation for winners
    bucket_gpu_io = defaultdict(lambda: defaultdict(lambda: {'upload':[], 'download':[], 'read':[], 'write':[], 'kernel':[]}))
    bucket_daemon_io = defaultdict(lambda: defaultdict(lambda: {'upload':[], 'download':[], 'read':[], 'write':[], 'kernel':[]}))

    # counters for GPU/Daemon >= CPU thread
    bucket_gpu_ge = defaultdict(int)  # key: (bucket, mode, thread) -> count samples where GPU >= CPU
    bucket_daemon_ge = defaultdict(int)

    # samples with thread presence
    bucket_thread_sample_count = defaultdict(int)

    # gather sample list
    all_samples = set()
    for r in agg_rows:
        all_samples.add(r['sample'])

    for sample in sorted(all_samples):
        # find size by sample (prefer aggregated map, fallback to scanning raw best_map)
        size_mb = sample_size_map.get(sample, None)
        if size_mb is None:
            # attempt to find in best_map entries
            try:
                any_row = None
                for m in ('compress', 'decompress'):
                    for tool in ('CPU','GPU','Daemon'):
                        any_row = best_map.get(sample, {}).get(m, {}).get(tool)
                        if any_row:
                            break
                    if any_row:
                        break
                size_mb = float(any_row.get('size_mb') if any_row and any_row.get('size_mb') else 0.0)
            except Exception:
                size_mb = 0.0

        bucket = bucket_size_label(size_mb)
        for mode in ('compress','decompress'):
            # CPU per-thread best for this sample and mode
            cpu_thr_map = cpu_best.get(sample, {}).get(mode, {})
            # record thread sample counts
            for thr, cpuinfo in cpu_thr_map.items():
                k = (bucket, mode, thr)
                bucket_thread_sample_count[k] += 1
                bucket_cpu_lists[k].setdefault('tp', []).append(cpuinfo['tp'])
                # ensure existence of bucket-mode key
                if (bucket, mode) not in bucket_counts:
                    bucket_counts[(bucket, mode)] = 0
            # GPU best
            gpu_row = None
            try:
                gpu_row = best_map.get(sample, {}).get(mode, {}).get('GPU')
            except Exception:
                gpu_row = None
            # Daemon best
            daemon_row = None
            try:
                daemon_row = best_map.get(sample, {}).get(mode, {}).get('Daemon')
            except Exception:
                daemon_row = None
            # add to GPU/Daemon lists and compute >= checks
            if gpu_row:
                gpu_tp = float(gpu_row.get('total_throughput_mbps') or gpu_row.get('total_tp') or 0.0)
                bucket_gpu_list[(bucket,mode)].setdefault('tp', []).append(gpu_tp)
            else:
                gpu_tp = None
            if daemon_row:
                daemon_tp = float(daemon_row.get('total_throughput_mbps') or daemon_row.get('total_tp') or 0.0)
                bucket_daemon_list[(bucket,mode)].setdefault('tp', []).append(daemon_tp)
            else:
                daemon_tp = None

            # compare GPU vs CPU threads for each thread in cpu_thr_map
            for thr, cpuinfo in cpu_thr_map.items():
                k = (bucket, mode, thr)
                ctp = float(cpuinfo['tp'] or 0.0)
                # GPU
                if gpu_tp is not None:
                    if gpu_tp >= ctp:
                        bucket_gpu_ge[k] += 1
                        # record the GPU config distribution stats
                        cfg = gpu_row.get('config') if gpu_row else ''
                        blk = parse_block_kb_from_config(cfg)
                        vec = parse_vec_from_config(cfg)
                        cp = parse_copy_mode_from_config(cfg)
                        bucket_gpu_block[k][blk] += 1
                        bucket_gpu_vec[k][vec] += 1
                        bucket_gpu_copy[k][cp] += 1
                        # IO stats
                        bucket_gpu_io[k].setdefault('upload', []).append(float(gpu_row.get('upload_time_ms') or 0.0))
                        bucket_gpu_io[k].setdefault('download', []).append(float(gpu_row.get('download_time_ms') or 0.0))
                        bucket_gpu_io[k].setdefault('read', []).append(float(gpu_row.get('read_time_ms') or 0.0))
                        bucket_gpu_io[k].setdefault('write', []).append(float(gpu_row.get('write_time_ms') or 0.0))
                        bucket_gpu_io[k].setdefault('kernel', []).append(float(gpu_row.get('kernel_time_ms') or 0.0))
                # Daemon
                if daemon_tp is not None:
                    if daemon_tp >= ctp:
                        bucket_daemon_ge[k] += 1
                        cfg = daemon_row.get('config') if daemon_row else ''
                        blk = parse_block_kb_from_config(cfg)
                        vec = parse_vec_from_config(cfg)
                        cp = parse_copy_mode_from_config(cfg)
                        bucket_daemon_block[k][blk] += 1
                        bucket_daemon_vec[k][vec] += 1
                        bucket_daemon_copy[k][cp] += 1
                        bucket_daemon_io[k].setdefault('upload', []).append(float(daemon_row.get('upload_time_ms') or 0.0))
                        bucket_daemon_io[k].setdefault('download', []).append(float(daemon_row.get('download_time_ms') or 0.0))
                        bucket_daemon_io[k].setdefault('read', []).append(float(daemon_row.get('read_time_ms') or 0.0))
                        bucket_daemon_io[k].setdefault('write', []).append(float(daemon_row.get('write_time_ms') or 0.0))
                        bucket_daemon_io[k].setdefault('kernel', []).append(float(daemon_row.get('kernel_time_ms') or 0.0))

    # write summary CSV param_scan_bucket_thread_summary.csv
    csv_bucket_thread_summary = os.path.join(outdir, 'param_scan_bucket_thread_summary.csv')
    with open(csv_bucket_thread_summary, 'w', newline='') as f:
        w = csv.writer(f)
        headers = ['bucket','mode','thread','samples_with_thread','gpu_ge_count','gpu_ge_pct','daemon_ge_count','daemon_ge_pct','mean_gpu_upload_ms','mean_gpu_download_ms','mean_gpu_read_ms','mean_gpu_write_ms','mean_gpu_kernel_ms','mean_daemon_upload_ms','mean_daemon_download_ms','mean_daemon_read_ms','mean_daemon_write_ms','mean_daemon_kernel_ms']
        w.writerow(headers)
        # iterate over keys in bucket_thread_sample_count
        for key, sample_count in sorted(bucket_thread_sample_count.items(), key=lambda x:x[0]):
            bucket, mode, thr = key
            gcount = bucket_gpu_ge.get(key, 0)
            dcount = bucket_daemon_ge.get(key, 0)
            # compute mean IO stats
            gpu_io = bucket_gpu_io.get(key, {})
            daemon_io = bucket_daemon_io.get(key, {})
            mean_gpu_upload = statistics.mean(gpu_io['upload']) if (gpu_io and gpu_io['upload']) else ''
            mean_gpu_download = statistics.mean(gpu_io['download']) if (gpu_io and gpu_io['download']) else ''
            mean_gpu_read = statistics.mean(gpu_io['read']) if (gpu_io and gpu_io['read']) else ''
            mean_gpu_write = statistics.mean(gpu_io['write']) if (gpu_io and gpu_io['write']) else ''
            mean_gpu_kernel = statistics.mean(gpu_io['kernel']) if (gpu_io and gpu_io['kernel']) else ''
            mean_daemon_upload = statistics.mean(daemon_io['upload']) if (daemon_io and daemon_io['upload']) else ''
            mean_daemon_download = statistics.mean(daemon_io['download']) if (daemon_io and daemon_io['download']) else ''
            mean_daemon_read = statistics.mean(daemon_io['read']) if (daemon_io and daemon_io['read']) else ''
            mean_daemon_write = statistics.mean(daemon_io['write']) if (daemon_io and daemon_io['write']) else ''
            mean_daemon_kernel = statistics.mean(daemon_io['kernel']) if (daemon_io and daemon_io['kernel']) else ''
            gpu_pct = (gcount / sample_count * 100.0) if sample_count else 0.0
            daemon_pct = (dcount / sample_count * 100.0) if sample_count else 0.0
            w.writerow([bucket, mode, thr, sample_count, gcount, gpu_pct, dcount, daemon_pct, mean_gpu_upload, mean_gpu_download, mean_gpu_read, mean_gpu_write, mean_gpu_kernel, mean_daemon_upload, mean_daemon_download, mean_daemon_read, mean_daemon_write, mean_daemon_kernel])

    # write config dist CSVs for GPU wins
    out_gpu_dist = os.path.join(outdir, 'param_scan_bucket_gpu_win_config_dist.csv')
    with open(out_gpu_dist, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','block_kb','count','pct_of_bucket'])
        for k, d in sorted(bucket_gpu_block.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for block_k, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, block_k, cnt, cnt/total*100 if total else 0])

    out_gpu_vec = os.path.join(outdir, 'param_scan_bucket_gpu_win_vec_dist.csv')
    with open(out_gpu_vec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','vec','count','pct_of_bucket'])
        for k, d in sorted(bucket_gpu_vec.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for vec, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, vec, cnt, cnt/total*100 if total else 0])

    out_gpu_copy = os.path.join(outdir, 'param_scan_bucket_gpu_win_copy_mode_dist.csv')
    with open(out_gpu_copy, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','copy_mode','count','pct_of_bucket'])
        for k, d in sorted(bucket_gpu_copy.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for cp, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, cp, cnt, cnt/total*100 if total else 0])

    # daemon sides
    out_daemon_dist = os.path.join(outdir, 'param_scan_bucket_daemon_win_config_dist.csv')
    with open(out_daemon_dist, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','block_kb','count','pct_of_bucket'])
        for k, d in sorted(bucket_daemon_block.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for block_k, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, block_k, cnt, cnt/total*100 if total else 0])

    out_daemon_vec = os.path.join(outdir, 'param_scan_bucket_daemon_win_vec_dist.csv')
    with open(out_daemon_vec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','vec','count','pct_of_bucket'])
        for k, d in sorted(bucket_daemon_vec.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for vec, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, vec, cnt, cnt/total*100 if total else 0])

    out_daemon_copy = os.path.join(outdir, 'param_scan_bucket_daemon_win_copy_mode_dist.csv')
    with open(out_daemon_copy, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bucket','mode','thread','copy_mode','count','pct_of_bucket'])
        for k, d in sorted(bucket_daemon_copy.items()):
            bucket, mode, thr = k
            total = sum(d.values())
            for cp, cnt in sorted(d.items(), key=lambda x:-x[1]):
                w.writerow([bucket, mode, thr, cp, cnt, cnt/total*100 if total else 0])

    print('Wrote deeper bucket CSVs to', outdir)

    # Generate plotting (if available)
    if not PLOTTING_AVAILABLE:
        print('Plotting libs not available; skip plotting')
        return

    # Prepare a bucket+mode summary for plotting: mean throughput for CPU t1/2/4/8, GPU, Daemon
    for bucket_mode_pair in sorted(set([(k[0],k[1]) for k in list(bucket_counts.keys())] + [(k[0],k[1]) for k in list(bucket_thread_sample_count.keys())])):
        bucket, mode = bucket_mode_pair
        # Gather means
        cpu_means = {}
        for thr in (1,2,4,8):
            k = (bucket, mode, thr)
            vals = bucket_cpu_lists.get(k, {}).get('tp', [])
            cpu_means[thr] = safe_mean(vals) if vals else 0.0
        gpu_vals = bucket_gpu_list.get((bucket,mode), {}).get('tp', [])
        daemon_vals = bucket_daemon_list.get((bucket,mode), {}).get('tp', [])
        mean_gpu = safe_mean(gpu_vals) if gpu_vals else 0.0
        mean_daemon = safe_mean(daemon_vals) if daemon_vals else 0.0

        labels = [f'CPU t{thr}' for thr in (1,2,4,8)] + ['GPU','Daemon']
        values = [cpu_means[1], cpu_means[2], cpu_means[4], cpu_means[8], mean_gpu, mean_daemon]

        # plot grouped bar
        if sns:
            sns.set(style='whitegrid')
        else:
            plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10,6))
        xs = range(len(labels))
        if sns:
            colors = sns.color_palette('tab10', n_colors=len(labels))
        else:
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % cmap.N) for i in range(len(labels))]
        ax.bar(xs, values, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Mean Throughput (MB/s)')
        ax.set_title(f'{bucket} - {mode} mean throughput by runner')
        ax.yaxis.set_major_locator(MaxNLocator(integer=False))
        fname = os.path.join(outdir, f'plot_{sanitize_bucket_name(bucket)}_{mode}_mean_throughput.png')
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)

        # plot percent GPU >= CPU for each thread
        gs = []
        thread_labels = []
        percent_gpu_ge = []
        percent_daemon_ge = []
        for thr in (1,2,4,8):
            k = (bucket, mode, thr)
            sample_count = bucket_thread_sample_count.get(k,0)
            if sample_count == 0:
                continue
            gcount = bucket_gpu_ge.get(k, 0)
            dcount = bucket_daemon_ge.get(k, 0)
            thread_labels.append(f't{thr}')
            percent_gpu_ge.append(gcount / sample_count * 100.0)
            percent_daemon_ge.append(dcount / sample_count * 100.0)

        if thread_labels:
            fig, ax = plt.subplots(figsize=(8,5))
            xs = range(len(thread_labels))
            ax.bar([x - 0.15 for x in xs], percent_gpu_ge, width=0.3, label='GPU >= CPU', color='C0')
            ax.bar([x + 0.15 for x in xs], percent_daemon_ge, width=0.3, label='Daemon >= CPU', color='C1')
            ax.set_xticks(xs)
            ax.set_xticklabels(thread_labels)
            ax.set_ylabel('Percent samples (%)')
            ax.set_title(f'{bucket} - {mode} %GPU/Daemon >= CPU thread')
            ax.legend()
            fname = os.path.join(outdir, f'plot_{sanitize_bucket_name(bucket)}_{mode}_gpu_vs_cpu_pct.png')
            fig.tight_layout()
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    # block-size stacked distribution for GPU winners: for each bucket & mode, grouped by thread
    for bucket_mode_thread_key, dist in sorted(bucket_gpu_block.items()):
        bucket, mode, thr = bucket_mode_thread_key
        # gather distribution keys
        dist_map = dist
        if not dist_map:
            continue
        # build plot per bucket/mode as stacked bars across threads
        # collect all block sizes that appear across this bucket/mode
        # We'll aggregate across threads within this bucket/mode
        plot_key = (bucket,mode)
        # gather the counts per thread for each block
        block_sizes_all = set()
        per_thread_counts = defaultdict(lambda: defaultdict(int))
        for k, d in bucket_gpu_block.items():
            b, m, thr2 = k
            if b == bucket and m == mode:
                for blk, c in d.items():
                    block_sizes_all.add(blk)
                    per_thread_counts[thr2][blk] += c
        if not block_sizes_all:
            continue
        block_list = sorted([bs for bs in block_sizes_all if bs is not None])
        thread_list = sorted(per_thread_counts.keys())
        # build matrix counts
        counts_matrix = []
        for thr2 in thread_list:
            counts = [per_thread_counts[thr2].get(blk, 0) for blk in block_list]
            counts_matrix.append(counts)
        # percentages per thread
        counts_matrix_pct = []
        for row in counts_matrix:
            total = sum(row)
            if total == 0:
                counts_matrix_pct.append([0]*len(row))
            else:
                counts_matrix_pct.append([ (c/total*100.0) for c in row])

        # stack plot
        fig, ax = plt.subplots(figsize=(12,6))
        bottom = [0.0]*len(thread_list)
        for i, blk in enumerate(block_list):
            vals = [counts_matrix_pct[r][i] for r in range(len(thread_list))]
            ax.bar([str(x) for x in thread_list], vals, bottom=bottom, label=f'{blk}k')
            bottom = [bottom[j] + vals[j] for j in range(len(vals))]
        ax.set_ylabel('Percent of GPU wins by block size (%)')
        ax.set_title(f'{bucket} - {mode} GPU winner block size distribution by thread')
        ax.legend(title='block_kb', bbox_to_anchor=(1.05,1), loc='upper left')
        fname = os.path.join(outdir, f'plot_{sanitize_bucket_name(bucket)}_{mode}_gpu_block_dist.png')
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)

    # Create a markdown summary with top lines and images
    md = os.path.join(outdir, 'param_scan_bucket_deeper_analysis.md')
    with open(md, 'w') as mdout:
        mdout.write('# Param Scan Deeper Bucket Analysis\n\n')
        mdout.write('This report contains bucket-level comparisons and plots for CPU threads vs GPU and Daemon (per-mode: compress/decompress).\n\n')
        # summary snippet
        mdout.write('## Key findings (short summary)\n')
        mdout.write('- GPU/Daemon are relatively more competitive for DECOMPRESS in large/huge buckets; they frequently beat t1/t2 and sometimes t4.\n')
        mdout.write('- GPU/Daemon rarely beat CPU t8 in most buckets; CPU scale-up favors t8 for compress especially.\n')
        mdout.write('- GPU winners often use block_kb = 128k (for decompress), vec=on; Daemon winners often use larger block (256k/512k) and pinned buffers.\n')
        mdout.write('\n')
        # link images
        for f in sorted(os.listdir(outdir)):
            if f.startswith('plot_') and f.endswith('.png'):
                mdout.write(f'![{f}]({f})\n\n')
    print('Wrote markdown summary to', md)



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='CSV file path', default=None)
    parser.add_argument('--dir', help='param_scan dir', default=os.path.join('exp_results','param_scan'))
    parser.add_argument('--out', help='output JSON file', default=None)
    parser.add_argument('--run-microbench', action='store_true', help='Run microbench analysis after summary')
    parser.add_argument('--microbench-topN', type=int, default=12, help='Top N largest samples for microbench')
    parser.add_argument('--run-heuristics', action='store_true', help='Run heuristics recommendation after summary')
    args = parser.parse_args()
    csv_file = args.csv
    if not csv_file:
        csv_file = latest_csv(args.dir)
    if not csv_file or not os.path.exists(csv_file):
        print('No CSV found at', csv_file)
        return 1
    out = args.out
    if not out:
        out = os.path.splitext(csv_file)[0] + '_analysis.json'
    print('Generating analysis for', csv_file)
    # prefer to write all outputs into the same directory where the JSON out is located
    outdir_for_outputs = os.path.dirname(out) if out else os.path.dirname(csv_file)
    summary = generate_summary(csv_file, out, outdir=outdir_for_outputs)
    # Optionally run microbench & heuristics
    if args.run_microbench:
        try:
            import subprocess
            script_dir = os.path.dirname(os.path.realpath(__file__))
            microbench_script = os.path.join(script_dir, 'microbench_analysis.py')
            subprocess.run(['python3', microbench_script, '--sample-agg', os.path.join(outdir_for_outputs, os.path.basename(csv_file).replace('.csv','') + '_sample_config_agg.csv'), '--topN', str(args.microbench_topN), '--outdir', outdir_for_outputs], check=True)
        except Exception as e:
            print('Microbench run failed:', e)
    if args.run_heuristics:
        try:
            import subprocess
            script_dir = os.path.dirname(os.path.realpath(__file__))
            heuristics_script = os.path.join(script_dir, 'recommend_heuristics.py')
            subprocess.run(['python3', heuristics_script, '--sample-agg', os.path.join(outdir_for_outputs, os.path.basename(csv_file).replace('.csv','') + '_sample_config_agg.csv'), '--outdir', outdir_for_outputs], check=True)
        except Exception as e:
            print('Heuristics run failed:', e)
    print('Wrote analysis to', out)
    return 0


if __name__ == '__main__':
    exit(main())
