#!/usr/bin/env python3
"""Deep kernel sweep for representative samples.

Runs standalone lzo_gpu in baseline/opt/opt-debug/debug kernels across
D_BITS × block sizes and aggregates median kernel exec times and
per-block counters (lookups/hits/match_bytes).

Usage example:
    python3 tools/deep_kernel_sweep.py --out /tmp/deep_scan --blocks 256B,512B,1k,2k,4k,8k --bits 6,7,8,9,10,11,12,14 --repeats 3

Defaults:
- samples: all regular files under /root/samples
- variants: baseline,opt,opt-debug,debug
"""

import argparse
import os
import subprocess
import time
import datetime
import re
import csv
import filecmp

from pathlib import Path
from statistics import median
import math
import tempfile

patt_full = re.compile(r'LZO_GPU_DEBUG BLOCK\s*(\d+)\s+IN\s*(\d+)\s+OUT\s*(\d+)\s+FLAG\s*(\d+)\s+LOOKUPS\s*(\d+)\s+HITS\s*(\d+)\s+MATCH_BYTES\s*(\d+)\s+UPDATES\s*(\d+)')


def run_cmd(cmd, timeout=300, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # If LZO_GPU_DIR is specified, run the command in that directory so that
    # the OpenCL compiler can find local headers (like lzo_gpu.h) via -I.
    cwd = env.get('LZO_GPU_DIR')
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=cwd)
    return p.returncode, p.stdout.decode('utf-8', errors='replace'), p.stderr.decode('utf-8', errors='replace')


def parse_gpu_debug(stderr_text):
    blocks = []
    patt_short = re.compile(r'LZO_GPU_DEBUG BLOCK\s*(\d+)\s+IN\s*(\d+)\s+OUT\s*(\d+)\s+FLAG\s*(\d+)', re.I)
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line.startswith('LZO_GPU_DEBUG BLOCK'):
            continue
        # Try full format first (LOOKUPS/HITS/MATCH_BYTES/UPDATES)
        m = patt_full.search(line)
        if m:
            blocks.append({'idx':int(m.group(1)),'in_len':int(m.group(2)),'out_len':int(m.group(3)),'flag':int(m.group(4)),'lookups':int(m.group(5)),'hits':int(m.group(6)),'matched_bytes':int(m.group(7)),'updates':int(m.group(8))})
            continue
        # Fallback: short format with only IN/OUT/FLAG
        m2 = patt_short.search(line)
        if m2:
            blocks.append({'idx':int(m2.group(1)),'in_len':int(m2.group(2)),'out_len':int(m2.group(3)),'flag':int(m2.group(4)),'lookups':0,'hits':0,'matched_bytes':0,'updates':0})
    return blocks


patt_kernel_ms = re.compile(r'Kernel Exec\s*:\s*([0-9.]+)\s*ms')


def parse_kernel_ms(stdout_text):
    comp_ms = None
    dec_ms = None

    # Split by Statistics headers to separate compression and decompression
    parts = re.split(r'=== (?:Compression|Decompression) Statistics ===', stdout_text)

    # Patt for "Kernel Exec : 123.456 ms"
    patt = re.compile(r'Kernel Exec\s*:\s*([0-9.]+)\s*ms')

    # If we have 3 parts: [before, compression_stats, decompression_stats]
    if len(parts) >= 3:
        # Compression is in parts[1], Decompression in parts[2]
        m1 = patt.search(parts[1])
        if m1: comp_ms = float(m1.group(1))
        m2 = patt.search(parts[2])
        if m2: dec_ms = float(m2.group(1))
        return comp_ms, dec_ms

    # Fallback for single match
    m = patt.search(stdout_text)
    if m: comp_ms = float(m.group(1))
    return comp_ms, None

def parse_kernel_thr(stdout_text):
    comp_thr = None
    dec_thr = None

    parts = re.split(r'=== (?:Compression|Decompression) Statistics ===', stdout_text)
    # Patt for "(kernel: 123.45 MB/s)"
    patt = re.compile(r'\(kernel:\s*([0-9.]+)\s*MB/s\)')

    if len(parts) >= 3:
        m1 = patt.search(parts[1])
        if m1: comp_thr = float(m1.group(1))
        m2 = patt.search(parts[2])
        if m2: dec_thr = float(m2.group(1))
        return comp_thr, dec_thr

    m = patt.search(stdout_text)
    if m: comp_thr = float(m.group(1))
    return comp_thr, None


def parse_size_bytes(s):
    """Parse size strings like '256B','8k','4K','1m' into bytes (int)."""
    if not s: return 0
    s = s.strip()
    m = re.match(r'^(\d+)([bBkKmMgG][bB]?)?$', s)
    if not m:
        return 0
    val = int(m.group(1))
    unit = m.group(2)
    if not unit:
        return val
    unit = unit.lower()
    if unit in ('b','bytes'):
        return val
    if unit in ('k','kb'):
        return val * 1024
    if unit in ('m','mb'):
        return val * 1024 * 1024
    if unit in ('g','gb'):
        return val * 1024 * 1024 * 1024
    return val


def median_or_none(lst):
    vals = [v for v in lst if v is not None]
    if not vals: return None
    return median(vals)


def format_size_bytes(n):
    """Format integer bytes into short size strings (e.g., 256B, 1k, 4M)."""
    if n is None:
        return ''
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n % (1024*1024) == 0:
        return f"{n // (1024*1024)}M"
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return f"{n}B"


def get_cl_device_info():
    """Query clinfo for device local mem and compute units; fallback to sane defaults."""
    try:
        out = subprocess.check_output(['clinfo'], stderr=subprocess.STDOUT, universal_newlines=True, timeout=5)
        max_cu = None; local_mem = None
        for line in out.splitlines():
            ll = line.strip()
            m = re.match(r'^Max compute units\s+(\d+)', ll)
            if m: max_cu = int(m.group(1))
            m2 = re.match(r'^Local memory size\s+(\d+)', ll)
            if m2: local_mem = int(m2.group(1))
        if max_cu is None:
            max_cu = max_cu
        if local_mem is None:
            local_mem = 64 * 1024
        return {'max_cu': max_cu, 'local_mem': local_mem}
    except Exception:
        return {'max_cu': 1, 'local_mem': 64 * 1024}

DBG_MAX_BYTES = 64 * 1024 * 1024
MAX_NBLK = 500000
# maximum block size to attempt (matches LZO_MAX_BLOCK_BYTES_DEFAULT)
MAX_BLOCK_BYTES = 64 * 1024  # updated to match LZO_MAX_BLOCK_BYTES_DEFAULT (64KB)
ADAPTIVE_SMALL_THRESHOLD = 1 * 1024 * 1024
TIMEOUT = 60  # default per-run timeout (seconds)


def ceil_div(a, b):
    return (a + b - 1) // b if b else 0


def align_up(x, align):
    if align <= 0:
        return x
    return ((x + align - 1) // align) * align


def choose_adaptive_blocks_bits(size_bytes, dev_info):
    """Return (blocks_list_bytes, bits_list, info_str) chosen for a sample size.

    Heuristic: small files -> small blocks/small dict; large files -> large
    blocks/large dict. When a candidate block would cause excessive nblk or
    debug buffer size, automatically increase the block size up to the
    device-allowed maximum instead of skipping the candidate.
    """
    if size_bytes is None:
        return None, None, 'size_unknown'

    KB = 1024
    MB = KB * KB

    # More permissive candidates (allow up to the device/C default max block)
    if size_bytes < 4 * MB:
        cand_blocks = [1024, 2048, 4096]
        cand_bits = [10, 11, 12]
    elif size_bytes < 38 * MB:
        cand_blocks = [4096, 8192, 16384]
        cand_bits = [10, 11, 12]
    elif size_bytes < 98 * MB:
        cand_blocks = [8192, 16384, 32768]
        cand_bits = [11, 12, 13]
    else:
        cand_blocks = [16384,32768, 65536]
        cand_bits = [11, 12, 13]

    # cap bits by device local memory (dict entry = 2 bytes)
    local_mem = dev_info.get('local_mem', 64 * 1024) if dev_info else 64 * 1024
    safety_frac = 0.85
    eff_local = max(2048, int(local_mem * safety_frac))
    dict_entry = 2
    try:
        max_bits_local = int(math.floor(math.log2(max(1, eff_local // dict_entry))))
    except Exception:
        max_bits_local = 14
    max_bits_allowed = min(max_bits_local, 14)
    cand_bits = [bb for bb in cand_bits if bb <= max_bits_allowed]
    if not cand_bits:
        cand_bits = [min(12, max_bits_allowed)]

    # compute max number of instrumented blocks permitted by DBG_MAX_BYTES
    max_instrumented_blocks = DBG_MAX_BYTES // (7 * 4) if DBG_MAX_BYTES else MAX_NBLK

    filtered_blocks = []
    warnings = []
    for blk in cand_blocks:
        blk = min(blk, MAX_BLOCK_BYTES)
        nblk_est = ceil_div(size_bytes, blk)
        dbg_bytes = nblk_est * 7 * 4

        # try to satisfy safety constraints by increasing the block size
        if (MAX_NBLK and nblk_est > MAX_NBLK) or (DBG_MAX_BYTES and dbg_bytes > DBG_MAX_BYTES):
            required_by_nblk = ceil_div(size_bytes, MAX_NBLK) if MAX_NBLK else blk
            required_by_dbg = ceil_div(size_bytes, max_instrumented_blocks) if max_instrumented_blocks else blk
            required_block = max(blk, required_by_nblk, required_by_dbg)
            required_block = align_up(required_block, 1024)
            if required_block <= MAX_BLOCK_BYTES:
                blk_used = required_block
                warnings.append(f'upscaled {format_size_bytes(blk)}->{format_size_bytes(blk_used)}')
            else:
                blk_used = MAX_BLOCK_BYTES
                warnings.append(f'capped {format_size_bytes(blk)}->{format_size_bytes(blk_used)} (max)')
        else:
            blk_used = blk

        if blk_used not in filtered_blocks:
            filtered_blocks.append(blk_used)

    if not filtered_blocks:
        # as a last resort, choose the maximum allowed block
        filtered_blocks = [MAX_BLOCK_BYTES]
        warnings.append(f'fallback->max({format_size_bytes(MAX_BLOCK_BYTES)})')

    filtered_blocks = sorted(set(filtered_blocks))
    info = f'auto_blocks={",".join(format_size_bytes(b) for b in filtered_blocks)};bits={",".join(str(x) for x in cand_bits)};local_mem={local_mem};warnings={"|".join(warnings)}'
    return filtered_blocks, cand_bits, info


def discover_samples(arg_samples):
    """Return a list of sample filenames.

    If arg_samples is provided, it can be a comma-separated list of filenames
    or a list of paths (if passed via shell glob).
    """
    if arg_samples:
        # Handle both comma-separated string and list of paths from glob
        if isinstance(arg_samples, str):
            items = [s.strip() for s in arg_samples.split(',') if s.strip()]
        else:
            items = arg_samples

        # If items are absolute paths, just take the basename but keep the path for later
        # Actually, the rest of the script expects filenames relative to /root/samples
        # or absolute paths. Let's normalize to absolute paths if they exist.
        results = []
        for item in items:
            p = Path(item)
            if p.exists():
                results.append(str(p.resolve()))
            else:
                # Try relative to /root/samples
                p2 = Path('/root/samples') / item
                if p2.exists():
                    results.append(str(p2.resolve()))
        return results

    root = Path('/root/samples')
    if not root.exists():
        return []
    # Include both regular files and symlinks (resolve symlinks to verify they point to files)
    files = []
    for p in root.iterdir():
        # Include if it's a regular file or if it's a symlink that resolves to a file
        if p.is_file() or (p.is_symlink() and p.resolve().is_file()):
            files.append(p.name)
    return sorted(files)


def normalize_variants(arg_variants):
    """Normalize and validate variant names.

    Accepts variants like: baseline, exp, exp2, opt, debug, opt-debug, exp-opt, exp-debug, exp2-opt, etc.
    Returns the normalized unique list in the same order provided.
    """
    mapped = []
    for v in arg_variants:
        if not v: continue
        v = v.strip().lower()
        # normalize alternative spellings
        v = v.replace('debug-opt','opt-debug')
        v = v.replace('_','-')
        # accept any token that looks reasonable (letters, digits, hyphen)
        if re.match(r'^[a-z0-9\-]+$', v):
            if v not in mapped:
                mapped.append(v)
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='Output directory')
    ap.add_argument('--samples', default=None, help='Comma-separated sample filenames; default: all files under /root/samples')
    ap.add_argument('--blocks', default=None, help='Comma-separated block sizes to consider; when omitted, per-sample adaptive selection is used')
    ap.add_argument('--bits', default='10,11,12,13', help='Comma-separated D_BITS values')
    ap.add_argument('--variants', default='baseline,opt,opt-debug,debug', help='Comma-separated kernel variants (baseline,opt,opt-debug,debug)')
    ap.add_argument('--repeats', type=int, default=3, help='Number of repeats per config')
    ap.add_argument('--algo', default='lzo1x', help='Algorithm name (lzo1x, lzo1y)')
    args = ap.parse_args()

    out_base = Path(args.out) if args.out else None

    # resolve path to the lzo_gpu binary in the repo root
    repo_root = Path(__file__).resolve().parents[1]
    lzo_bin = str(repo_root / 'lzo_gpu' / 'lzo_gpu')

    samples = discover_samples(args.samples)
    # DEFAULT_BLOCKS is used only when the user doesn't specify --blocks
    DEFAULT_BLOCKS = '1k,4k,8k,16k,32k'
    blocks_arg = args.blocks if args.blocks is not None else DEFAULT_BLOCKS
    blocks = [b.strip() for b in blocks_arg.split(',') if b.strip()]
    bits = [int(b.strip()) for b in args.bits.split(',') if b.strip()]
    variants = normalize_variants(args.variants.split(','))

    header = ['sample','sample_path','size_bytes','block_req','block_actual_kb','comp_level','kernel_variant','median_kernel_ms','median_kernel_thr','median_decomp_kernel_ms','median_decomp_kernel_thr','median_total_time_ms','median_nblk','median_total_lookups','median_total_hits','median_total_match_bytes','median_hit_rate','repeats','verified','compressed_size_bytes','compression_ratio']

    # prepare CSV writer: stdout when no out_dir (zero-trace), or file when out specified
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)
        csv_path = out_base / 'deep_summary.csv'
        fout = open(csv_path, 'w', newline='')
        close_fout = True
    else:
        import sys
        fout = sys.stdout
        close_fout = False

    w = csv.writer(fout)
    w.writerow(header)

    total = len(samples)*len(blocks)*len(bits)*len(variants)
    done = 0
    start = time.time()

    for sample in samples:
            # resolve sample path: accept absolute paths or relative to /root/samples, follow symlinks
            if os.path.isabs(sample):
                sample_path = sample
            else:
                sample_path = str(Path('/root/samples') / sample)
            # canonicalize (follow symlinks) and compute basename for safe filenames
            try:
                sample_path = os.path.realpath(sample_path)
            except Exception:
                pass
            sample_basename = os.path.basename(sample_path)
            safe_sample = re.sub(r'[^A-Za-z0-9._-]+', '_', sample_basename)
            try:
                size_bytes = os.path.getsize(sample_path)
            except Exception:
                size_bytes = None

            if not os.path.exists(sample_path):
                print(f"Sample missing: {sample_path} — marking combos as SKIPPED")
                sample_missing = True
            else:
                sample_missing = False

            # If the sample file is missing, mark the requested combinations as SKIPPED
            if sample_missing:
                for block in blocks:
                    for b in bits:
                        for v in variants:
                            done += 1
                            cfg = f'B{block}_L{b}_{v}'
                            print(f"[{done}] {sample_basename} {cfg}", flush=True)
                            row = [sample_basename, sample_path, size_bytes, block, 'SKIPPED', b, v, None, None, None, None, None, None, None, None, args.repeats, 'SKIPPED', '', '']
                            w.writerow(row); fout.flush()
                continue

            # Choose per-sample candidates (adaptive unless the user explicitly supplied --blocks)
            if size_bytes is not None:
                dev_info = get_cl_device_info()
                cand_blocks, cand_bits, _ = choose_adaptive_blocks_bits(size_bytes, dev_info)
                # If the user explicitly passed --blocks, honor those values instead of adaptive choices
                if args.blocks is not None:
                    # user explicitly provided blocks: honor those and use the requested bits
                    cand_blocks = [parse_size_bytes(b) for b in blocks]
                    cand_bits = bits
                # fall back to provided bits if adaptive did not return any
                if cand_bits is None:
                    cand_bits = bits
                if cand_blocks is None:
                    cand_blocks = [parse_size_bytes(b) for b in blocks]
            else:
                cand_blocks = [parse_size_bytes(b) for b in blocks]
                cand_bits = bits

            # Iterate chosen candidates for this sample
            for blk in cand_blocks:
                # string representation for command and CSV
                block = format_size_bytes(blk)
                for b in cand_bits:
                    for v in variants:
                        done += 1
                        cfg = f'B{block}_L{b}_{v}'
                        print(f"[{done}] {sample_basename} {cfg}", flush=True)

                        # variant tokens: debug and opt may be combined with exp/exp2
                        v_debug = ('debug' in v)
                        v_opt = ('opt' in v)
                        # algorithm name for -a flag
                        alg_name = args.algo
                        # prepare base extra env (no LZO_GPU_DIR needed - clbins in lzo_gpu dir)
                        extra_env_base = {}
                        if v_opt:
                            extra_env_base['LZO_KERNEL_OPT'] = '1'

                        # Safety checks: rely on adaptive candidates and buffer limits; min-block restriction removed
                        block_bytes = int(blk)
                        if block_bytes == 0:
                            print(f"Warning: cannot parse block '{block}'; skipping this block")
                            continue

                        # estimate debug buffer bytes (DBG_FIELDS=7, uint32)
                        nblk_est = (size_bytes + block_bytes - 1) // block_bytes if size_bytes else 0
                        dbg_bytes = nblk_est * 7 * 4

                        # If instrumented run would create too many blocks or too large debug buffer,
                        # try to increase block size (upscale) to meet safety constraints; if that's
                        # impossible (exceeds MAX_BLOCK_BYTES) then SKIP.
                        if v_debug and ( (MAX_NBLK and nblk_est > MAX_NBLK) or (DBG_MAX_BYTES and dbg_bytes > DBG_MAX_BYTES) ):
                            max_instrumented_blocks = DBG_MAX_BYTES // (7 * 4) if DBG_MAX_BYTES else MAX_NBLK
                            required_by_nblk = (size_bytes + MAX_NBLK - 1) // MAX_NBLK if MAX_NBLK else block_bytes
                            required_by_dbg = (size_bytes + max_instrumented_blocks - 1) // max_instrumented_blocks if max_instrumented_blocks else block_bytes
                            required_block = max(block_bytes, required_by_nblk, required_by_dbg)
                            # round to 1KB
                            required_block = ((required_block + 1023) // 1024) * 1024
                            if required_block <= MAX_BLOCK_BYTES:
                                print(f"ADJUST: {sample_basename} {cfg} block {format_size_bytes(block_bytes)} -> {format_size_bytes(required_block)} to respect safety bounds")
                                block = format_size_bytes(required_block)
                                block_bytes = required_block
                                # recompute estimates
                                nblk_est = (size_bytes + block_bytes - 1) // block_bytes if size_bytes else 0
                                dbg_bytes = nblk_est * 7 * 4
                            else:
                                print(f"SKIP: {sample_basename} {cfg} (required block {format_size_bytes(required_block)} > max {format_size_bytes(MAX_BLOCK_BYTES)})")
                                row = [sample_basename, sample_path, size_bytes, block, 'SKIPPED', b, v, None, None, None, None, None, None, None, None, args.repeats, 'SKIPPED', '', '']
                                w.writerow(row); fout.flush();
                                continue



                        # initial verification run (once per config) and capture compressed size
                        kernel_ms_vals = []
                        kernel_thr_vals = []
                        decomp_kernel_ms_vals = []
                        decomp_kernel_thr_vals = []
                        total_time_vals = []
                        nblk_vals = []
                        total_lookups_vals = []
                        total_hits_vals = []
                        total_match_bytes_vals = []
                        hit_rate_vals = []

                        verified = False
                        comp_size = None
                        comp_ratio = None

                        # File-based compress then decompress verify (restore original flow).
                        # Always use a temporary file for compressed output to avoid leaving artifacts.
                        tmpf = tempfile.NamedTemporaryFile(prefix=f'{safe_sample}_{cfg}_', suffix='_verify.lzo', delete=False)
                        verify_out = Path(tmpf.name)
                        tmpf.close()

                        comp_cmd = [lzo_bin, '-a', alg_name, '-L', str(b), '-B', block, sample_path, '-o', str(verify_out)]
                        if v_debug:
                            comp_cmd.insert(1,'--debug')
                        # copy the base extra env (contains LZO_KERNEL_OPT if opt variant)
                        extra_env = dict(extra_env_base)

                        # Add --verify flag to compression command for in-memory round-trip verification
                        comp_cmd.append('--verify')

                        try:
                            rc_comp, sout_comp, serr_comp = run_cmd(comp_cmd, timeout=TIMEOUT, extra_env=extra_env)
                        except Exception as e:
                            rc_comp = -1; sout_comp=''; serr_comp=str(e)

                        # compressed size and ratio (if file exists)
                        try:
                            if verify_out.exists():
                                comp_size = verify_out.stat().st_size
                                if size_bytes and comp_size:
                                    comp_ratio = float(size_bytes) / float(comp_size)
                        except Exception:
                            comp_size = None; comp_ratio = None

                        # Check if verification passed (now done by compression command with --verify)
                        comp_text = (sout_comp + '\n' + serr_comp).lower()
                        verified = (rc_comp == 0 and 'verify ok' in comp_text)

                        # parse metrics from the verification compress run (included as first sample)
                        output_combined = sout_comp + '\n' + serr_comp
                        k_ms, d_ms = parse_kernel_ms(output_combined)
                        k_thr, d_thr = parse_kernel_thr(output_combined)
                        gpu_blocks = parse_gpu_debug(serr_comp)
                        nblk = len(gpu_blocks)
                        total_lookups = sum(bk.get('lookups',0) for bk in gpu_blocks)
                        total_hits = sum(bk.get('hits',0) for bk in gpu_blocks)
                        total_match_bytes = sum(bk.get('matched_bytes',0) for bk in gpu_blocks)
                        hit_rate = (total_hits / total_lookups) if total_lookups else None

                        # If the kernel wrote a debug dump (sanity check failure), copy it into the output dir for inspection
                        dm = re.search(r'ERR: debug dump written to (\S+)', serr_comp)
                        if dm:
                            dump_path = dm.group(1)
                            try:
                                if out_base:
                                    import shutil
                                    dst = out_base / 'debug_dumps'
                                    dst.mkdir(parents=True, exist_ok=True)
                                    shutil.copy(dump_path, dst / f'{safe_sample}_{cfg}_dbg_dump.txt')
                                else:
                                    os.unlink(dump_path)
                            except Exception:
                                pass

                        # If no debug blocks were found but kernel indicates an instrumented build, do NOT save stderr (no traces policy)
                        if nblk == 0 and ('_debug' in (sout_comp + serr_comp) or '_opt_debug' in (sout_comp + serr_comp)):
                            pass

                        kernel_ms_vals.append(k_ms)
                        kernel_thr_vals.append(k_thr)
                        if d_ms: decomp_kernel_ms_vals.append(d_ms)
                        if d_thr: decomp_kernel_thr_vals.append(d_thr)
                        total_time_vals.append(None) # not parsing TOTAL here
                        nblk_vals.append(nblk)
                        total_lookups_vals.append(total_lookups)
                        total_hits_vals.append(total_hits)
                        total_match_bytes_vals.append(total_match_bytes)
                        hit_rate_vals.append(hit_rate)

                        if not verified:
                            # CPU Fallback Verification: determine if it's a compression or decompression issue
                            cpu_lzo = "/root/lzo-2.10/lzo_cpu/lzo_cpu"
                            failure_reason = "Unknown"
                            rc_cpu = -1
                            if os.path.exists(cpu_lzo) and verify_out.exists():
                                with tempfile.NamedTemporaryFile(prefix='cpu_dec_', suffix='.raw', delete=False) as tmp_cpu_dec:
                                    tmp_cpu_dec_path = tmp_cpu_dec.name

                                cpu_cmd = [cpu_lzo, '-d', '-a', alg_name, str(verify_out), '-o', tmp_cpu_dec_path]
                                rc_cpu, sout_cpu, serr_cpu = run_cmd(cpu_cmd)

                                if rc_cpu == 0 and os.path.exists(tmp_cpu_dec_path) and filecmp.cmp(sample_path, tmp_cpu_dec_path, shallow=False):
                                    failure_reason = "GPU_DECOMP_ISSUE"
                                else:
                                    failure_reason = "GPU_COMP_ISSUE"

                                if os.path.exists(tmp_cpu_dec_path): os.unlink(tmp_cpu_dec_path)

                            # Record failure (write failures CSV only if out_dir requested; otherwise keep zero-trace)
                            try:
                                if out_base:
                                    failures_file = out_base / 'verify_failures.csv'
                                    write_header = not failures_file.exists()
                                    with open(failures_file, 'a', newline='') as ff:
                                        if write_header:
                                            ff.write('timestamp,sample,sample_path,block,bits,variant,comp_size_bytes,compression_ratio,rc_comp,rc_cpu,failure_reason\n')
                                        ts = datetime.datetime.utcnow().isoformat() + 'Z'
                                        ff.write(f'{ts},{sample_basename},"{sample_path}",{block},{b},{v},{comp_size or ""},{comp_ratio or ""},{rc_comp},{rc_cpu},{failure_reason}\n')

                                    # Also write a more detailed log for human inspection
                                    with open(out_base / 'verify_failures.log', 'a') as fl:
                                        fl.write(f"[{ts}] FAILURE: {sample_basename} {cfg}\n")
                                        fl.write(f"Reason: {failure_reason}\n")
                                        fl.write(f"GPU Return Code: {rc_comp}\n")
                                        fl.write(f"CPU Return Code: {rc_cpu}\n")
                                        fl.write(f"GPU Stderr: {serr_comp[:1000]}\n")
                                        fl.write("-" * 40 + "\n")
                                else:
                                    # zero-trace mode: do not emit any textual traces (silent)
                                    pass
                            except Exception:
                                pass

                            # ensure compressed output file removed (never keep .lzo)
                            try:
                                if 'verify_out' in locals() and verify_out.exists(): verify_out.unlink()
                            except Exception:
                                pass

                            # compute medians from the single run
                            med_kernel_ms = median_or_none(kernel_ms_vals)
                            med_kernel_thr = median_or_none(kernel_thr_vals)
                            med_decomp_ms = median_or_none(decomp_kernel_ms_vals)
                            med_decomp_thr = median_or_none(decomp_kernel_thr_vals)
                            med_nblk = median_or_none(nblk_vals)
                            med_total_lookups = median_or_none(total_lookups_vals)
                            med_total_hits = median_or_none(total_hits_vals)
                            med_total_match_bytes = median_or_none(total_match_bytes_vals)
                            med_hit_rate = median_or_none([h for h in hit_rate_vals if h is not None])

                            # write summary row and skip further repeats
                            block_bytes_row = parse_size_bytes(block)
                            block_actual_kb = None if med_nblk is None else (0 if med_nblk==0 else int(med_nblk * block_bytes_row / 1024))
                            row = [sample_basename, sample_path, size_bytes, block, block_actual_kb, b, v, med_kernel_ms, med_kernel_thr, med_decomp_ms, med_decomp_thr, None, med_nblk, med_total_lookups, med_total_hits, med_total_match_bytes, med_hit_rate, args.repeats, 'NO', comp_size, comp_ratio]
                            w.writerow(row)
                            fout.flush()
                            continue

                        # run the remaining repeats (we already counted the verification run as one)
                        for rnum in range(max(0, args.repeats - 1)):
                            # file-based repeat run (use temporary file to avoid persisting artifacts)
                            tmpf = tempfile.NamedTemporaryFile(prefix=f'{safe_sample}_{cfg}_r{rnum+1}_', suffix='.lzo', delete=False)
                            rfil = Path(tmpf.name)
                            tmpf.close()

                            cmd = [lzo_bin, '-a', alg_name, '-L', str(b), '-B', block, sample_path, '-o', str(rfil)]
                            if v_debug:
                                cmd.insert(1,'--debug')
                            extra_env = dict(extra_env_base)
                            try:
                                rc, sout, serr = run_cmd(cmd, timeout=TIMEOUT, extra_env=extra_env)
                            except Exception as e:
                                rc = -1; sout=''; serr=str(e)

                            # parse metrics
                            output_combined = sout + '\n' + serr
                            k_ms, d_ms = parse_kernel_ms(output_combined)
                            k_thr, d_thr = parse_kernel_thr(output_combined)
                            gpu_blocks = parse_gpu_debug(serr)
                            nblk = len(gpu_blocks)
                            total_lookups = sum(bk.get('lookups',0) for bk in gpu_blocks)
                            total_hits = sum(bk.get('hits',0) for bk in gpu_blocks)
                            total_match_bytes = sum(bk.get('matched_bytes',0) for bk in gpu_blocks)
                            hit_rate = (total_hits / total_lookups) if total_lookups else None

                            kernel_ms_vals.append(k_ms)
                            kernel_thr_vals.append(k_thr)
                            if d_ms: decomp_kernel_ms_vals.append(d_ms)
                            if d_thr: decomp_kernel_thr_vals.append(d_thr)
                            total_time_vals.append(None) # not parsing TOTAL here
                            nblk_vals.append(nblk)
                            total_lookups_vals.append(total_lookups)
                            total_hits_vals.append(total_hits)
                            total_match_bytes_vals.append(total_match_bytes)
                            hit_rate_vals.append(hit_rate)

                            # delete temporary repeat file if present
                            try:
                                if rfil.exists(): rfil.unlink()
                            except Exception:
                                pass

                        med_kernel_ms = median_or_none(kernel_ms_vals)
                        med_kernel_thr = median_or_none(kernel_thr_vals)
                        med_decomp_ms = median_or_none(decomp_kernel_ms_vals)
                        med_decomp_thr = median_or_none(decomp_kernel_thr_vals)
                        med_nblk = median_or_none(nblk_vals)
                        med_total_lookups = median_or_none(total_lookups_vals)
                        med_total_hits = median_or_none(total_hits_vals)
                        med_total_match_bytes = median_or_none(total_match_bytes_vals)
                        med_hit_rate = median_or_none([h for h in hit_rate_vals if h is not None])

                        # write summary row
                        block_bytes_row = parse_size_bytes(block)
                        block_actual_kb = None if med_nblk is None else (0 if med_nblk==0 else int(med_nblk * block_bytes_row / 1024))
                        row = [sample_basename, sample_path, size_bytes, block, block_actual_kb, b, v, med_kernel_ms, med_kernel_thr, med_decomp_ms, med_decomp_thr, None, med_nblk, med_total_lookups, med_total_hits, med_total_match_bytes, med_hit_rate, args.repeats, ('YES' if verified else 'NO'), comp_size, comp_ratio]
                        w.writerow(row)
                        fout.flush()
                        # cleanup temporary verify file to avoid leaving artifacts
                        try:
                            if 'verify_out' in locals() and verify_out.exists():
                                verify_out.unlink()
                        except Exception:
                            pass

    if out_base:
        print(f"Deep sweep complete; summary at {csv_path}")
        if close_fout:
            fout.close()
    else:
        print("Deep sweep complete; summary printed to stdout")

if __name__ == '__main__':
    main()
