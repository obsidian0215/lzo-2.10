#!/usr/bin/env python3
"""
Aggregate per-level `EXPERIMENT_SUMMARY.md` files into a single combined report.
Usage: python3 tools/aggregate_experiment_markdowns.py [out_path]
Defaults to `lzo_gpu/experiments/logs/EXPERIMENT_SUMMARY_ALL.md`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / 'lzo_gpu' / 'experiments' / 'logs' / 'param_scans'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'lzo_gpu' / 'experiments' / 'logs' / 'EXPERIMENT_SUMMARY_ALL.md'

parts = []
parts.append('# Combined Parameter Scan Report')
parts.append('')
parts.append('This file aggregates per-level experiment summaries.')
parts.append('')

# Look for directories under param_scans (e.g. comp_<level>/decomp_<mode>/...)
if PARAM_DIR.exists():
    # collect experiment summary files under the param_scans tree
    md_files = sorted(PARAM_DIR.rglob('EXPERIMENT_SUMMARY.md'))
    if not md_files:
        parts.append('_No per-level EXPERIMENT_SUMMARY.md files found under param_scans._')
    else:
        for md in md_files:
            rel = md.relative_to(ROOT)
            parts.append(f"## {md.parent.name}")
            parts.append('')
            parts.append(f"Source: `{rel}`")
            parts.append('')
            try:
                content = md.read_text()
            except Exception:
                content = '_Failed to read file_'
            parts.append('```')
            parts.append(content)
            parts.append('```')
            parts.append('\n---\n')
else:
    parts.append('_param_scans directory not found; run param-scan first._')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(parts))
print('Wrote', OUT)
