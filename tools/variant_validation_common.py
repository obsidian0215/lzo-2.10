from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SAMPLES_ROOT = Path("/root/samples")

BENCH_COMP_RE = re.compile(
    r"Bench\s+Compress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+ratio=([0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)
BENCH_DEC_RE = re.compile(
    r"Bench\s+Decompress\s*:\s*kernel_tp=([0-9]+(?:\.[0-9]+)?)\s*MB/s\s+verify=(OK|FAIL)",
    re.IGNORECASE,
)
BENCH_ADAPTIVE_MEAN_RE = re.compile(
    r"Bench\s+Adaptive\s*:\s*gpu_ratio_mean=([0-9]+(?:\.[0-9]+)?)\s+min=([0-9]+(?:\.[0-9]+)?)\s+max=([0-9]+(?:\.[0-9]+)?)\s+samples=(\d+)",
    re.IGNORECASE,
)
BENCH_ADAPTIVE_SINGLE_RE = re.compile(
    r"Bench\s+Adaptive\s*:\s*gpu_ratio=([0-9]+(?:\.[0-9]+)?)\s+objective=([^\s]+)\s+min=([0-9]+(?:\.[0-9]+)?)\s+max=([0-9]+(?:\.[0-9]+)?)\s+samples=(\d+)",
    re.IGNORECASE,
)
TIMING_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9 _/\-\(\)]+?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*\([^\)]*\))?\s*$",
    re.MULTILINE,
)
INLINE_RATIO_RE = re.compile(r"\(([0-9]+(?:\.[0-9]+)?)%\s*ratio\)", re.IGNORECASE)
INCLUSIVE_TP_RE = re.compile(r"Inclusive\s+Throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MB/s", re.IGNORECASE)
KERNEL_TP_RE = re.compile(r"Kernel\s+Throughput\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*MB/s", re.IGNORECASE)
ADAPTIVE_TRACE_RE = re.compile(r"\br\*=([0-9]+(?:\.[0-9]+)?)")


class ValidationError(RuntimeError):
    """Raised when validation inputs or outputs violate the contract."""


@dataclass(slots=True)
class CommandResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    wall_s: float

    @property
    def combined_output(self) -> str:
        return f"{self.stdout or ''}\n{self.stderr or ''}".strip()


@dataclass(slots=True)
class VariantManifest:
    manifest: dict[str, Any]
    manifest_path: Path
    variant_dir: Path

    @property
    def id(self) -> str:
        return str(self.manifest.get("id") or self.variant_dir.name)

    @property
    def component(self) -> str:
        return str(self.manifest.get("component") or "")

    @property
    def optimization_object(self) -> str:
        return str(self.manifest.get("optimization_object") or "")


def utc_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def parse_timing_ms_map(text: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for raw_label, raw_value in TIMING_LINE_RE.findall(text or ""):
        try:
            timings[normalize_label(raw_label)] = float(raw_value)
        except ValueError:
            continue
    return timings


def parse_bench_output(text: str) -> dict[str, Any]:
    comp_match = BENCH_COMP_RE.search(text or "")
    dec_match = BENCH_DEC_RE.search(text or "")
    if not comp_match or not dec_match:
        raise ValidationError("bench output missing required Bench Compress/Decompress lines")

    payload: dict[str, Any] = {
        "bench_comp_kernel_mbs": float(comp_match.group(1)),
        "bench_ratio_pct": float(comp_match.group(2)),
        "bench_dec_kernel_mbs": float(dec_match.group(1)),
        "bench_verify_ok": dec_match.group(2).upper() == "OK",
    }

    adaptive_mean = BENCH_ADAPTIVE_MEAN_RE.search(text or "")
    adaptive_single = BENCH_ADAPTIVE_SINGLE_RE.search(text or "")
    if adaptive_mean:
        payload.update(
            {
                "adaptive_gpu_ratio_mean": float(adaptive_mean.group(1)),
                "adaptive_gpu_ratio_min": float(adaptive_mean.group(2)),
                "adaptive_gpu_ratio_max": float(adaptive_mean.group(3)),
                "adaptive_samples": int(adaptive_mean.group(4)),
            }
        )
    elif adaptive_single:
        payload.update(
            {
                "adaptive_gpu_ratio_mean": float(adaptive_single.group(1)),
                "adaptive_objective": adaptive_single.group(2),
                "adaptive_gpu_ratio_min": float(adaptive_single.group(3)),
                "adaptive_gpu_ratio_max": float(adaptive_single.group(4)),
                "adaptive_samples": int(adaptive_single.group(5)),
            }
        )
    return payload


def parse_inline_ratio_percent(text: str) -> float | None:
    match = INLINE_RATIO_RE.search(text or "")
    if not match:
        return None
    return float(match.group(1))


def parse_inclusive_throughput(text: str) -> float | None:
    match = INCLUSIVE_TP_RE.search(text or "")
    if not match:
        return None
    return float(match.group(1))


def parse_kernel_throughput(text: str) -> float | None:
    match = KERNEL_TP_RE.search(text or "")
    if not match:
        return None
    return float(match.group(1))


def parse_adaptive_trace_ratios(text: str) -> list[float]:
    return [float(value) for value in ADAPTIVE_TRACE_RE.findall(text or "")]


def safe_mean(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.mean(cleaned))


def safe_median(values: Iterable[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directional_delta_pct(candidate: float | None, baseline: float | None, *, higher_is_better: bool) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline == 0:
        if candidate == 0:
            return 0.0
        return None
    if higher_is_better:
        return ((candidate - baseline) / abs(baseline)) * 100.0
    return ((baseline - candidate) / abs(baseline)) * 100.0


def summarize_directional_deltas(deltas: Iterable[float | None]) -> dict[str, Any]:
    cleaned = [float(value) for value in deltas if value is not None]
    if not cleaned:
        return {"avg": None, "pos": 0, "neg": 0, "neg_worst": None}
    negative = [value for value in cleaned if value < 0.0]
    return {
        "avg": safe_mean(cleaned),
        "pos": sum(1 for value in cleaned if value > 0.0),
        "neg": sum(1 for value in cleaned if value < 0.0),
        "neg_worst": min(negative) if negative else 0.0,
    }


def resolve_variant_manifest(intel_root: Path, variant_ref: str, component: str | None = None) -> VariantManifest:
    maybe_path = Path(variant_ref)
    candidates: list[Path] = []

    if maybe_path.is_absolute() or maybe_path.parts:
        if maybe_path.exists() and maybe_path.is_file():
            manifest_path = maybe_path
            return VariantManifest(load_json(manifest_path), manifest_path, manifest_path.parent)

    if component:
        candidate_path = intel_root / "variants" / component / variant_ref / "variant.json"
        if candidate_path.exists():
            candidates.append(candidate_path)
    else:
        candidates.extend(sorted((intel_root / "variants").glob(f"*/{variant_ref}/variant.json")))

    if not candidates:
        raise ValidationError(f"variant manifest not found for '{variant_ref}' under {intel_root}")
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValidationError(f"variant '{variant_ref}' is ambiguous: {joined}")

    manifest_path = candidates[0]
    return VariantManifest(load_json(manifest_path), manifest_path, manifest_path.parent)


def resolve_path(path_value: str, *roots: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    for root in roots:
        path = root / candidate
        if path.exists():
            return path
    return roots[0] / candidate


def resolve_binary_path(variant: VariantManifest, repo_root: Path) -> Path:
    binary_relpath = variant.manifest.get("binary_relpath")
    if not binary_relpath:
        raise ValidationError(f"variant '{variant.id}' is missing binary_relpath")
    binary_path = resolve_path(str(binary_relpath), variant.variant_dir, repo_root)
    if not binary_path.exists():
        raise ValidationError(f"binary_relpath for variant '{variant.id}' does not exist: {binary_path}")
    return binary_path


def detect_variant_artifacts(variant: VariantManifest) -> list[str]:
    artifacts = [str(item) for item in variant.manifest.get("kernel_artifacts", []) if str(item).strip()]
    if artifacts:
        return artifacts
    detected = []
    for child in sorted(variant.variant_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in {".cl", ".clbin", ".h"}:
            detected.append(child.name)
    return detected


def prepare_runtime_overlay(
    *,
    default_runtime_dir: Path,
    variant: VariantManifest,
    work_root: Path,
) -> Path:
    artifacts = detect_variant_artifacts(variant)
    if not artifacts:
        return default_runtime_dir

    runtime_dir = ensure_dir(work_root / f"{variant.id}_runtime")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    default_runtime_dir = default_runtime_dir.resolve()
    work_root = work_root.resolve()
    runtime_dir = runtime_dir.resolve()

    def _ignore_copytree(src: str, names: list[str]) -> list[str]:
        ignored: list[str] = []
        src_path = Path(src).resolve()
        for name in names:
            candidate = (src_path / name).resolve()
            if name == "variant_validation":
                ignored.append(name)
                continue
            if candidate == work_root or work_root in candidate.parents:
                ignored.append(name)
                continue
            if candidate == runtime_dir or runtime_dir in candidate.parents:
                ignored.append(name)
                continue
        return ignored

    shutil.copytree(default_runtime_dir, runtime_dir, ignore=_ignore_copytree)

    for artifact in artifacts:
        source = resolve_path(artifact, variant.variant_dir)
        if not source.exists() or not source.is_file():
            raise ValidationError(f"artifact '{artifact}' for variant '{variant.id}' does not exist")
        shutil.copy2(source, runtime_dir / source.name)

    return runtime_dir


def discover_samples(
    *,
    samples_root: Path,
    sample_list: Path | None = None,
    sample_glob: str | None = None,
) -> list[Path]:
    if not samples_root.exists():
        raise ValidationError(f"samples root does not exist: {samples_root}")

    samples: list[Path] = []
    if sample_list:
        if not sample_list.exists():
            raise ValidationError(f"sample list file does not exist: {sample_list}")
        for raw_line in sample_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("/root/samples/"):
                line = str(samples_root / line.removeprefix("/root/samples/"))
            path = Path(line)
            if path.is_absolute() and (not path.exists() or not path.is_file()):
                try:
                    rel = path.relative_to(DEFAULT_SAMPLES_ROOT)
                    path = samples_root / rel
                except ValueError:
                    pass
            if not path.is_absolute():
                path = samples_root / path
            if not path.exists() or not path.is_file():
                raise ValidationError(f"sample listed in {sample_list} does not exist: {path}")
            samples.append(path.resolve())
    elif sample_glob:
        samples.extend(path.resolve() for path in sorted(samples_root.glob(sample_glob)) if path.is_file())
    else:
        samples.extend(path.resolve() for path in sorted(samples_root.rglob("*")) if path.is_file())

    if not samples:
        raise ValidationError(f"no samples discovered under {samples_root}")

    samples = sorted(dict.fromkeys(samples), key=lambda item: str(item.relative_to(samples_root.resolve())))
    return samples


def relative_sample_name(sample: Path, samples_root: Path) -> str:
    try:
        return sample.resolve().relative_to(samples_root.resolve()).as_posix()
    except Exception:
        return sample.name


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})

    start = time.perf_counter()
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    wall_s = max(0.0, time.perf_counter() - start)
    return CommandResult(list(cmd), completed.returncode, completed.stdout or "", completed.stderr or "", wall_s)


def build_metric_status(candidate_summary: Mapping[str, Any], control_summary: Mapping[str, Any]) -> str:
    candidate_avg = candidate_summary.get("avg")
    control_avg = control_summary.get("avg")
    if candidate_avg is None:
        return "reject"
    noise_floor = abs(float(control_avg)) if control_avg is not None else 0.0
    if float(candidate_avg) > max(noise_floor, 0.0):
        return "watch"
    return "reject"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def merge_env(base: Mapping[str, str] | None, extra: Mapping[str, Any] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    if base:
        merged.update({str(key): str(value) for key, value in base.items()})
    if extra:
        merged.update({str(key): str(value) for key, value in extra.items() if value is not None})
    return merged
