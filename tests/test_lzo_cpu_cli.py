#!/usr/bin/env python3
"""Smoke tests for the lzo_cpu CLI driver.

The script exercises compression and decompression across a handful of
compression levels and thread counts, verifies round-trip integrity, and
surfaces basic regressions when behaviour diverges from expectations.

Usage examples:
    python tests/test_lzo_cpu_cli.py
    python tests/test_lzo_cpu_cli.py --cli build\\lzo_frag.exe --levels 1,2,3,4

By default the script tries to locate the CLI binary at
    lzo_cpu/lzo_frag[.exe]
relative to the repository root. Override with --cli when the binary lives
elsewhere. Set LZO_CPU_CLI in the environment to provide a default override.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI_CANDIDATES = (
    REPO_ROOT / "lzo_cpu" / "lzo_frag.exe",
    REPO_ROOT / "lzo_cpu" / "lzo_frag",
)


class CLIError(RuntimeError):
    """Raised when the CLI returns a failing exit status."""


def discover_cli(explicit: str | None) -> Path:
    if explicit:
        cli_path = Path(explicit).resolve()
        if cli_path.exists():
            return cli_path
        raise FileNotFoundError(f"Specified CLI '{cli_path}' does not exist")

    env_override = os.environ.get("LZO_CPU_CLI")
    if env_override:
        cli_path = Path(env_override).resolve()
        if cli_path.exists():
            return cli_path

    for candidate in DEFAULT_CLI_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate lzo_frag binary. Provide --cli or set LZO_CPU_CLI."
    )


def run_cli(cli: Path, args: Iterable[str], *, expect_success: bool = True) -> None:
    cmd: List[str] = [str(cli), *args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if expect_success and proc.returncode != 0:
        raise CLIError(
            "Command failed with code {}\ncmd: {}\nstdout:\n{}\nstderr:\n{}".format(
                proc.returncode,
                " ".join(cmd),
                proc.stdout.decode(errors="ignore"),
                proc.stderr.decode(errors="ignore"),
            )
        )


def make_fixture(tmpdir: Path) -> Path:
    payload = bytearray()
    for i in range(256):
        payload.extend(bytes([i]) * (i + 1))
    payload.extend(b"LZO CPU CLI smoke test\n")
    payload.extend(os.urandom(4096))
    fixture_path = tmpdir / "fixture.bin"
    fixture_path.write_bytes(payload)
    return fixture_path


def roundtrip(
    cli: Path,
    fixture: Path,
    tmpdir: Path,
    level: int,
    threads: int,
    benchmark: bool,
) -> None:
    suffix = f"l{level}_t{threads}"
    compressed = tmpdir / f"{fixture.name}.{suffix}.lzo"
    restored = tmpdir / f"{fixture.name}.{suffix}.out"

    compress_args = [f"-{level}", "-t", str(threads)]
    if benchmark:
        compress_args.append("--benchmark")
    compress_args.extend([str(fixture), str(compressed)])
    run_cli(cli, compress_args)

    if not compressed.exists():
        raise AssertionError(f"Expected output '{compressed}' was not created")

    decompress_args = ["-d", "-t", str(threads), str(compressed), str(restored)]
    run_cli(cli, decompress_args)

    original = fixture.read_bytes()
    recovered = restored.read_bytes()
    if original != recovered:
        raise AssertionError(
            f"Content mismatch for level {level}, threads {threads}: {restored}"
        )


def parse_csv_ints(value: str) -> List[int]:
    result: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise argparse.ArgumentTypeError(f"Invalid integer '{item}'") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"Non-positive integer '{parsed}' is not allowed"
            )
        result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("List may not be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cli",
        help="Path to the lzo_frag executable (defaults to lzo_cpu/lzo_frag[.exe])",
    )
    parser.add_argument(
        "--levels",
        type=parse_csv_ints,
        default=[1, 3, 4],
        help="Comma-separated compression levels to exercise (default: 1,3,4)",
    )
    parser.add_argument(
        "--threads",
        type=parse_csv_ints,
        default=[1, 4],
        help="Comma-separated thread counts to exercise (default: 1,4)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable --benchmark flag during compression runs",
    )
    parser.add_argument(
        "--keep-temporary",
        action="store_true",
        help="Preserve the working directory for inspection",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cli_path = discover_cli(args.cli)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    workdir_obj = tempfile.TemporaryDirectory() if not args.keep_temporary else None
    workdir = (
        Path(workdir_obj.name)
        if workdir_obj is not None
        else Path(tempfile.mkdtemp(prefix="lzo_cpu_cli_"))
    )

    print(f"Using CLI: {cli_path}")
    print(f"Working directory: {workdir}")

    fixture = make_fixture(workdir)

    try:
        for level in args.levels:
            if level not in (1, 2, 3, 4):
                raise ValueError(f"Unsupported compression level requested: {level}")
            for threads in args.threads:
                print(f"- Roundtrip level={level} threads={threads}")
                roundtrip(cli_path, fixture, workdir, level, threads, args.benchmark)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep_temporary:
            print(f"Temporary files retained in {workdir}")
        elif workdir_obj is None:
            shutil.rmtree(workdir, ignore_errors=True)

    print("All roundtrip checks passed.")
    if workdir_obj is not None:
        workdir_obj.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
