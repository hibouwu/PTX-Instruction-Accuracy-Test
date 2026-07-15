#!/usr/bin/env python3
"""Offline, resumable launcher for the complete 512 GiB GB10 FP6 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import run_gb10_fp6_precheck as precheck
import run_gb10_ptx_accuracy as runner


ROOT = Path(__file__).resolve().parent
SHARD_COUNT = 16
DEFAULT_OUTPUT_DIR = ROOT / "results" / "fp6-full"
DEFAULT_PRECHECK_REPORT = ROOT / "results" / "fp6-precheck" / "precheck-report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or resume all 16 offline shards of the 512 GiB GB10 FP6 sweep."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--precheck-report", type=Path, default=DEFAULT_PRECHECK_REPORT)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--compat-dir", type=Path, default=Path("/usr/local/cuda-13.1/compat"))
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=SHARD_COUNT - 1)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--yes-large",
        action="store_true",
        help="confirm generation of up to approximately 512 GiB",
    )
    return parser.parse_args()


def expected_path(output_dir: Path, test: runner.Test, shard_index: int) -> Path:
    return (
        output_dir
        / runner.safe_name(test.name)
        / f"packed-b32__shard-{shard_index:05d}-of-{SHARD_COUNT:05d}.bin"
    )


def shard_complete(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> bool:
    start, count = runner.shard_slice(2**32, shard_index, SHARD_COUNT)
    for test in tests:
        path = expected_path(output_dir, test, shard_index)
        if not path.is_file():
            return False
        try:
            header = runner.read_header(path)
        except Exception:
            return False
        if (
            header["test_name"] != test.name
            or header["result_mask"] != 0xFFFF
            or header["total_records"] != 2**32
            or header["shard_start"] != start
            or header["shard_records"] != count
        ):
            return False
    return True


def selected_shards(args: argparse.Namespace) -> range:
    if not 0 <= args.start_shard <= args.end_shard < SHARD_COUNT:
        raise RuntimeError(
            f"require 0 <= start-shard <= end-shard < {SHARD_COUNT}"
        )
    return range(args.start_shard, args.end_shard + 1)


def competing_accuracy_processes() -> list[str]:
    matches: list[str] = []
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = entry.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        if not command:
            continue
        if "run_gb10_ptx_accuracy.py" in command or "/build/gb10_ptx_accuracy" in command:
            matches.append(f"pid={entry.parent.name} {command}")
    return matches


def load_precheck_report(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise RuntimeError(
            f"precheck report is missing: {path}; run run_gb10_fp6_precheck.py first"
        )
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("status") != "PASS":
        raise RuntimeError(f"precheck status is not PASS: {path}")
    reference = report.get("reference")
    if not isinstance(reference, dict) or reference.get("lanes") != 24_117_248:
        raise RuntimeError(f"precheck reference coverage is incomplete: {path}")
    return report, hashlib.sha256(raw).hexdigest()


def remove_stale_partials(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> None:
    for test in tests:
        partial = expected_path(output_dir, test, shard_index).with_suffix(".bin.partial")
        if partial.exists():
            runner.log(f"removing stale partial before resume: {partial}")
            partial.unlink()


def run_shard(
    output_dir: Path,
    nvcc: str,
    shard_index: int,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "run_gb10_ptx_accuracy.py"),
        "--tests",
        precheck.TEST_PATTERN,
        "--arch",
        precheck.ARCH,
        "--profile",
        "full",
        "--shard-count",
        str(SHARD_COUNT),
        "--shard-index",
        str(shard_index),
        "--output-dir",
        str(output_dir),
        "--yes-large",
        "--nvcc",
        nvcc,
    ]
    runner.log("+ " + " ".join(command))
    process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
    if process.returncode != 0:
        raise RuntimeError(f"full shard {shard_index} failed with exit code {process.returncode}")


def write_report(
    output_dir: Path,
    precheck_report: Path,
    precheck_sha256: str,
    tests: Sequence[runner.Test],
    elapsed: float,
) -> Path:
    binaries = [
        expected_path(output_dir, test, shard_index)
        for shard_index in range(SHARD_COUNT)
        for test in tests
    ]
    missing = [path for path in binaries if not path.is_file()]
    if missing:
        raise RuntimeError(f"full sweep is missing {len(missing)} binaries")
    partials = sorted(output_dir.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"full sweep has {len(partials)} incomplete .partial files")
    manifests = sorted(output_dir.glob("manifest-fp16x2_to_f6x2__shard-*.json"))
    if len(manifests) != SHARD_COUNT:
        raise RuntimeError(f"expected {SHARD_COUNT} manifests, found {len(manifests)}")
    report = {
        "status": "PASS",
        "instruction_family": "cvt.rn.satfinite{.relu}.{e2m3x2/e3m2x2}.{f16x2/bf16x2}",
        "arch": precheck.ARCH,
        "shard_count": SHARD_COUNT,
        "test_count": len(tests),
        "binary_count": len(binaries),
        "binary_bytes": sum(path.stat().st_size for path in binaries),
        "precheck_report": str(precheck_report),
        "precheck_report_sha256": precheck_sha256,
        "elapsed_seconds_this_invocation": elapsed,
    }
    path = output_dir / "full-run-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    args = parse_args()
    shards = selected_shards(args)
    tests = runner.select_tests([precheck.TEST_PATTERN])
    output_dir = args.output_dir.resolve()
    complete = [index for index in shards if shard_complete(output_dir, tests, index)]
    pending = [index for index in shards if index not in complete]
    bytes_per_shard = runner.projected_bytes(tests, "full", 0, SHARD_COUNT, None)

    print(f"requested shards: {args.start_shard}..{args.end_shard}")
    print(f"complete shards: {complete or 'none'}")
    print(f"pending shards: {pending or 'none'}")
    print(
        f"remaining output: {len(pending) * bytes_per_shard} bytes "
        f"({len(pending) * bytes_per_shard / 1024**3:.1f} GiB)"
    )
    if args.plan:
        return
    if not args.yes_large:
        raise RuntimeError("inspect --plan, then pass --yes-large to start or resume")

    competing = competing_accuracy_processes()
    if competing:
        details = "\n".join(competing)
        raise RuntimeError(
            "another GB10 accuracy process is already running; do not start two "
            f"writers for the same output tree:\n{details}"
        )

    _precheck, precheck_sha256 = load_precheck_report(args.precheck_report.resolve())
    nvcc = Path(args.nvcc)
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc not found: {nvcc}")
    precheck.configure_compatibility(args.compat_dir.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    required = len(pending) * bytes_per_shard
    available = shutil.disk_usage(output_dir).free
    if required > available:
        raise RuntimeError(
            f"remaining shards need {required} bytes but only {available} bytes are free"
        )

    started = time.time()
    for position, shard_index in enumerate(pending, 1):
        runner.log(
            f"=== full shard {shard_index}/{SHARD_COUNT - 1} "
            f"(pending {position}/{len(pending)}) ==="
        )
        remove_stale_partials(output_dir, tests, shard_index)
        run_shard(output_dir, str(nvcc), shard_index)
        if not shard_complete(output_dir, tests, shard_index):
            raise RuntimeError(f"full shard {shard_index} did not pass post-run validation")
        runner.log(f"FULL SHARD PASS {shard_index}/{SHARD_COUNT - 1}")

    all_complete = all(
        shard_complete(output_dir, tests, index) for index in range(SHARD_COUNT)
    )
    if not all_complete:
        runner.log("requested shard range passed; other shards remain incomplete")
        return
    report = write_report(
        output_dir,
        args.precheck_report.resolve(),
        precheck_sha256,
        tests,
        time.time() - started,
    )
    runner.log("FULL SWEEP PASS: 128 binaries, 16 shards, 8 concrete PTX instructions")
    runner.log(f"report: {report}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
