#!/usr/bin/env python3
"""Precheck, run, resume, and report the feasible GB10 conversion sweeps.

This single operational entry point covers the CUDA 13.1 conversion families
whose complete README-defined input spaces fit on the GB10 test machine.  It
reuses the generic CUDA generator/runner, but owns the complete workflow:

    precheck -> plan -> full/resume -> report

The global strided full capture is only a few MiB.  It is a structurally validated GB10
golden capture; the bounded precheck proves JIT execution and determinism, not
agreement with an independent numerical reference model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import run_gb10_ptx_accuracy as runner
import run_gb10_all_strided as integrity


ROOT = Path(__file__).resolve().parent
# The S2F6 conversions are SM121 application-specific.  A family-specific
# compute_120f/compute_121f target is intentionally rejected by ptxas.
ARCH = "compute_121a"
SHARD_COUNT = 16
TEST_PATTERNS = (
    "fp16x2_to_f4x2*",
    "bf16x2_to_ue8m0x2*",
    "bf16x2_to_s2f6x2*",
    "f6x2_to_f16x2*",
    "f4x2_to_f16x2*",
    "s2f6x2_to_bf16x2_scaled*",
    "ue8m0x2_to_bf16x2*",
)
EXPECTED_TESTS = 20
DEFAULT_PRECHECK_DIR = ROOT / "results" / "bounded-conversions-strided-precheck"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "bounded-conversions-strided-full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the globally strided full sweep for 20 bounded GB10 "
            "FP4/UE8M0/S2F6 and inverse conversion instructions."
        )
    )
    parser.add_argument("command", choices=("precheck", "plan", "full", "seal", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--precheck-dir", type=Path, default=DEFAULT_PRECHECK_DIR)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument(
        "--compat-dir",
        type=Path,
        default=Path("/usr/local/cuda-13.1/compat"),
    )
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=SHARD_COUNT - 1)
    parser.add_argument(
        "--overwrite-precheck",
        action="store_true",
        help="replace an existing bounded-conversion precheck directory",
    )
    parser.add_argument(
        "--yes-large",
        action="store_true",
        help="confirm generation of the complete globally strided golden data",
    )
    return parser.parse_args()


def selected_tests() -> list[runner.Test]:
    tests = runner.select_tests(TEST_PATTERNS)
    if len(tests) != EXPECTED_TESTS:
        raise RuntimeError(
            f"bounded conversion matrix changed: expected {EXPECTED_TESTS} tests, "
            f"found {len(tests)}"
        )
    if any(test.min_cuda > (13, 1) for test in tests):
        raise RuntimeError("bounded conversion selection unexpectedly requires CUDA 13.2+")
    if any(len(test.sweeps) != 1 for test in tests):
        raise RuntimeError("bounded conversion workflow requires exactly one sweep per test")
    return tests


def configure_compatibility(compat_dir: Path) -> None:
    if not compat_dir.is_dir():
        raise RuntimeError(
            f"CUDA compatibility directory is missing: {compat_dir}; "
            "install cuda-compat-13-1 before running PTX 9.1 instructions"
        )
    if not any(compat_dir.glob("libnvidia-ptxjitcompiler.so*")):
        raise RuntimeError(f"PTX JIT compatibility library is missing from {compat_dir}")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(":") if entry]
    compat = str(compat_dir)
    if compat not in entries:
        os.environ["LD_LIBRARY_PATH"] = ":".join([compat, *entries])
    runner.log(f"using CUDA compatibility libraries: {compat}")


def selected_shards(args: argparse.Namespace) -> range:
    if not 0 <= args.start_shard <= args.end_shard < SHARD_COUNT:
        raise RuntimeError(
            f"require 0 <= start-shard <= end-shard < {SHARD_COUNT}"
        )
    return range(args.start_shard, args.end_shard + 1)


def build_runner(tests: Sequence[runner.Test], nvcc: str) -> Path:
    source = runner.generate_cuda(tests, runner.DEFAULT_GENERATED_DIR)
    binary = runner.DEFAULT_BUILD_DIR / "gb10_bounded_conversions"
    runner.log(f"generated {source} ({len(tests)} concrete instructions)")
    runner.compile_cuda(source, binary, tests, nvcc, ARCH)
    runner.log(f"built {binary}")
    runner.preflight_tests(binary, tests)
    return binary


def capture_smoke(
    binary: Path,
    tests: Sequence[runner.Test],
    output_dir: Path,
) -> list[dict[str, object]]:
    summaries = runner.execute_tests(
        binary,
        tests,
        output_dir,
        None,
        "smoke",
        0,
        1,
        1_048_576,
        None,
    )
    runner.write_manifest(output_dir, tests, "smoke", 0, 1, summaries)
    return summaries


def compare_capture_trees(
    tests: Sequence[runner.Test],
    baseline: Path,
    repeat: Path,
) -> tuple[int, int]:
    files = 0
    bytes_compared = 0
    for test in tests:
        sweep = test.sweeps[0].smoke()
        relative = (
            Path(runner.safe_name(test.name))
            / f"{runner.safe_name(sweep.name)}__shard-00000-of-00001.bin"
        )
        first = baseline / relative
        second = repeat / relative
        runner.compare_binary(second, first)
        files += 1
        bytes_compared += first.stat().st_size
    return files, bytes_compared


def run_precheck(args: argparse.Namespace, tests: Sequence[runner.Test]) -> Path:
    output_dir = args.precheck_dir.resolve()
    if output_dir.exists():
        if not args.overwrite_precheck:
            raise RuntimeError(
                f"precheck output already exists: {output_dir}; "
                "pass --overwrite-precheck to replace it"
            )
        protected = {Path("/"), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
        if output_dir in protected or len(output_dir.parts) < 4:
            raise RuntimeError(f"refusing to remove unsafe precheck path: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started = time.time()
    binary = build_runner(tests, args.nvcc)
    baseline = output_dir / "baseline"
    repeat = output_dir / "repeat"
    capture_smoke(binary, tests, baseline)
    capture_smoke(binary, tests, repeat)
    files, bytes_compared = compare_capture_trees(tests, baseline, repeat)
    report = {
        "status": "PASS",
        "validation_scope": "JIT, binary structure, and repeatability; no independent numerical reference",
        "arch": ARCH,
        "compat_dir": str(args.compat_dir.resolve()),
        "test_count": len(tests),
        "tests": [test.name for test in tests],
        "determinism_files": files,
        "determinism_bytes": bytes_compared,
        "elapsed_seconds": time.time() - started,
    }
    report_path = output_dir / "precheck-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runner.log(
        f"PRECHECK PASS: {len(tests)} PTX preflights and {files} deterministic "
        "smoke binaries"
    )
    runner.log(f"report: {report_path}")
    return binary


def precheck_evidence(
    path: Path,
    tests: Sequence[runner.Test],
) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise RuntimeError(
            f"precheck report is missing: {path}; run "
            "run_gb10_bounded_conversions.py precheck first"
        )
    raw = path.read_bytes()
    report = json.loads(raw)
    expected_names = [test.name for test in tests]
    if (
        report.get("status") != "PASS"
        or report.get("test_count") != EXPECTED_TESTS
        or report.get("tests") != expected_names
    ):
        raise RuntimeError(f"precheck report is not a complete PASS: {path}")
    return report, hashlib.sha256(raw).hexdigest()


def expected_path(
    output_dir: Path,
    test: runner.Test,
    shard_index: int,
) -> Path:
    sweep = test.sweeps[0]
    return (
        output_dir
        / runner.safe_name(test.name)
        / (
            f"{runner.safe_name(sweep.name)}__"
            f"shard-{shard_index:05d}-of-{SHARD_COUNT:05d}.bin"
        )
    )


def shard_complete(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> bool:
    try:
        integrity.validate_shard_manifest(output_dir, tests, shard_index)
    except Exception:
        return False
    return True


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


def competing_accuracy_processes() -> list[str]:
    matches: list[str] = []
    own_pid = str(os.getpid())
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        if entry.parent.name == own_pid:
            continue
        try:
            command = entry.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not command:
            continue
        if "run_gb10_ptx_accuracy.py" in command or "/build/gb10_" in command:
            matches.append(f"pid={entry.parent.name} {command}")
    return matches


def print_plan(
    args: argparse.Namespace,
    tests: Sequence[runner.Test],
) -> tuple[list[int], list[int], int]:
    output_dir = args.output_dir.resolve()
    shards = selected_shards(args)
    complete = [index for index in shards if shard_complete(output_dir, tests, index)]
    pending = [index for index in shards if index not in complete]
    remaining = sum(
        runner.projected_bytes(tests, "full", index, SHARD_COUNT, None)
        for index in pending
    )
    total = sum(
        runner.projected_bytes(tests, "full", index, SHARD_COUNT, None)
        for index in range(SHARD_COUNT)
    )
    print(f"concrete instructions: {len(tests)}")
    print(f"requested shards: {args.start_shard}..{args.end_shard}")
    print(f"complete shards: {complete or 'none'}")
    print(f"pending shards: {pending or 'none'}")
    print(f"complete sweep: {total} bytes ({total / 1024**3:.3f} GiB)")
    print(f"remaining output: {remaining} bytes ({remaining / 1024**3:.3f} GiB)")
    return complete, pending, remaining


def run_one_shard(
    binary: Path,
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> None:
    summaries = runner.execute_tests(
        binary,
        tests,
        output_dir,
        None,
        "full",
        shard_index,
        SHARD_COUNT,
        1_048_576,
        None,
    )
    runner.write_manifest(
        output_dir,
        tests,
        "full",
        shard_index,
        SHARD_COUNT,
        summaries,
    )


def validate_all(
    output_dir: Path,
    tests: Sequence[runner.Test],
) -> tuple[list[Path], list[Path]]:
    return integrity.validate_all(output_dir, tests)


def write_report(
    output_dir: Path,
    precheck_report: Path,
    precheck_sha256: str,
    tests: Sequence[runner.Test],
    elapsed: float,
) -> Path:
    binaries, manifests = validate_all(output_dir, tests)
    report = {
        "status": "CAPTURE_COMPLETE",
        "capture_status": "PASS",
        "accuracy_status": "NOT_INDEPENDENTLY_VALIDATED",
        "result_kind": "GB10 golden capture with structural validation",
        "independent_numerical_reference": False,
        "arch": ARCH,
        "shard_count": SHARD_COUNT,
        "test_count": len(tests),
        "tests": [{"name": test.name, "ptx": test.ptx} for test in tests],
        "binary_count": len(binaries),
        "binary_bytes": sum(path.stat().st_size for path in binaries),
        "manifest_count": len(manifests),
        "matrix_sha256": runner.matrix_sha256(tests),
        "precheck_report": str(precheck_report),
        "precheck_report_sha256": precheck_sha256,
        "elapsed_seconds_this_invocation": elapsed,
    }
    path = output_dir / "full-run-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    args = parse_args()
    tests = selected_tests()
    output_dir = args.output_dir.resolve()
    precheck_dir = args.precheck_dir.resolve()
    if (
        output_dir == precheck_dir
        or output_dir in precheck_dir.parents
        or precheck_dir in output_dir.parents
    ):
        raise RuntimeError("output-dir and precheck-dir must be separate, non-nested trees")

    if args.command == "precheck":
        configure_compatibility(args.compat_dir.resolve())
        run_precheck(args, tests)
        return

    precheck_path = precheck_dir / "precheck-report.json"
    if args.command == "seal":
        _precheck, precheck_sha256 = precheck_evidence(precheck_path, tests)
        for shard_index in range(SHARD_COUNT):
            integrity.seal_shard_manifest(output_dir, tests, shard_index)
        report = write_report(output_dir, precheck_path, precheck_sha256, tests, 0.0)
        runner.log("CAPTURE COMPLETE: bounded results sealed")
        runner.log(f"report: {report}")
        return

    _complete, pending, remaining = print_plan(args, tests)
    if args.command == "plan":
        return

    prebuilt_binary: Path | None = None
    if args.command == "full":
        unsealed = integrity.unsealed_manifest_shards(output_dir, tests)
        if unsealed:
            raise RuntimeError("existing legacy manifests must be sealed before resume")
        if not args.yes_large:
            raise RuntimeError("inspect the plan, then pass --yes-large to start or resume")
        competing = competing_accuracy_processes()
        if competing:
            raise RuntimeError(
                "another GB10 accuracy process is running; do not start two writers:\n"
                + "\n".join(competing)
            )
        nvcc = Path(args.nvcc)
        if not nvcc.is_file():
            raise RuntimeError(f"nvcc not found: {nvcc}")
        output_dir.mkdir(parents=True, exist_ok=True)
        available = shutil.disk_usage(output_dir).free
        if remaining > available:
            raise RuntimeError(
                f"pending shards need {remaining} bytes but only {available} bytes are free"
            )
        configure_compatibility(args.compat_dir.resolve())
        if not precheck_path.is_file():
            runner.log("bounded precheck report is missing; running precheck automatically")
            prebuilt_binary = run_precheck(args, tests)
    _precheck, precheck_sha256 = precheck_evidence(precheck_path, tests)

    if args.command == "report":
        report = write_report(output_dir, precheck_path, precheck_sha256, tests, 0.0)
        runner.log(f"CAPTURE COMPLETE: {len(tests) * SHARD_COUNT} binaries")
        runner.log(f"report: {report}")
        return

    started = time.time()
    if pending:
        binary = prebuilt_binary or build_runner(tests, str(nvcc))
        for position, shard_index in enumerate(pending, 1):
            runner.log(
                f"=== bounded full shard {shard_index}/{SHARD_COUNT - 1} "
                f"(pending {position}/{len(pending)}) ==="
            )
            remove_stale_partials(output_dir, tests, shard_index)
            run_one_shard(binary, output_dir, tests, shard_index)
            if not shard_complete(output_dir, tests, shard_index):
                raise RuntimeError(f"shard {shard_index} failed post-run validation")
            runner.log(f"FULL SHARD PASS {shard_index}/{SHARD_COUNT - 1}")

    if not all(shard_complete(output_dir, tests, index) for index in range(SHARD_COUNT)):
        runner.log("requested shard range passed; other shards remain incomplete")
        return
    report = write_report(
        output_dir,
        precheck_path,
        precheck_sha256,
        tests,
        time.time() - started,
    )
    runner.log(
        f"CAPTURE COMPLETE: {len(tests) * SHARD_COUNT} binaries, "
        f"{SHARD_COUNT} shards, {len(tests)} concrete PTX instructions"
    )
    runner.log(f"report: {report}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
