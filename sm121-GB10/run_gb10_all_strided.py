#!/usr/bin/env python3
"""Precheck, run, resume, and report every strided GB10 PTX sweep.

This single operational entry point covers all CUDA-toolkit-supported GB10
rows after applying the global README strides (u32: 0xffffff, u16: 0xff).  It
reuses the generic CUDA generator/runner, but owns the complete workflow:

    precheck -> plan -> full/resume -> report

The CUDA 13.1 matrix is approximately 12.85 GiB.  It is a structurally
validated GB10 golden capture; the precheck proves JIT execution and
determinism, not agreement with an independent numerical reference model.
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


ROOT = Path(__file__).resolve().parent
# The S2F6 conversions are SM121 application-specific.  A family-specific
# compute_120f/compute_121f target is intentionally rejected by ptxas.
ARCH = "compute_121a"
SHARD_COUNT = 16
SUPPORTED_CUDA_VERSIONS = {(13, 1), (13, 2)}
DEFAULT_FP6_REFERENCE_REPORT = (
    ROOT / "results" / "fp6-strided-precheck" / "precheck-report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every README-listed GB10 test supported by the selected CUDA toolkit."
        )
    )
    parser.add_argument("command", choices=("precheck", "plan", "full", "seal", "report"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--precheck-dir", type=Path)
    parser.add_argument(
        "--fp6-reference-report",
        type=Path,
        default=DEFAULT_FP6_REFERENCE_REPORT,
        help="optional independently validated FP6 precheck report",
    )
    parser.add_argument(
        "--cuda-version",
        choices=("13.1", "13.2"),
        default="13.1",
        help="select the PTX matrix supported by this toolkit",
    )
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument(
        "--compat-dir",
        type=Path,
        help="CUDA compatibility library directory; defaults from --cuda-version",
    )
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=SHARD_COUNT - 1)
    parser.add_argument(
        "--overwrite-precheck",
        action="store_true",
        help="replace an existing all-strided precheck directory",
    )
    parser.add_argument(
        "--yes-large",
        action="store_true",
        help="confirm generation of the complete strided golden data",
    )
    return parser.parse_args()


def cuda_version(value: str) -> tuple[int, int]:
    version = tuple(int(part) for part in value.split("."))
    if version not in SUPPORTED_CUDA_VERSIONS:
        raise RuntimeError(f"unsupported CUDA matrix: {value}")
    return version  # type: ignore[return-value]


def selected_tests(version: tuple[int, int]) -> list[runner.Test]:
    tests = [test for test in runner.TESTS if test.min_cuda <= version]
    expected = 73 if version == (13, 1) else 85
    if len(tests) != expected:
        raise RuntimeError(f"expected {expected} tests for CUDA {version}, found {len(tests)}")
    return tests


def full_runs(tests: Sequence[runner.Test]) -> list[tuple[runner.Test, runner.Sweep]]:
    return [(test, sweep) for _test_id, test, sweep in runner.selected_runs(tests, "full")]


def default_paths(args: argparse.Namespace, version: tuple[int, int]) -> None:
    label = f"cuda{version[0]}.{version[1]}"
    if args.output_dir is None:
        args.output_dir = ROOT / "results" / f"all-strided-{label}-full"
    if args.precheck_dir is None:
        args.precheck_dir = ROOT / "results" / f"all-strided-{label}-precheck"
    if args.compat_dir is None:
        args.compat_dir = Path(f"/usr/local/cuda-{version[0]}.{version[1]}/compat")


def configure_compatibility(compat_dir: Path) -> None:
    if not compat_dir.is_dir():
        raise RuntimeError(
            f"CUDA compatibility directory is missing: {compat_dir}; "
            "install the matching cuda-compat package before running PTX JIT instructions"
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
    binary = runner.DEFAULT_BUILD_DIR / "gb10_all_strided"
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
    for _test_id, test, sweep in runner.selected_runs(tests, "smoke"):
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
        "sweep_count": len(full_runs(tests)),
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
            "run_gb10_all_strided.py precheck first"
        )
    raw = path.read_bytes()
    report = json.loads(raw)
    expected_names = [test.name for test in tests]
    if (
        report.get("status") != "PASS"
        or report.get("test_count") != len(tests)
        or report.get("sweep_count") != len(full_runs(tests))
        or report.get("tests") != expected_names
    ):
        raise RuntimeError(f"precheck report is not a complete PASS: {path}")
    return report, hashlib.sha256(raw).hexdigest()


def expected_path(
    output_dir: Path,
    test: runner.Test,
    sweep: runner.Sweep,
    shard_index: int,
) -> Path:
    return (
        output_dir
        / runner.safe_name(test.name)
        / (
            f"{runner.safe_name(sweep.name)}__"
            f"shard-{shard_index:05d}-of-{SHARD_COUNT:05d}.bin"
        )
    )


def expected_results(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> list[tuple[int, runner.Test, runner.Sweep, Path, str]]:
    expected: list[tuple[int, runner.Test, runner.Sweep, Path, str]] = []
    for test_id, test in enumerate(tests):
        for sweep in test.sweeps:
            path = expected_path(output_dir, test, sweep, shard_index)
            expected.append(
                (test_id, test, sweep, path, path.relative_to(output_dir).as_posix())
            )
    return expected


def validate_manifest_entry(
    entry: dict[str, object],
    test_id: int,
    test: runner.Test,
    sweep: runner.Sweep,
    path: Path,
    relative: str,
    shard_index: int,
    *,
    require_digests: bool,
) -> dict[str, object]:
    start, count = runner.shard_slice(sweep.count, shard_index, SHARD_COUNT)
    expected_metadata = {
        "test_id": test_id,
        "test_name": test.name,
        "result_mask": test.mask,
        "total_records": sweep.count,
        "shard_start": start,
        "shard_records": count,
        "bytes": runner.HEADER_SIZE + count * runner.RECORD_SIZE,
        "ptx": test.ptx,
        "sweep": sweep.name,
        "file": relative,
        "comparison": "golden-captured",
        "ranges": {
            "source_a": runner.range_metadata(sweep.a),
            "source_b": runner.range_metadata(sweep.b),
            "source_c": runner.range_metadata(sweep.c),
        },
    }
    for field, expected in expected_metadata.items():
        if entry.get(field) != expected:
            raise RuntimeError(
                f"manifest provenance mismatch for {relative}: "
                f"{field}={entry.get(field)!r}, expected={expected!r}"
            )
    layout = runner.read_payload_layout(path)
    if layout["bytes"] != expected_metadata["bytes"]:
        raise RuntimeError(
            f"binary payload size mismatch for {relative}: "
            f"{layout['bytes']} != {expected_metadata['bytes']}"
        )
    digest = runner.validate_binary_inputs(path, sweep, start, count)
    spec_digest = runner.sweep_spec_sha256(test, sweep)
    if require_digests:
        if entry.get("sha256") != digest:
            raise RuntimeError(f"binary SHA256 mismatch for {relative}")
        if entry.get("spec_sha256") != spec_digest:
            raise RuntimeError(f"specification SHA256 mismatch for {relative}")
    sealed = dict(entry)
    sealed["sha256"] = digest
    sealed["spec_sha256"] = spec_digest
    return sealed


def read_shard_manifest(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> tuple[Path, dict[str, object]]:
    path = runner.manifest_path(output_dir, tests, shard_index, SHARD_COUNT)
    if not path.is_file():
        raise RuntimeError(f"shard manifest is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid shard manifest: {path}")
    return path, payload


def validate_shard_manifest(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> Path:
    matrix_digest = runner.matrix_sha256(tests)
    path, payload = read_shard_manifest(output_dir, tests, shard_index)
    expected_top = {
        "manifest_version": 3,
        "profile": "full",
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "test_count": len(tests),
        "matrix_sha256": matrix_digest,
    }
    for field, expected in expected_top.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"manifest metadata mismatch in {path}: "
                f"{field}={payload.get(field)!r}, expected={expected!r}"
            )
    entries = payload.get("result_files")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise RuntimeError(f"invalid result_files in {path}")
    by_file = {entry.get("file"): entry for entry in entries}
    expected = expected_results(output_dir, tests, shard_index)
    expected_files = {relative for _id, _test, _sweep, _path, relative in expected}
    if len(by_file) != len(entries) or set(by_file) != expected_files:
        raise RuntimeError(f"manifest file set does not match the test matrix: {path}")
    for test_id, test, sweep, binary, relative in expected:
        validate_manifest_entry(
            by_file[relative],
            test_id,
            test,
            sweep,
            binary,
            relative,
            shard_index,
            require_digests=True,
        )
    return path


def shard_complete(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> bool:
    try:
        validate_shard_manifest(output_dir, tests, shard_index)
    except Exception:
        return False
    return True


def remove_stale_partials(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> None:
    for test, sweep in full_runs(tests):
        partial = expected_path(output_dir, test, sweep, shard_index).with_suffix(".bin.partial")
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


def unsealed_manifest_shards(
    output_dir: Path,
    tests: Sequence[runner.Test],
) -> list[int]:
    unsealed: list[int] = []
    for shard_index in range(SHARD_COUNT):
        path = runner.manifest_path(output_dir, tests, shard_index, SHARD_COUNT)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            unsealed.append(shard_index)
            continue
        if payload.get("manifest_version") != 3:
            unsealed.append(shard_index)
    return unsealed


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


def seal_shard_manifest(
    output_dir: Path,
    tests: Sequence[runner.Test],
    shard_index: int,
) -> Path:
    path, payload = read_shard_manifest(output_dir, tests, shard_index)
    if payload.get("manifest_version") == 3:
        return validate_shard_manifest(output_dir, tests, shard_index)
    expected_top = {
        "profile": "full",
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "test_count": len(tests),
    }
    for field, expected in expected_top.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"legacy manifest metadata mismatch in {path}: "
                f"{field}={payload.get(field)!r}, expected={expected!r}"
            )
    entries = payload.get("result_files")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise RuntimeError(f"invalid legacy result_files in {path}")
    by_file = {entry.get("file"): entry for entry in entries}
    expected = expected_results(output_dir, tests, shard_index)
    expected_files = {relative for _id, _test, _sweep, _path, relative in expected}
    if len(by_file) != len(entries) or set(by_file) != expected_files:
        raise RuntimeError(f"legacy manifest file set does not match the matrix: {path}")
    sealed_entries = [
        validate_manifest_entry(
            by_file[relative],
            test_id,
            test,
            sweep,
            binary,
            relative,
            shard_index,
            require_digests=False,
        )
        for test_id, test, sweep, binary, relative in expected
    ]
    sealed = dict(payload)
    sealed.update(
        {
            "manifest_version": 3,
            "matrix_sha256": runner.matrix_sha256(tests),
            "result_files": sealed_entries,
        }
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return validate_shard_manifest(output_dir, tests, shard_index)


def validate_all(
    output_dir: Path,
    tests: Sequence[runner.Test],
) -> tuple[list[Path], list[Path]]:
    binaries = [
        expected_path(output_dir, test, sweep, shard_index)
        for shard_index in range(SHARD_COUNT)
        for test, sweep in full_runs(tests)
    ]
    expected_set = set(binaries)
    actual_set = set(output_dir.rglob("*.bin"))
    if actual_set != expected_set:
        raise RuntimeError(
            "full sweep binary set mismatch: "
            f"missing={len(expected_set - actual_set)}, extra={len(actual_set - expected_set)}"
        )
    partials = sorted(output_dir.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"full sweep has {len(partials)} incomplete .partial files")
    manifests = [
        validate_shard_manifest(output_dir, tests, shard_index)
        for shard_index in range(SHARD_COUNT)
    ]
    return binaries, manifests


def write_report(
    output_dir: Path,
    precheck_report: Path,
    precheck_sha256: str,
    tests: Sequence[runner.Test],
    elapsed: float,
    fp6_reference_report: Path,
) -> Path:
    binaries, manifests = validate_all(output_dir, tests)
    numerical_reference = numerical_reference_coverage(fp6_reference_report, tests)
    manifest_digests = {
        path.relative_to(output_dir).as_posix(): runner.file_sha256(path)
        for path in manifests
    }
    binary_digests: list[dict[str, str]] = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        for entry in payload["result_files"]:
            binary_digests.append(
                {"file": entry["file"], "sha256": entry["sha256"]}
            )
    binary_digests.sort(key=lambda item: item["file"])
    report = {
        "status": "CAPTURE_COMPLETE",
        "capture_status": "PASS",
        "accuracy_status": numerical_reference["status"],
        "result_kind": "GB10 golden capture with structural validation",
        "independent_numerical_reference": False,
        "numerically_validated_test_count": numerical_reference["test_count"],
        "numerical_reference": numerical_reference,
        "arch": ARCH,
        "shard_count": SHARD_COUNT,
        "test_count": len(tests),
        "sweep_count": len(full_runs(tests)),
        "tests": [{"name": test.name, "ptx": test.ptx} for test in tests],
        "binary_count": len(binaries),
        "binary_bytes": sum(path.stat().st_size for path in binaries),
        "manifest_count": len(manifests),
        "manifest_sha256": manifest_digests,
        "binary_sha256_root": runner.canonical_sha256(binary_digests),
        "matrix_sha256": runner.matrix_sha256(tests),
        "precheck_report": str(precheck_report),
        "precheck_report_sha256": precheck_sha256,
        "elapsed_seconds_this_invocation": elapsed,
    }
    path = output_dir / "full-run-report.json"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def numerical_reference_coverage(
    report_path: Path,
    tests: Sequence[runner.Test],
) -> dict[str, object]:
    fp6_tests = [test for test in tests if test.name.startswith("fp16x2_to_f6x2__")]
    unavailable: dict[str, object] = {
        "status": "NOT_INDEPENDENTLY_VALIDATED",
        "test_count": 0,
        "report": str(report_path),
    }
    if not report_path.is_file() or not fp6_tests:
        return unavailable
    raw = report_path.read_bytes()
    try:
        report = json.loads(raw)
    except Exception:
        return unavailable
    reference = report.get("reference")
    expected_matrix = runner.matrix_sha256(fp6_tests)
    if (
        report.get("status") != "PASS"
        or report.get("test_count") != len(fp6_tests)
        or report.get("matrix_sha256") != expected_matrix
        or not isinstance(reference, dict)
        or reference.get("lanes") != 4_128
    ):
        return unavailable
    return {
        "status": "PARTIAL_REFERENCE_PASS",
        "test_count": len(fp6_tests),
        "tests": [test.name for test in fp6_tests],
        "report": str(report_path),
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "matrix_sha256": expected_matrix,
        "lanes": reference["lanes"],
    }


def main() -> None:
    args = parse_args()
    version = cuda_version(args.cuda_version)
    default_paths(args, version)
    tests = selected_tests(version)
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
            runner.log(f"sealing shard manifest {shard_index}/{SHARD_COUNT - 1}")
            seal_shard_manifest(output_dir, tests, shard_index)
        report = write_report(
            output_dir,
            precheck_path,
            precheck_sha256,
            tests,
            0.0,
            args.fp6_reference_report.resolve(),
        )
        runner.log("CAPTURE COMPLETE: existing binaries sealed without relabeling provenance")
        runner.log(f"report: {report}")
        return

    _complete, pending, remaining = print_plan(args, tests)
    if args.command == "plan":
        return

    prebuilt_binary: Path | None = None
    if args.command == "full":
        unsealed = unsealed_manifest_shards(output_dir, tests)
        if unsealed:
            raise RuntimeError(
                "existing legacy manifests must be sealed before resume; run "
                f"`python3 run_gb10_all_strided.py seal` (shards: {unsealed})"
            )
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
            runner.log("all-strided precheck report is missing; running precheck automatically")
            prebuilt_binary = run_precheck(args, tests)
    _precheck, precheck_sha256 = precheck_evidence(precheck_path, tests)

    if args.command == "report":
        report = write_report(
            output_dir,
            precheck_path,
            precheck_sha256,
            tests,
            0.0,
            args.fp6_reference_report.resolve(),
        )
        runner.log(
            f"CAPTURE COMPLETE: {len(full_runs(tests)) * SHARD_COUNT} binaries"
        )
        runner.log(f"report: {report}")
        return

    started = time.time()
    if pending:
        binary = prebuilt_binary or build_runner(tests, str(nvcc))
        for position, shard_index in enumerate(pending, 1):
            runner.log(
                f"=== all-strided full shard {shard_index}/{SHARD_COUNT - 1} "
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
        args.fp6_reference_report.resolve(),
    )
    runner.log(
        f"CAPTURE COMPLETE: {len(full_runs(tests)) * SHARD_COUNT} binaries, "
        f"{SHARD_COUNT} shards, {len(tests)} concrete PTX instructions, "
        f"{len(full_runs(tests))} sweeps"
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
