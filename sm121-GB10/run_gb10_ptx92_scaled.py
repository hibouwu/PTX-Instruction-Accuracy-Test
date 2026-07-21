#!/usr/bin/env python3
"""Formal GB10 workflow for the twelve PTX 9.2 scaled low-float conversions.

One user-facing entry point owns the complete workflow:

    selftest -> precheck -> full/resume -> report

``full`` reproduces the README Comments ranges exactly.  For E2M3/E3M2,
those raw strided values include encodings whose two padding bits are nonzero;
they remain useful GB10 golden observations but are outside the PTX-defined
numeric domain.  ``precheck`` therefore uses a separate legal-input matrix:
all 64x64 FP6 pairs (or all 16x16 FP4 pairs), with the strided packed scale
lattice covering every UE8M0 byte value in both lanes, and compares every
defined result lane against an independent exact integer/exponent model.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import run_gb10_all_strided as integrity
import run_gb10_ptx_accuracy as runner


ROOT = Path(__file__).resolve().parent
ARCH = "compute_120f"
SHARD_COUNT = integrity.SHARD_COUNT
EXPECTED_TESTS = 12
EXPECTED_REFERENCE_FILES = 516
EXPECTED_REFERENCE_RECORDS = 8_718_336
EXPECTED_REFERENCE_LANES = 17_436_672
DEFAULT_NVCC = "/usr/local/cuda-13.2/bin/nvcc"
DEFAULT_COMPAT_DIR = Path("/usr/local/cuda-13.2/compat")
DEFAULT_PRECHECK_DIR = ROOT / "results" / "ptx92-scaled-precheck"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "ptx92-scaled-full"
RECORD = runner.RECORD_STRUCT
BF16_CANONICAL_NAN = 0x7FFF
BF16_MAX = 0x7F7F
BF16_INFINITY = 0x7F80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and capture all twelve GB10 PTX 9.2 scaled FP4/FP6-to-BF16x2 instructions."
    )
    parser.add_argument(
        "command",
        choices=("selftest", "plan", "precheck", "full", "report", "all"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--precheck-dir", type=Path, default=DEFAULT_PRECHECK_DIR)
    parser.add_argument("--nvcc", default=DEFAULT_NVCC)
    parser.add_argument("--compat-dir", type=Path, default=DEFAULT_COMPAT_DIR)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=SHARD_COUNT - 1)
    parser.add_argument("--chunk-records", type=int, default=1_048_576)
    parser.add_argument("--overwrite-precheck", action="store_true")
    return parser.parse_args()


def selected_tests() -> list[runner.Test]:
    tests = [test for test in runner.TESTS if test.min_cuda == (13, 2)]
    expected_prefixes = ("f6x2_to_bf16x2_scaled__", "f4x2_to_bf16x2_scaled__")
    if len(tests) != EXPECTED_TESTS or any(
        not test.name.startswith(expected_prefixes) for test in tests
    ):
        raise RuntimeError(
            f"PTX 9.2 matrix changed: expected {EXPECTED_TESTS} scaled tests, found {len(tests)}"
        )
    if sum(test.name.startswith(expected_prefixes[0]) for test in tests) != 8:
        raise RuntimeError("expected eight FP6-to-BF16x2 tests")
    if sum(test.name.startswith(expected_prefixes[1]) for test in tests) != 4:
        raise RuntimeError("expected four FP4-to-BF16x2 tests")
    if any(len(test.sweeps) != 1 or test.mask != 0xFFFFFFFF for test in tests):
        raise RuntimeError("PTX 9.2 tests require one sweep and a full 32-bit result")
    return tests


def legal_reference_tests(tests: Sequence[runner.Test]) -> list[runner.Test]:
    reference: list[runner.Test] = []
    for test in tests:
        if test.name.startswith("f6x2_to_bf16x2_scaled__"):
            sweeps = tuple(
                runner.Sweep(
                    f"legal-upper-{upper:02x}",
                    runner.FIXED_ZERO,
                    runner.ValueRange(upper << 8, (upper << 8) | 0x3F, 1),
                    runner.SCALE_C,
                )
                for upper in range(64)
            )
        else:
            sweeps = (
                runner.Sweep(
                    "legal-packed-b8",
                    runner.FIXED_ZERO,
                    runner.ValueRange(0, 0xFF, 1),
                    runner.SCALE_C,
                ),
            )
        reference.append(dataclasses.replace(test, sweeps=sweeps))
    if sum(len(test.sweeps) for test in reference) != EXPECTED_REFERENCE_FILES:
        raise AssertionError("unexpected legal reference file count")
    if sum(sweep.count for test in reference for sweep in test.sweeps) != EXPECTED_REFERENCE_RECORDS:
        raise AssertionError("unexpected legal reference record count")
    return reference


def configure_compatibility(compat_dir: Path) -> None:
    if not compat_dir.is_dir():
        raise RuntimeError(f"CUDA 13.2 compatibility directory is missing: {compat_dir}")
    if not any(compat_dir.glob("libnvidia-ptxjitcompiler.so*")):
        raise RuntimeError(f"PTX JIT compiler is missing from {compat_dir}")
    entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    compat = str(compat_dir)
    if compat in entries:
        entries.remove(compat)
    os.environ["LD_LIBRARY_PATH"] = ":".join([compat, *entries])
    runner.log(f"using CUDA 13.2 compatibility libraries: {compat}")


def safe_remove_tree(path: Path) -> None:
    path = path.resolve()
    protected = {Path("/"), Path.home().resolve(), ROOT.resolve(), ROOT.parent.resolve()}
    if path in protected or len(path.parts) < 4:
        raise RuntimeError(f"refusing to remove unsafe path: {path}")
    shutil.rmtree(path)


@contextlib.contextmanager
def output_lock(directory: Path) -> Iterator[None]:
    # Keep the lock beside the result tree: --overwrite-precheck may remove the
    # tree itself, so a lock inside it would not protect against a second run.
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.run.lock"
    with lock_path.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another PTX 9.2 workflow holds {lock_path}") from error
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield


def test_tags(test: runner.Test) -> tuple[str, bool, bool]:
    parts = test.name.split("__")
    dtype = next((part for part in parts if part in {"e2m3x2", "e3m2x2"}), "e2m1x2")
    return dtype, "relu" in parts, "satfinite" in parts


def decode_low_float(code: int, dtype: str) -> tuple[bool, int, int]:
    if dtype == "e2m3x2":
        exponent_bits, mantissa_bits, bias = 2, 3, 1
    elif dtype == "e3m2x2":
        exponent_bits, mantissa_bits, bias = 3, 2, 3
    elif dtype == "e2m1x2":
        exponent_bits, mantissa_bits, bias = 2, 1, 1
    else:
        raise AssertionError(f"unsupported low-float type: {dtype}")
    width = 1 + exponent_bits + mantissa_bits
    sign = bool(code & (1 << (width - 1)))
    magnitude = code & ((1 << (width - 1)) - 1)
    exponent = magnitude >> mantissa_bits
    fraction = magnitude & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        return sign, fraction, 1 - bias - mantissa_bits
    return sign, (1 << mantissa_bits) | fraction, exponent - bias - mantissa_bits


def is_bf16_nan(bits: int) -> bool:
    return (bits & 0x7F80) == 0x7F80 and (bits & 0x007F) != 0


def expected_bf16(
    source_code: int,
    scale_code: int,
    dtype: str,
    relu: bool,
    satfinite: bool,
) -> tuple[str, int | None]:
    # UE8M0 0xff is its sole NaN.  PTX requires a canonical NaN with .relu;
    # without .relu the BF16 NaN payload is not part of the conformance check.
    if scale_code == 0xFF:
        return ("exact", BF16_CANONICAL_NAN) if relu else ("nan", None)

    sign, significand, power = decode_low_float(source_code, dtype)
    if significand == 0:
        return "exact", 0 if relu and sign else (0x8000 if sign else 0)
    if relu and sign:
        return "exact", 0

    power += scale_code - 127
    highest_power = significand.bit_length() - 1 + power
    sign_bit = 0x8000 if sign else 0
    if highest_power > 127:
        return "exact", sign_bit | (BF16_MAX if satfinite else BF16_INFINITY)

    if highest_power >= -126:
        significand8 = significand << (7 - (significand.bit_length() - 1))
        exponent_field = highest_power + 127
        fraction = significand8 - 128
        if not 1 <= exponent_field <= 254 or not 0 <= fraction <= 127:
            raise AssertionError("invalid exact BF16 normal encoding")
        return "exact", sign_bit | (exponent_field << 7) | fraction

    # All finite combinations in this instruction family remain exact BF16
    # subnormals: the smallest is E3M2 min-subnormal * 2^-127 = 2^-131.
    shift = power + 133
    if shift < 0:
        raise AssertionError("unexpected BF16 underflow requiring rounding")
    fraction = significand << shift
    if not 1 <= fraction <= 127:
        raise AssertionError("invalid exact BF16 subnormal encoding")
    return "exact", sign_bit | fraction


def selftest_reference() -> None:
    cases = (
        (0x08, 0x7F, "e2m3x2", False, False, 0x3F80),
        (0x1F, 0x7F, "e2m3x2", False, False, 0x40F0),
        (0x1F, 0xFC, "e2m3x2", False, False, 0x7F70),
        (0x1F, 0xFD, "e2m3x2", False, False, 0x7F80),
        (0x1F, 0xFD, "e2m3x2", False, True, 0x7F7F),
        (0x20, 0x7F, "e2m3x2", False, False, 0x8000),
        (0x20, 0x7F, "e2m3x2", True, False, 0x0000),
        (0x01, 0x00, "e3m2x2", False, False, 0x0004),
        (0x1F, 0x7F, "e3m2x2", False, False, 0x41E0),
        (0x07, 0x7F, "e2m1x2", False, False, 0x40C0),
    )
    for source, scale, dtype, relu, satfinite, expected in cases:
        kind, actual = expected_bf16(source, scale, dtype, relu, satfinite)
        if kind != "exact" or actual != expected:
            raise AssertionError(
                f"reference selftest failed: {dtype} source=0x{source:x} scale=0x{scale:02x}: "
                f"actual={actual!r}, expected=0x{expected:04x}"
            )
    if expected_bf16(0x01, 0xFF, "e2m3x2", False, False)[0] != "nan":
        raise AssertionError("UE8M0 NaN classification selftest failed")
    if expected_bf16(0x01, 0xFF, "e2m3x2", True, False) != ("exact", 0x7FFF):
        raise AssertionError("canonical BF16 NaN selftest failed")


def build_runner(tests: Sequence[runner.Test], nvcc: str, name: str) -> Path:
    source = runner.generate_cuda(tests, runner.DEFAULT_GENERATED_DIR)
    binary = runner.DEFAULT_BUILD_DIR / name
    runner.log(f"generated {source} ({len(tests)} concrete PTX 9.2 instructions)")
    runner.compile_cuda(source, binary, tests, nvcc, ARCH)
    runner.preflight_tests(binary, tests)
    return binary


def capture_tree(
    binary: Path,
    tests: Sequence[runner.Test],
    output_dir: Path,
    chunk_records: int,
    reference_dir: Path | None = None,
) -> list[dict[str, object]]:
    summaries = runner.execute_tests(
        binary,
        tests,
        output_dir,
        reference_dir,
        "full",
        0,
        1,
        chunk_records,
        None,
    )
    runner.write_manifest(output_dir, tests, "full", 0, 1, summaries)
    return summaries


def validate_reference_tree(
    output_dir: Path,
    tests: Sequence[runner.Test],
) -> dict[str, int]:
    files = records = lanes = exact = nan_class = 0
    positive = negative = zero = overflow = subnormal = 0
    for test_id, test in enumerate(tests):
        dtype, relu, satfinite = test_tags(test)
        for sweep in test.sweeps:
            path = (
                output_dir
                / runner.safe_name(test.name)
                / f"{runner.safe_name(sweep.name)}__shard-00000-of-00001.bin"
            )
            layout = runner.read_payload_layout(path)
            if layout["shard_records"] != sweep.count:
                raise RuntimeError(f"reference payload size mismatch: {path}")
            runner.validate_binary_inputs(path, sweep, 0, sweep.count)
            with path.open("rb") as stream:
                stream.seek(runner.HEADER_SIZE)
                for index in range(sweep.count):
                    raw = stream.read(RECORD.size)
                    if len(raw) != RECORD.size:
                        raise RuntimeError(f"truncated reference record {index}: {path}")
                    _source_a, source_b, source_c, result = RECORD.unpack(raw)
                    scale = (source_c >> 16) & 0xFFFF
                    if dtype == "e2m1x2":
                        packed = source_b & 0xFF
                        source_lanes = (packed & 0xF, (packed >> 4) & 0xF)
                    else:
                        packed = source_b & 0xFFFF
                        if packed & 0xC0C0:
                            raise RuntimeError(f"illegal FP6 padding in reference input: {path}")
                        source_lanes = (packed & 0x3F, (packed >> 8) & 0x3F)
                    scale_lanes = (scale & 0xFF, (scale >> 8) & 0xFF)
                    actual_lanes = (result & 0xFFFF, (result >> 16) & 0xFFFF)
                    for lane, (source_code, scale_code, actual) in enumerate(
                        zip(source_lanes, scale_lanes, actual_lanes)
                    ):
                        kind, expected = expected_bf16(
                            source_code, scale_code, dtype, relu, satfinite
                        )
                        if kind == "nan":
                            if not is_bf16_nan(actual):
                                raise RuntimeError(
                                    f"BF16 NaN-class mismatch: {path}, record={index}, lane={lane}, "
                                    f"source=0x{source_code:02x}, scale=0x{scale_code:02x}, actual=0x{actual:04x}"
                                )
                            nan_class += 1
                        else:
                            if actual != expected:
                                raise RuntimeError(
                                    f"BF16 reference mismatch: {path}, record={index}, lane={lane}, "
                                    f"source=0x{source_code:02x}, scale=0x{scale_code:02x}, "
                                    f"actual=0x{actual:04x}, expected=0x{expected:04x}"
                                )
                            exact += 1
                        sign, significand, _power = decode_low_float(source_code, dtype)
                        zero += int(significand == 0 and scale_code != 0xFF)
                        negative += int(sign and significand != 0 and scale_code != 0xFF)
                        positive += int(not sign and significand != 0 and scale_code != 0xFF)
                        overflow += int((actual & 0x7FFF) in {BF16_MAX, BF16_INFINITY})
                        subnormal += int((actual & 0x7F80) == 0 and (actual & 0x7F) != 0)
                        lanes += 1
                    records += 1
            files += 1
    if files != EXPECTED_REFERENCE_FILES:
        raise RuntimeError(f"expected {EXPECTED_REFERENCE_FILES} reference files, found {files}")
    if records != EXPECTED_REFERENCE_RECORDS or lanes != EXPECTED_REFERENCE_LANES:
        raise RuntimeError(
            f"reference coverage mismatch: records={records}, lanes={lanes}"
        )
    return {
        "files": files,
        "records": records,
        "lanes": lanes,
        "exact_bit_matches": exact,
        "nan_class_matches": nan_class,
        "positive_lanes": positive,
        "negative_lanes": negative,
        "zero_lanes": zero,
        "overflow_or_maxfinite_lanes": overflow,
        "subnormal_lanes": subnormal,
    }


def run_precheck(args: argparse.Namespace, tests: Sequence[runner.Test]) -> Path:
    output_dir = args.precheck_dir.resolve()
    reference_tests = legal_reference_tests(tests)
    required = 2 * runner.projected_bytes(reference_tests, "full", 0, 1, None)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with output_lock(output_dir):
        if output_dir.exists():
            if not args.overwrite_precheck:
                raise RuntimeError(
                    f"precheck output already exists: {output_dir}; use --overwrite-precheck"
                )
            safe_remove_tree(output_dir)
        if shutil.disk_usage(output_dir.parent).free < required + 1024**3:
            raise RuntimeError(f"precheck needs {required} bytes plus 1 GiB safety margin")
        output_dir.mkdir(parents=True)
        binary = build_runner(reference_tests, args.nvcc, "gb10_ptx92_scaled")
        baseline = output_dir / "baseline"
        repeat = output_dir / "repeat"
        runner.log("=== legal-input reference baseline ===")
        baseline_summaries = capture_tree(
            binary, reference_tests, baseline, args.chunk_records
        )
        runner.log("=== deterministic repeat and bitwise comparison ===")
        repeat_summaries = capture_tree(
            binary, reference_tests, repeat, args.chunk_records, baseline
        )
        runner.log("=== independent exact BF16 reference ===")
        reference = validate_reference_tree(baseline, reference_tests)
        report = {
            "status": "PASS",
            "accuracy_status": "DEFINED_DOMAIN_REFERENCE_PASS",
            "instruction_family": (
                "cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0."
                "bf16x2.{e2m3x2/e3m2x2/e2m1x2}"
            ),
            "arch": ARCH,
            "compat_dir": str(args.compat_dir.resolve()),
            "host": platform.node(),
            "test_count": len(tests),
            "tests": [test.name for test in tests],
            "capture_matrix_sha256": runner.matrix_sha256(tests),
            "reference_matrix_sha256": runner.matrix_sha256(reference_tests),
            "reference": reference,
            "determinism_files": len(repeat_summaries),
            "determinism_bytes": sum(item["bytes"] for item in repeat_summaries),
            "baseline_bytes": sum(item["bytes"] for item in baseline_summaries),
            "coverage_note": (
                "All legal packed FP6/FP4 sources; the packed scale lattice exposes "
                "all 256 UE8M0 codes in both lanes. Nonzero FP6 padding is excluded."
            ),
            "elapsed_seconds": time.time() - started,
        }
        path = output_dir / "precheck-report.json"
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    runner.log(
        f"PRECHECK PASS: {reference['lanes']} defined lanes matched the independent reference"
    )
    runner.log(f"report: {path}")
    return path


def load_precheck(path: Path, tests: Sequence[runner.Test]) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise RuntimeError(f"precheck report missing: {path}; run precheck first")
    raw = path.read_bytes()
    report = json.loads(raw)
    reference = report.get("reference")
    if (
        report.get("status") != "PASS"
        or report.get("accuracy_status") != "DEFINED_DOMAIN_REFERENCE_PASS"
        or report.get("test_count") != EXPECTED_TESTS
        or report.get("tests") != [test.name for test in tests]
        or report.get("capture_matrix_sha256") != runner.matrix_sha256(tests)
        or not isinstance(reference, dict)
        or reference.get("lanes") != EXPECTED_REFERENCE_LANES
    ):
        raise RuntimeError(f"precheck report is not a complete matrix-bound PASS: {path}")
    return report, hashlib.sha256(raw).hexdigest()


def requested_shards(args: argparse.Namespace) -> range:
    if not 0 <= args.start_shard <= args.end_shard < SHARD_COUNT:
        raise RuntimeError(f"require 0 <= start-shard <= end-shard < {SHARD_COUNT}")
    return range(args.start_shard, args.end_shard + 1)


def plan(args: argparse.Namespace, tests: Sequence[runner.Test]) -> tuple[list[int], int]:
    output_dir = args.output_dir.resolve()
    shards = requested_shards(args)
    complete = [index for index in shards if integrity.shard_complete(output_dir, tests, index)]
    pending = [index for index in shards if index not in complete]
    total = sum(
        runner.projected_bytes(tests, "full", index, SHARD_COUNT, None)
        for index in range(SHARD_COUNT)
    )
    remaining = sum(
        runner.projected_bytes(tests, "full", index, SHARD_COUNT, None)
        for index in pending
    )
    reference_tests = legal_reference_tests(tests)
    precheck_bytes = 2 * runner.projected_bytes(reference_tests, "full", 0, 1, None)
    print(f"PTX 9.2 concrete instructions: {len(tests)}")
    print(f"Comments records per instruction: {tests[0].sweeps[0].count}")
    print(f"Comments full capture: {total} bytes ({total / 1024**2:.3f} MiB)")
    print(f"legal reference baseline+repeat: {precheck_bytes} bytes ({precheck_bytes / 1024**2:.3f} MiB)")
    print(f"complete shards: {complete or 'none'}")
    print(f"pending shards: {pending or 'none'}")
    print(f"remaining full output: {remaining} bytes")
    print("scope: full follows README stride 0xff; precheck excludes undefined FP6 padding")
    return pending, remaining


def command_output(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(command), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, timeout=15,
        )
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def full_capture_stats(tests: Sequence[runner.Test]) -> dict[str, int]:
    records = sum(test.sweeps[0].count for test in tests)
    f6_tests = sum(test.name.startswith("f6x2_to_bf16x2_scaled__") for test in tests)
    f4_tests = len(tests) - f6_tests
    source_values = list(runner.U16.start + index * runner.U16.stride for index in range(runner.U16.count))
    source_values[-1] = runner.U16.maximum
    valid_lower = sum((value & 0xC0) == 0 for value in source_values)
    valid_upper = sum((value & 0xC000) == 0 for value in source_values)
    scale_count = runner.SCALE_C.count
    defined_f6_lanes = f6_tests * (valid_lower + valid_upper) * scale_count
    f6_lanes = f6_tests * runner.U16.count * scale_count * 2
    f4_lanes = f4_tests * runner.U16.count * scale_count * 2
    return {
        "records": records,
        "lane_observations": f6_lanes + f4_lanes,
        "defined_lane_observations": defined_f6_lanes + f4_lanes,
        "undefined_fp6_padding_lane_observations": f6_lanes - defined_f6_lanes,
        "f4_source_alias_records_per_test": (runner.U16.count - 256) * scale_count,
    }


def write_reports(
    output_dir: Path,
    precheck_path: Path,
    precheck_sha256: str,
    precheck: dict[str, object],
    tests: Sequence[runner.Test],
    nvcc: str,
    compat_dir: Path,
    elapsed: float,
) -> tuple[Path, Path]:
    binaries, manifests = integrity.validate_all(output_dir, tests)
    manifest_sha = {
        path.relative_to(output_dir).as_posix(): runner.file_sha256(path)
        for path in manifests
    }
    binary_digests: list[dict[str, str]] = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text())
        binary_digests.extend(
            {"file": entry["file"], "sha256": entry["sha256"]}
            for entry in payload["result_files"]
        )
    binary_digests.sort(key=lambda item: item["file"])
    report = {
        "status": "PASS",
        "capture_status": "COMMENTS_STRIDED_CAPTURE_COMPLETE",
        "accuracy_status": "DEFINED_DOMAIN_REFERENCE_PASS",
        "result_kind": "GB10 golden capture plus independent legal-domain reference",
        "instruction_family": (
            "cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0."
            "bf16x2.{e2m3x2/e3m2x2/e2m1x2}"
        ),
        "arch": ARCH,
        "test_count": len(tests),
        "tests": [{"name": test.name, "ptx": test.ptx} for test in tests],
        "matrix_sha256": runner.matrix_sha256(tests),
        "shard_count": SHARD_COUNT,
        "binary_count": len(binaries),
        "binary_bytes": sum(path.stat().st_size for path in binaries),
        "binary_sha256_root": runner.canonical_sha256(binary_digests),
        "manifest_count": len(manifests),
        "manifest_sha256": manifest_sha,
        "comments_capture": full_capture_stats(tests),
        "independent_reference": precheck["reference"],
        "precheck_report": str(precheck_path),
        "precheck_report_sha256": precheck_sha256,
        "provenance": {
            "host": platform.node(),
            "nvcc": command_output([nvcc, "--version"]),
            "compat_dir": str(compat_dir),
            "gpu": command_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"]
            ),
            "git_commit": command_output(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"]),
        },
        "scope_note": (
            "The full capture follows README packed source/scale stride 0xff. "
            "Nonzero FP6 padding is captured as GB10 observation only; numerical PASS "
            "comes from the separate all-legal-source, all-scalar-scale precheck."
        ),
        "elapsed_seconds_this_invocation": elapsed,
    }
    json_path = output_dir / "full-run-report.json"
    json_temp = json_path.with_suffix(".json.partial")
    json_temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    json_temp.replace(json_path)

    reference = precheck["reference"]
    stats = report["comments_capture"]
    markdown = f"""# GB10 PTX 9.2 Scaled Conversion Report

## 结论

- Comments strided golden capture: **COMPLETE**
- PTX 定义域独立参考检查: **PASS**
- 具体指令: {len(tests)}
- 结果文件: {len(binaries)} `.bin` / {len(manifests)} manifests
- 结果大小: {report['binary_bytes']} bytes ({report['binary_bytes'] / 1024**2:.3f} MiB)
- 独立参考比较: {reference['lanes']} lanes

## 指令族

```text
cvt.rn{{.relu}}{{.satfinite}}.scaled::n2::ue8m0.bf16x2.{{e2m3x2/e3m2x2/e2m1x2}}
```

## 覆盖范围

正式 capture 严格遵循 README Comments：packed source 与 packed scale 均从 `0x0000` 到 `0xffff`，逻辑 stride 为 `0xff`，每条指令 {tests[0].sweeps[0].count} records。

独立参考检查另外枚举全部合法 FP6/FP4 packed source；两条 lane 均覆盖全部 256 个 UE8M0 scale code。共比较 {reference['records']} records、{reference['lanes']} lanes，其中 exact-bit {reference['exact_bit_matches']}，NaN-class {reference['nan_class_matches']}。

## 结论边界

FP6 packed source 的每个 byte 高两位按 PTX 必须为零。Comments 的 raw stride 中有 {stats['undefined_fp6_padding_lane_observations']} 个 lane observation 不满足该条件；它们保留为 GB10 golden observation，但不计入规范数值 PASS。合法定义域的数值结论由独立 precheck 提供。

## 环境

```text
Host: {report['provenance']['host']}
Architecture target: {ARCH}
Compatibility libraries: {compat_dir}
GPU/driver: {report['provenance']['gpu']}
Git commit: {report['provenance']['git_commit']}
```

详细 SHA256、文件清单和统计见 `full-run-report.json`。
"""
    md_path = output_dir / "full-run-report.md"
    md_temp = md_path.with_suffix(".md.partial")
    md_temp.write_text(markdown)
    md_temp.replace(md_path)
    return json_path, md_path


def run_full(args: argparse.Namespace, tests: Sequence[runner.Test]) -> None:
    started = time.time()
    precheck_path = args.precheck_dir.resolve() / "precheck-report.json"
    precheck, precheck_sha = load_precheck(precheck_path, tests)
    output_dir = args.output_dir.resolve()
    pending, remaining = plan(args, tests)
    if pending:
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_compatibility(args.compat_dir.resolve())
        competitors = integrity.competing_accuracy_processes()
        if competitors:
            raise RuntimeError("another GB10 accuracy process is active:\n" + "\n".join(competitors))
        with output_lock(output_dir):
            # The plan was printed before locking.  Re-evaluate it here so a
            # just-finished peer cannot make us overwrite a completed shard.
            pending = [
                index for index in requested_shards(args)
                if not integrity.shard_complete(output_dir, tests, index)
            ]
            remaining = sum(
                runner.projected_bytes(tests, "full", index, SHARD_COUNT, None)
                for index in pending
            )
            if shutil.disk_usage(output_dir).free < remaining + 1024**3:
                raise RuntimeError(
                    "insufficient space for pending output plus 1 GiB safety margin"
                )
            if pending:
                binary = build_runner(tests, args.nvcc, "gb10_ptx92_scaled")
                for shard_index in pending:
                    runner.log(f"=== PTX 9.2 shard {shard_index}/{SHARD_COUNT - 1} ===")
                    integrity.remove_stale_partials(output_dir, tests, shard_index)
                    integrity.run_one_shard(binary, output_dir, tests, shard_index)
                    integrity.validate_shard_manifest(output_dir, tests, shard_index)
    complete_all = all(
        integrity.shard_complete(output_dir, tests, index) for index in range(SHARD_COUNT)
    )
    if complete_all:
        json_path, md_path = write_reports(
            output_dir,
            precheck_path,
            precheck_sha,
            precheck,
            tests,
            args.nvcc,
            args.compat_dir.resolve(),
            time.time() - started,
        )
        runner.log(
            f"FULL PASS: {EXPECTED_TESTS} PTX 9.2 instructions, {SHARD_COUNT} shards"
        )
        runner.log(f"reports: {json_path}, {md_path}")
    else:
        runner.log("requested shards complete; run full without a shard range to finish/report")


def main() -> None:
    args = parse_args()
    if args.chunk_records <= 0:
        raise RuntimeError("chunk-records must be positive")
    selftest_reference()
    tests = selected_tests()
    if args.command == "selftest":
        legal_reference_tests(tests)
        print("SELFTEST PASS: PTX 9.2 matrix and independent BF16 model")
        return
    if args.command == "plan":
        plan(args, tests)
        return
    if args.command == "precheck":
        configure_compatibility(args.compat_dir.resolve())
        run_precheck(args, tests)
        return
    if args.command == "report":
        precheck_path = args.precheck_dir.resolve() / "precheck-report.json"
        precheck, precheck_sha = load_precheck(precheck_path, tests)
        json_path, md_path = write_reports(
            args.output_dir.resolve(), precheck_path, precheck_sha, precheck,
            tests, args.nvcc, args.compat_dir.resolve(), 0.0,
        )
        print(f"REPORT PASS: {json_path}, {md_path}")
        return
    if args.command == "all":
        precheck_path = args.precheck_dir.resolve() / "precheck-report.json"
        if precheck_path.is_file():
            load_precheck(precheck_path, tests)
            runner.log(f"reusing validated precheck: {precheck_path}")
        elif args.precheck_dir.resolve().exists() and not args.overwrite_precheck:
            raise RuntimeError(
                f"incomplete precheck directory exists: {args.precheck_dir.resolve()}; "
                "rerun all with --overwrite-precheck"
            )
        else:
            configure_compatibility(args.compat_dir.resolve())
            run_precheck(args, tests)
    run_full(args, tests)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
