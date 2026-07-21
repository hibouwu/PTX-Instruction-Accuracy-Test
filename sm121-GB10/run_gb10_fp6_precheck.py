#!/usr/bin/env python3
"""Run every bounded check required before the globally strided GB10 FP6 sweep.

This is the single operational entry point for the PTX 9.1 conversions

    cvt.rn.satfinite{.relu}.{e2m3x2/e3m2x2}.{f16x2/bf16x2}

It compiles once, exercises driver JIT, captures smoke data, verifies the full
strided sweep twice bit-for-bit, and compares every captured output lane with
an independent software model.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import platform
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Sequence

import run_gb10_ptx_accuracy as runner


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "fp6-strided-precheck"
TEST_PATTERN = "fp16x2_to_f6x2*"
ARCH = "compute_120f"
SAMPLE_RECORDS = runner.U32.count
EXPECTED_REFERENCE_LANES = 8 * SAMPLE_RECORDS * 2
RECORD = struct.Struct("<IIII")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete precheck before the globally strided GB10 FP6 sweep."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--compat-dir", type=Path, default=Path("/usr/local/cuda-13.1/compat"))
    parser.add_argument("--chunk-records", type=int, default=1_048_576)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan", action="store_true", help="show stages and output size without running")
    return parser.parse_args()


def projected_bytes(tests: Sequence[runner.Test]) -> int:
    smoke = runner.projected_bytes(tests, "smoke", 0, 1, None)
    determinism = 2 * runner.projected_bytes(tests, "full", 0, 1, SAMPLE_RECORDS)
    return smoke + determinism


def print_plan(tests: Sequence[runner.Test], output_dir: Path) -> None:
    total = projected_bytes(tests)
    print(f"tests: {len(tests)}")
    print("stages: compile/JIT preflight, smoke, full deterministic repeat, software reference")
    print(f"deterministic records per test: {SAMPLE_RECORDS}")
    print(f"independent-reference lanes: {EXPECTED_REFERENCE_LANES}")
    print(f"projected binary bytes: {total} ({total / 1024**2:.1f} MiB)")
    print(f"output: {output_dir.resolve()}")


def configure_compatibility(compat_dir: Path) -> None:
    if not compat_dir.is_dir():
        raise RuntimeError(
            f"CUDA compatibility directory is missing: {compat_dir}; "
            "install cuda-compat-13-1 first"
        )
    if not any(compat_dir.glob("libnvidia-ptxjitcompiler.so*")):
        raise RuntimeError(f"PTX JIT compatibility library is missing from {compat_dir}")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(":") if entry]
    compat = str(compat_dir.resolve())
    os.environ["LD_LIBRARY_PATH"] = ":".join([compat, *[entry for entry in entries if entry != compat]])
    runner.log(f"using CUDA compatibility libraries: {compat}")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"output directory already exists: {output_dir}; "
                "choose another --output-dir or pass --overwrite"
            )
        runner.log(f"removing previous precheck output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def run_capture(
    binary: Path,
    tests: Sequence[runner.Test],
    output_dir: Path,
    *,
    profile: str,
    shard_index: int = 0,
    shard_count: int = 1,
    limit_records: int | None = None,
    reference_dir: Path | None = None,
    chunk_records: int,
) -> list[dict[str, object]]:
    summaries = runner.execute_tests(
        binary,
        tests,
        output_dir,
        reference_dir,
        profile,
        shard_index,
        shard_count,
        chunk_records,
        limit_records,
    )
    runner.write_manifest(
        output_dir, tests, profile, shard_index, shard_count, summaries
    )
    return summaries


def decode_source(bits: int, source: str) -> tuple[bool, str, float]:
    sign = bool(bits & 0x8000)
    if source == "f16x2":
        exponent_bits, mantissa_bits, bias = 5, 10, 15
    elif source == "bf16x2":
        exponent_bits, mantissa_bits, bias = 8, 7, 127
    else:
        raise AssertionError(f"unsupported source: {source}")
    exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return sign, "nan" if mantissa else "inf", math.nan if mantissa else math.inf
    if exponent == 0:
        value = math.ldexp(mantissa, 1 - bias - mantissa_bits)
    else:
        value = math.ldexp(
            (1 << mantissa_bits) + mantissa,
            exponent - bias - mantissa_bits,
        )
    return sign, "finite", value


@functools.lru_cache(maxsize=None)
def fp6_positive_values(dtype: str) -> tuple[float, ...]:
    if dtype == "e2m3x2":
        exponent_bits, mantissa_bits, bias = 2, 3, 1
    elif dtype == "e3m2x2":
        exponent_bits, mantissa_bits, bias = 3, 2, 3
    else:
        raise AssertionError(f"unsupported FP6 type: {dtype}")
    values: list[float] = []
    for code in range(32):
        exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1)
        mantissa = code & ((1 << mantissa_bits) - 1)
        if exponent == 0:
            value = math.ldexp(mantissa, 1 - bias - mantissa_bits)
        else:
            value = math.ldexp(
                (1 << mantissa_bits) + mantissa,
                exponent - bias - mantissa_bits,
            )
        values.append(value)
    return tuple(values)


@functools.lru_cache(maxsize=None)
def expected_table(source: str, dtype: str, relu: bool) -> tuple[int, ...]:
    values = fp6_positive_values(dtype)
    outputs: list[int] = []
    for bits in range(65_536):
        sign, kind, magnitude = decode_source(bits, source)
        if kind == "nan":
            # PTX satfinite maps FP6 NaNs to positive MAX_NORM.
            code = 0x1F
        elif relu and sign:
            # GB10 also canonicalizes negative zero to positive zero here.
            code = 0
        else:
            if kind == "inf" or magnitude >= values[-1]:
                magnitude_code = 0x1F
            else:
                # Inputs and FP6 values are exact binary rationals in f64.
                # At a midpoint, .rn selects an even mantissa LSB.
                magnitude_code = min(
                    range(32),
                    key=lambda candidate: (
                        abs(values[candidate] - magnitude),
                        candidate & 1,
                        candidate,
                    ),
                )
            code = magnitude_code | (0x20 if sign else 0)
        outputs.append(code)
    return tuple(outputs)


def validate_reference(output_dir: Path) -> dict[str, int]:
    paths = sorted(output_dir.glob("fp16x2_to_f6x2__*/*.bin"))
    if len(paths) != 8:
        raise RuntimeError(f"expected 8 FP6 reference binaries, found {len(paths)}")
    partials = list(output_dir.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"found {len(partials)} incomplete reference outputs")

    records = 0
    lanes = 0
    nan_lanes = 0
    negative_lanes = 0
    padding_checks = 0

    for path in paths:
        name = path.parent.name
        source = "bf16x2" if "__bf16x2__" in name else "f16x2"
        dtype = "e2m3x2" if "__e2m3x2__" in name else "e3m2x2"
        relu = name.endswith("__relu")
        table = expected_table(source, dtype, relu)
        with path.open("rb") as stream:
            layout = runner.read_payload_layout(path)
            if layout["shard_records"] != SAMPLE_RECORDS:
                raise RuntimeError(f"unexpected FP6 payload size: {path}")
            stream.seek(runner.HEADER_SIZE)
            for index in range(SAMPLE_RECORDS):
                raw = stream.read(RECORD.size)
                if len(raw) != RECORD.size:
                    raise RuntimeError(f"truncated record {index}: {path}")
                source_a, source_b, source_c, result = RECORD.unpack(raw)
                expected_source = min(
                    runner.U32.maximum,
                    runner.U32.start + index * runner.U32.stride,
                )
                if source_a != 0 or source_c != 0xDEADBEEF:
                    raise RuntimeError(f"fixed input mismatch at record {index}: {path}")
                if source_b != expected_source:
                    raise RuntimeError(f"enumeration mismatch at record {index}: {path}")
                if result & 0xC0C0:
                    raise RuntimeError(f"nonzero FP6 padding bits at record {index}: {path}")
                padding_checks += 2
                for input_shift, output_shift in ((0, 0), (16, 8)):
                    source_lane = (source_b >> input_shift) & 0xFFFF
                    actual = (result >> output_shift) & 0x3F
                    expected = table[source_lane]
                    sign, kind, _magnitude = decode_source(source_lane, source)
                    nan_lanes += int(kind == "nan")
                    negative_lanes += int(sign and kind != "nan")
                    if actual != expected:
                        raise RuntimeError(
                            f"FP6 reference mismatch: {path}, record={index}, "
                            f"input=0x{source_lane:04x}, actual=0x{actual:02x}, "
                            f"expected=0x{expected:02x}"
                        )
                    lanes += 1
                records += 1
            if stream.read(1):
                raise RuntimeError(f"trailing binary data: {path}")

    if lanes != EXPECTED_REFERENCE_LANES:
        raise RuntimeError(
            f"reference lane coverage mismatch: expected {EXPECTED_REFERENCE_LANES}, got {lanes}"
        )
    return {
        "files": len(paths),
        "records": records,
        "lanes": lanes,
        "padding_checks": padding_checks,
        "nan_lanes": nan_lanes,
        "negative_lanes": negative_lanes,
    }


def main() -> None:
    args = parse_args()
    if args.chunk_records <= 0:
        raise RuntimeError("chunk-records must be positive")
    tests = runner.select_tests([TEST_PATTERN])
    output_dir = args.output_dir.resolve()
    if args.plan:
        print_plan(tests, output_dir)
        return

    nvcc = Path(args.nvcc)
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc not found: {nvcc}")
    configure_compatibility(args.compat_dir.resolve())
    total_bytes = projected_bytes(tests)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < total_bytes:
        raise RuntimeError(f"precheck needs {total_bytes} bytes but the filesystem is too full")
    prepare_output(output_dir, args.overwrite)

    started = time.time()
    source = runner.generate_cuda(tests, runner.DEFAULT_GENERATED_DIR)
    binary = runner.DEFAULT_BUILD_DIR / "gb10_ptx_accuracy"
    runner.compile_cuda(source, binary, tests, str(nvcc), ARCH)
    runner.preflight_tests(binary, tests)

    runner.log("=== smoke capture ===")
    smoke_dir = output_dir / "smoke"
    run_capture(binary, tests, smoke_dir, profile="smoke", chunk_records=args.chunk_records)

    runner.log(f"=== deterministic full {SAMPLE_RECORDS}-record capture ===")
    baseline_dir = output_dir / "determinism" / "baseline"
    repeat_dir = output_dir / "determinism" / "repeat"
    run_capture(
        binary,
        tests,
        baseline_dir,
        profile="full",
        limit_records=SAMPLE_RECORDS,
        chunk_records=args.chunk_records,
    )
    run_capture(
        binary,
        tests,
        repeat_dir,
        profile="full",
        limit_records=SAMPLE_RECORDS,
        reference_dir=baseline_dir,
        chunk_records=args.chunk_records,
    )

    runner.log("=== independent FP6 software reference ===")
    reference_stats = validate_reference(baseline_dir)
    elapsed = time.time() - started
    report = {
        "status": "PASS",
        "instruction_family": "cvt.rn.satfinite{.relu}.{e2m3x2/e3m2x2}.{f16x2/bf16x2}",
        "arch": ARCH,
        "compat_dir": str(args.compat_dir.resolve()),
        "host": platform.node(),
        "test_count": len(tests),
        "matrix_sha256": runner.matrix_sha256(tests),
        "determinism_records_per_test": SAMPLE_RECORDS,
        "reference": reference_stats,
        "binary_bytes": sum(path.stat().st_size for path in output_dir.rglob("*.bin")),
        "elapsed_seconds": elapsed,
    }
    report_path = output_dir / "precheck-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    runner.log(
        f"PRECHECK PASS: {reference_stats['files']} strided reference binaries, "
        f"{reference_stats['lanes']} lanes matched the independent reference"
    )
    runner.log(f"report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
