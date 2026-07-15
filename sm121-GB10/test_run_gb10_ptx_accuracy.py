#!/usr/bin/env python3
"""CPU-only contract tests for the GB10 PTX accuracy runner."""

from __future__ import annotations

import importlib.util
import io
import dataclasses
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_gb10_ptx_accuracy.py")
SPEC = importlib.util.spec_from_file_location("gb10_ptx_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

SCRIPT_DIRECTORY = str(SCRIPT.parent)
if SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPT_DIRECTORY)
import run_gb10_bounded_conversions as bounded
import run_gb10_all_strided as all_strided


def values(value_range: object) -> tuple[int, ...]:
    return tuple(
        min(value_range.maximum, value_range.start + index * value_range.stride)
        for index in range(value_range.count)
    )


def write_synthetic_result(
    output: Path,
    test: object,
    *,
    test_id: int = 0,
    sealed: bool = True,
) -> tuple[Path, Path]:
    sweep = test.sweeps[0]
    shard_index = 0
    start, count = runner.shard_slice(sweep.count, shard_index, all_strided.SHARD_COUNT)
    path = all_strided.expected_path(output, test, sweep, shard_index)
    path.parent.mkdir(parents=True)
    name = test.name.encode()
    header = runner.HEADER_STRUCT.pack(
        runner.MAGIC,
        runner.FORMAT_VERSION,
        runner.HEADER_SIZE,
        runner.RECORD_SIZE,
        test_id,
        test.mask,
        sweep.count,
        start,
        count,
        name + b"\0" * (160 - len(name)),
        b"\0" * 44,
    )
    records = bytearray()
    for local in range(count):
        linear = start + local
        c_index = linear % sweep.c.count
        linear //= sweep.c.count
        b_index = linear % sweep.b.count
        linear //= sweep.b.count
        a_index = linear % sweep.a.count
        records.extend(
            runner.RECORD_STRUCT.pack(
                min(sweep.a.maximum, sweep.a.start + a_index * sweep.a.stride),
                min(sweep.b.maximum, sweep.b.start + b_index * sweep.b.stride),
                min(sweep.c.maximum, sweep.c.start + c_index * sweep.c.stride),
                0x1234,
            )
        )
    path.write_bytes(header + records)
    summary = runner.read_header(path)
    summary.update(
        {
            "ptx": test.ptx,
            "sweep": sweep.name,
            "file": path.relative_to(output).as_posix(),
            "ranges": {
                "source_a": runner.range_metadata(sweep.a),
                "source_b": runner.range_metadata(sweep.b),
                "source_c": runner.range_metadata(sweep.c),
            },
            "comparison": "golden-captured",
        }
    )
    if sealed:
        summary["spec_sha256"] = runner.sweep_spec_sha256(test, sweep)
        summary["sha256"] = runner.file_sha256(path)
        manifest = runner.write_manifest(
            output, [test], "full", shard_index, all_strided.SHARD_COUNT, [summary]
        )
    else:
        manifest = runner.manifest_path(
            output, [test], shard_index, all_strided.SHARD_COUNT
        )
        manifest.write_text(
            runner.json.dumps(
                {
                    "format": "GB10 PTX accuracy binary v1",
                    "profile": "full",
                    "shard_index": shard_index,
                    "shard_count": all_strided.SHARD_COUNT,
                    "test_count": 1,
                    "result_files": [summary],
                }
            )
        )
    return path, manifest


class RunnerContractTests(unittest.TestCase):
    def test_matrix_is_unique_and_complete(self) -> None:
        runner.validate_test_matrix(runner.TESTS)
        self.assertEqual(len(runner.TESTS), 85)

    def test_fp16x2_to_f6x2_expands_eight_strided_b16_results(self) -> None:
        tests = runner.select_tests(["fp16x2_to_f6x2*"])
        self.assertEqual(len(tests), 8)
        self.assertEqual({test.mask for test in tests}, {0xFFFF})
        self.assertEqual({test.sweeps[0].b.count for test in tests}, {258})
        self.assertEqual({test.sweeps[0].c.start for test in tests}, {0xDEADBEEF})

    def test_comments_ranges_for_add_and_fma(self) -> None:
        add_first, add_second = runner.ADD_SWEEPS
        self.assertEqual((add_first.a.maximum, add_first.a.stride), (0xFFFF, 0xFF))
        self.assertEqual((add_first.c.maximum, add_first.c.stride), (0xFFFFFFFF, 0xFFFFFF))
        self.assertEqual((add_second.a.maximum, add_second.a.stride), (0xFFFF, 0xFF))
        self.assertEqual((add_second.c.maximum, add_second.c.stride), (0xFFFFFFFF, 0xFFFFFF))

        fma_first, fma_second, fma_third = runner.FMA_SWEEPS
        self.assertEqual(fma_first.c.stride, 0xFFFFFF)
        self.assertEqual(fma_second.c.stride, 0xFFFFFF)
        self.assertEqual(fma_third.c.stride, 0xFFFFFF)

    def test_scaled_bf16_to_s2f6_uses_fixed_comment_source_c(self) -> None:
        tests = runner.select_tests(["bf16x2_to_s2f6x2__scaled*"])
        self.assertEqual(len(tests), 2)
        for test in tests:
            source_c = test.sweeps[0].c
            self.assertEqual((source_c.start, source_c.maximum, source_c.stride), (0xDEADBEEF,) * 2 + (0,))

    def test_every_declared_range_uses_the_global_stride(self) -> None:
        for test in runner.TESTS:
            for sweep in test.sweeps:
                for value_range in (sweep.a, sweep.b, sweep.c):
                    if (value_range.start, value_range.maximum) == (0, 0xFFFFFFFF):
                        self.assertEqual(value_range.stride, 0xFFFFFF)
                    if (value_range.start, value_range.maximum) == (0, 0xFFFF):
                        self.assertEqual(value_range.stride, 0xFF)
                    if (value_range.start, value_range.maximum) == (0, 0xFFFF0000):
                        self.assertEqual(value_range.stride, 0x00FF0000)

    def test_shards_cover_without_gaps_or_overlap(self) -> None:
        for total in (1, 2, 17, 2**32):
            for shard_count in (1, 3, 16):
                slices = [
                    runner.shard_slice(total, index, shard_count)
                    for index in range(shard_count)
                ]
                cursor = 0
                for start, count in slices:
                    self.assertEqual(start, cursor)
                    cursor += count
                self.assertEqual(cursor, total)

    def test_manifest_names_do_not_collide_between_families(self) -> None:
        add = runner.select_tests(["mixed_add__f16__rn__nosat"])
        fp6 = runner.select_tests(["fp16x2_to_f6x2*"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            add_manifest = runner.write_manifest(output, add, "smoke", 0, 1, [])
            fp6_manifest = runner.write_manifest(output, fp6, "smoke", 0, 1, [])
            self.assertNotEqual(add_manifest, fp6_manifest)
            self.assertTrue(add_manifest.exists())
            self.assertTrue(fp6_manifest.exists())

    def test_full_fp6_plan_uses_258_strided_inputs(self) -> None:
        tests = runner.select_tests(["fp16x2_to_f6x2*"])
        full = runner.projected_bytes(tests, "full", 0, 1, None)
        shard = runner.projected_bytes(tests, "full", 0, 16, None)
        self.assertEqual(full, 35_072)
        self.assertEqual(shard, 4_096)
        with redirect_stdout(io.StringIO()):
            runner.print_plan(tests, Path("/tmp/results"), "full", 0, 16, None)

    def test_bounded_conversion_selection_uses_global_strides(self) -> None:
        tests = bounded.selected_tests()
        self.assertEqual(bounded.ARCH, "compute_121a")
        self.assertEqual(len(tests), 20)
        self.assertEqual(sum(test.sweeps[0].count == 258 for test in tests), 16)
        self.assertEqual(sum(test.sweeps[0].count == 66_564 for test in tests), 4)
        self.assertTrue(all(test.min_cuda <= (13, 1) for test in tests))
        self.assertEqual(
            runner.projected_bytes(tests, "full", 0, 1, None),
            4_331_264,
        )
        self.assertAlmostEqual(
            runner.projected_bytes(tests, "full", 0, 16, None) / 1024**3,
            0.00025653743743896484,
        )

    def test_all_strided_matrix_covers_every_supported_gb10_test(self) -> None:
        cuda_131 = all_strided.selected_tests((13, 1))
        cuda_132 = all_strided.selected_tests((13, 2))
        self.assertEqual((len(cuda_131), len(all_strided.full_runs(cuda_131))), (73, 125))
        self.assertEqual((len(cuda_132), len(all_strided.full_runs(cuda_132))), (85, 137))
        total_131 = sum(
            runner.projected_bytes(cuda_131, "full", index, 16, None)
            for index in range(16)
        )
        self.assertEqual(total_131, 13_793_736_896)

    def test_integrity_manifest_rejects_payload_mutation(self) -> None:
        test = runner.select_tests(
            ["fp16x2_to_f6x2__f16x2__e2m3x2__norelu"]
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path, _manifest = write_synthetic_result(output, test)
            all_strided.validate_shard_manifest(output, [test], 0)
            with path.open("r+b") as stream:
                stream.seek(runner.HEADER_SIZE + 12)
                original = stream.read(1)
                stream.seek(runner.HEADER_SIZE + 12)
                stream.write(bytes([original[0] ^ 0x01]))
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                all_strided.validate_shard_manifest(output, [test], 0)

    def test_integrity_manifest_rejects_wrong_test_id(self) -> None:
        test = runner.select_tests(
            ["fp16x2_to_f6x2__f16x2__e2m3x2__norelu"]
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_synthetic_result(output, test, test_id=999)
            with self.assertRaisesRegex(RuntimeError, "test_id"):
                all_strided.validate_shard_manifest(output, [test], 0)

    def test_seal_refuses_to_relabel_legacy_provenance(self) -> None:
        test = runner.select_tests(
            ["fp16x2_to_f6x2__f16x2__e2m3x2__norelu"]
        )[0]
        changed = dataclasses.replace(test, ptx="totally.different.ptx")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_synthetic_result(output, test, sealed=False)
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                all_strided.seal_shard_manifest(output, [changed], 0)

    def test_seal_upgrades_legacy_manifest_without_relabeling(self) -> None:
        test = runner.select_tests(
            ["fp16x2_to_f6x2__f16x2__e2m3x2__norelu"]
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _binary, manifest = write_synthetic_result(output, test, sealed=False)
            all_strided.seal_shard_manifest(output, [test], 0)
            payload = runner.json.loads(manifest.read_text())
            self.assertEqual(payload["manifest_version"], 2)
            self.assertEqual(payload["matrix_sha256"], runner.matrix_sha256([test]))
            self.assertEqual(payload["result_files"][0]["ptx"], test.ptx)
            self.assertEqual(len(payload["result_files"][0]["sha256"]), 64)

    def test_numerical_coverage_requires_bound_fp6_matrix(self) -> None:
        tests = all_strided.selected_tests((13, 1))
        fp6_tests = [
            test for test in tests if test.name.startswith("fp16x2_to_f6x2__")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "fp6-reference.json"
            payload = {
                "status": "PASS",
                "test_count": len(fp6_tests),
                "matrix_sha256": runner.matrix_sha256(fp6_tests),
                "reference": {"lanes": 4_128},
            }
            report.write_text(runner.json.dumps(payload))
            coverage = all_strided.numerical_reference_coverage(report, tests)
            self.assertEqual(coverage["status"], "PARTIAL_REFERENCE_PASS")
            self.assertEqual(coverage["test_count"], 8)
            payload["matrix_sha256"] = "0" * 64
            report.write_text(runner.json.dumps(payload))
            rejected = all_strided.numerical_reference_coverage(report, tests)
            self.assertEqual(rejected["status"], "NOT_INDEPENDENTLY_VALIDATED")

    def test_bounded_full_runs_missing_precheck_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = [
                "run_gb10_bounded_conversions.py",
                "full",
                "--yes-large",
                "--output-dir",
                str(root / "full"),
                "--precheck-dir",
                str(root / "precheck"),
                "--nvcc",
                sys.executable,
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(bounded, "print_plan", return_value=([], [], 0)),
                mock.patch.object(bounded, "competing_accuracy_processes", return_value=[]),
                mock.patch.object(bounded, "configure_compatibility"),
                mock.patch.object(bounded, "run_precheck", return_value=root / "runner") as precheck,
                mock.patch.object(bounded, "precheck_evidence", return_value=({}, "sha256")),
                mock.patch.object(bounded, "shard_complete", return_value=True),
                mock.patch.object(bounded, "write_report", return_value=root / "report.json"),
            ):
                with redirect_stdout(io.StringIO()):
                    bounded.main()
            precheck.assert_called_once()


if __name__ == "__main__":
    unittest.main()
