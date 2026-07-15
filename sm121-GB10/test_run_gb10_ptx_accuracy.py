#!/usr/bin/env python3
"""CPU-only contract tests for the GB10 PTX accuracy runner."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_gb10_ptx_accuracy.py")
SPEC = importlib.util.spec_from_file_location("gb10_ptx_runner", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def values(value_range: object) -> tuple[int, ...]:
    return tuple(
        min(value_range.maximum, value_range.start + index * value_range.stride)
        for index in range(value_range.count)
    )


class RunnerContractTests(unittest.TestCase):
    def test_matrix_is_unique_and_complete(self) -> None:
        runner.validate_test_matrix(runner.TESTS)
        self.assertEqual(len(runner.TESTS), 85)

    def test_fp16x2_to_f6x2_expands_eight_full_b16_results(self) -> None:
        tests = runner.select_tests(["fp16x2_to_f6x2*"])
        self.assertEqual(len(tests), 8)
        self.assertEqual({test.mask for test in tests}, {0xFFFF})
        self.assertEqual({test.sweeps[0].b.count for test in tests}, {2**32})
        self.assertEqual({test.sweeps[0].c.start for test in tests}, {0xDEADBEEF})

    def test_comments_ranges_for_add_and_fma(self) -> None:
        add_first, add_second = runner.ADD_SWEEPS
        self.assertEqual((add_first.a.maximum, add_first.a.stride), (0xFFFF, 0xFF))
        self.assertEqual((add_first.c.maximum, add_first.c.stride), (0xFFFFFFFF, 1))
        self.assertEqual((add_second.a.maximum, add_second.a.stride), (0xFFFF, 1))
        self.assertEqual((add_second.c.maximum, add_second.c.stride), (0xFFFFFFFF, 0xFFFF))

        fma_first, fma_second, fma_third = runner.FMA_SWEEPS
        self.assertEqual(fma_first.c.stride, 1)
        self.assertEqual(fma_second.c.stride, 0xFFFFFF)
        self.assertEqual(fma_third.c.stride, 0xFFFF)

    def test_scaled_bf16_to_s2f6_uses_fixed_comment_source_c(self) -> None:
        tests = runner.select_tests(["bf16x2_to_s2f6x2__scaled*"])
        self.assertEqual(len(tests), 2)
        for test in tests:
            source_c = test.sweeps[0].c
            self.assertEqual((source_c.start, source_c.maximum, source_c.stride), (0xDEADBEEF,) * 2 + (0,))

    def test_smoke_preserves_lattice_and_distinguishes_add_sweeps(self) -> None:
        sparse_smoke = runner.ADD_SWEEPS[0].a.smoke()
        full_smoke = runner.ADD_SWEEPS[1].a.smoke()
        self.assertEqual(sparse_smoke.stride % runner.U16_SPARSE_FF.stride, 0)
        self.assertNotEqual(values(sparse_smoke), values(full_smoke))

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

    def test_full_fp6_plan_is_512_gib_and_shard_is_32_gib(self) -> None:
        tests = runner.select_tests(["fp16x2_to_f6x2*"])
        full = runner.projected_bytes(tests, "full", 0, 1, None)
        shard = runner.projected_bytes(tests, "full", 0, 16, None)
        self.assertEqual(full, 8 * (runner.HEADER_SIZE + 2**32 * runner.RECORD_SIZE))
        self.assertEqual(shard, 8 * (runner.HEADER_SIZE + 2**28 * runner.RECORD_SIZE))
        with redirect_stdout(io.StringIO()):
            runner.print_plan(tests, Path("/tmp/results"), "full", 0, 16, None)


if __name__ == "__main__":
    unittest.main()
