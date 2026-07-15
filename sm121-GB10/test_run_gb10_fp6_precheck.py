#!/usr/bin/env python3
"""CPU-only contract tests for the unified GB10 FP6 precheck."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_gb10_ptx_accuracy", ROOT / "run_gb10_ptx_accuracy.py"
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)

PRECHECK_SPEC = importlib.util.spec_from_file_location(
    "gb10_fp6_precheck", ROOT / "run_gb10_fp6_precheck.py"
)
assert PRECHECK_SPEC and PRECHECK_SPEC.loader
precheck = importlib.util.module_from_spec(PRECHECK_SPEC)
sys.modules[PRECHECK_SPEC.name] = precheck
PRECHECK_SPEC.loader.exec_module(precheck)


class FP6PrecheckContractTests(unittest.TestCase):
    def test_strided_reference_plan_is_bounded(self) -> None:
        tests = runner.select_tests([precheck.TEST_PATTERN])
        self.assertEqual(precheck.SAMPLE_RECORDS, 258)
        self.assertEqual(precheck.EXPECTED_REFERENCE_LANES, 4_128)
        total = precheck.projected_bytes(tests)
        self.assertEqual(total, 72_704)

    def test_software_reference_known_encodings(self) -> None:
        e2m3_f16 = precheck.expected_table("f16x2", "e2m3x2", False)
        e2m3_f16_relu = precheck.expected_table("f16x2", "e2m3x2", True)
        e3m2_bf16 = precheck.expected_table("bf16x2", "e3m2x2", False)

        self.assertEqual(e2m3_f16[0x0000], 0x00)  # +0
        self.assertEqual(e2m3_f16[0x8000], 0x20)  # -0
        self.assertEqual(e2m3_f16_relu[0x8000], 0x00)
        self.assertEqual(e2m3_f16[0x3C00], 0x08)  # +1
        self.assertEqual(e2m3_f16[0xBC00], 0x28)  # -1
        self.assertEqual(e2m3_f16_relu[0xBC00], 0x00)
        self.assertEqual(e2m3_f16[0x7C00], 0x1F)  # +inf -> +MAX_NORM
        self.assertEqual(e2m3_f16[0xFC00], 0x3F)  # -inf -> -MAX_NORM
        self.assertEqual(e2m3_f16[0x7E00], 0x1F)  # NaN -> +MAX_NORM
        self.assertEqual(e2m3_f16[0x3C40], 0x08)  # midpoint -> even LSB
        self.assertEqual(e2m3_f16[0x3CC0], 0x0A)  # midpoint -> even LSB
        self.assertEqual(e3m2_bf16[0x3F80], 0x0C)  # +1

    def test_all_reference_tables_cover_every_16_bit_input(self) -> None:
        for source in ("f16x2", "bf16x2"):
            for dtype in ("e2m3x2", "e3m2x2"):
                for relu in (False, True):
                    table = precheck.expected_table(source, dtype, relu)
                    self.assertEqual(len(table), 65_536)
                    self.assertTrue(all(0 <= value <= 0x3F for value in table))


if __name__ == "__main__":
    unittest.main()
