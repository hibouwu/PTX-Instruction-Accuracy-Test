#!/usr/bin/env python3
"""Generate, build, run, and validate the GB10 PTX accuracy tests.

The green/GB10 rows in the repository screenshots are the source of truth.  The
script expands PTX grammar metavariables (for example ``f6x2type`` and
``{.relu}``) into concrete instructions, generates one CUDA runner, builds it
for sm_121a, and writes fixed-width binary golden records.

Generated result files can be compared bit-for-bit with an existing result
directory by passing ``--reference-dir``.  Without a reference directory the
GB10 output is captured as golden data and its binary structure is validated;
that mode intentionally does not claim an independent numerical comparison.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_GENERATED_DIR = ROOT / "generated"
DEFAULT_BUILD_DIR = ROOT / "build"
DEFAULT_OUTPUT_DIR = ROOT / "results"

HEADER_SIZE = 0
RECORD_SIZE = 16
RECORD_STRUCT = struct.Struct("<IIII")
LARGE_OUTPUT_BYTES = 16 * 1024**3


@dataclasses.dataclass(frozen=True)
class ValueRange:
    start: int
    maximum: int
    stride: int

    @property
    def count(self) -> int:
        if self.stride == 0:
            return 1
        # Include the requested endpoint even when the stride does not land on
        # it naturally.  The generated runner clamps the last value to maximum.
        return (self.maximum - self.start + self.stride - 1) // self.stride + 1

    def smoke(self) -> "ValueRange":
        if self.count <= 4:
            return self
        # Preserve the original value lattice.  Replacing the stride from the
        # numeric span made distinct full sweeps collapse to identical smoke
        # inputs (for example ADD Test 1 and Test 2).
        index_step = (self.count - 1 + 2) // 3
        return ValueRange(self.start, self.maximum, self.stride * index_step)


@dataclasses.dataclass(frozen=True)
class Sweep:
    name: str
    a: ValueRange
    b: ValueRange
    c: ValueRange

    @property
    def count(self) -> int:
        return self.a.count * self.b.count * self.c.count

    def smoke(self) -> "Sweep":
        return Sweep(self.name + "-smoke", self.a.smoke(), self.b.smoke(), self.c.smoke())


@dataclasses.dataclass(frozen=True)
class Test:
    name: str
    ptx: str
    kind: str
    mask: int
    sweeps: tuple[Sweep, ...]
    min_cuda: tuple[int, int] = (13, 0)


# Test-spec global strides.  The endpoint is still included by ValueRange,
# even when the stride does not land on it naturally.
U32 = ValueRange(0, 0xFFFFFFFF, 0x00FFFFFF)
U32_SPARSE = U32
U16 = ValueRange(0, 0xFFFF, 0xFF)
U16_SPARSE_FF = U16
FIXED_ZERO = ValueRange(0, 0, 0)
FIXED_DEADBEEF = ValueRange(0xDEADBEEF, 0xDEADBEEF, 0)
SCALE_C = ValueRange(0, 0xFFFF0000, 0x00FF0000)

F32_PAIR = Sweep("f32-pair", U32, U32, FIXED_DEADBEEF)
F32_PAIR_SCALED = Sweep("f32-pair-scale", U32, U32, SCALE_C)
PACKED_B32 = Sweep("packed-b32", FIXED_ZERO, U32, FIXED_DEADBEEF)
PACKED_B32_SCALED = Sweep("packed-b32-scale", FIXED_ZERO, U32, SCALE_C)
PACKED_B32_FIXED_SCALE = Sweep("packed-b32-fixed-scale", FIXED_ZERO, U32, FIXED_DEADBEEF)
PACKED_B16 = Sweep("packed-b16", FIXED_ZERO, U16, FIXED_DEADBEEF)
PACKED_B16_SCALED = Sweep("packed-b16-scale", FIXED_ZERO, U16, SCALE_C)

ADD_SWEEPS = (
    Sweep("a-sparse-c-full", U16_SPARSE_FF, FIXED_ZERO, U32),
    Sweep("a-full-c-sparse", U16, FIXED_ZERO, U32),
)

FMA_SWEEPS = (
    Sweep("a-sparse-b-sparse-c-full", U16_SPARSE_FF, U16_SPARSE_FF, U32),
    Sweep(
        "a-sparse-b-full-c-sparse-ffffff",
        U16_SPARSE_FF,
        U16,
        U32,
    ),
    Sweep(
        "a-full-b-sparse-c-sparse-ffff",
        U16,
        U16_SPARSE_FF,
        U32,
    ),
)


def tagged(base: str, *modifiers: str) -> str:
    return base + "__" + "__".join(modifiers)


def build_tests() -> list[Test]:
    tests: list[Test] = []

    # image.png: FP32 pair conversions.
    for dtype in ("e2m3x2", "e3m2x2"):
        ptx = f"cvt.rn.satfinite.{dtype}.f32"
        tests.append(Test(tagged("f32_to_f6x2", dtype), ptx, "f32_pair_h", 0xFFFF, (F32_PAIR,)))

    tests.append(
        Test(
            tagged("f32_to_f4x2", "e2m1x2"),
            "cvt.rn.satfinite.e2m1x2.f32",
            "f32_pair_b8",
            0xFFFF,  # The table requests comparison of the lower 16 bits.
            (F32_PAIR,),
        )
    )

    for rounding in ("rz", "rp"):
        for satfinite in (False, True):
            sat = ".satfinite" if satfinite else ""
            ptx = f"cvt.{rounding}{sat}.ue8m0x2.f32"
            tests.append(
                Test(
                    tagged("f32_to_ue8m0x2", rounding, "satfinite" if satfinite else "nosat"),
                    ptx,
                    "f32_pair_h",
                    0xFFFF,
                    (F32_PAIR,),
                )
            )

    for relu in (False, True):
        relu_suffix = ".relu" if relu else ""
        ptx = f"cvt.rn.satfinite{relu_suffix}.scaled::n2::ue8m0.s2f6x2.f32"
        tests.append(
            Test(
                tagged("f32_to_s2f6x2_scaled", "relu" if relu else "norelu"),
                ptx,
                "f32_pair_h_scale",
                0xFFFF,
                (F32_PAIR_SCALED,),
                (13, 1),
            )
        )

    # image-1.png: packed conversions. f6x2type/fp16x2type and
    # f4x2type/fp16x2type are grammar metavariables and must be expanded.
    for dtype in ("e2m3x2", "e3m2x2"):
        for source in ("f16x2", "bf16x2"):
            for relu in (False, True):
                relu_suffix = ".relu" if relu else ""
                ptx = f"cvt.rn.satfinite{relu_suffix}.{dtype}.{source}"
                tests.append(
                    Test(
                        tagged("fp16x2_to_f6x2", source, dtype, "relu" if relu else "norelu"),
                        ptx,
                        "packed32_h",
                        # e2m3x2/e3m2x2 use a b16 destination containing two
                        # padded b8 lanes.  Keeping only 0xff silently drops
                        # the upper lane.
                        0xFFFF,
                        (PACKED_B32,),
                        (13, 1),
                    )
                )

    for source in ("f16x2", "bf16x2"):
        for relu in (False, True):
            relu_suffix = ".relu" if relu else ""
            ptx = f"cvt.rn.satfinite{relu_suffix}.e2m1x2.{source}"
            tests.append(
                Test(
                    tagged("fp16x2_to_f4x2", source, "relu" if relu else "norelu"),
                    ptx,
                    "packed32_b8",
                    0xFF,
                    (PACKED_B32,),
                    (13, 1),
                )
            )

    for rounding in ("rz", "rp"):
        for satfinite in (False, True):
            sat = ".satfinite" if satfinite else ""
            ptx = f"cvt.{rounding}{sat}.ue8m0x2.bf16x2"
            tests.append(
                Test(
                    tagged("bf16x2_to_ue8m0x2", rounding, "satfinite" if satfinite else "nosat"),
                    ptx,
                    "packed32_h",
                    0xFFFF,
                    (PACKED_B32,),
                )
            )

    for scaled in (False, True):
        for relu in (False, True):
            relu_suffix = ".relu" if relu else ""
            scale_suffix = ".scaled::n2::ue8m0" if scaled else ""
            ptx = f"cvt.rn.satfinite{relu_suffix}{scale_suffix}.s2f6x2.bf16x2"
            tests.append(
                Test(
                    tagged(
                        "bf16x2_to_s2f6x2",
                        "scaled" if scaled else "unscaled",
                        "relu" if relu else "norelu",
                    ),
                    ptx,
                    "packed32_h_scale" if scaled else "packed32_h",
                    0xFFFF,
                    # The table fixes Source C to 0xdeadbeef for this row, so
                    # the optional scale operand is 0xdead rather than a full
                    # 16-bit scale sweep.
                    (PACKED_B32_FIXED_SCALE if scaled else PACKED_B32,),
                    (13, 1),
                )
            )

    for source in ("e2m3x2", "e3m2x2"):
        ptx = f"cvt.rn.f16x2.{source}"
        tests.append(Test(tagged("f6x2_to_f16x2", source), ptx, "packed16_r", 0xFFFFFFFF, (PACKED_B16,)))

    tests.append(
        Test(
            tagged("f4x2_to_f16x2", "e2m1x2"),
            "cvt.rn.f16x2.e2m1x2",
            "packed8_r",
            0xFFFFFFFF,
            (PACKED_B16,),
        )
    )

    for source in ("e2m3x2", "e3m2x2"):
        for relu in (False, True):
            for satfinite in (False, True):
                relu_suffix = ".relu" if relu else ""
                sat_suffix = ".satfinite" if satfinite else ""
                ptx = (
                    f"cvt.rn{relu_suffix}{sat_suffix}.scaled::n2::ue8m0."
                    f"bf16x2.{source}"
                )
                tests.append(
                    Test(
                        tagged(
                            "f6x2_to_bf16x2_scaled",
                            source,
                            "relu" if relu else "norelu",
                            "satfinite" if satfinite else "nosat",
                        ),
                        ptx,
                        "packed16_r_scale",
                        0xFFFFFFFF,
                        (PACKED_B16_SCALED,),
                        (13, 2),
                    )
                )

    for relu in (False, True):
        for satfinite in (False, True):
            relu_suffix = ".relu" if relu else ""
            sat_suffix = ".satfinite" if satfinite else ""
            ptx = (
                f"cvt.rn{sat_suffix}{relu_suffix}.scaled::n2::ue8m0."
                "bf16x2.s2f6x2"
            )
            tests.append(
                Test(
                    tagged(
                        "s2f6x2_to_bf16x2_scaled",
                        "relu" if relu else "norelu",
                        "satfinite" if satfinite else "nosat",
                    ),
                    ptx,
                    "packed16_r_scale",
                    0xFFFFFFFF,
                    (PACKED_B16_SCALED,),
                    (13, 1),
                )
            )

    tests.append(
        Test(
            "ue8m0x2_to_bf16x2__rn",
            "cvt.rn.bf16x2.ue8m0x2",
            "packed16_r",
            0xFFFFFFFF,
            (PACKED_B16,),
        )
    )

    for relu in (False, True):
        for satfinite in (False, True):
            relu_suffix = ".relu" if relu else ""
            sat_suffix = ".satfinite" if satfinite else ""
            ptx = (
                f"cvt.rn{relu_suffix}{sat_suffix}.scaled::n2::ue8m0."
                "bf16x2.e2m1x2"
            )
            tests.append(
                Test(
                    tagged(
                        "f4x2_to_bf16x2_scaled",
                        "relu" if relu else "norelu",
                        "satfinite" if satfinite else "nosat",
                    ),
                    ptx,
                    "packed8_r_scale",
                    0xFFFFFFFF,
                    (PACKED_B16_SCALED,),
                    (13, 2),
                )
            )

    # image-2.png: mixed-precision add/fma.
    for source in ("f16", "bf16"):
        for rounding in (None, "rn", "rz", "rm", "rp"):
            for sat in (False, True):
                round_suffix = f".{rounding}" if rounding else ""
                sat_suffix = ".sat" if sat else ""
                ptx = f"add{round_suffix}.f32.{source}{sat_suffix}"
                tests.append(
                    Test(
                        tagged(
                            "mixed_add",
                            source,
                            rounding or "default_rn",
                            "sat" if sat else "nosat",
                        ),
                        ptx,
                        "mixed_add_f",
                        0xFFFFFFFF,
                        ADD_SWEEPS,
                    )
                )

    for source in ("f16", "bf16"):
        for rounding in ("rn", "rz", "rm", "rp"):
            for sat in (False, True):
                sat_suffix = ".sat" if sat else ""
                ptx = f"fma.{rounding}.f32.{source}{sat_suffix}"
                tests.append(
                    Test(
                        tagged("mixed_fma", source, rounding, "sat" if sat else "nosat"),
                        ptx,
                        "mixed_fma_f",
                        0xFFFFFFFF,
                        FMA_SWEEPS,
                    )
                )

    names = [test.name for test in tests]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate generated test names")
    return tests


TESTS = build_tests()


def validate_test_matrix(tests: Sequence[Test]) -> None:
    if len(tests) != 85:
        raise AssertionError(f"expected 85 concrete GB10 tests, found {len(tests)}")
    names = [test.name for test in tests]
    ptx = [test.ptx for test in tests]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate test names")
    if len(ptx) != len(set(ptx)):
        raise AssertionError("duplicate concrete PTX instructions")
    for test in tests:
        if not 0 <= test.mask <= 0xFFFFFFFF:
            raise AssertionError(f"invalid result mask for {test.name}")
        if not test.sweeps:
            raise AssertionError(f"test has no sweeps: {test.name}")
        for sweep in test.sweeps:
            for label, value_range in (
                ("a", sweep.a),
                ("b", sweep.b),
                ("c", sweep.c),
            ):
                if value_range.start > value_range.maximum:
                    raise AssertionError(f"{test.name}/{sweep.name}: {label} start > maximum")
                if value_range.stride < 0:
                    raise AssertionError(f"{test.name}/{sweep.name}: negative {label} stride")
                if value_range.stride == 0 and value_range.start != value_range.maximum:
                    raise AssertionError(
                        f"{test.name}/{sweep.name}: zero {label} stride for non-fixed range"
                    )
            if not 0 < sweep.count <= 0xFFFFFFFFFFFFFFFF:
                raise AssertionError(f"invalid sweep size: {test.name}/{sweep.name}")

    fp6_outputs = [test for test in tests if test.name.startswith("fp16x2_to_f6x2__")]
    if len(fp6_outputs) != 8 or any(test.mask != 0xFFFF for test in fp6_outputs):
        raise AssertionError("FP16/BF16-to-FP6x2 must preserve both padded b8 lanes")


validate_test_matrix(TESTS)


CU_TEMPLATE = r'''
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                \
  cudaError_t error__ = (call);                                              \
  if (error__ != cudaSuccess) {                                              \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                 cudaGetErrorString(error__));                               \
    std::exit(3);                                                            \
  }                                                                          \
} while (0)

static constexpr int kTestCount = @@TEST_COUNT@@;
static constexpr char const* kTestNames[kTestCount] = {
@@TEST_NAMES@@
};
static constexpr uint32_t kTestMasks[kTestCount] = {
@@TEST_MASKS@@
};

#pragma pack(push, 1)
struct Record {
  uint32_t source_a;
  uint32_t source_b;
  uint32_t source_c;
  uint32_t result;
};
#pragma pack(pop)

static_assert(sizeof(Record) == 16, "binary record must be 16 bytes");

struct RangeSpec {
  uint64_t start;
  uint64_t maximum;
  uint64_t stride;
  uint64_t count;
};

__host__ __device__ uint32_t range_value(RangeSpec range, uint64_t index) {
  if (range.count <= 1 || range.stride == 0) {
    return static_cast<uint32_t>(range.start);
  }
  uint64_t value = range.start + index * range.stride;
  return static_cast<uint32_t>(value > range.maximum ? range.maximum : value);
}

__global__ void accuracy_kernel(int test_id,
                                uint32_t result_mask,
                                uint64_t global_start,
                                uint64_t record_count,
                                RangeSpec a_range,
                                RangeSpec b_range,
                                RangeSpec c_range,
                                Record* records) {
  uint64_t local = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= record_count) {
    return;
  }

  uint64_t linear = global_start + local;
  uint64_t c_index = linear % c_range.count;
  linear /= c_range.count;
  uint64_t b_index = linear % b_range.count;
  linear /= b_range.count;
  uint64_t a_index = linear % a_range.count;

  uint32_t source_a = range_value(a_range, a_index);
  uint32_t source_b = range_value(b_range, b_index);
  uint32_t source_c = range_value(c_range, c_index);
  float fa = __uint_as_float(source_a);
  float fb = __uint_as_float(source_b);
  float fc = __uint_as_float(source_c);
  uint16_t a16 = static_cast<uint16_t>(source_a);
  uint16_t b16 = static_cast<uint16_t>(source_b);
  uint16_t scale = static_cast<uint16_t>(source_c >> 16);
  uint32_t result = 0;

  switch (test_id) {
@@SWITCH_CASES@@
    default:
      result = 0;
      break;
  }

  records[local] = Record{source_a, source_b, source_c, result & result_mask};
}

uint64_t parse_u64(char const* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') {
    std::fprintf(stderr, "invalid integer: %s\n", text);
    std::exit(2);
  }
  return static_cast<uint64_t>(value);
}

RangeSpec parse_range(char** argv, int offset) {
  return RangeSpec{
      parse_u64(argv[offset]),
      parse_u64(argv[offset + 1]),
      parse_u64(argv[offset + 2]),
      parse_u64(argv[offset + 3]),
  };
}

int main(int argc, char** argv) {
  if (argc != 19) {
    std::fprintf(stderr,
        "usage: %s test_id start count total "
        "a_start a_max a_stride a_count b_start b_max b_stride b_count "
        "c_start c_max c_stride c_count chunk_records output.bin\n", argv[0]);
    return 2;
  }

  int test_id = static_cast<int>(parse_u64(argv[1]));
  uint64_t shard_start = parse_u64(argv[2]);
  uint64_t shard_count = parse_u64(argv[3]);
  uint64_t total_records = parse_u64(argv[4]);
  RangeSpec a_range = parse_range(argv, 5);
  RangeSpec b_range = parse_range(argv, 9);
  RangeSpec c_range = parse_range(argv, 13);
  uint64_t chunk_records = parse_u64(argv[17]);
  char const* output_path = argv[18];

  if (test_id < 0 || test_id >= kTestCount || chunk_records == 0 ||
      shard_start > total_records || shard_count > total_records - shard_start) {
    std::fprintf(stderr, "invalid test/range arguments\n");
    return 2;
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 12 || properties.minor != 1) {
    std::fprintf(stderr, "GB10 SM121 required; detected %s (sm_%d%d)\n",
                 properties.name, properties.major, properties.minor);
    return 4;
  }

  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::fprintf(stderr, "cannot open output: %s\n", output_path);
    return 5;
  }
  uint64_t allocation_count = std::min(chunk_records, std::max<uint64_t>(1, shard_count));
  Record* device_records = nullptr;
  CUDA_CHECK(cudaMalloc(&device_records, allocation_count * sizeof(Record)));
  std::vector<Record> host_records(allocation_count);

  for (uint64_t done = 0; done < shard_count;) {
    uint64_t current = std::min(allocation_count, shard_count - done);
    int threads = 256;
    int blocks = static_cast<int>((current + threads - 1) / threads);
    accuracy_kernel<<<blocks, threads>>>(
        test_id, kTestMasks[test_id], shard_start + done, current,
        a_range, b_range, c_range, device_records);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_records.data(), device_records,
                          current * sizeof(Record), cudaMemcpyDeviceToHost));
    output.write(reinterpret_cast<char const*>(host_records.data()),
                 current * sizeof(Record));
    if (!output) {
      std::fprintf(stderr, "failed while writing: %s\n", output_path);
      CUDA_CHECK(cudaFree(device_records));
      return 5;
    }
    done += current;
  }

  CUDA_CHECK(cudaFree(device_records));
  output.flush();
  if (!output) {
    std::fprintf(stderr, "failed while flushing: %s\n", output_path);
    return 5;
  }
  output.close();
  std::printf("test=%s records=%llu output=%s\n", kTestNames[test_id],
              static_cast<unsigned long long>(shard_count), output_path);
  return 0;
}
'''


def cpp_quote(value: str) -> str:
    return json.dumps(value)


def asm_case(index: int, test: Test) -> str:
    instruction = test.ptx
    if test.kind == "f32_pair_h":
        body = (
            f'uint16_t out; asm volatile("{instruction} %0, %1, %2;" '
            ': "=h"(out) : "f"(fa), "f"(fb)); result = out;'
        )
    elif test.kind == "f32_pair_b8":
        body = (
            'uint16_t out; asm volatile("{ .reg .b8 tmp; '
            f'{instruction} tmp, %1, %2; mov.b16 %0, {{tmp, 0}}; }}" '
            ': "=h"(out) : "f"(fa), "f"(fb)); result = out;'
        )
    elif test.kind == "f32_pair_h_scale":
        body = (
            f'uint16_t out; asm volatile("{instruction} %0, %1, %2, %3;" '
            ': "=h"(out) : "f"(fa), "f"(fb), "h"(scale)); result = out;'
        )
    elif test.kind == "packed32_h":
        body = (
            f'uint16_t out; asm volatile("{instruction} %0, %1;" '
            ': "=h"(out) : "r"(source_b)); result = out;'
        )
    elif test.kind == "packed32_b8":
        body = (
            'uint16_t out; asm volatile("{ .reg .b8 tmp; '
            f'{instruction} tmp, %1; mov.b16 %0, {{tmp, 0}}; }}" '
            ': "=h"(out) : "r"(source_b)); result = out;'
        )
    elif test.kind == "packed32_h_scale":
        body = (
            f'uint16_t out; asm volatile("{instruction} %0, %1, %2;" '
            ': "=h"(out) : "r"(source_b), "h"(scale)); result = out;'
        )
    elif test.kind == "packed8_r":
        body = (
            "uint32_t out; asm volatile(\"{ .reg .b8 packed, zero; "
            f"mov.b16 {{packed, zero}}, %1; {instruction} %0, packed; }}\" "
            ": \"=r\"(out) : \"h\"(b16)); result = out;"
        )
    elif test.kind == "packed8_r_scale":
        body = (
            "uint32_t out; asm volatile(\"{ .reg .b8 packed, zero; "
            f"mov.b16 {{packed, zero}}, %1; {instruction} %0, packed, %2; }}\" "
            ": \"=r\"(out) : \"h\"(b16), \"h\"(scale)); result = out;"
        )
    elif test.kind == "packed16_r":
        body = (
            f'uint32_t out; asm volatile("{instruction} %0, %1;" '
            ': "=r"(out) : "h"(b16)); result = out;'
        )
    elif test.kind == "packed16_r_scale":
        body = (
            f'uint32_t out; asm volatile("{instruction} %0, %1, %2;" '
            ': "=r"(out) : "h"(b16), "h"(scale)); result = out;'
        )
    elif test.kind == "mixed_add_f":
        body = (
            f'float out; asm volatile("{instruction} %0, %1, %2;" '
            ': "=f"(out) : "h"(a16), "f"(fc)); result = __float_as_uint(out);'
        )
    elif test.kind == "mixed_fma_f":
        body = (
            f'float out; asm volatile("{instruction} %0, %1, %2, %3;" '
            ': "=f"(out) : "h"(a16), "h"(b16), "f"(fc)); result = __float_as_uint(out);'
        )
    else:
        raise ValueError(f"unsupported asm kind: {test.kind}")
    return f"    case {index}: {{ {body} break; }}"


def generate_cuda(tests: Sequence[Test], generated_dir: Path) -> Path:
    generated_dir.mkdir(parents=True, exist_ok=True)
    source = generated_dir / "gb10_ptx_accuracy_generated.cu"
    text = CU_TEMPLATE
    text = text.replace("@@TEST_COUNT@@", str(len(tests)))
    text = text.replace(
        "@@TEST_NAMES@@", "\n".join(f"  {cpp_quote(test.name)}," for test in tests)
    )
    text = text.replace(
        "@@TEST_MASKS@@", "\n".join(f"  0x{test.mask:08x}u," for test in tests)
    )
    text = text.replace(
        "@@SWITCH_CASES@@", "\n".join(asm_case(index, test) for index, test in enumerate(tests))
    )
    source.write_text(text)
    return source


def log(message: str) -> None:
    print(message, flush=True)


def run(
    command: Sequence[object],
    *,
    cwd: Path = ROOT,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = [str(item) for item in command]
    log("+ " + " ".join(args))
    process = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if capture and process.stdout:
        print(process.stdout, end="")
    if check and process.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {process.returncode}: {' '.join(args)}\n"
            f"{process.stdout or ''}"
        )
    return process


def nvcc_release(nvcc: str) -> tuple[int, int]:
    process = run([nvcc, "--version"])
    match = re.search(r"release\s+(\d+)\.(\d+)", process.stdout or "")
    if not match:
        raise RuntimeError("cannot determine nvcc release")
    return int(match.group(1)), int(match.group(2))


def compile_cuda(
    source: Path,
    binary: Path,
    tests: Sequence[Test],
    nvcc: str,
    arch: str,
) -> None:
    actual = nvcc_release(nvcc)
    required = max(test.min_cuda for test in tests)
    if actual < required:
        raise RuntimeError(
            f"selected instructions require CUDA {required[0]}.{required[1]}+, "
            f"but {nvcc} reports {actual[0]}.{actual[1]}"
        )
    binary.parent.mkdir(parents=True, exist_ok=True)
    command: list[object] = [nvcc, "-O3", "-std=c++17", "-lineinfo"]
    if arch.startswith("compute_"):
        # Embed family-specific PTX and defer stage-2 translation to the
        # driver.  This is required for PTX features whose ISA notes name
        # sm_*f but which ptxas cannot lower directly to the installed chip.
        command.extend([f"--gpu-architecture={arch}", f"--gpu-code={arch}"])
    else:
        command.append(f"-arch={arch}")
    command.extend([source, "-o", binary])
    run(command)


def select_tests(patterns: Sequence[str]) -> list[Test]:
    if not patterns:
        return list(TESTS)
    selected = [
        test
        for test in TESTS
        if any(fnmatch.fnmatch(test.name, pattern) or fnmatch.fnmatch(test.ptx, pattern) for pattern in patterns)
    ]
    if not selected:
        raise RuntimeError(f"no tests match: {', '.join(patterns)}")
    return selected


def range_args(value_range: ValueRange) -> list[str]:
    return [
        hex(value_range.start),
        hex(value_range.maximum),
        hex(value_range.stride),
        str(value_range.count),
    ]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def range_metadata(value_range: ValueRange) -> dict[str, int]:
    return {
        "start": value_range.start,
        "maximum": value_range.maximum,
        "stride": value_range.stride,
        "count": value_range.count,
    }


def sweep_spec(test: Test, sweep: Sweep) -> dict[str, object]:
    return {
        "test_name": test.name,
        "ptx": test.ptx,
        "kind": test.kind,
        "result_mask": test.mask,
        "sweep": sweep.name,
        "ranges": {
            "source_a": range_metadata(sweep.a),
            "source_b": range_metadata(sweep.b),
            "source_c": range_metadata(sweep.c),
        },
    }


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sweep_spec_sha256(test: Test, sweep: Sweep) -> str:
    return canonical_sha256(sweep_spec(test, sweep))


def matrix_sha256(tests: Sequence[Test]) -> str:
    return canonical_sha256(
        [sweep_spec(test, sweep) for test in tests for sweep in test.sweeps]
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def shard_slice(total: int, index: int, count: int) -> tuple[int, int]:
    start = total * index // count
    end = total * (index + 1) // count
    return start, end - start


def read_payload_layout(path: Path) -> dict[str, object]:
    """Validate and describe a headerless fixed-width record payload."""
    actual_size = path.stat().st_size
    if actual_size % RECORD_SIZE:
        raise RuntimeError(
            f"headerless binary size is not record-aligned: {path}: {actual_size}"
        )
    return {
        "header_size": 0,
        "record_size": RECORD_SIZE,
        "shard_records": actual_size // RECORD_SIZE,
        "bytes": actual_size,
    }


def validate_binary_inputs(
    path: Path,
    sweep: Sweep,
    shard_start: int,
    shard_records: int,
) -> str:
    """Verify every source tuple and return the SHA256 of the complete file."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        return _validate_binary_inputs_numpy(
            path, sweep, shard_start, shard_records, np
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise RuntimeError(f"truncated binary payload prefix: {path}")
        digest.update(header)
        for local_index in range(shard_records):
            raw = stream.read(RECORD_SIZE)
            if len(raw) != RECORD_SIZE:
                raise RuntimeError(f"truncated record {local_index}: {path}")
            digest.update(raw)
            source_a, source_b, source_c, _result = RECORD_STRUCT.unpack(raw)
            linear = shard_start + local_index
            c_index = linear % sweep.c.count
            linear //= sweep.c.count
            b_index = linear % sweep.b.count
            linear //= sweep.b.count
            a_index = linear % sweep.a.count
            expected = (
                min(sweep.a.maximum, sweep.a.start + a_index * sweep.a.stride),
                min(sweep.b.maximum, sweep.b.start + b_index * sweep.b.stride),
                min(sweep.c.maximum, sweep.c.start + c_index * sweep.c.stride),
            )
            actual = (source_a, source_b, source_c)
            if actual != expected:
                raise RuntimeError(
                    f"input enumeration mismatch in {path} at record {local_index}: "
                    f"actual={actual!r}, expected={expected!r}"
                )
        if stream.read(1):
            raise RuntimeError(f"trailing binary data: {path}")
    return digest.hexdigest()


def _validate_binary_inputs_numpy(
    path: Path,
    sweep: Sweep,
    shard_start: int,
    shard_records: int,
    np: object,
) -> str:
    """Vectorized full-record validator; ``np`` is injected for optional import."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise RuntimeError(f"truncated binary payload prefix: {path}")
    digest.update(header)
    records = np.memmap(
        path,
        dtype="<u4",
        mode="r",
        offset=HEADER_SIZE,
        shape=(shard_records, 4),
    )
    chunk_records = 4 * 1024 * 1024
    try:
        for begin in range(0, shard_records, chunk_records):
            end = min(shard_records, begin + chunk_records)
            chunk = records[begin:end]
            digest.update(chunk)
            linear = np.arange(
                shard_start + begin,
                shard_start + end,
                dtype=np.uint64,
            )
            c_index = linear % sweep.c.count
            linear //= sweep.c.count
            b_index = linear % sweep.b.count
            linear //= sweep.b.count
            a_index = linear % sweep.a.count
            expected_columns = (
                np.minimum(
                    sweep.a.maximum,
                    sweep.a.start + a_index * sweep.a.stride,
                ).astype(np.uint32),
                np.minimum(
                    sweep.b.maximum,
                    sweep.b.start + b_index * sweep.b.stride,
                ).astype(np.uint32),
                np.minimum(
                    sweep.c.maximum,
                    sweep.c.start + c_index * sweep.c.stride,
                ).astype(np.uint32),
            )
            for column, expected in enumerate(expected_columns):
                mismatches = np.flatnonzero(chunk[:, column] != expected)
                if mismatches.size:
                    local = begin + int(mismatches[0])
                    actual_tuple = tuple(int(value) for value in records[local, :3])
                    raise RuntimeError(
                        f"input enumeration mismatch in {path} at record {local}: "
                        f"actual={actual_tuple!r}"
                    )
    finally:
        del records
    return digest.hexdigest()


def compare_binary(actual: Path, reference: Path) -> None:
    if not reference.exists():
        raise RuntimeError(f"reference file missing: {reference}")
    actual_layout = read_payload_layout(actual)
    reference_layout = read_payload_layout(reference)
    if actual_layout["bytes"] != reference_layout["bytes"]:
        raise RuntimeError(
            f"reference payload size mismatch for {actual.name}: "
            f"actual={actual_layout['bytes']}, reference={reference_layout['bytes']}"
        )
    offset = 0
    with actual.open("rb") as actual_stream, reference.open("rb") as reference_stream:
        while True:
            actual_chunk = actual_stream.read(8 * 1024 * 1024)
            reference_chunk = reference_stream.read(8 * 1024 * 1024)
            if actual_chunk != reference_chunk:
                for index, (left, right) in enumerate(zip(actual_chunk, reference_chunk)):
                    if left != right:
                        raise RuntimeError(
                            f"binary mismatch: {actual.name} at byte {offset + index}: "
                            f"actual=0x{left:02x}, reference=0x{right:02x}"
                        )
                raise RuntimeError(f"binary mismatch: {actual.name} near byte {offset}")
            if not actual_chunk:
                break
            offset += len(actual_chunk)


def selected_runs(tests: Sequence[Test], profile: str) -> Iterable[tuple[int, Test, Sweep]]:
    for test_id, test in enumerate(tests):
        for sweep in test.sweeps:
            yield test_id, test, sweep.smoke() if profile == "smoke" else sweep


def projected_bytes(
    tests: Sequence[Test], profile: str, shard_index: int, shard_count: int, limit_records: int | None
) -> int:
    total_bytes = 0
    for _test_id, _test, sweep in selected_runs(tests, profile):
        _start, count = shard_slice(sweep.count, shard_index, shard_count)
        if limit_records is not None:
            count = min(count, limit_records)
        total_bytes += HEADER_SIZE + count * RECORD_SIZE
    return total_bytes


def execute_tests(
    binary: Path,
    tests: Sequence[Test],
    output_dir: Path,
    reference_dir: Path | None,
    profile: str,
    shard_index: int,
    shard_count: int,
    chunk_records: int,
    limit_records: int | None,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for test_id, test, sweep in selected_runs(tests, profile):
        start, count = shard_slice(sweep.count, shard_index, shard_count)
        if limit_records is not None:
            count = min(count, limit_records)
        test_directory = Path(safe_name(test.name))
        filename = (
            f"{safe_name(sweep.name)}__"
            f"shard-{shard_index:05d}-of-{shard_count:05d}.bin"
        )
        relative_path = test_directory / filename
        output = output_dir / relative_path
        partial_output = output.with_suffix(output.suffix + ".partial")
        output.parent.mkdir(parents=True, exist_ok=True)
        command: list[object] = [binary, test_id, start, count, sweep.count]
        command.extend(range_args(sweep.a))
        command.extend(range_args(sweep.b))
        command.extend(range_args(sweep.c))
        command.extend([chunk_records, partial_output])
        run(command)
        summary = read_payload_layout(partial_output)
        if summary["shard_records"] != count:
            raise RuntimeError(
                f"generated payload record mismatch for {partial_output}: "
                f"{summary['shard_records']} != {count}"
            )
        summary.update({
            "test_id": test_id,
            "test_name": test.name,
            "result_mask": test.mask,
            "total_records": sweep.count,
            "shard_start": start,
            "shard_records": count,
        })
        binary_sha256 = validate_binary_inputs(partial_output, sweep, start, count)
        summary.update(
            {
                "ptx": test.ptx,
                "sweep": sweep.name,
                "file": relative_path.as_posix(),
                "ranges": {
                    "source_a": range_metadata(sweep.a),
                    "source_b": range_metadata(sweep.b),
                    "source_c": range_metadata(sweep.c),
                },
                "spec_sha256": sweep_spec_sha256(test, sweep),
                "sha256": binary_sha256,
            }
        )
        if reference_dir is not None:
            compare_binary(partial_output, reference_dir / relative_path)
            summary["comparison"] = "pass"
            log(f"PASS {test.name} / {sweep.name}")
        else:
            summary["comparison"] = "golden-captured"
        partial_output.replace(output)
        summaries.append(summary)
    return summaries


def preflight_tests(binary: Path, tests: Sequence[Test]) -> None:
    with tempfile.TemporaryDirectory(prefix="gb10-ptx-preflight-") as temporary:
        temporary_dir = Path(temporary)
        for test_id, test in enumerate(tests):
            sweep = test.sweeps[0]
            output = temporary_dir / f"{safe_name(test.name)}.bin"
            command: list[object] = [binary, test_id, 0, 1, sweep.count]
            command.extend(range_args(sweep.a))
            command.extend(range_args(sweep.b))
            command.extend(range_args(sweep.c))
            command.extend([1, output])
            try:
                run(command)
                layout = read_payload_layout(output)
            except Exception as error:
                raise RuntimeError(
                    f"preflight failed for {test.name} ({test.ptx}); "
                    "do not start the full sweep until architecture, CUDA toolkit, "
                    f"and driver compatibility are resolved: {error}"
                ) from error
            if layout["shard_records"] != 1:
                raise RuntimeError(f"invalid preflight output for {test.name}")
            log(f"PREFLIGHT PASS {test.name}")


def write_manifest(
    output_dir: Path,
    tests: Sequence[Test],
    profile: str,
    shard_index: int,
    shard_count: int,
    summaries: Sequence[dict[str, object]],
) -> Path:
    manifest = manifest_path(output_dir, tests, shard_index, shard_count)
    payload = {
        "manifest_version": 3,
        "format": "GB10 PTX accuracy headerless records v2",
        "profile": profile,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "test_count": len(tests),
        "matrix_sha256": matrix_sha256(tests),
        "result_files": list(summaries),
    }
    temporary = manifest.with_suffix(manifest.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    return manifest


def manifest_path(
    output_dir: Path,
    tests: Sequence[Test],
    shard_index: int,
    shard_count: int,
) -> Path:
    families = {test.name.split("__", 1)[0] for test in tests}
    if len(families) == 1:
        selection = safe_name(next(iter(families)))
    elif len(tests) == len(TESTS):
        selection = "all"
    else:
        digest = hashlib.sha256("\n".join(test.name for test in tests).encode()).hexdigest()[:12]
        selection = f"selection-{digest}"
    return output_dir / (
        f"manifest-{selection}__shard-{shard_index:05d}-of-{shard_count:05d}.json"
    )


def print_plan(
    tests: Sequence[Test],
    output_dir: Path,
    profile: str,
    shard_index: int,
    shard_count: int,
    limit_records: int | None,
) -> None:
    total_bytes = 0
    for _test_id, test, sweep in selected_runs(tests, profile):
        start, count = shard_slice(sweep.count, shard_index, shard_count)
        if limit_records is not None:
            count = min(count, limit_records)
        size = HEADER_SIZE + count * RECORD_SIZE
        total_bytes += size
        filename = (
            f"{safe_name(sweep.name)}__"
            f"shard-{shard_index:05d}-of-{shard_count:05d}.bin"
        )
        print(f"{test.name}: {test.ptx}")
        print(
            f"  sweep={sweep.name} full_records={sweep.count} "
            f"shard_start={start} shard_records={count} bytes={size}"
        )
        print(f"  output={output_dir / safe_name(test.name) / filename}")
    print(f"tests={len(tests)} projected_bytes={total_bytes} projected_gib={total_bytes / 1024**3:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, compile, and run all README-listed GB10 PTX accuracy tests."
    )
    parser.add_argument(
        "--tests",
        action="append",
        default=[],
        metavar="GLOB",
        help="select test name/PTX glob; repeatable (default: every GB10 row/variant)",
    )
    parser.add_argument("--list", action="store_true", help="list expanded tests and exit")
    parser.add_argument("--plan", action="store_true", help="print ranges, shard sizes, and outputs")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="build and execute one temporary record per concrete PTX, then exit",
    )
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--arch", default="sm_121a")
    parser.add_argument("--chunk-records", type=int, default=1_048_576)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--limit-records",
        type=int,
        help="cap each test/sweep shard; useful for bring-up without changing enumeration order",
    )
    parser.add_argument(
        "--yes-large",
        action="store_true",
        help="confirm an output projection of at least 16 GiB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("require 0 <= shard-index < shard-count")
    if args.chunk_records <= 0:
        raise RuntimeError("chunk-records must be positive")
    if args.limit_records is not None and args.limit_records < 0:
        raise RuntimeError("limit-records cannot be negative")
    if args.reference_dir:
        output_path = args.output_dir.resolve()
        reference_path = args.reference_dir.resolve()
        if (
            output_path == reference_path
            or output_path in reference_path.parents
            or reference_path in output_path.parents
        ):
            raise RuntimeError("output-dir and reference-dir must be separate, non-nested trees")

    tests = select_tests(args.tests)
    if args.list:
        for index, test in enumerate(tests):
            versions = f"CUDA {test.min_cuda[0]}.{test.min_cuda[1]}+"
            print(f"{index:03d} {test.name}\n    {test.ptx};  mask=0x{test.mask:08x}  {versions}")
        print(f"expanded tests: {len(tests)}")
        return
    if args.plan:
        print_plan(
            tests,
            args.output_dir.resolve(),
            args.profile,
            args.shard_index,
            args.shard_count,
            args.limit_records,
        )
        return

    source = generate_cuda(tests, args.generated_dir.resolve())
    log(f"generated {source} ({len(tests)} concrete instructions)")
    if args.generate_only:
        return

    binary = args.build_dir.resolve() / "gb10_ptx_accuracy"
    compile_cuda(source, binary, tests, args.nvcc, args.arch)
    log(f"built {binary}")
    if args.build_only:
        return
    preflight_tests(binary, tests)
    if args.preflight_only:
        log(f"all {len(tests)} concrete PTX preflights passed")
        return

    estimate = projected_bytes(
        tests,
        args.profile,
        args.shard_index,
        args.shard_count,
        args.limit_records,
    )
    log(f"projected binary output: {estimate / 1024**3:.3f} GiB")
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_dir.resolve()).free
    if estimate > free_bytes:
        raise RuntimeError(
            f"projected output requires {estimate} bytes, but output filesystem has "
            f"only {free_bytes} bytes free"
        )
    if estimate >= LARGE_OUTPUT_BYTES and not args.yes_large:
        raise RuntimeError(
            "projected output is at least 16 GiB; inspect the ranges/sharding and rerun with --yes-large"
        )

    summaries = execute_tests(
        binary,
        tests,
        args.output_dir.resolve(),
        args.reference_dir.resolve() if args.reference_dir else None,
        args.profile,
        args.shard_index,
        args.shard_count,
        args.chunk_records,
        args.limit_records,
    )
    manifest = write_manifest(
        args.output_dir.resolve(),
        tests,
        args.profile,
        args.shard_index,
        args.shard_count,
        summaries,
    )
    if args.reference_dir:
        log(f"all {len(summaries)} binary comparisons passed")
    else:
        log(
            f"captured and structurally validated {len(summaries)} GB10 golden binaries; "
            "pass --reference-dir for independent bitwise comparison"
        )
    log(f"manifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
