#!/usr/bin/env python3
"""Formal B200 test for cvt.rs.satfinite.bf16x2.f32.

The generated CUDA runner deliberately uses this data path:

    global Input[] -> LDG -> inline PTX -> global Record[] -> binary dump

The full matrix follows the root README exactly.  A and B use the inclusive
0..0xffffffff lattice with stride 0x00ffffff.  A 16-bit random value uses the
inclusive 0..0xffff lattice with stride 0xff and is copied into both halves of
the .b32 rbits operand.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, Sequence


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated" / "b200_cvt_rs_bf16x2_generated.cu"
BUILD = ROOT / "build" / "b200_cvt_rs_bf16x2"
SASS = ROOT / "build" / "b200_cvt_rs_bf16x2.sass"
DEFAULT_OUTPUT = ROOT / "results" / "cvt-rs-satfinite-bf16x2-f32"
DEFAULT_NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "sm_100a"
TEST_NAME = "cvt.rs.satfinite.bf16x2.f32"
PTX = "cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits"
U32_MAX = 0xFFFFFFFF
U32_STRIDE = 0x00FFFFFF
U16_MAX = 0xFFFF
U16_STRIDE = 0xFF
U32_COUNT = (U32_MAX + U32_STRIDE - 1) // U32_STRIDE + 1
U16_COUNT = (U16_MAX + U16_STRIDE - 1) // U16_STRIDE + 1
TOTAL_RECORDS = U32_COUNT * U32_COUNT * U16_COUNT
DEFAULT_SHARDS = 16
HEADER_SIZE = 256
RECORD_SIZE = 16
HEADER = struct.Struct("<8sIIIIQQQ128s80s")
RECORD = struct.Struct("<IIII")
MAGIC = b"B200RS\0\0"
FORMAT_VERSION = 1
BF16_MAX = 0x7F7F
BF16_INFINITY = 0x7F80


CUDA_SOURCE = r'''#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                \
  cudaError_t status_ = (call);                                              \
  if (status_ != cudaSuccess) {                                              \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                 cudaGetErrorString(status_));                               \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

static constexpr std::uint32_t kU32Max = 0xffffffffu;
static constexpr std::uint64_t kU32Stride = 0x00ffffffull;
static constexpr std::uint64_t kU32Count = 258;
static constexpr std::uint32_t kU16Max = 0xffffu;
static constexpr std::uint64_t kU16Stride = 0xffull;
static constexpr std::uint64_t kU16Count = 258;
static constexpr std::uint64_t kTotalRecords =
    kU32Count * kU32Count * kU16Count;

#pragma pack(push, 1)
struct BinHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_size;
  std::uint32_t record_size;
  std::uint32_t result_mask;
  std::uint64_t total_records;
  std::uint64_t shard_start;
  std::uint64_t shard_records;
  char test_name[128];
  char reserved[80];
};
#pragma pack(pop)

struct Input {
  std::uint32_t a_bits;
  std::uint32_t b_bits;
  std::uint32_t rbits;
};

struct Record {
  std::uint32_t a_bits;
  std::uint32_t b_bits;
  std::uint32_t rbits;
  std::uint32_t result;
};

static_assert(sizeof(BinHeader) == 256, "bad header size");
static_assert(sizeof(Input) == 12, "bad input size");
static_assert(sizeof(Record) == 16, "bad record size");

__host__ __device__ std::uint32_t lattice_value(
    std::uint64_t index, std::uint64_t stride, std::uint32_t maximum) {
  std::uint64_t value = index * stride;
  return static_cast<std::uint32_t>(value > maximum ? maximum : value);
}

__global__ void initialize_inputs(
    std::uint64_t global_start, std::uint64_t count, Input* inputs) {
  std::uint64_t local = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= count) return;
  std::uint64_t linear = global_start + local;
  std::uint64_t r_index = linear % kU16Count;
  linear /= kU16Count;
  std::uint64_t b_index = linear % kU32Count;
  linear /= kU32Count;
  std::uint64_t a_index = linear;
  std::uint32_t r16 = lattice_value(r_index, kU16Stride, kU16Max);
  inputs[local] = Input{
      lattice_value(a_index, kU32Stride, kU32Max),
      lattice_value(b_index, kU32Stride, kU32Max),
      r16 | (r16 << 16)};
}

__global__ void convert_inputs(
    std::uint64_t count, Input const* inputs, Record* records) {
  std::uint64_t local = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= count) return;
  // Keep these as three explicit read-only global loads.  The formal script
  // checks the final SASS for LDG, PACK_AB.RS, and STG instructions.
  std::uint32_t a_bits = __ldg(&inputs[local].a_bits);
  std::uint32_t b_bits = __ldg(&inputs[local].b_bits);
  std::uint32_t rbits = __ldg(&inputs[local].rbits);
  float a = __uint_as_float(a_bits);
  float b = __uint_as_float(b_bits);
  std::uint32_t result;
  asm volatile(
      "cvt.rs.satfinite.bf16x2.f32 %0, %1, %2, %3;"
      : "=&r"(result)
      : "f"(a), "f"(b), "r"(rbits));
  records[local] = Record{a_bits, b_bits, rbits, result};
}

std::uint64_t parse_u64(char const* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') {
    std::fprintf(stderr, "invalid integer: %s\n", text);
    std::exit(2);
  }
  return static_cast<std::uint64_t>(value);
}

void write_header(std::ofstream& stream, std::uint64_t total,
                  std::uint64_t start, std::uint64_t count) {
  BinHeader header{};
  std::memcpy(header.magic, "B200RS", 6);
  header.version = 1;
  header.header_size = sizeof(BinHeader);
  header.record_size = sizeof(Record);
  header.result_mask = 0xffffffffu;
  header.total_records = total;
  header.shard_start = start;
  header.shard_records = count;
  std::snprintf(header.test_name, sizeof(header.test_name),
                "cvt.rs.satfinite.bf16x2.f32");
  stream.write(reinterpret_cast<char const*>(&header), sizeof(header));
}

void check_device() {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 10 || properties.minor != 0 ||
      std::strstr(properties.name, "B200") == nullptr) {
    std::fprintf(stderr, "B200 sm_100 required; detected %s sm_%d%d\n",
                 properties.name, properties.major, properties.minor);
    std::exit(4);
  }
}

void launch_conversion(Input const* input, Record* record, std::uint64_t count) {
  int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  convert_inputs<<<blocks, threads>>>(count, input, record);
  CUDA_CHECK(cudaGetLastError());
}

int run_range(std::uint64_t start, std::uint64_t count,
              std::uint64_t chunk_records, char const* output_path) {
  if (start > kTotalRecords || count > kTotalRecords - start ||
      chunk_records == 0) return 2;
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) return 5;
  write_header(output, kTotalRecords, start, count);
  std::uint64_t allocation = std::min(chunk_records, std::max<std::uint64_t>(1, count));
  Input* inputs = nullptr;
  Record* records = nullptr;
  CUDA_CHECK(cudaMalloc(&inputs, allocation * sizeof(Input)));
  CUDA_CHECK(cudaMalloc(&records, allocation * sizeof(Record)));
  std::vector<Record> host(allocation);
  for (std::uint64_t done = 0; done < count;) {
    std::uint64_t current = std::min(allocation, count - done);
    int threads = 256;
    int blocks = static_cast<int>((current + threads - 1) / threads);
    initialize_inputs<<<blocks, threads>>>(start + done, current, inputs);
    CUDA_CHECK(cudaGetLastError());
    launch_conversion(inputs, records, current);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), records, current * sizeof(Record),
                          cudaMemcpyDeviceToHost));
    output.write(reinterpret_cast<char const*>(host.data()),
                 current * sizeof(Record));
    if (!output) return 5;
    done += current;
  }
  CUDA_CHECK(cudaFree(records));
  CUDA_CHECK(cudaFree(inputs));
  output.close();
  return output ? 0 : 5;
}

int run_single(std::uint32_t a, std::uint32_t b, std::uint16_t r16,
               char const* output_path) {
  Input host_input{a, b, std::uint32_t(r16) | (std::uint32_t(r16) << 16)};
  Input* input = nullptr;
  Record* record = nullptr;
  CUDA_CHECK(cudaMalloc(&input, sizeof(Input)));
  CUDA_CHECK(cudaMalloc(&record, sizeof(Record)));
  CUDA_CHECK(cudaMemcpy(input, &host_input, sizeof(Input), cudaMemcpyHostToDevice));
  launch_conversion(input, record, 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  Record host_record{};
  CUDA_CHECK(cudaMemcpy(&host_record, record, sizeof(Record), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(record));
  CUDA_CHECK(cudaFree(input));
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) return 5;
  write_header(output, 1, 0, 1);
  output.write(reinterpret_cast<char const*>(&host_record), sizeof(host_record));
  return output ? 0 : 5;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr,
        "usage: %s range START COUNT CHUNK OUTPUT | single A B R16 OUTPUT\n",
        argv[0]);
    return 2;
  }
  check_device();
  std::string mode(argv[1]);
  if (mode == "range") {
    return run_range(parse_u64(argv[2]), parse_u64(argv[3]),
                     parse_u64(argv[4]), argv[5]);
  }
  if (mode == "single") {
    return run_single(static_cast<std::uint32_t>(parse_u64(argv[2])),
                      static_cast<std::uint32_t>(parse_u64(argv[3])),
                      static_cast<std::uint16_t>(parse_u64(argv[4])), argv[5]);
  }
  return 2;
}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"B200 formal test for {PTX}")
    parser.add_argument("command", choices=("selftest", "plan", "run", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nvcc", default=DEFAULT_NVCC)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--chunk-records", type=int, default=1_048_576)
    return parser.parse_args()


def run(
    command: Sequence[object], *, timeout: int | None = None, echo_output: bool = True
) -> str:
    args = [str(item) for item in command]
    print("+ " + " ".join(args), flush=True)
    result = subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=timeout,
    )
    if result.stdout and echo_output:
        print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}")
    return result.stdout


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lattice_value(index: int, maximum: int, stride: int) -> int:
    return min(maximum, index * stride)


def source_tuple(linear: int) -> tuple[int, int, int, int]:
    r_index = linear % U16_COUNT
    linear //= U16_COUNT
    b_index = linear % U32_COUNT
    linear //= U32_COUNT
    a_index = linear
    r16 = lattice_value(r_index, U16_MAX, U16_STRIDE)
    return (
        lattice_value(a_index, U32_MAX, U32_STRIDE),
        lattice_value(b_index, U32_MAX, U32_STRIDE),
        r16 | (r16 << 16),
        r16,
    )


def is_bf16_nan(bits: int) -> bool:
    return (bits & BF16_INFINITY) == BF16_INFINITY and (bits & 0x7F) != 0


def expected_bf16(source: int, random_bits: int) -> tuple[str, int | None]:
    magnitude = source & 0x7FFFFFFF
    sign = (source >> 16) & 0x8000
    if magnitude > 0x7F800000:
        return "nan", None
    if magnitude == 0x7F800000:
        return "exact", sign | BF16_MAX
    high = source >> 16
    carry = ((source & 0xFFFF) + random_bits) >> 16
    result = (high + carry) & 0xFFFF
    if (result & 0x7FFF) >= BF16_INFINITY:
        result = sign | BF16_MAX
    return "exact", result


def expected_result(a: int, b: int, r16: int) -> tuple[tuple[str, int | None], tuple[str, int | None]]:
    return expected_bf16(a, r16), expected_bf16(b, r16)


def selftest() -> None:
    if (U32_COUNT, U16_COUNT, TOTAL_RECORDS) != (258, 258, 17_173_512):
        raise AssertionError("range cardinality changed")
    if source_tuple(0) != (0, 0, 0, 0):
        raise AssertionError("first tuple mismatch")
    if source_tuple(TOTAL_RECORDS - 1) != (U32_MAX, U32_MAX, U32_MAX, U16_MAX):
        raise AssertionError("last tuple mismatch")
    cases = (
        (0x00000000, 0x0000, ("exact", 0x0000)),
        (0x3F800000, 0x1FFF, ("exact", 0x3F80)),
        (0x3F808001, 0x0000, ("exact", 0x3F80)),
        (0x3F808001, 0xFFFF, ("exact", 0x3F81)),
        (0x7F800000, 0x0000, ("exact", 0x7F7F)),
        (0xFF800000, 0x0000, ("exact", 0xFF7F)),
    )
    for source, random_bits, expected in cases:
        actual = expected_bf16(source, random_bits)
        if actual != expected:
            raise AssertionError((hex(source), hex(random_bits), actual, expected))
    if expected_bf16(0x7FC00001, 0)[0] != "nan":
        raise AssertionError("NaN classification failed")


def nvcc_version(nvcc: str) -> tuple[int, int]:
    output = run([nvcc, "--version"])
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        raise RuntimeError("cannot parse nvcc version")
    version = int(match.group(1)), int(match.group(2))
    if version < (12, 8):
        raise RuntimeError(f"{TEST_NAME} requires CUDA 12.8+, found {version}")
    return version


def build(nvcc: str) -> tuple[Path, dict[str, object]]:
    version = nvcc_version(nvcc)
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(CUDA_SOURCE)
    run([nvcc, "-O3", "-std=c++17", f"-arch={ARCH}", "-lineinfo", GENERATED, "-o", BUILD])
    sass = run(
        [str(Path(nvcc).with_name("cuobjdump")), "--dump-sass", BUILD],
        echo_output=False,
    )
    SASS.write_text(sass)
    mnemonic = "F2FP.SATFINITE.BF16.F32.PACK_AB.RS"
    function = re.search(
        r"Function\s*:\s*_Z14convert_inputs[^\n]*\n(.*?)(?=\n\s*Function\s*:|\Z)",
        sass,
        re.DOTALL,
    )
    if function is None:
        raise RuntimeError("cannot isolate convert_inputs in SASS")
    conversion_sass = function.group(1)
    if mnemonic not in conversion_sass:
        raise RuntimeError(f"SASS does not contain {mnemonic}")
    if conversion_sass.count("LDG") < 3 or "STG" not in conversion_sass:
        raise RuntimeError(
            "convert_inputs SASS does not prove three global inputs -> LDG -> global output"
        )
    return BUILD, {
        "nvcc": f"{version[0]}.{version[1]}",
        "arch": ARCH,
        "sass_mnemonic": mnemonic,
        "sass_sha256": file_sha256(SASS),
        "generated_cuda_sha256": file_sha256(GENERATED),
        "binary_sha256": file_sha256(BUILD),
    }


def parse_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw = stream.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise RuntimeError(f"truncated header: {path}")
    magic, version, header_size, record_size, mask, total, start, count, name, _ = HEADER.unpack(raw)
    if magic != MAGIC or version != FORMAT_VERSION:
        raise RuntimeError(f"bad magic/version: {path}")
    if header_size != HEADER_SIZE or record_size != RECORD_SIZE or mask != U32_MAX:
        raise RuntimeError(f"bad binary format: {path}")
    if path.stat().st_size != HEADER_SIZE + count * RECORD_SIZE:
        raise RuntimeError(f"bad binary length: {path}")
    return {
        "test_name": name.split(b"\0", 1)[0].decode(),
        "total_records": total,
        "shard_start": start,
        "shard_records": count,
    }


def validate_lanes(a: int, b: int, packed_rbits: int, result: int, context: str) -> tuple[int, int]:
    if (packed_rbits & 0xFFFF) != (packed_rbits >> 16):
        raise RuntimeError(f"Rbits halves differ: {context}")
    r16 = packed_rbits & 0xFFFF
    expectations = expected_result(a, b, r16)
    actuals = (result >> 16, result & 0xFFFF)
    exact = nan = 0
    for lane, (expected, actual) in enumerate(zip(expectations, actuals)):
        kind, value = expected
        if kind == "nan":
            if not is_bf16_nan(actual):
                raise RuntimeError(f"NaN mismatch {context} lane={lane} actual=0x{actual:04x}")
            nan += 1
        else:
            if actual != value:
                raise RuntimeError(
                    f"reference mismatch {context} lane={lane} actual=0x{actual:04x} "
                    f"expected=0x{value:04x}"
                )
            exact += 1
    return exact, nan


def validate_file(path: Path, start: int, count: int, total: int = TOTAL_RECORDS) -> dict[str, int | str]:
    header = parse_header(path)
    expected_header = {
        "test_name": TEST_NAME,
        "total_records": total,
        "shard_start": start,
        "shard_records": count,
    }
    if header != expected_header:
        raise RuntimeError(f"header mismatch: {path}: {header!r}")
    exact = nan = 0
    with path.open("rb") as stream:
        stream.seek(HEADER_SIZE)
        for local in range(count):
            raw = stream.read(RECORD_SIZE)
            if len(raw) != RECORD_SIZE:
                raise RuntimeError(f"truncated record {local}: {path}")
            a, b, rbits, result = RECORD.unpack(raw)
            if total == TOTAL_RECORDS:
                expected_a, expected_b, expected_r, _ = source_tuple(start + local)
                if (a, b, rbits) != (expected_a, expected_b, expected_r):
                    raise RuntimeError(f"input mismatch: {path} record={local}")
            lane_exact, lane_nan = validate_lanes(a, b, rbits, result, f"{path} record={local}")
            exact += lane_exact
            nan += lane_nan
        if stream.read(1):
            raise RuntimeError(f"trailing bytes: {path}")
    return {
        "records": count,
        "lanes": count * 2,
        "exact_bit_matches": exact,
        "nan_class_matches": nan,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def precheck(binary: Path, output_dir: Path, provenance: dict[str, object]) -> Path:
    cases = (
        ("raw-zero-one", 0x00000000, 0x00000001, 0x1FFF),
        ("exact-zero-one", 0x00000000, 0x3F800000, 0x1FFF),
        ("round-down", 0x3F808001, 0xBF808001, 0x0000),
        ("round-away", 0x3F808001, 0xBF808001, 0xFFFF),
        ("infinities", 0x7F800000, 0xFF800000, 0x55AA),
        ("nan", 0x7FC00001, 0xFFC00001, 0xAA55),
        ("signed-zero", 0x80000000, 0x00000000, 0xFFFF),
    )
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="b200-rs-precheck-") as temporary:
        root = Path(temporary)
        for name, a, b, r16 in cases:
            paths = [root / f"{name}-{iteration}.bin" for iteration in range(2)]
            for path in paths:
                run([binary, "single", hex(a), hex(b), hex(r16), path])
                validate_file(path, 0, 1, 1)
            if paths[0].read_bytes() != paths[1].read_bytes():
                raise RuntimeError(f"non-deterministic single-case output: {name}")
            _a, _b, packed_rbits, result = RECORD.unpack(
                paths[0].read_bytes()[HEADER_SIZE:HEADER_SIZE + RECORD_SIZE]
            )
            results.append({
                "name": name,
                "a_bits": f"0x{a:08x}",
                "b_bits": f"0x{b:08x}",
                "rbits": f"0x{packed_rbits:08x}",
                "result": f"0x{result:08x}",
            })
    report = {
        "status": "PASS",
        "test": TEST_NAME,
        "ptx": PTX,
        "platform": "NVIDIA B200",
        "data_path": "global input -> LDG -> inline PTX -> global output",
        "provenance": provenance,
        "cases": results,
    }
    path = output_dir / "precheck-report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def shard_slice(index: int, count: int) -> tuple[int, int]:
    start = TOTAL_RECORDS * index // count
    end = TOTAL_RECORDS * (index + 1) // count
    return start, end - start


def shard_path(output_dir: Path, index: int, count: int) -> Path:
    return output_dir / "full" / f"shard-{index:05d}-of-{count:05d}.bin"


def manifest_path(output_dir: Path, index: int, count: int) -> Path:
    return output_dir / f"manifest-shard-{index:05d}-of-{count:05d}.json"


def specification(shard_count: int) -> dict[str, object]:
    return {
        "test": TEST_NAME,
        "ptx": PTX,
        "platform": "NVIDIA B200",
        "arch": ARCH,
        "source_a": {"start": 0, "maximum": U32_MAX, "stride": U32_STRIDE, "count": U32_COUNT},
        "source_b": {"start": 0, "maximum": U32_MAX, "stride": U32_STRIDE, "count": U32_COUNT},
        "rbits_16": {"start": 0, "maximum": U16_MAX, "stride": U16_STRIDE, "count": U16_COUNT},
        "rbits_packing": "upper16 == lower16",
        "records": TOTAL_RECORDS,
        "lanes": TOTAL_RECORDS * 2,
        "shard_count": shard_count,
        "data_path": "global input -> LDG -> inline PTX -> global output",
    }


@contextlib.contextmanager
def output_lock(output_dir: Path) -> Iterator[None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.parent / f".{output_dir.name}.lock"
    with lock.open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another run holds {lock}") from error
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        yield


def write_manifest(output_dir: Path, shard_index: int, shard_count: int, summary: dict[str, object]) -> Path:
    start, count = shard_slice(shard_index, shard_count)
    payload = {
        "manifest_version": 1,
        "specification": specification(shard_count),
        "specification_sha256": canonical_sha256(specification(shard_count)),
        "shard_index": shard_index,
        "shard_start": start,
        "shard_records": count,
        "result": summary,
    }
    path = manifest_path(output_dir, shard_index, shard_count)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def validate_manifest(output_dir: Path, shard_index: int, shard_count: int) -> dict[str, object]:
    path = manifest_path(output_dir, shard_index, shard_count)
    if not path.is_file():
        raise RuntimeError(f"missing manifest: {path}")
    payload = json.loads(path.read_text())
    start, count = shard_slice(shard_index, shard_count)
    if payload.get("specification_sha256") != canonical_sha256(specification(shard_count)):
        raise RuntimeError(f"specification mismatch: {path}")
    if payload.get("shard_index") != shard_index or payload.get("shard_start") != start or payload.get("shard_records") != count:
        raise RuntimeError(f"manifest range mismatch: {path}")
    summary = validate_file(shard_path(output_dir, shard_index, shard_count), start, count)
    if payload.get("result") != summary:
        raise RuntimeError(f"manifest result mismatch: {path}")
    return summary


def print_plan(args: argparse.Namespace) -> list[int]:
    end = args.shard_count - 1 if args.end_shard is None else args.end_shard
    if args.shard_count <= 0 or not 0 <= args.start_shard <= end < args.shard_count:
        raise RuntimeError("invalid shard range")
    output_dir = args.output_dir.resolve()
    pending: list[int] = []
    for index in range(args.start_shard, end + 1):
        try:
            validate_manifest(output_dir, index, args.shard_count)
        except Exception:
            pending.append(index)
    total_bytes = TOTAL_RECORDS * RECORD_SIZE + args.shard_count * HEADER_SIZE
    remaining = sum(shard_slice(index, args.shard_count)[1] * RECORD_SIZE + HEADER_SIZE for index in pending)
    print(f"instruction: {PTX}")
    print(f"A values: {U32_COUNT}; B values: {U32_COUNT}; Rbits values: {U16_COUNT}")
    print(f"records: {TOTAL_RECORDS}; lanes: {TOTAL_RECORDS * 2}")
    print(f"shards: {args.shard_count}; pending: {pending or 'none'}")
    print(f"full binary bytes: {total_bytes} ({total_bytes / 1024**2:.3f} MiB)")
    print(f"remaining bytes: {remaining} ({remaining / 1024**2:.3f} MiB)")
    print("data path: global input -> LDG -> inline PTX -> global output")
    return pending


def write_full_report(output_dir: Path, shard_count: int, precheck_report: Path) -> Path:
    if not precheck_report.is_file() or json.loads(precheck_report.read_text()).get("status") != "PASS":
        raise RuntimeError("valid precheck report is required")
    summaries = [validate_manifest(output_dir, index, shard_count) for index in range(shard_count)]
    report = {
        "status": "PASS",
        "capture_status": "COMMENTS_STRIDED_CAPTURE_COMPLETE",
        "accuracy_status": "INDEPENDENT_REFERENCE_PASS",
        "test": TEST_NAME,
        "ptx": PTX,
        "platform": "NVIDIA B200",
        "specification": specification(shard_count),
        "specification_sha256": canonical_sha256(specification(shard_count)),
        "binary_count": shard_count,
        "binary_bytes": sum(int(item["bytes"]) for item in summaries),
        "exact_bit_matches": sum(int(item["exact_bit_matches"]) for item in summaries),
        "nan_class_matches": sum(int(item["nan_class_matches"]) for item in summaries),
        "binary_sha256_root": canonical_sha256([item["sha256"] for item in summaries]),
        "precheck_report": str(precheck_report),
        "precheck_report_sha256": file_sha256(precheck_report),
    }
    path = output_dir / "full-run-report.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def run_full(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    pending = print_plan(args)
    with output_lock(output_dir):
        binary, provenance = build(args.nvcc)
        precheck_report = precheck(binary, output_dir, provenance)
        end = args.shard_count - 1 if args.end_shard is None else args.end_shard
        pending = []
        for index in range(args.start_shard, end + 1):
            try:
                validate_manifest(output_dir, index, args.shard_count)
            except Exception:
                pending.append(index)
        required = sum(shard_slice(index, args.shard_count)[1] * RECORD_SIZE + HEADER_SIZE for index in pending)
        output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output_dir).free < required + 1024**3:
            raise RuntimeError("insufficient disk space (requires output plus 1 GiB margin)")
        (output_dir / "full").mkdir(parents=True, exist_ok=True)
        for index in pending:
            start, count = shard_slice(index, args.shard_count)
            path = shard_path(output_dir, index, args.shard_count)
            partial = path.with_suffix(".bin.partial")
            if partial.exists():
                partial.unlink()
            run([binary, "range", start, count, args.chunk_records, partial])
            summary = validate_file(partial, start, count)
            partial.replace(path)
            write_manifest(output_dir, index, args.shard_count, summary)
            print(f"PASS shard {index}/{args.shard_count - 1}: {count} records")
        complete = all(manifest_path(output_dir, index, args.shard_count).is_file() for index in range(args.shard_count))
        if complete:
            report = write_full_report(output_dir, args.shard_count, precheck_report)
            print(f"FULL PASS: {TOTAL_RECORDS} records, {TOTAL_RECORDS * 2} lanes")
            print(f"report: {report}")
        else:
            print("requested shards complete; run remaining shards before final report")


def main() -> None:
    args = parse_args()
    if args.chunk_records <= 0:
        raise RuntimeError("chunk-records must be positive")
    selftest()
    if args.command == "selftest":
        print("SELFTEST PASS: ranges, packing, and independent BF16 reference")
    elif args.command == "plan":
        print_plan(args)
    elif args.command == "run":
        run_full(args)
    else:
        path = write_full_report(
            args.output_dir.resolve(), args.shard_count,
            args.output_dir.resolve() / "precheck-report.json",
        )
        print(f"REPORT PASS: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
