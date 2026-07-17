#!/usr/bin/env python3
"""B200 stride-1 test for FP16x2/BF16x2 stochastic conversion.

The generated CUDA runner uses global Input[] -> 3x LDG -> inline PTX ->
global d[] -> binary dump. Each binary payload contains only uint32 d values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated" / "b200_cvt_rs_f16_bf16_generated.cu"
BUILD = ROOT / "build" / "b200_cvt_rs_f16_bf16"
SASS = ROOT / "build" / "b200_cvt_rs_f16_bf16.sass"
DEFAULT_OUTPUT = ROOT / "results" / "cvt-rs-satfinite-bf16x2-f32"
DEFAULT_NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "sm_100a"
A_BEGIN = 0x33000000
A_END = 0x34800000
FIXED_B = 0xDEADBEEF
FIXED_RBITS = 0x1FFF1FFF
TOTAL_RECORDS = A_END - A_BEGIN + 1
HEADER_SIZE = 256
RECORD_SIZE = 4
MAGIC = b"B2RS2\0\0\0"
HEADER = struct.Struct("<8sIIIIQQIIII128s72s")
RESULT = struct.Struct("<I")

TESTS = (
    {
        "id": 0,
        "slug": "f16x2",
        "name": "cvt.rs.satfinite.f16x2.f32",
        "kernel": "convert_f16",
        "sass": "F2FP.SATFINITE.F16.F32.PACK_AB.RS",
    },
    {
        "id": 1,
        "slug": "bf16x2",
        "name": "cvt.rs.satfinite.bf16x2.f32",
        "kernel": "convert_bf16",
        "sass": "F2FP.SATFINITE.BF16.F32.PACK_AB.RS",
    },
)


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
  cudaError_t s_ = (call);                                                   \
  if (s_ != cudaSuccess) {                                                   \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                 cudaGetErrorString(s_));                                    \
    std::exit(1);                                                            \
  }                                                                          \
} while (0)

static constexpr std::uint32_t kABegin = 0x33000000u;
static constexpr std::uint32_t kAEnd = 0x34800000u;
static constexpr std::uint32_t kFixedB = 0xdeadbeefu;
static constexpr std::uint32_t kFixedRbits = 0x1fff1fffu;
static constexpr std::uint64_t kTotal =
    std::uint64_t(kAEnd) - std::uint64_t(kABegin) + 1;

#pragma pack(push, 1)
struct Header {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_size;
  std::uint32_t record_size;
  std::uint32_t result_mask;
  std::uint64_t total_records;
  std::uint64_t records;
  std::uint32_t a_begin;
  std::uint32_t a_end;
  std::uint32_t fixed_b;
  std::uint32_t fixed_rbits;
  char test_name[128];
  char reserved[72];
};
#pragma pack(pop)

struct Input { std::uint32_t a, b, rbits; };
static_assert(sizeof(Header) == 256, "header size");
static_assert(sizeof(Input) == 12, "input size");

__global__ void initialize_inputs(std::uint64_t start, std::uint64_t count,
                                  Input* inputs) {
  std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  inputs[i] = Input{static_cast<std::uint32_t>(std::uint64_t(kABegin) + start + i),
                    kFixedB, kFixedRbits};
}

__global__ void convert_f16(std::uint64_t count, Input const* inputs,
                            std::uint32_t* output) {
  std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  std::uint32_t ai = __ldg(&inputs[i].a);
  std::uint32_t bi = __ldg(&inputs[i].b);
  std::uint32_t r = __ldg(&inputs[i].rbits);
  float a = __uint_as_float(ai), b = __uint_as_float(bi);
  std::uint32_t d;
  asm volatile("cvt.rs.satfinite.f16x2.f32 %0, %1, %2, %3;"
               : "=&r"(d) : "f"(a), "f"(b), "r"(r));
  output[i] = d;
}

__global__ void convert_bf16(std::uint64_t count, Input const* inputs,
                             std::uint32_t* output) {
  std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  std::uint32_t ai = __ldg(&inputs[i].a);
  std::uint32_t bi = __ldg(&inputs[i].b);
  std::uint32_t r = __ldg(&inputs[i].rbits);
  float a = __uint_as_float(ai), b = __uint_as_float(bi);
  std::uint32_t d;
  asm volatile("cvt.rs.satfinite.bf16x2.f32 %0, %1, %2, %3;"
               : "=&r"(d) : "f"(a), "f"(b), "r"(r));
  output[i] = d;
}

std::uint64_t parse(char const* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') std::exit(2);
  return static_cast<std::uint64_t>(value);
}

char const* name(int id) {
  return id == 0 ? "cvt.rs.satfinite.f16x2.f32"
                 : "cvt.rs.satfinite.bf16x2.f32";
}

void write_header(std::ofstream& stream, int id, std::uint64_t records) {
  Header h{};
  std::memcpy(h.magic, "B2RS2", 5);
  h.version = 2;
  h.header_size = sizeof(Header);
  h.record_size = sizeof(std::uint32_t);
  h.result_mask = 0xffffffffu;
  h.total_records = kTotal;
  h.records = records;
  h.a_begin = kABegin;
  h.a_end = kAEnd;
  h.fixed_b = kFixedB;
  h.fixed_rbits = kFixedRbits;
  std::snprintf(h.test_name, sizeof(h.test_name), "%s", name(id));
  stream.write(reinterpret_cast<char const*>(&h), sizeof(h));
}

void check_device() {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 10 || p.minor != 0 || std::strstr(p.name, "B200") == nullptr) {
    std::fprintf(stderr, "B200 required; detected %s sm_%d%d\n", p.name, p.major, p.minor);
    std::exit(4);
  }
}

void launch(int id, Input const* inputs, std::uint32_t* output,
            std::uint64_t count) {
  int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  if (id == 0) convert_f16<<<blocks, threads>>>(count, inputs, output);
  else convert_bf16<<<blocks, threads>>>(count, inputs, output);
  CUDA_CHECK(cudaGetLastError());
}

int run_range(int id, std::uint64_t start, std::uint64_t count,
              std::uint64_t chunk, char const* path) {
  if ((id != 0 && id != 1) || start > kTotal || count > kTotal - start || !chunk)
    return 2;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return 5;
  write_header(stream, id, count);
  std::uint64_t allocation = std::min(chunk, std::max<std::uint64_t>(1, count));
  Input* inputs = nullptr;
  std::uint32_t* output = nullptr;
  CUDA_CHECK(cudaMalloc(&inputs, allocation * sizeof(Input)));
  CUDA_CHECK(cudaMalloc(&output, allocation * sizeof(std::uint32_t)));
  std::vector<std::uint32_t> host(allocation);
  for (std::uint64_t done = 0; done < count;) {
    std::uint64_t current = std::min(allocation, count - done);
    int threads = 256;
    int blocks = static_cast<int>((current + threads - 1) / threads);
    initialize_inputs<<<blocks, threads>>>(start + done, current, inputs);
    CUDA_CHECK(cudaGetLastError());
    launch(id, inputs, output, current);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), output, current * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    stream.write(reinterpret_cast<char const*>(host.data()),
                 current * sizeof(std::uint32_t));
    if (!stream) return 5;
    done += current;
  }
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(inputs));
  return 0;
}

int run_single(int id, std::uint32_t a, char const* path) {
  Input h{a, kFixedB, kFixedRbits};
  Input* input = nullptr;
  std::uint32_t* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, sizeof(Input)));
  CUDA_CHECK(cudaMalloc(&output, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemcpy(input, &h, sizeof(h), cudaMemcpyHostToDevice));
  launch(id, input, output, 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::uint32_t d = 0;
  CUDA_CHECK(cudaMemcpy(&d, output, sizeof(d), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return 5;
  write_header(stream, id, 1);
  stream.write(reinterpret_cast<char const*>(&d), sizeof(d));
  return stream ? 0 : 5;
}

int main(int argc, char** argv) {
  check_device();
  if (argc == 7 && std::string(argv[1]) == "range")
    return run_range(int(parse(argv[2])), parse(argv[3]), parse(argv[4]),
                     parse(argv[5]), argv[6]);
  if (argc == 5 && std::string(argv[1]) == "single")
    return run_single(int(parse(argv[2])), std::uint32_t(parse(argv[3])), argv[4]);
  return 2;
}
'''


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B200 FP16x2/BF16x2 .rs stride-1 test")
    parser.add_argument("command", choices=("selftest", "plan", "precheck", "run", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nvcc", default=DEFAULT_NVCC)
    parser.add_argument("--chunk-records", type=int, default=1_048_576)
    return parser.parse_args()


def command(items: Sequence[object], *, echo: bool = True) -> str:
    argv = [str(item) for item in items]
    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.stdout and echo:
        print(proc.stdout, end="")
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}")
    return proc.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def nan16(bits: int, slug: str) -> bool:
    return ((bits & 0x7F80) == 0x7F80 and (bits & 0x7F) != 0) if slug == "bf16x2" \
        else ((bits & 0x7C00) == 0x7C00 and (bits & 0x3FF) != 0)


def ref_bf16(source: int, random16: int) -> tuple[str, int | None]:
    magnitude = source & 0x7FFFFFFF
    sign = (source >> 16) & 0x8000
    if magnitude > 0x7F800000:
        return "nan", None
    if magnitude == 0x7F800000:
        return "exact", sign | 0x7F7F
    result = (source >> 16) + (((source & 0xFFFF) + random16) >> 16)
    result &= 0xFFFF
    if (result & 0x7FFF) >= 0x7F80:
        result = sign | 0x7F7F
    return "exact", result


def ref_f16(source: int, random13: int) -> tuple[str, int | None]:
    magnitude = source & 0x7FFFFFFF
    sign = (source >> 16) & 0x8000
    exponent = (magnitude >> 23) & 0xFF
    fraction = magnitude & 0x7FFFFF
    if exponent == 0xFF:
        return ("nan", None) if fraction else ("exact", sign | 0x7BFF)
    if magnitude > 0x477FE000:
        return "exact", sign | 0x7BFF
    if exponent == 0:
        unbiased, significand = -126, fraction
    else:
        unbiased, significand = exponent - 127, 0x800000 | fraction
    if unbiased >= -14:
        base = ((unbiased + 15) << 10) | ((significand & 0x7FFFFF) >> 13)
        discarded = significand & 0x1FFF
    else:
        shift = -(unbiased + 1)
        base = significand >> shift if shift < 32 else 0
        remainder = significand - (base << shift) if shift < 32 else significand
        discarded = (remainder >> (shift - 13)) if shift >= 13 else (remainder << (13 - shift))
    result = base + ((discarded + (random13 & 0x1FFF)) >> 13)
    return "exact", sign | min(result, 0x7BFF)


def expected(test: dict[str, object], a: int) -> tuple[tuple[str, int | None], tuple[str, int | None]]:
    if test["slug"] == "f16x2":
        fn, random = ref_f16, FIXED_RBITS & 0x1FFF
    else:
        fn, random = ref_bf16, FIXED_RBITS & 0xFFFF
    return fn(a, random), fn(FIXED_B, random)


def check_d(test: dict[str, object], a: int, d: int, context: str) -> tuple[int, int]:
    exact = nan = 0
    for lane, ((kind, value), actual) in enumerate(zip(expected(test, a), (d >> 16, d & 0xFFFF))):
        if kind == "nan":
            if not nan16(actual, str(test["slug"])):
                raise RuntimeError(f"NaN mismatch {context} lane={lane}: 0x{actual:04x}")
            nan += 1
        elif actual != value:
            raise RuntimeError(f"reference mismatch {context} lane={lane}: actual=0x{actual:04x} expected=0x{int(value):04x}")
        else:
            exact += 1
    return exact, nan


def selftest() -> None:
    assert HEADER.size == HEADER_SIZE
    assert TOTAL_RECORDS == 25_165_825
    assert (FIXED_RBITS & 0xE000E000) == 0
    assert ref_f16(0x33800000, 0) == ("exact", 1)
    assert ref_f16(0x3F800000, 0x1FFF) == ("exact", 0x3C00)
    assert ref_bf16(0x3F800000, 0x1FFF) == ("exact", 0x3F80)
    assert ref_bf16(0x7F800000, 0) == ("exact", 0x7F7F)


def build(nvcc: str) -> tuple[Path, dict[str, object]]:
    version_text = command([nvcc, "--version"])
    match = re.search(r"release\s+(\d+)\.(\d+)", version_text)
    if not match or (int(match.group(1)), int(match.group(2))) < (12, 8):
        raise RuntimeError("CUDA 12.8+ required")
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(CUDA_SOURCE)
    command([nvcc, "-O3", "-std=c++17", f"-arch={ARCH}", "-lineinfo", GENERATED, "-o", BUILD])
    sass = command([str(Path(nvcc).with_name("cuobjdump")), "--dump-sass", BUILD], echo=False)
    SASS.write_text(sass)
    mapping: dict[str, str] = {}
    mnemonics = {str(test["sass"]) for test in TESTS}
    for test in TESTS:
        found = re.search(rf"Function\s*:\s*_Z\d+{test['kernel']}[^\n]*\n(.*?)(?=\n\s*Function\s*:|\Z)", sass, re.S)
        if not found:
            raise RuntimeError(f"cannot isolate {test['kernel']} in SASS")
        section = found.group(1)
        present = {mnemonic for mnemonic in mnemonics if mnemonic in section}
        if present != {test["sass"]}:
            raise RuntimeError(f"SASS mapping error for {test['name']}: {sorted(present)}")
        if section.count("LDG") < 3 or "STG" not in section:
            raise RuntimeError(f"LDG/STG proof missing for {test['name']}")
        mapping[str(test["name"])] = str(test["sass"])
    return BUILD, {"cuda": f"{match.group(1)}.{match.group(2)}", "arch": ARCH,
                   "sass_mapping": mapping, "sass_sha256": sha256(SASS),
                   "cuda_source_sha256": sha256(GENERATED), "executable_sha256": sha256(BUILD)}


def parse_header(path: Path, test: dict[str, object], records: int) -> None:
    raw = path.read_bytes()[:HEADER_SIZE]
    if len(raw) != HEADER_SIZE:
        raise RuntimeError(f"truncated header: {path}")
    values = HEADER.unpack(raw)
    magic, version, hsize, rsize, mask, total, count = values[:7]
    a_begin, a_end, fixed_b, fixed_rbits, name = values[7:12]
    expected_values = (MAGIC, 2, HEADER_SIZE, RECORD_SIZE, 0xFFFFFFFF,
                       TOTAL_RECORDS, records, A_BEGIN, A_END, FIXED_B, FIXED_RBITS,
                       test["name"])
    actual_values = (magic, version, hsize, rsize, mask, total, count, a_begin,
                     a_end, fixed_b, fixed_rbits, name.split(b"\0", 1)[0].decode())
    if actual_values != expected_values:
        raise RuntimeError(f"header mismatch: {path}: {actual_values!r}")
    if path.stat().st_size != HEADER_SIZE + records * RECORD_SIZE:
        raise RuntimeError(f"size mismatch: {path}")


def validate(path: Path, test: dict[str, object], *, first_a: int = A_BEGIN,
             records: int = TOTAL_RECORDS) -> dict[str, int | str]:
    parse_header(path, test, records)
    exact = nan = 0
    with path.open("rb") as stream:
        stream.seek(HEADER_SIZE)
        for index in range(records):
            raw = stream.read(RECORD_SIZE)
            if len(raw) != RECORD_SIZE:
                raise RuntimeError(f"truncated result {index}: {path}")
            d, = RESULT.unpack(raw)
            matched, nan_matched = check_d(test, first_a + index, d, f"{path} index={index}")
            exact += matched
            nan += nan_matched
    return {"records": records, "lanes": records * 2, "exact_bit_matches": exact,
            "nan_class_matches": nan, "bytes": path.stat().st_size, "sha256": sha256(path)}


def precheck(binary: Path, output: Path, provenance: dict[str, object]) -> Path:
    cases = (A_BEGIN, 0x337FFFFF, 0x33800000, 0x33FFFFFF, 0x34000000,
             A_END, 0x3F808001, 0x7FC00001)
    observations: dict[str, list[dict[str, str]]] = {}
    with tempfile.TemporaryDirectory(prefix="b200-rs-pair-") as directory:
        temp = Path(directory)
        for test in TESTS:
            rows = []
            for a in cases:
                files = [temp / f"{test['slug']}-{a:08x}-{repeat}.bin" for repeat in range(2)]
                for file in files:
                    command([binary, "single", test["id"], hex(a), file])
                    validate(file, test, first_a=a, records=1)
                if files[0].read_bytes() != files[1].read_bytes():
                    raise RuntimeError(f"non-deterministic {test['name']} A=0x{a:08x}")
                d, = RESULT.unpack(files[0].read_bytes()[HEADER_SIZE:HEADER_SIZE + 4])
                rows.append({"a": f"0x{a:08x}", "d": f"0x{d:08x}"})
            observations[str(test["name"])] = rows
    report = {"status": "PASS", "platform": "NVIDIA B200",
              "a": {"begin": f"0x{A_BEGIN:08x}", "end": f"0x{A_END:08x}", "stride": 1},
              "b": f"0x{FIXED_B:08x}", "rbits": f"0x{FIXED_RBITS:08x}",
              "payload": "d only", "provenance": provenance, "observations": observations}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "precheck-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def result_path(output: Path, test: dict[str, object]) -> Path:
    return output / "full" / str(test["slug"]) / "d.bin"


def report(output: Path, summaries: dict[str, dict[str, int | str]] | None = None) -> Path:
    precheck_path = output / "precheck-report.json"
    precheck_data = json.loads(precheck_path.read_text())
    if precheck_data.get("status") != "PASS":
        raise RuntimeError("valid precheck required")
    if summaries is None:
        summaries = {str(test["name"]): validate(result_path(output, test), test) for test in TESTS}
    payload = {"status": "PASS", "capture_status": "STRIDE1_CAPTURE_COMPLETE",
               "accuracy_status": "INDEPENDENT_REFERENCE_PASS",
               "sass_mapping_status": "PTX_TO_SASS_MAPPING_PASS",
               "platform": "NVIDIA B200", "records_per_test": TOTAL_RECORDS,
               "payload": "d only (uint32 little-endian)",
               "binary_count": 2, "binary_bytes": sum(int(item["bytes"]) for item in summaries.values()),
               "tests": summaries, "sass_mapping": precheck_data["provenance"]["sass_mapping"],
               "precheck_sha256": sha256(precheck_path)}
    path = output / "full-run-report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def plan(output: Path) -> None:
    per_test = HEADER_SIZE + TOTAL_RECORDS * RECORD_SIZE
    print(f"A: 0x{A_BEGIN:08x}..0x{A_END:08x}, stride 1, inclusive")
    print(f"A count: {TOTAL_RECORDS}")
    print(f"B: 0x{FIXED_B:08x}; Rbits: 0x{FIXED_RBITS:08x}")
    print(f"payload: d only, {RECORD_SIZE} bytes per A per instruction")
    print(f"per instruction: {per_test} bytes ({per_test / 1024**2:.3f} MiB)")
    print(f"two instructions: {per_test * 2} bytes ({per_test * 2 / 1024**2:.3f} MiB)")
    for test in TESTS:
        path = result_path(output, test)
        print(f"{test['slug']}: {'present' if path.is_file() else 'pending'} -> {path}")


def full_run(options: argparse.Namespace) -> None:
    output = options.output_dir.resolve()
    plan(output)
    required = (HEADER_SIZE + TOTAL_RECORDS * RECORD_SIZE) * 2
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < required + 1024**3:
        raise RuntimeError("output plus 1 GiB free space required")
    binary, provenance = build(options.nvcc)
    precheck(binary, output, provenance)
    summaries: dict[str, dict[str, int | str]] = {}
    for test in TESTS:
        path = result_path(output, test)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".bin.partial")
        partial.unlink(missing_ok=True)
        command([binary, "range", test["id"], 0, TOTAL_RECORDS,
                 options.chunk_records, partial])
        summaries[str(test["name"])] = validate(partial, test)
        partial.replace(path)
        print(f"PASS {test['name']}: {TOTAL_RECORDS} d values")
    path = report(output, summaries)
    print(f"FULL PASS: {TOTAL_RECORDS} A values x {len(TESTS)} instructions")
    print(f"report: {path}")


def precheck_only(options: argparse.Namespace) -> None:
    output = options.output_dir.resolve()
    binary, provenance = build(options.nvcc)
    path = precheck(binary, output, provenance)
    print(f"PRECHECK PASS: {path}")


def main() -> None:
    options = args()
    selftest()
    if options.chunk_records <= 0:
        raise RuntimeError("chunk-records must be positive")
    if options.command == "selftest":
        print("SELFTEST PASS: ranges, Rbits layouts, FP16/BF16 references")
    elif options.command == "plan":
        plan(options.output_dir.resolve())
    elif options.command == "precheck":
        precheck_only(options)
    elif options.command == "run":
        full_run(options)
    else:
        print(f"REPORT PASS: {report(options.output_dir.resolve())}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
