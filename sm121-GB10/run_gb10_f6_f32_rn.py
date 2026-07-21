#!/usr/bin/env python3
"""Exhaustive GB10 capture for the two public FP6 .RN conversions.

The single entry point generates CUDA, compiles PTX 9.2 for compute_120f,
checks JIT SASS and an independent FP6 reference, then captures all 2^32 A
bit patterns for E2M3x2 and E3M2x2.  Output files are headerless little-endian
uint32 d values and are split into restartable shards.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated" / "gb10_f6_f32_rn_generated.cu"
BUILD = ROOT / "build" / "gb10_f6_f32_rn"
SASS = ROOT / "build" / "gb10_f6_f32_rn.sass"
DEFAULT_OUTPUT = ROOT / "results" / "f6-f32-rn-full"
DEFAULT_PRECHECK = ROOT / "results" / "f6-f32-rn-precheck"
DEFAULT_NVCC = "/usr/local/cuda-13.2/bin/nvcc"
DEFAULT_COMPAT = Path("/usr/local/cuda-13.2/compat")
ARCH = "compute_120f"
TOTAL = 1 << 32
FIXED_B = 0xDEADBEEF
MERGE_SEED = 0xDEADBEEF
RESULT_SIZE = 4
DEFAULT_SHARDS = 64
RESULT = struct.Struct("<I")

TESTS = (
    {
        "id": 0,
        "slug": "e2m3x2-rn",
        "ptx": "cvt.rn.satfinite.e2m3x2.f32",
        "sass": "F2FP.SATFINITE.E2M3.F32.PACK_AB_MERGE_C",
    },
    {
        "id": 1,
        "slug": "e3m2x2-rn",
        "ptx": "cvt.rn.satfinite.e3m2x2.f32",
        "sass": "F2FP.SATFINITE.E3M2.F32.PACK_AB_MERGE_C",
    },
)


CUDA_SOURCE = r'''#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define CUDA_CHECK(call) do {                                                \
  cudaError_t e_ = (call);                                                   \
  if (e_ != cudaSuccess) {                                                   \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,               \
                 cudaGetErrorString(e_));                                    \
    std::exit(3);                                                            \
  }                                                                          \
} while (0)

static constexpr std::uint32_t kFixedB = 0xdeadbeefu;
static constexpr std::uint32_t kMergeSeed = 0xdeadbeefu;
struct Input { std::uint32_t a, b, c; };
static_assert(sizeof(Input) == 12, "input layout");

__global__ void initialize_inputs(std::uint64_t start, std::uint64_t stride,
                                  std::uint64_t count, Input* input) {
  std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  input[i] = Input{static_cast<std::uint32_t>(start + i * stride),
                   kFixedB, kMergeSeed};
}

__global__ void convert(int test_id, std::uint64_t count,
                        Input const* input, std::uint32_t* output) {
  std::uint64_t i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  std::uint32_t ai = __ldg(&input[i].a);
  std::uint32_t bi = __ldg(&input[i].b);
  std::uint32_t ci = __ldg(&input[i].c);
  float a = __uint_as_float(ai), b = __uint_as_float(bi);
  std::uint16_t low;
  if (test_id == 0) {
    asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %1, %2;"
                 : "=h"(low) : "f"(a), "f"(b));
  } else {
    asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %1, %2;"
                 : "=h"(low) : "f"(a), "f"(b));
  }
  // Public PTX exposes a b16 destination. Preserve C[31:16] explicitly so
  // the file contains the requested full d while accuracy compares d[15:0].
  output[i] = (ci & 0xffff0000u) | std::uint32_t(low);
}

std::uint64_t parse(char const* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 0);
  if (end == text || *end != '\0') std::exit(2);
  return static_cast<std::uint64_t>(value);
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s test start count stride chunk out partial-tag\n", argv[0]);
    return 2;
  }
  int test = int(parse(argv[1]));
  std::uint64_t start = parse(argv[2]), count = parse(argv[3]);
  std::uint64_t stride = parse(argv[4]), chunk = parse(argv[5]);
  char const* path = argv[6];
  if ((test != 0 && test != 1) || !count || !chunk) return 2;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  if (p.major != 12 || p.minor != 1) {
    std::fprintf(stderr, "GB10 SM121 required; detected %s sm_%d%d\n", p.name, p.major, p.minor);
    return 4;
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return 5;
  std::uint64_t allocation = std::min(chunk, count);
  Input* input = nullptr;
  std::uint32_t* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, allocation * sizeof(Input)));
  CUDA_CHECK(cudaMalloc(&output, allocation * sizeof(std::uint32_t)));
  std::vector<std::uint32_t> host(allocation);
  for (std::uint64_t done = 0; done < count;) {
    std::uint64_t current = std::min(allocation, count - done);
    int threads = 256;
    int blocks = int((current + threads - 1) / threads);
    initialize_inputs<<<blocks, threads>>>(start + done * stride, stride, current, input);
    CUDA_CHECK(cudaGetLastError());
    convert<<<blocks, threads>>>(test, current, input, output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), output, current * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    stream.write(reinterpret_cast<char const*>(host.data()),
                 current * sizeof(std::uint32_t));
    if (!stream) return 5;
    done += current;
  }
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  stream.flush();
  return stream ? 0 : 5;
}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GB10 exhaustive FP32-to-FP6x2 .RN capture")
    parser.add_argument("command", choices=("plan", "precheck", "run", "report", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--precheck-dir", type=Path, default=DEFAULT_PRECHECK)
    parser.add_argument("--nvcc", default=DEFAULT_NVCC)
    parser.add_argument("--compat-dir", type=Path, default=DEFAULT_COMPAT)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--chunk-records", type=int, default=4_194_304)
    parser.add_argument("--yes-large", action="store_true")
    return parser.parse_args()


def command(items: Sequence[object], *, env: dict[str, str] | None = None) -> str:
    argv = [str(item) for item in items]
    print("+ " + " ".join(argv), flush=True)
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(argv)}")
    return proc.stdout


def runtime_env(compat: Path, cache: Path) -> dict[str, str]:
    if not any(compat.glob("libnvidia-ptxjitcompiler.so*")):
        raise RuntimeError(f"PTX JIT compatibility library missing: {compat}")
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(compat) + (":" + old if old else "")
    env["CUDA_CACHE_PATH"] = str(cache)
    env["CUDA_CACHE_DISABLE"] = "0"
    return env


def build(nvcc: str) -> None:
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(CUDA_SOURCE)
    command((nvcc, "-O3", "-std=c++17", "-lineinfo",
             f"--gpu-architecture={ARCH}", f"--gpu-code={ARCH}", GENERATED, "-o", BUILD))


@functools.lru_cache(maxsize=None)
def fp6_values(dtype: str) -> tuple[float, ...]:
    ebits, mbits, bias = (2, 3, 1) if dtype == "e2m3x2" else (3, 2, 3)
    values = []
    for code in range(32):
        exponent, mantissa = code >> mbits, code & ((1 << mbits) - 1)
        if exponent == 0:
            value = math.ldexp(mantissa, 1 - bias - mbits)
        else:
            value = math.ldexp((1 << mbits) + mantissa, exponent - bias - mbits)
        values.append(value)
    return tuple(values)


def fp6(bits: int, dtype: str) -> int:
    sign = bool(bits & 0x80000000)
    exponent = (bits >> 23) & 0xff
    fraction = bits & 0x7fffff
    if exponent == 0xff and fraction:
        return 0x1f
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    magnitude = abs(value)
    values = fp6_values(dtype)
    if math.isinf(magnitude) or magnitude >= values[-1]:
        code = 0x1f
    else:
        code = min(range(32), key=lambda c: (abs(values[c] - magnitude), c & 1, c))
    return code | (0x20 if sign else 0)


def expected(a: int, dtype: str) -> int:
    # PTX packs b into d[5:0] and a into d[13:8]; padding bits remain zero.
    low16 = fp6(FIXED_B, dtype) | (fp6(a, dtype) << 8)
    return (MERGE_SEED & 0xffff0000) | low16


def run_binary(test: dict[str, object], start: int, count: int, stride: int,
               path: Path, chunk: int, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    command((BUILD, test["id"], start, count, stride, chunk, partial, "partial"), env=env)
    expected_size = count * RESULT_SIZE
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"wrong size for {partial}: {partial.stat().st_size} != {expected_size}")
    partial.replace(path)


def extract_sass(cache: Path, nvcc: str) -> str:
    nvdisasm = Path(nvcc).parent / "nvdisasm"
    if not nvdisasm.exists():
        for candidate in (Path("/usr/local/cuda-13.1/bin/nvdisasm"),
                          Path("/usr/local/cuda-13.0/bin/nvdisasm")):
            if candidate.exists():
                nvdisasm = candidate
                break
    if not nvdisasm.exists():
        raise RuntimeError("nvdisasm not found")
    texts = []
    with tempfile.TemporaryDirectory(prefix="gb10-cubin-") as tmp:
        for index, path in enumerate(cache.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            offset = data.find(b"\x7fELF")
            if offset < 0:
                continue
            cubin = Path(tmp) / f"{index}.cubin"
            cubin.write_bytes(data[offset:])
            proc = subprocess.run((str(nvdisasm), str(cubin)), text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if proc.returncode == 0:
                texts.append(proc.stdout)
    text = "\n".join(texts)
    for test in TESTS:
        if test["sass"] not in text:
            raise RuntimeError(f"expected SASS not found: {test['sass']}")
    SASS.write_text(text)
    return text


def precheck(args: argparse.Namespace) -> None:
    if args.precheck_dir.exists():
        shutil.rmtree(args.precheck_dir)
    cache = BUILD.parent / "f6-f32-rn-jit-cache"
    if cache.exists():
        shutil.rmtree(cache)
    env = runtime_env(args.compat_dir, cache)
    build(args.nvcc)
    sweeps = (
        ("spread-pos", 0x00000000, 65_536, 0x00010001),
        ("spread-neg", 0x80000000, 65_536, 0x00010001),
        ("pos-inf-nan", 0x7f800000, 65_536, 1),
        ("neg-inf-nan", 0xff800000, 65_536, 1),
    )
    checked = 0
    for test in TESTS:
        dtype = str(test["slug"]).split("-")[0]
        for name, start, count, stride in sweeps:
            path = args.precheck_dir / str(test["slug"]) / f"{name}.bin"
            run_binary(test, start, count, stride, path, args.chunk_records, env)
            data = path.read_bytes()
            for i, (actual,) in enumerate(struct.iter_unpack("<I", data)):
                a = (start + i * stride) & 0xffffffff
                wanted = expected(a, dtype)
                if actual != wanted:
                    raise RuntimeError(
                        f"reference mismatch {test['slug']} {name} i={i} "
                        f"a=0x{a:08x}: d=0x{actual:08x}, expected=0x{wanted:08x}"
                    )
                checked += 1
    sass = extract_sass(cache, args.nvcc)
    report = {
        "status": "PASS",
        "tests": len(TESTS),
        "records_checked": checked,
        "lower16_reference": "independent IEEE-754 binary32 to FP6 RN-satfinite model",
        "upper16": "C[31:16] preserved explicitly because public PTX destination is b16",
        "sass": [test["sass"] for test in TESTS],
        "sass_file": str(SASS),
        "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
    }
    (args.precheck_dir / "precheck-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PRECHECK PASS: {checked} results matched independent reference")


def shard_bounds(index: int, count: int) -> tuple[int, int]:
    begin = TOTAL * index // count
    end = TOTAL * (index + 1) // count
    return begin, end - begin


def full_run(args: argparse.Namespace) -> None:
    if not args.yes_large:
        raise RuntimeError("full run writes 32 GiB; pass --yes-large")
    if args.shard_count < 1:
        raise RuntimeError("--shard-count must be positive")
    end_shard = args.shard_count - 1 if args.end_shard is None else args.end_shard
    if not 0 <= args.start_shard <= end_shard < args.shard_count:
        raise RuntimeError("invalid shard interval")
    cache = BUILD.parent / "f6-f32-rn-jit-cache"
    env = runtime_env(args.compat_dir, cache)
    build(args.nvcc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for test in TESTS:
        directory = args.output_dir / str(test["slug"])
        directory.mkdir(parents=True, exist_ok=True)
        for shard in range(args.start_shard, end_shard + 1):
            start, count = shard_bounds(shard, args.shard_count)
            path = directory / f"d__shard-{shard:05d}-of-{args.shard_count:05d}.bin"
            wanted = count * RESULT_SIZE
            if path.exists() and path.stat().st_size == wanted:
                print(f"resume: {path} ({wanted} bytes)")
                continue
            if path.exists():
                path.unlink()
            print(f"capture {test['slug']} shard {shard + 1}/{args.shard_count} "
                  f"A=0x{start:08x} records={count}", flush=True)
            run_binary(test, start, count, 1, path, args.chunk_records, env)
    report(args)


def report(args: argparse.Namespace) -> None:
    manifests = []
    complete = True
    total_bytes = 0
    for test in TESTS:
        files = []
        for shard in range(args.shard_count):
            start, count = shard_bounds(shard, args.shard_count)
            path = args.output_dir / str(test["slug"]) / f"d__shard-{shard:05d}-of-{args.shard_count:05d}.bin"
            size = path.stat().st_size if path.exists() else None
            ok = size == count * RESULT_SIZE
            complete &= ok
            if size is not None:
                total_bytes += size
            files.append({"path": str(path), "a_start": f"0x{start:08x}",
                          "records": count, "bytes": size, "complete": ok})
        manifests.append({"test": test["slug"], "ptx": test["ptx"], "files": files})
    precheck_path = args.precheck_dir / "precheck-report.json"
    precheck = json.loads(precheck_path.read_text()) if precheck_path.exists() else {}
    reference_pass = precheck.get("status") == "PASS"
    doc = {
        "status": "PASS" if complete and reference_pass else "INCOMPLETE",
        "capture_status": "CAPTURE_COMPLETE" if complete else "CAPTURE_INCOMPLETE",
        "accuracy_status": "SAMPLED_REFERENCE_PASS" if reference_pass else "PRECHECK_MISSING_OR_FAILED",
        "reference_results_checked": precheck.get("records_checked", 0),
        "accuracy_scope": "compare d[15:0]; d[31:16] is explicit merge-seed preservation",
        "source_a": {"start": "0x00000000", "end": "0xffffffff", "stride": 1, "inclusive": True},
        "source_b": "0xdeadbeef",
        "source_c_merge_seed": "0xdeadbeef",
        "test_count": 2,
        "result_count": 2 * TOTAL,
        "binary_bytes_present": total_bytes,
        "binary_bytes_expected": 2 * TOTAL * RESULT_SIZE,
        "tests": manifests,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "full-run-report.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"status: {doc['status']}")
    print(f"binary_bytes: {total_bytes}/{doc['binary_bytes_expected']}")
    print(f"report: {path}")


def plan(args: argparse.Namespace) -> None:
    print("tests: 2 (.RN only: E2M3x2 and E3M2x2)")
    print(f"A: 0x00000000..0xffffffff, stride 1, inclusive ({TOTAL} values/test)")
    print("B: 0xdeadbeef; merge seed C: 0xdeadbeef")
    print(f"results: {2 * TOTAL} uint32 d values")
    print(f"binary output: {2 * TOTAL * RESULT_SIZE} bytes = 32 GiB")
    print(f"shards: {args.shard_count}/test")
    print(f"output: {args.output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.command == "plan":
        plan(args)
    elif args.command == "precheck":
        precheck(args)
    elif args.command == "run":
        full_run(args)
    elif args.command == "report":
        report(args)
    else:
        plan(args)
        precheck(args)
        full_run(args)
    print(f"elapsed_seconds: {time.time() - started:.1f}")


if __name__ == "__main__":
    main()
