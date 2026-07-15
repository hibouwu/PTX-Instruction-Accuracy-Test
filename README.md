# README

## FP32 Arithmetic Instructions

| PTX | Golden source | Comments | States |
|---|---|---|---|
| `add{.rnd}{.sat}.f32.atype d, a, c;` | GB10 | Test 1:<br>Source A range: `0x0~0xFFFF`; stride = `0xFF`<br>Source C range: `0x0~0xFFFFFFFF`<br><br>Test 2:<br>Source A range: `0x0~0xFFFF`<br>Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFF` | |
| `fma.rnd{.sat}.f32.abtype d, a, b, c;` | GB10 | Test 1:<br>Source A range: `0x0~0xFFFF`; stride = `0xFF`<br>Source B range: `0x0~0xFFFF`; stride = `0xFF`<br>Source C range: `0x0~0xFFFFFFFF`<br><br>Test 2:<br>Source A range: `0x0~0xFFFF`; stride = `0xFF`<br>Source B range: `0x0~0xFFFF`<br>Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFFFF`<br><br>Test 3:<br>Source A range: `0x0~0xFFFF`<br>Source B range: `0x0~0xFFFF`; stride = `0xFF`<br>Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFF` | |

## Conversion Instructions

| PTX | Golden source | Comments | States |
|---|---|---|---|
| `cvt.rn.satfinite{.relu}.f6x2type.fp16x2type d, a;` | GB10 | Source B range: `0~0xFFFFFFFF` | |
| `cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a;` | GB10 | Source B range: `0~0xFFFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 8 bits of result | |
| `cvt.frnd3{.satfinite}.ue8m0x2.bf16x2 d, a;` | GB10 | Source B range: `0~0xFFFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | |
| `cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2 d, a{, scale-factor};` | GB10 | Source B range: `0~0xFFFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | |
| `cvt.rn.f16x2.f6x2type d, a;` | GB10 | Source B range: `0~0xFFFF` | |
| `cvt.rn.f16x2.f4x2type d, a;` | GB10 | Source B range: `0~0xFFFF` | |
| `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f8x2type d, a, scale-factor;` | Ref model |  | |
| `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f6x2type d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`<br>Scale-factor range: `0~0xFFFF` | |
| `cvt.rn{.satfinite}{.relu}.scaled::n2::ue8m0.bf16x2.s2f6x2 d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`<br>Scale-factor range: `0~0xFFFF` | |
| `cvt.rn.bf16x2.ue8m0x2 d, a;` | GB10 | Source B range: `0~0xFFFF` | |
| `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f4x2type d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`<br>Scale-factor range: `0~0xFFFF` | |
| `cvt.rn.satfinite.f6x2type.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | |
| `cvt.rn.satfinite.f4x2type.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | |
| `cvt.{.rz,.rp}{.satfinite}.ue8m0x2.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | |
| `cvt.rn.satfinite{.relu}.scaled::n2::ue8m0.s2f6x2.f32 d, a, b, scale-factor;` | GB10 | Source A range: `0~0xFFFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `bit[31:16]`, range `0~0xFFFF`<br>Only compare lower 16 bits of result | |
| `cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits;` | SM110-THOR | Source A range: `0~0xFFFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C (`Rbits`): lower 16 bits = higher 16 bits; range `0~0xFFFF` | SM121 unsupported |

## GB10 FP6 正式全量测试前检查

在执行约 512 GiB 的 FP16/BF16 → FP6x2 全量测试前，先运行统一检查入口：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_fp6_precheck.py
```

该命令自动完成 CUDA 13.1 compatibility/JIT 检查、8 条具体 PTX preflight、smoke、65,536 条重复性比较、23 个边界分片，以及 E2M3/E3M2 独立软件参考逐 lane 校验。只有生成的 `results/fp6-precheck/precheck-report.json` 为 `PASS` 后才开始正式全量分片。详细说明见 [`sm121-GB10/README.md`](sm121-GB10/README.md)。

通过 precheck 后，可在 GB10 本机断网运行或续跑全部 16 个分片：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_fp6_full.py --yes-large
```

```bash
cd /home/jianyeshi/Note/PTX-Instruction-Accuracy-Test/sm121-GB10

# 查看全部 85 条指令
./run_gb10_ptx_accuracy.py --list

# GB10 上执行 smoke
./run_gb10_ptx_accuracy.py

# 完整范围分片
./run_gb10_ptx_accuracy.py \
  --profile full \
  --shard-count 16 \
  --shard-index 0 \
  --yes-large

# 与参考结果自动比较
./run_gb10_ptx_accuracy.py \
  --reference-dir /path/to/reference/results
```
