# README

## FP32 Arithmetic Instructions

| PTX | Golden source | Comments |
|---|---|---|
| add{.rnd}{.sat}.f32.atype d, a, c; | GB10 | Source A range: `0x0~0xFFFF`; stride = `0xFF`; Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFFFF` |
| fma.rnd{.sat}.f32.abtype d, a, b, c; | GB10 | Source A range: `0x0~0xFFFF`; stride = `0xFF`; Source B range: `0x0~0xFFFF`; stride = `0xFF`; Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFFFF` |

## Conversion Instructions

| PTX | Golden source | Comments |
|---|---|---|
| cvt.rn.satfinite{.relu}.f6x2type.fp16x2type d, a; | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF` |
| cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a; | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 8 bits of result |
| cvt.frnd3{.satfinite}.ue8m0x2.bf16x2 d, a; | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 16 bits of result |
| cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2 d, a{, scale-factor}; | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 16 bits of result |
| cvt.rn.f16x2.f6x2type d, a; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn.f16x2.f4x2type d, a; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f6x2type d, a, scale-factor; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`; Scale-factor range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn{.satfinite}{.relu}.scaled::n2::ue8m0.bf16x2.s2f6x2 d, a, scale-factor; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`; Scale-factor range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn.bf16x2.ue8m0x2 d, a; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f4x2type d, a, scale-factor; | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`; Scale-factor range: `0~0xFFFF`; stride = `0xFF` |
| cvt.rn.satfinite.f6x2type.f32 d, a, b; | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 16 bits of result |
| cvt.rn.satfinite.{e3m2x2,e2m3x2}.f32 d, a, b; | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `1`, inclusive; Source B: `0xDEADBEEF`; Source C (merge seed): `0xDEADBEEF`; Save full 32-bit `d`; compare lower 16 bits; RN matrix: `2 formats × 2^32 = 8,589,934,592` results, `32 GiB` |
| cvt.rn.satfinite.f4x2type.f32 d, a, b; | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 16 bits of result |
| cvt.{.rz,.rp}{.satfinite}.ue8m0x2.f32 d, a, b; | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `0xdeadbeef`; Only compare lower 16 bits of result |
| cvt.rn.satfinite{.relu}.scaled::n2::ue8m0.s2f6x2.f32 d, a, b, scale-factor; | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`; Source C: `bit[31:16]`, range `0~0xFFFF`; logical stride = `0xFF`; Only compare lower 16 bits of result |
| cvt.rs.satfinite.f16x2.f32 d, a, b, rbits; | B200 (`sm_100a`) | Source A: `0x33000000~0x34800000`, stride `1`, inclusive; Source B: `0xDEADBEEF`; Rbits: `0x1FFF1FFF` |
| cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits; | B200 (`sm_100a`) | Source A: `0x33000000~0x34800000`, stride `1`, inclusive; Source B: `0xDEADBEEF`; Rbits: `0x1FFF1FFF` |
| cvt.rs.satfinite.f16x2.f32 d, a, b, rbits; | B200 (`sm_100a`) | Source A: `0x23000000~0x33000000`, stride `1`, inclusive; Source B: `0xDEADBEEF`; Rbits: `0x1FFF1FFF`; Save full 32-bit `d`; `268,435,457` results (`1,073,741,828` bytes) |
| cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits; | B200 (`sm_100a`) | Source A: `0x23000000~0x33000000`, stride `1`, inclusive; Source B: `0xDEADBEEF`; Rbits: `0x1FFF1FFF`; Save full 32-bit `d`; `268,435,457` results (`1,073,741,828` bytes) |

## PTX 9.2 scaled FP4/FP6 → BF16x2 正式测试

这 12 条具体指令统一由一个脚本负责生成 `.cu`、编译、JIT preflight、重复性检查、独立参考比较、Comments 全量 capture、续跑和报告：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_ptx92_scaled.py all
```

默认使用 CUDA 13.2、`compute_120f` 和 `/usr/local/cuda-13.2/compat`，运行期间不需要联网。正式 Comments capture 为 12 条 × 66,564 records，约 12.235 MiB；实验前 baseline+repeat 约 266.314 MiB。输出位于：

```text
results/ptx92-scaled-precheck/
results/ptx92-scaled-full/
```

也可先查看容量或分阶段运行：

```bash
python3 run_gb10_ptx92_scaled.py plan
python3 run_gb10_ptx92_scaled.py precheck
python3 run_gb10_ptx92_scaled.py full
python3 run_gb10_ptx92_scaled.py report
```

FP6 packed source 的每个 byte 高两位是 padding，按 PTX 必须为零。Comments 的 raw stride 会包含非零 padding；脚本保留这些值作为 GB10 golden observation，但只把独立 precheck 中的合法编码计入数值 `PASS`。详细流程和交付物见 [`sm121-GB10/README.md`](sm121-GB10/README.md)。

## GB10 FP32 → FP6x2 `.RN` 全空间测试

`sm121-GB10/run_gb10_f6_f32_rn.py` 是 E2M3x2/E3M2x2 两条 `.RN` 指令的单文件入口。它会生成 `.cu`、以 CUDA 13.2 `compute_120f` 编译、通过兼容库在 GB10 上 JIT、检查实际 SASS，并先用独立 FP6 参考模型检查 524,288 个分布/Inf/NaN 输入。全量阶段按 stride 1 遍历 A 的全部 32-bit pattern：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_f6_f32_rn.py plan
python3 run_gb10_f6_f32_rn.py precheck
python3 run_gb10_f6_f32_rn.py run --yes-large
```

B 和 merge seed C 固定为 `0xDEADBEEF`。每条指令产生 `2^32` 个无 header、little-endian `uint32 d`，每种格式 16 GiB，合计 32 GiB；默认每种格式拆成 64 个 256 MiB 分片并支持按完整分片续跑。精度比较对象是 `d[15:0]`。由于公开 PTX 的 destination 是 `.b16`，文件中的 `d[31:16]` 由脚本显式保留 `C[31:16]`，不能把这部分当作 PTX 直接验证的硬件 merge 结果。结果目录为：

```text
sm121-GB10/results/f6-f32-rn-precheck/
sm121-GB10/results/f6-f32-rn-full/
```

## B200 FP16x2/BF16x2 stochastic-rounding mapping 测试

`B200/run_b200_cvt_rs_bf16x2.py` 是两条指令共用的单文件入口。它会自动生成 `.cu`、以 `sm_100a` 编译、检查 B200、反汇编核对 PTX→SASS 映射、运行重复性/边界 precheck、遍历完整 A 区间，并用独立 FP16/BF16 随机舍入参考逐 lane 比较。运行期间不需要联网：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/B200
python3 run_b200_cvt_rs_bf16x2.py plan
python3 run_b200_cvt_rs_bf16x2.py precheck
python3 run_b200_cvt_rs_bf16x2.py run
```

正式范围为 A=`0x33000000~0x34800000`、stride `1`、包含两端，共 25,165,825 个 bit pattern；B 固定为 `0xDEADBEEF`，Rbits 固定为 `0x1FFF1FFF`。FP16 使用每个半字的低 13 个随机位，BF16 使用完整 16 位。每个 `.bin` 没有 header，从 byte 0 开始只连续保存 little-endian `uint32 d`；输入规格保存在 JSON 报告中：

```text
B200/results/cvt-rs-satfinite-bf16x2-f32/full/f16x2/d.bin   ≈ 96.0 MiB
B200/results/cvt-rs-satfinite-bf16x2-f32/full/bf16x2/d.bin  ≈ 96.0 MiB
```

两条合计约 192.0 MiB。`full-run-report.json` 只有在两条完整结果都匹配参考后才写出 `STRIDE1_CAPTURE_COMPLETE` 和 `INDEPENDENT_REFERENCE_PASS`。由于 PTX 不规定 NaN payload，NaN 只检查目标格式的 NaN 类别。脚本预期并强制验证以下 PTX→SASS 对应关系，任一函数出现另一条 mnemonic 会立即失败：

```text
cvt.rs.satfinite.f16x2.f32  → F2FP.SATFINITE.F16.F32.PACK_AB.RS
cvt.rs.satfinite.bf16x2.f32 → F2FP.SATFINITE.BF16.F32.PACK_AB.RS
```

2026-07-17 B200 实测两条指令均完成 25,165,825 records / 50,331,650 lanes，所有 lane bitwise 匹配独立参考，且 SASS 对应关系未写反。结果 SHA256：

```text
f16x2:  225d47353206381d2b38c2dfabd1f51538b0473f193866e09a16ad13464fba44
bf16x2: 93c6e6c45b3333ed85c0e69648ccaeae80e28000c5706485fcf0279f89506126
```

最终 CUDA 数据通路为 `global Input{a,b,rbits}[] → 3×LDG → PTX → global d[] → D2H → .bin`。脚本还会保存生成的 `.cu`、可执行文件、SASS 及其 SHA256。`run` 会覆盖两条正式 `d.bin`；已有完整结果时可重新校验并生成报告：

```bash
python3 run_b200_cvt_rs_bf16x2.py report
```

## 当前测试数据通路与结论边界

GB10 统一 runner 生成的每个 `.bin` 均无 header，从 byte 0 开始直接保存固定宽度 record，元数据保存在 manifest JSON。

GB10 统一 runner 使用 **procedural/index-generated input**，即每个 GPU thread 根据全局索引和 `start / maximum / stride / count` 在 thread-local 标量中生成输入，再把它作为寄存器操作数传给 inline PTX。执行结果与对应输入一起写入 global output buffer，随后分块拷回 host 并 dump 为 `.bin`：

```text
Host 传入 RangeSpec
        ↓
GPU thread 根据 global index 计算 source_a/source_b/source_c
        ↓
thread-local scalar（通常由编译器分配到寄存器）
        ↓
inline PTX
        ↓
global Record{source_a, source_b, source_c, result}
        ↓
cudaMemcpy DeviceToHost
        ↓
binary dump
```

默认每个 record 为 16 bytes；runner 以最多 1,048,576 records 为一个执行 chunk，因此主要 device output buffer 约为 16 MiB。输入值不需要预先占用一块 16 GiB global memory。

该 GB10 路径与 `global input → LDG → PTX → global output` 方法不同：它没有预先初始化 global input array，也没有从该数组执行 LDG。B200 的 `.rs` 正式脚本已另行实现并通过反汇编验证 global-input/LDG 数据路径；两类结果应按各自报告中的 `data_path` 字段表述。

当前范围也必须按 manifest/Comments 解读：

- `0~0xffffffff, stride 0xffffff` 表示包含终点在内的 258 个采样值，不是 2³² 个 bit pattern 的 exhaustive 枚举。
- 多输入指令按 README 中定义的 sweep/笛卡尔积运行，不能表述为所有 2⁶⁴ 或 2⁹⁶ 输入组合。
- PTX 9.2 专用 precheck 会穷举全部合法 packed FP4/FP6 source，并让每条 lane 覆盖全部 256 个 UE8M0 scale code；它的数值 PASS 只适用于 PTX 定义域。

报告状态含义：

| 状态 | 可以证明 | 不能证明 |
|---|---|---|
| `CAPTURE_COMPLETE` | 编译、JIT、执行、输入枚举、分片、headerless payload、manifest、文件集合和 SHA256 完整 | GPU 输出与独立数学/指令语义模型一致 |
| `DEFINED_DOMAIN_REFERENCE_PASS` | 在完整 capture 基础上，PTX 合法定义域内的 GPU 输出逐 lane 匹配独立软件参考 | 保留位/padding 非法输入的数值正确性，或尚未运行的 LDG 数据路径 |

因此 CUDA 13.1 的 73 条结果应表述为 **GB10 strided golden capture with structural validation**；CUDA 13.2 的 12 条 PTX 9.2 scaled conversion 还可以表述为 **PTX-defined-domain independent reference PASS**。

## GB10 全部 strided 指令一键测试

全局范围规则：

```text
0~0xFFFFFFFF: stride = 0xFFFFFF，共 258 个值（包含终点）
0~0xFFFF:     stride = 0xFF，共 258 个值（包含终点）
高 16 位 scale: 物理 stride = 0x00FF0000，逻辑 stride = 0xFF
```

CUDA 13.1 当前覆盖 73 条具体 PTX、125 个 sweep，总输出约 12.846 GiB：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_all_strided.py full --yes-large
```

该命令在缺少 precheck 时会自动完成编译/JIT preflight 和重复性检查，然后运行或续跑全部 16 个分片。默认结果目录独立于历史 stride-1 数据：

```text
results/all-strided-cuda13.1-precheck/
results/all-strided-cuda13.1-full/
```

如果结果由旧版脚本生成，更新代码后先原地封存，不需要重跑 12.846 GiB：

```bash
python3 run_gb10_all_strided.py seal
```

`seal` 会核对运行时原始 manifest、逐记录输入枚举和 `test_id`，为每个二进制加入 SHA256 与测试规格摘要；它不会根据新代码重写旧 PTX/range。没有独立数值模型的统一报告状态为 `CAPTURE_COMPLETE`，不能表述为全部指令精度 `PASS`。

CUDA 13.2 环境运行全部 85 条：

```bash
python3 run_gb10_all_strided.py full --cuda-version 13.2 --yes-large
```

详细说明和二进制格式见 [`sm121-GB10/README.md`](sm121-GB10/README.md)。
