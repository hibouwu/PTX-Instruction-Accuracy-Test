# README

## FP32 Arithmetic Instructions

| PTX | Golden source | Comments | States |
|---|---|---|---|
| `add{.rnd}{.sat}.f32.atype d, a, c;` | GB10 | Source A range: `0x0~0xFFFF`; stride = `0xFF`<br>Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFFFF` | **READY** — all-strided CUDA 13.1 |
| `fma.rnd{.sat}.f32.abtype d, a, b, c;` | GB10 | Source A range: `0x0~0xFFFF`; stride = `0xFF`<br>Source B range: `0x0~0xFFFF`; stride = `0xFF`<br>Source C range: `0x0~0xFFFFFFFF`; stride = `0xFFFFFF` | **READY** — all-strided CUDA 13.1 |

## Conversion Instructions

| PTX | Golden source | Comments | States |
|---|---|---|---|
| `cvt.rn.satfinite{.relu}.f6x2type.fp16x2type d, a;` | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF` | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a;` | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 8 bits of result | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.frnd3{.satfinite}.ue8m0x2.bf16x2 d, a;` | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2 d, a{, scale-factor};` | GB10 | Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn.f16x2.f6x2type d, a;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn.f16x2.f4x2type d, a;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f6x2type d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`<br>Scale-factor range: `0~0xFFFF`; stride = `0xFF` | **FORMAL SCRIPT READY** — CUDA 13.2 |
| `cvt.rn{.satfinite}{.relu}.scaled::n2::ue8m0.bf16x2.s2f6x2 d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`<br>Scale-factor range: `0~0xFFFF`; stride = `0xFF` | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn.bf16x2.ue8m0x2 d, a;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF` | **LEGACY PASS** — stride 1 superset; **READY** — strided |
| `cvt.rn{.relu}{.satfinite}.scaled::n2::ue8m0.bf16x2.f4x2type d, a, scale-factor;` | GB10 | Source B range: `0~0xFFFF`; stride = `0xFF`<br>Scale-factor range: `0~0xFFFF`; stride = `0xFF` | **FORMAL SCRIPT READY** — CUDA 13.2 |
| `cvt.rn.satfinite.f6x2type.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | **READY** — all-strided CUDA 13.1 |
| `cvt.rn.satfinite.f4x2type.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | **READY** — all-strided CUDA 13.1 |
| `cvt.{.rz,.rp}{.satfinite}.ue8m0x2.f32 d, a, b;` | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `0xdeadbeef`<br>Only compare lower 16 bits of result | **READY** — all-strided CUDA 13.1 |
| `cvt.rn.satfinite{.relu}.scaled::n2::ue8m0.s2f6x2.f32 d, a, b, scale-factor;` | GB10 | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C: `bit[31:16]`, range `0~0xFFFF`; logical stride = `0xFF`<br>Only compare lower 16 bits of result | **READY** — all-strided CUDA 13.1 |
| `cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits;` | SM110-THOR | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C (`Rbits`): lower 16 bits = higher 16 bits; range `0~0xFFFF`; stride = `0xFF` | SM121 unsupported |

States 说明：

- **LEGACY PASS**：历史 stride-1 数据是新稀疏规范的严格超集，保留为证据，但其 manifest 不冒充新 stride 结果。
- **READY — all-strided CUDA 13.1**：已生成统一脚本，覆盖当前 Toolkit 可编译的 73 条具体 PTX。
- **READY — requires CUDA 13.2**：同一脚本已经包含这些变体，升级 Toolkit 后用 `--cuda-version 13.2` 运行全部 85 条。
- **FORMAL SCRIPT READY — CUDA 13.2**：12 条 PTX 9.2 变体已有独立的单文件正式入口；除 Comments capture 外，还对 PTX 定义域执行独立 BF16 数值参考检查。

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
