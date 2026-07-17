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
| `cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits;` | B200 (`sm_100a`) | Source A range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source B range: `0~0xFFFFFFFF`; stride = `0xFFFFFF`<br>Source C (`Rbits`): lower 16 bits = higher 16 bits; 16-bit value range: `0~0xFFFF`; stride = `0xFF` | **FORMAL SCRIPT READY — B200**; SM121 unsupported |

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

## B200 BF16x2 stochastic-rounding 正式测试

`B200/run_b200_cvt_rs_bf16x2.py` 是该指令唯一的用户入口。它会自动生成 `.cu`、以 `sm_100a` 编译、检查设备、反汇编验证 SASS、运行对抗样例、执行或续跑 Comments 全量矩阵、逐记录进行独立软件参考比较，并生成二进制和 JSON 报告。运行期间不需要联网：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/B200
python3 run_b200_cvt_rs_bf16x2.py plan
python3 run_b200_cvt_rs_bf16x2.py run
```

默认矩阵是 A、B 和 16-bit Rbits 三个 258 点集合的笛卡尔积，共 17,173,512 records、34,347,024 lanes；Rbits 的同一 16-bit 值复制到 `.b32` 的高低半字。16 个 `.bin` 分片合计约 262.05 MiB，位于：

```text
B200/results/cvt-rs-satfinite-bf16x2-f32/full/
```

最终 `full-run-report.json` 的 `COMMENTS_STRIDED_CAPTURE_COMPLETE` 表示 Comments 矩阵完整，`INDEPENDENT_REFERENCE_PASS` 表示所有非 NaN lane 均 bitwise 匹配独立 `.rs + .satfinite` 模型；由于 PTX 不规定 NaN payload，NaN lane 按 BF16 NaN 类别检查。脚本还会把 `.cu`、可执行文件和 SASS 分别生成到 `B200/generated/` 与 `B200/build/`，并在报告中记录 SHA256。实测 B200 SASS 数据通路为：

```text
global Input[] → 3 × LDG → F2FP.SATFINITE.BF16.F32.PACK_AB.RS
               → global Record{a, b, rbits, result} → D2H → .bin
```

如果完整运行被中断，重新执行同一条 `run` 命令即可按 manifest 和 SHA256 校验后续跑。只有在所有分片已存在时，也可单独重建最终报告：

```bash
python3 run_b200_cvt_rs_bf16x2.py report
```

## 当前测试数据通路与结论边界

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
| `CAPTURE_COMPLETE` | 编译、JIT、执行、输入枚举、分片、header、manifest、文件集合和 SHA256 完整 | GPU 输出与独立数学/指令语义模型一致 |
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
