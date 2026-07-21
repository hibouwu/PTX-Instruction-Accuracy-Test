# GB10 FP32 → FP6x2 `.RN` 全空间精度测试

## 1. 测试目的

本测试在 NVIDIA GB10（SM121）上验证下列两条 FP32 到 packed FP6 的 round-to-nearest-even 转换，并保存完整输入空间的硬件输出：

```ptx
cvt.rn.satfinite.e2m3x2.f32 d, a, b;
cvt.rn.satfinite.e3m2x2.f32 d, a, b;
```

对应待覆盖的 `.RN` XISA 形式为：

```text
VCVTP.F.F.SAT.E2M3.F32.PACK_AB_MERGE_C.RN
VCVTP.F.F.SAT.E3M2.F32.PACK_AB_MERGE_C.RN
```

本轮不测试 `.RZ` 变体。反汇编末尾出现的 `RZ` 是零寄存器操作数，不代表 `.RZ` 舍入模式。

## 2. 测试环境

| 项目 | 配置 |
|---|---|
| 测试平台 | NVIDIA GB10 / SM121 |
| CUDA Toolkit | CUDA 13.2 |
| PTX 编译目标 | `compute_120f` |
| Driver JIT compatibility | `/usr/local/cuda-13.2/compat` |
| 正式脚本 | `sm121-GB10/run_gb10_f6_f32_rn.py` |
| 正式测试日期 | 2026-07-20 |

测试机器运行期间不需要访问互联网。

## 3. 输入空间

| 输入 | 范围或固定值 |
|---|---|
| Source A | `0x00000000`～`0xFFFFFFFF`，stride 1，包含两端 |
| Source B | 固定为 `0xDEADBEEF` |
| Source C / merge seed | 固定为 `0xDEADBEEF` |

Source A 的每一个值都按原始 32-bit pattern 解释为 FP32。每条具体指令覆盖：

```text
2^32 = 4,294,967,296 个 A bit pattern
```

两条指令合计：

```text
2 × 2^32 = 8,589,934,592 个 uint32 d
```

## 4. 数据通路

脚本自动生成并编译 CUDA runner。正式数据通路为：

```text
global Input{a,b,c}[]
        │
        ├── GPU 分段初始化 A/B/C
        │
        └── LDG 读取
                │
                ▼
      inline PTX FP32 → FP6x2
                │
                ▼
          global uint32 d[]
                │
                ▼
              D2H
                │
                ▼
       headerless binary shards
```

完整输入空间按分片处理，不要求一次分配 16 GiB global input。默认每次内部处理 4,194,304 条记录，然后将结果写入当前分片。

## 5. 完整 `d` 与精度比较范围

公开 PTX 的 packed FP6 destination 类型是 `.b16`，因此 PTX 直接定义并产生的是 `d[15:0]`。实测 JIT SASS 中 conversion 的 merge-C 操作数为零寄存器 `RZ`：

```text
F2FP.SATFINITE.E2M3.F32.PACK_AB_MERGE_C ..., RZ
F2FP.SATFINITE.E3M2.F32.PACK_AB_MERGE_C ..., RZ
```

这里的 `RZ` 是寄存器，不是舍入修饰符。为了按照测试需求保存完整 32-bit `d`，runner 显式构造：

```text
d[15:0]  = PTX packed FP6 conversion result
d[31:16] = Source C[31:16] = 0xDEAD
```

因此：

- 数值精度比较只针对硬件转换产生的 `d[15:0]`；
- `d[31:16]` 用于保留约定的 merge seed；
- 不能将高 16 位表述为该公开 PTX 直接验证的硬件 merge-C 结果。

packed lane 顺序为：

```text
d[5:0]   = Source B 的 FP6 结果
d[13:8]  = Source A 的 FP6 结果
d[7:6]、d[15:14] = padding，必须为 0
```

## 6. 实验前检查

正式全量运行前，脚本执行以下检查：

1. 生成 `.cu` 并用 CUDA 13.2 编译为 `compute_120f` PTX；
2. 在 GB10 上完成 Driver JIT preflight；
3. 从独立 CUDA JIT cache 提取 cubin 并反汇编；
4. 确认 E2M3 与 E3M2 两条 `F2FP.SATFINITE.*.PACK_AB_MERGE_C` 指令存在；
5. 覆盖正负分布输入、Infinity 和 NaN；
6. 将 GPU 的 `d[15:0]` 与独立 FP6 round-to-nearest-even/satfinite 软件模型逐结果比较；
7. 检查高 16 位 merge seed 和 FP6 padding 位。

precheck 共比较：

```text
524,288 个 uint32 d
```

结果：

```text
PRECHECK PASS
accuracy_status = SAMPLED_REFERENCE_PASS
```

## 7. 全量运行方式

在 GB10 本机执行：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10

python3 run_gb10_f6_f32_rn.py plan
python3 run_gb10_f6_f32_rn.py precheck
python3 run_gb10_f6_f32_rn.py run --yes-large
python3 run_gb10_f6_f32_rn.py report
```

`run` 支持按完整分片续跑。已经存在且大小正确的分片会被跳过；未完成的 `.partial` 不会被当成正式结果。

## 8. 二进制格式

每个 `.bin` 都没有 header，从 byte 0 开始连续保存 little-endian `uint32 d`：

```text
offset 0:  uint32 d[0]
offset 4:  uint32 d[1]
offset 8:  uint32 d[2]
...
```

默认每种格式拆成 64 个分片：

```text
results/f6-f32-rn-full/
├── e2m3x2-rn/
│   ├── d__shard-00000-of-00064.bin
│   └── ...
├── e3m2x2-rn/
│   ├── d__shard-00000-of-00064.bin
│   └── ...
└── full-run-report.json
```

每个分片包含 67,108,864 个结果，大小为 256 MiB。每种格式为 16 GiB，两种格式合计 32 GiB。

分片 `i` 对应的 Source A 起点为：

```text
A_start = i × 0x04000000
```

分片内部严格按 stride 1 排列。因此，第 `i` 个分片内第 `j` 个 `uint32 d` 对应：

```text
A = i × 0x04000000 + j
```

输入规格、文件集合、记录数和状态保存在 `full-run-report.json`，不写入 `.bin`。

## 9. 正式结果

本次 GB10 正式执行结果：

| 检查项 | 结果 |
|---|---:|
| 具体指令数 | 2 |
| 全量结果数 | 8,589,934,592 |
| 正式 `.bin` 数 | 128 |
| 正式二进制字节数 | 34,359,738,368 bytes（32 GiB） |
| 残留 `.partial` | 0 |
| 文件大小异常 | 0 |
| 分片首尾结构样本 | 256/256 通过 |
| 独立参考结果 | 524,288/524,288 匹配 |
| 全量耗时 | 67.8 秒 |

报告状态：

```text
status = PASS
capture_status = CAPTURE_COMPLETE
accuracy_status = SAMPLED_REFERENCE_PASS
```

正式报告位置：

```text
/home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10/results/
  f6-f32-rn-full/full-run-report.json
```

## 10. 结论与边界

本测试可以证明：

- 两条 `.RN` PTX 均能在 GB10 上通过 CUDA 13.2 Driver JIT 并执行；
- 实际 SASS 包含对应 E2M3/E3M2 FP32 packed conversion；
- Source A 的全部 `2^32` bit pattern 均已执行并捕获输出；
- 128 个 headerless 分片的数量和长度完整；
- 524,288 个对抗性结果与独立 FP6 `.RN` 软件参考一致。

本测试不能证明：

- `.RZ` 舍入变体的行为；
- 每一个全空间结果都经过独立 CPU 参考逐值比较；
- `d[31:16]` 是公开 PTX conversion 指令直接产生的 merge-C 硬件结果。

因此，对外建议将结果表述为：

> GB10 上两条 FP32→FP6x2 `.RN` 指令完成全 32-bit Source A 空间的 golden capture；完整性检查通过，并有 524,288 个对抗样本通过独立 round-to-nearest-even 参考验证。
