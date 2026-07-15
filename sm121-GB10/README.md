# SM121 GB10 PTX accuracy runner

本目录使用统一 stride 规范生成 GB10 golden data：

```text
0~0xFFFFFFFF: stride = 0xFFFFFF
0~0xFFFF:     stride = 0xFF
```

`ValueRange` 会在 stride 未自然落到最大值时额外包含终点，因此两种范围都包含 258 个值。存放在 `source_c[31:16]` 的 scale-factor 使用物理 stride `0x00FF0000`，对应逻辑 16-bit scale stride `0xFF`。

## 推荐入口：全部 strided 测试

`run_gb10_all_strided.py` 是正式运行入口，自动完成：

1. 按 Toolkit 版本选择可编译的具体 PTX；
2. 生成一个 CUDA runner 并编译为 `compute_121a` PTX；
3. 加载对应 CUDA compatibility JIT；
4. 对每条具体 PTX 执行单记录 preflight；
5. 两次执行 smoke 并逐字节验证确定性；
6. 按 16 个分片运行或续跑全部 sweep；
7. 校验 header、范围、文件长度、manifest 和 `.partial`；
8. 生成最终 `full-run-report.json`。

CUDA 13.1 当前覆盖：

```text
73 条具体 PTX
125 个 sweep
16 个分片
约 12.846 GiB
```

查看计划：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_all_strided.py plan
```

一条命令运行或续跑；首次运行会自动先做 precheck：

```bash
python3 run_gb10_all_strided.py full --yes-large
```

只跑指定分片：

```bash
python3 run_gb10_all_strided.py full \
  --start-shard 3 \
  --end-shard 3 \
  --yes-large
```

验证已有结果并重新生成报告：

```bash
python3 run_gb10_all_strided.py report
```

默认输出：

```text
results/all-strided-cuda13.1-precheck/
└── precheck-report.json

results/all-strided-cuda13.1-full/
├── full-run-report.json
├── manifest-*.json
└── <test-name>/<sweep>__shard-*.bin
```

CUDA 13.2 下，同一脚本会加入 PTX ISA 9.2 的 F6/F4 → BF16 scaled 变体，共 85 条具体 PTX、137 个 sweep：

```bash
python3 run_gb10_all_strided.py plan --cuda-version 13.2
python3 run_gb10_all_strided.py full --cuda-version 13.2 --yes-large
```

需要安装 CUDA Toolkit 13.2，并通过 `--compat-dir` 指定匹配的 compatibility JIT（若默认目录不可用）。CUDA 13.1 的 compatibility package 不能让 CUDA 13.1 `nvcc/ptxas` 获得 PTX ISA 9.2 支持。

## 独立输出目录保护旧结果

历史 FP6 与 bounded 结果使用 stride 1，是新稀疏输入集合的严格超集，仍保留为历史证据：

```text
results/fp6-full/
results/fp6-precheck/
results/bounded-conversions-full/
results/bounded-conversions-precheck/
```

新脚本不会写入这些目录。专用 FP6/bounded 脚本的默认输出也已改为：

```text
results/fp6-strided-full/
results/fp6-strided-precheck/
results/bounded-conversions-strided-full/
results/bounded-conversions-strided-precheck/
```

因此修改 stride 后不会覆盖历史约 1.5 TiB 的 `.bin`。

## 专用 FP6 独立参考检查

需要单独验证 FP6 软件参考时运行：

```bash
python3 run_gb10_fp6_precheck.py
python3 run_gb10_fp6_full.py --yes-large
```

新的 FP6 precheck 对 8 条具体 PTX 的全部 258 个 packed 输入做重复性检查，并使用独立 E2M3/E3M2 软件模型比较 4,128 个 lane。默认报告：

```text
results/fp6-strided-precheck/precheck-report.json
results/fp6-strided-full/full-run-report.json
```

## 通用底层 runner

列出全部 85 条矩阵定义：

```bash
python3 run_gb10_ptx_accuracy.py --list
```

筛选某条指令并查看计划：

```bash
python3 run_gb10_ptx_accuracy.py \
  --tests 'mixed_fma*' \
  --profile full \
  --shard-count 16 \
  --shard-index 0 \
  --plan
```

`run_gb10_ptx_accuracy.py` 负责测试矩阵、CUDA 生成、编译、执行、二进制 header、manifest 和 reference-dir 比较。正常正式测试优先使用 `run_gb10_all_strided.py`，避免手工遗漏分片。

## 结果语义

没有 `--reference-dir` 时，结果属于：

```text
GB10 golden capture with structural validation
```

这证明输入范围、JIT 执行、二进制结构、分片完整性和重复性，但不能表述成与独立数值模型比较通过。FP6 专用 precheck 是当前具备独立软件数值参考的例外。

## 二进制格式

每个文件包含 256-byte little-endian header 和连续 16-byte records：

```text
uint32 source_a
uint32 source_b
uint32 source_c
uint32 masked_result
```

文件大小：

```text
256 + shard_records × 16 bytes
```

header 保存格式版本、具体指令名、result mask、完整 sweep 记录数、分片起点和分片记录数。输入范围和 stride 同时记录在 manifest 中。结果先写入 `.bin.partial`，校验成功后再原子替换为 `.bin`。

## CPU 合约测试

```bash
python3 -m unittest -v \
  test_run_gb10_ptx_accuracy.py \
  test_run_gb10_fp6_precheck.py
```

合约测试会检查全部 85 条矩阵定义、全局 stride、endpoint 包含规则、CUDA 13.1/13.2 选择数量、ADD/FMA 多 sweep、容量估算、分片覆盖和 FP6 软件参考表。
