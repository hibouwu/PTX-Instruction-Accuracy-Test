# SM121 GB10 PTX accuracy runner

`run_gb10_ptx_accuracy.py` 是该平台的统一测试入口。它读取脚本内根据仓库根 README 表格整理的测试矩阵，完成：

1. 将 `f6x2type`、`f4x2type`、`fp16x2type`、rounding 和可选 modifier 展开为具体 PTX 指令；
2. 自动生成 `generated/gb10_ptx_accuracy_generated.cu`；
3. 使用 `nvcc -arch=sm_121a` 编译；
4. 在 GB10 上分批执行测试；
5. 将输入和 masked 结果流式写入 `.bin`；
6. 校验二进制文件结构，并可与已有参考目录逐位比较。

正式写结果前，脚本会对每条选中的具体 PTX 执行一条临时记录作为 preflight。架构、PTX JIT、CUDA toolkit 或驱动不兼容时会在产生大文件前停止。

每条展开后的具体 PTX 指令使用独立结果目录：

```text
results/
└── <test-name>/
    ├── <sweep-name>__shard-00000-of-00001.bin
    └── ...
```

例如 `add.rn.f32.f16` 的结果位于：

```text
results/mixed_add__f16__rn__nosat/
```

Golden source 为 `Ref model` 的行不在本脚本中。`.rs.bf16x2.f32` 也不属于 SM121 GB10，因此未生成。其余 GB10 行展开后共 85 条具体指令。

## 常用命令

列出所有展开后的测试：

```bash
python3 run_gb10_ptx_accuracy.py --list
```

只生成 CUDA 源码：

```bash
python3 run_gb10_ptx_accuracy.py --generate-only
```

生成并编译：

```bash
python3 run_gb10_ptx_accuracy.py --build-only
```

默认 smoke 测试：

```bash
python3 run_gb10_ptx_accuracy.py
```

只查看实际范围、记录数、分片和预计输出，不生成或编译：

```bash
python3 run_gb10_ptx_accuracy.py --tests 'fp16x2_to_f6x2*' --profile full --shard-count 16 --shard-index 0 --plan
```

筛选指令：

```bash
python3 run_gb10_ptx_accuracy.py --tests 'f32_to_f6x2*'
```

## 只测试一条指令

先查看全部展开后的测试名称：

```bash
python3 run_gb10_ptx_accuracy.py --list
```

例如，只测试下面这条指令：

```ptx
cvt.rn.satfinite.e2m3x2.f32
```

可以使用测试名称：

```bash
python3 run_gb10_ptx_accuracy.py \
  --tests 'f32_to_f6x2__e2m3x2'
```

也可以直接使用完整 PTX 指令名：

```bash
python3 run_gb10_ptx_accuracy.py \
  --tests 'cvt.rn.satfinite.e2m3x2.f32'
```

上述命令默认只执行该指令的 smoke 测试。只生成或只编译这一条指令：

```bash
python3 run_gb10_ptx_accuracy.py \
  --tests 'f32_to_f6x2__e2m3x2' \
  --generate-only

python3 run_gb10_ptx_accuracy.py \
  --tests 'f32_to_f6x2__e2m3x2' \
  --build-only
```

对这一条指令执行完整范围中的一个分片：

```bash
python3 run_gb10_ptx_accuracy.py \
  --tests 'f32_to_f6x2__e2m3x2' \
  --profile full \
  --shard-count 16 \
  --shard-index 0 \
  --yes-large
```

## FP16/BF16 到 FP6x2：正式实验前检查

`run_gb10_fp6_precheck.py` 是这组 PTX 9.1 指令在正式全量测试前的唯一检查入口：

```ptx
cvt.rn.satfinite{.relu}.{e2m3x2/e3m2x2}.{f16x2/bf16x2}
```

它只生成/编译一次 CUDA runner，然后自动执行：

1. 检查 `/usr/local/cuda-13.1/compat` 中的新版 PTX JIT，并将其加入子进程的 `LD_LIBRARY_PATH`；
2. 使用 `compute_120f` 生成 family-specific PTX，对 8 条具体指令执行单记录编译/JIT preflight；
3. 捕获 smoke 结果并校验二进制结构；
4. 对每条指令运行两次 65,536 条连续记录并逐字节比较；
5. 选择 23 个 FP16/BF16 上半 lane 边界模式，每个模式穷举下半 lane 的全部 65,536 种编码；
6. 使用独立 CPU 模型验证 RN ties-to-even、E2M3/E3M2 编码、`satfinite`、`.relu`、NaN、无穷、次正规数、负零、双 lane 位置和 padding 位；
7. 写出 `precheck-report.json`。任何阶段失败都会返回非零退出码，不应开始正式全量测试。

安装 `cuda-compat-13-1` 后直接运行：

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10
python3 run_gb10_fp6_precheck.py
```

只查看阶段、预计容量和输出位置：

```bash
python3 run_gb10_fp6_precheck.py --plan
```

默认输出约 200 MiB，位于：

```text
results/fp6-precheck/
├── smoke/
├── determinism/
│   ├── baseline/
│   └── repeat/
├── adversarial/
└── precheck-report.json
```

默认不会覆盖已有检查证据。确认需要重新运行时使用：

```bash
python3 run_gb10_fp6_precheck.py --overwrite
```

成功结束时必须同时看到：

```text
PRECHECK PASS: 184 adversarial binaries, 24117248 lanes matched the independent reference
```

并确认 `precheck-report.json` 中：

```json
"status": "PASS"
```

## FP16/BF16 到 FP6x2 全量测试

`f6x2type × fp16x2type × {.relu}` 展开为 8 条具体 PTX。每条遍历 `2^32` 个 packed 输入，单条约 64 GiB，全部约 512 GiB。建议固定使用 16 个分片，每次运行约 32 GiB。

这些 PTX ISA 9.1 指令属于 `sm_120f` family-specific 特性。先进行 PTX-only 编译和单记录 JIT preflight：

```bash
python3 run_gb10_fp6_precheck.py
```

只有统一实验前检查生成 `PASS` 报告后，才逐个执行全量分片。正式 runner 也必须继续加载 CUDA 13.1 compatibility 库：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python3 run_gb10_ptx_accuracy.py --tests 'fp16x2_to_f6x2*' --arch compute_120f --profile full --shard-count 16 --shard-index 0 --yes-large
```

将 `--shard-index` 从 0 依次运行到 15。脚本会再次自动 preflight，并在磁盘不足时拒绝运行。

如果 precheck 报 `the provided PTX was compiled with an unsupported toolchain`，说明进程没有成功加载 CUDA 13.1 compatibility JIT，或当前驱动/toolkit 组合不兼容；不要开始全量任务。直接使用 `sm_121a`、`sm_121f` 或 `sm_120f` 生成 cubin 时，当前 ptxas 会以 feature not supported 拒绝这组指令。

按 16 个分片运行完整输入空间中的第 3 个分片：

```bash
python3 run_gb10_ptx_accuracy.py \
  --profile full --shard-count 16 --shard-index 3 --yes-large
```

与另一目录中的同名参考二进制逐位比较：

```bash
python3 run_gb10_ptx_accuracy.py \
  --reference-dir /path/to/reference/results
```

参考目录需要使用相同的 `<test-name>/<sweep-name>__shard-...bin` 层级。根目录下的 manifest 使用相对路径索引各指令子目录中的二进制文件。

涉及 `.s2f6x2` 以及 FP16/BF16 到 FP4/FP6 的 PTX 9.1 指令需要 CUDA 13.1 或更高版本。`bf16x2 <- f6x2/f4x2` 属于 PTX ISA 9.2，需要 CUDA 13.2 或更高版本。脚本会在编译前检查 `nvcc` 版本。

## 二进制格式

每个结果文件由 256 字节 little-endian header 和连续 16 字节 records 组成。

```text
Record:
  uint32 source_a
  uint32 source_b
  uint32 source_c
  uint32 masked_result
```

Header 保存格式版本、具体指令名、result mask、完整 sweep 记录数以及当前分片的起点和记录数。脚本会验证文件大小与 header 一致，并输出一个小型 JSON manifest 便于索引 `.bin` 文件。

结果先写入 `.bin.partial`，仅在 header 和文件长度验证成功后原子替换最终 `.bin`。manifest 名包含测试族，避免不同指令选择在相同 shard 上互相覆盖；manifest 同时记录 A/B/C 的完整范围。

未提供 `--reference-dir` 时，GB10 本身是表格指定的 golden source，因此脚本执行的是 golden capture 与完整性检查，不会把同一条硬件指令的输出伪装成独立 reference。提供参考目录后，任何 bit mismatch 都会返回非零退出码并报告首个不一致字节。

## CPU 合约测试

```bash
python3 -m unittest -v test_run_gb10_ptx_accuracy.py test_run_gb10_fp6_precheck.py
```

测试固定检查表格展开数量、FP6x2 双 lane mask、ADD/FMA stride、scaled 输入、smoke 采样、分片无缝覆盖、manifest 防覆盖、全量容量估算，以及 FP6 实验前检查的边界集合和独立参考编码。
