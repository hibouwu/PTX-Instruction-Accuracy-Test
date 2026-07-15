# SM121 GB10 PTX accuracy runner

`run_gb10_ptx_accuracy.py` 是该平台的统一测试入口。它读取脚本内根据仓库三张截图整理的测试矩阵，完成：

1. 将 `f6x2type`、`f4x2type`、`fp16x2type`、rounding 和可选 modifier 展开为具体 PTX 指令；
2. 自动生成 `generated/gb10_ptx_accuracy_generated.cu`；
3. 使用 `nvcc -arch=sm_121a` 编译；
4. 在 GB10 上分批执行测试；
5. 将输入和 masked 结果流式写入 `.bin`；
6. 校验二进制文件结构，并可与已有参考目录逐位比较。

图片中 Golden source 为 `Ref model` 的黑色行不在本脚本中。`.rs.bf16x2.f32` 也不属于 SM121 GB10，因此未生成。其余绿色 GB10 行展开后共 85 条具体指令。

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

涉及 `.s2f6x2` 以及 FP16/BF16 到 FP4/FP6 的 PTX 9.1 指令需要 CUDA 13.1 或更高版本。脚本会在编译前检查 `nvcc` 版本。

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

未提供 `--reference-dir` 时，GB10 本身是表格指定的 golden source，因此脚本执行的是 golden capture 与完整性检查，不会把同一条硬件指令的输出伪装成独立 reference。提供参考目录后，任何 bit mismatch 都会返回非零退出码并报告首个不一致字节。
