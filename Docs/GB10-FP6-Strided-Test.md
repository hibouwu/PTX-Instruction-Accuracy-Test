# GB10 FP16x2/BF16x2 → FP6x2 Strided 测试

## 1. 指令范围

本工作流验证 PTX 9.1 packed FP6 conversion：

```ptx
cvt.rn.satfinite{.relu}.{e2m3x2,e3m2x2}.{f16x2,bf16x2} d, a;
```

组合后共 8 条具体 PTX。两个脚本分工如下：

| 脚本 | 职责 |
|---|---|
| `run_gb10_fp6_precheck.py` | 编译/JIT、重复性、独立 FP6 软件参考 |
| `run_gb10_fp6_full.py` | 验证 precheck 后运行/续跑 16 个正式分片 |

## 2. 输入范围

packed source 使用：

```text
0x00000000～0xFFFFFFFF，stride 0xFFFFFF，包含末端
```

共 258 个 packed source。每个结果包含两个 FP6 lane；8 条指令的独立参考总计检查：

```text
8 × 258 × 2 = 4,128 lanes
```

## 3. Precheck

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10

python3 run_gb10_fp6_precheck.py --plan
python3 run_gb10_fp6_precheck.py --overwrite
```

默认使用 `compute_120f`、CUDA 13.1 compat。precheck 包含 smoke、两次完整 strided capture 的逐字节确定性比较，以及独立软件模型逐 lane 验证。

预期报告：

```text
results/fp6-strided-precheck/precheck-report.json
status = PASS
```

## 4. 正式 Capture

```bash
python3 run_gb10_fp6_full.py --plan
python3 run_gb10_fp6_full.py --yes-large
```

也可指定：

```bash
--start-shard N --end-shard M
```

正式输出默认位于 `results/fp6-strided-full/`。脚本拒绝缺失、失败或矩阵不匹配的 precheck 报告。

## 5. 二进制格式

该工作流复用统一 runner，每个 `.bin` 无 header，包含连续 16-byte records：

```text
uint32 source_a
uint32 source_b
uint32 source_c
uint32 masked_result
```

`masked_result` 仅保留该测试定义的有效结果位。输入和输出一起保存，便于独立参考重新计算。

## 6. 结论

precheck 通过并完成正式 capture 后，可以表述为：

```text
capture_status = PASS
accuracy_status = INDEPENDENT_REFERENCE_PASS
```

这里的 independent reference 覆盖 Comments 定义的 258 个 strided packed input，而不是 Source A 的完整 `2^32` bit pattern。
