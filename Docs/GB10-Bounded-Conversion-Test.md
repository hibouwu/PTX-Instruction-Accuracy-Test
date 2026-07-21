# GB10 Bounded Conversion 测试

## 1. 目的

该脚本提供一个规模较小、可独立运行的 GB10 conversion 子矩阵：

```text
sm121-GB10/run_gb10_bounded_conversions.py
```

它覆盖 20 条具体 PTX，主要包含 FP4、FP6、UE8M0 和 BF16 packed conversion。输入使用 README Comments 对应的全局 stride，因此完整 capture 仅为数 MiB。

## 2. 配置

| 项目 | 值 |
|---|---|
| 平台 | GB10 / SM121 |
| 编译目标 | `compute_121a` |
| CUDA | 13.1 及相容环境 |
| 具体指令数 | 20 |
| 分片数 | 16 |
| 正式无头 payload | 4,326,144 bytes |
| 默认输出 | `results/bounded-conversions-strided-full/` |

## 3. 运行

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10

python3 run_gb10_bounded_conversions.py precheck
python3 run_gb10_bounded_conversions.py plan
python3 run_gb10_bounded_conversions.py full --yes-large
python3 run_gb10_bounded_conversions.py report
```

可用以下参数限制分片：

```bash
--start-shard N --end-shard M
```

若 precheck 缺失，正式 `full` 会先要求或执行必要检查，避免直接产生未经 JIT/重复性验证的数据。

## 4. 输出格式

该脚本复用统一 GB10 runner。每个 `.bin` 无 header，payload 从 byte 0 开始，每条 record 为：

```text
uint32 source_a
uint32 source_b
uint32 source_c
uint32 masked_result
```

元数据和 SHA256 保存在 manifest。完整分片可续跑，`.partial` 不计入正式结果。

## 5. 结论边界

报告采用：

```text
status = CAPTURE_COMPLETE
capture_status = PASS
accuracy_status = NOT_INDEPENDENTLY_VALIDATED
```

这表示 20 条指令完成结构化 golden capture、输入枚举和重复性检查；除非另有独立参考报告，不能称为数值语义逐结果 PASS。
