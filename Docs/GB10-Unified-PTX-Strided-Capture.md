# GB10 统一 PTX Strided Capture

## 1. 适用范围

该工作流负责 README 测试矩阵中当前 CUDA Toolkit 可编译的 GB10 指令，包括 arithmetic、conversion、FP4/FP6、UE8M0 和 scaled conversion。正式入口为：

```text
sm121-GB10/run_gb10_all_strided.py
```

底层 `run_gb10_ptx_accuracy.py` 负责展开具体 PTX、生成 CUDA、编译、执行、payload 校验和 manifest；通常不应绕过正式入口手工运行全部矩阵。

## 2. 配置

| 项目 | 值 |
|---|---|
| 平台 | GB10 / SM121 |
| 编译目标 | `compute_121a` |
| 支持 Toolkit 矩阵 | CUDA 13.1、CUDA 13.2 |
| u32 全局 stride | `0xFFFFFF` |
| u16 全局 stride | `0xFF` |
| 正式分片数 | 16 |
| 默认 chunk | 1,048,576 records |

CUDA 13.1 矩阵包含 73 条具体 PTX、125 个具体 sweep，正式输出为 2,000 个 `.bin`，无头 payload 约 12.846 GiB。CUDA 13.2 在此基础上加入 12 条 PTX 9.2 指令。

## 3. 工作流

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10

python3 run_gb10_all_strided.py precheck --cuda-version 13.1
python3 run_gb10_all_strided.py plan     --cuda-version 13.1
python3 run_gb10_all_strided.py full     --cuda-version 13.1 --yes-large
python3 run_gb10_all_strided.py report   --cuda-version 13.1
```

需要 CUDA 13.2 全矩阵时，将 `13.1` 改为 `13.2`。可用 `--start-shard` 和 `--end-shard` 分段执行；已通过 manifest 完整性验证的分片会跳过。

`precheck` 验证编译、Driver JIT 和重复执行确定性；它不等价于所有指令都通过独立数学参考。

## 4. 数据通路

统一 runner 采用 procedural/index-generated input：

```text
global linear index
    → thread-local A/B/C 枚举
    → inline PTX
    → global Record[]
    → D2H
    → headerless .bin
```

它没有预先分配完整 16 GiB input array，也不把该工作流描述为 global-input/LDG 实验。

## 5. 二进制和 manifest

每个 `.bin` 无 header，从 byte 0 开始保存固定 16-byte little-endian record：

```text
uint32 source_a
uint32 source_b
uint32 source_c
uint32 masked_result
```

文件大小必须为：

```text
shard_records × 16 bytes
```

manifest 版本为 3，记录 PTX、`test_id`、result mask、完整 sweep 数量、分片起点、输入范围、spec SHA256 和文件 SHA256。旧的带 256-byte header 文件不能静默冒充当前格式。

## 6. 状态解释

```text
status = CAPTURE_COMPLETE
capture_status = PASS
accuracy_status = NOT_INDEPENDENTLY_VALIDATED 或 PARTIAL_REFERENCE_PASS
```

- `CAPTURE_COMPLETE`：文件集合、枚举、长度、manifest 和 SHA256 完整。
- `PARTIAL_REFERENCE_PASS`：绑定的 FP6 precheck 已通过，但不代表其他指令有独立参考。
- 只有专门工作流明确输出 reference PASS 时，才能宣称相应定义域数值正确。

## 7. 结果目录

目录由 `--output-dir` 或 CUDA 版本默认配置决定。每个测试/每个 sweep/每个 shard 单独保存，最终汇总为 `full-run-report.json`。
