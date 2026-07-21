# GB10 PTX 9.2 Scaled FP4/FP6 → BF16x2 测试

## 1. 目的与范围

正式入口：

```text
sm121-GB10/run_gb10_ptx92_scaled.py
```

该脚本统一验证 12 条 PTX 9.2 scaled low-float conversion：8 条 FP6→BF16x2 和 4 条 FP4→BF16x2，覆盖 E2M3、E3M2、E2M1、`.relu` 和 `.satfinite` 组合。

## 2. 环境和规模

| 项目 | 值 |
|---|---|
| 平台 | GB10 / SM121 |
| CUDA | 13.2 |
| 编译目标 | `compute_120f` |
| compat | `/usr/local/cuda-13.2/compat` |
| 具体 PTX | 12 |
| 正式分片 | 16 |
| 合法参考文件 | 516 |
| 独立参考 records | 8,718,336 |
| 独立参考 lanes | 17,436,672 |

## 3. 为什么分成 precheck 和 Comments capture

E2M3/E3M2 packed byte 的高两位是 padding。Comments 的 raw stride 会经过 padding 非零的 bit pattern；这些数据可以作为 GB10 golden observation，但不属于 PTX 定义的合法数值域。

因此脚本执行两条互补路径：

- `precheck`：只使用合法 FP4/FP6 编码，逐 lane 对照独立整数/指数 BF16 模型；
- `full`：严格复现 Comments 的 raw strided capture，并明确不为非法 padding 输入声明数值正确性。

## 4. 一键运行

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/sm121-GB10

python3 run_gb10_ptx92_scaled.py plan
python3 run_gb10_ptx92_scaled.py precheck
python3 run_gb10_ptx92_scaled.py full
python3 run_gb10_ptx92_scaled.py report
```

完整离线流程：

```bash
python3 run_gb10_ptx92_scaled.py all
```

必要时使用 `--start-shard`、`--end-shard` 续跑；`--overwrite-precheck` 会重新生成 precheck 数据。

## 5. 输出

```text
results/ptx92-scaled-precheck/
results/ptx92-scaled-full/
```

`.bin` 无 header，从 byte 0 开始保存统一 16-byte record：

```text
uint32 source_a
uint32 source_b
uint32 source_c
uint32 result
```

manifest/report 保存测试名、范围、分片、合法域统计和 SHA256。

## 6. 正式状态

完成后报告应包含：

```text
status = PASS
capture_status = COMMENTS_STRIDED_CAPTURE_COMPLETE
accuracy_status = DEFINED_DOMAIN_REFERENCE_PASS
test_count = 12
manifest_count = 16
```

`DEFINED_DOMAIN_REFERENCE_PASS` 只对 PTX 合法编码负责；`COMMENTS_STRIDED_CAPTURE_COMPLETE` 表示 raw Comments 范围已捕获。两者不能互相替代。
