# PTX 指令精度测试文档索引

本目录记录仓库内正式测试脚本的目的、输入空间、执行方式、二进制格式和结论边界。临时单值 probe 不作为正式交付脚本，不单独建档。

| 文档 | 平台 | 正式入口 | 用途 |
|---|---|---|---|
| [GB10 统一 PTX strided capture](GB10-Unified-PTX-Strided-Capture.md) | GB10 | `run_gb10_all_strided.py` | CUDA 13.1/13.2 支持矩阵的统一 16 分片 capture |
| [GB10 bounded conversion](GB10-Bounded-Conversion-Test.md) | GB10 | `run_gb10_bounded_conversions.py` | 20 条可控规模 conversion 的独立运行入口 |
| [GB10 FP6 strided test](GB10-FP6-Strided-Test.md) | GB10 | `run_gb10_fp6_precheck.py`、`run_gb10_fp6_full.py` | 8 条 FP16x2/BF16x2→FP6x2 指令的参考检查与 capture |
| [GB10 PTX 9.2 scaled test](GB10-PTX92-Scaled-Test.md) | GB10 | `run_gb10_ptx92_scaled.py` | 12 条 scaled FP4/FP6→BF16x2 指令 |
| [GB10 FP32→FP6x2 `.RN` 全空间测试](GB10-FP32-to-FP6x2-RN-Exhaustive-Test.md) | GB10 | `run_gb10_f6_f32_rn.py` | 两条 `.RN` 指令的完整 32-bit Source A capture |
| [B200 stochastic rounding test](B200-Stochastic-Rounding-Test.md) | B200 | `run_b200_cvt_rs_bf16x2.py` | FP16x2/BF16x2 `.rs` mapping 与独立参考验证 |

## 脚本层级

```text
run_gb10_ptx_accuracy.py          底层矩阵、CUDA 生成、执行与 payload 校验
├── run_gb10_all_strided.py       GB10 全矩阵正式入口
├── run_gb10_bounded_conversions.py
├── run_gb10_fp6_precheck.py
├── run_gb10_fp6_full.py
└── run_gb10_ptx92_scaled.py

run_gb10_f6_f32_rn.py            独立的一键全空间脚本
run_b200_cvt_rs_bf16x2.py         独立的 B200 一键脚本
```

`run_gb10_ptx_accuracy.py` 是可直接调用的底层 runner，但正式批量实验应优先使用对应工作流入口，以保留 precheck、续跑、manifest 和报告约束。

## 无 header 格式总览

| 脚本族 | `.bin` payload |
|---|---|
| GB10 统一 runner 及其包装器 | 连续 16-byte records：`source_a/source_b/source_c/result` |
| GB10 FP32→FP6x2 `.RN` | 连续 little-endian `uint32 d`，4 bytes/result |
| B200 `.rs` | 连续 little-endian `uint32 d`，4 bytes/result |

所有 `.bin` 均从 byte 0 开始保存 payload。指令名、范围、分片和状态放在 JSON manifest/report 中。
