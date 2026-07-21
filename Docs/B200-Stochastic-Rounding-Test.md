# B200 FP32 → FP16x2/BF16x2 Stochastic Rounding 测试

## 1. 指令

正式脚本：

```text
B200/run_b200_cvt_rs_bf16x2.py
```

验证两条 stochastic rounding conversion：

```ptx
cvt.rs.satfinite.f16x2.f32  d, a, b, rbits;
cvt.rs.satfinite.bf16x2.f32 d, a, b, rbits;
```

目标 SASS mapping：

```text
F2FP.SATFINITE.F16.F32.PACK_AB.RS
F2FP.SATFINITE.BF16.F32.PACK_AB.RS
```

## 2. 输入规范

| 输入 | 值 |
|---|---|
| Source A | `0x33000000`～`0x34800000`，stride 1，包含两端 |
| Source B | `0xDEADBEEF` |
| Rbits | `0x1FFF1FFF` |

每条指令覆盖 25,165,825 个 A bit pattern。FP16 每个 lane 使用低 13 个随机位；BF16 每个 lane 使用完整 16 个随机位。

## 3. 数据通路

```text
global Input{a,b,rbits}[]
    → 3 × LDG
    → inline PTX
    → global uint32 d[]
    → D2H
    → headerless d.bin
```

脚本还会保存生成的 CUDA source、可执行文件、SASS 和相应 SHA256。

## 4. 运行

```bash
cd /home/xp6/PTX-Instruction-Accuracy-Test/B200

python3 run_b200_cvt_rs_bf16x2.py selftest
python3 run_b200_cvt_rs_bf16x2.py plan
python3 run_b200_cvt_rs_bf16x2.py precheck
python3 run_b200_cvt_rs_bf16x2.py run
python3 run_b200_cvt_rs_bf16x2.py report
```

默认以 `sm_100a` 编译。`precheck` 检查设备、SASS mapping、边界、重复性和软件参考；`run` 生成两条正式结果并执行独立参考验证。

## 5. 二进制格式

每个 `.bin` 都没有 header，仅保存连续 little-endian `uint32 d`：

```text
B200/results/cvt-rs-satfinite-bf16x2-f32/full/f16x2/d.bin
B200/results/cvt-rs-satfinite-bf16x2-f32/full/bf16x2/d.bin
```

每个文件：

```text
25,165,825 × 4 = 100,663,300 bytes，约 96.0 MiB
```

第 `i` 个结果对应：

```text
A = 0x33000000 + i
```

输入规格和状态记录在 JSON，不嵌入 `.bin`。

## 6. 结果状态

正式报告使用：

```text
status = PASS
capture_status = STRIDE1_CAPTURE_COMPLETE
accuracy_status = INDEPENDENT_REFERENCE_PASS
sass_mapping_status = PTX_TO_SASS_MAPPING_PASS
```

这表示完整 stride-1 区间已捕获、每个结果通过独立 FP16/BF16 stochastic-rounding 参考，并确认 PTX 实际映射到预期 `.RS` SASS。
