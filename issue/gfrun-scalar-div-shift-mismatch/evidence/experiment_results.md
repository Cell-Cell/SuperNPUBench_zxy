# 证据：对照实验结果

## 实验总结

| 实验 | 修改 | R2（十六进制） | podInfoMatch | 结论 |
|---|---|---|---|---|
| 原始 | 无修改 | 0x2af8 | 48/1024 | 基线失败 |
| 实验 A | 算子 `>>6` → `/ expertPerPod` | 0x2740 | 48/1024 | `/` 和 `>>` 结果相同 |
| 实验 B | 算子和验证都改 `>> 6` | 0x2740 | 48/1024 | 确认 `/` 和 `>>` 编译为相同指令 |
| 实验 C | 清零移到循环开头 | 0x2740 | 48/1024 | store 重排不是根因 |
| 实验 D | `volatile dstPodLocal` | 0x2926 | 534/1024 | volatile 部分缓解 |
| 实验 E | volatile dstPodLocal + volatile 写入 | 0x2926 | 534/1024 | 写入 volatile 无额外效果 |
| 实验 F | volatile + 清零移到开头 | 0x2926 | 534/1024 | 两个修复一起仍只 534 |

## 关键结论

1. **`>> 6` 和 `/ 64` 在 gfrun 中结果完全相同**（实验 A、B）：编译器将二者编译为相同的 `bic + srli + bic` 指令序列，排除了 `>>` vs `/` 的差异假设。

2. **`volatile dstPodLocal` 使匹配数从 48→534**（实验 D）：说明编译器对 `dstPodLocal` 的优化（寄存器分配/store 重排）是部分原因。`volatile` 阻止优化后，部分位置的正确性恢复。

3. **534/1024 不完全通过**：`volatile` 只修复了 `dstPodLocal` 的 store/load 问题，但 `bic` 指令本身的位宽缺陷仍影响 `curDstPod` 的计算。

4. **`groupedTokenIds` 精确顺序完全匹配**（512/512）：scatter 顺序正确，问题仅在 `tokenSuperPodInfo` 的值计算。

## 诊断插桩结果

在验证逻辑中插桩收集首个失配点：

```
diagFirstMismatch[0] = 0x10001f
解码：
  section = 0
  pos     = 1
  pod     = 0
  actual  = 0 (算子写入，应为 1)
  expected= 1 (验证期望，正确)
```

**算子写入 0（错误），验证期望 1（正确）**。这说明 `bic` 指令的位宽缺陷导致算子侧 `curDstPod` 计算错误，`dstPodLocal[0]` 未被正确设置。

## 反汇编证据

编译器将 `expertId >> 6` 编译为：

```asm
srli a7, 4, ->t        # 右移 4 位
bic t#1, 0, 2, ->t     # 清除 bit[0:1] = 再右移 2 位 = 总共右移 6 位
```

`bic t#1, 0, 2`（M=0, N=2）在 32 位和 64 位语义下结果一致（只涉及低 2 位）。

但 `expertId % 4` 被编译为包含 `bic t#1, 2, 62` 的序列：

```asm
bic t#1, 2, 62, ->u    # M=2, N=62
```

`bic t#1, 2, 62`（M=2, N=62）：
- **32 位语义**（emulator）：N=62 >= 32 → 全掩码 → 结果 = 0
- **64 位语义**（RefModel）：清除 bit[2..63] → 保留低 2 位 = `% 4`

如果 `bic` 应使用 64 位语义（RefModel 正确），则 emulator 的 `CalcInstBIC` 有缺陷。
如果 `bic` 应使用 `srcWidth` 位语义（emulator 正确），则 RefModel 的 `ExecBIC` 有缺陷，且编译器生成的代码有误。

## Python golden 验证

| 指标 | Python（标准语义） | gfrun |
|---|---|---|
| `podInfoMatch` | 1024 / 1024 | 48 / 1024 |
| 返回值 | 0（全部通过） | 0x2af8 |
| 结果 | ✅ 全部通过 | ❌ 失败 |

Python 证明算子与验证逻辑在标准语义下功能等价，gfrun 失败是仿真器 `bic` 指令实现缺陷导致。
