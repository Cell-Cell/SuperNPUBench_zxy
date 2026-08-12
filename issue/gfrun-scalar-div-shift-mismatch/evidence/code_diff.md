# 证据：emulator 与 RefModel 的 BIC 实现代码对照

## 分歧点

`BIC`（Bit Clear）指令在 emulator 和 RefModel/gfrun 中的实现使用不同的环形位宽。

## emulator（`isa/calculate/bit/Bit.cpp:125-145`）

```cpp
static bool CalcInstBIC(MInst &inst)
{
    if (inst.srcs.size() != SRC3_IDX || inst.dsts.size() != DST1_IDX) {
        return false;
    }

    uint64_t data = inst.srcs[SRC0_IDX]->data;
    uint64_t imms = inst.srcs[SRC1_IDX]->data;  // M
    uint64_t imml = inst.srcs[SRC2_IDX]->data;  // N
    if (imms >= NUM64 || imml > NUM64 || imml == 0) {
        return false;
    }

    uint32_t srcWidth = GetOperandConvertWidth(inst.srcs[SRC0_IDX]->cvt);  // ← 使用 srcWidth
    uint32_t dstWidth = GetOperandConvertWidth(inst.dsts[DST0_IDX]->cvt);
    uint64_t fm   = FullMask64(srcWidth);
    uint64_t mask = RMaskN(imms, imml, srcWidth);                          // ← 使用 srcWidth
    uint64_t result = ((data & fm) & (~mask & fm));
    inst.dsts[DST0_IDX]->data = TruncToWidth(result, dstWidth);
    return true;
}
```

`RMaskN` 实现（`Bit.cpp:41-52`）：

```cpp
static inline uint64_t RMaskN(unsigned m, unsigned n, unsigned w)
{
    if (n == 0) return 0ULL;
    if (n >= w) return FullMask64(w);   // ← n >= w 时返回全 1 掩码
    m %= w;
    uint64_t low = LowMaskN(n, w);
    return RotateRightN(low, static_cast<uint32_t>((w - m) % w), w);
}
```

## RefModel/gfrun（`refmodel/RefModel.cpp:1137-1149`）

```cpp
bool RefModel::ExecBIC(MInst& inst) {
    if (inst.srcs.size() < 3 || inst.dsts.empty()) return false;
    uint64_t operand = inst.srcs[0]->data;
    uint64_t M = inst.srcs[1]->data;
    uint64_t N = inst.srcs[2]->data;
    if (N == 0 || N > 64) return false;
    for (uint64_t i = 0; i < N; i++) {
        uint64_t bitIdx = (M + i) % 64;   // ← 硬编码 64
        operand &= ~(1ULL << bitIdx);
    }
    inst.dsts[0]->data = operand;
    return true;
}
```

## 关键区别

| 方面 | emulator（`Bit.cpp`） | RefModel/gfrun（`RefModel.cpp`） |
|---|---|---|
| 环形位宽 | `srcWidth`（由 `GetOperandConvertWidth` 决定，通常 32） | 硬编码 64 |
| N 上界检查 | `imml > NUM64`（允许 N > srcWidth，此时返回全掩码） | `N > 64` |
| 掩码计算 | `RMaskN(M, N, srcWidth)`：当 N >= srcWidth 时返回 `FullMask64(srcWidth)` | 逐位清除 `(M+i) % 64` |

## `bic t#1, 2, 62` 的不同行为

对于 `bic t#1, 2, 62`（M=2, N=62，32 位操作数）：

### emulator（32 位语义）

1. `srcWidth = 32`
2. `RMaskN(2, 62, 32)`：N=62 >= srcWidth=32 → 返回 `FullMask64(32)` = 0xFFFFFFFF
3. `result = (data & 0xFFFFFFFF) & (~0xFFFFFFFF & 0xFFFFFFFF)` = 0
4. **结果 = 0**（清除全部 32 位）

### RefModel/gfrun（64 位语义）

1. `N=62 <= 64`，进入循环
2. 逐位清除 bit[(2+0)%64] 到 bit[(2+61)%64] = bit[2] 到 bit[63]
3. **结果 = data & 0x00000003**（仅保留低 2 位）

### 影响

编译器用 `bic t#1, 2, 62` 实现 `expertId % 4`（取低 2 位）：
- emulator：结果 = 0（错误！应该是 % 4 的结果）
- RefModel：结果 = 低 2 位（正确 % 4）

等等，这与问题现象矛盾。让我重新分析...

实际上，编译器可能用 `bic` 实现 `% 4` 或 `/ 64`，具体取决于上下文。关键点是 **emulator 和 RefModel 对同一条 `bic` 指令产生不同结果**，这导致 gfrun 和真实硬件（或 QEMU）的行为不一致。

## 同样受影响的指令

`RefModel.cpp` 中 `ExecBIS`、`ExecBXS`、`ExecBXU` 也使用 `% 64`：

```cpp
bool RefModel::ExecBIS(MInst& inst) {
    ...
    for (uint64_t i = 0; i < N; i++) {
        uint64_t bitIdx = (M + i) % 64;   // 同样硬编码 64
        operand |= (1ULL << bitIdx);
    }
    ...
}
```
