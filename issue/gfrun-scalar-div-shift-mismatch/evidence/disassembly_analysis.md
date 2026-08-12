# 证据：编译器反汇编分析

## 编译器将 `>> 6` 和 `/ 64` 编译为 `bic` 指令序列

编译器（clang-15，target `linx64v5`）将 C++ 的 `>> 6` 和 `/ 64` 优化为包含 `bic`（Bit Clear）指令的序列。

### 算子侧（`group_token_old.hpp:490`）

源码：
```cpp
uint32_t curDstPod = topkIndex[j] >> 6;
```

反汇编（`.LBB1_62`，地址 `0x11e6a`）：

```asm
11e6c:  lwu [s6, s7<<2], ->a7       # a7 = topkIndex[j] = expertId
11e70:  addi zero, 127, ->t          # t = 127 (expertNum - 1)
11e74:  setc.ltu t#1, a7             # bounds check: expertId < 128?
11e78:  C.BSTART.STD DIRECT, ->.LBB1_61
11e7a:  srli a7, 4, ->t              # t = expertId >> 4
11e7e:  bic t#1, 0, 2, ->t           # t = t & ~0b11 = t 清除 bit[0:1] = expertId >> 6
11e82:  addi sp, 96, ->t             # t = &dstPodLocal[0]
11e86:  c.movi 1, ->t                # t = 1
11e88:  sw.u t#1, [t#2, t#3]         # dstPodLocal[curDstPod] = 1
```

**关键**：`srli a7, 4` + `bic t#1, 0, 2` = 右移 4 位后清除低 2 位 = 右移 6 位 = `expertId >> 6`。

`bic t#1, 0, 2`（M=0, N=2）：
- **emulator（32位）**：在 32 位环内清除 bit[0:1] → `result = (expertId >> 4) & ~0x3`
- **RefModel（64位）**：在 64 位环内清除 bit[0:1] → `result = (expertId >> 4) & ~0x3`
- 此指令 M=0, N=2 在 32 位和 64 位下结果**一致**（因为只涉及低 2 位）

### 验证侧（`group_token_old.cpp:261`）

源码：
```cpp
uint32_t curDstPod = expertId / kExpertPerPod;  // / 64
```

编译器将 `/ 64` 编译为与 `>> 6` 相同的 `srli + bic` 序列。

### `expertId % 4` 的编译

源码：
```cpp
uint32_t curLocalExpId = topkIndex[j] % expertPerRank;  // % 4
```

反汇编中大量出现的模式（例如地址 `0x114d2`）：

```asm
114d2:  bic t#1, 2, 62, ->u          # M=2, N=62
114d6:  c.srli t#1, 4, ->t           # t = (result of bic) >> 4
114d8:  bic t#1, 0, 2, ->t           # 清除 bit[0:1]
```

`bic t#1, 2, 62`（M=2, N=62）：
- **emulator（32位）**：N=62 >= srcWidth=32 → 返回 `FullMask64(32)` = 0xFFFFFFFF → `result = data & ~0xFFFFFFFF = 0`
- **RefModel（64位）**：在 64 位环内清除 bit[2..63] → `result = data & 0x3`（保留低 2 位 = % 4）

**这是关键分歧点！**

在 32 位语义下（emulator），`bic t#1, 2, 62` 清除全部 32 位（结果 = 0），这不可能是 `% 4` 的正确实现。

实际上，编译器可能使用 `bic t#1, 2, 62` 来提取低 2 位（`% 4`），这在 64 位语义下是正确的（保留 bit[0:1]），但在 32 位语义下是错误的（清除全部）。

这说明 **emulator 的 `CalcInstBIC` 可能有缺陷**（N > srcWidth 时应保留低 N-srcWidth 位而非返回 0），或者 **编译器假设的是 64 位 `bic` 语义**。

## 结论

`bic` 指令的 `M=2, N=62` 编码在 32 位和 64 位环形掩码下产生完全不同的结果。emulator 和 RefModel 的实现不一致，导致 gfrun 的行为与预期不符。

需要确认：
1. ISA 规范中 `bic` 指令的环形位宽应该是 `srcWidth` 还是固定 64？
2. 编译器（clang-15）假设的是哪种语义？

如果 ISA 规范规定 `bic` 使用固定 64 位环形，则 emulator 的 `CalcInstBIC` 有缺陷。
如果 ISA 规范规定 `bic` 使用 `srcWidth` 位环形，则 RefModel 的 `ExecBIC` 有缺陷，且编译器生成的代码有误。
