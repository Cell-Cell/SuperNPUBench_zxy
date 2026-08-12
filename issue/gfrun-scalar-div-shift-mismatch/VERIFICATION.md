# 验证原理与方法论

## 一、验证总体设计

### 核心思路：4 阶段递进，逐步排除假设

```
阶段 1: bic 指令有代码差异吗？ ──有──→ 差异影响实际运算吗？
                                   │不影响
                                   └──→ 排除 bic，寻找真正根因
                                          │
阶段 2: >> 和 / 有差异吗？ ──────无──→ 排除 >>/ 差异
                                   │
阶段 3: bic 指令单独运行正确吗？ ──是──→ 排除 bic 指令 bug
                                   │
阶段 4: 写后清零模式有问题吗？ ────是──→ 确认根因
```

**一键运行完整验证链**：

```bash
python3 verify_full_chain.py
```

该脚本自动执行阶段 1-2（纯 Python），并检测 gfrun 是否可用以执行阶段 3-4。

---

## 二、各阶段详解

### 阶段 1：bic 指令实现差异（代码层面）

**原理**：直接对比两个文件的 `bic` 指令实现逻辑，确认是否存在代码差异。

**方法**：纯 Python 模拟两种实现：
- emulator（`Bit.cpp`）：`RMaskN(M, N, srcWidth=32)` — 32 位环形掩码
- RefModel（`RefModel.cpp`）：`(M + i) % 64` — 64 位环形掩码

**结果**：

```
bic(data=57, M=2, N=62):
  emulator(32位): 0  (N=62 > 32 → 全掩码 → 清零全部)
  refmodel(64位): 1  (64 位环内清除 bit[2..63] → 保留低 2 位)
  ❌ 不一致
```

**结论**：两个文件的 bic 实现**确实不同**（当 N > srcWidth 时）。

---

### 阶段 2：bic 差异的实际影响分析（逻辑层面）

**原理**：发现代码差异后，需要分析这个差异是否影响编译器生成的实际运算。如果 RefModel 的 64 位语义恰好对编译器生成的指令产生正确结果，那么 bic 差异就不是根因。

**方法**：分析编译器如何使用 `bic` 指令：
- `% 4` → `bic(2, 62)`：RefModel 64 位下 = 保留低 2 位 = 正确的 `% 4` 结果
- `>> 6` → `srli 4 + bic(0, 2)`：`bic(0, 2)` 只涉及低 2 位，32/64 位一致

逐值验证 `bic(2, 62)` 用于 `% 4` 的正确性：

```
 57 % 4 = 1, refmodel=1(✓), emulator=0(✗)
 70 % 4 = 2, refmodel=2(✓), emulator=0(✗)
100 % 4 = 0, refmodel=0(✓), emulator=0(✓)  ← 巧合一致
127 % 4 = 3, refmodel=3(✓), emulator=0(✗)
```

**结论**：
- RefModel 的 64 位 bic **恰好对 `% 4` 产生正确结果**
- emulator 的 32 位 bic 对 `% 4` 产生错误结果
- **gfrun 使用 RefModel，所以 `% 4` 在 gfrun 上实际是正确的**
- bic 差异**不是运算错误的直接根因**

---

### 阶段 3：最小测试验证（运行层面）

**原理**：用不依赖 `group_token_old` 的最小程序分别测试：
- 测试 A：`bic` 指令本身是否正确（`bic_test.elf`）
- 测试 B：写后清零模式是否正确（`store_load_order.elf`）

如果测试 A 通过但测试 B 失败，说明问题不在指令而在代码模式。

**测试 A：bic 指令正确性**

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/bic_test
make TESTCASE=bic_test PLAT=linx
bin/gfrun -f .../bic_test/elf/bic_test.elf
# R2 = 0 ✅ 通过
```

测试 `% 4`、`/ 64`、`>> 6` 在 gfrun 上的正确性。用运行时数组输入避免常量折叠，用减法循环计算期望值避免编译器用 `bic` 计算期望值。

**结果**：`R2 = 0`（全部通过）

**结论**：`bic` 指令在 gfrun 中**功能正确** — 排除了 bic 指令 bug。

---

**测试 B：写后清零模式**

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/store_load_order
make TESTCASE=store_load_order PLAT=linx
bin/gfrun -f .../store_load_order/elf/store_load_order.elf
# R2 = 0x2910 = 10512
```

模拟 `group_token_old` 的关键代码模式：

```cpp
// 基础写后清零（模拟 group_token_old 的 dstPodLocal）
temp[0] = 1;                    // 设置标志
output[i*2+0] = temp[0];        // 写入输出（store A）
temp[0] = 0;                    // 清零（store B）
```

对照实验：
- 测试 1：基础写后清零（编译器可优化 temp 到寄存器）
- 测试 2：`volatile temp`（阻止寄存器优化）
- 测试 3：清零移到循环开头（改变 store 顺序）
- 测试 4：间接索引写入（更接近原始代码）

**结果**：`R2 = 0x2910 = 10512`
- 测试 1（基础写后清零）：**失败，512 个元素全部写错**
- 测试 2/3/4：通过（否则 ret 会被覆盖为更小的测试编号）

**结论**：
- 基础写后清零模式在 gfrun 上**失败**
- `volatile` 阻止优化后**通过**
- **根因是编译器对 temp 的寄存器优化导致 store/load 顺序错误**

---

### 阶段 4：根因确认总结

**完整逻辑链**：

| 阶段 | 验证 | 结果 | 排除/确认 |
|---|---|---|---|
| 1 | bic 代码差异 | ❌ 差异存在 | 发现问题线索 |
| 2 | bic 差异影响分析 | RefModel 恰好正确 | **排除** bic 差异是根因 |
| 3A | bic_test on gfrun | ✅ R2=0 通过 | **排除** bic 指令 bug |
| 3B | store_load_order on gfrun | ❌ 基础模式失败 | **确认** 写后清零是根因 |
| 3B | volatile 对照 | ✅ 通过 | **确认** 根因是寄存器优化 |

**最终结论**：

根因是**写后清零模式**（`output[i] = temp; temp = 0;`）在编译器将 `temp` 优化到寄存器后，gfrun 的 store/load 顺序处理错误。gfrun 先执行了清零（`temp = 0`），再读取 `temp` 写入 `output`，导致 `output` 被写入 0 而非实际值。

**三个关键证据**：
1. `bic` 指令本身正确（`bic_test` R2=0）
2. 基础写后清零模式失败（`store_load_order` R2≠0）
3. `volatile` 阻止优化后通过（对照组）

---

## 三、验证工具一览

| 工具 | 类型 | 依赖 | 耗时 | 验证内容 |
|---|---|---|---|---|
| `verify_full_chain.py` | Python 脚本 | 无（阶段 1-2）/ gfrun（阶段 3-4） | 1-30 秒 | 完整验证链 |
| `bic_test/` | 最小 ELF | gfrun + linx 工具链 | 30 秒 | bic 指令正确性 |
| `store_load_order/` | 最小 ELF | gfrun + linx 工具链 | 30 秒 | 写后清零模式根因复现 |
| `evidence/python_golden_validation.py` | Python 脚本 | 无 | 1 秒 | 算法逻辑正确性（1024/1024） |
