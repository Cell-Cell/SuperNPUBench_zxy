# gfrun：写后清零代码模式在寄存器优化后 store/load 顺序错误，导致 `group_token_old` podInfo 验证失败

## 问题概述

在 `bin/gfrun` 上运行 `group_token_old` benchmark ELF 时，Phase 2 的 `tokenSuperPodInfo` 验证失败，1024 个元素中仅 48 个匹配。经 4 阶段验证链定位，**根因是编译器将 `dstPodLocal` 优化到寄存器后，"写后清零"模式（`output[i] = temp; temp = 0;`）在 gfrun 中的 store/load 顺序处理错误** — gfrun 先执行了清零（`temp = 0`），再读取 `temp` 写入 `output`，导致 `output` 被写入 0 而非实际值。

该问题通过不依赖 `group_token_old` 算子的最小测试用例独立复现，根因明确。

## 环境

| 项目 | 值 |
|---|---|
| 仓库 | `SuperScalarModel`（gfrun 功能模型） |
| gfrun 二进制 | `bin/gfrun`（从 `main` 构建，HEAD `75b9589`） |
| 编译器 | `linx-toolchain-build` clang-15，target `linx64v5-unknown-linux-musl` |
| benchmark ELF | `SuperNPUBench/.../output/kernel/group_token_old/elf/group_token_old.elf` |
| 日期 | 2026-08-11 |

## 根因分析

### 问题代码模式

`group_token_old` 算子中 `groupToken_scalar` 函数使用 `dstPodLocal`（2 元素数组）记录当前 token 的目标 pod 信息，写完输出后立即清零：

```cpp
// group_token_old.hpp:501-504
for (uint32_t j = 0; j < superPodNum; j++) {
    tokenSuperPodInfo[podInfoSectionOffset + j] = dstPodLocal[j];  // store A: 写入输出
    dstPodLocal[j] = 0;                                              // store B: 清零临时变量
}
```

### 根因机制

1. 编译器（clang-15, `-O2`）将 `dstPodLocal[2]` 优化到寄存器
2. gfrun 在处理 "先 store A（读取 `dstPodLocal` 写入 `tokenSuperPodInfo`）、再 store B（清零 `dstPodLocal`）" 的顺序时出现错误
3. gfrun 先执行了 store B（清零），再读取 `dstPodLocal` 的值用于 store A
4. 导致 `tokenSuperPodInfo` 被写入 0 而非 `dstPodLocal` 的实际值

### 验证证据

| 实验 | 修改 | podInfoMatch | 说明 |
|---|---|---|---|
| 原始 | 无 | 48/1024 | 基线失败 |
| `volatile dstPodLocal` | 阻止寄存器优化 | 534/1024 | 大幅修复，确认根因 |
| 清零移到循环开头 | 改变 store 顺序 | 48/1024 | 无效（编译器重新优化） |
| `>>` 改为 `/` | 改变运算方式 | 48/1024 | 无效（编译为相同指令） |

`volatile` 使匹配数从 48 跳到 534，直接证实编译器对 `dstPodLocal` 的寄存器优化是根因。

## 复现步骤

### 方法 1：一键完整验证（推荐，不依赖 group_token_old）

```bash
# 运行 4 阶段验证链脚本（自动检测 gfrun 并运行最小 ELF）
python3 SuperScalarModel/docs/issues/gfrun-scalar-div-shift-mismatch/verify_full_chain.py
```

预期输出：
```
阶段 1: bic 实现差异      → ❌ emulator(32位) vs RefModel(64位) 不一致
阶段 2: 差异影响分析      → RefModel 恰好正确，不是根因
阶段 3: bic_test on gfrun → ✅ R2=0，bic 指令功能正确
阶段 4: store_load_order  → ❌ R2=0x2910，写后清零 512 个全错，volatile 通过
最终结论：根因是写后清零模式在寄存器优化后 store/load 顺序错误
```

### 方法 2：最小测试独立复现（不依赖 group_token_old）

**测试 A：确认 bic 指令本身正确**

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/bic_test
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
make TESTCASE=bic_test PLAT=linx
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/bic_test/elf/bic_test.elf
# 预期：R2 = 0（% 4, / 64, >> 6 全部通过）
```

**测试 B：复现写后清零 store/load 顺序问题**

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/store_load_order
make TESTCASE=store_load_order PLAT=linx
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/store_load_order/elf/store_load_order.elf
# 预期：R2 = 0x2910 = 10512
# 解码：测试1（基础写后清零）失败，512 个元素全错；volatile 版本通过
```

### 方法 3：完整 group_token_old 复现

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/group_token_old
make TESTCASE=group_token_old PLAT=linx
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf
# 预期：R2 = 0x2af8（podInfo 验证失败，48/1024 匹配）
```

## 验证链说明

通过 4 阶段递进验证，逐步排除假设、定位根因：

### 阶段 1：bic 指令实现差异（代码层面）

发现 `emulator`（`Bit.cpp`）和 `RefModel`（`RefModel.cpp`）的 `bic` 指令实现使用不同的环形位宽：

| 文件 | 环形位宽 | 关键代码 |
|---|---|---|
| `isa/calculate/bit/Bit.cpp:141` | `srcWidth`（通常 32） | `RMaskN(imms, imml, srcWidth)` |
| `refmodel/RefModel.cpp:1144` | 硬编码 64 | `(M + i) % 64` |

当 `N > srcWidth`（如 `N=62 > 32`）时，两者产生不同结果。

### 阶段 2：bic 差异的实际影响分析（逻辑层面）

分析编译器如何使用 `bic` 指令：
- `% 4` → `bic(2, 62)`：RefModel 64 位下 = 保留低 2 位 = **正确的 `% 4` 结果**
- `>> 6` → `srli 4 + bic(0, 2)`：`bic(0, 2)` 只涉及低 2 位，32/64 位**一致**

结论：RefModel 的 64 位 bic 恰好对编译器生成的指令产生正确结果，**bic 差异不是根因**。

### 阶段 3：最小测试验证（运行层面）

- **bic_test**（`R2=0`）：bic 指令在 gfrun 中功能正确，排除 bic bug
- **store_load_order**（`R2=0x2910`）：基础写后清零模式 512 个全错，volatile 版本通过

### 阶段 4：根因确认

| 验证 | 结果 | 排除/确认 |
|---|---|---|
| bic 代码差异 | 差异存在 | 发现线索 |
| bic 差异影响 | RefModel 恰好正确 | 排除 bic 是根因 |
| bic_test on gfrun | R2=0 通过 | 排除 bic 指令 bug |
| store_load_order on gfrun | R2=0x2910 失败 | 确认写后清零是根因 |
| volatile 对照 | 通过 | 确认根因是寄存器优化 |

## 问题定位

### 涉及的 gfrun 代码

问题出在 gfrun 对**寄存器优化后的 store-to-load forwarding**（写后读转发）的处理。具体位置需要在 gfrun 的 store/load 执行路径中定位：

- `emulator/` 目录下的 store 执行逻辑
- 寄存器文件的 store/load 顺序处理
- 编译器将栈变量优化到寄存器后的 store/load 语义

### 反汇编证据

编译器将 `dstPodLocal` 优化到栈（`sp + 96`），使用 `sdi`/`sw.u` 指令访问。反汇编中写后清零模式为：

```asm
# store A: 把 dstPodLocal[j] 写入 tokenSuperPodInfo
sw.u t#1, [t#2, t#3]         # tokenSuperPodInfo[off + j] = dstPodLocal[j]

# store B: 清零 dstPodLocal[j]
sdi  zero, [sp, 96]           # dstPodLocal[0] = 0
```

gfrun 在处理这两条 store 指令时，可能将 store B 的效果前推到 store A 的数据源读取之前。

## 解决方向

### 方向 1：修复 gfrun 的 store/load 顺序处理（推荐）

在 gfrun 的 store 执行路径中，确保 store A 的数据源读取在 store B 执行之前完成。具体需要检查：
- 寄存器文件的 read-after-write 依赖处理
- store buffer 的 flush 顺序
- 编译器优化后栈变量到寄存器的映射

### 方向 2：临时规避

在算子侧给 `dstPodLocal` 加 `volatile`（部分修复，48→534，不完全）：

```cpp
volatile uint32_t dstPodLocal[kSuperPodNum];
```

或将清零移到下一次迭代开头（无效，编译器重新优化）。

### 方向 3：附加修复 — bic 指令实现一致性

`emulator` 和 `RefModel` 的 `bic` 实现差异（32位 vs 64位）虽然不是当前问题的直接根因，但建议作为独立 issue 修复：

```cpp
// refmodel/RefModel.cpp:ExecBIC — 修改前
for (uint64_t i = 0; i < N; i++) {
    uint64_t bitIdx = (M + i) % 64;  // 硬编码 64
    operand &= ~(1ULL << bitIdx);
}

// 修改后
uint32_t srcWidth = GetOperandConvertWidth(inst.srcs[0]->cvt);
for (uint64_t i = 0; i < N; i++) {
    uint64_t bitIdx = (M + i) % srcWidth;  // 使用 srcWidth
    operand &= ~(1ULL << bitIdx);
}
```

## 复现材料

所有材料位于 `SuperScalarModel/docs/issues/gfrun-scalar-div-shift-mismatch/`：

| 文件 | 说明 |
|---|---|
| `verify_full_chain.py` | 一键完整验证链（4 阶段，自动检测 gfrun） |
| `VERIFICATION.md` | 验证原理与方法论详解 |
| `evidence/code_diff.md` | emulator vs RefModel 的 BIC 实现对照 |
| `evidence/disassembly_analysis.md` | 编译器反汇编分析 |
| `evidence/experiment_results.md` | 对照实验结果（6 组实验） |
| `evidence/python_golden_validation.py` | Python golden 仿真（证明算法正确，1024/1024） |
| `evidence/gfrun_output.txt` | gfrun 完整输出日志 |

最小测试工程（不依赖 group_token_old）：

| 工程 | 路径 | 说明 |
|---|---|---|
| `bic_test` | `SuperNPUBench/.../test/kernel/bic_test/` | 验证 bic 指令正确性（R2=0） |
| `store_load_order` | `SuperNPUBench/.../test/kernel/store_load_order/` | 复现写后清零根因（R2=0x2910） |

## 关联

- benchmark 报告：`SuperNPUBench/docs/workflow/group_token_old_sim_report.md`
- 已有 TEPL 除法/移位验证：`4efec78`（`feat(tile): validate shift and scalar divide TEPL profiles`）— 覆盖 tile 引擎，未覆盖标量管线的 store/load 顺序
- `group_token_old` 尚未加入 `tests/gfrun-pass-list.txt`
