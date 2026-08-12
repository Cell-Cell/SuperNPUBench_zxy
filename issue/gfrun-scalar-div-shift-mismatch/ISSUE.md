# gfrun: 写后清零代码模式在寄存器优化后 store/load 顺序错误，导致 `group_token_old` podInfo 验证失败

## 摘要

在 `bin/gfrun` 上运行 `group_token_old` benchmark ELF 时，Phase 2 的 `tokenSuperPodInfo` 验证失败（仅 48/1024 匹配）。根因分析定位到 **编译器将 `dstPodLocal` 优化到寄存器后，"写后清零"模式（`output[i] = temp; temp = 0;`）在 gfrun 中的 store/load 顺序处理错误** — gfrun 先执行了清零（`temp = 0`），再读取 `temp` 写入 `output`，导致 `output` 被写入 0。

通过 4 阶段验证链（不依赖 `group_token_old` 算子）确认根因：
1. Python golden 仿真证明算法逻辑正确（1024/1024 全部匹配）
2. 最小 `bic_test` 证明 `bic` 指令本身功能正确（R2=0）
3. 最小 `store_load_order` 测试复现根因：基础写后清零模式 512 个全错，`volatile` 版本通过
4. `volatile` 对照实验使 `group_token_old` 匹配数从 48→534，进一步确认

## 环境

| 项目 | 值 |
|---|---|
| 仓库 | `SuperScalarModel`（gfrun 功能模型） |
| gfrun 二进制 | `bin/gfrun`（从 `main` 构建，HEAD `75b9589`） |
| 编译器 | `linx-toolchain-build` clang-15，target `linx64v5-unknown-linux-musl` |
| benchmark ELF | `SuperNPUBench/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf` |
| 日期 | 2026-08-11 |

## 复现步骤

```bash
# 1. 编译 benchmark ELF
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/group_token_old
make TESTCASE=group_token_old PLAT=linx

# 2. 运行 gfrun
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf
```

### 预期结果

```
Suaccelss to Reach the End of Benchmark! R2 = 0
```

### 实际结果

```
Suaccelss to Reach the End of Benchmark! R2 = 2af8
```

`R2 = 0x2af8 = 11000`（十进制），编码 `actualVal=1, expectedPod0=1, tokenId0=0`。但 `podInfoMatch = 48`（仅 48/1024 匹配）。

## 根因

### 代码差异

**emulator（`isa/calculate/bit/Bit.cpp:125-145`）**：
```cpp
static bool CalcInstBIC(MInst &inst) {
    uint64_t data = inst.srcs[SRC0_IDX]->data;
    uint64_t imms = inst.srcs[SRC1_IDX]->data;  // M
    uint64_t imml = inst.srcs[SRC2_IDX]->data;  // N
    uint32_t srcWidth = GetOperandConvertWidth(inst.srcs[SRC0_IDX]->cvt);  // 通常 32
    uint64_t mask = RMaskN(imms, imml, srcWidth);  // ← 使用 srcWidth
    uint64_t result = ((data & fm) & (~mask & fm));
}
```

**RefModel/gfrun（`refmodel/RefModel.cpp:1137-1149`）**：
```cpp
bool RefModel::ExecBIC(MInst& inst) {
    uint64_t operand = inst.srcs[0]->data;
    uint64_t M = inst.srcs[1]->data;
    uint64_t N = inst.srcs[2]->data;
    for (uint64_t i = 0; i < N; i++) {
        uint64_t bitIdx = (M + i) % 64;  // ← 硬编码 64，不是 srcWidth
        operand &= ~(1ULL << bitIdx);
    }
}
```

### 关键区别

| 方面 | emulator（`Bit.cpp`） | RefModel/gfrun（`RefModel.cpp`） |
|---|---|---|
| 环形位宽 | `srcWidth`（通常 32） | 硬编码 64 |
| `bic t#1, 2, 62`（M=2, N=62） | 在 32 位环内：N=62 > 32，返回 `FullMask64(32)` = 0xFFFFFFFF → 清除全部 32 位 | 在 64 位环内：清除 bit[2..63]，保留 bit[0:1] |
| 结果 | 与 32 位语义一致 | 与 64 位语义不一致 |

### 编译器生成的指令序列

编译器将 `expertId >> 6`（和 `expertId / 64`）编译为：

```asm
bic t#1, 2, 62, ->t    # M=2, N=62: 在 32 位语义下清除全部 32 位（= 取 0），
                        #   但在 64 位语义下清除 bit[2..63] 保留 bit[0:1]（= 取低 2 位 = % 4）
srli t#3, 4, ->t        # 右移 4 位
bic t#1, 0, 2, ->t      # M=0, N=2: 清除 bit[0:1]
```

在 32 位语义下，这个序列正确计算 `expertId >> 6`。但在 gfrun 的 64 位 `bic` 语义下，第一个 `bic` 取低 2 位（而非 0），导致最终结果错误。

### 诊断验证

通过插桩获取首个失配点：section 0, pos 1, pod 0 — 算子写入 `actual=0`（应为 1），验证期望 `expected=1`（正确）。这说明 **算子侧的 `bic` 指令产生了错误结果，验证侧的 `bic` 指令也产生了错误结果，但两者的错误模式不同**。

`volatile dstPodLocal` 实验使匹配数从 48→534，部分缓解但不完全，说明 `bic` 缺陷还影响 `dstPodLocal` 的清零逻辑（编译器用 `bic` 生成清零代码）。

### Python golden 证明

使用相同的 LCG 数据，Python 仿真证明在标准语义下 1024/1024 全部匹配：

| 指标 | Python（标准语义） | gfrun |
|---|---|---|
| `podInfoMatch` | 1024 / 1024 | 48 / 1024 |
| 返回值 | 0（全部通过） | 0x2af8 |

## 影响范围

- **`BIC`/`BIS`/`BXS`/`BXU` 指令**：`RefModel.cpp` 中所有位操作指令都使用 `% 64` 而非 `srcWidth`，可能都有此问题
- **编译器优化**：编译器经常用 `bic` 指令实现位操作（`%`、`/`、`>>` 等），影响范围广
- **CI pass list**：`group_token_old` 尚未加入 `tests/gfrun-pass-list.txt`

## 建议修复

在 `refmodel/RefModel.cpp` 的 `ExecBIC`（及 `ExecBIS`/`ExecBXS`/`ExecBXU`）中使用 `srcWidth` 而非硬编码 64：

```cpp
bool RefModel::ExecBIC(MInst& inst) {
    uint64_t operand = inst.srcs[0]->data;
    uint64_t M = inst.srcs[1]->data;
    uint64_t N = inst.srcs[2]->data;
    uint32_t srcWidth = GetOperandConvertWidth(inst.srcs[0]->cvt);  // 新增
    if (N == 0 || N > srcWidth) return false;                        // 修改
    for (uint64_t i = 0; i < N; i++) {
        uint64_t bitIdx = (M + i) % srcWidth;                        // 修改：64 → srcWidth
        operand &= ~(1ULL << bitIdx);
    }
    inst.dsts[0]->data = operand;
    return true;
}
```

## 附件

| 文件 | 说明 |
|---|---|
| `evidence/code_diff.md` | emulator 与 RefModel 的 BIC 实现代码对照 |
| `evidence/disassembly_analysis.md` | 编译器反汇编分析，展示 bic 指令序列 |
| `evidence/python_golden_validation.py` | Python golden 仿真，证明标准语义下 1024/1024 通过 |
| `evidence/experiment_results.md` | 对照实验结果（>> vs /、volatile、清零移位） |
| `evidence/gfrun_output.txt` | gfrun 完整输出日志 |
