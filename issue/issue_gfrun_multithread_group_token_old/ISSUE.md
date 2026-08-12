# [gfrun] `.uw` operand modifier 误解码为 `.not`，导致多线程算子无限循环

## 问题概述

`gfrun`（功能仿真器）在执行指令 `or zero, t#1.uw, ->x0` 时，将 `.uw`（unsigned word，取低 32 位零扩展）operand modifier **错误解码为 `.not`（按位取反）**。导致 `x0 = ~0x10 = 0xFFFFFFFFFFFFFFEF` 而非 `0x10`，循环上界变为负数（有符号 -17），循环条件永远为真，**程序陷入无限循环**。

该问题在 `multi_thread/group_token_old` 算子上复现：程序编译成功，gfrun 启动正常，但执行到 Phase 1 reduce 循环后无限不终止，2 小时不完成，Phase 2/3 从未开始。

## 影响范围

- `SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/group_token_old/`（4-PE SPMD MoE token grouping）
- 任何编译器生成 `.uw` modifier 的 `or` 指令的算子
- 单线程版 `group_token_old` 不受影响（不生成 `.uw` modifier 的 `or` 指令，正常完成）

## 环境信息

详见 `repro_materials/environment_info.txt`。摘要：

- **SuperScalarModel**：v300_0811，HEAD `69177439`
- **linx-toolchain-build**：TileOP-API `bb625b9`，LLVM `2a4648d08088`，clang 15.0.4
- **宿主**：aarch64 Linux 5.10, GCC 9.4.0

## 复现步骤

### 1. 编译

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/group_token_old
bash compile.all
```

编译成功，生成 ELF（约 4KB）。

### 2. 仿真

```bash
# 无 trace 模式：程序无限循环，2 小时不完成
bin/gfrun -f <elf>

# trace 模式：程序无限循环，30 秒执行 17 万次同一 block
bin/gfrun -f <elf> -t 1
```

### 3. 对比：单线程版正常完成

```bash
bin/gfrun -f <single_thread_group_token_old.elf>
# 输出: "Success to Reach the End of Benchmark! R2 = 2af8"
```

## 问题现象

程序不崩溃，但无限循环。30 秒 trace 采样：

```
170162 次 BPC 0x114dc（同一个 block，Phase 1 reduce 内层循环）
  2048 次 BPC 0x1145e（Phase 1 stride 循环，正常完成）
     1 次 BPC 0x114c8（reduce 外层循环入口，只执行 1 次后进入内层无限循环）
Phase 2/3 从未开始
```

`a6` 寄存器（循环计数器）在 30 秒内从 0 递增到 `0x2B87D`（178813），仍在循环，正常应只循环 16 次。

## 根因分析

### 错误指令

地址 `0x114d6`，反汇编（`llvm-objdump` 正确识别）：

```
114d6: 05803a05    or    zero, t#1.uw, ->x0
```

gfrun trace（错误解码为 `.not`）：

```
M78949  |TPC:0x114d2  I3  |t:0x10     |t#1:0x0  |0x10  |  |addi t#1, 0x10 -> t      (t#1 = 16 ✓)
M78950  |TPC:0x114d6  I4  |x0:0xffff..ef |zero:0x0 |[t#1.not]:0xffff..ef |  |or zero, [t#1.not] -> x0  (✗)
```

| | 预期（`.uw`） | 实际（`.not`） |
|---|---|---|
| 操作 | `0 \| (t#1 & 0xFFFFFFFF)` | `~t#1` |
| t#1 值 | `0x10` | `0x10` |
| x0 结果 | `0x10`（16） | `0xFFFFFFFFFFFFFFEF`（-17） |

### 后果

`x0` 被用作循环上界。循环条件指令 `setc.ltu a6, [x0.sw]`（地址 `0x114fc`）：

- `x0.sw = 0xFFFFFFEF`（32 位有符号 = -17）
- `a6` 从 0 无符号递增
- 有符号比较 `a6 < -17`：无符号值在有符号比较中总小于负数补码，**永远为真**
- 循环永远不终止

### OPConvertType 枚举

`isa/ISACommon/OperandType.h`：

```cpp
enum class OPConvertType {
    OPCVT_DW = -2,
    OPCVT_NOT,    // = -1
    OPCVT_NEG,    // = 0
    // ... 30+ 个其他类型 ...
    OPCVT_U32,    // .uw，在枚举中间
    // ...
};
```

`OPCVT_NOT`（-1）紧接 `OPCVT_DW`（-2），`OPCVT_U32` 在枚举深处。指令解码器 `DecodeInst32_extract_arith()` 提取了 `srcRType`（bits [26:25]）但未正确解析 operand modifier 字段，可能在 modifier 字段解码时将 `.uw` 的编码值映射到了 `OPCVT_NOT`。

## 代码位置

| 文件 | 行 | 说明 |
|------|-----|------|
| `isa/ISACommon/OperandType.h` | 318-360 | `OPConvertType` 枚举定义，`OPCVT_NOT` 与 `OPCVT_U32` 位置 |
| `isa/codec/generatedfiles/decode-inst32.cpp` | 348-370 | `DecodeInst32_extract_arith()` — modifier 解码缺失 |
| `isa/codec/generatedfiles/decode-inst32.cpp` | 930-933 | `OP_OR` 指令解码入口，调用 `extract_arith` |

## 问题解决思路

1. **确认 modifier 编码字段**：`or` 指令（bin `0x05803a05`）中哪个 bit 字段编码 `.uw` modifier
2. **修正 modifier 解码**：在 `extract_arith` 中正确解析该字段并映射到 `OPCVT_U32` 而非 `OPCVT_NOT`
3. **验证**：修复后 `x0 = 0x10`，循环执行 16 次后正常退出，程序继续执行 Phase 2/3

**临时规避方案**：如果无法立即修复解码器，可以在编译器侧避免生成 `.uw` modifier 的 `or` 指令（改用其他方式实现 32 位零扩展，如 `and t#1, 0xFFFFFFFF, ->x0`）。

## 复现材料清单

所有材料在 `repro_materials/` 目录下：

| 文件 | 说明 |
|------|------|
| `kernel_multi_thread_group_token_old_group_token_old.elf` | 测试 ELF（v300_0811 编译） |
| `group_token_old.cpp` | 多线程版测试源码 |
| `group_token_old.hpp` | kernel 头文件 |
| `Makefile` / `compile.all` | 编译文件 |
| `disassembly_full.txt` | 完整反汇编 |
| `disassembly_uw_modifier_area.txt` | `.uw` 指令附近反汇编（0x114c8-0x11510） |
| `trace_uw_as_not_modifier.txt` | trace：`.uw` 被解码为 `.not` 的证据 |
| `trace_block_distribution_30s.txt` | 30 秒 block 分布（170k 次 0x114dc） |
| `trace_x0_register_values.txt` | x0 寄存器值和 a6 递增证据 |
| `single_thread_gfrun_success.txt` | 单线程版正常完成的对比输出 |
| `environment_info.txt` | 环境信息 |
| `OPConvertType_enum.txt` | OPConvertType 枚举定义 |
| `decode_arith_function.txt` | `extract_arith` 解码函数（根因） |
| `source_key_lines.txt` | 源码关键行 |
