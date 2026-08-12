# gfrun `group_token_old` podInfo 验证失败 — Issue 材料索引

## 文件索引

```
gfrun-scalar-div-shift-mismatch/
├── ISSUE.md                          # Issue 正文
├── README.md                         # 本文件
├── VERIFICATION.md                   # 验证原理与方法论（4 阶段递进）
├── verify_full_chain.py              # ★ 完整验证链（一键运行，4 阶段）
├── reproduce/
│   ├── minimal_bic_test.c            # 最小 C 测试用例（bic 指令）
│   └── reproduce.sh                  # 一键复现脚本
└── evidence/
    ├── code_diff.md                  # emulator vs RefModel 的 BIC 实现对照
    ├── disassembly_analysis.md       # 编译器反汇编分析
    ├── experiment_results.md         # 对照实验结果（6 组实验）
    ├── python_golden_validation.py   # Python golden 仿真（1024/1024）
    └── gfrun_output.txt              # gfrun 完整输出日志
```

## 快速验证（不依赖 group_token_old）

### 一键完整验证（推荐）

```bash
python3 verify_full_chain.py
```

自动执行 4 个阶段：
1. **bic 代码差异**（纯 Python，1 秒）→ 发现差异
2. **差异影响分析**（纯 Python，1 秒）→ 排除 bic 是根因
3. **bic_test on gfrun**（30 秒）→ 确认 bic 指令正确
4. **store_load_order on gfrun**（30 秒）→ 复现真正根因

### 单独运行各验证

```bash
# 阶段 1-2：纯 Python
python3 verify_full_chain.py  # 自动检测 gfrun，不可用时只跑阶段 1-2

# 阶段 3A：bic 指令正确性
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/bic_test
make TESTCASE=bic_test PLAT=linx
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/bic_test/elf/bic_test.elf
# 预期 R2=0（通过）

# 阶段 3B：写后清零模式根因复现
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/store_load_order
make TESTCASE=store_load_order PLAT=linx
cd SuperScalarModel
bin/gfrun -f ../SuperNPUBench/benchmark/one-level-arch/output/kernel/store_load_order/elf/store_load_order.elf
# 预期 R2=0x2910（失败 — 基础写后清零模式 512 个全错，volatile 通过）
```

## 验证结论

| 阶段 | 验证 | 结果 | 说明 |
|---|---|---|---|
| 1 | bic 代码差异 | ❌ 差异存在 | emulator(32位) vs RefModel(64位) |
| 2 | bic 差异影响 | RefModel 恰好正确 | 不是直接根因 |
| 3A | bic_test on gfrun | ✅ R2=0 | bic 指令功能正确 |
| 3B | store_load_order on gfrun | ❌ R2=0x2910 | 写后清零模式失败 |
| 3B | volatile 对照 | ✅ 通过 | 确认根因是寄存器优化 |

**根因**：写后清零模式（`output[i]=temp; temp=0;`）在编译器将 `temp` 优化到寄存器后，gfrun 的 store/load 顺序处理错误。

**附加发现**：emulator 和 RefModel 的 bic 实现存在代码差异（32位 vs 64位），虽然不是当前问题的直接根因，但建议作为独立 issue 修复。
