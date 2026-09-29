# gfsim fourpe 两例冻结 —— 可复现材料包

关联: SuperNPUBench PR #198 (算子侧全部改动) + 本模型仓 issue (模型侧缺陷)

## 冻结用例
1. `solution_group_token_vec_group_token_vec_mt_dyn` — 冻结于 retired 99,741（首条 tile TLOAD [4×16] 0x19000）
2. `solution_mega_moe_mega_moe_sim_mt_dyn` — 冻结于 retired 1,657,024（cfgB 首条 decode TLOAD [32×32] E4M3 0x1c000）

## 目录
- `elf/` 两用例 ELF（冻结构建产物；算子侧 gfrun 14/14 R2=0）
- `disasm/` 全量反汇编（`llvm-objdump -d`，工具链 linx_blockisa_llvm_musl）
- `syms/` 符号表（`llvm-nm -n`）
- `logs/`
  - `gfrun_gtv_x4.log` / `gfrun_mm_y2.log`: 功能仿真 R2=0（算子逻辑正确性证明）
  - `gfsim_gtv_x4.log` / `gfsim_mm_y2c.log`: 本轮冻结现场（死锁头行 + retired）
  - `gfsim_gtv_dyn_stag.log`: 早期构建（含 l1d_refill_unmatched 丢弃洪流证据 C:581987-584781）
  - `gfsim_mm_dyn_stag.log`: mega SetACC 断言 (`distBid < distLast`) 现场
- `analyze_preq_stream.py`: SL2 viz trace 分析脚本（prior 流分桶 / INST 令牌 / 地址轮转）

## 复现命令
```bash
# 构建 (SuperNPUBench, COMPILER_DIR=linx-toolchain-build/output/linx_blockisa_llvm_musl/bin)
cd test/solution/{group_token_vec,mega_moe} && bash compile.all
# 功能仿真 (应 R2=0)
gfrun -f <elf> -s softcore.multiThreadNum=4
# 时序仿真 (冻结复现)
gfsim -f <elf> --conf fourpe -s core.bp_mode=1 -s core.deadlock_cycles=20000000
# SL2 可视化 trace (可选, 用于 §证据)
SL2_VIZ_TRACE=/tmp/sl2viz gfsim -f <elf> --conf fourpe -s core.bp_mode=1 \
  -s sl2.viz_start_cycle=0 -s sl2.viz_max_cycles=600000 -s sl2.viz_max_events=80000000
python3 analyze_preq_stream.py /tmp/sl2viz 300000
```

## 模型版本 (两代对照)

- 旧版 (issue 提交时): SuperScalarModel @ `701b5e728`
- 最新 main: `d21419033` (2026-09-29, 含 BRQ flush 修复 47df5876 与
  c9fed164 之后的合并; gtv mt_dyn 在本版 + `--pto-v02 true` 下完整跑完
  Total Cycles = 1,273,311)

## 最新 main 上的复现结论 (2026-09-29)

- gtv mt_dyn: `gfsim -f <elf> --conf fourpe --pto-v02 true -s tlsu.fake_l2_enable=false`
  → 完整跑完 1,273,311 cycles (见 logs/gfsim_gtv_latest_model_passthrough.log);
  不加 `--pto-v02` 则在 l2_agu.cpp:892 MSCATTER cacheline 断言 (模型团队已知)。
- mega mt_dyn (常规/无-finisher 诊断变体): 双双停在 main 内 mtBarrierDyn
  汇合自旋 (4 PE 同 BPC, retired 1,850,796, scb_waw_violation 33 条, 周期
  空转) → finisher 单写者非空转根因 (见 logs/gfsim_mega_diag_latest.log /
  gfsim_mega_reg_latest.log)。诊断变体 ELF:
  elf/solution_mega_moe_mega_moe_sim_mt_dyn_diag_nofinisher.elf
