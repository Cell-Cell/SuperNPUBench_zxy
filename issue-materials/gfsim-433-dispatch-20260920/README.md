# #433 附件材料索引（2026-09-20）

SuperNPUBench PR #163（4-PE 动态 shape MoE 算子）在 SuperScalarModel main@905ef557 上的 gfsim 卡死定位材料。

- **EVIDENCE.md** —— 指令级证据链（含对早前评论的勘误）：PC 表、时间线、因果链、分析命令
- **repro.sh** —— 一键复现（`SSM=/path/to/SuperScalarModel bash repro.sh`）
- **elf/**、**diss/** —— 4 个 `_mt_dyn` + 4 个静态对照 ELF 及反汇编（PR #163 @ 208f032 × llvm 4a3e0bdb 构建）
- **logs/** —— gfsim fourpe/SMT4/诊断构建日志、gfrun R2=0 尾部、dispatch 全量 trace（gz）与关键窗口摘录

结果速览：moe_combine_mt_dyn **PASS**；moe_dispatch_mt_dyn / group_token_vec_mt_dyn / mega_moe_sim_mt_dyn **FAIL**（静态对照：dispatch/combine 过，gtvec/mega 挂）。
