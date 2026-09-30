# 算子 ELF 打包分支（5 算子 × 14 用例）

- 来源: SuperNPUBench `benchmark/one-level-arch/output/solution/<op>/elf/`
- 算子版本: PR #198 (`Cell-Cell/SuperNPUBench_zxy` branch `feat/moe-tile-maximize-gfsim`, commit 4cf92ae)
- 工具链: linx_blockisa_llvm_musl (COMPILER_DIR)
- 验证状态: gfrun 14/14 R2=0; gfsim 最新模型复核见 LinxISA/SuperScalarModel issue #880

## 目录

- <op>/            14 个 .elf（按算子分目录）
- disasm/<op>/     全量反汇编（llvm-objdump -d，同名 .diss）
- syms/<op>/       符号表（llvm-nm -n，同名 .syms）

用例清单: group_token_old(2) / group_token_vec(3) / mega_moe(3) /
moe_combine(3) / moe_dispatch(3)。
