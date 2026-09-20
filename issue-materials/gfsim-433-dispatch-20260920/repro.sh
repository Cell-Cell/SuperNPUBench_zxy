#!/bin/bash
# SuperScalarModel #433 附件材料一键复现
# 前置：SuperScalarModel main @ 905ef557（或更新）已构建 bin/gfsim + bin/gfrun
# 用法：SSM=/path/to/SuperScalarModel bash repro.sh
set -u
SSM="${SSM:?set SSM=/path/to/SuperScalarModel}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E="$HERE/elf"

cd "$SSM"   # --conf fourpe 需以模型仓为 CWD 解析 configs/fourpe.conf（参见 #673）

run() { # name elf extra...
  local n=$1 f=$2; shift 2
  echo "==== $n ===="
  timeout 600 ./bin/gfsim -f "$f" "$@" 2>&1 | tail -3
  echo "rc=${PIPESTATUS[0]}"
}

# --- 失败用例（fourpe / SMT4 双配置）---
for cfg in "--conf fourpe" "-s core.threadCount=4 core.vec_core_num=4"; do
  echo "########## gfsim $cfg ##########"
  run moe_dispatch_mt_dyn    "$E/solution_moe_dispatch_moe_dispatch_mt_dyn.elf"    $cfg   # rc=134, C:25108
  run group_token_vec_mt_dyn "$E/solution_group_token_vec_group_token_vec_mt_dyn.elf" $cfg # rc=124 超时
  run mega_moe_sim_mt_dyn    "$E/solution_mega_moe_mega_moe_sim_mt_dyn.elf"        $cfg   # rc=134, C:223712
done

# --- 通过对照 ---
run PASS_combine_dyn    "$E/solution_moe_combine_moe_combine_mt_dyn.elf"        --conf fourpe
run PASS_dispatch_static "$E/solution_moe_dispatch_moe_dispatch_mt.elf"         --conf fourpe
run PASS_combine_static "$E/solution_moe_combine_moe_combine_mt.elf"            --conf fourpe

# --- 静态失败对照 ---
run FAIL_gtvec_static   "$E/solution_group_token_vec_group_token_vec_mt.elf"    --conf fourpe
run FAIL_mega_static    "$E/solution_mega_moe_mega_moe_sim_mt_BS16_H32_HD64.elf" --conf fourpe

# --- gfrun 功能对照（全部 R2=0）---
for n in moe_dispatch_mt_dyn moe_combine_mt_dyn group_token_vec_mt_dyn mega_moe_sim_mt_dyn; do
  case $n in
    moe_dispatch*)    f=$E/solution_moe_dispatch_moe_dispatch_mt_dyn.elf;;
    moe_combine*)     f=$E/solution_moe_combine_moe_combine_mt_dyn.elf;;
    group_token_vec*) f=$E/solution_group_token_vec_group_token_vec_mt_dyn.elf;;
    mega_moe*)        f=$E/solution_mega_moe_mega_moe_sim_mt_dyn.elf;;   # >300s，慢
  esac
  echo "==== gfrun $n ===="
  timeout 900 ./bin/gfrun -s softcore.multiThreadNum=4 -f "$f" 2>&1 | tail -2
done

# --- dispatch 关键 trace 复现（分析命令见 EVIDENCE.md §5）---
# ./bin/gfsim -f $E/solution_moe_dispatch_moe_dispatch_mt_dyn.elf --conf fourpe -t 1 > trace.log
