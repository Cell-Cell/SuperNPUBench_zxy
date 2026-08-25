#!/bin/bash
# ============================================================================
# gfsim 死锁复现脚本（moe_dispatch_v2: RET redirect resolveTarget=0 -> va=0
# fetch -> BHC pa=0 permanent stall）
#
# 基座: LinxISA/SuperScalarModel main @ 045c224（干净，无本地修改）
# 输入: 本目录自带 moe_dispatch_v2.elf（gfrun R2=0 精度通过版）
#
# 用法:
#   bash repro.sh              # 自动 clone + 编译 + 复现（gfsim 死锁 + gfrun 对照）
#   bash repro.sh trace        # 额外应用插桩 patch，打印 va=0 定位链
#
# 环境变量:
#   SSM_SRC   已有的 SuperScalarModel 源码目录（缺省自动 clone 到 ./SuperScalarModel）
#   BUILD_DIR 编译目录（缺省 $SSM_SRC/build）
# ============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ELF="$HERE/moe_dispatch_v2.elf"
SSM_SRC="${SSM_SRC:-$HERE/SuperScalarModel}"
BUILD_DIR="${BUILD_DIR:-$SSM_SRC/build}"
SSM_COMMIT="045c224"   # main, Merge PR #321

[ -f "$ELF" ] || { echo "ELF not found: $ELF"; exit 1; }

# ---------- [1] 获取模型源码 ----------
if [ ! -d "$SSM_SRC" ]; then
    echo "===== [1/4] clone SuperScalarModel (main @ $SSM_COMMIT) ====="
    git clone https://github.com/LinxISA/SuperScalarModel.git "$SSM_SRC" || exit 1
    git -C "$SSM_SRC" checkout "$SSM_COMMIT" || exit 1
else
    echo "===== [1/4] reuse SuperScalarModel at $SSM_SRC ====="
    cur=$(git -C "$SSM_SRC" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)
    echo "  (当前 HEAD=$cur, 本 issue 复现基座为 $SSM_COMMIT; 如行为不一致请 checkout 该基座)"
fi

# ---------- [2] 编译 gfsim / gfrun ----------
echo "===== [2/4] build gfsim & gfrun ====="
mkdir -p "$BUILD_DIR"
cmake -S "$SSM_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null || exit 1
cmake --build "$BUILD_DIR" --target gfsim gfrun -j "$(nproc)" > /dev/null 2>&1 || exit 1
GFSIM="$BUILD_DIR/bin/gfsim"; GFRUN="$BUILD_DIR/bin/gfrun"
[ -x "$GFSIM" ] || GFSIM="$SSM_SRC/bin/gfsim"
[ -x "$GFRUN" ] || GFRUN="$SSM_SRC/bin/gfrun"

# ---------- [3] gfsim: 预期死锁 ----------
echo "===== [3/4] gfsim (预期: Deadlock, exit 134) ====="
timeout 900 "$GFSIM" -f "$ELF" > /tmp/repro_gfsim.log 2>&1; rc=$?
if grep -q "Deadlock detected" /tmp/repro_gfsim.log; then
    echo "  [REPRODUCED] gfsim Deadlock (rc=$rc)"
    grep -m1 "Deadlock detected" /tmp/repro_gfsim.log | sed 's/^/    /'
    grep -A2 "ERROR:Deadlock execution at" /tmp/repro_gfsim.log | sed 's/^/    /'
else
    echo "  [NOT REPRODUCED] rc=$rc (见 /tmp/repro_gfsim.log)"
fi

# ---------- [4] gfrun 对照: 预期 R2=0（证明非算子问题） ----------
echo "===== [4/4] gfrun 对照 (预期: R2 = 0 正常完成) ====="
timeout 300 "$GFRUN" -t 1 -f "$ELF" > /tmp/repro_gfrun.log 2>&1; rc=$?
r2=$(grep -oE "R2 = [0-9a-f]+" /tmp/repro_gfrun.log | tail -1)
if [ "$r2" = "R2 = 0" ]; then
    echo "  [PASS] gfrun $r2 (功能/精度正确, 同一 ELF) — 死锁为 gfsim 模型问题"
else
    echo "  [UNEXPECTED] gfrun rc=$rc $r2 (见 /tmp/repro_gfrun.log)"
fi

# ---------- [可选] 应用插桩 patch, 打印 va=0 定位链 ----------
if [ "${1:-}" = "trace" ]; then
    echo "===== [trace] apply instrumentation patch & rebuild ====="
    git -C "$SSM_SRC" apply "$HERE/instrumentation/bfu_zero_target_trace.patch" || exit 1
    cmake --build "$BUILD_DIR" --target gfsim -j "$(nproc)" > /dev/null 2>&1 || exit 1
    timeout 900 "$GFSIM" -f "$ELF" > /tmp/repro_gfsim_trace.log 2>&1
    echo "  --- 定位链 (BFU 插桩日志) ---"
    grep -E "redir_zero_tgt|fetchq_zero_va|bhc_miss_zero_pa|Deadlock detected" \
        /tmp/repro_gfsim_trace.log | sed 's/^/    /'
    git -C "$SSM_SRC" apply -R "$HERE/instrumentation/bfu_zero_target_trace.patch"
fi
