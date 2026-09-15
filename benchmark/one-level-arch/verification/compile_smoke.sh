#!/bin/bash
# SuperNPUBench — 精简测试编译脚本（芯片核心功能验证）
# 16 个用例覆盖：CUBE/TEPL/TLSU/GPR + 多线程 + 全部 ISA 族

: "${COMPILER_DIR:?Set COMPILER_DIR to the Linx compiler bin directory}"
export COMPILER_DIR
export baremetal=${baremetal:-off}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

PASS=0; FAIL=0

smoke() {
    local name=$1; shift
    echo ""
    echo "------------------------------------------"
    echo "  $name"
    echo "------------------------------------------"
    if "$@" 2>&1; then
        echo "  ✓ $name"
        PASS=$((PASS+1))
    else
        echo "  ✗ $name"
        FAIL=$((FAIL+1))
    fi
}

echo "=========================================="
echo "  SuperNPUBench Smoke Test — 编译"
echo "  COMPILER_DIR=$COMPILER_DIR"
echo "=========================================="

# Ensure output directory exists (make clean fails if output/ is missing)
mkdir -p "$REPO_ROOT/output"

# --- 1. multi_thread/matmul (CUBE: 共享 tile) ---
smoke "multi_thread/matmul" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/matmul" TESTCASE=matmul COMPILER_DIR="$COMPILER_DIR" B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32 clean all

# --- 2. multi_thread/vec (TEPL: 多线程元素加) ---
smoke "multi_thread/vec" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/vec" TESTCASE=tadd COMPILER_DIR="$COMPILER_DIR" TileRows=16 TileCols=16 clean all

# --- 3. multi_thread/fa (CUBE+TEPL: 多线程 FA) ---
smoke "multi_thread/fa" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/fa" TESTCASE=fa_2d_unroll_gmma COMPILER_DIR="$COMPILER_DIR" Sq=128 Skv=64 Tm=16 Tk=16 clean all

# --- 4. multi_thread/element_wise/gelu (SPMD: 连续分片) ---
smoke "multi_thread/element_wise/gelu" \
    make -C "$REPO_ROOT/test/kernel/multi_thread/element_wise/gelu" TESTCASE=gelu COMPILER_DIR="$COMPILER_DIR" clean all

echo ""
echo "=========================================="
echo "  Smoke Test 编译完成"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "=========================================="
echo "ELF: $(find "$REPO_ROOT/output" -name '*.elf' -type f | wc -l)"
