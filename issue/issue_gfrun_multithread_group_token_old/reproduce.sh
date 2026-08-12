#!/bin/bash
# Reproduction script for gfrun .uw operand modifier misdecoding bug
#
# Bug: or zero, t#1.uw, ->x0 is decoded as .not instead of .uw
#      x0 = ~0x10 = 0xFFFFFFFFFFFFFFEF instead of 0x10
#      Loop condition a6 < x0.sw always true (unsigned < -17 signed)
#      Program enters infinite loop
#
# Prerequisites:
#   - linx-toolchain-build toolchain built (COMPILER_DIR)
#   - SuperScalarModel gfrun built (bin/gfrun)
#
# Usage:
#   export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
#   export GFRUN=/path/to/SuperScalarModel/bin/gfrun
#   bash reproduce.sh

set -euo pipefail

: "${COMPILER_DIR:?Set COMPILER_DIR to the Linx compiler bin directory}"
: "${GFRUN:?Set GFRUN to the gfrun binary path}"

ELF_DIR="$(cd "$(dirname "$0")" && pwd)/repro_materials"
ELF="$ELF_DIR/kernel_multi_thread_group_token_old_group_token_old.elf"

echo "=========================================="
echo "Test 1: gfrun no trace (expect infinite loop / timeout)"
echo "=========================================="
timeout 10 "$GFRUN" -f "$ELF" 2>&1 || true
echo "[exit code: $?] (124 = timeout = bug reproduced)"
echo ""

echo "=========================================="
echo "Test 2: gfrun trace -t 1 (expect infinite loop, not crash)"
echo "=========================================="
timeout 10 "$GFRUN" -f "$ELF" -t 1 2>&1 | tail -5 || true
echo "[exit code: $?] (124 = timeout = bug reproduced)"
echo ""

echo "=========================================="
echo "Test 3: Verify .uw modifier misdecoding"
echo "  Expected: 'or zero, t#1.uw'"
echo "  Actual:   'or zero, [t#1.not]'"
echo "=========================================="
timeout 30 "$GFRUN" -f "$ELF" -t 1 2>&1 | grep "TPC:0x114d6" | head -3 || true
echo ""

echo "=========================================="
echo "Test 4: Verify x0 register value"
echo "  Expected: x0 = 0x10 (16)"
echo "  Actual:   x0 = 0xffffffffffffffef (-17, = ~0x10)"
echo "=========================================="
timeout 30 "$GFRUN" -f "$ELF" -t 1 2>&1 | grep "TPC:0x114d6" | head -1 || true
echo ""

echo "=========================================="
echo "Test 5: Block distribution (30s)"
echo "  Expected: diverse BPC values (Phase 1/2/3)"
echo "  Actual: >170k times BPC 0x114dc (stuck in one loop)"
echo "=========================================="
timeout 30 "$GFRUN" -f "$ELF" -t 1 2>&1 | grep "^B[0-9]" | grep -oP "BPC 0x[0-9a-f]+" | sort | uniq -c | sort -rn | head -5 || true
echo ""

echo "=========================================="
echo "Bug reproduction complete."
echo ""
echo "If Test 1/2 show timeout and Test 3 shows '.not' instead of '.uw',"
echo "and Test 4 shows x0 = 0xffffffffffffffef,"
echo "the .uw modifier misdecoding bug is confirmed."
echo "=========================================="
