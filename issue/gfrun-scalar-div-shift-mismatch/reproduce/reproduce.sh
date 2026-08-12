#!/bin/bash
# reproduce.sh — gfrun 标量 `/` 与 `>>` 缺陷一键复现脚本
#
# 用法：
#   bash reproduce.sh
#
# 前置条件：
#   - SuperScalarModel 已构建（bin/gfrun 存在）
#   - linx-toolchain-build 已编译（COMPILER_DIR 已设置或可自动检测）
#   - SuperNPUBench 可用（用于 _start.s 和 group_token_old ELF）

set -e

# === 配置 ===
REPO_ROOT="/mnt/workspace/gitCode/cann/Dev-experience/v300"
SIM="$REPO_ROOT/SuperScalarModel"
BENCH="$REPO_ROOT/SuperNPUBench"
COMPILER_DIR="${COMPILER_DIR:-$REPO_ROOT/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin}"
ISSUE_DIR="$(cd "$(dirname "$0")" && pwd)"

export COMPILER_DIR

echo "=== gfrun 标量 \`/\` 与 \`>>\` 缺陷复现 ==="
echo ""

# === 步骤 0：验证环境 ===
echo "[0] 验证环境..."

if [ ! -f "$SIM/bin/gfrun" ]; then
    echo "错误：bin/gfrun 未找到，路径 $SIM/bin/gfrun"
    echo "      请先构建：cd $SIM && python3 build.py build --target gfrun -j8"
    exit 1
fi

if [ ! -f "$COMPILER_DIR/clang++" ]; then
    echo "错误：clang++ 未找到，路径 $COMPILER_DIR/clang++"
    echo "      请先构建 linx-toolchain 或设置 COMPILER_DIR"
    exit 1
fi

echo "    gfrun:    $SIM/bin/gfrun"
echo "    clang++:  $COMPILER_DIR/clang++"
echo ""

# === 步骤 1：复现 group_token_old R2=6 失败 ===
echo "[1] 复现 group_token_old R2=6 失败..."
ELF="$BENCH/benchmark/one-level-arch/output/kernel/group_token_old/elf/group_token_old.elf"

if [ ! -f "$ELF" ]; then
    echo "    ELF 未找到，开始构建..."
    cd "$BENCH/benchmark/one-level-arch/test/kernel/group_token_old"
    make TESTCASE=group_token_old PLAT=linx
fi

echo "    运行 gfrun..."
cd "$SIM"
GFRUN_OUTPUT=$(bin/gfrun -f "$ELF" 2>&1)
R2_VALUE=$(echo "$GFRUN_OUTPUT" | grep -oP 'R2 = \K\d+' || echo "unknown")

echo "    R2 = $R2_VALUE"
if [ "$R2_VALUE" = "6" ]; then
    echo "    ✅ 已复现：R2=6（podInfo 验证失败）"
else
    echo "    ⚠️  R2=$R2_VALUE（预期 6）"
fi
echo ""

# === 步骤 2：运行最小 div/shift 测试 ===
echo "[2] 运行最小标量 div/shift 测试..."
MINIMAL_SRC="$ISSUE_DIR/minimal_div_shift_test.c"
MINIMAL_DIR="/tmp/gfrun_div_shift_test"
mkdir -p "$MINIMAL_DIR"

# 从任意 benchmark 测试 kernel 目录复制 _start.s
START_S="$BENCH/benchmark/one-level-arch/test/kernel/group_token_old/_start.s"
if [ ! -f "$START_S" ]; then
    echo "错误：_start.s 未找到，路径 $START_S"
    exit 1
fi
cp "$START_S" "$MINIMAL_DIR/"
cp "$MINIMAL_SRC" "$MINIMAL_DIR/"

cd "$MINIMAL_DIR"
echo "    编译最小测试用例..."
$COMPILER_DIR/clang++ -c -mlxbc -O2 -std=c++20 -D__linx \
    minimal_div_shift_test.c -o minimal_div_shift_test.o 2>&1
$COMPILER_DIR/clang++ -nostartfiles _start.s minimal_div_shift_test.o \
    -o minimal_div_shift_test.elf 2>&1

echo "    运行 gfrun..."
cd "$SIM"
MINIMAL_OUTPUT=$(bin/gfrun -f "$MINIMAL_DIR/minimal_div_shift_test.elf" 2>&1)
MINIMAL_R2=$(echo "$MINIMAL_OUTPUT" | grep -oP 'R2 = \K\d+' || echo "unknown")

echo "    R2 = $MINIMAL_R2"
if [ "$MINIMAL_R2" = "3" ]; then
    echo "    ✅ 通过：>> 和 / 产生一致的正确结果（R2=3）"
    echo "    这表示缺陷可能比预期更具体。"
elif [ "$MINIMAL_R2" = "0" ]; then
    echo "    ❌ 失败：>> 和 / 彼此匹配但都返回 0（R2=0）"
    echo "    标量 >> 和 / 均有缺陷。"
elif [ "$MINIMAL_R2" -gt 1000 ] 2>/dev/null; then
    echo "    ❌ 失败：>> 和 / 产生不同结果（R2=$MINIMAL_R2）"
    echo "    检测到不匹配 — 确认 gfrun 中标量 \`/\` 与 \`>>\` 分歧。"
    MISMATCH_FLAGS=$((MINIMAL_R2 / 10000))
    B_SHIFT1=$(((MINIMAL_R2 / 100) % 100))
    B_DIV1=$((MINIMAL_R2 % 100))
    echo "    不匹配标志：$MISMATCH_FLAGS（位 1=测试1, 位 2=测试2, ...）"
    echo "    测试1：>>6 = $B_SHIFT1，/64 = $B_DIV1"
else
    echo "    ⚠️  非预期的 R2=$MINIMAL_R2"
fi
echo ""

# === 汇总 ===
echo "=== 汇总 ==="
echo "  group_token_old：R2=$R2_VALUE（预期 6，podInfo 验证失败）"
echo "  最小测试：       R2=$MINIMAL_R2（预期 3 为通过，>1000 为不匹配）"
echo ""
echo "完整 gfrun 输出已保存到："
echo "  $ISSUE_DIR/evidence/gfrun_output.txt（group_token_old）"
echo "  $MINIMAL_DIR/gfrun_minimal_output.txt（最小测试）"

# 保存输出
echo "$GFRUN_OUTPUT" > "$ISSUE_DIR/evidence/gfrun_output.txt" 2>/dev/null || true
echo "$MINIMAL_OUTPUT" > "$MINIMAL_DIR/gfrun_minimal_output.txt" 2>/dev/null || true
