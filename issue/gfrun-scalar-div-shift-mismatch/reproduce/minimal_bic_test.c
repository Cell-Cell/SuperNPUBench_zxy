/*
 * minimal_bic_test.c — 不依赖任何 benchmark 的最小 bic 指令验证
 *
 * 构建：
 *   export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
 *   $COMPILER_DIR/clang++ -c -mlxbc -O2 -std=c++20 -D__linx \
 *     minimal_bic_test.c -o minimal_bic_test.o
 *   cp /path/to/SuperNPUBench/.../test/common/_start.s .
 *   $COMPILER_DIR/clang++ -nostartfiles _start.s minimal_bic_test.o \
 *     -o minimal_bic_test.elf
 *
 * 运行：
 *   bin/gfrun -f minimal_bic_test.elf
 *
 * 预期 R2 = 0（全部通过）。若 R2 != 0，bic 指令有缺陷。
 */

#include <stdint.h>

int main() {
    static uint32_t fail_count;
    fail_count = 0;

    /* 测试 1: % 4（编译器生成 bic(data, 2, 62)） */
    if (57 % 4 != 1) fail_count++;
    if (70 % 4 != 2) fail_count++;
    if (100 % 4 != 0) fail_count++;
    if (127 % 4 != 3) fail_count++;
    if (0 % 4 != 0) fail_count++;
    if (3 % 4 != 3) fail_count++;
    if (64 % 4 != 0) fail_count++;

    /* 测试 2: / 64（编译器生成 srli + bic 序列） */
    if (57 / 64 != 0) fail_count += 100;
    if (70 / 64 != 1) fail_count += 100;
    if (100 / 64 != 1) fail_count += 100;
    if (127 / 64 != 1) fail_count += 100;
    if (64 / 64 != 1) fail_count += 100;

    /* 测试 3: >> 6（应与 / 64 等价） */
    if ((57u >> 6) != 0) fail_count += 10000;
    if ((70u >> 6) != 1) fail_count += 10000;
    if ((100u >> 6) != 1) fail_count += 10000;
    if ((127u >> 6) != 1) fail_count += 10000;

    return fail_count;
}
