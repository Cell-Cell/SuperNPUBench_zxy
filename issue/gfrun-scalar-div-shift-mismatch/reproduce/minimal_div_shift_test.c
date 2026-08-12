/*
 * minimal_div_shift_test.c — gfrun 标量 `/` 与 `>>` 缺陷的最小测试用例
 *
 * Issue: gfrun-scalar-div-shift-mismatch
 *
 * 本程序测试 gfrun 是否对 uint32_t 的 `a >> 6` 和 `a / 64` 求值一致。
 * 在标准 C++ 语义下二者等价（64 = 2^6，无符号右移为逻辑右移）。
 *
 * 返回值（R2 寄存器）：
 *   b_shift * 100 + b_div
 *
 * 预期（标准语义）：
 *   a = 100
 *   b_shift = 100 >> 6 = 1
 *   b_div   = 100 / 64 = 1
 *   R2 = 1 * 100 + 1 = 101
 *
 * 若 gfrun 存在缺陷：
 *   R2 != 101（如两者都错则 R2 = 0，如只有一个错则 R2 = 1 或 100）
 *
 * 构建（需要 linx 工具链）：
 *   export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
 *   $COMPILER_DIR/clang++ -c -mlxbc -O2 -std=c++20 -D__linx \
 *     minimal_div_shift_test.c -o minimal_div_shift_test.o
 *   $COMPILER_DIR/clang++ -nostartfiles _start.s minimal_div_shift_test.o \
 *     -o minimal_div_shift_test.elf
 *
 * 运行：
 *   bin/gfrun -f minimal_div_shift_test.elf
 *
 * 注意：_start.s 可从任意 SuperNPUBench 测试 kernel 目录复制，例如：
 *   cp SuperNPUBench/benchmark/one-level-arch/test/kernel/group_token_old/_start.s .
 */

#include <stdint.h>

int main() {
    /* 测试用例 1：100 / 64 = 1，100 >> 6 = 1 */
    uint32_t a1 = 100;
    uint32_t b_shift1 = a1 >> 6;    /* 期望 1 */
    uint32_t b_div1   = a1 / 64;    /* 期望 1 */

    /* 测试用例 2：127 / 64 = 1，127 >> 6 = 1 */
    uint32_t a2 = 127;
    uint32_t b_shift2 = a2 >> 6;    /* 期望 1 */
    uint32_t b_div2   = a2 / 64;    /* 期望 1 */

    /* 测试用例 3：64 / 64 = 1，64 >> 6 = 1 */
    uint32_t a3 = 64;
    uint32_t b_shift3 = a3 >> 6;    /* 期望 1 */
    uint32_t b_div3   = a3 / 64;    /* 期望 1 */

    /* 测试用例 4：63 / 64 = 0，63 >> 6 = 0 */
    uint32_t a4 = 63;
    uint32_t b_shift4 = a4 >> 6;    /* 期望 0 */
    uint32_t b_div4   = a4 / 64;    /* 期望 0 */

    /* 编码所有结果：
     *   R2 = (b_shift1 == b_div1 ? 0 : 1) * 1000
     *      + (b_shift2 == b_div2 ? 0 : 1) * 100
     *      + (b_shift3 == b_div3 ? 0 : 1) * 10
     *      + (b_shift4 == b_div4 ? 0 : 1) * 1
     *
     * 预期：R2 = 0（所有配对匹配）
     * 任何非零位表示该测试用例的 >> 与 / 不一致。
     *
     * 额外编码实际值用于诊断：
     *   若 R2 == 0：所有配对匹配
     *   若 R2 != 0：至少有一对不匹配
     */
    uint32_t mismatch = 0;
    if (b_shift1 != b_div1) mismatch += 1000;
    if (b_shift2 != b_div2) mismatch += 100;
    if (b_shift3 != b_div3) mismatch += 10;
    if (b_shift4 != b_div4) mismatch += 1;

    /* 若全部匹配，返回所有 shift 结果之和（应为 1+1+1+0 = 3）
     * 以同时验证值正确，而非仅仅彼此相等。
     *
     * R2 = 0：所有配对匹配且和为 0（>> 和 / 都错误地返回 0）
     * R2 = 3：所有配对匹配且值正确（通过）
     * R2 = 1000+：至少有一对不匹配（失败）
     */
    if (mismatch == 0) {
        /* 全部匹配 — 返回 shift 结果之和以验证正确性 */
        return b_shift1 + b_shift2 + b_shift3 + b_shift4;  /* 期望 3 */
    } else {
        /* 不匹配 — 编码失败的测试用例 + 值 */
        /* 格式：mismatch_flag(4位) + b_shift1(2位) + b_div1(2位) */
        return mismatch * 10000 + b_shift1 * 100 + b_div1;
    }
}
