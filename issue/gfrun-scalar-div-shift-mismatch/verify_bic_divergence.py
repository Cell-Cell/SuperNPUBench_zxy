#!/usr/bin/env python3
"""
verify_bic_divergence.py — 直接验证 BIC 指令在 emulator 和 RefModel 中的差异

不依赖任何 benchmark 算子，纯 Python 模拟两种实现逻辑。

emulator (Bit.cpp):  RMaskN(M, N, srcWidth) — 使用 srcWidth 位环形
RefModel (RefModel.cpp): (M + i) % 64       — 硬编码 64 位环形

用法：
    python3 verify_bic_divergence.py
"""

def full_mask(w):
    """FullMask64(w): w 位全 1 掩码"""
    return (1 << w) - 1

def low_mask(n, w):
    """LowMaskN(n, w): 低 n 位为 1，在 w 位内"""
    if n == 0:
        return 0
    if n >= w:
        return full_mask(w)
    return (1 << n) - 1

def rotate_right(x, s, w):
    """RotateRightN(x, s, w): 在 w 位内右旋 s 位"""
    x &= full_mask(w)
    if s == 0:
        return x
    s %= w
    if s == 0:
        return x
    return ((x >> s) | (x << (w - s))) & full_mask(w)

def rmask_n(m, n, w):
    """RMaskN(m, n, w): emulator 的环形掩码"""
    if n == 0:
        return 0
    if n >= w:
        return full_mask(w)
    m %= w
    low = low_mask(n, w)
    return rotate_right(low, (w - m) % w, w)

def bic_emulator(data, m, n, srcWidth=32):
    """emulator (Bit.cpp) 的 CalcInstBIC 实现"""
    fm = full_mask(srcWidth)
    mask = rmask_n(m, n, srcWidth)
    result = (data & fm) & (~mask & fm)
    return result

def bic_refmodel(data, m, n):
    """RefModel (RefModel.cpp) 的 ExecBIC 实现"""
    operand = data
    if n == 0 or n > 64:
        return None  # 返回 false
    for i in range(n):
        bitIdx = (m + i) % 64
        operand &= ~(1 << bitIdx)
    return operand

def test_bic(data, m, n, srcWidth=32, label=""):
    """测试单条 bic 指令在两种实现下的结果"""
    emu = bic_emulator(data, m, n, srcWidth)
    ref = bic_refmodel(data, m, n)
    match = (emu == ref) if ref is not None else False
    status = "✅ 一致" if match else "❌ 不一致"
    print(f"  bic(data=0x{data:08x}, M={m}, N={n}, srcWidth={srcWidth}) {label}")
    print(f"    emulator(32位): 0x{emu:08x} ({emu})")
    if ref is not None:
        print(f"    refmodel(64位): 0x{ref:016x} ({ref})")
    else:
        print(f"    refmodel(64位): 返回 false")
    print(f"    {status}")
    print()
    return match

def main():
    print("=" * 70)
    print("BIC 指令实现差异验证（不依赖任何 benchmark 算子）")
    print("emulator: RMaskN(M, N, srcWidth) — srcWidth 位环形")
    print("RefModel: (M + i) % 64           — 64 位环形")
    print("=" * 70)
    print()

    all_match = True

    # 测试 1: 编译器生成的关键指令 bic t#1, 2, 62 (M=2, N=62)
    # 用于实现 expertId % 4（取低 2 位）
    print("--- 测试 1: bic(data, M=2, N=62) — 编译器用于 % 4 ---")
    for data in [0, 57, 100, 127, 255]:
        match = test_bic(data, 2, 62, 32, f"(data={data})")
        all_match = all_match and match

    # 测试 2: bic t#1, 0, 2 (M=0, N=2) — 清除低 2 位
    # 用于实现 >> 6 的最后一步
    print("--- 测试 2: bic(data, M=0, N=2) — 清除低 2 位 ---")
    for data in [0, 3, 7, 15, 255]:
        match = test_bic(data, 0, 2, 32, f"(data={data})")
        all_match = all_match and match

    # 测试 3: 完整的 >> 6 指令序列模拟
    print("--- 测试 3: 完整 >> 6 序列 (srli 4 + bic 0,2) ---")
    for expertId in [57, 70, 100, 127]:
        expected = expertId >> 6
        # emulator (32位)
        step1_emu = bic_emulator(expertId, 2, 62, 32)  # 第一个 bic（% 4 部分）
        step2_emu = step1_emu >> 4
        step3_emu = bic_emulator(step2_emu, 0, 2, 32)
        # refmodel (64位)
        step1_ref = bic_refmodel(expertId, 2, 62)
        step2_ref = step1_ref >> 4
        step3_ref = bic_refmodel(step2_ref, 0, 2)

        print(f"  expertId={expertId:3d}, 期望 >>6 = {expected}")
        print(f"    emulator: bic(2,62)=0x{step1_emu:08x} → >>4=0x{step2_emu:08x} → bic(0,2)=0x{step3_emu:08x} ({step3_emu})")
        print(f"    refmodel: bic(2,62)=0x{step1_ref:016x} → >>4=0x{step2_ref:016x} → bic(0,2)=0x{step3_ref:016x} ({step3_ref})")
        match = (step3_emu == expected) and (step3_ref == expected)
        print(f"    {'✅ 两者都正确' if match else '❌ 结果不同或错误'}")
        print()
        all_match = all_match and match

    # 测试 4: N > srcWidth 的情况（根因所在）
    print("--- 测试 4: N > srcWidth 的情况（根因）---")
    for m, n in [(2, 62), (0, 33), (5, 40), (0, 64)]:
        data = 0xDEADBEEF
        match = test_bic(data, m, n, 32, f"(N={n} > srcWidth=32)")
        all_match = all_match and match

    # 汇总
    print("=" * 70)
    if all_match:
        print("全部一致 — 未检测到差异（可能需要检查测试用例）")
    else:
        print("❌ 检测到差异！emulator 和 RefModel 的 bic 实现不一致。")
        print("   根因：emulator 使用 srcWidth(32) 位环形，RefModel 硬编码 64 位环形。")
        print("   当 N > srcWidth 时，两者产生完全不同的结果。")
    print("=" * 70)

if __name__ == "__main__":
    main()
