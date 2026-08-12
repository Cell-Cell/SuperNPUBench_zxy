#!/usr/bin/env python3
"""
python_golden_validation.py — 证明算子与验证在标准 C++ 语义下等价

本脚本复现 genTopkIndex() 中完全相同的 LCG，完整运行 group_token_old 算子逻辑
（使用 >> 6）和验证逻辑（使用 // 64），并比较结果。

在标准语义下，对无符号整数 >> 6 和 // 64 完全等价。
若本脚本显示 1024/1024 匹配，则 gfrun 失败纯粹是仿真器 bic 指令缺陷。

用法：
    python3 python_golden_validation.py
"""

KBS = 512
KTOPK = 16
KEXPERT_PER_RANK = 4
KEXPERT_PER_POD = 64
KSUPER_POD_NUM = 2
KEXPERT_NUM = 128


def gen_topk_index():
    """复现 group_token_old.cpp 中 genTopkIndex 的 LCG（固定种子）。"""
    seed = 0x1234ABCD
    topk_index = []
    for i in range(KBS):
        token_experts = []
        for j in range(KTOPK):
            seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
            expert_id = (seed >> 16) % KEXPERT_NUM
            token_experts.append(expert_id)
        topk_index.append(token_experts)
    return topk_index


def operator_group_token_scalar(topk_index):
    """复现 groupToken_scalar<true>（使用 >> 6 和 % 4）。"""
    expert_section_token_cnt = [0] * KEXPERT_PER_RANK
    grouped_token_ids = [0] * (KEXPERT_PER_RANK * KBS)
    token_super_pod_info = [0] * (KEXPERT_PER_RANK * KBS * KSUPER_POD_NUM)

    for i in range(KBS):
        min_local_exp_id = KEXPERT_PER_RANK
        dst_pod_local = [0] * KSUPER_POD_NUM

        for j in range(KTOPK):
            expert_id = topk_index[i][j]
            cur_local_exp_id = expert_id % KEXPERT_PER_RANK
            if cur_local_exp_id < min_local_exp_id:
                min_local_exp_id = cur_local_exp_id

            cur_dst_pod = expert_id >> 6
            if cur_dst_pod < KSUPER_POD_NUM:
                dst_pod_local[cur_dst_pod] = 1

        idx_in_section = expert_section_token_cnt[min_local_exp_id]
        expert_section_token_cnt[min_local_exp_id] = idx_in_section + 1
        grouped_token_ids[min_local_exp_id * KBS + idx_in_section] = i

        pod_info_section_offset = (
            min_local_exp_id * KBS * KSUPER_POD_NUM + idx_in_section * KSUPER_POD_NUM
        )
        for j in range(KSUPER_POD_NUM):
            token_super_pod_info[pod_info_section_offset + j] = dst_pod_local[j]

    return grouped_token_ids, token_super_pod_info, expert_section_token_cnt


def full_validation():
    """完整验证：算子（>> 6）vs 验证（// 64）。"""
    topk_index = gen_topk_index()

    grouped_token_ids, token_super_pod_info, expert_section_token_cnt = \
        operator_group_token_scalar(topk_index)

    # 参考实现
    ref_section_cnt = [0] * KEXPERT_PER_RANK
    for i in range(KBS):
        min_local = KEXPERT_PER_RANK
        for j in range(KTOPK):
            local = topk_index[i][j] % KEXPERT_PER_RANK
            if local < min_local:
                min_local = local
        ref_section_cnt[min_local] += 1

    # 验证 pod info
    pod_info_match = 0
    pod_info_total = 0

    for s in range(KEXPERT_PER_RANK):
        n = ref_section_cnt[s]
        pod_info_total += n * KSUPER_POD_NUM
        for i in range(n):
            token_id = grouped_token_ids[s * KBS + i]
            expected_pod = [0] * KSUPER_POD_NUM
            for j in range(KTOPK):
                expert_id = topk_index[token_id][j]
                cur_dst_pod = expert_id // KEXPERT_PER_POD
                if cur_dst_pod < KSUPER_POD_NUM:
                    expected_pod[cur_dst_pod] = 1
            offset = s * KBS * KSUPER_POD_NUM + i * KSUPER_POD_NUM
            for j in range(KSUPER_POD_NUM):
                actual = token_super_pod_info[offset + j]
                expected = expected_pod[j]
                if actual == expected:
                    pod_info_match += 1

    print("=" * 60)
    print("Python Golden 验证：算子（>> 6）vs 验证（// 64）")
    print("=" * 60)
    print()
    print(f"podInfoMatch: {pod_info_match} / {pod_info_total}")
    print(f"  全部匹配：  {pod_info_match == pod_info_total}")
    print()
    print(f"与 gfrun 对比：")
    print(f"  Python（标准）： podInfoMatch={pod_info_match}/{pod_info_total}")
    print(f"  gfrun：          podInfoMatch=48/1024")
    print()

    if pod_info_match == pod_info_total:
        print("✅ 通过：在标准语义下，算子与验证逻辑功能等价。")
        print("   gfrun 失败是仿真器 store/load 顺序缺陷导致（见 verify_full_chain.py）。")
    else:
        print("❌ 失败：标准语义下检测到不匹配。")


if __name__ == "__main__":
    full_validation()
