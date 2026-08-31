// 触发版: 去掉 volatile 后编译即崩（v4i64 BUILD_VECTOR Cannot select）
    // SendAndQuantBuffInit (9731): 统计槽清零 — 每 PE 只清自己 4 个伪核
    {
        // 注: volatile 清零 —— 4 伪核 × 2 expert = 8 个连续 i32 零 store 会被
        //     linxv5 continuous-mem-opt 合并为 32B tile store (v4i64 BUILD_VECTOR,
        //     Cannot select 崩溃), 与 mega_moe_sim.hpp 的 tokRef/tilingData 同款规避
        int32_t* stats = reinterpret_cast<int32_t*>(g_mmWorkspace + statsOffset);
        for (uint32_t lc = 0U; lc < kCoresPerPE; ++lc) {
            const uint32_t core = tid * kCoresPerPE + lc;
            for (uint32_t e = 0; e < tilingData.moeExpertPerRank; ++e) {
                stats[core * tilingData.moeExpertPerRank + e] = 0;
            }
        }
    }
