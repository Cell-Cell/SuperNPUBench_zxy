/**
 * Group Token Vec — 4-PE SPMD 动态 shape test driver (group_token_vec_mt_dyn)
 *
 * 基于 group_token_vec_mt.cpp 的动态 shape 版驱动:
 *   - 单次运行内顺序执行 2 组运行时 shape:
 *       cfgA: {bs=512, topK=16, epr=4, epp=64, spn=2}  — 与静态版同值, 等价回归
 *       cfgB: {bs=517, topK=16, epr=8, epp=128, spn=2} — bs=517%16=5 触发
 *             尾块 ValidRow 路径 (PE0 vr=4/PE1 vr=1/PE2..3 空转),
 *             expertPerRank=8 覆盖专家域运行时路径
 *   - GM 缓冲按两组 shape 的最大值定长分配, 运行时只使用有效段;
 *   - barrier 相位由 kernel 内 per-PE 调用计数 (sInvCnt) 单调编号 (inv*8+k),
 *     跨调用陈旧 flag 天然失效 —— 不做任何跨 PE 的 flag 复位写; flag 本体
 *     为多副本轮换 strip (header mtBarrier 注, gfsim 跨 PE 可见性规避);
 *   - 验证 PE0 独占, 逐组比对 (5 项检查与静态版一致, 边界运行时化):
 *     cfgA 失败返回静态版同款诊断码 1..5, cfgB 失败返回 +10 偏移码。
 *
 * 运行 (gfrun 功能仿真, 必须四线程):
 *   gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
 */

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "benchmark.h"
#include "solution/group_token_vec/group_token_vec_mt_dyn.hpp"

// ---- max-shape 缓冲尺寸 (cfgA/cfgB 的逐维最大值) ----
constexpr uint32_t kBSMax = 517;
constexpr uint32_t kTopKMax = 16;
constexpr uint32_t kExpertPerRankMax = 8;
constexpr uint32_t kSuperPodNumMax = 2;
constexpr uint32_t kExpertPodMax = kExpertPerRankMax * 16U;         // 128
constexpr uint32_t kExpertNumMax = kExpertPodMax * kSuperPodNumMax; // 256
constexpr uint32_t kTopKEleNumMax = kBSMax * kTopKMax;              // 8272
constexpr uint32_t kBsPerPEMax = (kBSMax + kThreadsPerBlock - 1U) / kThreadsPerBlock;

static void genTopkIndex(uint32_t *topkIndex, uint32_t bs, uint32_t k,
                          uint32_t expertNum)
{
    uint32_t seed = 0x1234ABCDu;
    for (uint32_t i = 0; i < bs; i++) {
        for (uint32_t j = 0; j < k; j++) {
            seed = seed * 1103515245u + 12345u;
            topkIndex[i * k + j] = (seed >> 16) % expertNum;
        }
    }
}

static void refCalTokenPerExpertCnt(const uint32_t *topkIndex,
                                     uint32_t *refExpertCnt,
                                     uint32_t expertNum, uint32_t topkEleNum)
{
    for (uint32_t i = 0; i < expertNum; i++) refExpertCnt[i] = 0;
    for (uint32_t i = 0; i < topkEleNum; i++) {
        if (topkIndex[i] < expertNum) refExpertCnt[topkIndex[i]]++;
    }
}

static void refGroupToken(const uint32_t *topkIndex,
                           uint32_t *refGroupedIds,
                           uint32_t *refSectionCnt,
                           uint32_t bs, uint32_t topk, uint32_t expertPerRank)
{
    for (uint32_t i = 0; i < expertPerRank; i++) refSectionCnt[i] = 0;
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t local = topkIndex[i * topk + j] % expertPerRank;
            if (local < minLocal) minLocal = local;
        }
        uint32_t idx = refSectionCnt[minLocal]++;
        refGroupedIds[minLocal * bs + idx] = i;
    }
}

static void refSortByLocalExpId(const uint32_t *topkIndex,
                                 uint32_t *refSortedIds,
                                 uint32_t *refSectionStarts,
                                 uint32_t bs, uint32_t topk,
                                 uint32_t expertPerRank)
{
    // per-PE 私有 scratch: refPreCompute 全 PE 冗余执行, 共享静态上的
    // ++ 读改写会跨 PE 交错互相抹除 (gfrun R2=1 实证)
    const uint32_t tid = get_thread_idx();
    static uint32_t minLocalExpIds[kThreadsPerBlock][kBSMax];
    static uint32_t counts[kThreadsPerBlock][kExpertPerRankMax];
    static uint32_t writePos[kThreadsPerBlock][kExpertPerRankMax];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t minLocal = expertPerRank;
        for (uint32_t j = 0; j < topk; j++) {
            uint32_t local = topkIndex[i * topk + j] % expertPerRank;
            if (local < minLocal) minLocal = local;
        }
        minLocalExpIds[tid][i] = minLocal;
    }
    for (uint32_t i = 0; i < expertPerRank; i++) counts[tid][i] = 0;
    for (uint32_t i = 0; i < bs; i++) counts[tid][minLocalExpIds[tid][i]]++;
    refSectionStarts[0] = 0;
    for (uint32_t i = 0; i < expertPerRank; i++)
        refSectionStarts[i + 1] = refSectionStarts[i] + counts[tid][i];
    for (uint32_t i = 0; i < expertPerRank; i++)
        writePos[tid][i] = refSectionStarts[i];
    for (uint32_t i = 0; i < bs; i++) {
        uint32_t section = minLocalExpIds[tid][i];
        refSortedIds[writePos[tid][section]++] = i;
    }
}

// GM 缓冲 (verify/main 共同引用, 声明置于使用之前)
static uint32_t topkIndex[kTopKEleNumMax + 2 * 4096];
// topkIndex 对齐指针 = 纯立即数运算 (数组地址为链接期常量), 每次调用点就地
// 计算。原实现为全局指针变量 (落在 .sbss, 与工具链 GlobalTensor 惰性初始化
// 守卫变量相邻) —— gfsim 取证: 该内存槽在 cfgA 期间被相邻 16B 合并 store
// 破坏为 0 (C9 类后端合并缺陷), cfgB 从槽读入 NULL → 全 PE tile TLOAD 基址
// 退化为 tid*0x100 (未映射) → SL2 无响应硬停摆 (确定性冻结于 retired
// 2,717,014)。改为栈/寄存器局部值后无共享内存槽可破坏 (4 PE 各自独立)。
// 单输入缓冲 (两轮共用, ballast 构建实证地址血脉): cfgB 的 genTopkIndex
// 会覆写本缓冲 → cfgA 的验证参考值改由全 PE 在 gen 后、kernel 前冗余同值
// 预计算 (refPreCompute, 纯标量), 延迟验证只做比对 (读 per-cfg 输出缓冲),
// 不再依赖 cfgA 输入数据。bmin 构建取证: 独立 topkIndexB (链接器落点
// 0x28140 全局 bss 区) 上 cfgB p2_min 的全 4 PE Row=4 TLOAD 永无响应
// (SL2 停摆 addr:0x0 ready:0, retired 170,958 冻结 14M+ cycles); 同构建
// cfgA 在 0x19000 区的同款 TLOAD 正常 —— 输入回退单缓冲即回到实证地址。
static inline uint32_t *topkAligned()
{
    return (uint32_t *)(((uint64_t)topkIndex & ~0xFFFu) + 0x1000);
}
// .sbss 压舱槽 (sacrificial ballast): 与原 topkIndexAligned 全局指针同款
// 声明/初始化/文件位置 —— 恢复 .sbss 内 "指针槽@+0, 工具链 GlobalTensor
// 惰性初始化守卫@+8..." 的原始相邻布局。取证结论: 启动期构造器 store 与
// 首个守卫 init store 相邻, 被后端合并为 16B tile store 且丢失一半 (C9 类
// 缺陷): 原布局丢的是本槽 (无引用则良性; 旧版 kernel 读它 → NULL 基址
// 冻结), 移除本槽后丢失落在守卫/默认元数据上 → 运行期视图元数据损坏
// (AGU 非对齐断言)。压舱槽把丢失吸收回无害位置; kernel 一律用
// topkAligned() 立即数运算, 永不读本槽。
__attribute__((used)) static uint32_t *topkIndexAlignedBallast =
    (uint32_t *)(((uint64_t)topkIndex & ~0xFFFu) + 0x1000);

static uint32_t tokenPerExpertCnt[kExpertNumMax];
static uint32_t groupedTokenIds[kExpertPerRankMax * kBSMax];
static uint32_t tokenSuperPodInfo[kExpertPerRankMax * kBSMax * kSuperPodNumMax];
static uint32_t expertSectionTokenCnt[kExpertPerRankMax];
static uint32_t sortedTokenIds[kBSMax];
static uint32_t sectionStarts[kExpertPerRankMax + 1];

static uint32_t cntLocal[kThreadsPerBlock * kExpertNumMax];
static uint32_t perPegroupedIds[kExpertPerRankMax * kThreadsPerBlock * kBsPerPEMax];
static uint32_t perPeSectionCnt[kExpertPerRankMax * kThreadsPerBlock];
static uint32_t perPePodInfo[kExpertPerRankMax * kThreadsPerBlock * kBsPerPEMax * kSuperPodNumMax];
static uint32_t minLocalExpIds[kBSMax];
static uint32_t podScratch[kThreadsPerBlock * kSuperPodNumMax];
static uint32_t sortScratch[2 * kExpertPerRankMax];

// ---- cfgB 输出缓冲 (追加于全部原有静态之后; 输入为单缓冲, 见 topkAligned
// 注)。tokenSuperPodInfo 不参与验证 → 共享。----
static uint32_t tokenPerExpertCntB[kExpertNumMax];
static uint32_t groupedTokenIdsB[kExpertPerRankMax * kBSMax];
static uint32_t expertSectionTokenCntB[kExpertPerRankMax];
static uint32_t sortedTokenIdsB[kBSMax];
static uint32_t sectionStartsB[kExpertPerRankMax + 1];

// ---- per-cfg 验证参考值 ([PE][cfg] 双下标: refPreCompute 全 PE 冗余执行
// —— 消除 PE0 独占预计算时其余 PE 在 hist 屏障的长驱逐自旋 (refpre 取证:
// 该前导流量窗口后首个 TLOAD 冻结); 各 PE 只写自己切片, 零共享零竞态;
// verify (PE0 独占) 只读 [0][c] 切片) ----
static uint32_t refExpertCnt2[kThreadsPerBlock][2][kExpertNumMax];
static uint32_t refGroupedIds2[kThreadsPerBlock][2][kExpertPerRankMax * kBSMax];
static uint32_t refSectionCnt2[kThreadsPerBlock][2][kExpertPerRankMax];
static uint32_t refSortedIds2[kThreadsPerBlock][2][kBSMax];
static uint32_t refSectionStarts2[kThreadsPerBlock][2][kExpertPerRankMax + 1];

// 验证参考值预计算 (全 PE 冗余同值执行, 各写私有 [tid][c] 切片 —— 含
// ++ 累加, 严禁跨 PE 共享目标; 必须在本轮 genTopkIndex 之后、下一轮
// genTopkIndex 覆写输入之前执行 —— 放在 kernel 前即天然满足)。
static void refPreCompute(const int64_t *tiling, uint32_t c)
{
    const uint32_t tid = get_thread_idx();
    const uint32_t bs = static_cast<uint32_t>(tiling[0]);
    const uint32_t topK = static_cast<uint32_t>(tiling[1]);
    const uint32_t expertPerRank = static_cast<uint32_t>(tiling[2]);
    const uint32_t expertPerPod = static_cast<uint32_t>(tiling[3]);
    const uint32_t superPodNum = static_cast<uint32_t>(tiling[4]);
    const uint32_t expertNum = expertPerPod * superPodNum;
    const uint32_t topkEleNum = bs * topK;

    for (uint32_t i = 0; i < expertPerRank * bs; i++) refGroupedIds2[tid][c][i] = 0;
    refCalTokenPerExpertCnt(topkAligned(), refExpertCnt2[tid][c], expertNum, topkEleNum);
    refGroupToken(topkAligned(), refGroupedIds2[tid][c], refSectionCnt2[tid][c], bs, topK, expertPerRank);
    refSortByLocalExpId(topkAligned(), refSortedIds2[tid][c], refSectionStarts2[tid][c], bs, topK, expertPerRank);
}

// 验证 (静态版 group_token_vec_mt.cpp 5 项检查同逻辑, 边界运行时化)。
// 返回 0 = PASS; 1..5 = 静态版同款诊断码。仅比对 (参考值已由
// refPreCompute 预计算); PE0 独占执行。
static int verify(const int64_t *tiling, uint32_t c)
{
    const uint32_t bs = static_cast<uint32_t>(tiling[0]);
    const uint32_t expertPerRank = static_cast<uint32_t>(tiling[2]);
    const uint32_t expertPerPod = static_cast<uint32_t>(tiling[3]);
    const uint32_t superPodNum = static_cast<uint32_t>(tiling[4]);
    const uint32_t expertNum = expertPerPod * superPodNum;

    const uint32_t *refExpertCnt = refExpertCnt2[0][c];
    const uint32_t *refGroupedIds = refGroupedIds2[0][c];
    const uint32_t *refSectionCnt = refSectionCnt2[0][c];
    const uint32_t *refSortedIds = refSortedIds2[0][c];
    const uint32_t *refSectionStarts = refSectionStarts2[0][c];
    // PE0-private verification scratch
    static uint32_t verBuf[kBSMax];
    static uint32_t verRef[kBSMax];

    // 参考值现算 (输入未覆写, 即时可得 —— 不做 pre-kernel 预计算,
    // 见 main 注: 稠密标量前导是首 tile 饥饿触发器)
    refPreCompute(tiling, c);

    int cntMatch = 0;
    for (uint32_t i = 0; i < expertNum; i++) {
        if (((c == 0u) ? tokenPerExpertCnt : tokenPerExpertCntB)[i] == refExpertCnt[i]) cntMatch++;
    }

    int secMatch = 0;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        if (((c == 0u) ? expertSectionTokenCnt : expertSectionTokenCntB)[i] == refSectionCnt[i]) secMatch++;
    }

    int idMatch = 0;
    int idTotal = 0;
    for (uint32_t s = 0; s < expertPerRank; s++) {
        uint32_t n = ((c == 0u) ? expertSectionTokenCnt : expertSectionTokenCntB)[s];
        idTotal += n;
        for (uint32_t i = 0; i < n; i++) {
            verBuf[i] = ((c == 0u) ? groupedTokenIds : groupedTokenIdsB)[s * bs + i];
            verRef[i] = refGroupedIds[s * bs + i];
        }
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = i + 1; j < n; j++) {
                if (verBuf[i] > verBuf[j]) { uint32_t t = verBuf[i]; verBuf[i] = verBuf[j]; verBuf[j] = t; }
                if (verRef[i] > verRef[j]) { uint32_t t = verRef[i]; verRef[i] = verRef[j]; verRef[j] = t; }
            }
        }
        for (uint32_t i = 0; i < n; i++) {
            if (verBuf[i] == verRef[i]) idMatch++;
        }
    }

    int boundMatch = 0;
    for (uint32_t i = 0; i <= expertPerRank; i++) {
        if (((c == 0u) ? sectionStarts : sectionStartsB)[i] == refSectionStarts[i]) boundMatch++;
    }

    int sortMatch = 0;
    for (uint32_t i = 0; i < bs; i++) {
        if (((c == 0u) ? sortedTokenIds : sortedTokenIdsB)[i] == refSortedIds[i]) sortMatch++;
    }

    if (cntMatch != (int)expertNum) return 1;
    if (secMatch != (int)expertPerRank) return 2;
    if (idMatch != idTotal) return 3;
    if (boundMatch != (int)(expertPerRank + 1)) return 4;
    if (sortMatch != (int)bs) return 5;
    return 0;
}

int main()
{
    const uint32_t tid = get_thread_idx();

    const int64_t cfgA[5] = {512, 16, 4, 4 * 16, 2};      // 与静态版同值 (等价回归)
    const int64_t cfgB[5] = {517, 16, 8, 8 * 16, 2};      // 尾块 + 专家域运行时覆盖
    const int64_t *cfgs[2] = {cfgA, cfgB};

    // 工具链视图惰性初始化预热: global_tensor 各实例化的 defaultShape/
    // defaultStride + 守卫变量为函数级静态 (首次构造时初始化, 写 .sbss)。
    // 在多 PE tile 流量开始前 (启动期, 各 PE 冗余同值) 逐一构造两轮, 把
    // 全部惰性 init 写收敛到良性窗口 (第二轮吸收首轮可能的合并写丢失)。
    {
        static uint32_t warmupBuf[64];
        for (int rep = 0; rep < 2; ++rep) {
            global_tensor<uint32_t, RowMajor<-1, -1>> w0(warmupBuf, 8, 8);
            global_tensor<uint32_t, RowMajor<1, 1>> w1(warmupBuf);
            global_tensor<uint32_t, RowMajor<1, 4>> w2(warmupBuf);
            global_tensor<uint32_t, RowMajor<4, 1>> w3(warmupBuf);
            global_tensor<uint32_t, RowMajor<1, 8>> w4(warmupBuf);
            global_tensor<uint32_t, RowMajor<8, 1>> w5(warmupBuf);
            global_tensor<uint32_t, RowMajor<1, 32>> w6(warmupBuf);
            global_tensor<uint32_t, RowMajor<32, 1>> w7(warmupBuf);
            (void)w0; (void)w1; (void)w2; (void)w3;
            (void)w4; (void)w5; (void)w6; (void)w7;
        }
    }

    // 单输入 + per-cfg 输出 + 即时验证 (ballast 实证血脉): 关键差异
    // 修订 —— 不设 pre-kernel 参考值预计算段: refpre/stag 取证显示,
    // refPreCompute 的稠密标量前导 (每 PE ~150K 块) 触发 l1d_refill
    // _unmatched refill 丢弃洪流 (C:581987-584781), 随即首条 tile TLOAD
    // (gen:1) 进入 L2 INST_BUF 永不派发 (SL2 viz 轨迹实证) —— 稠密标量
    // 前导窗口是首 tile 饥饿的确定性触发器。参考值改由 verify 内部现算
    // (输入未覆写, 即时可得)。cfgA 验证后 worker 在 mtBarrier(6) 自旋,
    // 验证结束才进入 cfgB gen; kernel B 末端屏障后 worker 直接退出
    // (零等待), PE0 独占验 cfgB。
    int failA = 0;
    for (int c = 0; c < 2; ++c) {
        // Input init runs redundantly on every PE (deterministic, identical
        // writes — same convention as group_token_vec_mt's genTopkIndex).
        genTopkIndex(topkAligned(),
                     static_cast<uint32_t>(cfgs[c][0]),
                     static_cast<uint32_t>(cfgs[c][1]),
                     static_cast<uint32_t>(cfgs[c][3]) *
                         static_cast<uint32_t>(cfgs[c][4]));

        // L1D 预暖 pass (首 tile 饥饿修复): 顺序触碰 topkIndex 全部行,
        // 使 hist 阶段每 PE 8K 稠密标量读全部命中私有 L1D → L2 prior
        // 流量归零 → 首条 scatter TLOAD 到达安静 L2 而可派发。stag/drain/
        // X1 取证: hist 冷读流在首 TLOAD 时刻持续占满 prior 路径, L2 的
        // prior 绝对优先调度使 tile 派发静默饿死 (INST_BUF 永驻); ballast
        // 靠时序间隙侥幸。预暖自身的 miss 流发生在无 tile 请求在飞的安全
        // 窗口; 4 PE 各自预热私有 L1D (同值冗余, 值弃用)。
        {
            volatile uint32_t warmAcc = 0u;
            const uint32_t warmElems = static_cast<uint32_t>(cfgs[c][0]) *
                                       static_cast<uint32_t>(cfgs[c][1]);
            for (uint32_t i = 0; i < warmElems; ++i) {
                warmAcc += topkAligned()[i];
            }
            (void)warmAcc;
        }

        BENCHSTART;

        runGroupTokenVecMTDyn(topkAligned(),
                              (c == 0) ? tokenPerExpertCnt : tokenPerExpertCntB,
                              (c == 0) ? groupedTokenIds : groupedTokenIdsB,
                              tokenSuperPodInfo,
                              (c == 0) ? expertSectionTokenCnt : expertSectionTokenCntB,
                              (c == 0) ? sortedTokenIds : sortedTokenIdsB,
                              (c == 0) ? sectionStarts : sectionStartsB,
                              cntLocal, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                              minLocalExpIds, podScratch, sortScratch, cfgs[c]);

        BENCHEND;

        // cfgA 即时验证 (PE0 独占; 诊断码记下, 不在循环内早退 ——
        // worker 仍在 mtBarrier(6) 自旋, 早退会造成互等死锁);
        // worker 在验证期间于汇合点自旋 (ballast 实证该窗口可存活)。
        if (c == 0) {
            if (tid == 0U) {
                failA = verify(cfgs[0], 0u);
            }
            mtBarrier(6U);   // 相位 6: cfgA 验证汇合 (kernel=1..5, cfgB=9..13)
        }
    }

    // --- cfgB 验证 (PE0 独占, worker 已退出零等待) ---
    if (tid != 0) {
        return 0;
    }

    if (failA != 0) return failA;          // cfgA: 静态版同款诊断码
    int rcB = verify(cfgs[1], 1u);
    return rcB == 0 ? 0 : 10 + rcB;        // cfgB: +10 偏移诊断码
}
