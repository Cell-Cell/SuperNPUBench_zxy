#ifndef GROUP_TOKEN_VEC_MT_HPP
#define GROUP_TOKEN_VEC_MT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// MoE Token Grouping — Multi-thread Vector (Tile) variant
//
// 4-PE SPMD + TLOAD data loading via Vector核 TMA channel.
// Each PE uses get_thread_idx() for stride parallelism.
//
// Phase 1: TLOAD tile per PE + scalar histogram (stride mode)
// Phase 2: TLOAD tile per PE + scalar scatter (stride mode)
// Phase 3a: TLOAD tile per PE + scalar floorFunc (stride mode)
// Phase 3b: scalar counting sort (PE 0 only)
// ============================================================================

constexpr uint32_t kBS            = 512;
constexpr uint32_t kTopK          = 16;
constexpr uint32_t kExpertPerRank = 4;
constexpr uint32_t kRankPerPod    = 16;
constexpr uint32_t kSuperPodNum   = 2;
constexpr uint32_t kExpertPerPod  = kExpertPerRank * kRankPerPod;
constexpr uint32_t kExpertNum     = kExpertPerPod * kSuperPodNum;
constexpr uint32_t kTopKEleNum    = kBS * kTopK;

constexpr uint32_t kThreadsPerBlock = 4;
constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 16;

using TileU32 = Tile<Location::Vec, uint32_t, kTileM, kTileN, BLayout::RowMajor>;
using GmTopkIndex = global_tensor<uint32_t, RowMajor<kBS, kTopK>>;

// ============================================================================
// Phase 1 (Tile + multi-thread): TLOAD tile + scalar histogram
//
// Each PE loads topkIndex tiles via TLOAD (Vector核 TMA channel),
// then does scalar histogram counting on its stride of tokens.
// ============================================================================
static inline void calTokenPerExpertCnt_mt_tile(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *cntLocal,
    uint32_t expertNum,
    uint32_t topkEleNum)
{
    const uint32_t tid = get_thread_idx();

    uint32_t *myCnt = cntLocal + tid * expertNum;
    for (uint32_t i = 0; i < expertNum; i++) {
        myCnt[i] = 0;
    }

    // TLOAD tiles and scalar count
    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    TileU32 dataTile;

    for (uint32_t blk = 0; blk < Mb; ++blk) {
        auto gI = gIter(blk, 0);
        TLOAD(dataTile, gI);

        // Each PE counts its stride of tokens within this tile
        // Tile covers tokens [blk*16, blk*16+15], each token has 16 expert ids
        for (uint32_t row = tid; row < kTileM; row += kThreadsPerBlock) {
            uint32_t tokenId = blk * kTileM + row;
            uint32_t base = tokenId * kTopK;
            for (uint32_t col = 0; col < kTileN; ++col) {
                uint32_t expertId = topkIndex[base + col];
                if (expertId < expertNum) {
                    myCnt[expertId]++;
                }
            }
        }
    }

    // Reduce: each PE writes its assigned expert range
    uint32_t expertsPerPE = expertNum / kThreadsPerBlock;
    for (uint32_t e = 0; e < expertsPerPE; e++) {
        uint32_t globalExpert = tid * expertsPerPE + e;
        uint32_t sum = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            sum += cntLocal[t * expertNum + globalExpert];
        }
        tokenPerExpertCnt[globalExpert] = sum;
    }
}

// ============================================================================
// Phase 2 (Tile + multi-thread): TLOAD tile + scalar scatter
//
// Each PE loads tiles via TLOAD, then does scalar scatter for its stride.
// ============================================================================
static inline void groupToken_mt_tile(
    uint32_t *topkIndex,
    uint32_t *perPegroupedIds,
    uint32_t *perPeSectionCnt,
    uint32_t *perPePodInfo,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank,
    uint32_t expertPerPod,
    uint32_t superPodNum)
{
    const uint32_t tid = get_thread_idx();
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;

    uint32_t *mySectionCnt = perPeSectionCnt + tid * expertPerRank;
    for (uint32_t i = 0; i < expertPerRank; i++) {
        mySectionCnt[i] = 0;
    }
    uint32_t dstPodLocal[kSuperPodNum];

    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    TileU32 dataTile;

    for (uint32_t blk = 0; blk < Mb; ++blk) {
        auto gI = gIter(blk, 0);
        TLOAD(dataTile, gI);

        for (uint32_t row = tid; row < kTileM; row += kThreadsPerBlock) {
            uint32_t tokenId = blk * kTileM + row;
            uint32_t minLocalExpId = expertPerRank;
            for (uint32_t s = 0; s < superPodNum; s++) dstPodLocal[s] = 0;

            uint32_t base = tokenId * topk;
            for (uint32_t col = 0; col < kTileN; ++col) {
                uint32_t expertId = topkIndex[base + col];
                uint32_t curLocalExpId = expertId % expertPerRank;
                if (curLocalExpId < minLocalExpId) {
                    minLocalExpId = curLocalExpId;
                }
                uint32_t curDstPod = expertId / expertPerPod;
                if (curDstPod < superPodNum) {
                    dstPodLocal[curDstPod] = 1;
                }
            }

            uint32_t idxInSection = mySectionCnt[minLocalExpId]++;
            uint32_t peOffset = minLocalExpId * kThreadsPerBlock * kBsPerPE
                              + tid * kBsPerPE + idxInSection;
            perPegroupedIds[peOffset] = tokenId;

            uint32_t podPeOffset = minLocalExpId * kThreadsPerBlock * kBsPerPE * superPodNum
                                 + tid * kBsPerPE * superPodNum
                                 + idxInSection * superPodNum;
            for (uint32_t s = 0; s < superPodNum; s++) {
                perPePodInfo[podPeOffset + s] = dstPodLocal[s];
            }
        }
    }
}

// ============================================================================
// Host-side merge (scalar, single-PE)
// ============================================================================
static inline void mergeGroupTokenResults(
    const uint32_t *perPegroupedIds,
    const uint32_t *perPeSectionCnt,
    const uint32_t *perPePodInfo,
    uint32_t *groupedTokenIds,
    uint32_t *tokenSuperPodInfo,
    uint32_t *expertSectionTokenCnt,
    uint32_t expertPerRank,
    uint32_t superPodNum)
{
    constexpr uint32_t kBsPerPE = kBS / kThreadsPerBlock;

    for (uint32_t s = 0; s < expertPerRank; s++) {
        uint32_t globalIdx = 0;
        for (uint32_t t = 0; t < kThreadsPerBlock; t++) {
            uint32_t peCnt = perPeSectionCnt[t * expertPerRank + s];
            for (uint32_t i = 0; i < peCnt; i++) {
                uint32_t peOffset = s * kThreadsPerBlock * kBsPerPE
                                  + t * kBsPerPE + i;
                groupedTokenIds[s * kBS + globalIdx] = perPegroupedIds[peOffset];

                uint32_t podPeOffset = s * kThreadsPerBlock * kBsPerPE * superPodNum
                                     + t * kBsPerPE * superPodNum
                                     + i * superPodNum;
                for (uint32_t j = 0; j < superPodNum; j++) {
                    tokenSuperPodInfo[s * kBS * superPodNum + globalIdx * superPodNum + j]
                        = perPePodInfo[podPeOffset + j];
                }
                globalIdx++;
            }
        }
        expertSectionTokenCnt[s] = globalIdx;
    }
}

// ============================================================================
// Phase 3 (Tile + multi-thread): TLOAD tile + scalar FloorFunc + sort
// ============================================================================
static inline void sortKernel_mt_tile(
    uint32_t *topkIndex,
    uint32_t *minLocalExpIds,
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t batchSize,
    uint32_t topk,
    uint32_t expertPerRank)
{
    const uint32_t tid = get_thread_idx();

    // Phase 3a: FloorFunc — TLOAD + scalar, stride mode
    using itTopk = global_iterator<GmTopkIndex, TileU32>;
    itTopk gIter(topkIndex);

    constexpr uint32_t Mb = kBS / kTileM;
    TileU32 dataTile;

    for (uint32_t blk = 0; blk < Mb; ++blk) {
        auto gI = gIter(blk, 0);
        TLOAD(dataTile, gI);

        for (uint32_t row = tid; row < kTileM; row += kThreadsPerBlock) {
            uint32_t tokenId = blk * kTileM + row;
            uint32_t minLocalExpId = expertPerRank;
            uint32_t base = tokenId * topk;
            for (uint32_t col = 0; col < kTileN; ++col) {
                uint32_t expertId = topkIndex[base + col];
                uint32_t curLocalExpId = expertId % expertPerRank;
                if (curLocalExpId < minLocalExpId) {
                    minLocalExpId = curLocalExpId;
                }
            }
            minLocalExpIds[tokenId] = minLocalExpId;
        }
    }

    // Phase 3b: Counting sort — only PE 0
    if (tid == 0) {
        uint32_t counts[kExpertPerRank];
        for (uint32_t i = 0; i < expertPerRank; i++) {
            counts[i] = 0;
        }
        for (uint32_t i = 0; i < batchSize; i++) {
            counts[minLocalExpIds[i]]++;
        }
        sectionStarts[0] = 0;
        for (uint32_t i = 0; i < expertPerRank; i++) {
            sectionStarts[i + 1] = sectionStarts[i] + counts[i];
        }
        uint32_t writePos[kExpertPerRank];
        for (uint32_t i = 0; i < expertPerRank; i++) {
            writePos[i] = sectionStarts[i];
        }
        for (uint32_t i = 0; i < batchSize; i++) {
            uint32_t section = minLocalExpIds[i];
            sortedTokenIds[writePos[section]++] = i;
        }
    }
}

// ============================================================================
// Entry point
// ============================================================================
static inline void runGroupTokenVecMT(
    uint32_t *topkIndex,
    uint32_t *tokenPerExpertCnt,
    uint32_t *groupedTokenIds,
    uint32_t *tokenSuperPodInfo,
    uint32_t *expertSectionTokenCnt,
    uint32_t *sortedTokenIds,
    uint32_t *sectionStarts,
    uint32_t *cntLocal,
    uint32_t *perPegroupedIds,
    uint32_t *perPeSectionCnt,
    uint32_t *perPePodInfo,
    uint32_t *minLocalExpIds)
{
    calTokenPerExpertCnt_mt_tile(topkIndex, tokenPerExpertCnt, cntLocal,
                                   kExpertNum, kTopKEleNum);

    groupToken_mt_tile(topkIndex, perPegroupedIds, perPeSectionCnt, perPePodInfo,
                         kBS, kTopK, kExpertPerRank, kExpertPerPod, kSuperPodNum);

    mergeGroupTokenResults(perPegroupedIds, perPeSectionCnt, perPePodInfo,
                            groupedTokenIds, tokenSuperPodInfo, expertSectionTokenCnt,
                            kExpertPerRank, kSuperPodNum);

    sortKernel_mt_tile(topkIndex, minLocalExpIds, sortedTokenIds, sectionStarts,
                         kBS, kTopK, kExpertPerRank);
}

#endif // GROUP_TOKEN_VEC_MT_HPP
