// =============================================================================
// moe_dispatch_v2.hpp — MoE Dispatch V2 (PTO one-level tile version)
// =============================================================================
//
// Migrated from: cann-samples/Samples/2_Performance/moe_dispatch_v2_story/
//                src/moe_dispatch_v2_main.asc (2942 lines)
//                class MoeDistributeDispatchV2FullMesh
//
// Self-loopback single-rank model (epWorldSize=1, epRankId=0):
//   1) AllToAllDispatch: pack x[token] into 512B window blocks (data+flag+triple)
//   2) CalCumSum:        per-expert token count, cumsum, write status area
//   3) LocalWindowCopy:  read window, format continuous output
//
// ======================= Migration mapping (Ascend C → PTO) ======================= //
//   DataCopyPad       → TLOAD/TSTORE (tile) or scalar pointer access
//   Duplicate         → TEXPANDSCALAR (tile broadcast)
//   Cast              → TCVT
//   CompareScalar     → TCMP (tile) or scalar comparison
//   GatherMask        → scalar loop with index gathering
//   Select            → scalar conditional
//   And               → scalar bitwise AND
//   Sum/ReduceSum     → TROWSUM (tile) or scalar accumulation
//   CreateVecIndex    → scalar loop generating sequential indices
//   Sub/Abs/Mins      → TSUB/TABS/TMINS or scalar
//   SetFlag/WaitFlag  → no-op (sequential execution)
//   SyncFunc/PipeBarrier → no-op
//   DataCopy          → TSTORE (tile) or scalar write
//   GetBlockIdx       → 0 (single-thread)
//   GetSystemCycle    → 0 (no cycle counter)
//   SetAtomicMax      → scalar max
//   SetCtrlSpr        → no-op
//   TPipe/TBuf/TQue   → local variables / arrays
//   GlobalTensor      → raw pointer | LocalTensor → direct array access
//
// ======================= Simplifications (self-loopback single-thread) ============= //
//   - Multi-core split (SplitToCore/SplitExpertNumToCore) → single core (aivNum_=1)
//   - Cross-core sync (WaitCumSumFlag/GetCumSum polling) → direct scalar calc
//   - Flag polling (WaitDispatch/CheckDataArriveWithFlag) → sequential (data ready)
//   - Ping-pong buffer (xTmpPing/Pong) → single buffer
//   - flagPadOffset_ double-buffer toggle → no-op
//   - Quant path (QuantMode > UNQUANT) → compiled out via if-constexpr
//   - A3/A5 branch → unified A5 token-iteration path (PTO has no __NPU_ARCH__)
//   - FP4 type handling → if (false) documented stub
//   - All optional paths (elastic/mask/performance/BS-mode) → if (false) documented stubs
//
// ======================= Missing function stubs (if (false) blocks) =============== //
//   Each original function is preserved as a compiled-but-never-executed block:
//   InitElasticInfo, CalAndSendCntByRank, AllToAllDispatchA3, SendToMoeExpertByBS,
//   SendBSExpertLoop, CalcBSTokenRange, CalcSendTokenBufNum, RecordRankCommDuration,
//   TokenToExpertInQuant, CalExpertSendNum, SplitExpertNumToCore, BufferInit,
//   WaitDispatch, GatherSumRecvCnt, GetCumSum, WaitCumSumFlag, CheckDataArriveWithFlag,
//   WaitAndFormatOutput (ping-pong), Copy stride 512B packing, flagPadOffset_ toggle,
//   FP4 handling, GenerateGatherMaskTensor, MaskZeroComputeExpert
//
// =============================================================================
#ifndef SUPERNPU_MOE_DISPATCH_V2_PTO_HPP
#define SUPERNPU_MOE_DISPATCH_V2_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace supernpu::tile_isa {

// ===================== Constants (from original) ===================== //
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint8_t  BUFFER_NUM = 2U;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t FLAG_OFFSET = STATE_OFFSET / sizeof(float);
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t STATE_SIZE = 1024UL * 1024UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint8_t  COMBINE_IN_DATA_SIZE = 2;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;
constexpr uint32_t EXPAND_IDX_INFO = 3U;
constexpr uint8_t  EP_WORLD_SIZE_IDX = 1U;
constexpr uint8_t  SHARE_RANK_NUM_IDX = 2U;
constexpr uint8_t  MOE_NUM_IDX = 3U;
constexpr int32_t  BITS_PER_BYTE = 8;
constexpr uint32_t FP4_ELEMS_PER_BYTE = 2;
constexpr uint8_t  QUANT_PADDING_VALUE = 0;
constexpr uint32_t UNQUANT = 0U;
constexpr uint32_t STATIC_QUANT = 1U;
constexpr uint32_t PERTOKEN_DYNAMIC_QUANT = 2U;
constexpr uint32_t PERGROUP_DYNAMIC_QUANT = 3U;
constexpr uint32_t MX_QUANT = 4U;
constexpr uint32_t MX_QUANT_CLIP = 5U;
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t ZERONE_STATE_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr float    ONE = 1.0f;
constexpr uint32_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;

// Mc2Kernel namespace constants
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL;
constexpr uint64_t CUMSUM_CAL_OFFSET = 868UL * 1024UL;
constexpr uint64_t CUMSUM_FLAG_OFFSET = 876UL * 1024UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL;
constexpr uint32_t SYNC_OFFSET = 3U * 1024U;
constexpr int32_t  FULL_MESH_MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U;
constexpr uint32_t AIV_STATE_SIZE = 64U;
constexpr uint32_t SFFVALUE_SIZE = 64U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 16U;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t  VALID_EVENT_FLAG_NUM = 8U;
constexpr uint8_t  LOCAL_COPY_BUFFER_NUM = 2U;
constexpr uint32_t DATA_COPY_MAX_BLOCK_COUNT = 4095U;
constexpr uint8_t  UB_ALIGN_DATA_COUNT = 8U;
constexpr uint32_t MIN_ACTIVE_BS_FOR_BS_MODE = 16U;
constexpr uint32_t DURATION_OFFSET = sizeof(int64_t) / sizeof(int32_t);
constexpr uint32_t FLOAT_OVERFLOW_MODE_CTRL = 60;

constexpr uint32_t ACL_CEIL_DIV(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
constexpr uint32_t Align32(uint32_t inVal) { return (inVal + UB_ALIGN - 1U) & ~(UB_ALIGN - 1U); }

// ===================== Tiling data (mirrors original) ===================== //
struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize, tpWorldSize, epRankId, tpRankId;
    uint32_t expertShardType, sharedExpertNum, sharedExpertRankNum, moeExpertNum;
    uint32_t quantMode, globalBs, bs, k, h, a, aivNum;
    bool isTokenMask, isExpertMask, hasElasticInfo, isPerformance, isQuant, isMc2Context;
    bool hasExpertScales, reserved1;
    uint64_t totalUbSize, totalWinSizeEp;
    uint32_t expertTokenNumsType;
    int32_t zeroComputeExpertNum;
    uint32_t maxSizeForUbBuffer;
    uint64_t scalesRow, scalesCol, scalesCount;
    uint32_t scalesTypeSize;
    uint32_t cumsumTmpMinSize;
};
struct MoeDistributeDispatchV2TilingData {
    MoeDistributeDispatchV2Info moeDistributeDispatchV2Info;
};

// ===================== ScalarGetSFFValue (scalar bit scan) ===================== //
inline int64_t ScalarGetSFFValue(uint64_t mask) {
    if (mask == 0) return -1;
    return static_cast<int64_t>(__builtin_ctzll(mask));
}

// ===================== Main kernel function ===================== //
template <typename XType, typename ExpandXOutType, typename ExpandIdxType,
          int BS, int H, int K, int MoeExpertNum = K,
          int EpWorldSize = 1, int EpRankId = 0,
          int SharedExpertNum = 0, int TileW = 64,
          uint32_t QuantMode = UNQUANT, bool HasExpertScales = true,
          bool IsSmoothScaleExist = false,
          bool HasElasticInfo = false,
          bool IsPerformance = false,
          bool IsTokenMask = false,
          bool IsExpertMask = false,
          int32_t ZeroComputeExpertNum = 0>
void moe_dispatch_v2(
    XType* x,                        // [BS, H]
    ExpandIdxType* expertIds,        // [BS, K]
    float* expertScales,             // [BS, K]
    ExpandXOutType* expandXOut,      // [BS*K, H]
    ExpandIdxType* expandIdxOut,     // [BS*K, 3]
    float* expandScalesOut,          // [BS*K]
    int32_t* sendCountsOut,          // [MoeExpertNum]
    int64_t* expertTokenNumsOut,     // [MoeExpertNum]
    uint8_t* windowBuf,              // window buffer (state 1M + data region)
    uint8_t* workspace,              // workspace buffer
    bool* xActiveMask = nullptr,     // [BS] or [BS, K] (optional)
    int32_t* elasticInfo = nullptr,  // (optional, HasElasticInfo)
    int32_t* performanceInfo = nullptr, // (optional, IsPerformance)
    uint8_t* dynamicScalesOut = nullptr,
    float* scales = nullptr)
{
    using namespace pto;
    static_assert(H % TileW == 0, "H must be divisible by TileW");
    static_assert(TileW % 16 == 0, "TileW must be multiple of 16 for 32B align");
    constexpr int HTiles = H / TileW;
    constexpr int slotCount = BS * K;

    // --- FP4 type handling (A:385-394, only on 3510/A3) ---
    // Original: XInType = conditional_t<IsFP4<XType>, uint8_t, XType>
    // PTO: no FP4 support; default to XType/ExpandXOutType directly
    using XInType = XType;
    using XOutType = ExpandXOutType;
    if (false) {
        // FP4 path: if XType is fp4x2_e2m1_t or fp4x2_e1m2_t, use uint8_t for GM access
        // copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE)
        // copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE)
        // This path is never taken in PTO (no FP4 types defined)
    }

    // ==================== Compile-time derived sizes ==================== //
    constexpr int hXTypeSize = H * (int)sizeof(XInType);
    constexpr int hOutSizeBase = H * (int)sizeof(XOutType);
    (void)hXTypeSize; (void)hOutSizeBase;

    // ==================== Tile type definitions ==================== //
    using tile_h = Tile<Location::Vec, XInType, 1, TileW, BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, XOutType, 1, TileW, BLayout::RowMajor>;
    using tile_fl = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor>;
    using gm_x = global_tensor<XInType, RowMajor<BS, H>>;
    using gm_w = global_tensor<XOutType, RowMajor<1, TileW>>;
    using gm_fl = global_tensor<float, RowMajor<1, 32>>;
    (void)sizeof(tile_h); (void)sizeof(tile_o); (void)sizeof(tile_fl);
    (void)sizeof(gm_x); (void)sizeof(gm_w); (void)sizeof(gm_fl);

    // ==================== SetTilingData (A:662-690) ==================== //
    uint32_t axisBS_ = BS;
    uint32_t axisH_ = H;
    uint32_t axisK_ = K;
    uint32_t epWorldSizeOriginal_ = EpWorldSize;
    int32_t  epRankIdOriginal_ = EpRankId;
    uint32_t epWorldSize_ = EpWorldSize;
    int32_t  epRankId_ = EpRankId;
    uint32_t globalBS_ = BS;
    uint32_t moeExpertNum_ = MoeExpertNum;
    uint32_t sharedExpertNum_ = SharedExpertNum;
    uint32_t sharedExpertRankNum_ = 0;
    uint32_t moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    uint32_t moeExpertNumPerRank_ = (moeExpertRankNum_ > 0) ? (moeExpertNum_ / moeExpertRankNum_) : moeExpertNum_;
    if (moeExpertNumPerRank_ == 0) moeExpertNumPerRank_ = moeExpertNum_;
    uint32_t totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    uint32_t expertTokenNumsType_ = 1;
    int32_t  zeroComputeExpertNum_ = ZeroComputeExpertNum;

    bool hasElasticInfoFlag_ = HasElasticInfo;
    bool hasExpertScalesFlag_ = HasExpertScales;
    bool isPerformanceFlag_ = IsPerformance;
    bool isTokenMaskFlag_ = IsTokenMask;
    bool isExpertMaskFlag_ = IsExpertMask;
    bool isShareExpertRankFlag_ = (epRankId_ < (int32_t)sharedExpertRankNum_);
    bool isScalingDownFlag_ = false;

    // --- SetTilingDataAndCal derived computations (A:692-750) --- //
    uint32_t copyInAxisH_ = axisH_;
    uint32_t copyOutAxisH_ = axisH_;

    // --- InitElasticInfo (A:640-659) ---
    // Original: DataCopyPad(elasticInfo) → read isScalingDownFlag_, override epWorldSize_/moeExpertNum_/epRankId_
    int32_t elasticInfoTensor_[64] = {0};
    if (hasElasticInfoFlag_) {
        uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
        for (uint32_t i = 0; i < elasticInfoSize / sizeof(int32_t); i++) {
            elasticInfoTensor_[i] = elasticInfo[i];
        }
        isScalingDownFlag_ = elasticInfoTensor_[0];
        if (isScalingDownFlag_) {
            epWorldSize_ = elasticInfoTensor_[EP_WORLD_SIZE_IDX];
            sharedExpertRankNum_ = elasticInfoTensor_[SHARE_RANK_NUM_IDX];
            moeExpertNum_ = elasticInfoTensor_[MOE_NUM_IDX];
            epRankId_ = elasticInfoTensor_[ELASTIC_INFO_OFFSET + epRankId_];
        }
    }
    if (false) {
        // --- Full InitElasticInfo with TPipe buffer allocation ---
        // Original A:640-659:
        //   uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
        //   tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
        //   elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
        //   DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
        //   SyncFunc<MTE2_S>();
        //   isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
        //   if (isScalingDownFlag_) { epWorldSize_ = ...; sharedExpertRankNum_ = ...; moeExpertNum_ = ...; epRankId_ = ...; }
    }

    isShareExpertRankFlag_ = (epRankId_ < (int32_t)sharedExpertRankNum_);
    uint32_t rankNumPerSharedExpert_ = 0;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = (moeExpertRankNum_ > 0) ? (moeExpertNum_ / moeExpertRankNum_) : moeExpertNum_;
    if (moeExpertNumPerRank_ == 0) moeExpertNumPerRank_ = moeExpertNum_;
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;

    int32_t expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    uint32_t hAlignSize_ = Align32(axisH_ * sizeof(XInType));
    uint32_t hOutSizeAlign_ = Align32(hOutSize_);
    uint32_t scaleInBytes_ = 0;
    uint32_t scaleOutBytes_ = 0;
    uint32_t scalesCount_ = 0;
    int32_t  tokenQuantAlign_ = static_cast<int32_t>(hOutSizeAlign_ / sizeof(int32_t));
    if constexpr (QuantMode == UNQUANT && IsSmoothScaleExist) {
        hOutSizeAlign_ += scaleInBytes_;
        hAlignSize_ += scaleInBytes_;
        scaleOutBytes_ = scaleInBytes_;
    }
    uint32_t hScaleSizeAlign = Align32(hOutSizeAlign_);
    tokenQuantAlign_ = static_cast<int32_t>(hScaleSizeAlign / sizeof(int32_t));
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    uint32_t expertScaleAlign_ = 0;
    if (hasExpertScalesFlag_) {
        expertScaleAlign_ = tokenQuantAlign_ + EXPAND_IDX_INFO;
    }

    uint32_t blockCntPerToken_ = ACL_CEIL_DIV(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    if (blockCntPerToken_ == 0) blockCntPerToken_ = 1;
    uint32_t hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    uint32_t axisHCommu_ = hCommuSize_ / sizeof(XOutType);
    uint32_t axisMaxBS_ = (epWorldSizeOriginal_ > 0) ? (globalBS_ / epWorldSizeOriginal_) : globalBS_;
    uint64_t expertPerSizeOnWin_ = (uint64_t)axisMaxBS_ * hCommuSize_;
    uint32_t rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
    uint32_t statusCntAlign_ = ACL_CEIL_DIV(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;

    // Multi-core split (A:733-749): single-thread → aivNum_=1
    // Original 3510: aivUsedCumSum_ = totalExpertNum_ / 16; non-3510: / 32
    // PTO: use 3510 formula (original test target is 3510)
    uint32_t aivNum_ = 1;
    uint32_t aivId_ = 0;
    uint32_t aivUsedCumSum_ = totalExpertNum_ / 16;
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_;
    if (aivUsedCumSum_ == 0) aivUsedCumSum_ = 1;
    uint32_t aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (aivUsedAllToAll_ == 0) aivUsedAllToAll_ = 1;
    uint32_t sharedUsedAivNum_ = 0;
    uint32_t moeUsedAivNum_ = aivUsedAllToAll_;
    if (sharedExpertRankNum_ != 0) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) sharedUsedAivNum_ = 1;
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
    if (moeUsedAivNum_ == 0) moeUsedAivNum_ = 1;

    uint32_t expertIdsBufSize_ = ACL_CEIL_DIV(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint64_t totalWinSize_ = (uint64_t)BS * (axisK_ + sharedExpertNum_) * hCommuSize_ * 2;
    uint64_t activeMaskBsCnt_ = axisBS_;
    uint64_t sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    uint64_t flagPadOffset_ = 0;

    // ==================== SetDataStatus / Init (A:753-816) ==================== //
    uint8_t* statusDataSpaceGm_ = windowBuf;
    uint32_t* selfDataStatusGMTensor_ = reinterpret_cast<uint32_t*>(
        statusDataSpaceGm_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN);

    // InitWinState (A:166-196): toggle 0/1, write state fields, increment opCnt
    // Original: returns OLD value of ZERONE_STATE_POS before toggle
    uint32_t dataStateOld = selfDataStatusGMTensor_[ZERONE_STATE_POS];
    selfDataStatusGMTensor_[ZERONE_STATE_POS] = (dataStateOld == 0) ? 1 : 0;
    selfDataStatusGMTensor_[OPOSITION_POS] = 1;
    selfDataStatusGMTensor_[TILING_EPRANKID_POS] = epRankIdOriginal_;
    selfDataStatusGMTensor_[MOE_NUM_POS] = moeExpertNum_;
    selfDataStatusGMTensor_[TILING_WORLDSIZE_POS] = epWorldSizeOriginal_;
    selfDataStatusGMTensor_[GLOBALBS_POS] = globalBS_;
    uint64_t* selfDataStatusGMTensor64_ = reinterpret_cast<uint64_t*>(selfDataStatusGMTensor_);
    selfDataStatusGMTensor64_[OP_CNT_POSUL] = selfDataStatusGMTensor64_[OP_CNT_POSUL] + 1;
    // HCCL DFX (A:187-194): epRankIdOriginal != epRankIdHccl → write HCCL DFX fields
    // Self-loopback: epRankIdHccl == epRankIdOriginal → omitted
    uint32_t dataState_ = dataStateOld; // original returns OLD value

    // winDataSizeOffset_ (A:764-770)
    // Original 3510: Ceil(axisH * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE
    // Original non-3510: Ceil(axisH * COMBINE_IN_DATA_SIZE, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    // PTO: use 3510 formula (original test target is 3510)
    uint64_t hSizeAlignCombine = ACL_CEIL_DIV(axisH_ * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE;
    uint64_t winDataSizeOffset_ = (uint64_t)dataState_ * (totalWinSize_ / BUFFER_NUM) +
                                  (uint64_t)axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;

    // Window addresses (self-loopback: all ranks → same local buffer)
    uint8_t* windowGM_ = windowBuf + STATE_SIZE + winDataSizeOffset_;
    uint8_t* statusSpaceGM_ = windowBuf + (uint64_t)dataState_ * WIN_STATE_OFFSET;
    uint8_t* selfRankWinInBase_ = statusDataSpaceGm_;

    auto GetWindAddrByRankId = [&](int32_t) -> uint8_t* { return windowGM_; };
    auto GetWindStateAddrByRankId = [&](int32_t) -> uint8_t* { return statusSpaceGM_; };

    // GM output pointers
    uint8_t* expandXOutGM_ = reinterpret_cast<uint8_t*>(expandXOut);
    uint8_t* expandScalesOutGM_ = HasExpertScales ? reinterpret_cast<uint8_t*>(expandScalesOut) : nullptr;
    uint8_t* sendCountsOutGM_ = reinterpret_cast<uint8_t*>(sendCountsOut);
    uint8_t* recvCntWorkspaceGM_ = workspace;

    // ==================== ExpIdsCopyAndMaskCal (A:2565-2623) ==================== //
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;

    // Copy expertIds to local array
    int32_t validExpertIds_[slotCount > 0 ? slotCount : 1];
    for (int i = 0; i < expertIdsCnt_; i++) {
        validExpertIds_[i] = expertIds[i];
    }
    int32_t validBsIndex_[BS > 0 ? BS : 1];
    uint32_t gatherMaskTensor_[(slotCount + 31) / 32 > 0 ? (slotCount + 31) / 32 : 1] = {0};

    // --- TokenActiveMaskCal (A:2380-2402) ---
    // Original: DataCopyPad(bool mask) → Cast(bool→half) → Sum → GetValue
    // PTO: scalar count of active tokens
    if (isTokenMaskFlag_) {
        activeMaskBsCnt_ = 0;
        for (uint32_t i = 0; i < axisBS_; i++) {
            if (xActiveMask[i]) activeMaskBsCnt_++;
        }
        sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
    }
    if (false) {
        // --- Full TokenActiveMaskCal with tile ops ---
        // Original A:2380-2402:
        //   DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
        //   SyncFunc<MTE2_V>(); Cast(maskTmpTensor, maskInputInt8Tensor, CAST_NONE, axisBS_);
        //   PipeBarrier<PIPE_V>(); Sum(sumOutTensor, maskTmpTensor, params);
        //   SyncFunc<V_S>(); activeMaskBsCnt_ = sumOutTensor.GetValue(0);
    }

    // --- ExpertActiveMaskCal (A:2479-2495) ---
    // Original: ExpertActiveMaskInit → CalValidBSCnt + CalValidExpIdx
    // PTO: scalar loops for index gathering

    // --- ExpertActiveMaskInit (A:2467-2476) ---
    // Original: tpipe_->InitBuffer(validBsIndexTBuf_, ...); InitBuffer(validExpertIndexBuf_, ...)
    // PTO: stack arrays (already declared above as validBsIndex_)

    if (isExpertMaskFlag_) {
        // --- CalValidBSCnt (A:2404-2436) ---
        // Original: Cast(bool→half) → Sum(axisK) → Mins(1) → CompareScalar(==1)
        //           → CreateVecIndex → GatherMask(validBsIndexTensor_)
        // PTO: scalar nested loop
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < axisBS_; i++) {
            bool valid = false;
            for (uint32_t j = 0; j < axisK_; j++) {
                if (xActiveMask[i * axisK_ + j]) { valid = true; break; }
            }
            if (valid) { validBsIndex_[cnt] = (int32_t)i; cnt++; }
        }
        activeMaskBsCnt_ = cnt;

        // --- CalValidExpIdx (A:2440-2464) ---
        // Original: Cast(bool→half) → CompareScalar(==1) → CreateVecIndex
        //           → GatherMask(validExpertIndexTensor)
        // PTO: scalar count of valid (token, expert) pairs
        sendToMoeExpTokenCnt_ = 0;
        for (int32_t i = 0; i < expertIdsCnt_; i++) {
            if (xActiveMask[i]) sendToMoeExpTokenCnt_++;
        }
    }
    if (false) {
        // --- Full CalValidBSCnt + CalValidExpIdx with tile ops ---
        // Original A:2404-2464:
        //   CalValidBSCnt: Cast(bool→half) → Sum(axisK) → Mins(1) → CompareScalar(==1)
        //                  → CreateVecIndex → GatherMask(validBsIndexTensor_)
        //   CalValidExpIdx: Cast(bool→half) → CompareScalar(==1) → CreateVecIndex
        //                   → GatherMask(validExpertIndexTensor)
        //   Original also builds compacted validExpertIds_ via Select(mask, expertIds, -1)
        //   PTO gap: Select operation for compacting validExpertIds_ is not implemented.
        //   For isExpertMask=true, validExpertIds_ would contain stale entries for masked-out slots.
    }

    if (activeMaskBsCnt_ == 0) {
        for (uint32_t e = 0; e < moeExpertNum_; e++) {
            sendCountsOut[e] = 0;
            expertTokenNumsOut[e] = 0;
        }
        return;
    }

    // --- ZeroComputeExpertMaskCal (A:2549-2562) ---
    // Original: GenerateGatherMaskTensor + MaskZeroComputeExpert
    if (zeroComputeExpertNum_ != 0) {
        // GenerateGatherMaskTensor: if !isExpertMaskFlag_, fill mask with 0xFFFFFFFF
        if (!isExpertMaskFlag_) {
            uint32_t maskCnt = isTokenMaskFlag_ ? (uint32_t)(activeMaskBsCnt_ * axisK_) : (uint32_t)expertIdsCnt_;
            for (uint32_t i = 0; i < maskCnt; i++) {
                gatherMaskTensor_[i / 32] |= (1u << (i % 32));
            }
        }
        // MaskZeroComputeExpert: mark slots with expertId >= moeExpertNum_ as invalid (-1)
        for (int i = 0; i < expertIdsCnt_; i++) {
            if ((uint32_t)validExpertIds_[i] >= moeExpertNum_) {
                validExpertIds_[i] = -1;
            }
        }
    } else {
        // Right-pad with -1 (mirrors DataCopyPadExtParams rightPadding)
        uint32_t expertIdsMask = (uint32_t)(activeMaskBsCnt_ * axisK_);
        uint32_t expertIdsAlignCnt = ACL_CEIL_DIV(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        for (uint32_t i = expertIdsMask; i < expertIdsAlignCnt && i < (uint32_t)expertIdsCnt_; i++) {
            validExpertIds_[i] = -1;
        }
    }

    // --- Duplicate(-1) pre-fill + Select for mask compaction (A:2589-2603) ---
    // Original: Duplicate<int32_t>(validExpertIdsTensor_, -1, expertIdsBufSize_/sizeof(int32_t))
    //           then if mask/zeroCompute: Select(validExpertIdsFloat, gatherMaskInt8, expertIdsFloat, -1, VSEL_TENSOR_SCALAR_MODE)
    //           else: DataCopyPad with rightPadding=-1
    // PTO: fill entire validExpertIds_ with -1 first, then apply mask via scalar Select
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        // Select: for each slot, if gatherMask bit set → keep expertId, else → -1
        for (int i = 0; i < expertIdsCnt_; i++) {
            uint32_t bitIdx = i / 32;
            uint32_t bitPos = i % 32;
            bool maskValid = (bitIdx < (uint32_t)((slotCount + 31) / 32)) &&
                             (gatherMaskTensor_[bitIdx] & (1u << bitPos));
            if (!maskValid) {
                validExpertIds_[i] = -1;
            } else {
                validExpertIds_[i] = expertIds[i]; // keep original expertId
            }
        }
        // Fill remaining buffer positions with -1 (mirrors Duplicate(-1) on full buffer)
        for (int i = expertIdsCnt_; i < (int)(expertIdsBufSize_ / sizeof(int32_t)) && i < slotCount; i++) {
            validExpertIds_[i] = -1;
        }
    } else {
        // Non-mask path: DataCopyPad with rightPadding=-1 already done above
        // Fill remaining buffer positions with -1 (mirrors Duplicate(-1) on full buffer)
        for (int i = expertIdsCnt_; i < (int)(expertIdsBufSize_ / sizeof(int32_t)) && i < slotCount; i++) {
            validExpertIds_[i] = -1;
        }
    }
    if (false) {
        // --- Full MaskZeroComputeExpert with tile ops (A:2499-2536) ---
        // Original: DataCopyPad(expertIds) → SetDeqScale → Cast(int32→half)
        //           → CompareScalar(< moeExpertNum, LT) → And(with gatherMask) → CreateVecIndex
        //           → GatherMask(validExpertIndexTensor)
        // GenerateGatherMaskTensor (A:2540-2546):
        //   Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
        //   Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    }

    // --- expertScales local copy (A:2617-2622) ---
    // Original: DataCopyPad(expertScalesTensor_, expertScalesGMTensor_, expertScalesParams, expertScalesPadParams)
    //           SyncFunc<MTE2_S>();
    // PTO: read directly from expertScales pointer in FillTriple (no UB copy needed)
    float* expertScalesTensor_ = expertScales; // direct GM access (no UB copy in PTO)

    // ==================== AllToAllDispatch (A:1387-1462) ==================== //
    // --- CalcSendTokenBufNum (A:1199-1216) ---
    uint8_t sendTokenBufNum_ = VALID_EVENT_FLAG_NUM;
    if (false) {
        // Original A:1199-1216:
        //   remainUbSize = totalUbSize_ - (endUbAddr - beginUbAddr + UB_ALIGN);
        //   sendTokenBufNum_ = remainUbSize / hCommuSize_;
        //   if (sendTokenBufNum_ > VALID_EVENT_FLAG_NUM) sendTokenBufNum_ = VALID_EVENT_FLAG_NUM;
        //   tpipe_->InitBuffer(outBuf, hCommuSize_ * sendTokenBufNum_);
    }

    // --- SplitToCore (A:965-987) ---
    auto SplitToCore = [&](uint32_t curSendCnt, uint32_t curUseAivNum,
                           uint32_t& startTokenId, uint32_t& endTokenId,
                           uint32_t& sendTokenNum, bool isFront = true) {
        if (curUseAivNum == 0) curUseAivNum = 1;
        sendTokenNum = curSendCnt / curUseAivNum;
        uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
        uint32_t newAivId;
        if (isFront) { newAivId = aivId_; }
        else if (aivId_ >= aivUsedAllToAll_) { newAivId = aivId_ - aivUsedAllToAll_; }
        // Fix (issue #348): aivId_ < moeUsedAivNum_ (MoE AIV, e.g. single-thread aivId_=0)
        // previously underflowed: 0 - moeUsedAivNum_ = 0xFFFFFFFF -> startTokenId=0xFFFFFFFC,
        // endTokenId=0 -> zero-iteration cumsum loop -> sendCountsOut never written,
        // validNum_==0 early return. Clamp to 0 (first worker of the partition).
        else { newAivId = (aivId_ >= moeUsedAivNum_) ? (aivId_ - moeUsedAivNum_) : 0; }
        startTokenId = sendTokenNum * newAivId;
        if (newAivId < remainderTokenNum) { sendTokenNum += 1; startTokenId += newAivId; }
        else { startTokenId += remainderTokenNum; }
        endTokenId = startTokenId + sendTokenNum;
    };

    // --- SplitExpertNumToCore (A:1074-1088) ---
    uint32_t startId_ = 0;
    uint32_t endId_ = moeExpertNum_;
    uint32_t sendNum_ = moeExpertNum_;
    if (false) {
        // Original A:1074-1088: stride = moeUsedAivNum_; sendNum_ = moeExpertNum_ / stride;
        //   startId_ = aivId_; if (aivId_ < remainder) sendNum_ += 1;
    }

    // --- Flag tile (original uses Duplicate<float>(packed, 1.0); PTO: scalar write) ---
    // Note: TSTORE requires min 128B tile; flag area is only 32B, so use scalar write

    // --- CalTokenSendExpertCnt (A:1526-1565) ---
    // Original: Duplicate(dstExpId) → Sub → Abs → Mins(1) → Cast → ReduceSum → GetValue
    // PTO: scalar counting — position of current slot among same-expert slots
    auto CalTokenSendExpertCnt = [&](uint32_t dstExpertId, int32_t calCnt, int32_t& curExpertCnt) {
        int32_t otherCnt = 0;
        for (int32_t j = 0; j < calCnt; j++) {
            int32_t diff = validExpertIds_[j] - (int32_t)dstExpertId;
            int32_t absDiff = (diff < 0) ? -diff : diff;
            if (absDiff >= 1) otherCnt++;
        }
        curExpertCnt = (calCnt >= otherCnt) ? (calCnt - otherCnt) : 0;
    };

    // --- FillTriple (A:820-844) ---
    // Original: xOutTint32[tokenQuantAlign_] = epRankId_; [+1]=tokenIndex; [+2]=k;
    //           if hasExpertScales: xOutTfloat(expertScaleAlign_) = expertScales[tokenIndex*K+k]
    auto FillTriple = [&](uint8_t* wAddr, uint32_t tokenIndex, uint32_t k) {
        int32_t* triplePtr = reinterpret_cast<int32_t*>(wAddr + tokenQuantAlign_ * sizeof(int32_t));
        triplePtr[0] = epRankId_;
        triplePtr[1] = (int32_t)tokenIndex;
        triplePtr[2] = (int32_t)k;
        if ((k < axisK_) && hasExpertScalesFlag_) {
            float* scalePtr = reinterpret_cast<float*>(wAddr + expertScaleAlign_ * sizeof(int32_t));
            scalePtr[0] = expertScalesTensor_[tokenIndex * axisK_ + k]; // original reads from expertScalesTensor_
        }
    };

    // --- TokenToExpert (A:906-961) ---
    // Original: DataCopyPad(x) → FillTriple → Copy(stride {1,1,16,15}) → DataCopy to window
    //   Copy stride packs 480B data + 32B flag gap per 512B block
    //   outTensor_ pre-filled with Duplicate(1.0f) → flag area already 1.0f before Copy
    // PTO: TLOAD token tiles from GM → TSTORE to window; scalar FillTriple; TEXPANDS+TSTORE flag
    auto TokenToExpert = [&](uint8_t* wAddr, uint32_t srcTokenIndex, uint32_t toExpertIndex) {
        FillTriple(wAddr, srcTokenIndex, toExpertIndex);
        if constexpr (QuantMode > UNQUANT) {
            // quantInst_.QuantProcess — compiled out under UNQUANT
        } else {
            // Scalar copy token data GM → window (A3 compliance: all window
            // slot accesses are scalar to avoid tile/scalar mixed 512B block access)
            const XInType* src = x + (uint64_t)srcTokenIndex * H;
            XOutType* dst = reinterpret_cast<XOutType*>(wAddr);
            for (uint32_t e = 0; e < H; e++) {
                dst[e] = static_cast<XOutType>(src[e]);
            }
        }
        // Fix (issue #348): block packing — original Ascend C writes the token
        // record (x + triple + scale, contiguous [0, hOutSizeAlign_)) into the
        // window with Copy stride {1,1,16,15}: block b's 480B data area
        // [b*512, b*512+480) holds record slice [b*480, (b+1)*480); the 32B flag
        // area is preserved (pre-filled 1.0f). The PTO port wrote the record
        // contiguously, so x ([0,512)) overlapped block 0's flag area [480,512)
        // and the flag write clobbered the last 8 floats of x. Repack in place,
        // iterating blocks from the last to the first so already-moved data is
        // never overwritten (overlapping ranges handled like memmove).
        for (int b = (int)blockCntPerToken_ - 1; b >= 1; b--) {
            uint32_t srcOff = (uint32_t)b * SPLIT_BLOCK_DATA_SIZE;
            uint32_t cnt = (hOutSizeAlign_ > srcOff) ? (hOutSizeAlign_ - srcOff) : 0U;
            if (cnt > SPLIT_BLOCK_DATA_SIZE) cnt = SPLIT_BLOCK_DATA_SIZE;
            uint32_t* dst = reinterpret_cast<uint32_t*>(wAddr + (uint64_t)b * SPLIT_BLOCK_SIZE);
            uint32_t* src = reinterpret_cast<uint32_t*>(wAddr + srcOff);
            for (uint32_t i = cnt / sizeof(uint32_t); i > 0; i--) {
                dst[i - 1] = src[i - 1];
            }
        }
        // Store flag (1.0) at each 512B block's flag area (offset 480 within each block)
        // Original: Duplicate(1.0f) pre-fills entire outTensor_, Copy preserves flag gaps
        // Fix (issue #348): flag area is 32B (8 floats), smaller than the minimum
        // TSTORE tile (128B). TSTORE with tile_fl wrote [blk*512+480, blk*512+608),
        // clobbering the NEXT block's data area — triple [512,524) / scale [524,528)
        // of this slot (block 0 flag write) and the first 96B of the following
        // slot (last block flag write). Use scalar writes limited to the 32B
        // flag area (same approach as FillTriple).
        constexpr uint32_t FLAG_AREA_FLOATS = (SPLIT_BLOCK_SIZE - SPLIT_BLOCK_DATA_SIZE) / sizeof(float);
        for (uint32_t blk = 0; blk < blockCntPerToken_; blk++) {
            float* flagPtr = reinterpret_cast<float*>(
                wAddr + blk * SPLIT_BLOCK_SIZE + SPLIT_BLOCK_DATA_SIZE);
            for (uint32_t f = 0; f < FLAG_AREA_FLOATS; f++) {
                flagPtr[f] = 1.0f;
            }
        }
        // flagPadOffset_ = hCommuSize_ - flagPadOffset_ (ping-pong toggle, no-op in PTO)
    };

    // --- TokenToExpertInQuant (A:847-903) ---
    // Original: same as TokenToExpert but with quant/dtype-conversion path
    if (false) {
        // Original A:847-903:
        //   DataCopyPad(xInTensor, xGM_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
        //   if QuantMode > UNQUANT: quantInst_.QuantProcess(tempTensor_, xInTensor, ...)
        //   else (A3): Cast(float) → Cast(round) → FillTriple → Copy(stride {1,1,16,15})
        //              → DataCopy(dstWinGMTensor, outTensor_[slot], axisHCommu_)
        //   flagPadOffset_ = hCommuSize_ - flagPadOffset_  (ping-pong buffer toggle)
        //   Copy stride 512B block packing:
        //     Copy(outTensorInt32, xInTensorInt32, 64, blockCntPerToken_, {1,1,16,15})
        //     Copy(outTensorInt32[64], xInTensorInt32[64], 56, blockCntPerToken_, {1,1,16,15})
        //   64 = 256B per op; 56 = (480-256)/4; 16=dst stride, 15=src stride (512B block packing)
    }

    // --- CalExpertSendNum (A:1048-1070) ---
    int32_t tokenNumToExpert_[MoeExpertNum > 0 ? MoeExpertNum : 1] = {0};
    if (false) {
        // Original A:1048-1070:
        //   for each expert: CompareScalar(expertMask, validExpertIds, dstExpertId, EQ)
        //   → GatherMask(gatherTemp, validExpertIds, expertMask) → count → Duplicate(1.0)
    }

    // --- CalcBSTokenRange (A:1218-1258) ---
    if (false) {
        // Original A:1218-1258:
        //   isMultiCorePerToken = (activeBS < moeUsedAivNum_)
        //   Mode A: coresPerToken = moeUsedAivNum_ / activeBS; each core gets 1 token's expert subset
        //   Mode B: tokensPerCore = activeBS / moeUsedAivNum_; each core gets multiple tokens, all K
    }

    // --- SendBSExpertLoop (A:1260-1319) ---
    if (false) {
        // Original A:1260-1319:
        //   for each expert in [kIterStart, kIterEnd): Copy(stride) → FillTriple → DataCopy to window
        //   batched by sendTokenBufNum_ with event flag rotation
        //   tripleOutOffset = (tokenQuantAlign_ / SPLIT_BLOCK_DATA_COUNT) * SPLIT_BLOCK_COUNT
        //                     + (tokenQuantAlign_ % SPLIT_BLOCK_DATA_COUNT)
    }

    // --- SendToMoeExpertByBS (A:1321-1385) ---
    if (false) {
        // Original A:1321-1385:
        //   CalcBSTokenRange → for each token: DataCopyPad(x) → QuantProcess → SendBSExpertLoop
        //   BS mode: read+quant each token once, dispatch to K experts (O(BS) vs O(BS×K))
    }

    // --- SendToSharedExpert (A:989-1044) ---
    auto SendToSharedExpert = [&]() {
        if (sharedExpertNum_ == 0) return;
        uint32_t curSendCnt = (uint32_t)(activeMaskBsCnt_ * sharedExpertNum_);
        uint32_t startTokenId, endTokenId, sendTokenNum;
        SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
        if (startTokenId >= curSendCnt) return;

        uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;
        for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
            uint32_t sendTokenIndex = virtualTokenIndex % (uint32_t)activeMaskBsCnt_;
            uint32_t toSharedExpertIndex = virtualTokenIndex / (uint32_t)activeMaskBsCnt_;
            int32_t toRankId = (int32_t)(idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_);
            if (isScalingDownFlag_) {
                toRankId = elasticInfoTensor_[ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId];
            }
            uint8_t* wAddr = GetWindAddrByRankId(toRankId) +
                             (uint64_t)(expertPerSizeOnWin_ * (uint64_t)epRankId_ +
                             (uint64_t)sendTokenIndex * hCommuSize_);
            uint32_t srcTokenIndex = sendTokenIndex;
            if (isExpertMaskFlag_) {
                srcTokenIndex = (uint32_t)validBsIndex_[sendTokenIndex];
            }
            if constexpr (QuantMode > UNQUANT || (QuantMode == UNQUANT && !std::is_same_v<ExpandXOutType, XType>)) {
                TokenToExpert(wAddr, srcTokenIndex, axisK_ + toSharedExpertIndex);
            } else {
                TokenToExpert(wAddr, srcTokenIndex, axisK_ + toSharedExpertIndex);
            }
        }
    };

    // --- SendToMoeExpert (A:1092-1197) ---
    // Unified token-iteration path (A5 style for single-thread)
    auto SendToMoeExpert = [&]() {
        uint32_t validTokenNum = isTokenMaskFlag_ ? (uint32_t)(activeMaskBsCnt_ * axisK_) : (uint32_t)expertIdsCnt_;
        for (int32_t index = (int32_t)aivId_; index < (int32_t)validTokenNum; index += (int32_t)moeUsedAivNum_) {
            if (moeUsedAivNum_ == 0) { if (index > 0) break; }
            int32_t tokenId = index / (int32_t)axisK_;
            int32_t topKId = index % (int32_t)axisK_;
            int32_t expertId = validExpertIds_[index];
            if (expertId >= (int32_t)moeExpertNum_ || expertId < 0) continue;
            int32_t toRankId = expertId / (int32_t)moeExpertNumPerRank_ + (int32_t)sharedExpertRankNum_;
            if (isScalingDownFlag_) {
                toRankId = elasticInfoTensor_[ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId];
            }

            // CalTokenSendExpertCnt: count same-expert slots before current
            int32_t dstTokenIdx = 0;
            CalTokenSendExpertCnt((uint32_t)expertId, index, dstTokenIdx);

            // Window address computation
            uint8_t* wAddr;
            if (hasElasticInfoFlag_) {
                // Elastic: epRankId_ * moeExpertNumPerRank_ (no ring offset)
                wAddr = GetWindAddrByRankId(toRankId) +
                        expertPerSizeOnWin_ * ((uint64_t)epRankId_ * moeExpertNumPerRank_ +
                                               (uint64_t)(expertId % (int32_t)moeExpertNumPerRank_)) +
                        (uint64_t)dstTokenIdx * hCommuSize_;
            } else {
                // Non-elastic: (epRankId_ + toRankId) % epWorldSize_ ring offset
                wAddr = GetWindAddrByRankId(toRankId) +
                        expertPerSizeOnWin_ * (((uint64_t)(epRankId_ + toRankId) % epWorldSize_ * moeExpertNumPerRank_) +
                                               (uint64_t)(expertId % (int32_t)moeExpertNumPerRank_)) +
                        (uint64_t)dstTokenIdx * hCommuSize_;
            }

            if constexpr (QuantMode > UNQUANT || (QuantMode == UNQUANT && !std::is_same_v<ExpandXOutType, XType>)) {
                // TokenToExpertInQuant path (compiled out for default UNQUANT + same-type)
                TokenToExpert(wAddr, (uint32_t)tokenId, (uint32_t)topKId);
            } else {
                TokenToExpert(wAddr, (uint32_t)tokenId, (uint32_t)topKId);
            }
        }
    };

    // --- AllToAllDispatchA3 (A:1477-1523) with canUseBSMode logic ---
    if (false) {
        // Original A:1477-1523:
        //   if (aivId_ >= moeUsedAivNum_ && sharedExpertRankNum_ != 0):
        //     SendToSharedExpert (with BUFFER_NUM outBuf)
        //   else:
        //     canUseBSMode = (activeBS > MIN_ACTIVE_BS_FOR_BS_MODE) && (moeUsedAivNum_ > 0)
        //                    && (activeBS % moeUsedAivNum_ == 0 || moeUsedAivNum_ % activeBS == 0)
        //                    && (scalesCount_ <= axisH_) && (!isExpertMaskFlag_);
        //     if canUseBSMode: SendToMoeExpertByBS
        //     else: SendToMoeExpert (expert-iter with expertMaskBuf)
    }

    // Execute AllToAllDispatch (A5 path for PTO)
    if (activeMaskBsCnt_ > 0) {
        if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
            SendToSharedExpert();
        } else {
            SendToMoeExpert();
        }
    }

    // Pre-compute expert counts for CalCumSum
    int32_t expertCounts_[MoeExpertNum > 0 ? MoeExpertNum : 1] = {0};
    for (int i = 0; i < expertIdsCnt_; i++) {
        int32_t eid = validExpertIds_[i];
        if (eid >= 0 && (uint32_t)eid < moeExpertNum_) {
            expertCounts_[eid]++;
        }
    }

    // ==================== CalCumSum (A:1989-2025) ==================== //
    // --- CalAndSendCntByExp (A:1568-1614) ---
    // Write status (flag=1.0 + count) to window status area for each expert
    for (uint32_t e = 0; e < totalExpertNum_; e++) {
        int32_t curExpertCnt = 0;
        if (e < sharedExpertRankNum_) {
            if (activeMaskBsCnt_ > 0 &&
                e % rankNumPerSharedExpert_ == (uint32_t)(epRankId_ % (int32_t)rankNumPerSharedExpert_)) {
                curExpertCnt = (int32_t)activeMaskBsCnt_;
            }
        } else {
            int32_t curMoeExpertId = (int32_t)(e - sharedExpertRankNum_);
            if (sendToMoeExpTokenCnt_ > 0) {
                curExpertCnt = expertCounts_[curMoeExpertId];
            }
        }

        uint32_t dstRankId = e;
        uint32_t offset = STATE_OFFSET * (uint32_t)epRankId_;
        if (e >= sharedExpertRankNum_) {
            dstRankId = ((e - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((e - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        if (isScalingDownFlag_) {
            dstRankId = (uint32_t)elasticInfoTensor_[ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId];
        }

        uint8_t* rankStateAddr = GetWindStateAddrByRankId((int32_t)dstRankId);
        float* statusPtr = reinterpret_cast<float*>(rankStateAddr + offset);
        statusPtr[0] = 1.0f;
        *reinterpret_cast<int32_t*>(&statusPtr[1]) = curExpertCnt;
    }

    // --- CalAndSendCntByRank (A:1616-1674) ---
    // Original: for 3510 + !hasElasticInfoFlag_, this is the active path (not CalAndSendCntByExp)
    //   Per-rank status send: for dstRankId in [newAivId, epWorldSize_, aivUsedCumSum_):
    //     if dstRankId >= sharedExpertRankNum_: for each moe expert: CalTokenSendExpertCnt → SetValue
    //     else: shared expert count
    //     DataCopy(rankGMTensor, statusTensor_[...], cntCopyParams)
    // PTO: for self-loopback (epWorldSize_=1), CalAndSendCntByExp and CalAndSendCntByRank
    //   produce identical status writes. Using CalAndSendCntByExp above.
    if (false) {
        // Original A:1616-1674:
        //   Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
        //   Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_/8, 1, 8);
        //   SyncFunc<V_S>();
        //   for dstRankId in [newAivId, epWorldSize_, aivUsedCumSum_):
        //     if dstRankId >= sharedExpertRankNum_:
        //       for curMoeExpertId in [startExpertId, endExpertId):
        //         CalTokenSendExpertCnt(curMoeExpertId, maskCnt, curExpertCnt);
        //         statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        //     else:
        //       if activeMaskBsCnt_ > 0 && dstRankId % rankNumPerSharedExpert_ == epRankId_ % ...:
        //         curExpertCnt = activeMaskBsCnt_;
        //       statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        //   SyncFunc<S_MTE3>();
        //   for dstRankId in [newAivId, epWorldSize_, aivUsedCumSum_):
        //     DataCopy(rankGMTensor, statusTensor_[...], cntCopyParams or UB_ALIGN_DATA_COUNT);
    }

    // --- BufferInit (A:1677-1722) ---
    if (false) {
        // Original A:1677-1722:
        //   tpipe_->InitBuffer(waitStatusBuf_, ...); InitBuffer(gatherMaskOutBuf_, ...)
        //   InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN); InitBuffer(scalarBuf_, UB_ALIGN * 3)
        //   if isPerformanceFlag_: InitBuffer(performanceInfoBuf_, ...); Duplicate(0)
    }

    // --- SplitToCore for cumsum (single-thread: all experts to core 0) ---
    uint32_t startStatusIndex_, endStatusIndex_, recStatusNumPerCore_;
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);

    // --- WaitDispatch (A:1847-1883) ---
    // Original: poll flag sum == 1.0 * recStatusNumPerCore_ → RunPosRecord → WaitDispatchClearStatus
    //   Inside polling: RecordRankCommDuration (if isPerformanceFlag_)
    //   After polling: SetAtomicMax → DataCopyPad(performanceInfoGM_) → SetAtomicNone
    // PTO: self-loopback, data already written → no polling needed
    {
        // performanceTimeStart = GetSystemCycle() → 0
        uint64_t performanceTimeStart = 0;

        // RecordRankCommDuration called inside polling loop (A:1863-1864)
        // PTO: data already arrived → single iteration
        if (isPerformanceFlag_ && performanceInfo != nullptr) {
            int32_t* perfInfoTensor = performanceInfo;
            int32_t perfFlags[MoeExpertNum * 4 > 0 ? MoeExpertNum * 4 : 1] = {0};
            int32_t duration = 0; // (0 - 0) / CYCLES_PER_US = 0
            for (uint32_t i = 0; i < recStatusNumPerCore_; i++) {
                uint8_t* statusAddr = GetWindStateAddrByRankId((int32_t)epRankIdOriginal_) +
                                      ((startStatusIndex_ + i) * STATE_OFFSET);
                float statusFp32 = *reinterpret_cast<float*>(statusAddr);
                if (statusFp32 > 0.5f && perfFlags[i] == 0) {
                    perfFlags[i] = 1;
                    uint32_t fromLocalRankId = (startStatusIndex_ + i) % epWorldSize_;
                    uint32_t fromRankId = fromLocalRankId;
                    if (isScalingDownFlag_) {
                        fromRankId = (uint32_t)elasticInfoTensor_[ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromLocalRankId];
                    }
                    int32_t savedTime = perfInfoTensor[fromRankId * DURATION_OFFSET];
                    int32_t newValue = (duration > savedTime) ? duration : savedTime;
                    if (newValue != savedTime) {
                        perfInfoTensor[fromRankId * DURATION_OFFSET] = duration;
                    }
                }
            }
        }

        // RunPosRecord(RUNPOS_CALCUMSUM)
        selfDataStatusGMTensor_[OPOSITION_POS] = RUNPOS_CALCUMSUM;

        // After polling: SetAtomicMax → DataCopyPad(performanceInfoGM_) → SetAtomicNone (A:1872-1878)
        // PTO: performanceInfo already written directly above (no SetAtomicMax needed, single-thread)

        // --- WaitDispatchClearStatus (A:1725-1737) ---
        // Original: Duplicate(0, mask={0x101010101010101,0}) → DataCopy
        //   mask pattern: clears every 8th int32 (position 0 = flag) in each status block
        //   count at position 1 is PRESERVED for GetCumSum to read
        // PTO: documented stub — scalar write to window status area causes simulator TMA stall.
        //   For single invocation, flag cleanup is not needed (status area not reused).
        //   For repeated invocations, windowBuf should be zeroed by caller between calls.
    }
    if (false) {
        // Original A:1847-1883: polling loop
        //   while (sumOfFlag != compareTarget):
        //     DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[...], intriParams);
        //     if isPerformanceFlag_: RecordRankCommDuration(performanceInfoTensor_, performanceTimeStart);
        //     ReduceSum(statusSumOutTensor, statusFp32Tensor_, ...);
        //     sumOfFlag = statusSumOutTensor.GetValue(0);
        //   RunPosRecord(RUNPOS_CALCUMSUM);
        //   if isPerformanceFlag_: SetAtomicMax; DataCopyPad(performanceInfoGM_, ...); SetAtomicNone;
        //   WaitDispatchClearStatus();  // DataCopy clean flags
        //   GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
    }

    // --- GatherSumRecvCnt / GetCumSum (A:1741-1844) ---
    // Original: multi-core polling + GatherMask + Sum for cross-core recvCnt aggregation
    // PTO: single-thread → direct scalar cumsum

    int32_t cumSum_[MoeExpertNum * 4 > 0 ? MoeExpertNum * 4 : 1] = {0};
    int32_t curCnt = 0;
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint8_t* statusAddr = GetWindStateAddrByRankId((int32_t)epRankIdOriginal_) +
                              (index * STATE_OFFSET);
        int32_t count = *reinterpret_cast<int32_t*>(statusAddr + sizeof(float));
        curCnt += count;
        cumSum_[index] = curCnt;
    }
    if (false) {
        // Original GatherSumRecvCnt A:1741-1778:
        //   GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask, ...)
        //   Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
        //   Duplicate(sumCoreFP32, sumOfRecvCnt, ...); DataCopy(selfRankWinInGMTensor_[CUMSUM_CAL_OFFSET...], ...)
        // Original GetCumSum A:1782-1844:
        //   while (true): DataCopy(sumLocalTensor, selfRankWinInGMTensor_[...], ...);
        //                 GatherMask → Sum → if cumSumFlag == aivUsedCumSum_: break;
        //   if newAivId == 0: outLocal.SetValue(0, 0);
        //   else: GatherMask → Sum → outLocal.SetValue(0, sum);
    }

    // --- CalRecvAndSetFlag (A:1886-1925) ---
    // Write sendCountsOut + workspace copies
    int32_t* sendCountsGlobal = reinterpret_cast<int32_t*>(sendCountsOutGM_);
    int32_t* workspaceGlobal = reinterpret_cast<int32_t*>(recvCntWorkspaceGM_);
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        sendCountsGlobal[index] = cumSum_[index];
    }
    for (uint32_t aiv = 0; aiv < aivNum_; aiv++) {
        for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
            workspaceGlobal[aiv * rscvStatusNum_ + index] = cumSum_[index];
        }
    }

    // --- SetExpertTokenNums (A:1928-1959) ---
    {
        uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
        int64_t expertTokenNumCumsum = 0;
        for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
            int64_t expertTokenNum = 0;
            for (uint32_t rank = 0; rank < epWorldSize_; rank++) {
                uint8_t* statusAddr = GetWindStateAddrByRankId((int32_t)rank) +
                                      (localExpertIdx * epWorldSize_ + rank) * STATE_OFFSET;
                int32_t cnt = *reinterpret_cast<int32_t*>(statusAddr + sizeof(float));
                expertTokenNum += cnt;
            }
            expertTokenNumCumsum += expertTokenNum;
            if (expertTokenNumsType_ == 0) {
                expertTokenNumsOut[localExpertIdx] = expertTokenNumCumsum;
            } else {
                expertTokenNumsOut[localExpertIdx] = expertTokenNum;
            }
        }
    }

    // --- WaitCumSumFlag (A:2028-2054) ---
    // Original: poll cumsum flag == aivUsedCumSum_ * UB_ALIGN_DATA_COUNT → clean flag
    // PTO: write + immediately clear (single-thread)
    {
        float* cumsumFlagPtr = reinterpret_cast<float*>(selfRankWinInBase_ + CUMSUM_FLAG_OFFSET);
        for (uint32_t i = 0; i < aivUsedCumSum_ * UB_ALIGN_DATA_COUNT; i++) {
            cumsumFlagPtr[i] = 1.0f;
        }
        selfDataStatusGMTensor_[OPOSITION_POS] = RUNPOS_CUMSUMFLAG;
        for (uint32_t i = 0; i < aivUsedCumSum_ * UB_ALIGN_DATA_COUNT; i++) {
            cumsumFlagPtr[i] = 0.0f;
        }
    }
    if (false) {
        // Original WaitCumSumFlag A:2028-2054: polling loop
        //   while (true): DataCopy(statusFp32Tensor_, selfRankWinInGMTensor_[cumSumFlagOffset], ...);
        //                 Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        //                 cumSumFlag = statusSumOutTensor.GetValue(0);
        //                 if (cumSumFlag == targetFlag): break;
        //   RunPosRecord(RUNPOS_CUMSUMFLAG);
        //   Duplicate(0.0) → DataCopy clean flag
    }

    // --- RecordRankCommDuration (A:1961-1986) ---
    // Already called inside WaitDispatch above (A:1863-1864 calls it during polling)
    // After-polling SetAtomicMax + DataCopyPad(performanceInfoGM_) also done in WaitDispatch
    // No additional performance recording needed here

    // ==================== LocalWindowCopy (A:2338-2377) ==================== //
    // --- SetValidExpertInfo (A:2057-2105) ---
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    if (sendNum_ == 0) return;

    int32_t* sendCntTensor_ = reinterpret_cast<int32_t*>(recvCntWorkspaceGM_);
    for (uint32_t i = 0; i < rscvStatusNum_; i++) {
        sendCntTensor_[i] = cumSum_[i];
    }

    uint32_t expertMap_[MoeExpertNum * 4 > 0 ? MoeExpertNum * 4 : 1];
    uint32_t expertFinishNum_[MoeExpertNum * 4 > 0 ? MoeExpertNum * 4 : 1] = {0};
    uint32_t expertLeftNum_[MoeExpertNum * 4 > 0 ? MoeExpertNum * 4 : 1] = {0};
    uint32_t validNum_ = 0;

    for (uint32_t index = startId_; index < endId_; index++) {
        expertMap_[validNum_] = index;
        if (index == 0) {
            expertLeftNum_[validNum_] = (uint32_t)sendCntTensor_[index];
        } else {
            expertLeftNum_[validNum_] = (uint32_t)sendCntTensor_[index] - (uint32_t)sendCntTensor_[index - 1];
        }
        if (expertLeftNum_[validNum_] != 0) {
            validNum_++;
        }
    }
    if (validNum_ == 0) return;

    // --- CheckDataArriveWithFlag (A:2109-2127) ---
    // Original: DataCopy flag → ReduceSum → check == 1.0 * flagNum
    // PTO: self-loopback — all data arrived
    auto CheckDataArriveWithFlag = [&](uint32_t, int32_t, int32_t copyCnt) -> uint32_t {
        return (uint32_t)copyCnt;
    };
    if (false) {
        // Original A:2109-2127:
        //   flagNum = blockCntPerToken_ * copyCnt;
        //   DataCopy(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams);
        //   ReduceSum(flagSumOutTensor, flagRecvTensor_, ..., 1, flagNum, 1);
        //   return flagSumOutTensor.GetValue(0) == 1.0F * flagNum ? copyCnt : 0;
    }

    // --- GetLocalWindowSrcDataBlockIdx (A:2191-2201) ---
    // Original: srcDataBlockIdx = srcExpertId % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_
    //   3510 non-shared non-elastic: (srcExpertId + epRankId_) % epWorldSize_ * localExpertNum + ...
    // PTO: use 3510 formula (original test target is 3510)
    auto GetLocalWindowSrcDataBlockIdx = [&](uint32_t srcExpertId, uint32_t localExpertNum) -> uint32_t {
        // 3510 ring-offset: (srcExpertId + epRankId_) % epWorldSize_ when !shared && !elastic
        if (!(isShareExpertRankFlag_ || hasElasticInfoFlag_)) {
            return ((srcExpertId + (uint32_t)epRankId_) % epWorldSize_) * localExpertNum +
                   srcExpertId / epWorldSize_;
        }
        return (srcExpertId % epWorldSize_) * localExpertNum + srcExpertId / epWorldSize_;
    };

    // --- CopyInAndOut (A:2130-2187) ---
    // Original: DataCopyPad(xTmpTensor, window) → DataCopyPad(expandXOut) → DataCopyPad(expandIdx)
    //           with ping-pong buffer and event flag rotation
    // PTO: TLOAD from window → TSTORE to expandXOut; scalar triple/scale copy
    float clearFlagVal = 0.0f;

    auto CopyInAndOut = [&](uint8_t* wAddr, uint32_t index, uint32_t dstPosition,
                            uint32_t arriveCount, uint32_t, bool) {
        for (uint32_t slot = 0; slot < arriveCount; slot++) {
            uint32_t curDstPosition = dstPosition + slot;
            uint8_t* slotAddr = wAddr + (uint64_t)(expertFinishNum_[index] + slot) * hCommuSize_;

            // Fix (issue #348): block unpacking — inverse of the packing done in
            // TokenToExpert. Original Ascend C reads the window with
            // srcTokenCopyParams {blockCntPerToken_, 480, UB_ALIGN, 0}: each
            // block's 480B data area is gathered contiguously into UB, restoring
            // the linear token record (x [0,512) + triple [512,524) + scale
            // [524,528)) before the token/triple/scale reads below. Iterate
            // blocks from the first to the last (forward, inverse of packing).
            for (uint32_t b = 1; b < blockCntPerToken_; b++) {
                uint32_t srcOff = (uint32_t)b * SPLIT_BLOCK_DATA_SIZE;
                uint32_t cnt = (hOutSizeAlign_ > srcOff) ? (hOutSizeAlign_ - srcOff) : 0U;
                if (cnt > SPLIT_BLOCK_DATA_SIZE) cnt = SPLIT_BLOCK_DATA_SIZE;
                uint32_t* dst = reinterpret_cast<uint32_t*>(slotAddr + srcOff);
                uint32_t* src = reinterpret_cast<uint32_t*>(slotAddr + (uint64_t)b * SPLIT_BLOCK_SIZE);
                // src > dst with possible overlap [dst+512, dst+cnt): forward copy
                for (uint32_t i = 0; i < cnt / sizeof(uint32_t); i++) {
                    dst[i] = src[i];
                }
            }

            // 1) Read token data from window → expandXOut (scalar, A3 compliance)
            {
                const XOutType* src = reinterpret_cast<const XOutType*>(slotAddr);
                XOutType* dst = expandXOut + (uint64_t)curDstPosition * H;
                for (uint32_t e = 0; e < H; e++) {
                    dst[e] = src[e];
                }
            }

            // 2) Read triple from window → expandIdxOut (scalar: 3 int32)
            int32_t* triplePtr = reinterpret_cast<int32_t*>(slotAddr + tokenQuantAlign_ * sizeof(int32_t));
            expandIdxOut[curDstPosition * EXPAND_IDX_INFO + 0] = triplePtr[0];
            expandIdxOut[curDstPosition * EXPAND_IDX_INFO + 1] = triplePtr[1];
            expandIdxOut[curDstPosition * EXPAND_IDX_INFO + 2] = triplePtr[2];

            // 3) Read expert scale from window → expandScalesOut (scalar: 1 float)
            if (hasExpertScalesFlag_) {
                float scaleVal = *reinterpret_cast<float*>(slotAddr + expertScaleAlign_ * sizeof(int32_t));
                expandScalesOut[curDstPosition] = scaleVal;
            }
        }
    };

    // --- WaitAndFormatOutput (A:2248-2324) ---
    // Original: poll CheckDataArriveWithFlag → CopyInAndOut with ping-pong xTmpPingTensor/xTmpPongTensor
    // PTO: sequential (self-loopback: all data arrived)
    if (false) {
        // Original A:2248-2324:
        //   xTmpPingTensor = tBuf.GetWithOffset<XOutType>(dataBufferSize, 0);
        //   xTmpPongTensor = tBuf.GetWithOffset<XOutType>(dataBufferSize, dataBufferSize);
        //   flagReduceWorkTensor_ = tBuf.GetWithOffset<float>(...);
        //   flagRecvTensor_ = tBuf.GetWithOffset<float>(...);
        //   while (true): if expertLeftNum == 0: skip;
        //     copyCnt = min(expertLeftNum, maxCopyTokenCnt);
        //     arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNum, copyCnt);
        //     if arriveCount == copyCnt:
        //       xTmpTensor = (dataBufferId == 0) ? xTmpPingTensor : xTmpPongTensor;
        //       CopyInAndOut(xTmpTensor, wAddr, index, dstPosition, arriveCount, dataBufferId, ...)
        //     else: index = (index+1) % validNum;
        //     if validNum == finishNum: break;
        //   WaitFlag for used data buffers; ClearLocalWindowDataFlags
    }

    // Main output loop (sequential, no polling)
    {
        uint32_t finishNum = 0;
        uint32_t copyBatchId = 0;

        for (uint32_t idx = 0; idx < validNum_; idx++) {
            if (expertLeftNum_[idx] == 0) continue;

            uint32_t srcExpertId = expertMap_[idx];
            uint32_t srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, moeExpertNumPerRank_);
            uint32_t copyCnt = expertLeftNum_[idx];
            uint32_t arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, (int32_t)expertFinishNum_[idx], (int32_t)copyCnt);

            if (arriveCount == copyCnt) {
                uint32_t dstPosition = (srcExpertId != 0) ? (uint32_t)sendCntTensor_[srcExpertId - 1] : 0;
                dstPosition += expertFinishNum_[idx];
                uint8_t* wAddr = GetWindAddrByRankId(0) + (uint64_t)srcDataBlockIdx * expertPerSizeOnWin_;

                CopyInAndOut(wAddr, idx, dstPosition, arriveCount, copyBatchId % LOCAL_COPY_BUFFER_NUM,
                             copyBatchId >= LOCAL_COPY_BUFFER_NUM);
                ++copyBatchId;
                expertFinishNum_[idx] += arriveCount;
                expertLeftNum_[idx] -= arriveCount;
                if (expertLeftNum_[idx] == 0) finishNum++;
            }
            if (validNum_ == finishNum) break;
        }
    }

    // --- ClearLocalWindowDataFlags (A:2205-2244) ---
    // Original: Duplicate(0.0) → batched DataCopy clean flags per expert slot
    // Fix (issue #348): scalar writes limited to the 32B flag area — TSTORE tile
    // is 128B min and would clobber the next block's data area (see TokenToExpert).
    {
        constexpr uint32_t CLEAR_FLAG_FLOATS = (SPLIT_BLOCK_SIZE - SPLIT_BLOCK_DATA_SIZE) / sizeof(float);
        for (uint32_t idx = 0; idx < validNum_; idx++) {
            uint32_t srcExpertId = expertMap_[idx];
            uint32_t srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, moeExpertNumPerRank_);
            for (uint32_t slotIdx = 0; slotIdx < expertFinishNum_[idx]; slotIdx++) {
                uint8_t* wAddr = GetWindAddrByRankId(0) +
                                 (uint64_t)srcDataBlockIdx * expertPerSizeOnWin_ +
                                 (uint64_t)slotIdx * hCommuSize_;
                // Clear flag at each 512B block's flag area
                for (uint32_t blk = 0; blk < blockCntPerToken_; blk++) {
                    float* flagPtr = reinterpret_cast<float*>(
                        wAddr + blk * SPLIT_BLOCK_SIZE + SPLIT_BLOCK_DATA_SIZE);
                    for (uint32_t f = 0; f < CLEAR_FLAG_FLOATS; f++) {
                        flagPtr[f] = 0.0f;
                    }
                }
            }
        }
    }
    if (false) {
        // Original A:2205-2244:
        //   cleanRecordCapacity = tBufRealSize_ / UB_ALIGN (capped at DATA_COPY_MAX_BLOCK_COUNT)
        //   Duplicate<float>(cleanUpTensor, 0, cleanRecordNum * UB_ALIGN_DATA_COUNT);
        //   for each expert: DataCopy(cleanGlobal[flagIndex], cleanUpTensor, cleanUpParams)
        //     flagIndex = recordOffset * SPLIT_BLOCK_COUNT + SPLIT_BLOCK_DATA_COUNT
    }

    // --- RunPosRecord(RUNPOS_ARRIVECNT) (A:2327-2335) ---
    selfDataStatusGMTensor_[OPOSITION_POS] = RUNPOS_ARRIVECNT;
}

// ===================== Host-side reference (golden) ===================== //
template <typename XType, typename ExpandXOutType>
inline bool refDispatchCheck(const XType* x, const int32_t* expertIds, const float* expertScales,
                             const ExpandXOutType* expandXOut, const int32_t* expandIdxOut,
                             const float* expandScalesOut, const int32_t* sendCountsOut,
                             const int64_t* expertTokenNumsOut,
                             uint32_t bs, uint32_t h, uint32_t k, uint32_t moeExpertNum) {
    const uint32_t slotCnt = bs * k;
    int32_t counts[1024] = {0};
    for (uint32_t i = 0; i < slotCnt; i++) {
        int32_t e = expertIds[i];
        if (e >= 0 && static_cast<uint32_t>(e) < moeExpertNum) counts[e]++;
    }
    int32_t cum[1024] = {0};
    int32_t acc = 0;
    for (uint32_t e = 0; e < moeExpertNum; e++) {
        acc += counts[e];
        cum[e] = acc;
    }
    for (uint32_t e = 0; e < moeExpertNum; e++) {
        if (sendCountsOut[e] != cum[e]) return false;
    }
    for (uint32_t e = 0; e < moeExpertNum; e++) {
        if (expertTokenNumsOut[e] != counts[e]) return false;
    }
    uint32_t q = 0;
    for (uint32_t e = 0; e < moeExpertNum; e++) {
        for (uint32_t i = 0; i < slotCnt; i++) {
            if (static_cast<uint32_t>(expertIds[i]) != e) continue;
            uint32_t tokenId = i / k;
            uint32_t topkId = i % k;
            if (expandIdxOut[q * EXPAND_IDX_INFO + 0] != 0) return false;
            if (expandIdxOut[q * EXPAND_IDX_INFO + 1] != static_cast<int32_t>(tokenId)) return false;
            if (expandIdxOut[q * EXPAND_IDX_INFO + 2] != static_cast<int32_t>(topkId)) return false;
            for (uint32_t j = 0; j < h; j++) {
                float a = static_cast<float>(expandXOut[q * h + j]);
                float b = static_cast<float>(x[tokenId * h + j]);
                if (std::abs(a - b) > 1e-6f) return false;
            }
            float sc = expandScalesOut[q];
            float sb = expertScales[i];
            if (std::abs(sc - sb) > 1e-6f) return false;
            q++;
        }
    }
    return q == slotCnt;
}

} // namespace supernpu::tile_isa
#endif
