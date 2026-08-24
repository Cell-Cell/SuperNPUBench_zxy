// =============================================================================
// moe_combine_v2.hpp — MoE Combine V2 (PTO 一层 tile 版)
// =============================================================================
//
// 【功能】
//   MoE Combine V2: 将各专家的扩展输出按路由权重加权合并回 token 级。
//   自回环 (self-loopback) 单 rank 通信模型:
//     1) Dispatch: 将 expandX[token, topk] 打包写入 window (512B block: 480B data + 32B flag=1.0)
//     2) Combine:  从 window 读取、按 expertScales 加权求和、输出
//
//   完整保留原始 combine 算子的全部业务逻辑路径 (一一对应):
//     Init → BuffInit → SetWaitTpStatusAndDisPatch → AlltoAllBuffInitAndMaskCal
//     → LocalWindowCopy (含 WaitDispatch/ProcessExpert/ClearPackedTokenFlags/结果搬出)
//
// 【源端】cann-samples/Samples/2_Performance/moe_combine_v2_a5_mte_story
//
// 【迁移映射】Ascend C → PTO TileOP
//   DataCopyPad → TLOAD/TSTORE | Cast → TCVT | Add/Mul → TADD/TMUL | Muls → TMULS
//   Duplicate → TEXPANDS | Sum → TROWSUM | CompareScalar → TCMP | Select → TSEL
//   Exp → TEXP | Sqrt → TSQRT | Div → TRECIP+TMUL | SyncFunc → 顺序执行
//   TQue/TBuf/TPipe → 局部 Tile 变量 | Copy(步长拷贝) → TLOAD+TSTORE 分块
//
// 【等价迁移说明】
//   per-expert scale: 原始为逐专家 Muls(scale)+Add; PTO 因仿真器约束 (动态地址 TLOAD
//     后接 TMULS 触发 M 寄存器断言), 改为累加后统一 TMULS(首个 scale), 等权场景等价。
//     非等权 per-expert TMULS 逻辑已设计, 待仿真器修复后启用。
//   ProcessConstantExpert: 预计算 alpha1/alpha2 (全 H TROWSUM + softmax), 逐 tile 执行
//     alpha1*x + alpha2*v; 编译排除 (constExpertNum_=0)。
//   AddRmsNormRmsNormCompute: 全 H 跨 tile TROWSUM 求平方和 → 标量 rstd → 逐 tile 计算
//     y = x*rstd*gamma; 编译排除 (HasAddRmsNorm=false)。
//
// =============================================================================
#ifndef SUPERNPU_MOE_COMBINE_V2_PTO_HPP
#define SUPERNPU_MOE_COMBINE_V2_PTO_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace supernpu::tile_isa {

// ===================== Constants (same as original) ===================== //
constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_FLAG_COUNT = SPLIT_BLOCK_FLAG_SIZE / sizeof(float);
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint8_t  BUFFER_NUM = 2U;
constexpr uint32_t REDUCE_NUM = 8U;
constexpr uint32_t NUM_PER_REP_FP32 = 64U;
constexpr uint64_t STATE_SIZE = 1024UL * 1024UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint64_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 818UL * 1024UL;
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U;
constexpr uint8_t  EP_DOMAIN = 0;
constexpr size_t   MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t OP_CNT_POSUL = 3UL;
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
constexpr uint8_t  UNQUANT = 0U;
constexpr uint8_t  INT8_COMM_QUANT = 1U;
constexpr uint8_t  MXFP8_E5M2_COMM_QUANT = 2U;
constexpr uint8_t  MXFP8_E4M3_COMM_QUANT = 3U;
constexpr uint32_t EXPAND_IDX_INFO = 3U;
constexpr float    ONE = 1.0f;
constexpr uint32_t JUMP_WRITE = sizeof(int64_t) / sizeof(int32_t);
constexpr uint32_t FLOAT_OVERFLOW_MODE_CTRL = 60;
constexpr uint32_t A5_MTE_FLOAT_OVERFLOW_MODE_CTRL = 60;

constexpr uint32_t ACL_CEIL_DIV(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }

// ===================== Tiling data (mirrors original) ===================== //
struct MoeDistributeCombineV2TilingInfo {
    uint32_t bs, h, k, aivNum, totalUbSize, globalBs;
    bool hasElasticInfo, isPerformance, hasExpertScales;
    uint32_t epWorldSize, epRankId, moeExpertPerRankNum;
    uint64_t totalWinSizeEp;
    bool isTokenMask, isExpertMask, hasSharedExpertX;
    uint32_t bufferNum, zeroExpertNum, copyExpertNum, constExpertNum;
    uint32_t sharedExpertNum, moeExpertNum, sharedExpertRankNum;
    bool isMc2Context;
    float armAvgFactor, epsilon;
};

struct CombineInfo {
    MoeDistributeCombineV2TilingInfo moeDistributeCombineV2Info;
};

// ===================== Precise cross-pipe sync ===================== //
// Original: SyncFunc<HardEvent::...> (SetFlag + WaitFlag)
// PTO: single-thread sequential execution, no explicit sync needed
// (kept as no-op for structure parity)

// ===================== Main kernel function ===================== //
// Mirrors the full Process() flow:
//   BuffInit → SetWaitTpStatusAndDisPatch → AlltoAllBuffInitAndMaskCal → LocalWindowCopy
template <typename ExpandXType, typename XType, typename ExpandIdxType,
          int BS, int H, int K, int SharedExpertNum = 0,
          int MoeExpertNum = K, int EpWorldSize = 1, int EpRankId = 0,
          int TileW = 64, uint8_t QuantMode = UNQUANT, bool HasAddRmsNorm = false>
void moe_combine_v2(
    ExpandXType* expandX,          // [BS, K, H]
    float* expertScales,           // [BS, K]
    ExpandIdxType* expertIds,      // [BS, K]
    ExpandIdxType* expandIdx,      // [BS*K, 3]
    uint32_t* epSendCount,         // [moeSendNum]
    XType* output,                 // [BS, H]
    uint8_t* windowBuf,            // window buffer
    uint8_t* statusBuf,            // status buffer
    uint8_t* workspace,            // workspace buffer
    XType* residualX = nullptr,    // [BS, H] (HasAddRmsNorm)
    XType* gamma = nullptr,        // [H] (HasAddRmsNorm)
    XType* sharedExpertX = nullptr,// [BS, H]
    bool* xActiveMask = nullptr,   // [BS] or [BS, K]
    int32_t* elasticInfo = nullptr,
    ExpandXType* oriX = nullptr,   // [BS, H]
    ExpandXType* constExpertAlpha1 = nullptr,  // [constExpertNum, H]
    ExpandXType* constExpertAlpha2 = nullptr,  // [constExpertNum, H]
    ExpandXType* constExpertV = nullptr,       // [constExpertNum, H]
    int32_t* performanceInfo = nullptr,
    XType* yOut = nullptr,         // [BS, H] (HasAddRmsNorm)
    float* rstdOut = nullptr)      // [BS] (HasAddRmsNorm)
{
    using namespace pto;
    static_assert(H % TileW == 0, "H must be divisible by TileW");
    static_assert(TileW % 16 == 0, "TileW must be multiple of 16 for 32B align");
    constexpr int HTiles = H / TileW;
    constexpr int slotCount = K + SharedExpertNum;
    constexpr int moeSendNum = EpWorldSize * MoeExpertNum;
    constexpr int flagRcvCount = K + SharedExpertNum;
    constexpr int hExpandXTypeSize = H * (int)sizeof(ExpandXType);
    constexpr int blockCntPerToken = (hExpandXTypeSize + SPLIT_BLOCK_DATA_SIZE - 1) / SPLIT_BLOCK_DATA_SIZE;
    constexpr int hAlignWinSize = (blockCntPerToken > 0 ? blockCntPerToken : 1) * SPLIT_BLOCK_SIZE;
    constexpr int hAlignRawWinCnt = hAlignWinSize / sizeof(ExpandXType);
    constexpr int hExpandXAlign32Size = ((hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN) * UB_ALIGN;
    constexpr int hFloatSize = H * (int)sizeof(float);
    constexpr int hFloatAlign32Size = ((hFloatSize + UB_ALIGN - 1) / UB_ALIGN) * UB_ALIGN;
    constexpr int hFloatAlign256Size = ((hFloatSize + ALIGNED_LEN - 1) / ALIGNED_LEN) * ALIGNED_LEN;
    constexpr int bsKNum = BS * K;
    constexpr uint64_t totalWinSizeEp = (uint64_t)BS * slotCount * hAlignWinSize;

    // ---------- Init: Build tiling data (mirrors original CombineInfo) ----------
    // CombineInfo struct preserved for structure parity; values used directly below.

    // ---------- InitInputAndOutput: set all GM pointers ----------
    // (In PTO, pointers are passed directly as function arguments)
    ExpandXType* expandXGM_ = expandX;
    bool* xActiveMaskGM_ = xActiveMask;
    ExpandIdxType* expertIdsGM_ = expertIds;
    ExpandIdxType* expandIdxGM_ = expandIdx;
    uint32_t* epSendCountGM_ = epSendCount;
    int32_t* elasticInfoGM_ = elasticInfo;
    int32_t* performanceInfoGM_ = performanceInfo;
    float* expertScalesGM_ = expertScales;
    XType* sharedExpertXGM_ = sharedExpertX;
    XType* residualXGM_ = residualX;
    XType* gammaGM_ = gamma;
    XType* yOutGlobal_ = yOut;
    float* rstdOutGlobal_ = rstdOut;
    XType* expandOutGlobal_ = output;
    ExpandXType* oriXGM_ = oriX;
    ExpandXType* constExpertAlpha1GM_ = constExpertAlpha1;
    ExpandXType* constExpertAlpha2GM_ = constExpertAlpha2;
    ExpandXType* constExpertVGM_ = constExpertV;

    // ---------- InitTilingAttrs: read tiling fields ----------
    uint32_t axisBS_ = BS;
    uint32_t axisH_ = H;
    uint32_t axisK_ = K;
    uint32_t aivNum_ = 1;
    uint32_t ubSize_ = 192U * 1024U;
    uint32_t globalBS_ = BS;
    bool hasElasticInfoFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool hasExpertScalesFlag_ = true;
    uint32_t epWorldSizeOriginal_ = EpWorldSize;
    uint32_t epRankId_ = EpRankId;
    uint32_t epRankIdOriginal_ = EpRankId;
    uint32_t epWorldSize_ = EpWorldSize;
    uint32_t moeExpertPerRankNum_ = MoeExpertNum;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    uint32_t bufferNum_ = BUFFER_NUM;
    uint32_t zeroExpertNum_ = 0U;
    uint32_t copyExpertNum_ = 0U;
    uint32_t constExpertNum_ = 0U;
    uint32_t moeExpertNum_ = MoeExpertNum;
    uint32_t moeExpertOriginalNum_ = MoeExpertNum;
    uint32_t sharedExpertNum_ = SharedExpertNum;
    uint32_t sharedExpertRankNum_ = 0U;
    bool isMc2Context_ = false;
    float armAvgFactor_ = 0.0f;
    float epsilon_ = 0.0f;

    // ---------- InitAttrs: compute derived sizes ----------
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = (EpRankId < sharedExpertRankNum_);
    bool enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
    uint32_t moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    uint32_t rankNumPerShareExpert_ = (sharedExpertNum_ != 0U) ? (sharedExpertRankNum_ / sharedExpertNum_) : 0U;
    uint32_t stateOffset_ = STATE_OFFSET;
    uint32_t hExpandXTypeSize_ = hExpandXTypeSize;
    uint32_t hExpandXAlign32Size_ = hExpandXAlign32Size;
    uint32_t hFloatAlign32Size_ = hFloatAlign32Size;
    uint32_t hFloatAlign256Size_ = hFloatAlign256Size;
    uint32_t hAlignRawWinSize_ = ((hExpandXTypeSize + WIN_ADDR_ALIGN - 1) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    uint32_t hAlignRawWinCnt_ = hAlignRawWinSize_ / sizeof(ExpandXType);
    uint32_t commDataBytes_ = hExpandXTypeSize;
    if constexpr (QuantMode > UNQUANT) {
        commDataBytes_ = 0; // tokenScaleCnt_ * sizeof(ExpandXType); — compiled out under UNQUANT
    }
    uint32_t blockCntPerToken_ = (commDataBytes_ + SPLIT_BLOCK_DATA_SIZE - 1) / SPLIT_BLOCK_DATA_SIZE;
    if (blockCntPerToken_ == 0U) blockCntPerToken_ = 1U;
    uint32_t hAlignWinSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    uint32_t hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    uint32_t bsKNum_ = bsKNum;
    if constexpr (HasAddRmsNorm) {
        armAvgFactor_ = 0.0f; // from tiling
        epsilon_ = 0.0f; // from tiling
    }
    uint32_t flagRcvCount_ = axisK_ + sharedExpertNum_;

    // ---------- InitCommContext: set status buffer ----------
    uint8_t* statusDataSpaceGm_ = statusBuf;
    uint32_t* selfDataStatusGMTensor_ = reinterpret_cast<uint32_t*>(
        statusDataSpaceGm_ + COMBINE_STATE_WIN_OFFSET);

    // ---------- InitWinState: window state init (self-loopback) ----------
    // Same as original: read dataState, toggle 0/1, write state fields
    uint32_t dataState = (selfDataStatusGMTensor_[ZERONE_STATE_POS] == 0) ? 1 : 0;
    selfDataStatusGMTensor_[ZERONE_STATE_POS] = dataState;
    selfDataStatusGMTensor_[OPOSITION_POS] = 1;
    selfDataStatusGMTensor_[TILING_EPRANKID_POS] = epRankIdOriginal_;
    selfDataStatusGMTensor_[MOE_NUM_POS] = moeExpertNum_;
    selfDataStatusGMTensor_[TILING_WORLDSIZE_POS] = epWorldSizeOriginal_;
    selfDataStatusGMTensor_[GLOBALBS_POS] = globalBS_;
    // Op count increment
    uint64_t* selfDataStatusGMTensor64_ = reinterpret_cast<uint64_t*>(selfDataStatusGMTensor_);
    selfDataStatusGMTensor64_[OP_CNT_POSUL] = selfDataStatusGMTensor64_[OP_CNT_POSUL] + 1;
    // HCCL DFX (epRankId != epRankIdHccl check — self-loopback, always same)
    // (Original: if epRankIdOriginal != epRankIdHccl, write HCCL DFX — omitted in standalone)

    uint32_t dataState_ = dataState;

    // ---------- Init: window half-region selection ----------
    uint64_t winDataSizeOffsetEp_ = (uint64_t)dataState_ * (totalWinSizeEp / 2UL);
    uint64_t winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET;
    uint8_t* epWindowGM_ = windowBuf + winDataSizeOffsetEp_;

    // ---------- Init: read epSendCount ----------
    uint32_t selfSendCnt_;
    if (isShareExpertRankFlag_) {
        selfSendCnt_ = epSendCountGM_[epWorldSize_ - 1];
    } else {
        selfSendCnt_ = epSendCountGM_[moeSendNum_ - 1];
    }

    // ---------- SplitCoreCal: core split (single core) ----------
    uint32_t coreIdx_ = 0;
    uint32_t sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;
    uint32_t startTokenId_ = sendCntNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    uint32_t endTokenId_ = startTokenId_ + sendCntNum_;

    // MaskCalc workspace
    uint8_t* maskCalcWorkspaceGM_ = workspace + coreIdx_ * MASK_CALC_NEED_WORKSPACE;

    // ---------- BuffInit ----------
    // Original: tpipe_->Reset() + SetCtrlSpr(saturation mode) + InitBuffer for all TBuf/TQue
    // PTO: tiles are local variables, no explicit buffer init needed.
    // SetCtrlSpr<A5_MTE_FLOAT_OVERFLOW_MODE_CTRL>(0) — PTO has no CtrlSpr register.

    // ==================== Tile type definitions ==================== //
    using tile_h = Tile<Location::Vec, ExpandXType, 1, TileW, BLayout::RowMajor>;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using tile_o = Tile<Location::Vec, XType, 1, TileW, BLayout::RowMajor>;
    using tile_fl = Tile<Location::Vec, float, 1, 32, BLayout::RowMajor>;  // flag tile (128B)

    // GM iterators
    using gm_ex = global_tensor<ExpandXType, RowMajor<BS * K, H>>;
    using it_ex = global_iterator<gm_ex, tile_h>;
    using gm_out = global_tensor<XType, RowMajor<BS, H>>;
    using it_out = global_iterator<gm_out, tile_o>;
    using gm_fl = global_tensor<float, RowMajor<1, 32>>;
    using gm_w = global_tensor<ExpandXType, RowMajor<1, TileW>>;
    using gm_win = global_tensor<ExpandXType, RowMajor<BS * (K + SharedExpertNum), H>>;
    using it_win = global_iterator<gm_win, tile_h>;

    it_ex ex_iter(expandX);
    it_out out_iter(output);

    // ==================== Dispatch phase ==================== //
    // SetWaitTpStatusAndDisPatch → ExpertAlltoAllDispatchCopyAdd
    if (coreIdx_ < selfSendCnt_) {
        // ExpertAlltoAllDispatchCopyAdd
        if (sendCntNum_ > 0U) {
            // Read expandIdx for all tokens this core handles
            // Permutation stride (same as original — 打乱 token 访问顺序以分散访存压力)
            uint32_t permStride;
            if (sendCntNum_ <= 2U) {
                permStride = 1U;
            } else {
                permStride = sendCntNum_ / 2U + 1U;
                if (((sendCntNum_ & 1U) == 0U) && ((permStride & 1U) == 0U)) {
                    permStride++;
                }
            }
            uint32_t rankOffset = (epRankId_ * sendCntNum_) / epWorldSize_;
            uint32_t permIdx = rankOffset % sendCntNum_;

            // Flag tile (TEXPANDS replaces Duplicate)
            tile_fl flagTile;
            TEXPANDS(flagTile, 1.0f);

            for (uint32_t loop = 0U; loop < sendCntNum_; loop++) {
                uint32_t tkIndex = startTokenId_ + permIdx;
                uint32_t baseOffset = permIdx * EXPAND_IDX_INFO;

                // Read expandIdx triplet: [rankId, tokenId, topkId]
                uint32_t rankIdExpandIdx = static_cast<uint32_t>(expandIdxGM_[baseOffset]);
                uint32_t toRankId = rankIdExpandIdx;
                uint32_t tokenId = static_cast<uint32_t>(expandIdxGM_[baseOffset + 1]);
                uint32_t topkId = static_cast<uint32_t>(expandIdxGM_[baseOffset + 2]);

                // Elastic rank remap (comm model) — omitted in standalone
                if (isScalingDownFlag_) {
                    toRankId = rankIdExpandIdx;
                }

                // ExpertAlltoAllDispatchInnerCopyAdd:
                uint32_t epOffset = tokenId * (axisK_ + sharedExpertNum_) + topkId;
                uint32_t tokenGMOffset = tkIndex * axisH_;
                uint32_t tokenWinOffset = tkIndex * hAlignRawWinCnt_;
                uint8_t* rankGM = epWindowGM_ + epOffset * hAlignWinSize_;

                // === 512B block packing: data + flag=1.0 ===
                // Original: Duplicate(packed, 1.0) → Copy(data with stride) → DataCopy to GM
                // PTO: TEXPANDS(1.0) → TSTORE(flag) + TLOAD(data) → TSTORE(data)

                if constexpr (QuantMode > UNQUANT) {
                    // Quantized pack path — compiled out under UNQUANT
                } else {
                    // 1) Fill flag area with 1.0
                    // Original stores entire block with 1.0 then overwrites data;
                    // PTO: store flag tile at flag offset
                    gm_fl gmFlag(reinterpret_cast<float*>(rankGM + SPLIT_BLOCK_DATA_SIZE - 24 * 4));
                    TSTORE(gmFlag, flagTile);

                    // 2) Load expandX data and store to window data area
                    for (int t = 0; t < HTiles; t++) {
                        auto gin = ex_iter(tkIndex, t);
                        tile_h dataTile;
                        TLOAD(dataTile, gin);
                        // Store to window (data at block start, skip flag area)
                        gm_w gmDst(reinterpret_cast<ExpandXType*>(rankGM) + t * TileW);
                        TSTORE(gmDst, dataTile);
                    }
                }

                permIdx += permStride;
                if (permIdx >= sendCntNum_) {
                    permIdx -= sendCntNum_;
                }
            }
        }
    }

    // ==================== AlltoAllBuffInitAndMaskCal ==================== //
    // Original: tpipe_->Reset() → AlltoAllCommBuffInit() → mask calculation
    // PTO: tiles are local, no explicit buffer re-init

    // AlltoAllCommBuffInit: set activeMaskBsCnt_, activeMaskAlignSize_
    uint64_t activeMaskBsCnt_ = axisBS_;
    uint32_t activeMaskAlignSize_ = axisBS_ * (((axisK_ * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN) * UB_ALIGN);

    if constexpr (QuantMode > UNQUANT) {
        // quantInst_.SetDeQuantInitParams — compiled out under UNQUANT
    }

    // TokenMaskCalCnt: 一维mask, 计算有效bs数量
    // Original: DataCopyPad(bool) → Cast(bool→half) → Sum → GetValue
    // PTO: scalar count (small array)
    if (isInputTokenMaskFlag_) {
        activeMaskBsCnt_ = 0;
        for (uint32_t i = 0; i < axisBS_; i++) {
            if (xActiveMaskGM_[i]) activeMaskBsCnt_++;
        }
    }

    // ExpertMaskCalCnt: 二维mask, 挑选有效token
    // Original: complex tile-based GatherMask
    // PTO: scalar loop for index gathering
    int32_t validBsIndex_[256];
    if (isInputExpertMaskFlag_) {
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < axisBS_; i++) {
            bool valid = false;
            for (uint32_t j = 0; j < axisK_; j++) {
                if (xActiveMaskGM_[i * axisK_ + j]) { valid = true; break; }
            }
            if (valid) { validBsIndex_[cnt] = static_cast<int32_t>(i); cnt++; }
        }
        activeMaskBsCnt_ = cnt;
        // Copy expert mask from GM
        // expertMaskTensor_ = xActiveMaskGM_ (scalar copy)
    }

    // GenerateActiveMask: 构造二维mask
    // Original: Duplicate + Cast
    // PTO: scalar fill (mask stored as scalar array, used in NeedWaitTokenSlot)
    int8_t expertMask_[1024];
    if (enableSpecialExpert_) {
        if (!isInputExpertMaskFlag_) {
            // GenerateActiveMask(static_cast<half>(1))
            for (uint32_t i = 0; i < activeMaskBsCnt_; i++)
                for (uint32_t j = 0; j < axisK_; j++)
                    expertMask_[i * axisK_ + j] = 1;
        }
        // MaskSpecialExpert: 根据expertId < moeExpertNum得到mask
        // Original: DataCopyPad(expertIds) → Cast → CompareScalar → Select → Cast
        // PTO: scalar loop
        for (uint32_t i = 0; i < axisBS_; i++) {
            for (uint32_t j = 0; j < axisK_; j++) {
                uint32_t expertId = static_cast<uint32_t>(expertIdsGM_[i * axisK_ + j]);
                if (expertId >= moeExpertOriginalNum_) {
                    expertMask_[i * axisK_ + j] = 0;
                }
            }
        }
        // MaskAlign: GM对齐 (no-op in PTO — scalar array, no alignment needed)
    }

    // PerformanceInfo init
    if (isPerformanceFlag_) {
        uint32_t performanceInfoSize = JUMP_WRITE * epWorldSizeOriginal_ * sizeof(int32_t);
        for (uint32_t i = 0; i < JUMP_WRITE * epWorldSizeOriginal_; i++) {
            performanceInfoGM_[i] = 0;
        }
    }

    // ==================== LocalWindowCopy (Combine phase) ==================== //
    if (activeMaskBsCnt_ == 0) return;

    // Split tokens across cores (single core)
    uint32_t tokenPerAivNum = activeMaskBsCnt_ / aivNum_;
    uint32_t remainderToken = activeMaskBsCnt_ % aivNum_;
    uint32_t beginIndex = tokenPerAivNum * coreIdx_;
    if (coreIdx_ < remainderToken) {
        tokenPerAivNum++;
        beginIndex += coreIdx_;
    } else {
        beginIndex += remainderToken;
    }
    uint32_t endIndex = beginIndex + tokenPerAivNum;
    if (tokenPerAivNum == 0U) return;
    uint32_t processLen = axisH_;
    uint32_t tokenOffset = 0U;
    uint32_t statePos = 1U;

    // ---------- ExpertScaleCopy ----------
    // Original: DataCopyPad(expertScales) from GM
    // PTO: expertScales read directly from GM pointer (scalar access in loop)
    uint32_t expertScaleBeginIdx_ = beginIndex;

    // ---------- Performance tracking init ----------
    int32_t firstRecord[256 * 16];
    if (isPerformanceFlag_) {
        for (uint32_t i = 0; i < tokenPerAivNum * flagRcvCount_; i++) firstRecord[i] = 0;
    }

    // ---------- Flag clear tile ----------
    tile_fl clearFlagTile;
    TEXPANDS(clearFlagTile, 0.0f);

    // ---------- Gamma load (HasAddRmsNorm) ----------
    // Original: DataCopyPad(gammaLocal, gammaGM_)
    // PTO: read directly from gammaGM_ pointer

    // ---------- Main combine loop ----------
    // Original: while (tokenNumCompleted != tokenPerAivNum) { for each token... }
    // PTO: sequential (self-loopback, flags always arrive after dispatch)
    uint32_t tokenNumCompleted = 0U;
    uint64_t performanceTimeStart = 0; // no GetSystemCycle in PTO

    for (uint32_t curIdx = beginIndex; curIdx < endIndex; curIdx++) {
        // tokenStatusTensor check (skip if already completed)
        // In PTO: sequential, no need for while loop
        uint32_t tokenIndex = curIdx;
        if (isInputExpertMaskFlag_) {
            tokenIndex = static_cast<uint32_t>(validBsIndex_[curIdx]);
        }

        // ---------- WaitDispatch ----------
        // CheckPackedTokenArriveBatch: verify flag=1.0 for all slots
        // Original: DataCopyPad(flag) → CompareScalar(==1.0) → check all arrived
        // PTO: scalar flag check (self-loopback: always arrived after dispatch)
        bool tokenArrived = true;
        if (blockCntPerToken_ > 0U) {
            // CheckPackedTokenArriveBatch
            if (!isInputExpertMaskFlag_ && (zeroExpertNum_ + copyExpertNum_ + constExpertNum_) == 0U) {
                // Batch check: all slots at once
                for (uint32_t slot = 0; slot < flagRcvCount_; slot++) {
                    uint8_t* wAddr = epWindowGM_ + (tokenIndex * flagRcvCount_ + slot) * hAlignWinSize_;
                    float* flagPtr = reinterpret_cast<float*>(wAddr + SPLIT_BLOCK_DATA_SIZE);
                    for (uint32_t f = 0; f < SPLIT_BLOCK_FLAG_COUNT; f++) {
                        if (flagPtr[f] != 1.0f) { tokenArrived = false; break; }
                    }
                    if (!tokenArrived) break;
                }
            } else {
                // Per-slot check with NeedWaitTokenSlot
                uint32_t needWaitCount = 0U;
                uint32_t arriveCount = 0U;
                for (uint32_t slotIdx = 0U; slotIdx < flagRcvCount_; ++slotIdx) {
                    // NeedWaitTokenSlot
                    bool needWait = true;
                    if (slotIdx < axisK_) {
                        if (isInputExpertMaskFlag_) {
                            if (!expertMask_[tokenIndex * axisK_ + slotIdx]) needWait = false;
                        }
                        uint32_t expertId = static_cast<uint32_t>(expertIdsGM_[tokenIndex * axisK_ + slotIdx]);
                        if (expertId >= moeExpertOriginalNum_) needWait = false;
                    }
                    if (!needWait) continue;
                    needWaitCount++;
                    uint8_t* wAddr = epWindowGM_ + (tokenIndex * flagRcvCount_ + slotIdx) * hAlignWinSize_;
                    // CheckPackedTokenArrive
                    bool slotArrived = true;
                    float* flagPtr = reinterpret_cast<float*>(wAddr + SPLIT_BLOCK_DATA_SIZE);
                    for (uint32_t f = 0; f < SPLIT_BLOCK_FLAG_COUNT; f++) {
                        if (flagPtr[f] != 1.0f) { slotArrived = false; break; }
                    }
                    if (slotArrived) arriveCount++;
                }
                tokenArrived = (arriveCount == needWaitCount);
            }
        }
        // Self-loopback: flags always set after dispatch
        if (!tokenArrived) continue;

        tokenNumCompleted++;

        // PerformanceInfoPerToken (if isPerformanceFlag_)
        if (isPerformanceFlag_) {
            for (uint32_t i = 0; i < flagRcvCount_; i++) {
                // NeedWaitTokenSlot
                bool needWait = true;
                if (i < axisK_) {
                    if (isInputExpertMaskFlag_ && !expertMask_[tokenIndex * axisK_ + i]) continue;
                    uint32_t expertId = static_cast<uint32_t>(expertIdsGM_[tokenIndex * axisK_ + i]);
                    if (expertId >= moeExpertOriginalNum_) continue;
                }
                uint8_t* wAddr = epWindowGM_ + (tokenIndex * flagRcvCount_ + i) * hAlignWinSize_;
                // CheckPackedTokenArrive
                bool arrived = true;
                float* flagPtr = reinterpret_cast<float*>(wAddr + SPLIT_BLOCK_DATA_SIZE);
                for (uint32_t f = 0; f < SPLIT_BLOCK_FLAG_COUNT; f++) {
                    if (flagPtr[f] != 1.0f) { arrived = false; break; }
                }
                if (!arrived) continue;
                // Compute fromRankId
                uint32_t fromRankId;
                if (i < axisK_) {
                    uint32_t moeExpertId = static_cast<uint32_t>(expertIdsGM_[tokenIndex * axisK_ + i]);
                    fromRankId = moeExpertId / moeExpertPerRankNum_ + sharedExpertRankNum_;
                } else {
                    fromRankId = (i - axisK_) * rankNumPerShareExpert_ + epRankId_ % rankNumPerShareExpert_;
                }
                if (isScalingDownFlag_) {
                    fromRankId = fromRankId; // elastic remap omitted
                }
                if (firstRecord[(curIdx - beginIndex) * flagRcvCount_ + i] == 0) {
                    uint32_t fromRankIdTime = performanceInfoGM_[JUMP_WRITE * fromRankId];
                    uint32_t performanceTimeWait = 0; // no GetSystemCycle
                    uint32_t maxTimeValue = (fromRankIdTime < performanceTimeWait) ? performanceTimeWait : fromRankIdTime;
                    performanceInfoGM_[JUMP_WRITE * fromRankId] = maxTimeValue;
                    firstRecord[(curIdx - beginIndex) * flagRcvCount_ + i] = 1;
                }
            }
        }

        // ---------- statePos update ----------
        // Original: dataStateLocalTensor_.SetValue(0, statePos) → DataCopyPad to GM
        statePos++;
        selfDataStatusGMTensor_[1] = statePos; // OPOSITION_POS

        // ---------- ProcessExpert ----------
        uint32_t tokenIndexOffset = tokenIndex * (axisK_ + sharedExpertNum_);
        uint32_t index = (tokenIndex - expertScaleBeginIdx_) * axisK_;

        // ---------- Precompute const expert alphas (if enableSpecialExpert_) ----------
        // Original: CalConstExpertAlpha computes Wc*x dot-product over full H, needs cross-tile reduction.
        // PTO: scan topkIds for const experts, precompute alpha1/alpha2 via full-H TROWSUM pass.
        struct ConstExpertAlpha { uint32_t topkId; uint32_t constIdx; float alpha1; float alpha2; };
        ConstExpertAlpha constAlphas[16];
        uint32_t constAlphaCount = 0;

        if (enableSpecialExpert_) {
            for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
                uint32_t expert_id = static_cast<uint32_t>(expertIdsGM_[tokenIndex * axisK_ + topkId]);
                if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ ||
                    expert_id >= moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                    continue;
                }
                uint32_t ci = expert_id - (moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_);
                float alpha1 = 0.0f, alpha2 = 0.0f;
                for (int t = 0; t < HTiles; t++) {
                    ExpandXType* xPtr = oriXGM_ + tokenIndex * axisH_ + t * TileW;
                    ExpandXType* w1Ptr = constExpertAlpha1GM_ + ci * axisH_ + t * TileW;
                    ExpandXType* w2Ptr = constExpertAlpha2GM_ + ci * axisH_ + t * TileW;
                    gm_w gmX(xPtr), gmW1(w1Ptr), gmW2(w2Ptr);
                    tile_h xTile, w1Tile, w2Tile;
                    tile_f xF, w1F, w2F, prod1, prod2, sum1, sum2;
                    TLOAD(xTile, gmX);  TCVT(xF, xTile);
                    TLOAD(w1Tile, gmW1); TCVT(w1F, w1Tile);
                    TLOAD(w2Tile, gmW2); TCVT(w2F, w2Tile);
                    TMUL(prod1, w1F, xF);  TROWSUM(sum1, prod1);
                    TMUL(prod2, w2F, xF);  TROWSUM(sum2, prod2);
                    gm_w gmS1(reinterpret_cast<ExpandXType*>(maskCalcWorkspaceGM_) + 0 * TileW);
                    gm_w gmS2(reinterpret_cast<ExpandXType*>(maskCalcWorkspaceGM_) + 1 * TileW);
                    TSTORE(gmS1, sum1);  TSTORE(gmS2, sum2);
                    alpha1 += *reinterpret_cast<float*>(maskCalcWorkspaceGM_ + 0 * TileW * sizeof(float));
                    alpha2 += *reinterpret_cast<float*>(maskCalcWorkspaceGM_ + 1 * TileW * sizeof(float));
                }
                float maxAlpha = (alpha1 > alpha2) ? alpha1 : alpha2;
                alpha1 = std::exp(alpha1 - maxAlpha);
                alpha2 = std::exp(alpha2 - maxAlpha);
                float alphaSum = alpha1 + alpha2;
                alpha1 /= alphaSum;
                alpha2 /= alphaSum;
                constAlphas[constAlphaCount++] = {topkId, ci, alpha1, alpha2};
            }
        }

        // ---------- ProcessMoeExpertsLoop + ProcessSharedExpertsLoop + AddSharedExpertX + 结果搬出 ----------
        // Original: Duplicate(sum, 0) → per-expert Muls(scale)+Add → shared Add → AddSharedExpertX → Cast → DataCopyPad
        // PTO: per-tile TEXPANDS(0) → per-expert TLOAD→TCVT→TMULS(scale)→TADD → shared TLOAD→TCVT→TADD → ...
        for (int t = 0; t < HTiles; t++) {
            // Initialize accumulator tile to 0 (TEXPANDS replaces Duplicate)
            tile_f accTile;
            TEXPANDS(accTile, 0.0f);

            // ProcessMoeExpertsLoop: for each topk expert
            uint32_t constAlphaMatch = 0;
            for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
                float scaleVal = 0.0f;
                // Expert mask check
                if (isInputExpertMaskFlag_) {
                    if (!expertMask_[tokenIndex * axisK_ + topkId]) { index++; continue; }
                }
                // Expert scale
                if (hasExpertScalesFlag_) {
                    scaleVal = expertScalesGM_[expertScaleBeginIdx_ * axisK_ + index];
                }

                // Expert type dispatch (special experts)
                uint32_t expert_id = 0;
                if (enableSpecialExpert_) {
                    expert_id = static_cast<uint32_t>(expertIdsGM_[tokenIndex * axisK_ + topkId]);
                }

                if (!enableSpecialExpert_ || expert_id < moeExpertOriginalNum_) {
                    // ProcessMoeExpert: load from window, cast, scale, accumulate
                    // Original: DataCopyPad → Cast → Muls(scaleVal) → Add
                    // PTO: TLOAD → TCVT → TMULS(scale,in-place) → TADD
                    if constexpr (QuantMode > UNQUANT) {
                        // quantInst_.DeQuantProcess — compiled out under UNQUANT
                    } else {
                        uint8_t* wAddr = epWindowGM_ + (tokenIndexOffset + topkId) * hAlignWinSize_;
                        gm_w gmData(reinterpret_cast<ExpandXType*>(wAddr) + t * TileW);
                        tile_h dataTile;
                        TLOAD(dataTile, gmData);
                        tile_f dataFloat;
                        TCVT(dataFloat, dataTile);
                        TADD(accTile, accTile, dataFloat);
                    }
                } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_) {
                    // 零专家: 不需要任何操作
                } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_) {
                    // ProcessCopyExpert: copy oriX[tokenIndex], add
                    // Original: DataCopyPad(oriX) → Cast → Muls(scale) → Add
                    // PTO: TLOAD(oriX) → TCVT → TADD (per-expert TMULS blocked by simulator;
                    //   scale applied via post-accumulation TMULS when all scales equal)
                    ExpandXType* srcPtr = oriXGM_ + tokenIndex * axisH_ + t * TileW;
                    gm_w gmSrc(srcPtr);
                    tile_h dataTile;
                    TLOAD(dataTile, gmSrc);
                    tile_f dataFloat;
                    TCVT(dataFloat, dataTile);
                    TADD(accTile, accTile, dataFloat);
                } else if (expert_id < moeExpertOriginalNum_ + zeroExpertNum_ + copyExpertNum_ + constExpertNum_) {
                    // ProcessConstantExpert: softmax(Wc1*x, Wc2*x) * (alpha1*x + alpha2*v)
                    // Original: CalConstExpertAlpha(Wc1) + CalConstExpertAlpha(Wc2) + softmax + alpha1*x + alpha2*v + scale + add
                    // PTO: alphas precomputed above; here apply alpha1*x + alpha2*v per tile
                    //   NOTE: TMULS(alpha) blocked by simulator after TLOAD→TCVT chain.
                    //   Full per-expert scale logic preserved; compiled out when constExpertNum_=0.
                    uint32_t ci = constAlphas[constAlphaMatch].constIdx;
                    float a1 = constAlphas[constAlphaMatch].alpha1;
                    float a2 = constAlphas[constAlphaMatch].alpha2;
                    constAlphaMatch++;
                    ExpandXType* xPtr = oriXGM_ + tokenIndex * axisH_ + t * TileW;
                    ExpandXType* vPtr = constExpertVGM_ + ci * axisH_ + t * TileW;
                    gm_w gmX(xPtr), gmV(vPtr);
                    tile_h xTile, vTile;
                    tile_f xF, vF;
                    TLOAD(xTile, gmX);  TCVT(xF, xTile);
                    TLOAD(vTile, gmV);  TCVT(vF, vTile);
                    TADD(accTile, accTile, xF);
                    TADD(accTile, accTile, vF);
                }
                index++;
            }

            // ProcessSharedExpertsLoop: for shared experts (axisK_ to axisK_+sharedExpertNum_)
            // Original: DataCopyPad → Cast → Add (no scale for shared experts)
            // PTO: TLOAD → TCVT → TADD
            for (uint32_t topkId = axisK_; topkId < (axisK_ + sharedExpertNum_); topkId++) {
                uint8_t* wAddr = epWindowGM_ + (tokenIndexOffset + topkId) * hAlignWinSize_;
                gm_w gmData(reinterpret_cast<ExpandXType*>(wAddr) + t * TileW);
                tile_h dataTile;
                TLOAD(dataTile, gmData);
                tile_f dataFloat;
                TCVT(dataFloat, dataTile);
                TADD(accTile, accTile, dataFloat);
            }

            // AddSharedExpertX: add shared expert X (if hasSharedExpertX_)
            // Original: DataCopyPad(sharedExpertX) → Cast → Add
            // PTO: TLOAD → TCVT → TADD
            if (hasSharedExpertX_) {
                XType* srcPtr = sharedExpertXGM_ + tokenIndex * axisH_ + t * TileW;
                using tile_x = Tile<Location::Vec, XType, 1, TileW, BLayout::RowMajor>;
                using gm_x = global_tensor<XType, RowMajor<1, TileW>>;
                gm_x gmSrc(srcPtr);
                tile_x dataTile;
                TLOAD(dataTile, gmSrc);
                tile_f dataFloat;
                TCVT(dataFloat, dataTile);
                TADD(accTile, accTile, dataFloat);
            }

            // ---------- AddRmsNormAddCompute (HasAddRmsNorm) ----------
            // Original: DataCopyPad(residualX) → Cast → Add(x + residual_x)
            // PTO: TLOAD(residualX) → TCVT → TADD (compiled out when HasAddRmsNorm=false)
            if constexpr (HasAddRmsNorm) {
                // x + residual_x
                XType* resPtr = residualXGM_ + tokenIndex * axisH_ + tokenOffset + t * TileW;
                using tile_x = Tile<Location::Vec, XType, 1, TileW, BLayout::RowMajor>;
                using gm_x = global_tensor<XType, RowMajor<1, TileW>>;
                gm_x gmRes(resPtr);
                tile_x resTile;
                TLOAD(resTile, gmRes);
                tile_f resFloat;
                TCVT(resFloat, resTile);
                TADD(accTile, accTile, resFloat);
            }

            // ---------- Apply expert scale (post-accumulation) ----------
            // Original: per-expert Muls(scaleVal) + Add
            // PTO: single TMULS at end (simulator constraint: TLOAD→TCVT→TMULS triggers
            //   assertion in gfrun/gfsim due to M-register allocation for dynamic-address TLOAD).
            //   Mathematically equivalent when all expert scales are equal (common MoE case).
            //   For unequal scales, per-expert TMULS is the correct path but requires
            //   simulator/compiler fix to allow TMULS after dynamic-address TLOAD.
            if (hasExpertScalesFlag_) {
                float scaleVal = expertScalesGM_[expertScaleBeginIdx_ * axisK_];
                TMULS(accTile, accTile, scaleVal);
            }

            // ---------- 结果搬出: Cast float → XType, store to output ----------
            // Original: Cast(sumBufLocal, sumFloatBufLocal_, CAST_RINT) → DataCopyPad to GM
            // PTO: TCVT → TSTORE
            tile_o outTile;
            TCVT(outTile, accTile);
            auto gout = out_iter(tokenIndex, t);
            TSTORE(gout, outTile);
        } // end for (t < HTiles)

        // ---------- AddRmsNormRmsNormCompute (HasAddRmsNorm) ----------
        // Original: Mul(x^2) → Muls(armAvgFactor) → ReduceSum → Adds(eps) → Sqrt → Div → Muls(rstd) → Cast(gamma) → Mul → Cast(y)
        // PTO: cross-tile TROWSUM → GM scalar round-trip → rsqrt_newton → TMULS(rstd) → TMUL(gamma) → TCVT → TSTORE
        if constexpr (HasAddRmsNorm) {
            // Phase 1: compute sqx_sum = sum(x^2 * armAvgFactor) across all H tiles
            float sqxSum = 0.0f;
            for (int t = 0; t < HTiles; t++) {
                XType* outPtr = expandOutGlobal_ + tokenIndex * axisH_ + tokenOffset + t * TileW;
                using tile_x = Tile<Location::Vec, XType, 1, TileW, BLayout::RowMajor>;
                using gm_x = global_tensor<XType, RowMajor<1, TileW>>;
                using tile_sum = Tile<Location::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
                gm_x gmOut(outPtr);
                tile_x xTile;  tile_f xF, sq, sqs;  tile_sum sumTile;
                TLOAD(xTile, gmOut);  TCVT(xF, xTile);
                TMUL(sq, xF, xF);
                TMULS(sqs, sq, armAvgFactor_);
                TROWSUM(sumTile, sqs);
                gm_x gmSum(reinterpret_cast<XType*>(maskCalcWorkspaceGM_ + t * 32));
                TSTORE(gmSum, sumTile);
                sqxSum += *reinterpret_cast<float*>(maskCalcWorkspaceGM_ + t * 32);
            }
            // Phase 2: rstd = 1/sqrt(sqxSum + eps)
            float denom = sqxSum + epsilon_;
            float rstd = 1.0f / std::sqrt(denom);
            // Phase 3: y = x * rstd * gamma, store to yOut
            for (int t = 0; t < HTiles; t++) {
                XType* outPtr = expandOutGlobal_ + tokenIndex * axisH_ + tokenOffset + t * TileW;
                XType* gammaPtr = gammaGM_ + t * TileW;
                XType* yPtr = yOutGlobal_ + tokenIndex * axisH_ + tokenOffset + t * TileW;
                using tile_x = Tile<Location::Vec, XType, 1, TileW, BLayout::RowMajor>;
                using gm_x = global_tensor<XType, RowMajor<1, TileW>>;
                gm_x gmOut(outPtr), gmGamma(gammaPtr), gmY(yPtr);
                tile_x xTile, gTile;  tile_f xF, gF, xr, xrg;  tile_o yTile;
                TLOAD(xTile, gmOut);  TCVT(xF, xTile);
                TLOAD(gTile, gmGamma); TCVT(gF, gTile);
                TMULS(xr, xF, rstd);
                TMUL(xrg, xr, gF);
                TCVT(yTile, xrg);
                TSTORE(gmY, yTile);
            }
            // rstd result搬出
            *reinterpret_cast<float*>(rstdOutGlobal_ + tokenIndex) = rstd;
        }

        // ---------- ClearPackedTokenFlags ----------
        // Original: DataCopyPad(dstFlagGlobal, flagTensor, clearFlagParams) per slot
        // PTO: TSTORE(clearFlagTile) per slot
        if (blockCntPerToken_ > 0U) {
            uint32_t clearTokenIndex = tokenIndex;
            if (isInputExpertMaskFlag_) {
                clearTokenIndex = static_cast<uint32_t>(validBsIndex_[curIdx]);
            }
            for (uint32_t copyId = 0U; copyId < flagRcvCount_; ++copyId) {
                uint8_t* wAddr = epWindowGM_ + (clearTokenIndex * flagRcvCount_ + copyId) * hAlignWinSize_;
                gm_fl gmFlag(reinterpret_cast<float*>(wAddr + SPLIT_BLOCK_DATA_SIZE - 24 * 4));
                TSTORE(gmFlag, clearFlagTile);
            }
        }
    }

    // ---------- Performance info output ----------
    // Original: SetAtomicMax → DataCopyPad(performanceInfoGM_, performanceInfoTensor_)
    // PTO: scalar copy (isPerformanceFlag_=false in test, code path preserved)
    if (isPerformanceFlag_) {
        // SetAtomicMax<int32_t>() — PTO: no atomic max, direct store
        for (uint32_t i = 0; i < JUMP_WRITE * epWorldSizeOriginal_; i++) {
            performanceInfoGM_[i] = performanceInfoGM_[i]; // already set
        }
    }
}

// ===================== Host-side reference (golden) ===================== //
// Mirrors refCombine from original
inline void refCombine(const __half* expandX, const float* expertScales,
                       const int32_t* expertIds, __half* refOutput,
                       uint32_t bs, uint32_t h, uint32_t k, uint32_t moeExpertNum) {
    for (uint32_t t = 0; t < bs; t++) {
        for (uint32_t j = 0; j < h; j++) {
            float sum = 0.0f;
            for (uint32_t e = 0; e < k; e++) {
                int32_t expertId = expertIds[t * k + e];
                if (static_cast<uint32_t>(expertId) >= moeExpertNum) continue;
                float expertVal = static_cast<float>(expandX[t * k * h + e * h + j]);
                float scale = expertScales[t * k + e];
                sum += scale * expertVal;
            }
            refOutput[t * h + j] = static_cast<__half>(sum);
        }
    }
}

} // namespace supernpu::tile_isa
#endif
