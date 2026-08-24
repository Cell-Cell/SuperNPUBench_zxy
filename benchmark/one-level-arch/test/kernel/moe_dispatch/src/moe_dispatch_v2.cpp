#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cmath>
#include "moe_dispatch/moe_dispatch_v2.hpp"

using namespace supernpu::tile_isa;

using dtype = float;

constexpr int kBS = 8;
constexpr int kH = 128;
constexpr int kK = 4;
constexpr int kMoeExpertNum = 4;
constexpr int kTileW = 64;

constexpr int kSlotCount = kBS * kK;
constexpr int kHOutSize = kH * sizeof(dtype);
constexpr int kHOutSizeAlign = ((kHOutSize + 31) / 32) * 32;
constexpr int kTokenQuantAlign = kHOutSizeAlign / sizeof(int32_t);
constexpr int kBlockCntPerToken = (kHOutSizeAlign + 479) / 480;
constexpr int kHCommuSize = kBlockCntPerToken * 512;
constexpr int kExpertPerSizeOnWin = kBS * kHCommuSize;
constexpr uint64_t kStateSize = 1024UL * 1024UL;
constexpr uint64_t kTotalWinSize = kStateSize + 128UL * 1024UL;
constexpr uint64_t kWorkspaceSize = 10UL * 1024UL;

static dtype x[kBS * kH] __attribute__((aligned(4096))) = {};
static int32_t expertIds[kBS * kK] __attribute__((aligned(4096))) = {};
static float expertScales[kBS * kK] __attribute__((aligned(4096))) = {};
static dtype expandXOut[kBS * kK * kH] __attribute__((aligned(4096))) = {};
static int32_t expandIdxOut[kBS * kK * 3] __attribute__((aligned(4096))) = {};
static float expandScalesOut[kBS * kK] __attribute__((aligned(4096))) = {};
static int32_t sendCountsOut[kMoeExpertNum] __attribute__((aligned(4096))) = {};
static int64_t expertTokenNumsOut[kMoeExpertNum] __attribute__((aligned(4096))) = {};
static uint8_t windowBuf[kTotalWinSize] __attribute__((aligned(4096))) = {};
static uint8_t workspace[kWorkspaceSize] __attribute__((aligned(4096))) = {};

int main() {
    for (int i = 0; i < kBS * kH; i++) {
        x[i] = static_cast<dtype>(static_cast<float>(i) * 0.1f);
    }
    for (int i = 0; i < kBS * kK; i++) {
        expertIds[i] = i % kMoeExpertNum;
        expertScales[i] = 0.25f;
    }
    moe_dispatch_v2<dtype, dtype, int32_t, kBS, kH, kK, kMoeExpertNum,
                    1, 0, 0, kTileW>(
        x, expertIds, expertScales,
        expandXOut, expandIdxOut, expandScalesOut,
        sendCountsOut, expertTokenNumsOut,
        windowBuf, workspace);

    bool ok = refDispatchCheck(x, expertIds, expertScales,
                               expandXOut, expandIdxOut, expandScalesOut,
                               sendCountsOut, expertTokenNumsOut,
                               kBS, kH, kK, kMoeExpertNum);
    return ok ? 0 : 1;
}
