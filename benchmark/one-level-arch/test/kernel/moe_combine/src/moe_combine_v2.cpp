#include <common/pto_tileop.hpp>
#include <cstdint>
#include "moe_combine/moe_combine_v2.hpp"

using namespace supernpu::tile_isa;

constexpr int kBS = 8;
constexpr int kH = 128;
constexpr int kK = 4;
constexpr int kMoeExpertNum = 4;
constexpr int kSharedExpertNum = 0;
constexpr int kTileW = 64;

constexpr int kSlotCount = kK + kSharedExpertNum;
constexpr int kHBytes = kH * sizeof(__half);
constexpr int kBlockCntPerToken = (kHBytes + 479) / 480;
constexpr int kAlignWinSize = kBlockCntPerToken * 512;
constexpr uint64_t kTotalWinSize = static_cast<uint64_t>(kBS) * kSlotCount * kAlignWinSize;
constexpr int kMoeSendNum = 1 * kMoeExpertNum;
constexpr uint64_t kStatusSize = 820UL * 1024UL;
constexpr uint64_t kWorkspaceSize = 10UL * 1024UL;

static __half expandX[kBS * kK * kH] __attribute__((aligned(4096)));
static float expertScales[kBS * kK] __attribute__((aligned(4096)));
static int32_t expertIds[kBS * kK] __attribute__((aligned(4096)));
static int32_t expandIdx[kBS * kK * 3] __attribute__((aligned(4096)));
static uint32_t epSendCount[kMoeSendNum] __attribute__((aligned(4096)));
static __half output[kBS * kH] __attribute__((aligned(4096)));
static uint8_t windowBuf[kTotalWinSize] __attribute__((aligned(4096)));
static uint8_t statusBuf[kStatusSize] __attribute__((aligned(4096)));
static uint8_t workspace[kWorkspaceSize] __attribute__((aligned(4096)));

int main() {
    epSendCount[0] = 0; epSendCount[1] = 0; epSendCount[2] = 0; epSendCount[3] = kBS * kK;
    for (int i = 0; i < kBS * kK; i++) {
        expertIds[i] = i % kMoeExpertNum;
        expertScales[i] = 0.25f;
        expandIdx[i * 3] = 0;
        expandIdx[i * 3 + 1] = i / kK;
        expandIdx[i * 3 + 2] = i % kK;
    }
    for (int i = 0; i < kBS * kK * kH; i++) {
        expandX[i] = static_cast<__half>(static_cast<float>(i % 100) * 0.01f);
    }
    moe_combine_v2<__half, __half, int32_t, kBS, kH, kK, kSharedExpertNum, kMoeExpertNum,
                   1, 0, kTileW>(
        expandX, expertScales, expertIds, expandIdx, epSendCount,
        output, windowBuf, statusBuf, workspace);

    static __half refOutput[kBS * kH];
    refCombine(expandX, expertScales, expertIds, refOutput, kBS, kH, kK, kMoeExpertNum);

    int ret = 0;
    for (int i = 0; i < kBS * kH; i++) {
        float a = static_cast<float>(output[i]);
        float b = static_cast<float>(refOutput[i]);
        float diff = a - b;
        if (diff < 0.0f) diff = -diff;
        if (diff > 0.01f) { ret = 1; break; }
    }
    return ret;
}
