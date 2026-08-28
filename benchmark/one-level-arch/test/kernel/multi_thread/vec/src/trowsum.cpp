#include "multi_thread/reduction/trowsum_multithread.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

#ifndef TileRows
#define kTileRows 16
#else
#define kTileRows TileRows
#endif

#ifndef TileCols
#define kTileCols 16
#else
#define kTileCols TileCols
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    static_assert(kTileRows % 4 == 0,
                  "TileRows must be divisible by the four PE threads");

    float in_buf[kTileRows * kTileCols + 2 * ALIGN];
    float out_buf[kTileRows + 2 * ALIGN];
    float *in = (float *)(((uint64_t)in_buf & ALIGN_MASK) + ALIGN);
    float *out = (float *)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC_PATH CHK_DIR "/src.bin"
    readBinaryFile(SRC_PATH, (uint8_t *)in,
                   kTileRows * kTileCols * sizeof(float));
#else
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        in[i] = 1.0f;
    }
#endif

    BENCHSTART;
    trowsum_multithread<kTileRows / 4, kTileCols>(out, in);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/trowsum_out.bin"
    writeBinaryFile(OUT_PATH, (uint8_t *)out,
                    kTileRows * sizeof(float));
#endif

    return 0;
}
