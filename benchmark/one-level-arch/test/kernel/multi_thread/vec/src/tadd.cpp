#include "multi_thread/element_wise/tadd_multithread.hpp"

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

    float a_buf[kTileRows * kTileCols + 2 * ALIGN];
    float b_buf[kTileRows * kTileCols + 2 * ALIGN];
    float out_buf[kTileRows * kTileCols + 2 * ALIGN];
    float *a = (float *)(((uint64_t)a_buf & ALIGN_MASK) + ALIGN);
    float *b = (float *)(((uint64_t)b_buf & ALIGN_MASK) + ALIGN);
    float *out = (float *)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC_A_PATH CHK_DIR "/src_a.bin"
#define SRC_B_PATH CHK_DIR "/src_b.bin"
    readBinaryFile(SRC_A_PATH, (uint8_t *)a,
                   kTileRows * kTileCols * sizeof(float));
    readBinaryFile(SRC_B_PATH, (uint8_t *)b,
                   kTileRows * kTileCols * sizeof(float));
#else
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
#endif

    BENCHSTART;
    vec_multithread<kTileRows / 4, kTileCols>(out, a, b);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/vec_out.bin"
    writeBinaryFile(OUT_PATH, (uint8_t *)out,
                    kTileRows * kTileCols * sizeof(float));
#endif

    return 0;
}
