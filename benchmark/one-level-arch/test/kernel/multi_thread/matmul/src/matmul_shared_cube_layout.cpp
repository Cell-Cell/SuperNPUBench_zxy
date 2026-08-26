#include "matmul/matmul_shared_cube_layout.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

#ifndef DTYPE
#define DTYPE float
#endif
#ifndef globM
#define globM 256
#endif
#ifndef globN
#define globN 256
#endif
#ifndef globK
#define globK 256
#endif
#ifndef tilM
#define tilM 64
#endif
#ifndef tilN
#define tilN 32
#endif
#ifndef tilK
#define tilK 32
#endif
#ifndef Batch
#define Batch 1
#endif
#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    using dtype = DTYPE;
    dtype src0p[Batch * globM * globK + 2 * ALIGN];
    dtype src1p[Batch * globK * globN + 2 * ALIGN];
    float dstp[Batch * globM * globN + 2 * ALIGN];

    dtype *src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
    dtype *src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
    float *dst = (float *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH CHK_DIR "/src0.bin"
#define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t *)src0,
                   Batch * globM * globK * sizeof(dtype));
    readBinaryFile(SRC1_PATH, (uint8_t *)src1,
                   Batch * globK * globN * sizeof(dtype));
#endif

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_shared_cube_layout<dtype, globM, globN, globK,
                                  tilM, tilN, tilK>(
            dst + b * globM * globN,
            src0 + b * globM * globK,
            src1 + b * globK * globN);
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst,
                    Batch * globM * globN * sizeof(float));
#endif
    return 0;
}
