#include "matmul/matmul_shared_reuseB.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "linx_group_runtime.h"

// Element data type for the A/B input tiles. Set via -DDTYPE=<token> from the
// Makefile (float / __bf16 / __half). The output C tile stays FP32 inside the
// kernel template (see matmul_shared.hpp).
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
#define tilM 32
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

using dtype = DTYPE;

struct MatmulReuseContext {
    dtype *src0;
    dtype *src1;
    float *dst;
};

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    (void)peId;
    MatmulReuseContext *context =
        static_cast<MatmulReuseContext *>(opaque);

    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        matmul_shared_reuseB<dtype, globM, globN, globK,
                             tilM, tilN, tilK>(
            context->dst + b * globM * globN,
            context->src0 + b * globM * globK,
            context->src1 + b * globK * globN);
    }
    BENCHEND;
    return 0;
}

int main() {
    constexpr int kPeNum = 4;

    static_assert(globM % kPeNum == 0,
                  "global M must be divisible by the PE count");

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

    MatmulReuseContext context{src0, src1, dst};
#ifdef LINX_GROUP_RUNTIME
    const int status = linx_group_run(&context);
#else
    const int status = __linx_group_worker_main(0, &context);
#endif

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst,
                    Batch * globM * globN * sizeof(float));
#endif

    return status;
}
