#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"

#ifndef PE_NUM
#define PE_NUM 1
#endif

#if PE_NUM == 1
#include "single_thread/normalization/rms_norm/rms_norm.hpp"
#else
#include "multi_thread/normalization/rms_norm/rms_norm.hpp"
#endif

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

// Same as dynamic rms_norm.cpp tiling_info {16,512,2,512}
#ifndef G_A
#define G_A 16
#endif
#ifndef G_R
#define G_R 512
#endif
#ifndef TILE_A
#define TILE_A 2
#endif
#ifndef TILE_R
#define TILE_R 512
#endif

int main() {
    using dtype = DType;

    static_assert(PE_NUM > 0, "PE_NUM must be positive");
    static_assert(G_A % PE_NUM == 0, "G_A must be divisible by PE_NUM");
    static_assert((G_A / PE_NUM) >= TILE_A, "PE-local G_A must cover one tile_a");

    constexpr int pe_a = G_A / PE_NUM;

    dtype input_buf[G_A * G_R];
    dtype output_buf[G_A * G_R];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                   static_cast<size_t>(G_A) * G_R * sizeof(dtype));
#endif

#if PE_NUM == 1
    rms_norm<dtype, G_A, G_R, TILE_A, TILE_R>(input, output, EPS);
#else
    // Full [G_A, G_R] buffers; kernel splits A with get_thread_idx().
    rms_norm<dtype, pe_a, G_A, G_R, TILE_A, TILE_R>(input, output, EPS);
#endif

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                    static_cast<size_t>(G_A) * G_R * sizeof(dtype));
#endif
}
