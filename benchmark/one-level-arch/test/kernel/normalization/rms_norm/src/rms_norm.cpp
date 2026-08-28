#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"

#ifndef PE_NUM
#define PE_NUM 1
#endif

#if PE_NUM == 1
#include "single_thread/normalization/rms_norm/rms_norm_pto.hpp"
#else
#include "multi_thread/normalization/rms_norm/rms_norm_pto.hpp"
#endif

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

int main() {
    using dtype = DType;

    static_assert(PE_NUM > 0, "PE_NUM must be positive");
    static_assert(16 % PE_NUM == 0, "g_a=16 must be divisible by PE_NUM");

    // tiling_info: {g_a, g_r, tile_a, tile_r} — host-visible full A.
    // 4PE: split A across PEs like multi_thread/vec (get_thread_idx).
    int64_t tiling_info[4] = {16, 512, 1, -1};

    const int64_t g_a = tiling_info[0];
    const int64_t g_r = tiling_info[1];
    constexpr int64_t pe_a = 16 / PE_NUM;

    dtype input_buf[16 * 512];
    dtype output_buf[16 * 512];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                   static_cast<size_t>(g_a) * g_r * sizeof(dtype));
#endif

#if PE_NUM == 1
    rms_norm<dtype>(input, tiling_info, output, EPS);
#else
    const uint32_t tid = get_thread_idx();
    int64_t tiling_pe[4] = {pe_a, g_r, tiling_info[2], tiling_info[3]};
    rms_norm<dtype>(input + tid * pe_a * g_r, tiling_pe,
                    output + tid * pe_a * g_r, EPS);
#endif

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                    static_cast<size_t>(g_a) * g_r * sizeof(dtype));
#endif
}
