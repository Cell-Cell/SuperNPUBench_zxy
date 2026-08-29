#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

template <int Rows, int Cols>
void trowsum_multithread(float *out_ptr, float *in_ptr) {
    constexpr int kTileByteLimit = 4 * 1024;

    static_assert(Rows * Cols * sizeof(float) <= kTileByteLimit,
                  "each PE input tile must not exceed 4 KiB");
    static_assert(Rows * 8 * sizeof(float) <= kTileByteLimit,
                  "each PE row-sum tile must not exceed 4 KiB");

    using tileIn =
        Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    // TROWSUM produces one value per row. Keep 8 physical columns for Tile
    // alignment while exposing only column 0 as valid data.
    using tileSum =
        Tile<Location::Vec, float, Rows, 8, BLayout::RowMajor, Rows, 1>;
    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, 1>>;
    using itIn = global_iterator<gmIn, tileIn>;
    using itOut = global_iterator<gmOut, tileSum>;

    const uint32_t tid = get_thread_idx();
    const uint32_t in_offset = tid * Rows * Cols;
    const uint32_t out_offset = tid * Rows;

    itIn in_iter(in_ptr + in_offset);
    itOut out_iter(out_ptr + out_offset);

    tileIn tIn;
    tileSum tSum;
    auto src = in_iter(0, 0);
    auto dst = out_iter(0, 0);
    TLOAD(tIn, src);
    TROWSUM(tSum, tIn);
    TSTORE(dst, tSum);
}
