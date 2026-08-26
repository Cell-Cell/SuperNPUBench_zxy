#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

using namespace pto;

// Explicit PTO v0.58 CUBE-layout multi-thread matmul.
//
// A and B are cooperative Shared matrices. TMATMUL therefore consumes two
// complete SharedTile operands and publishes one PE-local accumulator:
//
//   Shared A group storage : tM x tK,        ordinary ND
//   Shared B storage       : tK x tN,       ordinary ND
//   Local D per PE         : PeM x tN,      CUBE_M16 or CUBE_M32
//
// CUBE_N8 is only the CELL layout for a Local Right primary. It is
// intentionally absent here because the right primary remains Shared.
template <typename DType, int gM, int gN, int gK,
          int tM, int tN, int tK>
void matmul_shared_cube_layout(float *c_ptr, DType *a_ptr, DType *b_ptr) {
    constexpr int kPeNum = 4;
    constexpr int kPeM = tM / kPeNum;

    static_assert(gM % tM == 0 && gN % tN == 0 && gK % tK == 0,
                  "global dimensions must be divisible by tile dimensions");
    static_assert(tM % kPeNum == 0,
                  "tM must be divisible by the four PEs");
    static_assert(kPeM > 0 && kPeM <= 32,
                  "the PE-local destination supports at most 32 rows");

    const uint32_t tid = get_thread_idx();

    using gmA = global_tensor<DType, RowMajor<gM, gK>>;
    using gmB = global_tensor<DType, RowMajor<gK, gN>>;
    using gmC = global_tensor<float, RowMajor<gM, gN>>;

    // A Shared tile describes the complete cooperative tM x tK matrix. The
    // hardware distributes its rows across the four PEs; kPeM is only the
    // row count of each PE-local accumulator.
    using tileAMatrix = SharedMatrixLeft<DType, tM, tK>;
    using tileBMatrix = SharedMatrixRight<DType, tK, tN>;
    using tileA = SharedTile<tileAMatrix>;
    using tileB = SharedTile<tileBMatrix>;
    using tileCM16 = CubeAccumulatorM16<float, kPeM, tN>;
    using tileCM32 = CubeAccumulatorM32<float, kPeM, tN>;
    using tileC = std::conditional_t<(kPeM <= 16), tileCM16, tileCM32>;

    using itA = global_iterator<gmA, tileAMatrix>;
    using itB = global_iterator<gmB, tileBMatrix>;
    using itC = global_iterator<gmC, tileC>;

    itA gIterA(a_ptr);
    itB gIterB(b_ptr);
    itC gIterC(c_ptr);

    constexpr int Mb = gM / tM;
    constexpr int Nb = gN / tN;
    constexpr int Kb = gK / tK;

#pragma clang loop unroll(full)
    for (int i = 0; i < Mb; ++i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < Nb; ++j) {
            tileC tC[Kb];
#pragma clang loop unroll(full)
            for (int k = 0; k < Kb; ++k) {
                auto gA = gIterA(i, k);
                auto gB = gIterB(k, j);
                tileA tA;
                tileB tB;
                TLOAD<tileAMatrix, 1>(tA, gA);
                TLOAD<tileBMatrix, 1>(tB, gB);
                if (k == 0)
                    TMATMUL(tC[k], tA, tB);
                else
                    TMATMUL_ACC(tC[k], tC[k - 1], tA, tB);
            }

            auto gC = gIterC(i * kPeNum + tid, j);
            TSTORE_CUBE(gC, tC[Kb - 1]);
        }
    }
}
