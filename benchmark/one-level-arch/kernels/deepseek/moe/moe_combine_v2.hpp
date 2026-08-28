#ifndef SUPERNPU_MOE_COMBINE_V2_HPP
#define SUPERNPU_MOE_COMBINE_V2_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

template <typename DTypeIn, int NumExpanded, int H, int K, int TileW>
void pack_phase(DTypeIn *expandX, std::int32_t *expandIdx,
                DTypeIn *windowData, float *windowFlag)
{
    constexpr int kTiles = H / TileW;
    using namespace pto;

    using gm_x    = global_tensor<DTypeIn, RowMajor<NumExpanded, H>>;
    using gm_win  = global_tensor<DTypeIn, RowMajor<NumExpanded, H>>;
    using gm_flag = global_tensor<float,   RowMajor<NumExpanded, TileW>>;
    using tile_d  = Tile<Location::Vec, DTypeIn, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,   1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_win  = global_iterator<gm_win,  tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    it_x    x_iter(expandX);
    it_win  win_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = 0; tk < NumExpanded; tk++) {
        int tokenId = expandIdx[tk * 3 + 1];
        int topkId  = expandIdx[tk * 3 + 2];
        int slot    = tokenId * K + topkId;

        for (int t = 0; t < kTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tk, t);
            TLOAD(xq, gx);

            tile_d sync_d;
            TMOV(sync_d, xq);

            tile_f flagTile;
            TEXPANDS(flagTile, 1.0f);

            auto gw = win_iter(slot, t);
            TSTORE(gw, xq);

            auto gf = flag_iter(slot, t);
            TSTORE(gf, flagTile);
        }
    }
}

template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW>
void combine_phase(float *expertScales, DTypeIn *windowData, float *windowFlag,
                   float *predBuf, DTypeOut *out)
{
    constexpr int kTiles = H / TileW;
    using namespace pto;

    using gm_win  = global_tensor<DTypeIn,  RowMajor<NumExpanded, H>>;
    using gm_flag = global_tensor<float,    RowMajor<NumExpanded, TileW>>;
    using gm_pred = global_tensor<float,    RowMajor<NumExpanded, TileW>>;
    using gm_out  = global_tensor<DTypeOut, RowMajor<BS, H>>;
    using tile_d  = Tile<Location::Vec, DTypeIn,  1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,   1, TileW, BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, DTypeOut, 1, TileW, BLayout::RowMajor>;
    using it_win  = global_iterator<gm_win,  tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    using it_pred = global_iterator<gm_pred, tile_f>;
    using it_out  = global_iterator<gm_out,  tile_o>;

    it_win  win_iter(windowData);
    it_flag flag_iter(windowFlag);
    it_pred pred_iter(predBuf);
    it_out  out_iter(out);

    for (int n = 0; n < BS; n++) {
        for (int t = 0; t < kTiles; t++) {
            tile_f refFlag;
            TEXPANDS(refFlag, 1.0f);

            for (int k = 0; k < K; k++) {
                int slot = n * K + k;

                tile_f flagTile;
                auto gf = flag_iter(slot, t);
                TLOAD(flagTile, gf);

                tile_f sync_f;
                TMOV(sync_f, flagTile);

                tile_f predTile;
                TCMP<CmpMode::EQ>(predTile, flagTile, refFlag);

                auto gp = pred_iter(slot, t);
                TSTORE(gp, predTile);
            }

            for (int k = 0; k < K; k++) {
                int slot = n * K + k;
                if (predBuf[slot * TileW] < 0.5f) break;
            }

            tile_f acc;
            TEXPANDS(acc, 0.0f);

            for (int k = 0; k < K; k++) {
                int slot = n * K + k;
                float scale = expertScales[n * K + k];

                tile_d xq;
                auto gw = win_iter(slot, t);
                TLOAD(xq, gw);

                tile_d sync_d;
                TMOV(sync_d, xq);

                tile_f xf;
                TCVT(xf, xq);
                TMULS(xf, xf, scale);
                TADD(acc, acc, xf);
            }

            tile_o oq;
            TCVT(oq, acc);
            auto gout = out_iter(n, t);
            TSTORE(gout, oq);
        }

        tile_f zeroFlag;
        TEXPANDS(zeroFlag, 0.0f);
        for (int k = 0; k < K; k++) {
            int slot = n * K + k;
            auto gf = flag_iter(slot, 0);
            TSTORE(gf, zeroFlag);
        }
    }
}

template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW = 128>
void moe_combine_v2(DTypeIn *expandX, float *expertScales,
                      std::int32_t *expandIdx,
                      DTypeIn *windowData, float *windowFlag,
                      uint32_t *windowState, float *predBuf,
                      DTypeOut *out)
{
    static_assert(BS > 0 && H > 0 && K > 0 && NumExpanded > 0, "dim must be positive");
    static_assert(TileW % 8 == 0, "TileW must be multiple of 8");

    uint32_t dataState = windowState[0];
    windowState[0] = (dataState == 0) ? 1 : 0;
    windowState[1] = 1;
    windowState[2] = 0;

    pack_phase<DTypeIn, NumExpanded, H, K, TileW>(
        expandX, expandIdx, windowData, windowFlag);

    combine_phase<DTypeIn, DTypeOut, BS, H, K, NumExpanded, TileW>(
        expertScales, windowData, windowFlag, predBuf, out);

    windowState[4] = BS;
}

} // namespace supernpu::tile_isa
#endif
