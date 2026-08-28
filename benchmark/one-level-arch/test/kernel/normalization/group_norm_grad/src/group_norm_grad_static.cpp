#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "single_thread/normalization/group_norm_grad/group_norm_grad.hpp"

#ifndef DType
#define DType __half
#endif

// Same as dynamic group_norm_grad.cpp: N=2 C=16 G=4 HxW=16 tile_hw=8
#ifndef N_BATCH
#define N_BATCH 2
#endif
#ifndef C_CH
#define C_CH 16
#endif
#ifndef G_GRP
#define G_GRP 4
#endif
#ifndef HxW_SZ
#define HxW_SZ 16
#endif
#ifndef TILE_HW
#define TILE_HW 8
#endif

int main() {
    using dtype = DType;

    constexpr int64_t kElems = static_cast<int64_t>(N_BATCH) * C_CH * HxW_SZ;
    constexpr int64_t kWs =
        2 * static_cast<int64_t>(N_BATCH) * C_CH +
        2 * static_cast<int64_t>(N_BATCH) * G_GRP;

    dtype dy_buf[kElems];
    dtype x_buf[kElems];
    float mean_buf[N_BATCH * G_GRP];
    float rstd_buf[N_BATCH * G_GRP];
    dtype gamma_buf[C_CH];
    dtype dx_buf[kElems];
    dtype dgamma_buf[C_CH];
    dtype dbeta_buf[C_CH];
    float workspace_buf[kWs];

    dtype *dy = dy_buf;
    dtype *x = x_buf;
    float *mean = mean_buf;
    float *rstd = rstd_buf;
    dtype *gamma = gamma_buf;
    dtype *dx = dx_buf;
    dtype *dgamma = dgamma_buf;
    dtype *dbeta = dbeta_buf;
    float *workspace = workspace_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/dy.bin", (uint8_t *)dy,
                   static_cast<size_t>(kElems) * sizeof(dtype));
    readBinaryFile(CHK_DIR "/x.bin", (uint8_t *)x,
                   static_cast<size_t>(kElems) * sizeof(dtype));
    readBinaryFile(CHK_DIR "/mean.bin", (uint8_t *)mean,
                   static_cast<size_t>(N_BATCH) * G_GRP * sizeof(float));
    readBinaryFile(CHK_DIR "/rstd.bin", (uint8_t *)rstd,
                   static_cast<size_t>(N_BATCH) * G_GRP * sizeof(float));
    readBinaryFile(CHK_DIR "/gamma.bin", (uint8_t *)gamma,
                   static_cast<size_t>(C_CH) * sizeof(dtype));
#endif

    group_norm_grad<dtype, N_BATCH, C_CH, G_GRP, HxW_SZ, TILE_HW>(
        dy, x, mean, rstd, gamma, dx, dgamma, dbeta, workspace);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/dx.bin", (uint8_t *)dx,
                    static_cast<size_t>(kElems) * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dgamma.bin", (uint8_t *)dgamma,
                    static_cast<size_t>(C_CH) * sizeof(dtype));
    writeBinaryFile(CHK_DIR "/dbeta.bin", (uint8_t *)dbeta,
                    static_cast<size_t>(C_CH) * sizeof(dtype));
#endif
}
