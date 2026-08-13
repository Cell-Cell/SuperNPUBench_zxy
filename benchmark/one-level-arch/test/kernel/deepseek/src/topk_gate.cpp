#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/moe/topk_gate.hpp"
using namespace pto;
using namespace supernpu::tile_isa;

static float scores[1024] __attribute__((aligned(4096))) = {};
static std::int32_t topk_idx[256] __attribute__((aligned(4096))) = {};

int main() {
    topk_gate<16, 32, 4>(scores, topk_idx);
    return 0;
}
