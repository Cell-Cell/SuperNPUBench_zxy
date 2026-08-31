// Minimal repro: 8 consecutive i32 zero stores -> merged BLK_TSTORE (32B) ->
// "Cannot select: v4i64 = BUILD_VECTOR" backend crash (LinxV5, -mlxbc -O1/-O2).
// 9 lines of user code; no tile APIs required (get_thread_idx only sets the PE base).
#include <common/pto_tileop.hpp>
#include <cstdint>

static volatile int32_t g_buf[32] __attribute__((aligned(4096)));

int main() {
    const uint32_t tid = get_thread_idx();
    for (uint32_t lc = 0U; lc < 4U; ++lc) {
        const uint32_t core = tid * 4U + lc;
        for (uint32_t e = 0U; e < 2U; ++e) {
            g_buf[core * 2U + e] = 0;
        }
    }
    return g_buf[0] + g_buf[7];
}
