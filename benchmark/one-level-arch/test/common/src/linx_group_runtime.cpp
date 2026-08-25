#include "linx_group_runtime.h"

#include <stdint.h>

namespace {

constexpr uint32_t kPeCount = 4;

struct alignas(64) GroupControl {
    volatile uint32_t ready;
    volatile uint32_t done0;
    volatile uint32_t done1;
    volatile uint32_t done2;
    volatile uint32_t done3;
    volatile int32_t status0;
    volatile int32_t status1;
    volatile int32_t status2;
    volatile int32_t status3;
    void *context;
};

GroupControl control = {};

uint32_t CurrentPe()
{
#ifdef __linx
    uint32_t peId;
    asm volatile("SSRGET 0x802, ->%0" : "=r"(peId));
    return peId;
#else
    return 0;
#endif
}

inline void CompilerBarrier()
{
    asm volatile("" : : : "memory");
}

inline uint32_t LoadPublished(const volatile uint32_t *value)
{
    const uint32_t data = *value;
    CompilerBarrier();
    return data;
}

inline void StorePublished(volatile uint32_t *value, uint32_t data)
{
    CompilerBarrier();
    *value = data;
}

} // namespace

extern "C" int linx_group_run(void *context)
{
    if (CurrentPe() != 0) {
        return -1;
    }

    control.status0 = 0;
    control.status1 = 0;
    control.status2 = 0;
    control.status3 = 0;
    control.done0 = 0;
    control.done1 = 0;
    control.done2 = 0;
    control.done3 = 0;
    control.context = context;
    StorePublished(&control.ready, 1);

    control.status0 = __linx_group_worker_main(0, context);
    StorePublished(&control.done0, 1);

    while (LoadPublished(&control.done1) == 0) {
    }
    while (LoadPublished(&control.done2) == 0) {
    }
    while (LoadPublished(&control.done3) == 0) {
    }

    if (control.status0 != 0) {
        return control.status0;
    }
    if (control.status1 != 0) {
        return control.status1;
    }
    if (control.status2 != 0) {
        return control.status2;
    }
    if (control.status3 != 0) {
        return control.status3;
    }
    return 0;
}

extern "C" __attribute__((noreturn, used, visibility("default")))
void __linx_group_worker_start(void)
{
    const uint32_t peId = CurrentPe();
    if (peId == 0 || peId >= kPeCount) {
        for (;;) {
        }
    }
    while (LoadPublished(&control.ready) == 0) {
    }

    const int status = __linx_group_worker_main(peId, control.context);
    if (peId == 1) {
        control.status1 = status;
        StorePublished(&control.done1, 1);
    } else if (peId == 2) {
        control.status2 = status;
        StorePublished(&control.done2, 1);
    } else {
        control.status3 = status;
        StorePublished(&control.done3, 1);
    }

    for (;;) {
    }
}
