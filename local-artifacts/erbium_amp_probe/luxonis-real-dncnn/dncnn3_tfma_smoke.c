/* Multi-minion scaffolding smoke test: T0 of each minion writes a per-hart
 * magic and slot, then bench_barrier, then hart 0 writes a global summary. */
#include <stdint.h>
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define ACTIVE_HARTS 8u
#define BENCH_FLB    2u
#define BENCH_FCC    FCC_0
#define MAGIC        0x534D4F4Bu  /* 'SMOK' */

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
    }
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
    }
    return ptr;
}

static inline uint32_t mask_t0(void)
{
    uint32_t m = 0;
    for (uint32_t i = 0; i < ACTIVE_HARTS; i++) m |= 1u << i;
    return m;
}

int main(uintptr_t arg_area)
{
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;
    const uint32_t hart_id = raw_hart >> 1;

    uint8_t *base = (uint8_t *)buffer_base_from_args(arg_area);
    volatile uint32_t *slots = (volatile uint32_t *)(base + 0x10000u);

    /* Each minion stamps its slot before the barrier. */
    slots[hart_id * 4 + 0] = MAGIC;
    slots[hart_id * 4 + 1] = hart_id;
    slots[hart_id * 4 + 2] = get_minion_id();
    slots[hart_id * 4 + 3] = get_thread_id();
    evict((const void *)&slots[hart_id * 4], 16);
    WAIT_CACHEOPS;

    shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS, mask_t0(), 0u);

    if (hart_id == 0u) {
        volatile uint32_t *summary = (volatile uint32_t *)(base + 0x11000u);
        uint32_t mask = 0, count = 0;
        for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
            if (slots[h * 4] == MAGIC) {
                mask |= 1u << slots[h * 4 + 1];
                count++;
            }
        }
        summary[0] = MAGIC;
        summary[1] = mask;
        summary[2] = count;
        summary[3] = ACTIVE_HARTS;
        evict((const void *)summary, 64);
        WAIT_CACHEOPS;
    }
    shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS, mask_t0(), 0u);
    return 0;
}
