/* Smoke v2: like smoke v1 but adds the BIG bss array to reproduce the hang. */
#include <stdint.h>
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define ACTIVE_HARTS 8u
#define BENCH_FLB    2u
#define BENCH_FCC    FCC_0
#define MAGIC        0x534D4F4Bu
#define ACT_SIZE     ((242u * 322u * 64u))   /* same as TFMA kernel: 4,989,184 floats = 19.95 MB */

#ifndef SKIP_BSS
static float static_act0[ACT_SIZE] __attribute__((aligned(64)));
static float static_act1[ACT_SIZE] __attribute__((aligned(64)));
#endif

#ifdef ADD_VPU_MASK
static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}
#endif

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

struct slot64 {
    uint32_t magic, hart_id, minion_id, thread_id;
    uint32_t r[12];
};

int main(uintptr_t arg_area)
{
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;
    const uint32_t hart_id = raw_hart >> 1;

#ifdef ADD_VPU_MASK
    setM0MaskFF();
#endif

#ifndef SKIP_BSS
    /* Touch the bss arrays to ensure they exist (volatile to prevent gc) */
    volatile float *t0 = static_act0;
    volatile float *t1 = static_act1;
    (void)t0; (void)t1;
#endif

    uint8_t *base = (uint8_t *)buffer_base_from_args(arg_area);
    volatile struct slot64 *slots = (volatile struct slot64 *)(base + 0x10000u);

    slots[hart_id].magic = MAGIC;
    slots[hart_id].hart_id = hart_id;
    slots[hart_id].minion_id = get_minion_id();
    slots[hart_id].thread_id = get_thread_id();
    evict((const void *)&slots[hart_id], 64);
    WAIT_CACHEOPS;

    shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS, mask_t0(), 0u);

    if (hart_id == 0u) {
        volatile uint32_t *summary = (volatile uint32_t *)(base + 0x11000u);
        uint32_t mask = 0, count = 0;
        for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
            if (slots[h].magic == MAGIC) {
                mask |= 1u << slots[h].hart_id;
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
