/* Tiny test: hart 0 zeros static_act1's halo row, evicts, exits.
 * Then dump should show row 0 of act1 = all zero. */
#include <stdint.h>
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define ACTIVE_HARTS 8u
#define ACT_W 322u
#define CH 64u
#define ACT_PIXELS (242u * 322u)

static float static_act1[ACT_PIXELS * CH] __attribute__((aligned(64)));

int main(uintptr_t arg_area)
{
    (void)arg_area;
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;
    const uint32_t hart_id = raw_hart >> 1;

    if (hart_id == 0u) {
        /* Zero row 0 of static_act1 (= 322*64 = 20608 floats = 82432 bytes) */
        volatile uint64_t *p = (volatile uint64_t *)static_act1;
        for (uint64_t i = 0; i < (322u * 64u) / 2u; i++) {
            p[i] = 0;
        }
        evict((const void *)static_act1, 322u * 64u * 4u);
        WAIT_CACHEOPS;
        FENCE;
    }
    shire_barrier(2u, FCC_0, ACTIVE_HARTS, 0xff, 0u);
    return 0;
}
