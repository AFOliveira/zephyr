/*
 * Minimal TFMA throughput probe for Erbium / ET-SoC1.
 *
 * Bisect-friendly build:
 *   -DPROBE_STAGE=0   summary-only (no tensor instrs) — sanity check the scaffold
 *   -DPROBE_STAGE=1   add tensor_load + tensor_wait
 *   -DPROBE_STAGE=2   add VPU mask (mov.m.x m0)
 *   -DPROBE_STAGE=3   full TFMA throughput measurement
 */

#include <stdint.h>

#ifndef PROBE_STAGE
#define PROBE_STAGE 3
#endif

#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#if PROBE_STAGE >= 1
#include "erbium/isa/tensors.h"
#endif

extern char heap0_end[];

#define TFMA_PROBE_MAGIC 0x54464D41u  /* 'TFMA' */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
/* Past code (~4 KB), data (~0), and stack (~4 KB) — well into heap0 space.
 * Within the 16 MB dump window. */
#define SUMMARY_OFFSET   0x010000u

#ifndef TFMA_AROWS
#define TFMA_AROWS 0xFu
#endif
#ifndef TFMA_ACOLS
#define TFMA_ACOLS 0xFu
#endif
#ifndef TFMA_BCOLS
#define TFMA_BCOLS 0x3u
#endif
#ifndef TFMA_TYPE
#define TFMA_TYPE  0u
#endif
#ifndef TFMA_ITERS
#define TFMA_ITERS 1000u
#endif

#define VLENW 8u

struct tfma_summary {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t arows;
    uint32_t acols;
    uint32_t bcols;
    uint32_t type;
    uint32_t iters;
    uint32_t probe_stage;
    uint64_t cycles;
    uint64_t macs_total;
    uint64_t macs_per_iter;
    uint64_t cycles_per_iter_x1000;
};

static inline uint64_t read_cycles(void)
{
    /* hpmcounter3 (0xC03) is enabled in U-mode by Erbium's mcounteren
     * (proven by dncnn3-pmc/pmc_snapshot). cycle (0xC00) traps. */
    uint64_t v;
    __asm__ __volatile__("csrr %0, hpmcounter3" : "=r"(v));
    return v;
}

#if PROBE_STAGE >= 2
static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}
#endif

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - DUMP_REGION_SIZE;
    }
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - DUMP_REGION_SIZE;
    }
    return ptr;
}

int main(uintptr_t arg_area)
{
    if (get_hart_id() != 0u) return 0;

    const uintptr_t base = buffer_base_from_args(arg_area);
    volatile struct tfma_summary *s =
        (volatile struct tfma_summary *)(base + SUMMARY_OFFSET);

    /* Eagerly write the magic so we can confirm the kernel reached this point
     * even if a later instruction faults. */
    s->magic = TFMA_PROBE_MAGIC;
    s->hart_id = 0;
    s->arows = TFMA_AROWS;
    s->acols = TFMA_ACOLS;
    s->bcols = TFMA_BCOLS;
    s->type = TFMA_TYPE;
    s->iters = TFMA_ITERS;
    s->probe_stage = PROBE_STAGE;

    uint64_t cycles = 0;
    uint64_t macs_total = 0;
    uint64_t macs_per_iter = (uint64_t)TFMA_AROWS * (uint64_t)TFMA_ACOLS *
                             VLENW * (uint64_t)TFMA_BCOLS;

#if PROBE_STAGE >= 1
    const uintptr_t scp_src = (base + 0x100000ull) & ~0x3Full;
    tensor_load(0, 0, 0,  0, 0, scp_src, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);
    tensor_load(0, 0, 16, 0, 0, scp_src, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);
    s->probe_stage = (uint32_t)(PROBE_STAGE) | 0x10000u;  /* mark "tensor_load done" */
#endif

#if PROBE_STAGE >= 2
    setM0MaskFF();
    s->probe_stage = (uint32_t)(PROBE_STAGE) | 0x20000u;  /* "VPU mask set" */
#endif

#if PROBE_STAGE >= 3
    /* Warm-up TFMA. */
    tensor_fma(0, TFMA_BCOLS, TFMA_AROWS, TFMA_ACOLS, 0, 0, 0, 0,
               0, 16, 0, TFMA_TYPE, 1);
    tensor_wait(TENSOR_FMA_WAIT);
    s->probe_stage = (uint32_t)(PROBE_STAGE) | 0x40000u;  /* "first TFMA OK" */

    const uint64_t t0 = read_cycles();
    for (uint32_t i = 0; i < TFMA_ITERS; ++i) {
        tensor_fma(0, TFMA_BCOLS, TFMA_AROWS, TFMA_ACOLS, 0, 0, 0, 0,
                   0, 16, 0, TFMA_TYPE, (i == 0));
    }
    tensor_wait(TENSOR_FMA_WAIT);
    const uint64_t t1 = read_cycles();
    cycles = t1 - t0;
    macs_total = macs_per_iter * (uint64_t)TFMA_ITERS;
#else
    /* Just measure mcycle around an empty loop for sanity. */
    const uint64_t t0 = read_cycles();
    for (uint32_t i = 0; i < TFMA_ITERS; ++i) {
        __asm__ __volatile__("nop\n");
    }
    const uint64_t t1 = read_cycles();
    cycles = t1 - t0;
    macs_total = 0;
#endif

    s->cycles = cycles;
    s->macs_total = macs_total;
    s->macs_per_iter = macs_per_iter;
    s->cycles_per_iter_x1000 =
        (TFMA_ITERS > 0u) ? (cycles * 1000u) / TFMA_ITERS : 0u;
    s->probe_stage |= 0x80000u;  /* "summary written" */

    /* L1D is non-coherent — eject so --dump_after sees the writes. */
    evict((const void *)s, sizeof(*s));
    WAIT_CACHEOPS;
    return 0;
}
