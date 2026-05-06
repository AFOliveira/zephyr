/*
 * TFMA correctness test:
 *   1. Pre-fill heap buffer with all-ones FP32.
 *   2. tensor_load 16 cache-lines into scp[0..15] (A) and scp[16..31] (B).
 *      Each line = 64 B = 16 FP32s, all 1.0f.
 *   3. tensor_fma fp32, arows=16 acols=16 bcols=16, first_pass=1.
 *      Expected output: C[i,j] = sum_k 1.0 * 1.0 = 16 over k=0..15 = 16.0f.
 *   4. tensor_store the 16x16 fp32 result back to memory.
 *   5. evict, summary.
 *
 * Host audits: every output cell == 16.0f.
 */

#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

extern char heap0_end[];

#define MAGIC_OK 0x544F4B41u  /* 'TOKA' = TFMA OK */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
#define SUMMARY_OFFSET   0x010000u
#define INPUT_OFFSET     0x100000u   /* all-ones source for tensor_load */
#define OUTPUT_OFFSET    0x200000u   /* tensor_store target */

#define AROWS_FIELD 0xFu   /* arows = 16 */
#define ACOLS_FIELD 0xFu   /* acols = 16 */
#define BCOLS_FIELD 0x3u   /* bcols = (3+1)*4 = 16 */

struct correctness_summary {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t mismatches;        /* count of output cells != 16.0f */
    uint32_t first_mismatch_idx;
    float    first_mismatch_val;
    float    expected;
    uint32_t total_cells;
    uint32_t reserved;
};

static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}

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
    setM0MaskFF();

    const uintptr_t base = buffer_base_from_args(arg_area);
    volatile struct correctness_summary *s =
        (volatile struct correctness_summary *)(base + SUMMARY_OFFSET);
    s->magic = MAGIC_OK;
    s->hart_id = 0;
    s->mismatches = 0xFFFFFFFFu;     /* sentinel — overwritten on success */
    s->total_cells = 16 * 16;
    s->expected = 16.0f;

    /* 1. Fill 16 cache lines starting at INPUT_OFFSET with 1.0f. 16 lines = 256 fp32. */
    volatile float *src = (volatile float *)(base + INPUT_OFFSET);
    for (uint32_t i = 0; i < 16u * 16u; i++) src[i] = 1.0f;
    evict((const void *)src, 16u * 64u);
    WAIT_CACHEOPS;

    /* 2. tensor_load: load 16 lines into scp[0..15] (A) and scp[16..31] (B). */
    tensor_load(0, 0, /*scp_dst*/ 0,  0, 0, base + INPUT_OFFSET, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);
    tensor_load(0, 0, /*scp_dst*/ 16, 0, 0, base + INPUT_OFFSET, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* 3. TFMA: A * B → FREGs, with first_pass=1 (so it's a pure mul, no accum). */
    tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
               /*aoffset*/ 0, /*tenc_loc*/ 0, /*tenb_uns*/ 0, /*tena_uns*/ 0,
               /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
               /*opcode (type)*/ 0u, /*first_pass*/ 1);
    tensor_wait(TENSOR_FMA_WAIT);

    /* 4. tensor_store: 16 rows × 16 cols of fp32 (= 16 lines) → memory at OUTPUT_OFFSET.
     *
     *   reg_stride: 0 (FREGs packed sequentially)
     *   start_reg:  0 (FREG 0 = row 0)
     *   cols:       (bcols/8 - 1) = 1 → 16 fp32 per row (2 vregs of 8)
     *               TFMA writes row i to FREG i*TFMA_REGS_PER_ROW = i*2 (since
     *               16 fp32 = 2 vregs of 8 lanes each)
     *   Arows:      0xF = 16 rows
     *   stride:     64 (one row = one cache line)
     */
    /* reg_stride=1 → srcinc=2 (advance 2 FREGs per row, since TFMA writes
     *   16 cols across 2 FREGs of 8 lanes each).
     * cols=3 → actual cols=4 → 4 * 128b = 64 B = 16 fp32 per row. */
    tensor_store(/*reg_stride*/ 1, /*start_reg*/ 0, /*cols*/ 3, /*Arows*/ 0xF,
                 /*addr*/ base + OUTPUT_OFFSET, /*coop_store*/ 0, /*stride*/ 64);
    tensor_wait(TENSOR_STORE_WAIT);

    /* 5. Read back the 16x16 output (in DRAM/L2 — evict to be sure visible to us). */
    evict((const void *)(base + OUTPUT_OFFSET), 16u * 64u);
    WAIT_CACHEOPS;
    FENCE;

    volatile float *out = (volatile float *)(base + OUTPUT_OFFSET);
    uint32_t mismatches = 0;
    uint32_t first_idx = 0xFFFFFFFFu;
    float first_val = 0.0f;
    for (uint32_t k = 0; k < 16u * 16u; k++) {
        float v = out[k];
        if (v != 16.0f) {
            if (mismatches == 0u) {
                first_idx = k;
                first_val = v;
            }
            mismatches++;
        }
    }

    s->mismatches = mismatches;
    s->first_mismatch_idx = first_idx;
    s->first_mismatch_val = first_val;

    evict((const void *)s, sizeof(*s));
    WAIT_CACHEOPS;
    return 0;
}
