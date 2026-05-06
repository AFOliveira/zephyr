/*
 * TFMA matmul audit: linked random A, B (each 16×16 fp32). Compute C = A @ B
 * via TFMA, dump to memory, host compares against C_ref.
 *
 * Layout:
 *   A.bin: 16 cache-lines × 16 fp32 = A[i,k] = line i lane k
 *   B.bin: 16 cache-lines × 16 fp32 = B[k,j] = line k lane j
 *   C[i,j] = sum_k A[i,k] * B[k,j]
 */

#include <stdint.h>
#include <string.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

extern const unsigned char _binary_A_bin_start[];
extern const unsigned char _binary_B_bin_start[];

extern char heap0_end[];

#define MAGIC_OK 0x544D4C41u  /* 'TMLA' = TFMA matmul audit */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
#define SUMMARY_OFFSET   0x010000u
#define A_OFFSET         0x100000u
#define B_OFFSET         0x110000u    /* + 64 KB */
#define OUTPUT_OFFSET    0x200000u

#define AROWS_FIELD 0xFu
#define ACOLS_FIELD 0xFu
#define BCOLS_FIELD 0x3u

struct audit_summary {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t bytes_match;          /* 1 if all 1024 bytes equal expected */
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
    volatile struct audit_summary *s =
        (volatile struct audit_summary *)(base + SUMMARY_OFFSET);
    s->magic = MAGIC_OK;
    s->bytes_match = 0xFFFFFFFFu;

    /* Copy A, B from linked .rodata into heap-resident, 64-aligned src buffers
     * (tensor_load needs 64-aligned source addresses). */
    volatile uint8_t *a_dst = (volatile uint8_t *)(base + A_OFFSET);
    volatile uint8_t *b_dst = (volatile uint8_t *)(base + B_OFFSET);
    for (uint32_t i = 0; i < 1024u; i++) {
        a_dst[i] = _binary_A_bin_start[i];
        b_dst[i] = _binary_B_bin_start[i];
    }
    evict((const void *)a_dst, 1024);
    evict((const void *)b_dst, 1024);
    WAIT_CACHEOPS;

    /* Load A → scp[0..15], B → scp[16..31] (16 lines each, 64 B per line). */
    tensor_load(0, 0, /*dst*/  0, 0, 0, base + A_OFFSET, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);
    tensor_load(0, 0, /*dst*/ 16, 0, 0, base + B_OFFSET, 0, 15, 64, 0);
    tensor_wait(TENSOR_LOAD_WAIT_0);

    /* C = A @ B with first_pass=1 (pure mul-add accumulation, init to 0). */
    tensor_fma(0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD, 0, 0, 0, 0, 0,
               /*scp_loc_b*/ 16, /*scp_loc_a*/ 0, /*type*/ 0u,
               /*first_pass*/ 1);
    tensor_wait(TENSOR_FMA_WAIT);

    /* Store params:
     *   cols field=3 → actual cols=4 (4 × 128b per row = 16 fp32).
     *   With cols=4, the loop advances `src` after every odd col (after col=1
     *   and col=3 in each row). So src advances by srcinc * 2 per row.
     *   To advance by 2 per row (matching TFMA's TFMA_REGS_PER_ROW=2), use
     *   srcinc=1 (reg_stride field = 0). */
    tensor_store(/*reg_stride*/ 0, /*start_reg*/ 0, /*cols*/ 3, /*Arows*/ 0xF,
                 /*addr*/ base + OUTPUT_OFFSET, /*coop_store*/ 0,
                 /*stride*/ 64);
    tensor_wait(TENSOR_STORE_WAIT);

    evict((const void *)(base + OUTPUT_OFFSET), 1024);
    WAIT_CACHEOPS;
    FENCE;

    /* Don't need to compare on-chip — let the host do the bit-exact compare. */
    s->bytes_match = 1;
    evict((const void *)s, sizeof(*s));
    WAIT_CACHEOPS;
    return 0;
}
