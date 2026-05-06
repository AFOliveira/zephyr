/*
 * TFMA-based 3x3 conv proof of correctness.
 *
 * Layer: input [IC=16, 6, 6] (4x4 output + 2-pixel halo on each axis),
 *        weights [OC=16, IC=16, K=3, K=3], output [OC=16, 4, 4].
 * Inputs are pre-im2col'd / pre-packed by gen_conv_inputs.py:
 *   im2col_pack: 9 chunks of [IC=16, 16 fp32] (144 cache lines total)
 *   weight_pack: 9 chunks of [OC=16, IC=16] (144 cache lines total)
 *
 * For each ky*3+kx ∈ [0..9): tensor_load weight chunk → scp[0..15],
 *                            tensor_load im2col chunk → scp[16..31],
 *                            tensor_fma() with first_pass = (k == 0).
 * Then tensor_store output [OC=16, 16] back to memory.
 * Host audits vs conv_ref.bin at 1e-5.
 */

#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

extern const unsigned char _binary_im2col_pack_bin_start[];
extern const unsigned char _binary_weight_pack_bin_start[];

extern char heap0_end[];

#define MAGIC 0x434F4E56u  /* 'CONV' */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
#define SUMMARY_OFFSET   0x010000u
#define IM2COL_OFFSET    0x100000u   /* 9*16*16*4 = 9216 B */
#define WEIGHT_OFFSET    0x110000u
#define OUTPUT_OFFSET    0x200000u   /* 16 OC × 16 pix × 4 = 1024 B */

#define IC      16u
#define OC      16u
#define K       3u
#define P       16u   /* output pixels (4x4 flattened) */

#define AROWS_FIELD 0xFu  /* 16 */
#define ACOLS_FIELD 0xFu  /* 16 */
#define BCOLS_FIELD 0x3u  /* 16 */
#define CHUNK_BYTES 1024u  /* 16 cache lines × 64 B */

struct conv_summary {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t status;     /* 1 if reached end, 0 otherwise */
    uint32_t reserved;
    uint64_t dispatch_cycles;
};

static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}

static inline uint64_t read_cycles(void)
{
    uint64_t v;
    __asm__ __volatile__("csrr %0, hpmcounter3" : "=r"(v));
    return v;
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
    volatile struct conv_summary *s =
        (volatile struct conv_summary *)(base + SUMMARY_OFFSET);
    s->magic = MAGIC;
    s->status = 0;
    s->dispatch_cycles = 0;

    /* Copy linked .rodata blobs into 64-aligned heap-resident buffers. */
    volatile uint8_t *im_dst = (volatile uint8_t *)(base + IM2COL_OFFSET);
    volatile uint8_t *w_dst  = (volatile uint8_t *)(base + WEIGHT_OFFSET);
    for (uint32_t i = 0; i < 9u * 1024u; i++) {
        im_dst[i] = _binary_im2col_pack_bin_start[i];
        w_dst[i]  = _binary_weight_pack_bin_start[i];
    }
    evict((const void *)im_dst, 9u * 1024u);
    evict((const void *)w_dst,  9u * 1024u);
    WAIT_CACHEOPS;

    const uint64_t t0 = read_cycles();

    /* For each (ky,kx) chunk in [0..9): load W and Im2col into scp, accumulate via TFMA. */
    for (uint32_t k = 0; k < K * K; k++) {
        const uintptr_t w_addr  = base + WEIGHT_OFFSET + k * CHUNK_BYTES;
        const uintptr_t im_addr = base + IM2COL_OFFSET + k * CHUNK_BYTES;

        /* Load weight chunk (16 cache lines = 16 OC rows) → scp[0..15] (A operand). */
        tensor_load(0, 0, /*dst*/  0, 0, 0, w_addr,  0, 15, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);
        /* Load im2col chunk (16 cache lines = 16 IC rows) → scp[16..31] (B operand). */
        tensor_load(0, 0, /*dst*/ 16, 0, 0, im_addr, 0, 15, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);

        /* C = W_k @ A_k (k=0) or C += W_k @ A_k (k>0). */
        tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                   /*aoffset*/ 0, /*tenc_loc*/ 0, /*tenb_uns*/ 0, /*tena_uns*/ 0,
                   /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                   /*type*/ 0u, /*first_pass*/ (k == 0u));
        tensor_wait(TENSOR_FMA_WAIT);
    }

    const uint64_t t1 = read_cycles();
    s->dispatch_cycles = t1 - t0;

    /* Store FREGs → output[OC=16, P=16]. cols=3 (4 × 128b = 16 fp32 per row),
     * srcinc=1 (FREGs advance by 2 per row, accounting for the inner advance). */
    tensor_store(/*reg_stride*/ 0, /*start_reg*/ 0, /*cols*/ 3, /*Arows*/ 0xF,
                 base + OUTPUT_OFFSET, /*coop_store*/ 0, /*stride*/ 64);
    tensor_wait(TENSOR_STORE_WAIT);

    evict((const void *)(base + OUTPUT_OFFSET), 1024u);
    WAIT_CACHEOPS;
    FENCE;

    s->status = 1;
    evict((const void *)s, sizeof(*s));
    WAIT_CACHEOPS;
    return 0;
}
