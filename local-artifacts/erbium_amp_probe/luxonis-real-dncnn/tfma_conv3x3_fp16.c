/*
 * FP16 TFMA-based 3x3 conv proof of correctness.
 *
 * TFMA fp16->fp32 path. ACOLS field=7 → actual=16 (fp16 path multiplies field+1
 * by 2). 9 dispatches (one per ky,kx) accumulating.
 *
 * Layout differs from FP32:
 *   - weight_pack_fp16: 9 × 16 lines × 32 fp16 lanes (only first 16 lanes used)
 *   - im2col_pack_fp16: 9 × 8 lines × 32 fp16 lanes (interleaved per
 *     sysemu's tensor_fma16a32 algorithm)
 *
 * Output: 16 OC × 16 P fp32, same as FP32 path (TFMA accumulator is fp32).
 */

#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

extern const unsigned char _binary_weight_pack_fp16_bin_start[];
extern const unsigned char _binary_im2col_pack_fp16_bin_start[];

extern char heap0_end[];

#define MAGIC 0x46313643u  /* 'F16C' */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
#define SUMMARY_OFFSET   0x010000u
#define WEIGHT_OFFSET    0x100000u   /* 9 × 16 × 32 fp16 = 9216 B */
#define IM2COL_OFFSET    0x110000u   /* 9 ×  8 × 32 fp16 = 4608 B */
#define OUTPUT_OFFSET    0x200000u   /* 16 OC × 16 P × 4 B = 1024 B */

#define K  3u

#define AROWS_FIELD 0xFu  /* 16 OC rows */
#define ACOLS_FIELD 0x7u  /* fp16: (7+1)*2 = 16 inner-dim */
#define BCOLS_FIELD 0x3u  /* (3+1)*4 = 16 P columns */

/* Per chunk (ky,kx):
 *   weight bytes = 16 lines × 64 B = 1024 B
 *   im2col bytes =  8 lines × 64 B =  512 B
 */
#define W_CHUNK_BYTES   1024u
#define IM_CHUNK_BYTES   512u

struct conv_summary {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t status;
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

    /* Copy linked .rodata blobs to 64-aligned heap-resident buffers. */
    volatile uint8_t *w_dst  = (volatile uint8_t *)(base + WEIGHT_OFFSET);
    volatile uint8_t *im_dst = (volatile uint8_t *)(base + IM2COL_OFFSET);
    for (uint32_t i = 0; i < 9u * 1024u; i++) {
        w_dst[i] = _binary_weight_pack_fp16_bin_start[i];
    }
    for (uint32_t i = 0; i < 9u * 512u; i++) {
        im_dst[i] = _binary_im2col_pack_fp16_bin_start[i];
    }
    evict((const void *)w_dst,  9u * 1024u);
    evict((const void *)im_dst, 9u * 512u);
    WAIT_CACHEOPS;

    const uint64_t t0 = read_cycles();

    for (uint32_t k = 0; k < K * K; k++) {
        const uintptr_t w_addr  = base + WEIGHT_OFFSET + k * W_CHUNK_BYTES;
        const uintptr_t im_addr = base + IM2COL_OFFSET + k * IM_CHUNK_BYTES;

        /* A: 16 lines (one per OC row), each 32 fp16 lanes. */
        tensor_load(0, 0, /*dst*/  0, 0, 0, w_addr,  0, 15, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);
        /* B: 8 lines (one per acols/2 = k-pair), each 32 fp16 lanes. */
        tensor_load(0, 0, /*dst*/ 16, 0, 0, im_addr, 0,  7, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);

        tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                   /*aoffset*/ 0, /*tenc_loc*/ 0, /*tenb_uns*/ 0, /*tena_uns*/ 0,
                   /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                   /*type*/ 1u, /*first_pass*/ (k == 0u));
        tensor_wait(TENSOR_FMA_WAIT);
    }

    const uint64_t t1 = read_cycles();
    s->dispatch_cycles = t1 - t0;

    /* Output is fp32 (TFMA fp16 accumulator is fp32) — same store as FP32 path. */
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
