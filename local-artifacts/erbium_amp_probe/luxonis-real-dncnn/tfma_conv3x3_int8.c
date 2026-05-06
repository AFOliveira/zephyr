/*
 * INT8 TFMA-based 3x3 conv proof of correctness.
 *
 * tensor_ima8a32: int8 inputs, int32 accumulator.  ACOLS field=3 → actual=16
 * (INT8 path multiplies field+1 by 4, so field=3 = actual=16). 9 dispatches,
 * accumulating across (ky,kx) chunks.
 *
 * Layout:
 *   weight_pack_int8: 9 × 16 lines × 64 int8 lanes (16 used per row, lanes 0..15)
 *   im2col_pack_int8: 9 × 4 lines × 64 int8 lanes (k0=0..3, interleaved 4-deep)
 *
 * Output: 16 OC × 16 P int32. Host scales by w_scale * a_scale to get fp32.
 */

#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/tensors.h"

extern const unsigned char _binary_weight_pack_int8_bin_start[];
extern const unsigned char _binary_im2col_pack_int8_bin_start[];

extern char heap0_end[];

#define MAGIC 0x49383843u  /* 'I88C' */
#define DUMP_REGION_SIZE (16u * 1024u * 1024u)
#define SUMMARY_OFFSET   0x010000u
#define WEIGHT_OFFSET    0x100000u   /* 9 × 16 × 64 = 9216 B */
#define IM2COL_OFFSET    0x110000u   /* 9 ×  4 × 64 = 2304 B */
#define OUTPUT_OFFSET    0x200000u   /* 16 × 16 × 4 = 1024 B */

#define K  3u

#define AROWS_FIELD 0xFu  /* 16 */
#define ACOLS_FIELD 0x3u  /* INT8: (3+1)*4 = 16 */
#define BCOLS_FIELD 0x3u  /* (3+1)*4 = 16 */

#define W_CHUNK_BYTES   1024u
#define IM_CHUNK_BYTES   256u

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

    volatile uint8_t *w_dst  = (volatile uint8_t *)(base + WEIGHT_OFFSET);
    volatile uint8_t *im_dst = (volatile uint8_t *)(base + IM2COL_OFFSET);
    for (uint32_t i = 0; i < 9u * 1024u; i++) {
        w_dst[i] = _binary_weight_pack_int8_bin_start[i];
    }
    for (uint32_t i = 0; i < 9u * 256u; i++) {
        im_dst[i] = _binary_im2col_pack_int8_bin_start[i];
    }
    evict((const void *)w_dst,  9u * 1024u);
    evict((const void *)im_dst, 9u * 256u);
    WAIT_CACHEOPS;

    const uint64_t t0 = read_cycles();

    for (uint32_t k = 0; k < K * K; k++) {
        const uintptr_t w_addr  = base + WEIGHT_OFFSET + k * W_CHUNK_BYTES;
        const uintptr_t im_addr = base + IM2COL_OFFSET + k * IM_CHUNK_BYTES;

        /* A: 16 lines (one per OC), int8 packed. */
        tensor_load(0, 0, /*dst*/  0, 0, 0, w_addr,  0, 15, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);
        /* B: 4 lines (one per k-quad), int8 packed. */
        tensor_load(0, 0, /*dst*/ 16, 0, 0, im_addr, 0,  3, 64, 0);
        tensor_wait(TENSOR_LOAD_WAIT_0);

        /* tfma_type_int8 = 3. tena_unsigned/tenb_unsigned = 0 → signed int8.
         * INT8 writes results to TenC by default; set tenc_loc=1 (= tenc2rf
         * in sysemu) on the LAST dispatch to copy TenC → FREGs for tensor_store. */
        const uint32_t last = (k == K * K - 1u);
        tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                   /*aoffset*/ 0, /*tenc_loc*/ last, /*tenb_uns*/ 0, /*tena_uns*/ 0,
                   /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                   /*type*/ 3u, /*first_pass*/ (k == 0u));
        tensor_wait(TENSOR_FMA_WAIT);
    }

    const uint64_t t1 = read_cycles();
    s->dispatch_cycles = t1 - t0;

    /* Output: int32 in FREGs, store as 32-bit values (cols=3 = 16 32-bit/row). */
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
