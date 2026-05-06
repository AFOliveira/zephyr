/*
 * DnCNN3 240x320 with TFMA-based hidden layers, multi-minion.
 *
 * Activation layout: NHWC, with a 1-pixel zero-padded border (242×322×64).
 * Hidden conv = 9 (ky,kx) × 4 ic_chunks × 4 oc_chunks accumulating TFMAs per
 * 16-spatial-pixel × 16-OC output tile. First/final convs use scalar FP32
 * (~0.04 % of total compute).
 *
 * Build variants (selected at compile time):
 *   -DTFMA_PARTITION=A   row-stripe per minion, no coop loads (default)
 *   -DTFMA_PARTITION=B   OC-chunk × spatial-half partition (8 minions = 4 OC-tiles × 2 halves)
 *   -DTFMA_PARTITION=C   row-stripe + cooperative weight loads (tensor_coop)
 *
 * Eight T0 harts (even hart_id) participate; T1 harts return early.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/tensors.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 8u
#endif
#ifndef NUM_HARTS
#define NUM_HARTS 16
#endif

#ifndef TFMA_PARTITION
#define TFMA_PARTITION 'A'
#endif

#ifndef COOP_DRAIN_EVERY
#define COOP_DRAIN_EVERY 4u
#endif

#define DNCNN_MAGIC       0xD3C11004u
#define IMG_W             320u
#define IMG_H             240u
#define CH                64u
#ifndef HIDDEN_LAYERS
#define HIDDEN_LAYERS     18u
#endif
#define K                 3u

#define CHECKPOINT_OFFSET 0x800u   /* hart 0 writes phase markers here */

#define IMG_PIXELS        (IMG_W * IMG_H)
#define INPUT_BYTES       (IMG_PIXELS * sizeof(float))
#define OUTPUT_BYTES      INPUT_BYTES

/* Unpadded activation buffer (matches existing dncnn3_luxonis_real_vpu.c
 * approach). Boundary handling is done in conv_hidden_scalar / _tfma. */
#define ACT_H             (IMG_H)
#define ACT_W             (IMG_W)
#define ACT_PIXELS        (ACT_H * ACT_W)
#define ACT_BYTES         (ACT_PIXELS * CH * sizeof(float))

/* Memory layout offsets within the per-kernel buffer base. */
#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x000000u
#define SUMMARY_OFFSET    0x001000u
#define OUTPUT_OFFSET     0x300000u   /* final output = input - residual */

/* TFMA encoding constants (FP32 path, max-config dispatch). */
#define AROWS_FIELD       0xFu        /* arows = 16 spatial pixels */
#define ACOLS_FIELD       0xFu        /* acols = 16 IC values */
#define BCOLS_FIELD       0x3u        /* bcols field 3 → actual = (3+1)*4 = 16 OC */

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct dncnn_slot {
    uint32_t magic;
    uint32_t hart_id;
    uint32_t minion_id;
    uint32_t thread_id;
    uint32_t row0;
    uint32_t row1;
    uint32_t active_harts;
    uint32_t checksum;
    uint32_t done;
    uint32_t reserved[7];
};

struct dncnn_summary {
    uint32_t magic;
    uint32_t active_harts;
    uint32_t passes;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t layers;
    uint32_t active_mask;
    uint32_t done_count;
    uint32_t output_hash;
    uint32_t slot_checksum_sum;
    uint32_t ops_lo;
    uint32_t ops_hi;
    uint32_t reserved[3];
};

/* Linked-in blobs */
extern const unsigned char _binary_dncnn3_luxonis_240x320_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_hidden_weights_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_hidden_biases_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_first_weights_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_first_biases_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_final_weights_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_final_biases_fp32_bin_start[];
/* Original packed weights — used for scalar conv_hidden reference and to lay
 * out hidden_orig_weights blob (= WH[18, OC, ky, kx, IC] flat). */
extern const unsigned char _binary_dncnn3_tfma_hidden_orig_weights_fp32_bin_start[];

#define INPUT_BLOB         ((const float *)_binary_dncnn3_luxonis_240x320_input_f32_bin_start)
#define REF_BLOB           ((const float *)_binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start)
#define WH_TFMA_BLOB       ((const float *)_binary_dncnn3_tfma_hidden_weights_fp32_bin_start)
#define WH_ORIG_BLOB       ((const float *)_binary_dncnn3_tfma_hidden_orig_weights_fp32_bin_start)
#define BH_BLOB            ((const float *)_binary_dncnn3_tfma_hidden_biases_fp32_bin_start)
#define W0_BLOB            ((const float *)_binary_dncnn3_tfma_first_weights_fp32_bin_start)
#define B0_BLOB            ((const float *)_binary_dncnn3_tfma_first_biases_fp32_bin_start)
#define WF_BLOB            ((const float *)_binary_dncnn3_tfma_final_weights_fp32_bin_start)
#define BF_BLOB            ((const float *)_binary_dncnn3_tfma_final_biases_fp32_bin_start)

/* Activation banks live in .bss (region0_size extended to 64 MB). Padded. */
static float static_act0[ACT_PIXELS * CH] __attribute__((aligned(64)));
static float static_act1[ACT_PIXELS * CH] __attribute__((aligned(64)));
static float static_output[IMG_PIXELS] __attribute__((aligned(64)));

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    }
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u) {
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    }
    return ptr;
}

static inline uint32_t active_mask_t0(void)
{
    /* T0-only: 8 even harts. */
    uint32_t mask = 0;
    for (uint32_t m = 0; m < ACTIVE_HARTS; m++) mask |= 1u << m;
    return mask;
}

static inline void bench_barrier(void)
{
    shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
                  active_mask_t0(), 0u);
}

static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}

static inline float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }

/* ------------------------------------------------------------------------
 * First conv: IC=1, OC=64, K=3, output to padded act0 (with halo offsets).
 * Scalar reference; ~4.6 M MACs total = trivial compared to hidden layers.
 * Each hart owns 1/ACTIVE_HARTS of output rows.
 * Writes act_padded[(y_out + 1)*ACT_W + (x_out + 1)*CH + oc] for oc=0..63.
 * Boundary pixels in the input image are zero-padded (input is 1-channel
 * so the halo is implicit in indexing — guard with `if`).
 * ------------------------------------------------------------------------ */
static void conv_first_scalar(const float *input, const float *w0, const float *b0,
                              float *act,
                              uint32_t row0, uint32_t row1)
{
    /* Unpadded NHWC output: act[(y * IMG_W + x) * CH + oc] */
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            for (uint32_t oc = 0; oc < CH; oc++) {
                float acc = b0[oc];
                for (uint32_t ky = 0; ky < K; ky++) {
                    for (uint32_t kx = 0; kx < K; kx++) {
                        const int yy = (int)y + (int)ky - 1;
                        const int xx = (int)x + (int)kx - 1;
                        float v = 0.0f;
                        if (yy >= 0 && yy < (int)IMG_H && xx >= 0 && xx < (int)IMG_W) {
                            v = input[yy * IMG_W + xx];
                        }
                        acc += w0[oc * 9 + ky * 3 + kx] * v;
                    }
                }
                act[((uint64_t)y * IMG_W + x) * CH + oc] = relu_f32(acc);
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * Final conv: IC=64, OC=1, K=3, residual subtract: out = input - conv(act).
 * Scalar reference; ~5 M MACs.
 * Reads act_padded[(y+1)*ACT_W + (x+1)*CH + ic], writes output[y*IMG_W + x].
 * Final conv weight layout: WF[oc=1, ic=64, ky=3, kx=3].
 * ------------------------------------------------------------------------ */
static void conv_final_scalar(const float *act, const float *input_image,
                              const float *wf, const float *bf,
                              float *output,
                              uint32_t row0, uint32_t row1)
{
    /* Unpadded NHWC input; bound-check boundary reads. */
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            float acc = bf[0];
            for (uint32_t ic = 0; ic < CH; ic++) {
                for (uint32_t ky = 0; ky < K; ky++) {
                    const int y_in = (int)y + (int)ky - 1;
                    if (y_in < 0 || y_in >= (int)IMG_H) continue;
                    for (uint32_t kx = 0; kx < K; kx++) {
                        const int x_in = (int)x + (int)kx - 1;
                        if (x_in < 0 || x_in >= (int)IMG_W) continue;
                        const float v = act[((uint64_t)y_in * IMG_W + (uint32_t)x_in) * CH + ic];
                        /* WF is packed as [OC=1, ky, kx, IC] (NOT [OC,IC,ky,kx] like ONNX) */
                        acc += wf[(ky * K + kx) * CH + ic] * v;
                    }
                }
            }
            output[(uint64_t)y * IMG_W + x] = input_image[(uint64_t)y * IMG_W + x] - acc;
        }
    }
}

/* ------------------------------------------------------------------------
 * Hidden conv via TFMA (NHWC layout, NHWC output).
 *
 * For each output tile of 16 spatial-x × 16 OC at (y_out, x_run, oc_chunk):
 *   for k=0..8 (ky*3+kx): for ic_chunk=0..3:
 *     A: tensor_load 16 lines of activation [act_pad[y_out+ky][x_run+kx..+15][ic_chunk*16+0..15]]
 *        (stride=CH*4=256, num_lines=16, offset within line = ic_chunk * 64)
 *     B: tensor_load 16 lines of weight tile [layer, k, oc_chunk, ic_chunk] (1024 B)
 *     tensor_fma(first_pass = (k==0 && ic_chunk==0))
 *   tensor_store: 16 lines of output [y_out][x_run+0..+15][oc_chunk*16+0..15]
 *   bias + ReLU per cell (read back, fmadd, fmax, write back)
 *
 * Each minion handles `row0..row1` (row-stripe partition).
 * ------------------------------------------------------------------------ */
/* ------------------------------------------------------------------------
 * Hidden conv: SCALAR REFERENCE (unpadded NHWC, boundary checks inline).
 * Reads from src (unpadded), writes to dst (unpadded). Bias + ReLU baked in.
 * Slow but no halo dependency — robust to BSS uninit.
 * ------------------------------------------------------------------------ */
static void conv_hidden_scalar(const float *src, float *dst,
                               const float *layer_w, const float *layer_b,
                               uint32_t row0, uint32_t row1)
{
    /* layer_w shape [OC, ky, kx, IC] — flat index oc*K*K*CH + ky*K*CH + kx*CH + ic */
    for (uint32_t y_out = row0; y_out < row1; y_out++) {
        for (uint32_t x_out = 0; x_out < IMG_W; x_out++) {
            for (uint32_t oc = 0; oc < CH; oc++) {
                float acc = layer_b[oc];
                for (uint32_t ky = 0; ky < K; ky++) {
                    const int y_in = (int)y_out + (int)ky - 1;
                    if (y_in < 0 || y_in >= (int)IMG_H) continue;
                    for (uint32_t kx = 0; kx < K; kx++) {
                        const int x_in = (int)x_out + (int)kx - 1;
                        if (x_in < 0 || x_in >= (int)IMG_W) continue;
                        for (uint32_t ic = 0; ic < CH; ic++) {
                            const float v = src[((uint64_t)y_in * IMG_W + (uint32_t)x_in) * CH + ic];
                            const float w = layer_w[((uint64_t)oc * K + ky) * K * CH
                                                    + kx * CH + ic];
                            acc += w * v;
                        }
                    }
                }
                dst[((uint64_t)y_out * IMG_W + x_out) * CH + oc] = relu_f32(acc);
            }
        }
    }
}

static void conv_hidden_tfma(const float *src_pad, const float *dst_pad_base,
                             const float *layer_w_pack, const float *layer_b,
                             uint32_t row0, uint32_t row1)
{
    const uint64_t ROW_STRIDE_B = (uint64_t)ACT_W * CH * 4u;   /* 322*64*4 = 82,432 */
    const uint64_t PIXEL_STRIDE_B = (uint64_t)CH * 4u;          /* 256 */

    /* dst_pad_base is the .bss padded buffer; results land at +1,+1 offset. */
    float *const dst_pad = (float *)dst_pad_base;

    for (uint32_t oc_c = 0; oc_c < CH / 16u; oc_c++) {
        for (uint32_t y_out = row0; y_out < row1; y_out++) {
            /* y_out maps to padded row y_out+1; reading rows y_out..y_out+2. */
            for (uint32_t x_run = 0; x_run < IMG_W; x_run += 16u) {
                /* Output goes to padded position (y_out+1, x_run+1..x_run+16, oc_c*16..oc_c*16+15). */
                for (uint32_t k = 0; k < K * K; k++) {
                    const uint32_t ky = k / K;
                    const uint32_t kx = k % K;
                    for (uint32_t ic_c = 0; ic_c < CH / 16u; ic_c++) {
                        /* B operand: weight tile [layer][k][oc_c][ic_c][16 IC × 16 OC] */
                        const uint64_t w_addr =
                            (uint64_t)(uintptr_t)(layer_w_pack
                                + ((k * 4u * 4u) + (oc_c * 4u) + ic_c) * 16u * 16u);
                        tensor_load(/*tmask*/ 0, /*coop*/ 0,
                                    /*scp_dst*/ 16, /*xform*/ 0, /*tenb*/ 0,
                                    w_addr, /*offset*/ 0, /*num_lines*/ 15,
                                    /*stride*/ 64, /*id*/ 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        /* A operand: activation tile, 16 spatial pixels × 16 IC.
                         * Each cache line is one (y_pad, x_pad) pixel's 16 ICs.
                         * 16 lines stride = PIXEL_STRIDE_B (one full pixel = 256 B). */
                        const uint64_t a_addr =
                            (uint64_t)(uintptr_t)src_pad
                            + (uint64_t)(y_out + ky) * ROW_STRIDE_B
                            + (uint64_t)(x_run + kx) * PIXEL_STRIDE_B
                            + (uint64_t)(ic_c * 16u) * 4u;
                        tensor_load(/*tmask*/ 0, /*coop*/ 0,
                                    /*scp_dst*/ 0, /*xform*/ 0, /*tenb*/ 0,
                                    a_addr, /*offset*/ 0, /*num_lines*/ 15,
                                    /*stride*/ PIXEL_STRIDE_B, /*id*/ 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        const uint32_t first_pass = (k == 0u && ic_c == 0u) ? 1u : 0u;
                        tensor_fma(/*tmask*/ 0,
                                   BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                                   /*aoffset*/ 0,
                                   /*tenc_loc*/ 0,
                                   /*tenb_uns*/ 0, /*tena_uns*/ 0,
                                   /*tenb_loc*/ 0,
                                   /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                                   /*type*/ 0u, /*first_pass*/ first_pass);
                        tensor_wait(TENSOR_FMA_WAIT);
                    }
                }

                /* Store 16 spatial pixels × 16 OC FP32 values out of FREGs.
                 * Stride = PIXEL_STRIDE_B between pixels. Each pixel writes
                 * 16 fp32 starting at oc_c*16 lane within the pixel's NHWC slot. */
                const uint64_t s_addr =
                    (uint64_t)(uintptr_t)dst_pad
                    + (uint64_t)(y_out + 1u) * ROW_STRIDE_B
                    + (uint64_t)(x_run + 1u) * PIXEL_STRIDE_B
                    + (uint64_t)(oc_c * 16u) * 4u;
                tensor_store(/*reg_stride*/ 0, /*start_reg*/ 0,
                             /*cols*/ 3, /*Arows*/ 0xF,
                             s_addr, /*coop_store*/ 0,
                             /*stride*/ PIXEL_STRIDE_B);
                tensor_wait(TENSOR_STORE_WAIT);
                /* tensor_store goes to L2 directly (per CACHEOP_DST_L2 semantics).
                 * Make L1D coherent for the read by evicting the just-written
                 * region (forces refetch from L2 on next read). */
                evict((const void *)s_addr, 16u * PIXEL_STRIDE_B);
                WAIT_CACHEOPS;
                FENCE;

                /* Bias + ReLU, in-place at the just-stored 16 spatial pixels.
                 * Memory now coherent in L2 after evict + FENCE. */
                volatile float *out_ptr = (volatile float *)(uintptr_t)s_addr;
                for (uint32_t p = 0; p < 16u; p++) {
                    volatile float *cell = (volatile float *)((uintptr_t)out_ptr
                                                              + (uint64_t)p * PIXEL_STRIDE_B);
                    for (uint32_t oc_in = 0; oc_in < 16u; oc_in++) {
                        const float bias = layer_b[oc_c * 16u + oc_in];
                        const float v = cell[oc_in] + bias;
                        cell[oc_in] = v > 0.0f ? v : 0.0f;
                    }
                }
                evict((const void *)s_addr, 16u * PIXEL_STRIDE_B);
                WAIT_CACHEOPS;
            }
        }
    }

    /* Evict the entire stripe so the next layer can read it. The padded halo
     * stays zero (bss-init). */
    const uintptr_t stripe_lo =
        (uintptr_t)dst_pad + (uint64_t)(row0 + 1u) * ROW_STRIDE_B;
    const uint64_t stripe_bytes = (uint64_t)(row1 - row0) * ROW_STRIDE_B;
    evict((const void *)stripe_lo, stripe_bytes);
    WAIT_CACHEOPS;
}

/* ------------------------------------------------------------------------
 * Entry point.
 * ------------------------------------------------------------------------ */
int main(uintptr_t arg_area)
{
    /* T0-only mode: only 8 even harts of minions 0..7 do work. */
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;
    const uint32_t hart_id = raw_hart >> 1;     /* 0..7 */

    setM0MaskFF();

    uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
    const float *const input    = INPUT_BLOB;
    const float *const reference = REF_BLOB;
    float *const final_output   = (float *)(base + OUTPUT_OFFSET);
    float *const act0 = static_act0;
    float *const act1 = static_act1;

    volatile struct dncnn_slot *const slots =
        (volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
    volatile struct dncnn_summary *const summary =
        (volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);

    const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
    const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

    /* Checkpoint marker — hart 0 stamps progress so we can see hangs. */
    volatile uint32_t *const ckpt = (volatile uint32_t *)(base + CHECKPOINT_OFFSET);
    if (hart_id == 0u) {
        ckpt[0] = 0xC0DE0001u;       /* reached main */
        evict((const void *)ckpt, 64u); WAIT_CACHEOPS;
    }

    /* No halo dependence anymore (unpadded buffers + boundary checks in the
     * conv routines), so no need to pre-zero. */

#ifndef DNCNN_TFMA_DEBUG_NOFIRSTBARRIER
    bench_barrier();
#endif
    if (hart_id == 0u) { ckpt[1] = 0xC0DE0002u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }

#ifndef DNCNN_TFMA_SKIP_FIRST
    if (hart_id == 0u) { ckpt[20] = 0xC0DE0F10u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS; }
    /* ---------- First conv: scalar, write into act0 unpadded buffer ---------- */
    conv_first_scalar(input, W0_BLOB, B0_BLOB, act0, row0, row1);
    if (hart_id == 0u) { ckpt[21] = 0xC0DE0F11u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS; }
    /* Evict our stripe so other harts can read it during the next layer. */
    {
        const uint64_t ROW_B = (uint64_t)IMG_W * CH * 4u;
        const uintptr_t lo = (uintptr_t)act0 + (uint64_t)row0 * ROW_B;
        const uint64_t bytes = (uint64_t)(row1 - row0) * ROW_B;
        evict((const void *)lo, bytes);
        WAIT_CACHEOPS;
    }
    bench_barrier();
#endif
    if (hart_id == 0u) { ckpt[2] = 0xC0DE0003u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }

    /* ---------- Hidden layers ---------- */
#ifndef DNCNN_TFMA_SKIP_HIDDEN
    for (uint32_t L = 0; L < HIDDEN_LAYERS; L++) {
        const float *const src = (L & 1u) ? act1 : act0;
        float       *const dst = (L & 1u) ? act0 : act1;
        const float *const b   = BH_BLOB + L * CH;
#ifdef DNCNN_TFMA_USE_SCALAR_HIDDEN
        const float *const w_orig = WH_ORIG_BLOB + L * CH * K * K * CH;
        const uint64_t ROW_B = (uint64_t)IMG_W * CH * 4u;
        /* Pre-read evict: invalidate halo rows adjacent to our stripe in our
         * local L1D so the read sees fresh data from L2 (written by neighbor
         * harts in the previous layer's post-write evict). */
        if (row0 > 0u) {
            evict((const void *)((uintptr_t)src + (uint64_t)(row0 - 1u) * ROW_B), ROW_B);
        }
        if (row1 < IMG_H) {
            evict((const void *)((uintptr_t)src + (uint64_t)row1 * ROW_B), ROW_B);
        }
        WAIT_CACHEOPS;
        conv_hidden_scalar(src, dst, w_orig, b, row0, row1);
        /* Post-write evict: push our stripe to L2 so neighbors see it. */
        evict((const void *)((uintptr_t)dst + (uint64_t)row0 * ROW_B),
              (uint64_t)(row1 - row0) * ROW_B);
        WAIT_CACHEOPS;
#else
        const float *const w = WH_TFMA_BLOB + L * 9u * 4u * 4u * 16u * 16u;
        conv_hidden_tfma(src, dst, w, b, row0, row1);
#endif
        bench_barrier();
        if (hart_id == 0u) {
            ckpt[3 + L] = 0xC0DE0010u + L;
            evict((const void *)ckpt, 128u); WAIT_CACHEOPS;
        }
    }
#endif
    if (hart_id == 0u) { ckpt[2] = 0xC0DE000Au; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }

    /* ---------- Final conv: scalar, residual subtract ---------- */
#ifndef DNCNN_TFMA_SKIP_FINAL
    if (hart_id == 0u) { ckpt[22] = 0xC0DE0F20u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS; }
    {
        const float *const last = (HIDDEN_LAYERS & 1u) ? act1 : act0;
        conv_final_scalar(last, input, WF_BLOB, BF_BLOB, final_output, row0, row1);
        evict(final_output + row0 * IMG_W,
              (uint64_t)(row1 - row0) * IMG_W * sizeof(float));
        WAIT_CACHEOPS;
    }
    if (hart_id == 0u) { ckpt[23] = 0xC0DE0F21u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS; }
    bench_barrier();
#endif

    /* ---------- Slot summary write per hart, then hart-0 global summary ---------- */
#ifdef DNCNN_TFMA_DEBUG_NOSLOT
    bench_barrier();
    return 0;
#endif
#ifdef DNCNN_TFMA_DEBUG_NOCHK
    const uint32_t checksum = 0;
#else
    /* Cheap stripe checksum over our region */
    uint32_t checksum = 0;
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            const uint32_t bits = *(const uint32_t *)&final_output[y * IMG_W + x];
            checksum = ((checksum << 5) | (checksum >> 27)) ^ bits;
        }
    }
#endif

    if (hart_id == 0u) { ckpt[10] = 0xC0DEAA01u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }
    volatile struct dncnn_slot *const slot = &slots[hart_id];
    if (hart_id == 0u) { ckpt[11] = 0xC0DEAA02u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }
    slot->magic = DNCNN_MAGIC;
    slot->hart_id = hart_id;
    slot->minion_id = get_minion_id();
    slot->thread_id = get_thread_id();
    slot->row0 = row0;
    slot->row1 = row1;
    slot->active_harts = ACTIVE_HARTS;
    slot->checksum = checksum;
    slot->done = 1;
    if (hart_id == 0u) { ckpt[12] = 0xC0DEAA03u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }
    evict((const void *)slot, sizeof(*slot));
    if (hart_id == 0u) { ckpt[13] = 0xC0DEAA04u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }
    WAIT_CACHEOPS;
    if (hart_id == 0u) { ckpt[14] = 0xC0DEAA05u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }
    bench_barrier();
    if (hart_id == 0u) { ckpt[15] = 0xC0DEAA06u; evict((const void *)ckpt, 64u); WAIT_CACHEOPS; }

    if (hart_id == 0u) {
        uint32_t mask = 0, done = 0, sum = 0;
        for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
            if (slots[h].magic == DNCNN_MAGIC && slots[h].done) {
                mask |= 1u << slots[h].hart_id;
                done++;
                sum += slots[h].checksum;
            }
        }
        summary->magic = DNCNN_MAGIC;
        summary->active_harts = ACTIVE_HARTS;
        summary->passes = 1;
        summary->width = IMG_W;
        summary->height = IMG_H;
        summary->channels = CH;
        summary->layers = HIDDEN_LAYERS + 2u;
        summary->active_mask = mask;
        summary->done_count = done;
        summary->slot_checksum_sum = sum;

#ifndef DNCNN_TFMA_DEBUG_NOMAX
        ckpt[16] = 0xC0DE5101u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS;
        /* Compute max_abs vs ORT reference (only hart 0). */
        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        for (uint32_t i = 0; i < IMG_PIXELS; i++) {
            const float d = final_output[i] - reference[i];
            const float ad = d < 0 ? -d : d;
            if (ad > max_abs) max_abs = ad;
            mean_abs += ad;
        }
        ckpt[17] = 0xC0DE5102u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS;
        mean_abs *= (1.0f / (float)IMG_PIXELS);   /* avoid FP divide */
        ckpt[18] = 0xC0DE5103u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS;
        summary->reserved[0] = (uint32_t)(max_abs * 1e9f);
        summary->reserved[1] = (uint32_t)(mean_abs * 1e9f);
        ckpt[19] = 0xC0DE5104u; evict((const void *)ckpt, 128u); WAIT_CACHEOPS;
#endif

        evict((const void *)summary, sizeof(*summary));
        WAIT_CACHEOPS;
    }
    bench_barrier();
    return 0;
}
