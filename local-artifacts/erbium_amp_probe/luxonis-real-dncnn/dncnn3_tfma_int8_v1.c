/*
 * DnCNN3 INT8 (per-channel PTQ) multi-minion kernel.
 *
 * Pipeline:
 *   conv_first scalar FP32  → quantize to int8 → 18 INT8 hidden layers →
 *   dequantize to FP32 → conv_final scalar FP32 → residual subtract.
 *
 * Hidden layer N:
 *   for each output (y_out, x_out, oc):
 *     int32 acc = 0
 *     for each (ky, kx, ic):
 *       acc += int32(W_int8[L, oc, ky, kx, ic]) * int32(A_int8[y_in, x_in, ic])
 *     fp32 acc_dq = acc * W_scale[L, oc] * A_scale[L]
 *     fp32 out = relu(acc_dq + bias[L, oc])
 *     int8 next = sat_int8(round(out / A_scale[L+1]))
 *
 * Activations between hidden layers stored as int8 (saves 4× memory).
 * Audit gate: ~5e-2 vs FP32 ORT.
 */
#include <stdint.h>
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 8u
#endif
#ifndef NUM_HARTS
#define NUM_HARTS 16
#endif

#define DNCNN_MAGIC       0xD3C11008u
#define IMG_W             320u
#define IMG_H             240u
#define CH                64u
#define HIDDEN_LAYERS     18u
#define K                 3u
#define IMG_PIXELS        (IMG_W * IMG_H)

#define SLOTS_OFFSET      0x000000u
#define SUMMARY_OFFSET    0x001000u
#define OUTPUT_OFFSET     0x300000u
#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct dncnn_slot {
    uint32_t magic, hart_id, minion_id, thread_id;
    uint32_t row0, row1, active_harts, checksum, done;
    uint32_t reserved[7];
};
struct dncnn_summary {
    uint32_t magic, active_harts, passes, width, height, channels, layers;
    uint32_t active_mask, done_count, output_hash, slot_checksum_sum;
    uint32_t ops_lo, ops_hi;
    uint32_t reserved[3];
};

extern const unsigned char _binary_dncnn3_luxonis_240x320_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_first_weights_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_first_biases_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_final_weights_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_final_biases_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_tfma_hidden_biases_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_int8_hidden_weights_int8_bin_start[];
extern const unsigned char _binary_dncnn3_int8_hidden_w_scales_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_int8_act_scales_fp32_bin_start[];
extern const unsigned char _binary_dncnn3_int8_inv_act_scales_fp32_bin_start[];

#define INPUT_BLOB        ((const float *)_binary_dncnn3_luxonis_240x320_input_f32_bin_start)
#define REF_BLOB          ((const float *)_binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start)
#define W0_BLOB           ((const float *)_binary_dncnn3_tfma_first_weights_fp32_bin_start)
#define B0_BLOB           ((const float *)_binary_dncnn3_tfma_first_biases_fp32_bin_start)
#define WF_BLOB           ((const float *)_binary_dncnn3_tfma_final_weights_fp32_bin_start)
#define BF_BLOB           ((const float *)_binary_dncnn3_tfma_final_biases_fp32_bin_start)
#define BH_BLOB           ((const float *)_binary_dncnn3_tfma_hidden_biases_fp32_bin_start)
#define WH_INT8_BLOB      ((const int8_t *)_binary_dncnn3_int8_hidden_weights_int8_bin_start)
#define WS_BLOB           ((const float *)_binary_dncnn3_int8_hidden_w_scales_fp32_bin_start)
#define AS_BLOB           ((const float *)_binary_dncnn3_int8_act_scales_fp32_bin_start)
#define INV_AS_BLOB       ((const float *)_binary_dncnn3_int8_inv_act_scales_fp32_bin_start)

static int8_t static_act0[IMG_PIXELS * CH] __attribute__((aligned(64)));
static int8_t static_act1[IMG_PIXELS * CH] __attribute__((aligned(64)));
static float  static_act_fp32[IMG_PIXELS * CH] __attribute__((aligned(64)));   /* conv_first output */
static float  static_output[IMG_PIXELS] __attribute__((aligned(64)));

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    return ptr;
}

static inline uint32_t mask_t0(void) { uint32_t m = 0; for (uint32_t i = 0; i < ACTIVE_HARTS; i++) m |= 1u << i; return m; }
static inline void bench_barrier(void) { shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS, mask_t0(), 0u); }
static inline float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }
static inline int8_t sat_int8(float x) {
    int v = (int)(x + (x >= 0 ? 0.5f : -0.5f));
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}

/* Conv_first: FP32 input → FP32 output, then quantize to int8 for hidden. */
static void conv_first_to_int8(const float *input, const float *w0, const float *b0,
                                int8_t *act_int8, float inv_a_scale_out,
                                uint32_t row0, uint32_t row1)
{
    /* Caller passes precomputed reciprocal; FP divide hangs in this U-mode runtime. */
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            for (uint32_t oc = 0; oc < CH; oc++) {
                float acc = b0[oc];
                for (uint32_t ky = 0; ky < K; ky++) {
                    for (uint32_t kx = 0; kx < K; kx++) {
                        const int yy = (int)y + (int)ky - 1;
                        const int xx = (int)x + (int)kx - 1;
                        float v = 0.0f;
                        if (yy >= 0 && yy < (int)IMG_H && xx >= 0 && xx < (int)IMG_W)
                            v = input[yy * IMG_W + xx];
                        acc += w0[oc * 9 + ky * 3 + kx] * v;
                    }
                }
                const float relu = acc > 0.0f ? acc : 0.0f;
                act_int8[((uint64_t)y * IMG_W + x) * CH + oc] = sat_int8(relu * inv_a_scale_out);
            }
        }
    }
}

/* Hidden conv: int8 in → int8 out. Applies bias + ReLU + requantize for next layer. */
static void conv_hidden_int8(const int8_t *src, int8_t *dst,
                              const int8_t *layer_w_int8, const float *layer_w_scales,
                              const float *layer_b, float a_scale_in, float inv_a_scale_out,
                              uint32_t row0, uint32_t row1)
{
    /* layer_w_int8 layout: [OC, ky, kx, IC]; inv_a_scale_out precomputed by caller. */
    for (uint32_t y_out = row0; y_out < row1; y_out++) {
        for (uint32_t x_out = 0; x_out < IMG_W; x_out++) {
            for (uint32_t oc = 0; oc < CH; oc++) {
                int32_t acc = 0;
                for (uint32_t ky = 0; ky < K; ky++) {
                    const int y_in = (int)y_out + (int)ky - 1;
                    if (y_in < 0 || y_in >= (int)IMG_H) continue;
                    for (uint32_t kx = 0; kx < K; kx++) {
                        const int x_in = (int)x_out + (int)kx - 1;
                        if (x_in < 0 || x_in >= (int)IMG_W) continue;
                        for (uint32_t ic = 0; ic < CH; ic++) {
                            const int32_t a = (int32_t)src[((uint64_t)y_in * IMG_W + (uint32_t)x_in) * CH + ic];
                            const int32_t w = (int32_t)layer_w_int8[((uint64_t)oc * K + ky) * K * CH + kx * CH + ic];
                            acc += w * a;
                        }
                    }
                }
                const float dq = (float)acc * layer_w_scales[oc] * a_scale_in;
                const float r = relu_f32(dq + layer_b[oc]);
                dst[((uint64_t)y_out * IMG_W + x_out) * CH + oc] = sat_int8(r * inv_a_scale_out);
            }
        }
    }
}

/* Conv_final: int8 in → FP32 residual subtract output. */
static void conv_final_int8_to_fp32(const int8_t *act_int8, float a_scale_in,
                                    const float *input_image,
                                    const float *wf, const float *bf,
                                    float *output, uint32_t row0, uint32_t row1)
{
    /* WF layout: [OC=1, ky, kx, IC] (per discovery) */
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            float acc = bf[0];
            for (uint32_t ky = 0; ky < K; ky++) {
                const int y_in = (int)y + (int)ky - 1;
                if (y_in < 0 || y_in >= (int)IMG_H) continue;
                for (uint32_t kx = 0; kx < K; kx++) {
                    const int x_in = (int)x + (int)kx - 1;
                    if (x_in < 0 || x_in >= (int)IMG_W) continue;
                    for (uint32_t ic = 0; ic < CH; ic++) {
                        const float v = (float)act_int8[((uint64_t)y_in * IMG_W + (uint32_t)x_in) * CH + ic] * a_scale_in;
                        acc += wf[(ky * K + kx) * CH + ic] * v;
                    }
                }
            }
            output[(uint64_t)y * IMG_W + x] = input_image[(uint64_t)y * IMG_W + x] - acc;
        }
    }
}

int main(uintptr_t arg_area)
{
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;
    const uint32_t hart_id = raw_hart >> 1;

    uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
    const float *const input    = INPUT_BLOB;
    const float *const reference = REF_BLOB;
    float       *const final_output = (float *)(base + OUTPUT_OFFSET);
    int8_t      *const act0 = static_act0;
    int8_t      *const act1 = static_act1;

    volatile struct dncnn_slot *const slots = (volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
    volatile struct dncnn_summary *const summary = (volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);

    const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
    const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;
    bench_barrier();

    /* Reciprocals are precomputed on host (FP divide hangs in this U-mode runtime). */
    conv_first_to_int8(input, W0_BLOB, B0_BLOB, act0, INV_AS_BLOB[0], row0, row1);
    {
        const uint64_t ROW_B = (uint64_t)IMG_W * CH;  /* int8: 1 byte per element */
        evict((const void *)((uintptr_t)act0 + (uint64_t)row0 * ROW_B), (uint64_t)(row1 - row0) * ROW_B);
        WAIT_CACHEOPS;
    }
    bench_barrier();

    /* 18 hidden layers, ping-pong int8 buffers */
    for (uint32_t L = 0; L < HIDDEN_LAYERS; L++) {
        const int8_t *const src = (L & 1u) ? act1 : act0;
        int8_t       *const dst = (L & 1u) ? act0 : act1;
        const int8_t *const w_int8 = WH_INT8_BLOB + L * CH * K * K * CH;
        const float  *const w_scl  = WS_BLOB + L * CH;
        const float  *const b      = BH_BLOB + L * CH;
        const float a_scale_in  = AS_BLOB[L];
        const float inv_a_out   = INV_AS_BLOB[L + 1u];

        /* Pre-read evict halo rows */
        const uint64_t ROW_B = (uint64_t)IMG_W * CH;  /* int8 */
        if (row0 > 0u)        evict((const void *)((uintptr_t)src + (uint64_t)(row0 - 1u) * ROW_B), ROW_B);
        if (row1 < IMG_H)     evict((const void *)((uintptr_t)src + (uint64_t)row1 * ROW_B), ROW_B);
        WAIT_CACHEOPS;

        conv_hidden_int8(src, dst, w_int8, w_scl, b, a_scale_in, inv_a_out, row0, row1);

        evict((const void *)((uintptr_t)dst + (uint64_t)row0 * ROW_B), (uint64_t)(row1 - row0) * ROW_B);
        WAIT_CACHEOPS;
        bench_barrier();
    }

    /* conv_final: int8 → FP32 residual subtract */
    {
        const int8_t *const last = (HIDDEN_LAYERS & 1u) ? act1 : act0;
        const float a_scale_last = AS_BLOB[HIDDEN_LAYERS];
        conv_final_int8_to_fp32(last, a_scale_last, input, WF_BLOB, BF_BLOB,
                                final_output, row0, row1);
        evict(final_output + row0 * IMG_W, (uint64_t)(row1 - row0) * IMG_W * sizeof(float));
        WAIT_CACHEOPS;
    }
    bench_barrier();

    /* Slot fill + summary */
    uint32_t checksum = 0;
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            const uint32_t bits = *(const uint32_t *)&final_output[y * IMG_W + x];
            checksum = ((checksum << 5) | (checksum >> 27)) ^ bits;
        }
    }
    volatile struct dncnn_slot *const slot = &slots[hart_id];
    slot->magic = DNCNN_MAGIC;
    slot->hart_id = hart_id;
    slot->minion_id = get_minion_id();
    slot->row0 = row0;
    slot->row1 = row1;
    slot->checksum = checksum;
    slot->done = 1;
    evict((const void *)slot, sizeof(*slot));
    WAIT_CACHEOPS;
    bench_barrier();

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
        summary->done_count = done;
        summary->active_mask = mask;
        summary->slot_checksum_sum = sum;

        /* Compute max_abs vs ORT (loose 5e-2 gate for INT8) */
        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        for (uint32_t i = 0; i < IMG_PIXELS; i++) {
            const float d = final_output[i] - reference[i];
            const float ad = d < 0 ? -d : d;
            if (ad > max_abs) max_abs = ad;
            mean_abs += ad;
        }
        mean_abs *= (1.0f / (float)IMG_PIXELS);
        summary->reserved[0] = (uint32_t)(max_abs * 1e9f);
        summary->reserved[1] = (uint32_t)(mean_abs * 1e9f);
        evict((const void *)summary, sizeof(*summary));
        WAIT_CACHEOPS;
    }
    bench_barrier();
    return 0;
}
