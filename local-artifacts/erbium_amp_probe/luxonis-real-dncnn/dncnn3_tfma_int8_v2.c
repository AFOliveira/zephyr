/*
 * DnCNN3 INT8 multi-minion kernel — TFMA INT8 path for hidden layers.
 *
 * Hidden conv replaced by tensor_fma(type=3) chain:
 *   - 16 OC × 16 spatial × 64 IC tile per dispatch
 *   - 9 dispatches per output tile (one per (ky, kx) tap), tenc_loc=1 on tap 8
 *   - Per dispatch: 16,384 MACs in ~318 cyc → 51.5 MACs/cyc/minion peak
 *
 * Conv_first / conv_final stay scalar FP32 (already cheap).
 *
 * Output tile: 1 row × 16 cols × 16 OC. 4 oc_tiles × 20 spatial_tiles per row.
 * Per minion (8-way row-stripe): 30 rows × 4 × 20 = 2400 tiles per layer.
 *
 * B-side packing: per tile, build 9216 B staging (9 taps × 16 lines × 64 lanes)
 * from NHWC activation src with zero-padding for out-of-bounds halo.
 *
 * Pipeline (same as v1):
 *   conv_first FP32 → quantize to int8 → 18 INT8 hidden via TFMA →
 *   conv_final int8→FP32 → residual subtract.
 */
#include <stdint.h>
#include <string.h>
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

/* DNCNN_SMT_PIPELINE: T1 of each minion runs pack_b_tile for tile N+1 in
 * parallel with T0 running TFMA + marshal for tile N. Producer/consumer
 * synchronized via shared L1D (T0 and T1 are SMT siblings on one minion).
 * When undefined, falls back to the original mode where T1 returns at entry. */
#ifndef DNCNN_SMT_PIPELINE
#define DNCNN_SMT_PIPELINE 0   /* See journal note: blocked by SMT-sibling cache coherence on real silicon. */
#endif

/* Number of back-to-back inferences in a single kernel launch. Each iteration
 * runs the full conv_first → 18 hidden → conv_final pipeline on the same
 * input image (the only one baked in), and writes its output to
 *   OUTPUT_OFFSET + i * IMG_PIXELS * sizeof(float)
 * Use to amortize launcher / one-time-init overhead across many frames. */
#ifndef DNCNN_N_INFERENCES
#define DNCNN_N_INFERENCES 1u
#endif
#define OUTPUT_STRIDE_BYTES  (IMG_PIXELS * 4u)   /* one denoised FP32 image */

/* DNCNN_STREAMING: when enabled, kernel runs an indefinite polling loop over
 * a control struct in DRAM. Host writes new image at STREAM_INPUT_OFFSET +
 * sets cmd=STREAM_CMD_PROCESS; kernel processes; writes done counter.
 * Host can shutdown via cmd=STREAM_CMD_SHUTDOWN. */
#ifndef DNCNN_STREAMING
#define DNCNN_STREAMING 0
#endif

/* Streaming offsets (kernel-side, relative to its heap base). */
#define STREAM_CTRL_OFFSET    0x002000u   /* shared control struct (host R/W) */
#define STREAM_INPUT_OFFSET   0x010000u   /* host writes input image here */
/* Output goes to OUTPUT_OFFSET + frame_idx * OUTPUT_STRIDE_BYTES (mod ring). */

/* Streaming control struct fields (laid out at STREAM_CTRL_OFFSET). */
#define STREAM_CMD_IDLE       0u
#define STREAM_CMD_PROCESS    1u
#define STREAM_CMD_SHUTDOWN   2u
struct dncnn_stream_ctrl {
    volatile uint32_t cmd;          /* host writes; kernel reads */
    volatile uint32_t frame_idx;    /* host increments per frame */
    volatile uint32_t done_idx;     /* kernel writes when frame complete */
    volatile uint32_t output_off;   /* kernel writes (frame's output offset) */
    volatile uint64_t cycles_start; /* kernel writes per-frame timing */
    volatile uint64_t cycles_end;
    uint32_t reserved[8];
};

/* Always 8 minions doing real compute work (row partition is per-minion) */
#define ACTIVE_MINIONS    8u

#if DNCNN_SMT_PIPELINE
/* 16 harts active (8 T0 compute + 8 T1 prefetch). All participate in the
 * shire barrier so layer transitions sync everyone. */
#define BARRIER_HARTS     16u
#define BARRIER_MASK_T0   0xffu
#define BARRIER_MASK_T1   0xffu
#else
#define BARRIER_HARTS     ACTIVE_HARTS
#define BARRIER_MASK_T0   0xffu
#define BARRIER_MASK_T1   0u
#endif

#define DNCNN_MAGIC       0xD3C11108u
#define IMG_W             320u
#define IMG_H             240u
#define CH                64u
#define HIDDEN_LAYERS     18u
#define K                 3u
#define IMG_PIXELS        (IMG_W * IMG_H)

/* Padded activation buffer: 1-pixel zero halo on each side, so the inner
 * pack_b_tile reads need no boundary check. Real data lives at
 *   act_pad[1..240][1..320][:]   (1-indexed in y and x)
 * Halo at y=0/241 and x=0/321 stays zero (initialized once at kernel start). */
#define PAD               1u
#define H_PAD             (IMG_H + 2u * PAD)   /* 242 */
#define W_PAD             (IMG_W + 2u * PAD)   /* 322 */
#define ACT_PAD_PIXELS    (H_PAD * W_PAD)      /* 77,924 */

#define TILE_W            16u
#define OC_PER_TILE       16u
#define OC_TILES          (CH / OC_PER_TILE)   /* = 4 */
#define K0_PER_DISPATCH   16u                  /* 16 k0 lines × 4 IC = 64 IC */

#define SLOTS_OFFSET      0x000000u
#define SUMMARY_OFFSET    0x001000u
#define OUTPUT_OFFSET     0x300000u
#define DEBUG_TILE_OFFSET 0x900000u   /* hart 0 first-tile outbuf dump */
#define PHASE_TS_OFFSET   0xA00000u   /* per-phase HPM[0] timestamps (hart 0) */
/* Layout at PHASE_TS_OFFSET, all uint64_t:
 *   [0]  start
 *   [1]  after conv_first
 *   [2]  after barrier
 *   [3..20]  after each of 18 hidden layers
 *   [21] after conv_final
 *   [22] after barrier
 *   [23] end
 *   [24..71] reserved (48 slots × 8 = 384 B)
 */
#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

#define AROWS_FIELD       0xFu  /* 16 OC rows */
#ifdef DNCNN_ACOLS16
#define ACOLS_FIELD       0x3u  /* (3+1)*4 = 16 IC cols, 4 dispatches/tap */
#else
#define ACOLS_FIELD       0xFu  /* (15+1)*4 = 64 IC cols, 1 dispatch/tap */
#endif
#define BCOLS_FIELD       0x3u  /* (3+1)*4 = 16 spatial cols */

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
extern const unsigned char _binary_dncnn3_tfma_int8_a_pack_bin_start[];
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
#define WA_INT8_BLOB      ((const int8_t *)_binary_dncnn3_tfma_int8_a_pack_bin_start)
#define WS_BLOB           ((const float *)_binary_dncnn3_int8_hidden_w_scales_fp32_bin_start)
#define AS_BLOB           ((const float *)_binary_dncnn3_int8_act_scales_fp32_bin_start)
#define INV_AS_BLOB       ((const float *)_binary_dncnn3_int8_inv_act_scales_fp32_bin_start)

/* Static activation banks (padded NHWC int8). Pad rows/cols stay zero so
 * pack_b_tile can read without boundary checks. */
static int8_t static_act0[ACT_PAD_PIXELS * CH] __attribute__((aligned(64)));
static int8_t static_act1[ACT_PAD_PIXELS * CH] __attribute__((aligned(64)));
static float  static_output[IMG_PIXELS] __attribute__((aligned(64)));

/* TFMA tensor_load encodes addr & ~0x3F (silently rounds DOWN to 64-byte
 * boundary). The linker-placed binary blobs (.o-converted) are only 4-byte
 * aligned, so we MUST copy the A-pack into a 64-byte-aligned static buffer.
 * Same total bytes as the source blob: 18 × 4 × 9 × 16 × 64 = 663,552. */
#define A_PACK_BYTES (18u * 4u * 9u * 16u * 64u)
static int8_t static_a_pack[A_PACK_BYTES] __attribute__((aligned(64)));

/* conv_first weight repack: from [OC=64][9] (ONNX/baked layout) to
 * [v=0..7][k=0..8][oc_in_v=0..7] — 8 OC-vectors, each with 9 taps × 8 OCs
 * laid out so vpu_conv_first_8oc can do contiguous flq2 loads stride 32 B. */
static float static_w0_packed[8u * 9u * 8u] __attribute__((aligned(64)));

/* conv_final scaled weights: wf[ky][kx][ic] * a_scale_in, precomputed at
 * kernel startup. Layout matches the baked WF_BLOB ([OC=1][ky][kx][IC]),
 * so wf_scaled[(ky*K + kx) * CH + ic] is the access. */
static float static_wf_scaled[9u * 64u] __attribute__((aligned(64)));

/* spatial-vec offset table for conv_final's fgb.ps gather: each lane reads
 * one of 8 spatial positions, stride CH=64 bytes between them. */
static const int32_t CF_SPATIAL_OFS[8] __attribute__((aligned(32))) =
    {0, 64, 128, 192, 256, 320, 384, 448};

/* Per-minion SMT pipeline state: T1 packs B for tile N+1 while T0 consumes
 * tile N. Double-buffered bpack so producer and consumer don't collide.
 *
 * 9 taps × 16 k0 × 64 lanes = 9216 B per slot. With 2 slots (double-buffered)
 * = 18,432 B per minion. ×8 minions = 147 KB total in BSS.
 *
 * `produced` and `consumed` are absolute counters (monotonic across all 18
 * layers). T0 waits for produced > tile_idx; T1 waits for slot to be free
 * (consumed + 2 > tile_idx, so T1 is at most 1 tile ahead).
 *
 * NON-volatile bpack so the compiler fuses byte stores; ordering enforced by
 * explicit FENCE + evict + WAIT_CACHEOPS + memory-clobber asm. */
#define BPACK_BYTES_PER_SLOT  (9u * K0_PER_DISPATCH * 64u)
struct minion_pipe {
    int8_t bpack[2][BPACK_BYTES_PER_SLOT] __attribute__((aligned(64)));
    volatile uint32_t produced;
    volatile uint32_t consumed;
    uint32_t pad[14];                       /* fill out to a clean cache-line boundary */
};
static struct minion_pipe minion_pipes[ACTIVE_MINIONS] __attribute__((aligned(64)));

/* Per-minion int32 output staging (only T0 uses; one per minion not per hart). */
static int32_t static_outbuf[ACTIVE_MINIONS][OC_PER_TILE * TILE_W] __attribute__((aligned(64)));

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
    if (arg_area == 0u || arg_area == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
    if (ptr == 0u || ptr == ~(uintptr_t)0u)
        return (uintptr_t)heap0_end - (64u * 1024u * 1024u);
    return ptr;
}

static inline void bench_barrier(void) {
    shire_barrier(BENCH_FLB, BENCH_FCC, BARRIER_HARTS,
                  BARRIER_MASK_T0, BARRIER_MASK_T1);
}

static inline uint64_t read_cycles(void)
{
    uint64_t v;
    __asm__ __volatile__("csrr %0, hpmcounter3" : "=r"(v));
    return v;
}

static inline void setM0MaskFF(void)
{
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}

static inline float relu_f32(float x) { return x > 0.0f ? x : 0.0f; }
static inline int8_t sat_int8(float x) {
    int v = (int)(x + (x >= 0 ? 0.5f : -0.5f));
    if (v < -128) v = -128;
    if (v > 127) v = 127;
    return (int8_t)v;
}

/* VPU-vectorized marshalling: for one OC's row of 16 int32 values (in
 * outbuf[r*16..r*16+15]), apply the full per-channel int8 quantization chain
 * using packed-fp32 ops. Produces 16 int32 lanes in `staging` clamped to
 * [0, 127] and rounded RNE; caller does scalar (int8_t)cast + scattered store.
 *
 * Chain: rounded_int = clamp(0, 127, fmadd(scale, fp32_of_int32, bias))
 * where ReLU is implicit via fmax against 0. */
static inline void vpu_marshal_row16(const int32_t *src, float cs_r,
                                     float cb_r, int32_t *staging)
{
    union { float f; uint32_t u; } cs_un, cb_un, hi_un, zero_un;
    cs_un.f   = cs_r;
    cb_un.f   = cb_r;
    hi_un.f   = 127.0f;
    zero_un.u = 0u;

    __asm__ __volatile__(
        "flq2       f0,  0(%[src])\n"     /* int32 lanes 0..7 */
        "flq2       f1, 32(%[src])\n"     /* int32 lanes 8..15 */
        "fcvt.ps.pw f0, f0\n"             /* int32 → fp32 lanes 0..7 */
        "fcvt.ps.pw f1, f1\n"             /* int32 → fp32 lanes 8..15 */
        "fbcx.ps    f4, %[cs]\n"          /* broadcast scale */
        "fbcx.ps    f5, %[cb]\n"          /* broadcast bias */
        "fbcx.ps    f6, %[hi]\n"          /* broadcast 127.0 */
        "fbcx.ps    f7, %[lo]\n"          /* broadcast 0.0 */
        "fmadd.ps   f0, f0, f4, f5\n"     /* f0 = f0*cs + cb */
        "fmadd.ps   f1, f1, f4, f5\n"
        "fmax.ps    f0, f0, f7\n"         /* ReLU */
        "fmax.ps    f1, f1, f7\n"
        "fmin.ps    f0, f0, f6\n"         /* clamp ≤ 127 */
        "fmin.ps    f1, f1, f6\n"
        "fcvt.pw.ps f0, f0, rne\n"        /* fp32 → int32 round-nearest-even */
        "fcvt.pw.ps f1, f1, rne\n"
        "fsq2       f0,  0(%[dst])\n"
        "fsq2       f1, 32(%[dst])\n"
        :
        : [src] "r"(src), [dst] "r"(staging),
          [cs] "r"((uint64_t)cs_un.u),
          [cb] "r"((uint64_t)cb_un.u),
          [hi] "r"((uint64_t)hi_un.u),
          [lo] "r"((uint64_t)zero_un.u)
        : "memory", "f0", "f1", "f4", "f5", "f6", "f7"
    );
}

/* VPU-vectorized inner block of conv_first: process 8 OCs in parallel.
 *
 *   acc[8] = bias[8] + sum_{k=0..8} w[k][8] * inputs[k]
 *   out[8] = sat_int8(round(clamp(0, 127, relu(acc) * inv_a_scale)))
 *
 * `w_8x9` points to 9 contiguous tap-rows of 8 fp32s each (= 9*8*4 = 288 B).
 * `bias8` points to 8 contiguous fp32 biases.
 * `inputs9` is 9 fp32 input values for the 9 (ky,kx) taps (already
 * boundary-resolved to 0 for OOB taps by the caller).
 * `staging` is 8 int32 results clamped to [0, 127]; caller casts to int8. */
static inline void vpu_conv_first_8oc(const float *bias8,
                                       const float *w_8x9,
                                       const float *inputs9,
                                       float inv_a_scale_out,
                                       int32_t *staging)
{
    union { float f; uint32_t u; } in[9], ia, hi, lo;
    for (uint32_t k = 0; k < 9u; k++) in[k].f = inputs9[k];
    ia.f = inv_a_scale_out;
    hi.f = 127.0f;
    lo.f = 0.0f;

    __asm__ __volatile__(
        "flq2       f0, 0(%[bias])\n"        /* 8 fp32 bias → f0 */
        "flq2       f1,   0(%[w])\n"
        "fbcx.ps    f8, %[i0]\n"
        "fmadd.ps   f0, f1, f8, f0\n"        /* acc += w[0] * input[0] */
        "flq2       f1,  32(%[w])\n"
        "fbcx.ps    f8, %[i1]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1,  64(%[w])\n"
        "fbcx.ps    f8, %[i2]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1,  96(%[w])\n"
        "fbcx.ps    f8, %[i3]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1, 128(%[w])\n"
        "fbcx.ps    f8, %[i4]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1, 160(%[w])\n"
        "fbcx.ps    f8, %[i5]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1, 192(%[w])\n"
        "fbcx.ps    f8, %[i6]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1, 224(%[w])\n"
        "fbcx.ps    f8, %[i7]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "flq2       f1, 256(%[w])\n"
        "fbcx.ps    f8, %[i8]\n"
        "fmadd.ps   f0, f1, f8, f0\n"
        "fbcx.ps    f9, %[lo]\n"
        "fmax.ps    f0, f0, f9\n"            /* ReLU */
        "fbcx.ps    f10, %[ia]\n"
        "fmul.ps    f0, f0, f10\n"           /* * inv_a_scale */
        "fbcx.ps    f11, %[hi]\n"
        "fmin.ps    f0, f0, f11\n"           /* clamp ≤ 127 */
        "fcvt.pw.ps f0, f0, rne\n"           /* fp32 → int32 RNE */
        "fsq2       f0, 0(%[stage])\n"
        :
        : [bias]"r"(bias8), [w]"r"(w_8x9), [stage]"r"(staging),
          [i0]"r"((uint64_t)in[0].u), [i1]"r"((uint64_t)in[1].u),
          [i2]"r"((uint64_t)in[2].u), [i3]"r"((uint64_t)in[3].u),
          [i4]"r"((uint64_t)in[4].u), [i5]"r"((uint64_t)in[5].u),
          [i6]"r"((uint64_t)in[6].u), [i7]"r"((uint64_t)in[7].u),
          [i8]"r"((uint64_t)in[8].u),
          [ia]"r"((uint64_t)ia.u), [hi]"r"((uint64_t)hi.u), [lo]"r"((uint64_t)lo.u)
        : "memory", "f0", "f1", "f8", "f9", "f10", "f11"
    );
}

/* Conv_first: FP32 input → FP32 output, then quantize to int8.
 * Writes into the padded activation buffer at (y+PAD, x+PAD).
 *
 * VPU-vectorized: 64 OCs computed as 8 vectors of 8 OCs each. Each vector
 * uses vpu_conv_first_8oc to do 9 taps × 8 OCs in one fmadd.ps chain.
 *
 * Reads w0_packed (repacked at kernel startup to OC-vector layout) so the
 * weight loads are contiguous flq2 stride-32. */
static void conv_first_to_int8(const float *input, const float *w0_packed,
                                const float *b0,
                                int8_t *act_pad, float inv_a_scale_out,
                                uint32_t row0, uint32_t row1)
{
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            /* Gather 9 input values (boundary-resolved to 0). Cheap — one
             * scalar fp32 read per tap. */
            float inputs[9];
            for (uint32_t k = 0; k < 9u; k++) {
                const int ky = (int)(k / 3u);
                const int kx = (int)(k % 3u);
                const int yy = (int)y + ky - 1;
                const int xx = (int)x + kx - 1;
                inputs[k] = (yy >= 0 && yy < (int)IMG_H &&
                             xx >= 0 && xx < (int)IMG_W)
                            ? input[yy * IMG_W + xx] : 0.0f;
            }

            /* 8 OC-vectors × 8 OCs each = 64 OCs total */
            int32_t staging[CH] __attribute__((aligned(64)));
            for (uint32_t v = 0; v < 8u; v++) {
                vpu_conv_first_8oc(&b0[v * 8u],
                                    &w0_packed[v * 9u * 8u],   /* [v][k=0..8][oc=0..7] */
                                    inputs,
                                    inv_a_scale_out,
                                    &staging[v * 8u]);
            }

            /* Scatter 64 int8 values into NHWC-padded dst. Stride 1 within
             * IC, so this is contiguous within one (y, x) cache line. */
            int8_t *dst_p = &act_pad[((uint64_t)(y + PAD) * W_PAD + (x + PAD)) * CH];
            for (uint32_t oc = 0; oc < CH; oc++) {
                dst_p[oc] = (int8_t)staging[oc];
            }
        }
    }
}

/* Pack B for one output tile starting at (oy_base, ox_base, j=0..15).
 *
 * Layout: bpack[tap=0..8][k0=0..15][lane=0..63]
 *   lane = j*4 + d, d ∈ [0..3]
 *   bpack[tap, k0, j*4+d] = act_pad[y_in_pad, x_in_pad][4*k0 + d]
 * where (with padded layout, real data at indices 1..H, 1..W):
 *   y_in_pad = oy_base + ky      (the +PAD=+1 cancels the -1 conv shift)
 *   x_in_pad = ox_base + kx + j
 *
 * No boundary checks needed — halo cells are pre-zeroed in act_pad.
 *
 * VPU path: per (tap, j), one 64-byte src cache line is gathered (2× flq2)
 * and scattered to 16 dst positions (k0*64 + j*4) using two 8-lane fscw.ps
 * with constant offset-vectors. The 64-byte source reads land on contiguous
 * L1D lines; the 16 scattered destinations stay warm in L1D across j. */
static const int32_t SCATTER_OFS_LO[8] __attribute__((aligned(32))) =
    {0, 64, 128, 192, 256, 320, 384, 448};
static const int32_t SCATTER_OFS_HI[8] __attribute__((aligned(32))) =
    {512, 576, 640, 704, 768, 832, 896, 960};

static inline void pack_b_tile(const int8_t *src_pad, int8_t *bpack,
                                uint32_t oy_base, uint32_t ox_base)
{
    for (uint32_t ky = 0; ky < K; ky++) {
        const uint32_t y_in_pad = oy_base + ky;
        for (uint32_t kx = 0; kx < K; kx++) {
            const uint32_t tap = ky * K + kx;
            int8_t *const tap_base = &bpack[tap * (K0_PER_DISPATCH * 64u)];
            for (uint32_t j = 0; j < TILE_W; j++) {
                const uint32_t x_in_pad = ox_base + kx + j;
                const int8_t *src_p =
                    &src_pad[((uint64_t)y_in_pad * W_PAD + x_in_pad) * CH];
                const uint32_t off = j * 4u;
                for (uint32_t k0 = 0; k0 < K0_PER_DISPATCH; k0++) {
                    *(uint32_t *)&tap_base[k0 * 64u + off] =
                        *(const uint32_t *)&src_p[k0 * 4u];
                }
            }
        }
    }
}

/* Sub-phase cycle accumulators for ONE hidden layer (hart 0, layer 0).
 * Dumped at PHASE_TS_OFFSET + 24 × 8 (= 0xA000C0) as 4 uint64_t. */
struct sub_phase_acc {
    uint64_t pack_b;
    uint64_t tfma;
    uint64_t store_evict;
    uint64_t marshal;
};

/* T1 path: produce B-pack for each tile (oy, ox) in this minion's stripe and
 * publish via the per-minion pipe. Bounded to 1 tile ahead of T0.
 *
 * NOTE: Erbium SMT siblings (T0/T1 on one minion) do NOT share L1D; sync
 * through pipe counters must go via L2 (evict our writes, invalidate before
 * reads — same model as cross-minion coherence). */
static void conv_hidden_t1_produce(const int8_t *src,
                                    struct minion_pipe *pipe,
                                    uint32_t row0, uint32_t row1,
                                    uint32_t base_tile_idx)
{
    uint32_t tile_idx = base_tile_idx;
    for (uint32_t y_out = row0; y_out < row1; y_out++) {
        for (uint32_t x_out = 0; x_out < IMG_W; x_out += TILE_W) {
            /* Wait for T0 to free this slot. Invalidate-and-reload pipe
             * counter every poll (T0's writes may live in T0's L1D). */
            for (;;) {
                evict((const void *)&pipe->consumed, 4u);
                WAIT_CACHEOPS;
                FENCE;
                __asm__ __volatile__("" ::: "memory");
                if (pipe->consumed + 2u > tile_idx) break;
            }
            int8_t *const bpack = pipe->bpack[tile_idx & 1u];

            pack_b_tile(src, bpack, y_out, x_out);

            /* Push our writes to L2 so T0's tensor_load can see them. */
            __asm__ __volatile__("" ::: "memory");
            FENCE;
            evict((const void *)bpack, BPACK_BYTES_PER_SLOT);
            WAIT_CACHEOPS;
            FENCE;
            __asm__ __volatile__("" ::: "memory");

            /* Publish: tile_idx is now ready in slot (tile_idx & 1).
             * Evict the flag write so T0's poll (which invalidates its L1D)
             * pulls our update from L2. */
            pipe->produced = tile_idx + 1u;
            __asm__ __volatile__("" ::: "memory");
            FENCE;
            evict((const void *)&pipe->produced, 4u);
            WAIT_CACHEOPS;
            FENCE;
            tile_idx++;
        }
    }
}

/* T0 path: consume B-pack from pipe, run TFMA + marshal to dst. */
static void conv_hidden_t0_consume(const int8_t *layer_a_pack,
                                    const float *layer_w_scales,
                                    const float *layer_b,
                                    float a_scale_in, float inv_a_scale_out,
                                    int8_t *dst,
                                    struct minion_pipe *pipe,
                                    int32_t *outbuf,
                                    uint32_t row0, uint32_t row1,
                                    uint32_t base_tile_idx,
                                    uint32_t hart_id, uint32_t layer_idx,
                                    uint8_t *base);

/* Hidden conv via TFMA INT8 (legacy single-thread path; both pack and consume
 * on T0). Used when DNCNN_SMT_PIPELINE is 0. */
static void conv_hidden_tfma_int8(const int8_t *src, int8_t *dst,
                                   const int8_t *layer_a_pack,
                                   const float *layer_w_scales,
                                   const float *layer_b,
                                   float a_scale_in, float inv_a_scale_out,
                                   uint32_t row0, uint32_t row1,
                                   int8_t *bpack, int32_t *outbuf,
                                   uint32_t hart_id, uint32_t layer_idx,
                                   uint8_t *base)
{
    /* Sub-phase accumulators (only meaningful for hart 0, layer 0). */
    struct sub_phase_acc acc = {0, 0, 0, 0};
    const uint32_t profile_layer = (hart_id == 0u && layer_idx == 0u);

    for (uint32_t y_out = row0; y_out < row1; y_out++) {
        for (uint32_t x_out = 0; x_out < IMG_W; x_out += TILE_W) {
            const uint64_t t_pack0 = profile_layer ? read_cycles() : 0u;
            /* Pack B once per spatial tile (used by all OC tiles). */
            pack_b_tile(src, bpack, y_out, x_out);

            /* Compiler memory barrier: ensure pack stores are seen by the
             * subsequent CSR-driven evict/tensor_load. evict() is asm without
             * a memory clobber, so without this barrier DCE can elide the
             * pack stores entirely. */
            __asm__ __volatile__("" ::: "memory");
            FENCE;
            evict((const void *)bpack, BPACK_BYTES_PER_SLOT);
            WAIT_CACHEOPS;
            FENCE;
            const uint64_t t_pack1 = profile_layer ? read_cycles() : 0u;
            if (profile_layer) acc.pack_b += (t_pack1 - t_pack0);

            for (uint32_t oc_tile = 0; oc_tile < OC_TILES; oc_tile++) {
                const uint64_t t_tfma0 = profile_layer ? read_cycles() : 0u;
                const int8_t *const a_base =
                    layer_a_pack + (uint64_t)oc_tile * 9u * OC_PER_TILE * CH;

#ifdef DNCNN_USE_SCALAR_REF
                /* Bisect path: skip TFMA, fill outbuf via scalar dot product
                 * over the same A-pack and B-pack so we can verify the outer
                 * tile loop / marshalling without the TFMA hardware path. */
                (void)a_base;  /* unused if we use B-pack directly */
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    for (uint32_t j = 0; j < TILE_W; j++) {
                        int32_t acc = 0;
                        for (uint32_t tap = 0; tap < 9u; tap++) {
                            const int8_t *a_line =
                                a_base + (uint64_t)tap * (OC_PER_TILE * CH) + (uint64_t)r * CH;
                            for (uint32_t k = 0; k < 64u; k += 4u) {
                                const uint32_t k0 = k >> 2;
                                const int8_t *b_line =
                                    bpack + (uint64_t)tap * (K0_PER_DISPATCH * 64u) + (uint64_t)k0 * 64u;
                                const int32_t a1 = a_line[k+0];
                                const int32_t a2 = a_line[k+1];
                                const int32_t a3 = a_line[k+2];
                                const int32_t a4 = a_line[k+3];
                                const int32_t b1 = b_line[j*4+0];
                                const int32_t b2 = b_line[j*4+1];
                                const int32_t b3 = b_line[j*4+2];
                                const int32_t b4 = b_line[j*4+3];
                                acc += a1*b1 + a2*b2 + a3*b3 + a4*b4;
                            }
                        }
                        ((volatile int32_t *)outbuf)[r * TILE_W + j] = acc;
                    }
                }
#else
#ifdef DNCNN_ACOLS16
                /* 36 dispatches per tile: 9 taps × 4 IC sub-slabs of 16 each.
                 * Mirrors the validated tfma_conv3x3_int8.c pattern. */
                for (uint32_t tap = 0; tap < 9u; tap++) {
                    const uintptr_t a_addr =
                        (uintptr_t)a_base + (uint64_t)tap * (OC_PER_TILE * CH);
                    const uintptr_t b_addr =
                        (uintptr_t)bpack + (uint64_t)tap * (K0_PER_DISPATCH * 64u);

                    /* A: 16 lines × 64 IC. We'll re-use this load across the 4
                     * sub-dispatches via aoffset (in 4-byte units). */
                    tensor_load(0, 0, /*dst*/  0, 0, 0, a_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    /* B: 16 lines × 64 lanes (will be split 4-line slabs below). */
                    tensor_load(0, 0, /*dst*/ 16, 0, 0, b_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);

                    for (uint32_t s = 0; s < 4u; s++) {
                        /* Sub-slab s covers IC [16s, 16s+16). aoffset is
                         * encoded in 4-byte units → field = 4*s. */
                        const uint32_t aoff_field = 4u * s;
                        const uint32_t scp_b = 16u + 4u * s;   /* 4 B lines per slab */
                        const uint32_t is_last = (tap == 8u) && (s == 3u);
                        const uint32_t is_first = (tap == 0u) && (s == 0u);
                        tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                                   /*aoffset*/ aoff_field, /*tenc_loc*/ is_last,
                                   /*tenb_uns*/ 0, /*tena_uns*/ 0,
                                   /*tenb_loc*/ 0, /*scp_loc_b*/ scp_b, /*scp_loc_a*/ 0,
                                   /*type=INT8*/ 3u, /*first_pass*/ is_first);
                        tensor_wait(TENSOR_FMA_WAIT);
                    }
                }
#else
                for (uint32_t tap = 0; tap < 9u; tap++) {
                    const uintptr_t a_addr =
                        (uintptr_t)a_base + (uint64_t)tap * (OC_PER_TILE * CH);
                    const uintptr_t b_addr =
                        (uintptr_t)bpack + (uint64_t)tap * (K0_PER_DISPATCH * 64u);

                    /* A operand: 16 lines (one per OC), 64 IC int8 each. */
                    tensor_load(0, 0, /*dst*/  0, 0, 0, a_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    /* B operand: 16 lines (k0=0..15), 64 lanes int8 each. */
                    tensor_load(0, 0, /*dst*/ 16, 0, 0, b_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);

                    /* INT8 signed×signed; tenc_loc=1 on last to copy TenC→FREGs. */
                    const uint32_t last = (tap == 8u);
                    tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                               /*aoffset*/ 0, /*tenc_loc*/ last,
                               /*tenb_uns*/ 0, /*tena_uns*/ 0,
                               /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                               /*type=INT8*/ 3u, /*first_pass*/ (tap == 0u));
                    tensor_wait(TENSOR_FMA_WAIT);
                }
#endif

                const uint64_t t_tfma1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.tfma += (t_tfma1 - t_tfma0);
                const uint64_t t_store0 = profile_layer ? read_cycles() : 0u;

                /* Store 16 OC × 16 P int32 to outbuf. cols=3 means 16 32-bit
                 * elements per row. */
                tensor_store(/*reg_stride*/ 0, /*start_reg*/ 0, /*cols*/ 3,
                             /*Arows*/ 0xF, (uintptr_t)outbuf,
                             /*coop_store*/ 0, /*stride*/ 64);
                tensor_wait(TENSOR_STORE_WAIT);

                /* Make tensor_store output visible to scalar reads. The
                 * tensor_store is a CSR write without compiler-visible memory
                 * effect, so we add an explicit barrier to keep the read
                 * after this evict. */
                __asm__ __volatile__("" ::: "memory");
                FENCE;
                evict((const void *)outbuf, OC_PER_TILE * TILE_W * sizeof(int32_t));
                WAIT_CACHEOPS;
                FENCE;
                const uint64_t t_store1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.store_evict += (t_store1 - t_store0);
                /* tensor_fma with tenc_loc=1 writes int32 results into FREGs
                 * (the floating-point register file). The asm CSR write does
                 * not declare an FP-register clobber, so the compiler may have
                 * hoisted FP values like a_scale_in / inv_a_scale_out into
                 * FREGs that TFMA just trashed. Force a full FREG spill+reload
                 * before the FP-heavy marshalling that follows. */
                __asm__ __volatile__("" ::: "memory",
                    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
                    "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
                    "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
                    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
#endif

                /* Debug dump: hart 0, layer 0, ALL 20 tiles in row 0.
                 * Layout: tile_x at +0x2000*tile_x. 20 × 0x2000 = 0x28000.
                 * Each tile slot has 4 oc_tiles × 1024 B = 4096 B (in 8 KB slot). */
                if (hart_id == 0u && layer_idx == 0u && y_out == row0) {
                    const uint32_t tile_off = (x_out / TILE_W) * 0x2000u;
                    volatile int32_t *dbg = (volatile int32_t *)
                        (base + DEBUG_TILE_OFFSET + tile_off
                         + oc_tile * (OC_PER_TILE * TILE_W) * sizeof(int32_t));
                    for (uint32_t r2 = 0; r2 < OC_PER_TILE; r2++)
                        for (uint32_t j2 = 0; j2 < TILE_W; j2++)
                            dbg[r2 * TILE_W + j2] = ((volatile int32_t *)outbuf)[r2 * TILE_W + j2];
                    evict((const void *)dbg, OC_PER_TILE * TILE_W * sizeof(int32_t));
                    WAIT_CACHEOPS;
                    FENCE;
                }

                const uint64_t t_marsh0 = profile_layer ? read_cycles() : 0u;

                /* Marshal: dequant + bias + ReLU + requant → int8 dst.
                 *
                 * Combined-form math:
                 *   cs[r] = w_scale[oc] * a_scale_in * inv_a_scale_out
                 *   cb[r] = bias[oc]    * inv_a_scale_out
                 *   out_int8 = sat_int8(round( max(0, fmadd(cs, v, cb)) ))
                 *
                 * (Original was sat_int8(relu(v*w_scale*a_scale + bias) * inv_a_scale).
                 * Folding inv_a_scale into both terms is exact for ReLU input ≥ 0
                 * and matches the original to 0 elsewhere.) */
                const uint32_t oc_offset = oc_tile * OC_PER_TILE;
                const float *const ws_oc = &layer_w_scales[oc_offset];
                const float *const b_oc  = &layer_b[oc_offset];
                const float a_scale_combined = a_scale_in * inv_a_scale_out;

#ifdef DNCNN_SCALAR_MARSHAL
                volatile const int32_t *const vout = (volatile const int32_t *)outbuf;
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    const uint32_t oc = oc_offset + r;
                    const float scale_r = ws_oc[r] * a_scale_in;
                    const float bias_r = b_oc[r];
                    for (uint32_t j = 0; j < TILE_W; j++) {
                        const int32_t v = vout[r * TILE_W + j];
                        const float dq = (float)v * scale_r + bias_r;
                        const float rl = relu_f32(dq);
                        const uint32_t x_pad = x_out + PAD + j;
                        dst[((uint64_t)(y_out + PAD) * W_PAD + x_pad) * CH + oc] =
                            sat_int8(rl * inv_a_scale_out);
                    }
                }
#else
                /* VPU path: compute combined scale/bias scalar once per OC,
                 * then 16 lanes via fmadd.ps + fmax/fmin + fcvt.pw.ps. */
                int32_t staging[OC_PER_TILE * TILE_W] __attribute__((aligned(64)));
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    const float cs_r = ws_oc[r] * a_scale_combined;
                    const float cb_r = b_oc[r] * inv_a_scale_out;
                    vpu_marshal_row16(&((const int32_t *)outbuf)[r * TILE_W],
                                      cs_r, cb_r, &staging[r * TILE_W]);
                }
                /* Scatter staging → dst (padded NHWC, stride 64 B per x). */
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    const uint32_t oc = oc_offset + r;
                    for (uint32_t j = 0; j < TILE_W; j++) {
                        const uint32_t x_pad = x_out + PAD + j;
                        dst[((uint64_t)(y_out + PAD) * W_PAD + x_pad) * CH + oc] =
                            (int8_t)staging[r * TILE_W + j];
                    }
                }
#endif
                const uint64_t t_marsh1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.marshal += (t_marsh1 - t_marsh0);
            }
        }
    }

    /* Dump sub-phase accumulators (hart 0, layer 0 only). */
    if (profile_layer) {
        volatile uint64_t *acc_dump = (volatile uint64_t *)
            (base + PHASE_TS_OFFSET + 24u * 8u);
        acc_dump[0] = acc.pack_b;
        acc_dump[1] = acc.tfma;
        acc_dump[2] = acc.store_evict;
        acc_dump[3] = acc.marshal;
        evict((const void *)acc_dump, 4u * sizeof(uint64_t));
        WAIT_CACHEOPS;
        FENCE;
    }
}

/* T0 consume path: pull B-pack from pipe (T1 produced), run TFMA + marshal. */
static void conv_hidden_t0_consume(const int8_t *layer_a_pack,
                                    const float *layer_w_scales,
                                    const float *layer_b,
                                    float a_scale_in, float inv_a_scale_out,
                                    int8_t *dst,
                                    struct minion_pipe *pipe,
                                    int32_t *outbuf,
                                    uint32_t row0, uint32_t row1,
                                    uint32_t base_tile_idx,
                                    uint32_t hart_id, uint32_t layer_idx,
                                    uint8_t *base)
{
    struct sub_phase_acc acc = {0, 0, 0, 0};
    const uint32_t profile_layer = (hart_id == 0u && layer_idx == 0u);
    uint32_t tile_idx = base_tile_idx;

    for (uint32_t y_out = row0; y_out < row1; y_out++) {
        for (uint32_t x_out = 0; x_out < IMG_W; x_out += TILE_W) {
            const uint64_t t_wait0 = profile_layer ? read_cycles() : 0u;

            /* Wait for T1 (SMT sibling) to publish tile_idx. Invalidate L1D
             * each poll to pick up T1's write that lives in T1's L1D /
             * L2-evicted state. */
            for (;;) {
                evict((const void *)&pipe->produced, 4u);
                WAIT_CACHEOPS;
                FENCE;
                __asm__ __volatile__("" ::: "memory");
                if (pipe->produced > tile_idx) break;
            }
            const int8_t *const bpack = pipe->bpack[tile_idx & 1u];

            const uint64_t t_wait1 = profile_layer ? read_cycles() : 0u;
            if (profile_layer) acc.pack_b += (t_wait1 - t_wait0);  /* now: T1-wait time */

            for (uint32_t oc_tile = 0; oc_tile < OC_TILES; oc_tile++) {
                const uint64_t t_tfma0 = profile_layer ? read_cycles() : 0u;
                const int8_t *const a_base =
                    layer_a_pack + (uint64_t)oc_tile * 9u * OC_PER_TILE * CH;

                for (uint32_t tap = 0; tap < 9u; tap++) {
                    const uintptr_t a_addr =
                        (uintptr_t)a_base + (uint64_t)tap * (OC_PER_TILE * CH);
                    const uintptr_t b_addr =
                        (uintptr_t)bpack + (uint64_t)tap * (K0_PER_DISPATCH * 64u);
                    tensor_load(0, 0, /*dst*/  0, 0, 0, a_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    tensor_load(0, 0, /*dst*/ 16, 0, 0, b_addr, 0, 15, 64, 0);
                    tensor_wait(TENSOR_LOAD_WAIT_0);
                    const uint32_t last = (tap == 8u);
                    tensor_fma(/*tmask*/ 0, BCOLS_FIELD, AROWS_FIELD, ACOLS_FIELD,
                               /*aoffset*/ 0, /*tenc_loc*/ last,
                               /*tenb_uns*/ 0, /*tena_uns*/ 0,
                               /*tenb_loc*/ 0, /*scp_loc_b*/ 16, /*scp_loc_a*/ 0,
                               /*type=INT8*/ 3u, /*first_pass*/ (tap == 0u));
                    tensor_wait(TENSOR_FMA_WAIT);
                }

                const uint64_t t_tfma1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.tfma += (t_tfma1 - t_tfma0);
                const uint64_t t_store0 = profile_layer ? read_cycles() : 0u;

                tensor_store(/*reg_stride*/ 0, /*start_reg*/ 0, /*cols*/ 3,
                             /*Arows*/ 0xF, (uintptr_t)outbuf,
                             /*coop_store*/ 0, /*stride*/ 64);
                tensor_wait(TENSOR_STORE_WAIT);
                __asm__ __volatile__("" ::: "memory");
                FENCE;
                evict((const void *)outbuf, OC_PER_TILE * TILE_W * sizeof(int32_t));
                WAIT_CACHEOPS;
                FENCE;
                const uint64_t t_store1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.store_evict += (t_store1 - t_store0);
                __asm__ __volatile__("" ::: "memory",
                    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
                    "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
                    "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
                    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");

                const uint64_t t_marsh0 = profile_layer ? read_cycles() : 0u;

                /* Marshal: dequant + bias + ReLU + requant → int8 dst. VPU path. */
                const uint32_t oc_offset = oc_tile * OC_PER_TILE;
                const float *const ws_oc = &layer_w_scales[oc_offset];
                const float *const b_oc  = &layer_b[oc_offset];
                const float a_scale_combined = a_scale_in * inv_a_scale_out;
                int32_t staging[OC_PER_TILE * TILE_W] __attribute__((aligned(64)));
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    const float cs_r = ws_oc[r] * a_scale_combined;
                    const float cb_r = b_oc[r] * inv_a_scale_out;
                    vpu_marshal_row16(&((const int32_t *)outbuf)[r * TILE_W],
                                      cs_r, cb_r, &staging[r * TILE_W]);
                }
                for (uint32_t r = 0; r < OC_PER_TILE; r++) {
                    const uint32_t oc = oc_offset + r;
                    for (uint32_t j = 0; j < TILE_W; j++) {
                        const uint32_t x_pad = x_out + PAD + j;
                        dst[((uint64_t)(y_out + PAD) * W_PAD + x_pad) * CH + oc] =
                            (int8_t)staging[r * TILE_W + j];
                    }
                }

                const uint64_t t_marsh1 = profile_layer ? read_cycles() : 0u;
                if (profile_layer) acc.marshal += (t_marsh1 - t_marsh0);
            }

            /* Tile fully consumed — release this slot to T1 (evict so T1's
             * invalidate-and-reload poll sees our update). */
            pipe->consumed = tile_idx + 1u;
            __asm__ __volatile__("" ::: "memory");
            FENCE;
            evict((const void *)&pipe->consumed, 4u);
            WAIT_CACHEOPS;
            FENCE;
            tile_idx++;
        }
    }

    if (profile_layer) {
        volatile uint64_t *acc_dump = (volatile uint64_t *)
            (base + PHASE_TS_OFFSET + 24u * 8u);
        acc_dump[0] = acc.pack_b;
        acc_dump[1] = acc.tfma;
        acc_dump[2] = acc.store_evict;
        acc_dump[3] = acc.marshal;
        evict((const void *)acc_dump, 4u * sizeof(uint64_t));
        WAIT_CACHEOPS;
        FENCE;
    }
}

/* Conv_final: int8 in (padded layout) → FP32 residual subtract output.
 * Padded read: act_pad[(y+ky)*W_PAD + (x+kx)] (the +1 pad shift cancels the
 * -1 conv shift). No boundary checks needed — halo is zero.
 *
 * NOTE: We tried VPU spatial-vec with fgb.ps gather + fcvt.ps.pw + fmadd.ps
 * (8 outputs per pass, vectorize across spatial). It was ~85% SLOWER than
 * this scalar form (175 ms vs 95 ms) because the compiler can't keep
 * accumulator/offset FP regs live across separate inline-asm blocks — each
 * inner iter spills+reloads. To fix that we'd need to write the entire
 * (ky, kx, ic) inner loop as ONE asm block with internal counter management.
 * Not worth the complexity for ~95 ms (4% of total wait). Left as future work. */
static void conv_final_int8_to_fp32(const int8_t *act_pad, float a_scale_in,
                                    const float *input_image,
                                    const float *wf, const float *bf,
                                    float *output, uint32_t row0, uint32_t row1)
{
    for (uint32_t y = row0; y < row1; y++) {
        for (uint32_t x = 0; x < IMG_W; x++) {
            float acc = bf[0];
            for (uint32_t ky = 0; ky < K; ky++) {
                for (uint32_t kx = 0; kx < K; kx++) {
                    for (uint32_t ic = 0; ic < CH; ic++) {
                        const float v = (float)act_pad[((uint64_t)(y + ky) * W_PAD + (x + kx)) * CH + ic] * a_scale_in;
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
    if (raw_hart >= 16u) return 0;
    const uint32_t minion_id = raw_hart >> 1;
#if DNCNN_SMT_PIPELINE
    const bool is_t0 = ((raw_hart & 1u) == 0u);
#else
    if ((raw_hart & 1u) != 0u) return 0;
    const bool is_t0 = true;
#endif
    /* hart_id (logical) = minion_id (8 minions doing real compute work) */
    const uint32_t hart_id = minion_id;

    setM0MaskFF();

    uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
#if DNCNN_STREAMING
    /* In streaming mode, host DMA's the input image to base+STREAM_INPUT_OFFSET
     * for each frame. The baked-in INPUT_BLOB is unused. */
    const float *const input    = (const float *)(base + STREAM_INPUT_OFFSET);
#else
    const float *const input    = INPUT_BLOB;
#endif
    const float *const reference = REF_BLOB;
    int8_t      *const act0 = static_act0;
    int8_t      *const act1 = static_act1;
    int32_t     *const outbuf = static_outbuf[minion_id];
    struct minion_pipe *const pipe = &minion_pipes[minion_id];

    volatile struct dncnn_slot *const slots = (volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
    volatile struct dncnn_summary *const summary = (volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);

    const uint32_t row0 = (IMG_H * minion_id) / ACTIVE_MINIONS;
    const uint32_t row1 = (IMG_H * (minion_id + 1u)) / ACTIVE_MINIONS;

    /* Init pipe counters (T0 of each minion does this). */
    if (is_t0) {
        pipe->produced = 0u;
        pipe->consumed = 0u;
    }

    /* Per-hart progress marker (debug aid for hangs). At 0xB00000:
     *   uint32_t progress[16] indexed by raw_hart, last-written value tells
     *   us how far that hart got. */
    volatile uint32_t *const progress = (volatile uint32_t *)(base + 0xB00000u);
    progress[raw_hart] = 1u;
    evict((const void *)&progress[raw_hart], 4u);
    WAIT_CACHEOPS;
    FENCE;

    /* Cooperative zero of static_act0 + static_act1 (halo rows are read by
     * pack_b_tile but never written by conv outputs; with -fno-zero-init-bss
     * they would otherwise hold arbitrary memory contents). All BARRIER_HARTS
     * harts (T0+T1) cooperate to halve init time. */
    {
        const uint64_t buf_bytes = (uint64_t)ACT_PAD_PIXELS * CH;
        const uint64_t bytes_per_hart = (buf_bytes + BARRIER_HARTS - 1u) / BARRIER_HARTS;
        const uint64_t off0 = (uint64_t)raw_hart * bytes_per_hart;
        const uint64_t off1 = (off0 + bytes_per_hart > buf_bytes) ? buf_bytes : (off0 + bytes_per_hart);
        if (off0 < off1) {
            uint64_t *p0 = (uint64_t *)((uintptr_t)static_act0 + off0);
            uint64_t *p1 = (uint64_t *)((uintptr_t)static_act1 + off0);
            const uint64_t n8 = (off1 - off0) / 8u;
            for (uint64_t i = 0; i < n8; i++) { p0[i] = 0; p1[i] = 0; }
            evict((const void *)((uintptr_t)static_act0 + off0), off1 - off0);
            evict((const void *)((uintptr_t)static_act1 + off0), off1 - off0);
            WAIT_CACHEOPS;
            FENCE;
        }
    }
    bench_barrier();

    /* Copy A-pack to 64-byte-aligned static buffer so tensor_load addresses
     * round to the same address (no silent off-by-60-byte). All harts copy a
     * disjoint stripe to parallelize. Use byte copy: src is only 4-byte
     * aligned (linker placed it at +0x3c offset within a section), so 8-byte
     * loads would be misaligned. */
    {
        const uint32_t bytes_per_hart = (A_PACK_BYTES + ACTIVE_HARTS - 1u) / ACTIVE_HARTS;
        const uint32_t off0 = hart_id * bytes_per_hart;
        const uint32_t off1 = (off0 + bytes_per_hart > A_PACK_BYTES) ? A_PACK_BYTES : (off0 + bytes_per_hart);
        const int8_t *src = WA_INT8_BLOB + off0;
        int8_t       *dst = static_a_pack + off0;
        const uint32_t n  = off1 - off0;
        for (uint32_t i = 0; i < n; i++) dst[i] = src[i];
        evict((const void *)((uintptr_t)static_a_pack + off0), n);
        WAIT_CACHEOPS;
        FENCE;
    }
    bench_barrier();

    /* Repack conv_first weights from [OC=64][9] (linker layout) into
     * [v=0..7][k=0..8][oc_in_v=0..7] for VPU-vectorized conv_first.
     * Hart 0 only — only 2304 B = trivial. */
    if (raw_hart == 0u) {
        for (uint32_t v = 0; v < 8u; v++) {
            for (uint32_t k = 0; k < 9u; k++) {
                for (uint32_t oc_in_v = 0; oc_in_v < 8u; oc_in_v++) {
                    const uint32_t oc = v * 8u + oc_in_v;
                    static_w0_packed[v * (9u * 8u) + k * 8u + oc_in_v] =
                        W0_BLOB[oc * 9u + k];
                }
            }
        }
        /* Pre-scale conv_final weights by a_scale_in (the activation scale
         * for the LAST hidden layer's output, which is conv_final's input). */
        const float a_scale_last = AS_BLOB[HIDDEN_LAYERS];
        for (uint32_t k = 0; k < 9u; k++) {
            for (uint32_t ic = 0; ic < CH; ic++) {
                static_wf_scaled[k * CH + ic] = WF_BLOB[k * CH + ic] * a_scale_last;
            }
        }
        evict((const void *)static_w0_packed, sizeof(static_w0_packed));
        evict((const void *)static_wf_scaled, sizeof(static_wf_scaled));
        WAIT_CACHEOPS;
        FENCE;
    }
    bench_barrier();

    /* Phase timestamps. With multiple inferences, each one gets a 24-slot
     * block at PHASE_TS_OFFSET + img_idx * (24*8). Slot meanings unchanged. */
    volatile uint64_t *const phase_ts_base = (volatile uint64_t *)(base + PHASE_TS_OFFSET);

    const uint32_t total_tiles_per_layer = (row1 - row0) * (IMG_W / TILE_W);

  /* === Outer loop: N back-to-back inferences === */
  for (uint32_t img_idx = 0u; img_idx < (uint32_t)DNCNN_N_INFERENCES; img_idx++) {
    volatile uint64_t *const phase_ts = phase_ts_base + (uint64_t)img_idx * 24u;
    float *const final_output_i = (float *)(base + OUTPUT_OFFSET +
                                             (uint64_t)img_idx * OUTPUT_STRIDE_BYTES);

    if (hart_id == 0u && is_t0) phase_ts[0] = read_cycles();

#if DNCNN_STREAMING
    /* Streaming mode: host DMA'd a fresh image into DRAM at STREAM_INPUT_OFFSET.
     * Invalidate this hart's L1D for the input region so we read from DRAM. */
    if (is_t0) {
        evict((const void *)input, IMG_PIXELS * sizeof(float));
        WAIT_CACHEOPS;
        FENCE;
    }
    bench_barrier();
#endif

    /* conv_first VPU → int8 (writes padded layout). T0 only. Uses repacked
     * weights for OC-vector flq2 access. */
    if (is_t0) {
        conv_first_to_int8(input, static_w0_packed, B0_BLOB, act0,
                            INV_AS_BLOB[0], row0, row1);
        /* VPU writes to FREGs; hidden conv path reads FP scalars right after
         * (a_scale_in etc.). Force GCC to spill+reload its FP-cached values. */
        __asm__ __volatile__("" ::: "memory",
            "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
            "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
            "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
            "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
        const uint64_t ROW_B = (uint64_t)W_PAD * CH;
        evict((const void *)((uintptr_t)act0 + (uint64_t)(row0 + PAD) * ROW_B),
              (uint64_t)(row1 - row0) * ROW_B);
        WAIT_CACHEOPS;
    }
    if (hart_id == 0u && is_t0) phase_ts[1] = read_cycles();
    bench_barrier();
    if (hart_id == 0u && is_t0) phase_ts[2] = read_cycles();

    /* 18 hidden layers via TFMA. */
    for (uint32_t L = 0; L < HIDDEN_LAYERS; L++) {
        const int8_t *const src = (L & 1u) ? act1 : act0;
        int8_t       *const dst = (L & 1u) ? act0 : act1;
        const int8_t *const a_pack = static_a_pack + (uint64_t)L * OC_TILES * 9u * OC_PER_TILE * CH;
        const float  *const w_scl  = WS_BLOB + (uint64_t)L * CH;
        const float  *const b      = BH_BLOB + (uint64_t)L * CH;
        const float a_scale_in = AS_BLOB[L];
        const float inv_a_out  = INV_AS_BLOB[L + 1u];

        /* Halo eviction (T0 only — T0 and T1 share L1D within a minion, so
         * one evict invalidates for both). */
        if (is_t0) {
            const uint64_t ROW_B = (uint64_t)W_PAD * CH;
            if (row0 > 0u)
                evict((const void *)((uintptr_t)src + (uint64_t)(row0 + PAD - 1u) * ROW_B), ROW_B);
            if (row1 < IMG_H)
                evict((const void *)((uintptr_t)src + (uint64_t)(row1 + PAD) * ROW_B), ROW_B);
            WAIT_CACHEOPS;
        }
        bench_barrier();   /* ensure halo is invalidated before T1 starts packing */

        const uint32_t base_tile_idx = L * total_tiles_per_layer;

#if DNCNN_SMT_PIPELINE
#ifdef DNCNN_SMT_T1_NOOP
        /* Bisect: T1 attends barriers but does no work; T0 runs the legacy
         * single-thread path that does pack itself. */
        if (is_t0) {
            conv_hidden_tfma_int8(src, dst, a_pack, w_scl, b, a_scale_in, inv_a_out,
                                  row0, row1, pipe->bpack[0], outbuf,
                                  hart_id, L, base);
        }
#else
        if (is_t0) {
            conv_hidden_t0_consume(a_pack, w_scl, b, a_scale_in, inv_a_out,
                                   dst, pipe, outbuf,
                                   row0, row1, base_tile_idx,
                                   hart_id, L, base);
        } else {
            conv_hidden_t1_produce(src, pipe, row0, row1, base_tile_idx);
        }
#endif
#else
        conv_hidden_tfma_int8(src, dst, a_pack, w_scl, b, a_scale_in, inv_a_out,
                              row0, row1, pipe->bpack[0], outbuf,
                              hart_id, L, base);
#endif

        if (is_t0) {
            const uint64_t ROW_B = (uint64_t)W_PAD * CH;
            evict((const void *)((uintptr_t)dst + (uint64_t)(row0 + PAD) * ROW_B),
                  (uint64_t)(row1 - row0) * ROW_B);
            WAIT_CACHEOPS;
        }
        progress[raw_hart] = 100u + L;
        evict((const void *)&progress[raw_hart], 4u);
        WAIT_CACHEOPS;
        bench_barrier();
        if (hart_id == 0u && is_t0) phase_ts[3 + L] = read_cycles();
    }

    /* conv_final scalar (T0 only). VPU attempt was slower; see comment in
     * conv_final_int8_to_fp32. */
    if (is_t0) {
        const int8_t *const last = (HIDDEN_LAYERS & 1u) ? act1 : act0;
        const float a_scale_last = AS_BLOB[HIDDEN_LAYERS];
        conv_final_int8_to_fp32(last, a_scale_last, input, WF_BLOB, BF_BLOB,
                                final_output_i, row0, row1);
        evict(final_output_i + row0 * IMG_W, (uint64_t)(row1 - row0) * IMG_W * sizeof(float));
        WAIT_CACHEOPS;
    }
    if (hart_id == 0u && is_t0) phase_ts[21] = read_cycles();
    bench_barrier();
    if (hart_id == 0u && is_t0) phase_ts[22] = read_cycles();
    if (hart_id == 0u && is_t0) phase_ts[23] = read_cycles();
  } /* === end outer N-inference loop === */

    /* Use the LAST inference's output for the audit / slot fill. */
    float *const final_output = (float *)(base + OUTPUT_OFFSET +
        (uint64_t)(DNCNN_N_INFERENCES - 1u) * OUTPUT_STRIDE_BYTES);

    /* Slot fill + summary (T0 only — T1 just barriers and exits). */
    if (!is_t0) {
        bench_barrier();   /* match T0's slot-fill barrier */
        bench_barrier();   /* match the trailing barrier at end */
        return 0;
    }

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
        flush_va(0, (uint64_t)slots, 8u, 64u, 0);
        WAIT_CACHEOPS;
        FENCE;
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
    if (hart_id == 0u) {
        evict((const void *)phase_ts_base, 24u * (uint64_t)DNCNN_N_INFERENCES * sizeof(uint64_t));
        WAIT_CACHEOPS;
        FENCE;
    }
    return 0;
}
