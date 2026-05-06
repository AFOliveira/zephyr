/*
 * Luxonis DnCNN3 240x320 FP32 benchmark for ET-SoC1 Erbium U-mode.
 *
 * This uses the real Luxonis ONNX topology and weights:
 *   - input:  [1, 1, 240, 320] FP32
 *   - layers: Conv/ReLU, 18x Conv/ReLU, Conv, then output = input - residual
 *   - hidden width: 64 channels
 *
 * The weights are loaded by the host launcher in a pre-packed layout.  Hidden
 * and final weights are arranged as {output channel, kernel point, input
 * channel}, so the VPU can consume the 64 input channels as four packed
 * 16-float dot products per kernel point.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif

#ifndef DNCNN_PASSES
#define DNCNN_PASSES 1u
#endif

#define DNCNN_MAGIC       0xD3C11004u
#ifndef IMG_W
#define IMG_W             320u
#endif

#ifndef IMG_H
#define IMG_H             240u
#endif
#define CH                64u
#ifndef HIDDEN_LAYERS
#define HIDDEN_LAYERS     18u
#endif
#define K                 3u

#define IMG_PIXELS        (IMG_W * IMG_H)
#define INPUT_BYTES       (IMG_PIXELS * sizeof(float))
#define OUTPUT_BYTES      INPUT_BYTES
#define ACT_BYTES         (IMG_PIXELS * CH * sizeof(float))

#define W0_FLOATS         (CH * K * K)
#define B0_FLOATS         CH
#define WH_PACK_FLOATS    (HIDDEN_LAYERS * CH * K * K * CH)
#define BH_FLOATS         (HIDDEN_LAYERS * CH)
#define WF_PACK_FLOATS    (K * K * CH)
#define BF_FLOATS         1u

#define W0_OFFSET_FLOATS  0u
#define B0_OFFSET_FLOATS  (W0_OFFSET_FLOATS + W0_FLOATS)
#define WH_OFFSET_FLOATS  (B0_OFFSET_FLOATS + B0_FLOATS)
#define BH_OFFSET_FLOATS  (WH_OFFSET_FLOATS + WH_PACK_FLOATS)
#define WF_OFFSET_FLOATS  (BH_OFFSET_FLOATS + BH_FLOATS)
#define BF_OFFSET_FLOATS  (WF_OFFSET_FLOATS + WF_PACK_FLOATS)
#define WEIGHTS_FLOATS    (BF_OFFSET_FLOATS + BF_FLOATS)

#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x0000u
#define SUMMARY_OFFSET    0x1000u
#define INPUT_OFFSET      0x2000u
#define WEIGHTS_OFFSET    0x60000u
#define OUTPUT_OFFSET     0x300000u
#define ACT0_OFFSET       0x400000u
#define ACT1_OFFSET       (ACT0_OFFSET + ACT_BYTES)
#define BUFFER_BYTES      (ACT1_OFFSET + ACT_BYTES)

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

#ifdef DNCNN_STATIC_BLOBS
#ifdef DNCNN_TILE_BLOBS
extern const unsigned char _binary_dncnn_tile_input_bin_start[];
extern const unsigned char _binary_dncnn_tile_ref_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn_tile_input_bin_start
#define DNCNN_REF_BLOB _binary_dncnn_tile_ref_bin_start
#elif defined(DNCNN_TILE1_BLOBS)
extern const unsigned char _binary_dncnn3_luxonis_1x1_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_1x1_ort_output_f32_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn3_luxonis_1x1_input_f32_bin_start
#define DNCNN_REF_BLOB _binary_dncnn3_luxonis_1x1_ort_output_f32_bin_start
#elif defined(DNCNN_TILE4_BLOBS)
extern const unsigned char _binary_dncnn3_luxonis_4x4_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_4x4_ort_output_f32_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn3_luxonis_4x4_input_f32_bin_start
#define DNCNN_REF_BLOB _binary_dncnn3_luxonis_4x4_ort_output_f32_bin_start
#elif defined(DNCNN_TILE16_BLOBS)
extern const unsigned char _binary_dncnn3_luxonis_16x16_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_16x16_ort_output_f32_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn3_luxonis_16x16_input_f32_bin_start
#define DNCNN_REF_BLOB _binary_dncnn3_luxonis_16x16_ort_output_f32_bin_start
#elif defined(DNCNN_TILE64_BLOBS)
extern const unsigned char _binary_dncnn3_luxonis_64x64_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_64x64_ort_output_f32_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn3_luxonis_64x64_input_f32_bin_start
#define DNCNN_REF_BLOB _binary_dncnn3_luxonis_64x64_ort_output_f32_bin_start
#else
extern const unsigned char _binary_dncnn3_luxonis_240x320_input_f32_bin_start[];
extern const unsigned char _binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start[];
#define DNCNN_INPUT_BLOB _binary_dncnn3_luxonis_240x320_input_f32_bin_start
#define DNCNN_REF_BLOB _binary_dncnn3_luxonis_240x320_ort_output_f32_bin_start
#endif
extern const unsigned char _binary_dncnn3_luxonis_240x320_weights_packed_f32_bin_start[];
#define DNCNN_WEIGHTS_BLOB _binary_dncnn3_luxonis_240x320_weights_packed_f32_bin_start

static float static_output[IMG_PIXELS] __attribute__((aligned(64)));
static float static_act0[IMG_PIXELS * CH] __attribute__((aligned(64)));
static float static_act1[IMG_PIXELS * CH] __attribute__((aligned(64)));
#endif

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
#ifdef DNCNN_SMT_HETERO_T0_ONLY
	/* All ACTIVE_HARTS (=8) participants are T0 of distinct minions 0..7. */
	uint32_t mask = 0;
	for (uint32_t m = 0; m < ACTIVE_HARTS; m++) {
		mask |= 1u << m;
	}
	return mask;
#else
	uint32_t mask = 0;

	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
#endif
}

static inline uint32_t active_mask_t1(void)
{
#ifdef DNCNN_SMT_HETERO_T0_ONLY
	/* No T1 participants in T0-only mode. */
	return 0;
#else
	uint32_t mask = 0;

	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
#endif
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

static inline float relu_f32(float v)
{
	return v < 0.0f ? 0.0f : v;
}

static inline uint32_t float_bits(float v)
{
	union {
		float f;
		uint32_t u;
	} x;

	x.f = v;
	return x.u;
}

static inline uint32_t mix_hash(uint32_t h, uint32_t v)
{
	h ^= v;
	h *= 16777619u;
	return h;
}

static inline float vpu_dot16_f32(const float *a, const float *b)
{
	__attribute__((aligned(32))) float tmp[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"flq2    f1, 0(%[a0])\n"
		"flq2    f2, 0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f1, 32(%[a0])\n"
		"flq2    f2, 32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2    f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [b0] "r"(b), [tmp] "r"(tmp)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum += tmp[i];
	}

	return sum;
}

static inline float dot64_vpu(const float *a, const float *b)
{
	float acc = 0.0f;

	acc += vpu_dot16_f32(a, b);
	acc += vpu_dot16_f32(a + 16u, b + 16u);
	acc += vpu_dot16_f32(a + 32u, b + 32u);
	acc += vpu_dot16_f32(a + 48u, b + 48u);

	return acc;
}

static inline void dot64x2_vpu(const float *a, const float *b0,
			       const float *b1, float *out0, float *out1)
{
#ifdef DNCNN_VPU_FUSED64X2
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"flq2    f1,   0(%[a])\n"
		"flq2    f2,   0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   0(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,  32(%[a])\n"
		"flq2    f2,  32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  32(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,  64(%[a])\n"
		"flq2    f2,  64(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  64(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,  96(%[a])\n"
		"flq2    f2,  96(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  96(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 128(%[a])\n"
		"flq2    f2, 128(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 128(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 160(%[a])\n"
		"flq2    f2, 160(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 160(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 192(%[a])\n"
		"flq2    f2, 192(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 192(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 224(%[a])\n"
		"flq2    f2, 224(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 224(%[b1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"fsq2    f0, 0(%[t0])\n"
		"fsq2    f3, 0(%[t1])\n"
		:
		: [zero] "r"(zero), [a] "r"(a), [b0] "r"(b0), [b1] "r"(b1),
		  [t0] "r"(tmp0), [t1] "r"(tmp1)
		: "memory", "f0", "f1", "f2", "f3", "f4");

	float s0 = 0.0f;
	float s1 = 0.0f;
	for (uint32_t i = 0; i < 8u; i++) {
		s0 += tmp0[i];
		s1 += tmp1[i];
	}
	*out0 = s0;
	*out1 = s1;
#else
	float acc0 = 0.0f;
	float acc1 = 0.0f;

	acc0 += vpu_dot16_f32(a, b0);
	acc1 += vpu_dot16_f32(a, b1);
	acc0 += vpu_dot16_f32(a + 16u, b0 + 16u);
	acc1 += vpu_dot16_f32(a + 16u, b1 + 16u);
	acc0 += vpu_dot16_f32(a + 32u, b0 + 32u);
	acc1 += vpu_dot16_f32(a + 32u, b1 + 32u);
	acc0 += vpu_dot16_f32(a + 48u, b0 + 48u);
	acc1 += vpu_dot16_f32(a + 48u, b1 + 48u);

	*out0 = acc0;
	*out1 = acc1;
#endif
}

#ifdef DNCNN_VPU_FUSED9TAP
static inline void vpu_dot64x9x2_fused_f32(
	const float *p0, const float *p1, const float *p2,
	const float *p3, const float *p4, const float *p5,
	const float *p6, const float *p7, const float *p8,
	const float *w0, const float *w1,
	float *out0, float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;
	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"addi t0, %[w0], 1024\n"
		"addi t1, %[w1], 1024\n"
		"flq2    f1,     0(%[p0])\n"
		"flq2    f2, -1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, -1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p0])\n"
		"flq2    f2,  -992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p0])\n"
		"flq2    f2,  -960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p0])\n"
		"flq2    f2,  -928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p0])\n"
		"flq2    f2,  -896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p0])\n"
		"flq2    f2,  -864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p0])\n"
		"flq2    f2,  -832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p0])\n"
		"flq2    f2,  -800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p1])\n"
		"flq2    f2,  -768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p1])\n"
		"flq2    f2,  -736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p1])\n"
		"flq2    f2,  -704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p1])\n"
		"flq2    f2,  -672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p1])\n"
		"flq2    f2,  -640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p1])\n"
		"flq2    f2,  -608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p1])\n"
		"flq2    f2,  -576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p1])\n"
		"flq2    f2,  -544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p2])\n"
		"flq2    f2,  -512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p2])\n"
		"flq2    f2,  -480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p2])\n"
		"flq2    f2,  -448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p2])\n"
		"flq2    f2,  -416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p2])\n"
		"flq2    f2,  -384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p2])\n"
		"flq2    f2,  -352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p2])\n"
		"flq2    f2,  -320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p2])\n"
		"flq2    f2,  -288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p3])\n"
		"flq2    f2,  -256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p3])\n"
		"flq2    f2,  -224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p3])\n"
		"flq2    f2,  -192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p3])\n"
		"flq2    f2,  -160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p3])\n"
		"flq2    f2,  -128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p3])\n"
		"flq2    f2,   -96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p3])\n"
		"flq2    f2,   -64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p3])\n"
		"flq2    f2,   -32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p4])\n"
		"flq2    f2,     0(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,     0(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p4])\n"
		"flq2    f2,    32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p4])\n"
		"flq2    f2,    64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p4])\n"
		"flq2    f2,    96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p4])\n"
		"flq2    f2,   128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p4])\n"
		"flq2    f2,   160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p4])\n"
		"flq2    f2,   192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p4])\n"
		"flq2    f2,   224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p5])\n"
		"flq2    f2,   256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p5])\n"
		"flq2    f2,   288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p5])\n"
		"flq2    f2,   320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p5])\n"
		"flq2    f2,   352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p5])\n"
		"flq2    f2,   384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p5])\n"
		"flq2    f2,   416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p5])\n"
		"flq2    f2,   448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p5])\n"
		"flq2    f2,   480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p6])\n"
		"flq2    f2,   512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p6])\n"
		"flq2    f2,   544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p6])\n"
		"flq2    f2,   576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p6])\n"
		"flq2    f2,   608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p6])\n"
		"flq2    f2,   640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p6])\n"
		"flq2    f2,   672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p6])\n"
		"flq2    f2,   704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p6])\n"
		"flq2    f2,   736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p7])\n"
		"flq2    f2,   768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p7])\n"
		"flq2    f2,   800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p7])\n"
		"flq2    f2,   832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p7])\n"
		"flq2    f2,   864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p7])\n"
		"flq2    f2,   896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p7])\n"
		"flq2    f2,   928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p7])\n"
		"flq2    f2,   960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p7])\n"
		"flq2    f2,   992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p8])\n"
		"flq2    f2,  1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p8])\n"
		"flq2    f2,  1056(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1056(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p8])\n"
		"flq2    f2,  1088(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1088(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p8])\n"
		"flq2    f2,  1120(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1120(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p8])\n"
		"flq2    f2,  1152(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1152(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p8])\n"
		"flq2    f2,  1184(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1184(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p8])\n"
		"flq2    f2,  1216(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1216(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p8])\n"
		"flq2    f2,  1248(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1248(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"fsq2    f0, 0(%[t0o])\n"
		"fsq2    f3, 0(%[t1o])\n"
		:
		: [p0] "r"(p0),
		  [p1] "r"(p1),
		  [p2] "r"(p2),
		  [p3] "r"(p3),
		  [p4] "r"(p4),
		  [p5] "r"(p5),
		  [p6] "r"(p6),
		  [p7] "r"(p7),
		  [p8] "r"(p8),
		  [w0] "r"(w0),
		  [w1] "r"(w1),
		  [t0o] "r"(tmp0),
		  [t1o] "r"(tmp1),
		  [zero] "r"(zero)
		: "memory", "t0", "t1", "f0", "f1", "f2", "f3", "f4");

	float s0 = 0.0f;
	float s1 = 0.0f;
	for (uint32_t i = 0; i < 8u; i++) {
		s0 += tmp0[i];
		s1 += tmp1[i];
	}
	*out0 = s0;
	*out1 = s1;
}
#endif

#ifdef DNCNN_VPU_FUSED9TAP_OC4
static inline void vpu_dot64x9x4_fused_f32(
	const float *p0, const float *p1, const float *p2,
	const float *p3, const float *p4, const float *p5,
	const float *p6, const float *p7, const float *p8,
	const float *w0, const float *w1, const float *w2, const float *w3,
	float *out0, float *out1, float *out2, float *out3)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	__attribute__((aligned(32))) float tmp2[8];
	__attribute__((aligned(32))) float tmp3[8];
	const uint64_t zero = 0;
	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"fbcx.ps f6, %[zero]\n"
		"fbcx.ps f9, %[zero]\n"
		"addi t0, %[w0], 1024\n"
		"addi t1, %[w1], 1024\n"
		"addi t2, %[w2], 1024\n"
		"addi t3, %[w3], 1024\n"
		"flq2    f1,     0(%[p0])\n"
		"flq2    f2, -1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, -1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5, -1024(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7, -1024(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p0])\n"
		"flq2    f2,  -992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -992(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -992(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p0])\n"
		"flq2    f2,  -960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -960(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -960(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p0])\n"
		"flq2    f2,  -928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -928(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -928(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p0])\n"
		"flq2    f2,  -896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -896(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -896(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p0])\n"
		"flq2    f2,  -864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -864(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -864(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p0])\n"
		"flq2    f2,  -832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -832(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -832(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p0])\n"
		"flq2    f2,  -800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -800(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -800(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p1])\n"
		"flq2    f2,  -768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -768(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -768(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p1])\n"
		"flq2    f2,  -736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -736(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -736(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p1])\n"
		"flq2    f2,  -704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -704(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -704(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p1])\n"
		"flq2    f2,  -672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -672(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -672(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p1])\n"
		"flq2    f2,  -640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -640(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -640(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p1])\n"
		"flq2    f2,  -608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -608(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -608(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p1])\n"
		"flq2    f2,  -576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -576(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -576(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p1])\n"
		"flq2    f2,  -544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -544(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -544(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p2])\n"
		"flq2    f2,  -512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -512(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -512(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p2])\n"
		"flq2    f2,  -480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -480(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -480(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p2])\n"
		"flq2    f2,  -448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -448(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -448(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p2])\n"
		"flq2    f2,  -416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -416(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -416(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p2])\n"
		"flq2    f2,  -384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -384(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -384(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p2])\n"
		"flq2    f2,  -352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -352(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -352(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p2])\n"
		"flq2    f2,  -320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -320(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -320(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p2])\n"
		"flq2    f2,  -288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -288(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -288(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p3])\n"
		"flq2    f2,  -256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -256(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -256(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p3])\n"
		"flq2    f2,  -224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -224(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -224(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p3])\n"
		"flq2    f2,  -192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -192(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -192(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p3])\n"
		"flq2    f2,  -160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -160(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -160(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p3])\n"
		"flq2    f2,  -128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  -128(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  -128(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p3])\n"
		"flq2    f2,   -96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   -96(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   -96(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p3])\n"
		"flq2    f2,   -64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   -64(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   -64(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p3])\n"
		"flq2    f2,   -32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   -32(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   -32(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p4])\n"
		"flq2    f2,     0(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,     0(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,     0(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,     0(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p4])\n"
		"flq2    f2,    32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,    32(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,    32(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p4])\n"
		"flq2    f2,    64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,    64(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,    64(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p4])\n"
		"flq2    f2,    96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,    96(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,    96(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p4])\n"
		"flq2    f2,   128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   128(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   128(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p4])\n"
		"flq2    f2,   160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   160(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   160(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p4])\n"
		"flq2    f2,   192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   192(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   192(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p4])\n"
		"flq2    f2,   224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   224(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   224(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p5])\n"
		"flq2    f2,   256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   256(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   256(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p5])\n"
		"flq2    f2,   288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   288(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   288(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p5])\n"
		"flq2    f2,   320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   320(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   320(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p5])\n"
		"flq2    f2,   352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   352(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   352(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p5])\n"
		"flq2    f2,   384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   384(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   384(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p5])\n"
		"flq2    f2,   416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   416(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   416(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p5])\n"
		"flq2    f2,   448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   448(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   448(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p5])\n"
		"flq2    f2,   480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   480(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   480(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p6])\n"
		"flq2    f2,   512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   512(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   512(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p6])\n"
		"flq2    f2,   544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   544(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   544(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p6])\n"
		"flq2    f2,   576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   576(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   576(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p6])\n"
		"flq2    f2,   608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   608(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   608(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p6])\n"
		"flq2    f2,   640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   640(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   640(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p6])\n"
		"flq2    f2,   672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   672(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   672(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p6])\n"
		"flq2    f2,   704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   704(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   704(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p6])\n"
		"flq2    f2,   736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   736(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   736(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p7])\n"
		"flq2    f2,   768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   768(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   768(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p7])\n"
		"flq2    f2,   800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   800(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   800(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p7])\n"
		"flq2    f2,   832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   832(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   832(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p7])\n"
		"flq2    f2,   864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   864(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   864(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p7])\n"
		"flq2    f2,   896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   896(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   896(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p7])\n"
		"flq2    f2,   928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   928(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   928(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p7])\n"
		"flq2    f2,   960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   960(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   960(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p7])\n"
		"flq2    f2,   992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,   992(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,   992(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,     0(%[p8])\n"
		"flq2    f2,  1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1024(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1024(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    32(%[p8])\n"
		"flq2    f2,  1056(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1056(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1056(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1056(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    64(%[p8])\n"
		"flq2    f2,  1088(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1088(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1088(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1088(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,    96(%[p8])\n"
		"flq2    f2,  1120(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1120(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1120(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1120(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   128(%[p8])\n"
		"flq2    f2,  1152(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1152(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1152(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1152(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   160(%[p8])\n"
		"flq2    f2,  1184(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1184(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1184(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1184(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   192(%[p8])\n"
		"flq2    f2,  1216(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1216(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1216(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1216(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"flq2    f1,   224(%[p8])\n"
		"flq2    f2,  1248(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1248(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f5,  1248(t2)\n"
		"fmadd.ps f6, f1, f5, f6\n"
		"flq2    f7,  1248(t3)\n"
		"fmadd.ps f9, f1, f7, f9\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		"fsq2    f6, 0(%[tmp2])\n"
		"fsq2    f9, 0(%[tmp3])\n"
		:
		: [p0] "r"(p0),
		  [p1] "r"(p1),
		  [p2] "r"(p2),
		  [p3] "r"(p3),
		  [p4] "r"(p4),
		  [p5] "r"(p5),
		  [p6] "r"(p6),
		  [p7] "r"(p7),
		  [p8] "r"(p8),
		  [w0] "r"(w0),
		  [w1] "r"(w1),
		  [w2] "r"(w2),
		  [w3] "r"(w3),
		  [tmp0] "r"(tmp0),
		  [tmp1] "r"(tmp1),
		  [tmp2] "r"(tmp2),
		  [tmp3] "r"(tmp3),
		  [zero] "r"(zero)
		: "memory", "t0", "t1", "t2", "t3",
		  "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f9");

	float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
	for (uint32_t i = 0; i < 8u; i++) {
		s0 += tmp0[i]; s1 += tmp1[i]; s2 += tmp2[i]; s3 += tmp3[i];
	}
	*out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
}
#endif

#ifdef DNCNN_VPU_SPATIAL2
static inline void vpu_dot64x9x2_spatial2_f32(
	const float *p_0_0, const float *p_0_1, const float *p_0_2, const float *p_0_3, const float *p_1_0, const float *p_1_1, const float *p_1_2, const float *p_1_3, const float *p_2_0, const float *p_2_1, const float *p_2_2, const float *p_2_3,
	const float *w0, const float *w1,
	float *out_p1_oc0, float *out_p1_oc1,
	float *out_p2_oc0, float *out_p2_oc1)
{
	__attribute__((aligned(32))) float t1_0[8];  /* pixel1 OC0 */
	__attribute__((aligned(32))) float t1_1[8];  /* pixel1 OC1 */
	__attribute__((aligned(32))) float t2_0[8];  /* pixel2 OC0 */
	__attribute__((aligned(32))) float t2_1[8];  /* pixel2 OC1 */
	const uint64_t zero = 0;
	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"  /* pixel1 OC0 */
		"fbcx.ps f3, %[zero]\n"  /* pixel1 OC1 */
		"fbcx.ps f6, %[zero]\n"  /* pixel2 OC0 */
		"fbcx.ps f9, %[zero]\n"  /* pixel2 OC1 */
		"addi t0, %[w0], 1024\n"
		"addi t1, %[w1], 1024\n"
		"flq2    f1,     0(%[p_0_0])\n"
		"flq2    f2, -1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, -1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p_0_1])\n"
		"flq2    f2,  -768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2, -1024(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4, -1024(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_0_2])\n"
		"flq2    f2,  -512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -768(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -768(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_0_3])\n"
		"flq2    f2,  -512(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -512(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_1_0])\n"
		"flq2    f2,  -256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p_1_1])\n"
		"flq2    f2,     0(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,     0(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -256(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -256(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_1_2])\n"
		"flq2    f2,   256(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   256(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,     0(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,     0(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_1_3])\n"
		"flq2    f2,   256(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   256(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_2_0])\n"
		"flq2    f2,   512(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   512(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,     0(%[p_2_1])\n"
		"flq2    f2,   768(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   768(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   512(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   512(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_2_2])\n"
		"flq2    f2,  1024(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1024(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   768(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   768(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,     0(%[p_2_3])\n"
		"flq2    f2,  1024(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1024(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_0_0])\n"
		"flq2    f2,  -992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p_0_1])\n"
		"flq2    f2,  -736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -992(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -992(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_0_2])\n"
		"flq2    f2,  -480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -736(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -736(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_0_3])\n"
		"flq2    f2,  -480(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -480(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_1_0])\n"
		"flq2    f2,  -224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p_1_1])\n"
		"flq2    f2,    32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -224(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -224(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_1_2])\n"
		"flq2    f2,   288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,    32(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,    32(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_1_3])\n"
		"flq2    f2,   288(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   288(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_2_0])\n"
		"flq2    f2,   544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    32(%[p_2_1])\n"
		"flq2    f2,   800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   544(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   544(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_2_2])\n"
		"flq2    f2,  1056(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1056(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   800(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   800(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    32(%[p_2_3])\n"
		"flq2    f2,  1056(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1056(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_0_0])\n"
		"flq2    f2,  -960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p_0_1])\n"
		"flq2    f2,  -704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -960(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -960(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_0_2])\n"
		"flq2    f2,  -448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -704(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -704(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_0_3])\n"
		"flq2    f2,  -448(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -448(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_1_0])\n"
		"flq2    f2,  -192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p_1_1])\n"
		"flq2    f2,    64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -192(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -192(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_1_2])\n"
		"flq2    f2,   320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,    64(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,    64(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_1_3])\n"
		"flq2    f2,   320(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   320(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_2_0])\n"
		"flq2    f2,   576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    64(%[p_2_1])\n"
		"flq2    f2,   832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   576(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   576(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_2_2])\n"
		"flq2    f2,  1088(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1088(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   832(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   832(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    64(%[p_2_3])\n"
		"flq2    f2,  1088(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1088(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_0_0])\n"
		"flq2    f2,  -928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p_0_1])\n"
		"flq2    f2,  -672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -928(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -928(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_0_2])\n"
		"flq2    f2,  -416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -672(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -672(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_0_3])\n"
		"flq2    f2,  -416(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -416(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_1_0])\n"
		"flq2    f2,  -160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p_1_1])\n"
		"flq2    f2,    96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,    96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -160(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -160(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_1_2])\n"
		"flq2    f2,   352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,    96(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,    96(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_1_3])\n"
		"flq2    f2,   352(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   352(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_2_0])\n"
		"flq2    f2,   608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,    96(%[p_2_1])\n"
		"flq2    f2,   864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   608(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   608(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_2_2])\n"
		"flq2    f2,  1120(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1120(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   864(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   864(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,    96(%[p_2_3])\n"
		"flq2    f2,  1120(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1120(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_0_0])\n"
		"flq2    f2,  -896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p_0_1])\n"
		"flq2    f2,  -640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -896(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -896(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_0_2])\n"
		"flq2    f2,  -384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -640(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -640(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_0_3])\n"
		"flq2    f2,  -384(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -384(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_1_0])\n"
		"flq2    f2,  -128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p_1_1])\n"
		"flq2    f2,   128(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   128(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -128(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -128(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_1_2])\n"
		"flq2    f2,   384(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   384(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   128(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   128(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_1_3])\n"
		"flq2    f2,   384(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   384(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_2_0])\n"
		"flq2    f2,   640(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   640(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   128(%[p_2_1])\n"
		"flq2    f2,   896(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   896(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   640(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   640(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_2_2])\n"
		"flq2    f2,  1152(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1152(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   896(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   896(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   128(%[p_2_3])\n"
		"flq2    f2,  1152(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1152(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_0_0])\n"
		"flq2    f2,  -864(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -864(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p_0_1])\n"
		"flq2    f2,  -608(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -608(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -864(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -864(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_0_2])\n"
		"flq2    f2,  -352(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -352(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -608(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -608(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_0_3])\n"
		"flq2    f2,  -352(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -352(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_1_0])\n"
		"flq2    f2,   -96(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -96(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p_1_1])\n"
		"flq2    f2,   160(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   160(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   -96(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   -96(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_1_2])\n"
		"flq2    f2,   416(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   416(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   160(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   160(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_1_3])\n"
		"flq2    f2,   416(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   416(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_2_0])\n"
		"flq2    f2,   672(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   672(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   160(%[p_2_1])\n"
		"flq2    f2,   928(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   928(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   672(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   672(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_2_2])\n"
		"flq2    f2,  1184(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1184(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   928(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   928(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   160(%[p_2_3])\n"
		"flq2    f2,  1184(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1184(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_0_0])\n"
		"flq2    f2,  -832(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -832(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p_0_1])\n"
		"flq2    f2,  -576(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -576(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -832(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -832(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_0_2])\n"
		"flq2    f2,  -320(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -320(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -576(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -576(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_0_3])\n"
		"flq2    f2,  -320(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -320(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_1_0])\n"
		"flq2    f2,   -64(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -64(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p_1_1])\n"
		"flq2    f2,   192(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   192(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   -64(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   -64(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_1_2])\n"
		"flq2    f2,   448(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   448(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   192(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   192(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_1_3])\n"
		"flq2    f2,   448(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   448(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_2_0])\n"
		"flq2    f2,   704(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   704(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   192(%[p_2_1])\n"
		"flq2    f2,   960(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   960(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   704(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   704(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_2_2])\n"
		"flq2    f2,  1216(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1216(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   960(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   960(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   192(%[p_2_3])\n"
		"flq2    f2,  1216(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1216(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_0_0])\n"
		"flq2    f2,  -800(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -800(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p_0_1])\n"
		"flq2    f2,  -544(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -544(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -800(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -800(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_0_2])\n"
		"flq2    f2,  -288(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  -288(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,  -544(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -544(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_0_3])\n"
		"flq2    f2,  -288(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  -288(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_1_0])\n"
		"flq2    f2,   -32(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   -32(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p_1_1])\n"
		"flq2    f2,   224(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   224(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   -32(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   -32(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_1_2])\n"
		"flq2    f2,   480(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   480(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   224(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   224(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_1_3])\n"
		"flq2    f2,   480(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   480(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_2_0])\n"
		"flq2    f2,   736(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   736(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1,   224(%[p_2_1])\n"
		"flq2    f2,   992(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,   992(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   736(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   736(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_2_2])\n"
		"flq2    f2,  1248(t0)\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4,  1248(t1)\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f2,   992(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,   992(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"flq2    f1,   224(%[p_2_3])\n"
		"flq2    f2,  1248(t0)\n"
		"fmadd.ps f6, f1, f2, f6\n"
		"flq2    f4,  1248(t1)\n"
		"fmadd.ps f9, f1, f4, f9\n"
		"fsq2    f0, 0(%[t1_0])\n"
		"fsq2    f3, 0(%[t1_1])\n"
		"fsq2    f6, 0(%[t2_0])\n"
		"fsq2    f9, 0(%[t2_1])\n"
		:
		: [p_0_0] "r"(p_0_0),
		  [p_0_1] "r"(p_0_1),
		  [p_0_2] "r"(p_0_2),
		  [p_0_3] "r"(p_0_3),
		  [p_1_0] "r"(p_1_0),
		  [p_1_1] "r"(p_1_1),
		  [p_1_2] "r"(p_1_2),
		  [p_1_3] "r"(p_1_3),
		  [p_2_0] "r"(p_2_0),
		  [p_2_1] "r"(p_2_1),
		  [p_2_2] "r"(p_2_2),
		  [p_2_3] "r"(p_2_3),
		  [w0] "r"(w0),
		  [w1] "r"(w1),
		  [t1_0] "r"(t1_0),
		  [t1_1] "r"(t1_1),
		  [t2_0] "r"(t2_0),
		  [t2_1] "r"(t2_1),
		  [zero] "r"(zero)
		: "memory", "t0", "t1",
		  "f0", "f1", "f2", "f3", "f4", "f6", "f9");

	float p1_oc0 = 0.0f, p1_oc1 = 0.0f, p2_oc0 = 0.0f, p2_oc1 = 0.0f;
	for (uint32_t i = 0; i < 8u; i++) {
		p1_oc0 += t1_0[i]; p1_oc1 += t1_1[i];
		p2_oc0 += t2_0[i]; p2_oc1 += t2_1[i];
	}
	*out_p1_oc0 = p1_oc0; *out_p1_oc1 = p1_oc1;
	*out_p2_oc0 = p2_oc0; *out_p2_oc1 = p2_oc1;
}
#endif

static float hidden_acc_scalar(const float *input, const float *w_oc,
			       uint32_t y, uint32_t x)
{
	float acc = 0.0f;

	for (int ky = -1; ky <= 1; ky++) {
		const int yy = (int)y + ky;

		if (yy < 0 || yy >= (int)IMG_H) {
			continue;
		}

		for (int kx = -1; kx <= 1; kx++) {
			const int xx = (int)x + kx;

			if (xx < 0 || xx >= (int)IMG_W) {
				continue;
			}

			const uint32_t k = (uint32_t)(ky + 1) * 3u +
					   (uint32_t)(kx + 1);
			const float *pix = input + ((uint32_t)yy * IMG_W +
						    (uint32_t)xx) * CH;
			const float *w = w_oc + k * CH;

			for (uint32_t ic = 0; ic < CH; ic++) {
				acc += pix[ic] * w[ic];
			}
		}
	}

	return acc;
}

static float hidden_acc_vpu(const float *input, const float *w_oc,
			    uint32_t y, uint32_t x)
{
	const float *const p = input + (y * IMG_W + x) * CH;
	const int row = IMG_W * CH;
	float acc = 0.0f;

	acc += dot64_vpu(p - row - (int)CH, w_oc + 0u * CH);
	acc += dot64_vpu(p - row, w_oc + 1u * CH);
	acc += dot64_vpu(p - row + (int)CH, w_oc + 2u * CH);
	acc += dot64_vpu(p - (int)CH, w_oc + 3u * CH);
	acc += dot64_vpu(p, w_oc + 4u * CH);
	acc += dot64_vpu(p + (int)CH, w_oc + 5u * CH);
	acc += dot64_vpu(p + row - (int)CH, w_oc + 6u * CH);
	acc += dot64_vpu(p + row, w_oc + 7u * CH);
	acc += dot64_vpu(p + row + (int)CH, w_oc + 8u * CH);

	return acc;
}

static void conv_first_fp(const float *input, const float *w0,
			  const float *b0, float *output,
			  uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const float *const w = w0 + oc * K * K;

		for (uint32_t y = row0; y < row1; y++) {
			for (uint32_t x = 0; x < IMG_W; x++) {
				float acc = b0[oc];

				for (int ky = -1; ky <= 1; ky++) {
					const int yy = (int)y + ky;

					if (yy < 0 || yy >= (int)IMG_H) {
						continue;
					}

					for (int kx = -1; kx <= 1; kx++) {
						const int xx = (int)x + kx;

						if (xx < 0 || xx >= (int)IMG_W) {
							continue;
						}

						const uint32_t k =
							(uint32_t)(ky + 1) * 3u +
							(uint32_t)(kx + 1);
						acc += input[(uint32_t)yy * IMG_W +
							     (uint32_t)xx] * w[k];
					}
				}

				output[(y * IMG_W + x) * CH + oc] = relu_f32(acc);
			}
		}
	}
}

static void conv_hidden_fp(const float *input, const float *weights,
			   const float *biases, float *output,
			   uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_SPATIAL2
	for (uint32_t oc = 0; oc < CH; oc += 2u) {
		const float *const w0 = weights + oc * K * K * CH;
		const float *const w1 = w0 + K * K * CH;
		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);
			uint32_t x = 0;
			/* Process pairs of interior pixels (x, x+1) where both are interior, i.e. x in [1, IMG_W-2). */
			if (interior_y) {
				/* boundary x=0 falls through to scalar */
				if (x == 0u) {
					float acc0 = biases[oc] + hidden_acc_scalar(input, w0, y, x);
					float acc1 = biases[oc + 1u] + hidden_acc_scalar(input, w1, y, x);
					output[(y * IMG_W + x) * CH + oc] = relu_f32(acc0);
					output[(y * IMG_W + x) * CH + oc + 1u] = relu_f32(acc1);
					x = 1u;
				}
				while (x + 1u < (IMG_W - 1u)) {
					const float *const p =
						input + (y * IMG_W + x) * CH;
					const int row = IMG_W * CH;
					float p1_oc0, p1_oc1, p2_oc0, p2_oc1;
					vpu_dot64x9x2_spatial2_f32(
						p - row - (int)CH, p - row, p - row + (int)CH, p - row + 2 * (int)CH,
						p - (int)CH,        p,       p + (int)CH,       p + 2 * (int)CH,
						p + row - (int)CH, p + row, p + row + (int)CH, p + row + 2 * (int)CH,
						w0, w1,
						&p1_oc0, &p1_oc1, &p2_oc0, &p2_oc1);
					output[(y * IMG_W + x) * CH + oc + 0u]      = relu_f32(biases[oc]      + p1_oc0);
					output[(y * IMG_W + x) * CH + oc + 1u]      = relu_f32(biases[oc + 1u] + p1_oc1);
					output[(y * IMG_W + (x + 1u)) * CH + oc + 0u] = relu_f32(biases[oc]      + p2_oc0);
					output[(y * IMG_W + (x + 1u)) * CH + oc + 1u] = relu_f32(biases[oc + 1u] + p2_oc1);
					x += 2u;
				}
			}
			/* boundary tail (and entire row if !interior_y) */
			for (; x < IMG_W; x++) {
				float acc0 = biases[oc] + hidden_acc_scalar(input, w0, y, x);
				float acc1 = biases[oc + 1u] + hidden_acc_scalar(input, w1, y, x);
				output[(y * IMG_W + x) * CH + oc] = relu_f32(acc0);
				output[(y * IMG_W + x) * CH + oc + 1u] = relu_f32(acc1);
			}
		}
	}
	return;
#endif
#ifdef DNCNN_VPU_FUSED9TAP_OC4
	for (uint32_t oc = 0; oc < CH; oc += 4u) {
		const float *const w0 = weights + oc * K * K * CH;
		const float *const w1 = w0 + K * K * CH;
		const float *const w2 = w1 + K * K * CH;
		const float *const w3 = w2 + K * K * CH;
		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);
			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior = interior_y &&
					x > 0u && x < (IMG_W - 1u);
				float acc0 = biases[oc];
				float acc1 = biases[oc + 1u];
				float acc2 = biases[oc + 2u];
				float acc3 = biases[oc + 3u];
				if (interior) {
					const float *const p =
						input + (y * IMG_W + x) * CH;
					const int row = IMG_W * CH;
					float d0, d1, d2, d3;
					vpu_dot64x9x4_fused_f32(
						p - row - (int)CH, p - row, p - row + (int)CH,
						p - (int)CH,        p,       p + (int)CH,
						p + row - (int)CH, p + row, p + row + (int)CH,
						w0, w1, w2, w3,
						&d0, &d1, &d2, &d3);
					acc0 += d0; acc1 += d1; acc2 += d2; acc3 += d3;
				} else {
					acc0 += hidden_acc_scalar(input, w0, y, x);
					acc1 += hidden_acc_scalar(input, w1, y, x);
					acc2 += hidden_acc_scalar(input, w2, y, x);
					acc3 += hidden_acc_scalar(input, w3, y, x);
				}
				output[(y * IMG_W + x) * CH + oc + 0u] = relu_f32(acc0);
				output[(y * IMG_W + x) * CH + oc + 1u] = relu_f32(acc1);
				output[(y * IMG_W + x) * CH + oc + 2u] = relu_f32(acc2);
				output[(y * IMG_W + x) * CH + oc + 3u] = relu_f32(acc3);
			}
		}
	}
	return;
#endif
	for (uint32_t oc = 0; oc < CH; oc += 2u) {
		const float *const w0 = weights + oc * K * K * CH;
		const float *const w1 = w0 + K * K * CH;

		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);

			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior = interior_y &&
					x > 0u && x < (IMG_W - 1u);
				float acc0 = biases[oc];
				float acc1 = biases[oc + 1u];

				if (interior) {
					const float *const p =
						input + (y * IMG_W + x) * CH;
					const int row = IMG_W * CH;
#ifdef DNCNN_VPU_FUSED9TAP
					float d0;
					float d1;
					vpu_dot64x9x2_fused_f32(
						p - row - (int)CH, p - row, p - row + (int)CH,
						p - (int)CH,        p,       p + (int)CH,
						p + row - (int)CH, p + row, p + row + (int)CH,
						w0, w1, &d0, &d1);
					acc0 += d0;
					acc1 += d1;
#else
					for (uint32_t k = 0; k < K * K; k++) {
						float d0;
						float d1;

						dot64x2_vpu(p + ((int)(k / 3u) - 1) *
							    row +
							    ((int)(k % 3u) - 1) *
							    (int)CH,
							    w0 + k * CH,
							    w1 + k * CH,
							    &d0, &d1);
						acc0 += d0;
						acc1 += d1;
					}
#endif
				} else {
					acc0 += hidden_acc_scalar(input, w0, y, x);
					acc1 += hidden_acc_scalar(input, w1, y, x);
				}

				output[(y * IMG_W + x) * CH + oc] = relu_f32(acc0);
				output[(y * IMG_W + x) * CH + oc + 1u] =
					relu_f32(acc1);
			}
		}
	}
}

static void conv_final_fp(const float *input, const float *source_image,
			  const float *weights, const float *bias,
			  float *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++) {
		const int interior_y = y > 0u && y < (IMG_H - 1u);

		for (uint32_t x = 0; x < IMG_W; x++) {
			const int interior = interior_y &&
				x > 0u && x < (IMG_W - 1u);
			float residual = bias[0];

			residual += interior ?
				hidden_acc_vpu(input, weights, y, x) :
				hidden_acc_scalar(input, weights, y, x);

			output[y * IMG_W + x] =
				source_image[y * IMG_W + x] - residual;
		}
	}
}

static uint32_t stripe_hash(const float *output, uint32_t row0, uint32_t row1)
{
	uint32_t h = 2166136261u;

	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			h = mix_hash(h, float_bits(output[y * IMG_W + x]));
		}
	}

	return h;
}

static void evict_activation_write_float(float *buffer,
					 uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_BOUNDARY_ONLY_EVICT
	/* Only evict the first and last rows (the halos that other harts read).
	 * Saves DRAM write bandwidth on interior rows. */
	if (row0 > 0u) {
		evict(buffer + row0 * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
	if (row1 < IMG_H && row1 > row0 + 1u) {
		evict(buffer + (row1 - 1u) * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
#else
	evict(buffer + row0 * IMG_W * CH,
	      (row1 - row0) * IMG_W * CH * sizeof(float));
#endif
}

static void evict_activation_read_float(const float *buffer,
					uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_SKIP_READ_EVICT
	(void)buffer; (void)row0; (void)row1;
#elif defined(DNCNN_VPU_BOUNDARY_ONLY_EVICT)
	/* Only invalidate the halo rows since own range stayed in L1D from the
	 * previous layer's read. */
	if (row0 > 0u) {
		evict(buffer + (row0 - 1u) * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
	if (row1 < IMG_H) {
		evict(buffer + row1 * IMG_W * CH,
		      IMG_W * CH * sizeof(float));
	}
#else
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;

	evict(buffer + read_row0 * IMG_W * CH,
	      (read_row1 - read_row0) * IMG_W * CH * sizeof(float));
#endif
}

static void prefetch_activation_read_window(const float *buffer,
					    uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_VPU_PREFETCH_READ_WINDOW
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;
	uint64_t addr = (uint64_t)(buffer + read_row0 * IMG_W * CH);
	uint64_t lines = ((read_row1 - read_row0) * IMG_W * CH *
			  sizeof(float) + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
#else
	(void)buffer;
	(void)row0;
	(void)row1;
#endif
}

static void prefetch_weight_block(const float *weights, uint32_t floats)
{
	uint64_t addr = (uint64_t)weights;
	uint64_t lines = (floats * sizeof(float) + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
}

int main(uintptr_t arg_area)
{
#ifdef DNCNN_SMT_HETERO_T0_ONLY
	/* Hetero-SMT campaign mode: only T0 (even harts) of each minion participate
	 * in compute. Odd harts exit immediately. The remaining 8 even harts are
	 * remapped to logical indices 0..7 so the row-stripe partition divides
	 * IMG_H into 8 instead of 16. Build with -DACTIVE_HARTS=8 to keep barrier
	 * counts consistent. */
	const uint32_t raw_hart = get_hart_id();
	if (raw_hart >= 16u || (raw_hart & 1u) != 0u) {
		return 0;
	}
	const uint32_t hart_id = raw_hart >> 1;
#else
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}
#endif

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
#ifdef DNCNN_STATIC_BLOBS
	const float *const input =
		(const float *)DNCNN_INPUT_BLOB;
	const float *const weights =
		(const float *)DNCNN_WEIGHTS_BLOB;
	const float *const reference =
		(const float *)DNCNN_REF_BLOB;
	float *const final_output = (float *)(base + OUTPUT_OFFSET);
	float *const act0 = static_act0;
	float *const act1 = static_act1;
#else
	const float *const input = (const float *)(base + INPUT_OFFSET);
	const float *const weights = (const float *)(base + WEIGHTS_OFFSET);
	float *const final_output = (float *)(base + OUTPUT_OFFSET);
	float *const act0 = (float *)(base + ACT0_OFFSET);
	float *const act1 = (float *)(base + ACT1_OFFSET);
	const float *const reference = (const float *)0;
#endif

	const float *const w0 = weights + W0_OFFSET_FLOATS;
	const float *const b0 = weights + B0_OFFSET_FLOATS;
	const float *const hidden_w = weights + WH_OFFSET_FLOATS;
	const float *const hidden_b = weights + BH_OFFSET_FLOATS;
	const float *const final_w = weights + WF_OFFSET_FLOATS;
	const float *const final_b = weights + BF_OFFSET_FLOATS;

	volatile struct dncnn_slot *const slots =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
	volatile struct dncnn_summary *const summary =
		(volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

	evict((void *)input, INPUT_BYTES);
	evict((void *)weights, WEIGHTS_FLOATS * sizeof(float));
	WAIT_CACHEOPS;
	prefetch_weight_block(hidden_w, WH_PACK_FLOATS);
	FENCE;
	bench_barrier();

	for (uint32_t pass = 0; pass < DNCNN_PASSES; pass++) {
		conv_first_fp(input, w0, b0, act0, row0, row1);
		FENCE;
		evict_activation_write_float(act0, row0, row1);
		WAIT_CACHEOPS;
		bench_barrier();

		for (uint32_t layer = 0; layer < HIDDEN_LAYERS; layer++) {
			const float *const src = (layer & 1u) ? act1 : act0;
			float *const dst = (layer & 1u) ? act0 : act1;

			evict_activation_read_float(src, row0, row1);
			WAIT_CACHEOPS;
			prefetch_activation_read_window(src, row0, row1);
			conv_hidden_fp(src,
				       hidden_w + layer * CH * K * K * CH,
				       hidden_b + layer * CH,
				       dst, row0, row1);
			FENCE;
			evict_activation_write_float(dst, row0, row1);
			WAIT_CACHEOPS;
			bench_barrier();
		}

		const float *const last_hidden =
			(HIDDEN_LAYERS & 1u) ? act1 : act0;

		evict_activation_read_float(last_hidden, row0, row1);
		WAIT_CACHEOPS;
		conv_final_fp(last_hidden, input, final_w, final_b,
			      final_output, row0, row1);
		FENCE;
		evict(final_output + row0 * IMG_W,
		      (row1 - row0) * IMG_W * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();
	}

	const uint32_t checksum = stripe_hash(final_output, row0, row1);
	volatile struct dncnn_slot *const slot =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = DNCNN_MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->row0 = row0;
	slot->row1 = row1;
	slot->active_harts = ACTIVE_HARTS;
	slot->checksum = checksum;
	slot->done = 1u;

	FENCE;
	evict((void *)slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t slot_checksum_sum = 0;
		uint32_t output_hash = 2166136261u;
#ifdef DNCNN_STATIC_BLOBS
		uint32_t ref_hash = 2166136261u;
		float max_abs = 0.0f;
		float sum_abs = 0.0f;
#endif

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == DNCNN_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}

		for (uint32_t i = 0; i < IMG_PIXELS; i++) {
			output_hash = mix_hash(output_hash,
					       float_bits(final_output[i]));
#ifdef DNCNN_STATIC_BLOBS
			ref_hash = mix_hash(ref_hash, float_bits(reference[i]));
			float diff = final_output[i] - reference[i];

			if (diff < 0.0f) {
				diff = -diff;
			}
			if (diff > max_abs) {
				max_abs = diff;
			}
			sum_abs += diff;
#endif
		}

		const uint64_t macs_per_pass =
			(uint64_t)IMG_PIXELS *
			((uint64_t)CH * K * K +
			 (uint64_t)HIDDEN_LAYERS * CH * CH * K * K +
			 (uint64_t)CH * K * K);
		const uint64_t ops = macs_per_pass * DNCNN_PASSES * 2u;

		summary->magic = DNCNN_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->passes = DNCNN_PASSES;
		summary->width = IMG_W;
		summary->height = IMG_H;
		summary->channels = CH;
		summary->layers = HIDDEN_LAYERS + 2u;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
#ifdef DNCNN_STATIC_BLOBS
		summary->slot_checksum_sum = ref_hash;
		summary->reserved[0] = (uint32_t)(max_abs * 1000000000.0f);
		summary->reserved[1] =
			(uint32_t)((sum_abs / (float)IMG_PIXELS) *
				   1000000000.0f);
		summary->reserved[2] = slot_checksum_sum;
#else
		summary->slot_checksum_sum = slot_checksum_sum;
#endif
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);

		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
