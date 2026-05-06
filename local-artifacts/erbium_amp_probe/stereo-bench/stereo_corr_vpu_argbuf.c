/*
 * Foundation-Stereo-shaped Erbium VPU FP32 benchmark.
 *
 * This is a cost-volume/correlation kernel benchmark, not a full stereo model.
 * It computes left/right feature dot products for a disparity sweep.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 1u
#endif

#ifndef STEREO_PASSES
#define STEREO_PASSES 1u
#endif

#define STEREO_MAGIC      0x53773001u
#define IMG_W             80u
#define IMG_H             48u
#define CH                32u
#define DISP              32u
#define FEAT_FLOATS       (IMG_W * IMG_H * CH)
#define COST_FLOATS       (IMG_W * IMG_H * DISP)

#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x0000u
#define SUMMARY_OFFSET    0x1000u
#define LEFT_OFFSET       0x4000u
#define RIGHT_OFFSET      0x80000u
#define COST_OFFSET       0x100000u

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct stereo_slot {
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

struct stereo_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t passes;
	uint32_t width;
	uint32_t height;
	uint32_t channels;
	uint32_t disparities;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t slot_checksum_sum;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t reserved[3];
};

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	return ptr;
}

static inline uint32_t active_mask_t0(void)
{
	uint32_t mask = 0;

	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
}

static inline uint32_t active_mask_t1(void)
{
	uint32_t mask = 0;

	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1);
	}

	return mask;
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

static inline uint8_t clamp_u8_from_f32(float v)
{
	if (v < 0.0f) {
		return 0u;
	}
	if (v > 255.0f) {
		return 255u;
	}
	return (uint8_t)(v + 0.5f);
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

static inline void vpu_dot16x2_f32(const float *a, const float *w0,
				   const float *w1, float *out0,
				   float *out1)
{
	__attribute__((aligned(32))) float tmp0[8];
	__attribute__((aligned(32))) float tmp1[8];
	const uint64_t zero = 0;

	__asm__ __volatile__(
		"fbcx.ps f0, %[zero]\n"
		"fbcx.ps f3, %[zero]\n"
		"flq2    f1, 0(%[a0])\n"
		"flq2    f2, 0(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 0(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"flq2    f1, 32(%[a0])\n"
		"flq2    f2, 32(%[w0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2    f4, 32(%[w1])\n"
		"fmadd.ps f3, f1, f4, f3\n"
		"fsq2    f0, 0(%[tmp0])\n"
		"fsq2    f3, 0(%[tmp1])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [w0] "r"(w0),
		  [w1] "r"(w1), [tmp0] "r"(tmp0), [tmp1] "r"(tmp1)
		: "memory", "f0", "f1", "f2", "f3", "f4");

	float sum0 = 0.0f;
	float sum1 = 0.0f;

	for (uint32_t i = 0; i < 8u; i++) {
		sum0 += tmp0[i];
		sum1 += tmp1[i];
	}

	*out0 = sum0;
	*out1 = sum1;
}

static float dot32_f32(const float *a, const float *b)
{
	return vpu_dot16_f32(a, b) + vpu_dot16_f32(a + 16u, b + 16u);
}

static void dot32x2_f32(const float *a, const float *b0, const float *b1,
			float *out0, float *out1)
{
	float p0;
	float p1;
	float q0;
	float q1;

	vpu_dot16x2_f32(a, b0, b1, &p0, &p1);
	vpu_dot16x2_f32(a + 16u, b0 + 16u, b1 + 16u, &q0, &q1);
	*out0 = p0 + q0;
	*out1 = p1 + q1;
}

static void prefetch_block(const void *ptr, uint32_t bytes)
{
	uint64_t addr = (uint64_t)ptr;
	uint64_t lines = (bytes + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	if (lines > 0u) {
		prefetch_va(0, addr, lines, 64u, 0);
	}
}

static void init_features(float *left, float *right)
{
	for (uint32_t i = 0; i < FEAT_FLOATS; i++) {
		left[i] = (float)((i * 13u + 17u) & 0xffu) * (1.0f / 255.0f);
		right[i] = (float)((i * 29u + 3u) & 0xffu) * (1.0f / 255.0f);
	}
}

static void maybe_evict_read(const float *ptr, uint32_t bytes)
{
#ifndef STEREO_SKIP_READ_EVICT
	evict(ptr, bytes);
	WAIT_CACHEOPS;
#else
	(void)ptr;
	(void)bytes;
#endif
}

static void stereo_corr(const float *left, const float *right, float *cost,
			uint32_t row0, uint32_t row1)
{
	maybe_evict_read(left + row0 * IMG_W * CH,
			 (row1 - row0) * IMG_W * CH * sizeof(float));
#ifdef STEREO_PREFETCH_RIGHT
	prefetch_block(right, FEAT_FLOATS * sizeof(float));
#endif
#ifdef STEREO_PREFETCH_LEFT_ROWS
	prefetch_block(left + row0 * IMG_W * CH,
		       (row1 - row0) * IMG_W * CH * sizeof(float));
#endif

	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			const float *const l = left + (y * IMG_W + x) * CH;

#ifdef STEREO_DISP2
			for (uint32_t d = 0; d < DISP; d += 2u) {
				const uint32_t x0 = x >= d ? x - d : 0u;
				const uint32_t d1 = d + 1u;
				const uint32_t x1 = x >= d1 ? x - d1 : 0u;
				const float *const r0 =
					right + (y * IMG_W + x0) * CH;
				const float *const r1 =
					right + (y * IMG_W + x1) * CH;
				float acc0;
				float acc1;

				dot32x2_f32(l, r0, r1, &acc0, &acc1);
				cost[(y * IMG_W + x) * DISP + d] =
					acc0 * (1.0f / (float)CH);
				cost[(y * IMG_W + x) * DISP + d + 1u] =
					acc1 * (1.0f / (float)CH);
			}
#else
			for (uint32_t d = 0; d < DISP; d++) {
				const uint32_t xx = x >= d ? x - d : 0u;
				const float *const r =
					right + (y * IMG_W + xx) * CH;
				const float acc = dot32_f32(l, r);

				cost[(y * IMG_W + x) * DISP + d] =
					acc * (1.0f / (float)CH);
			}
#endif
		}
	}
}

static uint32_t stripe_checksum(const float *cost, uint32_t row0,
				uint32_t row1)
{
	uint32_t sum = 0;

	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			for (uint32_t d = 0; d < DISP; d++) {
				sum += clamp_u8_from_f32(
					64.0f + cost[(y * IMG_W + x) * DISP + d] *
					128.0f);
			}
		}
	}

	return sum;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const left = (float *)(base + LEFT_OFFSET);
	float *const right = (float *)(base + RIGHT_OFFSET);
	float *const cost = (float *)(base + COST_OFFSET);
	volatile struct stereo_slot *const slots =
		(volatile struct stereo_slot *)(base + SLOTS_OFFSET);
	volatile struct stereo_summary *const summary =
		(volatile struct stereo_summary *)(base + SUMMARY_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

	if (hart_id == 0u) {
		init_features(left, right);
		FENCE;
		evict(left, FEAT_FLOATS * sizeof(float));
		evict(right, FEAT_FLOATS * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	for (uint32_t pass = 0; pass < STEREO_PASSES; pass++) {
		stereo_corr(left, right, cost, row0, row1);
		FENCE;
#ifdef STEREO_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == STEREO_PASSES)
#endif
		{
			evict(cost + row0 * IMG_W * DISP,
			      (row1 - row0) * IMG_W * DISP * sizeof(float));
			WAIT_CACHEOPS;
		}
		bench_barrier();
	}

	const uint32_t checksum = stripe_checksum(cost, row0, row1);
	volatile struct stereo_slot *const slot =
		(volatile struct stereo_slot *)(base + SLOTS_OFFSET +
						hart_id * SLOT_BYTES);
	slot->magic = STEREO_MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->row0 = row0;
	slot->row1 = row1;
	slot->active_harts = ACTIVE_HARTS;
	slot->checksum = checksum;
	slot->done = 1u;

	FENCE;
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t slot_checksum_sum = 0;
		uint32_t output_sum = 0;

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == STEREO_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}

		output_sum = stripe_checksum(cost, 0u, IMG_H);

		const uint64_t macs_per_pass =
			(uint64_t)IMG_W * IMG_H * DISP * CH;
		const uint64_t ops = macs_per_pass * STEREO_PASSES * 2u;

		summary->magic = STEREO_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->passes = STEREO_PASSES;
		summary->width = IMG_W;
		summary->height = IMG_H;
		summary->channels = CH;
		summary->disparities = DISP;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_sum = output_sum;
		summary->slot_checksum_sum = slot_checksum_sum;
		summary->ops_lo = (uint32_t)ops;
		summary->ops_hi = (uint32_t)(ops >> 32);

		FENCE;
		evict(summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
