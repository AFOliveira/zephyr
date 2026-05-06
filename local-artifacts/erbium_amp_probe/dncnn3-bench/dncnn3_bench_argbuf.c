/*
 * DnCNN-shaped ET-SoC1/Erbium scaling benchmark.
 *
 * This is a hardware bring-up benchmark, not a trained Luxonis DnCNN3
 * accuracy port.  The host loads deterministic int8 input/weights into a
 * device buffer and passes that buffer pointer as the kernel argument.  Active
 * harts split image rows, run a small DnCNN-like stack of 3x3 convolutions,
 * synchronize between layers, and publish slots plus a summary for host
 * validation.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 1u
#endif

#ifndef DNCNN_PASSES
#define DNCNN_PASSES 8u
#endif

#define DNCNN_MAGIC       0xD3C11003u
#ifndef IMG_W
#define IMG_W             64u
#endif
#ifndef IMG_H
#define IMG_H             64u
#endif
#ifndef DNCNN_CH
#define DNCNN_CH          16u
#endif
#define CH                DNCNN_CH
#ifndef DNCNN_LAYERS
#define DNCNN_LAYERS      5u
#endif
#define LAYERS            DNCNN_LAYERS
#define K                 3u
#define IMG_BYTES         (IMG_W * IMG_H)
#define ACT_BYTES         (IMG_W * IMG_H * CH)
#define W0_BYTES          (CH * K * K)
#define WH_BYTES          ((LAYERS - 2u) * CH * CH * K * K)
#define WF_BYTES          (CH * K * K)
#define WEIGHTS_BYTES     (W0_BYTES + WH_BYTES + WF_BYTES)
#define FIRST_SHIFT       7u
#define HIDDEN_SHIFT      8u
#define FINAL_SHIFT       4u

#ifndef SLOT_BYTES
#define SLOT_BYTES        64u
#endif
#ifndef SLOTS_OFFSET
#define SLOTS_OFFSET      0x0000u
#endif
#ifndef SUMMARY_OFFSET
#define SUMMARY_OFFSET    0x1000u
#endif
#ifndef INPUT_OFFSET
#define INPUT_OFFSET      0x2000u
#endif
#ifndef WEIGHTS_OFFSET
#define WEIGHTS_OFFSET    0x4000u
#endif
#ifndef OUTPUT_OFFSET
#define OUTPUT_OFFSET     0x10000u
#endif
#ifndef ACT0_OFFSET
#define ACT0_OFFSET       0x20000u
#endif
#ifndef ACT1_OFFSET
#define ACT1_OFFSET       (ACT0_OFFSET + ACT_BYTES)
#endif
#ifndef PRIVATE_OFFSET
#define PRIVATE_OFFSET    (ACT1_OFFSET + ACT_BYTES)
#endif
#ifndef PRIVATE_STRIDE
#define PRIVATE_STRIDE    (2u * ACT_BYTES)
#endif

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

static inline int clamp_coord(int v, unsigned limit)
{
	if (v < 0) {
		return 0;
	}
	if (v >= (int)limit) {
		return (int)limit - 1;
	}
	return v;
}

static inline int8_t clamp_i8(int32_t v)
{
	if (v > 127) {
		return 127;
	}
	if (v < -128) {
		return -128;
	}
	return (int8_t)v;
}

static inline int8_t relu_i8(int8_t v)
{
	return v < 0 ? 0 : v;
}

static inline int32_t conv_first_acc_clamped(const uint8_t *input,
					     const int8_t *w,
					     uint32_t y, uint32_t x)
{
	int32_t acc = 0;

	for (int ky = -1; ky <= 1; ky++) {
		const int yy = clamp_coord((int)y + ky, IMG_H);

		for (int kx = -1; kx <= 1; kx++) {
			const int xx = clamp_coord((int)x + kx, IMG_W);
			const int32_t pix =
				(int32_t)input[yy * IMG_W + (unsigned)xx] - 128;

			acc += pix * w[(ky + 1) * 3 + (kx + 1)];
		}
	}

	return acc;
}

static inline int32_t conv_first_acc_interior(const uint8_t *input,
					      const int8_t *w,
					      uint32_t y, uint32_t x,
					      int32_t bias)
{
	const uint8_t *const p = input + y * IMG_W + x;

	return bias +
	       (int32_t)p[-(int)IMG_W - 1] * w[0] +
	       (int32_t)p[-(int)IMG_W] * w[1] +
	       (int32_t)p[-(int)IMG_W + 1] * w[2] +
	       (int32_t)p[-1] * w[3] +
	       (int32_t)p[0] * w[4] +
	       (int32_t)p[1] * w[5] +
	       (int32_t)p[IMG_W - 1] * w[6] +
	       (int32_t)p[IMG_W] * w[7] +
	       (int32_t)p[IMG_W + 1] * w[8];
}

static inline int32_t conv_hidden_acc_clamped(const int8_t *input,
					      const int8_t *w_oc,
					      uint32_t y, uint32_t x)
{
	int32_t acc = 0;

	for (uint32_t ic = 0; ic < CH; ic++) {
		const int8_t *const w_ic = w_oc + ic * K * K;

		for (int ky = -1; ky <= 1; ky++) {
			const int yy = clamp_coord((int)y + ky, IMG_H);

			for (int kx = -1; kx <= 1; kx++) {
				const int xx = clamp_coord((int)x + kx, IMG_W);
				const int8_t pix =
					input[((unsigned)yy * IMG_W +
					       (unsigned)xx) * CH + ic];

				acc += (int32_t)pix *
				       w_ic[(ky + 1) * 3 + (kx + 1)];
			}
		}
	}

	return acc;
}

static inline int32_t conv_hidden_acc_interior(const int8_t *input,
					       const int8_t *w_oc,
					       uint32_t y, uint32_t x)
{
	const int8_t *const p = input + (y * IMG_W + x) * CH;
	const int row = IMG_W * CH;
	int32_t acc = 0;

#ifdef DNCNN_UNROLL_HIDDEN
#if DNCNN_CH != 16u
#error "DNCNN_UNROLL_HIDDEN currently assumes DNCNN_CH=16"
#endif
#define DNCNN_ACC_IC(ic_) do {						\
		const int8_t *const w = w_oc + (ic_) * K * K;		\
		acc += (int32_t)p[-row - (int)CH + (ic_)] * w[0];	\
		acc += (int32_t)p[-row + (ic_)] * w[1];			\
		acc += (int32_t)p[-row + (int)CH + (ic_)] * w[2];	\
		acc += (int32_t)p[-(int)CH + (ic_)] * w[3];		\
		acc += (int32_t)p[(ic_)] * w[4];			\
		acc += (int32_t)p[(int)CH + (ic_)] * w[5];		\
		acc += (int32_t)p[row - (int)CH + (ic_)] * w[6];	\
		acc += (int32_t)p[row + (ic_)] * w[7];			\
		acc += (int32_t)p[row + (int)CH + (ic_)] * w[8];	\
	} while (0)

	DNCNN_ACC_IC(0);
	DNCNN_ACC_IC(1);
	DNCNN_ACC_IC(2);
	DNCNN_ACC_IC(3);
	DNCNN_ACC_IC(4);
	DNCNN_ACC_IC(5);
	DNCNN_ACC_IC(6);
	DNCNN_ACC_IC(7);
	DNCNN_ACC_IC(8);
	DNCNN_ACC_IC(9);
	DNCNN_ACC_IC(10);
	DNCNN_ACC_IC(11);
	DNCNN_ACC_IC(12);
	DNCNN_ACC_IC(13);
	DNCNN_ACC_IC(14);
	DNCNN_ACC_IC(15);

#undef DNCNN_ACC_IC
#else
	for (uint32_t ic = 0; ic < CH; ic++) {
		const int8_t *const w = w_oc + ic * K * K;

		acc += (int32_t)p[-row - (int)CH + (int)ic] * w[0];
		acc += (int32_t)p[-row + (int)ic] * w[1];
		acc += (int32_t)p[-row + (int)CH + (int)ic] * w[2];
		acc += (int32_t)p[-(int)CH + (int)ic] * w[3];
		acc += (int32_t)p[(int)ic] * w[4];
		acc += (int32_t)p[(int)CH + (int)ic] * w[5];
		acc += (int32_t)p[row - (int)CH + (int)ic] * w[6];
		acc += (int32_t)p[row + (int)ic] * w[7];
		acc += (int32_t)p[row + (int)CH + (int)ic] * w[8];
	}
#endif

	return acc;
}

static void conv_first(const uint8_t *input, const int8_t *weights,
		       int8_t *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const int8_t *const w = weights + oc * K * K;
		int32_t bias = 0;

		for (uint32_t i = 0; i < K * K; i++) {
			bias -= 128 * (int32_t)w[i];
		}

		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);

			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior =
					interior_y && x > 0u && x < (IMG_W - 1u);
				const int32_t acc = interior ?
					conv_first_acc_interior(input, w, y, x, bias) :
					conv_first_acc_clamped(input, w, y, x);
				output[(y * IMG_W + x) * CH + oc] =
					relu_i8(clamp_i8((acc + (1 << (FIRST_SHIFT - 1))) >>
							 FIRST_SHIFT));
			}
		}
	}
}

static void conv_hidden(const int8_t *input, const int8_t *weights,
			int8_t *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const int8_t *const w_oc = weights + oc * CH * K * K;

		for (uint32_t y = row0; y < row1; y++) {
			const int interior_y = y > 0u && y < (IMG_H - 1u);

			for (uint32_t x = 0; x < IMG_W; x++) {
				const int interior =
					interior_y && x > 0u && x < (IMG_W - 1u);
				const int32_t acc = interior ?
					conv_hidden_acc_interior(input, w_oc, y, x) :
					conv_hidden_acc_clamped(input, w_oc, y, x);
				output[(y * IMG_W + x) * CH + oc] =
					relu_i8(clamp_i8((acc + (1 << (HIDDEN_SHIFT - 1))) >>
							 HIDDEN_SHIFT));
			}
		}
	}
}

static void conv_final(const int8_t *input, const int8_t *weights,
		       uint8_t *output, uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++) {
		const int interior_y = y > 0u && y < (IMG_H - 1u);

		for (uint32_t x = 0; x < IMG_W; x++) {
			const int interior =
				interior_y && x > 0u && x < (IMG_W - 1u);
			const int32_t acc = interior ?
				conv_hidden_acc_interior(input, weights, y, x) :
				conv_hidden_acc_clamped(input, weights, y, x);

			int32_t v = 128 + ((acc + (1 << (FINAL_SHIFT - 1))) >>
					    FINAL_SHIFT);

			if (v < 0) {
				v = 0;
			} else if (v > 255) {
				v = 255;
			}

			output[y * IMG_W + x] = (uint8_t)v;
		}
	}
}

static uint32_t stripe_checksum(const uint8_t *output,
				uint32_t row0, uint32_t row1)
{
	uint32_t sum = 0;

	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			sum += output[y * IMG_W + x];
		}
	}

	return sum;
}

static void evict_activation_stripe(int8_t *buffer,
				    uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_BOUNDARY_ONLY_EVICT
	if (row0 > 0u) {
		evict(buffer + row0 * IMG_W * CH, IMG_W * CH);
	}
	if (row1 < IMG_H && row1 > row0 + 1u) {
		evict(buffer + (row1 - 1u) * IMG_W * CH, IMG_W * CH);
	}
#else
	evict(buffer + row0 * IMG_W * CH,
	      (row1 - row0) * IMG_W * CH);
#endif
}

static void evict_activation_read_window(const int8_t *buffer,
					 uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_BOUNDARY_ONLY_EVICT
	if (row0 > 0u) {
		evict(buffer + (row0 - 1u) * IMG_W * CH, IMG_W * CH);
	}
	if (row1 < IMG_H) {
		evict(buffer + row1 * IMG_W * CH, IMG_W * CH);
	}
#else
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;

	evict(buffer + read_row0 * IMG_W * CH,
	      (read_row1 - read_row0) * IMG_W * CH);
#endif
}

static void prefetch_activation_read_window(const int8_t *buffer,
					    uint32_t row0, uint32_t row1)
{
#ifdef DNCNN_PREFETCH_READ_WINDOW
	const uint32_t read_row0 = row0 == 0u ? 0u : row0 - 1u;
	const uint32_t read_row1 = row1 == IMG_H ? IMG_H : row1 + 1u;
	uint64_t addr = (uint64_t)(buffer + read_row0 * IMG_W * CH);
	uint64_t lines = ((read_row1 - read_row0) * IMG_W * CH + 63u) >> 6;

	while (lines > 15u) {
		prefetch_va(0, addr, 15u, 64u, 0);
		addr += 16u * 64u;
		lines -= 15u;
	}
	prefetch_va(0, addr, lines, 64u, 0);
#else
	(void)buffer;
	(void)row0;
	(void)row1;
#endif
}

static inline uint32_t expand_row0(uint32_t row, uint32_t halo)
{
	return row > halo ? row - halo : 0u;
}

static inline uint32_t expand_row1(uint32_t row, uint32_t halo)
{
	const uint32_t expanded = row + halo;

	return expanded < IMG_H ? expanded : IMG_H;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	const uint8_t *const input = base + INPUT_OFFSET;
	const int8_t *const weights = (const int8_t *)(base + WEIGHTS_OFFSET);
	uint8_t *const final_output = base + OUTPUT_OFFSET;
	int8_t *const act0 = (int8_t *)(base + ACT0_OFFSET);
	int8_t *const act1 = (int8_t *)(base + ACT1_OFFSET);
	int8_t *const priv0 = (int8_t *)(base + PRIVATE_OFFSET +
					 hart_id * PRIVATE_STRIDE);
	int8_t *const priv1 = priv0 + ACT_BYTES;
	volatile struct dncnn_slot *const slots =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
	volatile struct dncnn_summary *const summary =
		(volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

	for (uint32_t pass = 0; pass < DNCNN_PASSES; pass++) {
#ifdef DNCNN_PRIVATE_FUSED
		const uint32_t first_row0 = expand_row0(row0, 4u);
		const uint32_t first_row1 = expand_row1(row1, 4u);
		const uint32_t hidden0_row0 = expand_row0(row0, 3u);
		const uint32_t hidden0_row1 = expand_row1(row1, 3u);
		const uint32_t hidden1_row0 = expand_row0(row0, 2u);
		const uint32_t hidden1_row1 = expand_row1(row1, 2u);
		const uint32_t hidden2_row0 = expand_row0(row0, 1u);
		const uint32_t hidden2_row1 = expand_row1(row1, 1u);
		const int8_t *const hidden_w = weights + W0_BYTES;
		const int8_t *const final_w = weights + W0_BYTES + WH_BYTES;

		conv_first(input, weights, priv0, first_row0, first_row1);
		FENCE;
		conv_hidden(priv0, hidden_w, priv1, hidden0_row0, hidden0_row1);
		FENCE;
		conv_hidden(priv1, hidden_w + CH * CH * K * K, priv0,
			    hidden1_row0, hidden1_row1);
		FENCE;
		conv_hidden(priv0, hidden_w + 2u * CH * CH * K * K, priv1,
			    hidden2_row0, hidden2_row1);
		FENCE;
		conv_final(priv1, final_w, final_output, row0, row1);
		FENCE;
#ifdef DNCNN_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DNCNN_PASSES)
#endif
		{
			evict(final_output + row0 * IMG_W,
			      (row1 - row0) * IMG_W);
			WAIT_CACHEOPS;
		}
#ifdef DNCNN_PRIVATE_FUSED_PASS_BARRIER
		bench_barrier();
#endif
#elif defined(DNCNN_FUSED_HALO)
		const uint32_t first_row0 = expand_row0(row0, 4u);
		const uint32_t first_row1 = expand_row1(row1, 4u);
		const uint32_t hidden0_row0 = expand_row0(row0, 3u);
		const uint32_t hidden0_row1 = expand_row1(row1, 3u);
		const uint32_t hidden1_row0 = expand_row0(row0, 2u);
		const uint32_t hidden1_row1 = expand_row1(row1, 2u);
		const uint32_t hidden2_row0 = expand_row0(row0, 1u);
		const uint32_t hidden2_row1 = expand_row1(row1, 1u);
		const int8_t *const hidden_w = weights + W0_BYTES;
		const int8_t *const final_w = weights + W0_BYTES + WH_BYTES;

		conv_first(input, weights, act0, first_row0, first_row1);
		FENCE;
		conv_hidden(act0, hidden_w, act1, hidden0_row0, hidden0_row1);
		FENCE;
		conv_hidden(act1, hidden_w + CH * CH * K * K, act0,
			    hidden1_row0, hidden1_row1);
		FENCE;
		conv_hidden(act0, hidden_w + 2u * CH * CH * K * K, act1,
			    hidden2_row0, hidden2_row1);
		FENCE;
		conv_final(act1, final_w, final_output, row0, row1);
		FENCE;
#ifdef DNCNN_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DNCNN_PASSES)
#endif
		{
			evict(final_output + row0 * IMG_W,
			      (row1 - row0) * IMG_W);
			WAIT_CACHEOPS;
		}
#ifndef DNCNN_SKIP_FINAL_BARRIER
		bench_barrier();
#endif
#else
		conv_first(input, weights, act0, row0, row1);
		FENCE;
		evict_activation_stripe(act0, row0, row1);
		WAIT_CACHEOPS;
#ifndef DNCNN_SKIP_LAYER_BARRIERS
		bench_barrier();
#endif

		const int8_t *hidden_w = weights + W0_BYTES;
		for (uint32_t layer = 0; layer < (LAYERS - 2u); layer++) {
			const int8_t *const src = (layer & 1u) ? act1 : act0;
			int8_t *const dst = (layer & 1u) ? act0 : act1;

			evict_activation_read_window(src, row0, row1);
			WAIT_CACHEOPS;
			prefetch_activation_read_window(src, row0, row1);

			conv_hidden(src, hidden_w + layer * CH * CH * K * K,
				    dst, row0, row1);
			FENCE;
			evict_activation_stripe(dst, row0, row1);
			WAIT_CACHEOPS;
#ifndef DNCNN_SKIP_LAYER_BARRIERS
			bench_barrier();
#endif
		}

		const int8_t *const last_hidden =
			((LAYERS - 2u) & 1u) ? act1 : act0;
		const int8_t *const final_w = weights + W0_BYTES + WH_BYTES;

		evict_activation_read_window(last_hidden, row0, row1);
		WAIT_CACHEOPS;
		prefetch_activation_read_window(last_hidden, row0, row1);

		conv_final(last_hidden, final_w, final_output, row0, row1);
		FENCE;
#ifdef DNCNN_EVICT_OUTPUT_LAST_ONLY
		if (pass + 1u == DNCNN_PASSES)
#endif
		{
		evict(final_output + row0 * IMG_W, (row1 - row0) * IMG_W);
		WAIT_CACHEOPS;
		}
#ifndef DNCNN_SKIP_FINAL_BARRIER
#ifndef DNCNN_SKIP_NONLAST_FINAL_BARRIER
		bench_barrier();
#else
		if (pass + 1u == DNCNN_PASSES) {
			bench_barrier();
		}
#endif
#endif
#endif
	}

	const uint32_t checksum = stripe_checksum(final_output, row0, row1);
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
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t slot_checksum_sum = 0;
		uint32_t output_sum = 0;

		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == DNCNN_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}

		for (uint32_t i = 0; i < IMG_BYTES; i++) {
			output_sum += final_output[i];
		}

		const uint64_t macs_per_pass =
			(uint64_t)IMG_W * IMG_H *
			((uint64_t)CH * K * K +
			 (uint64_t)(LAYERS - 2u) * CH * CH * K * K +
			 (uint64_t)CH * K * K);
		const uint64_t ops = macs_per_pass * DNCNN_PASSES * 2u;

		summary->magic = DNCNN_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->passes = DNCNN_PASSES;
		summary->width = IMG_W;
		summary->height = IMG_H;
		summary->channels = CH;
		summary->layers = LAYERS;
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
