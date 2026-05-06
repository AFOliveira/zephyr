/*
 * ET-SoC1/Erbium all-hart compute probe.
 *
 * The launcher passes a host-allocated 16 MiB device buffer pointer as the
 * kernel argument.  All 16 local harts compute disjoint stripes of one 128x128
 * 3x3 blur, publish one cache-line-sized status slot, then hart 0 summarizes
 * the result after a shire barrier.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define CONV_ARG_MAGIC   0xC03A3A02u
#define IMG_W            128u
#define IMG_H            128u
#define ACTIVE_HARTS     16u
#define ROWS_PER_HART    (IMG_H / ACTIVE_HARTS)
#define IMG_BYTES        (IMG_W * IMG_H)

#define SLOT_BYTES       64u
#define SLOTS_OFFSET     0x0000u
#define SUMMARY_OFFSET   0x1000u
#define INPUT_OFFSET     0x2000u
#define OUTPUT_OFFSET    0x6000u

#define BARRIER_FLB      2u
#define BARRIER_FCC      FCC_0
#define BARRIER_MASK_T0  0xffu
#define BARRIER_MASK_T1  0xffu

struct hart_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t row0;
	uint32_t rows;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[8];
};

struct conv_summary {
	uint32_t magic;
	uint32_t width;
	uint32_t height;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t expected_harts;
	uint32_t reserved[9];
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

static inline uint8_t clamp_pixel(const uint8_t *image, int y, int x)
{
	if (y < 0) {
		y = 0;
	} else if (y >= (int)IMG_H) {
		y = (int)IMG_H - 1;
	}

	if (x < 0) {
		x = 0;
	} else if (x >= (int)IMG_W) {
		x = (int)IMG_W - 1;
	}

	return image[(unsigned)y * IMG_W + (unsigned)x];
}

static inline uint8_t conv3x3_q4(const uint8_t *image, unsigned y, unsigned x)
{
	const int yy = (int)y;
	const int xx = (int)x;
	uint32_t acc = 0;

	acc += clamp_pixel(image, yy - 1, xx - 1);
	acc += 2u * clamp_pixel(image, yy - 1, xx);
	acc += clamp_pixel(image, yy - 1, xx + 1);
	acc += 2u * clamp_pixel(image, yy, xx - 1);
	acc += 4u * clamp_pixel(image, yy, xx);
	acc += 2u * clamp_pixel(image, yy, xx + 1);
	acc += clamp_pixel(image, yy + 1, xx - 1);
	acc += 2u * clamp_pixel(image, yy + 1, xx);
	acc += clamp_pixel(image, yy + 1, xx + 1);

	return (uint8_t)((acc + 8u) >> 4);
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	const uint8_t *const input = base + INPUT_OFFSET;
	uint8_t *const output = base + OUTPUT_OFFSET;
	volatile struct hart_slot *const slots =
		(volatile struct hart_slot *)(base + SLOTS_OFFSET);
	volatile struct conv_summary *const summary =
		(volatile struct conv_summary *)(base + SUMMARY_OFFSET);

	const unsigned row0 = hart_id * ROWS_PER_HART;
	const unsigned row1 = row0 + ROWS_PER_HART;
	uint32_t checksum = 0;

	for (unsigned y = row0; y < row1; y++) {
		for (unsigned x = 0; x < IMG_W; x++) {
			const uint8_t v = conv3x3_q4(input, y, x);
			output[y * IMG_W + x] = v;
			checksum += v;
		}
	}

	volatile struct hart_slot *const slot =
		(volatile struct hart_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);
	slot->magic = CONV_ARG_MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->row0 = row0;
	slot->rows = ROWS_PER_HART;
	slot->checksum = checksum;
	slot->done = 1u;

	FENCE;
	evict(output + row0 * IMG_W, ROWS_PER_HART * IMG_W);
	evict(slot, sizeof(*slot));
	WAIT_CACHEOPS;

	shire_barrier(BARRIER_FLB, BARRIER_FCC, ACTIVE_HARTS,
		      BARRIER_MASK_T0, BARRIER_MASK_T1);

	if (hart_id == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t output_sum = 0;

		for (unsigned h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == CONV_ARG_MAGIC &&
			    slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
			}
		}

		for (unsigned i = 0; i < IMG_BYTES; i++) {
			output_sum += output[i];
		}

		summary->magic = CONV_ARG_MAGIC;
		summary->width = IMG_W;
		summary->height = IMG_H;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_sum = output_sum;
		summary->expected_harts = ACTIVE_HARTS;

		FENCE;
		evict(summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
