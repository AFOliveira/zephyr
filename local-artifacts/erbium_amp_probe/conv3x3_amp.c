/*
 * Local AMP probe for native Erbium.
 *
 * Runs one thread per minion (even hart IDs only).  Each active hart owns a
 * 16-row stripe of a 128x128 input image and writes a disjoint output stripe.
 * Output stripes and per-hart status slots are evicted before hart 0 reads
 * cross-hart data, because Erbium L1D is not coherent across minions.
 */

#include <stddef.h>
#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define CONV_MAGIC              0xC03A3A01u
#define IMG_W                   128u
#define IMG_H                   128u
#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS            8u
#endif
#define ROWS_PER_HART           (IMG_H / ACTIVE_HARTS)
#define IMG_BYTES               (IMG_W * IMG_H)
#define SLOT_BYTES              64u
#define SLOTS_BYTES             (ACTIVE_HARTS * SLOT_BYTES)
#define SUMMARY_BYTES           64u

#define SUMMARY_OFFSET_FROM_END 0x40u
#define SLOTS_OFFSET_FROM_END   (SUMMARY_OFFSET_FROM_END + SLOTS_BYTES)
#define OUTPUT_OFFSET_FROM_END  (SLOTS_OFFSET_FROM_END + IMG_BYTES)
#define INPUT_OFFSET_FROM_END   (OUTPUT_OFFSET_FROM_END + IMG_BYTES)

#define BARRIER_FLB             2u
#define INPUT_BARRIER_FLB       3u
#ifndef BARRIER_MASK_T0
#define BARRIER_MASK_T0         0xFFu
#endif
#ifndef BARRIER_MASK_T1
#define BARRIER_MASK_T1         0x00u
#endif

struct hart_slot {
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t row0;
	uint32_t rows;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[9];
};

struct summary {
	uint32_t magic;
	uint32_t width;
	uint32_t height;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t reserved[10];
};

#ifdef STATIC_BUFFERS
static uint8_t static_input[IMG_BYTES] __attribute__((aligned(64)));
static uint8_t static_output[IMG_BYTES] __attribute__((aligned(64)));
static struct hart_slot static_slots[ACTIVE_HARTS] __attribute__((aligned(64)));
volatile struct summary conv_summary __attribute__((aligned(64), used));
#endif

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

static void init_input(uint8_t *input)
{
	uint32_t state = 0x12345678u;

	for (unsigned i = 0; i < IMG_BYTES; i++) {
		const unsigned y = i / IMG_W;
		const unsigned x = i % IMG_W;

		state = (1664525u * state) + 1013904223u;
		input[i] = (uint8_t)(((state >> 24) + x * 3u + y * 5u) & 0xffu);
	}
}

int main(void)
{
	const unsigned hart_id = get_hart_id();

#if ACTIVE_HARTS <= 8u
	if (hart_id & 1u) {
		return 0;
	}
	const unsigned hart = hart_id >> 1;
#else
	const unsigned hart = hart_id;
#endif
	if (hart >= ACTIVE_HARTS) {
		return 0;
	}

#ifdef STATIC_BUFFERS
	uint8_t *const input = static_input;
	uint8_t *const output = static_output;
	struct hart_slot *const slots = static_slots;
	volatile struct summary *const sum = &conv_summary;
#else
	uint8_t *const heap_end = (uint8_t *)heap0_end;
	uint8_t *const input = heap_end - INPUT_OFFSET_FROM_END;
	uint8_t *const output = heap_end - OUTPUT_OFFSET_FROM_END;
	struct hart_slot *const slots =
		(struct hart_slot *)(heap_end - SLOTS_OFFSET_FROM_END);
	struct summary *const sum =
		(struct summary *)(heap_end - SUMMARY_OFFSET_FROM_END);
#endif

#ifdef SELF_INIT_INPUT
	if (hart == 0u) {
		init_input(input);
		FENCE;
		evict(input, IMG_BYTES);
		WAIT_CACHEOPS;
	}

	shire_barrier(INPUT_BARRIER_FLB, FCC_1, ACTIVE_HARTS,
		      BARRIER_MASK_T0, BARRIER_MASK_T1);
#endif

	const unsigned row0 = hart * ROWS_PER_HART;
	const unsigned row1 = row0 + ROWS_PER_HART;
	uint32_t checksum = 0;

	for (unsigned y = row0; y < row1; y++) {
		for (unsigned x = 0; x < IMG_W; x++) {
			const uint8_t v = conv3x3_q4(input, y, x);
			output[y * IMG_W + x] = v;
			checksum += v;
		}
	}

	struct hart_slot *const slot = &slots[hart];
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

	shire_barrier(BARRIER_FLB, FCC_0, ACTIVE_HARTS,
		      BARRIER_MASK_T0, BARRIER_MASK_T1);

	if (hart == 0u) {
		uint32_t active_mask = 0;
		uint32_t done_count = 0;
		uint32_t output_sum = 0;

		for (unsigned h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
			}
		}

		for (unsigned i = 0; i < IMG_BYTES; i++) {
			output_sum += output[i];
		}

		sum->magic = CONV_MAGIC;
		sum->width = IMG_W;
		sum->height = IMG_H;
		sum->active_mask = active_mask;
		sum->done_count = done_count;
		sum->output_sum = output_sum;

		FENCE;
		evict(sum, sizeof(*sum));
		WAIT_CACHEOPS;
	}

	return 0;
}
