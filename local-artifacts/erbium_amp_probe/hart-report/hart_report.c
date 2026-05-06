/*
 * Minimal ET-SoC1/Erbium hart report probe.
 *
 * The launcher passes a 16 MiB device buffer pointer as kernel arg.  Each
 * selected hart writes one cache-line-sized record at buffer[hartid * 64],
 * evicts that line, and returns.  The host dump then tells us which U-mode
 * harts actually executed without relying on a shared summary or barrier.
 */

#include <stdint.h>

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#define HART_REPORT_MAGIC 0xA17E0001u
#define HART_RECORD_BYTES 64u

#ifndef SELECT_MODE
#define SELECT_MODE 0
#endif

struct hart_record {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t selected;
	uint32_t select_mode;
	uint32_t seq;
	uint32_t reserved[9];
};

static uintptr_t report_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (64u * HART_RECORD_BYTES);
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (64u * HART_RECORD_BYTES);
	}

	return ptr;
}

static int hart_selected(uint32_t hart_id)
{
#if SELECT_MODE == 0
	return hart_id < 16u;
#elif SELECT_MODE == 1
	return hart_id < 16u && ((hart_id & 1u) == 0u);
#elif SELECT_MODE == 2
	return hart_id < 16u && ((hart_id & 1u) == 1u);
#else
	return 0;
#endif
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (!hart_selected(hart_id)) {
		return 0;
	}

	volatile struct hart_record *const records =
		(volatile struct hart_record *)report_base_from_args(arg_area);
	volatile struct hart_record *const rec = &records[hart_id];

	rec->magic = HART_REPORT_MAGIC;
	rec->hart_id = hart_id;
	rec->minion_id = get_minion_id();
	rec->thread_id = get_thread_id();
	rec->selected = 1u;
	rec->select_mode = SELECT_MODE;
	rec->seq = 0x1000u + hart_id;

	FENCE;
	evict((void *)rec, sizeof(*rec));
	WAIT_CACHEOPS;

	return 0;
}
