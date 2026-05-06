/*
 * Probe edge PMC IDs through the public U-mode SC/MS PMC sample syscalls.
 */

#include <stdint.h>

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/syscall.h"

extern char heap0_end[];

#define EDGE_MAGIC 0x504D4345u
#define EDGE_OFFSET 0x0000u
#define SC_BANKS 4u
#define MS_COUNT 8u
#define PMC_IDS 4u

struct edge_record {
	uint32_t magic;
	uint32_t kind;
	uint32_t block;
	uint32_t pmc;
	uint32_t shire;
	uint32_t hart;
	uint32_t reserved0;
	uint32_t reserved1;
	uint64_t value;
	uint64_t reserved2;
	uint64_t reserved3;
	uint64_t reserved4;
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

int main(uintptr_t arg_area)
{
	const uintptr_t base = buffer_base_from_args(arg_area);
	volatile struct edge_record *records =
		(volatile struct edge_record *)(base + EDGE_OFFSET);
	const uint32_t shire = get_shire_id();
	const uint32_t hart = get_hart_id();
	uint32_t idx = 0;

	for (uint32_t bank = 0; bank < SC_BANKS; bank++) {
		for (uint32_t pmc = 0; pmc < PMC_IDS; pmc++) {
			volatile struct edge_record *r = &records[idx++];

			r->magic = EDGE_MAGIC;
			r->kind = 0;
			r->block = bank;
			r->pmc = pmc;
			r->shire = shire;
			r->hart = hart;
			r->reserved0 = 0;
			r->reserved1 = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_SC_SAMPLE,
						     shire, bank, pmc);
			r->reserved2 = 0;
			r->reserved3 = 0;
			r->reserved4 = 0;
			evict((const void *)r, sizeof(*r));
		}
	}

	for (uint32_t ms = 0; ms < MS_COUNT; ms++) {
		for (uint32_t pmc = 0; pmc < PMC_IDS; pmc++) {
			volatile struct edge_record *r = &records[idx++];

			r->magic = EDGE_MAGIC;
			r->kind = 1;
			r->block = ms;
			r->pmc = pmc;
			r->shire = shire;
			r->hart = hart;
			r->reserved0 = 0;
			r->reserved1 = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_MS_SAMPLE,
						     ms, pmc, 0);
			r->reserved2 = 0;
			r->reserved3 = 0;
			r->reserved4 = 0;
			evict((const void *)r, sizeof(*r));
		}
	}

	return 0;
}
