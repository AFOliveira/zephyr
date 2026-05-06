/*
 * Single-CSR negative/positive control for ET-SoC1 U-mode PMC probing.
 *
 * Build this source once per CSR_NUM.  If the CSR read traps, the host dump
 * still contains the START magic but not the DONE magic.
 */

#include <stdint.h>

#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef CSR_NUM
#define CSR_NUM 0xC03
#endif

#ifndef CSR_ID
#define CSR_ID 3
#endif

#define CSR_PROBE_START_MAGIC 0x43535253u
#define CSR_PROBE_DONE_MAGIC  0x43535244u

#define STR2(x) #x
#define STR(x) STR2(x)

struct csr_probe_record {
	uint32_t start_magic;
	uint32_t done_magic;
	uint32_t csr_num;
	uint32_t csr_id;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t reserved;
	uint64_t before;
	uint64_t after;
	uint64_t delta;
	uint64_t checksum;
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

static inline uint64_t read_probe_csr(void)
{
	uint64_t value;

	__asm__ __volatile__(".p2align 4\n"
			     "csrr %0, " STR(CSR_NUM) "\n"
			     "csrr %0, " STR(CSR_NUM) "\n"
			     "csrr %0, " STR(CSR_NUM) "\n"
			     "csrr %0, " STR(CSR_NUM) "\n"
			     : "=r"(value));
	return value;
}

static uint64_t small_work(void)
{
	uint64_t x = 0xfeedface12345678ULL;

	for (uint32_t i = 0; i < 100000u; i++) {
		x ^= x << 5;
		x += i + (x >> 19);
		x ^= x >> 7;
		__asm__ __volatile__("" : "+r"(x) :: "memory");
	}

	return x;
}

int main(uintptr_t arg_area)
{
	const uintptr_t base = buffer_base_from_args(arg_area);
	volatile struct csr_probe_record *rec =
		(volatile struct csr_probe_record *)base;

	rec->start_magic = CSR_PROBE_START_MAGIC;
	rec->done_magic = 0;
	rec->csr_num = CSR_NUM;
	rec->csr_id = CSR_ID;
	rec->hart_id = get_hart_id();
	rec->minion_id = get_minion_id();
	rec->thread_id = get_thread_id();
	rec->reserved = 0;
	rec->before = 0;
	rec->after = 0;
	rec->delta = 0;
	rec->checksum = 0;
	evict((const void *)rec, sizeof(*rec));

	const uint64_t before = read_probe_csr();
	const uint64_t checksum = small_work();
	const uint64_t after = read_probe_csr();

	rec->before = before;
	rec->after = after;
	rec->delta = after - before;
	rec->checksum = checksum;
	rec->done_magic = CSR_PROBE_DONE_MAGIC;
	evict((const void *)rec, sizeof(*rec));

	return 0;
}
