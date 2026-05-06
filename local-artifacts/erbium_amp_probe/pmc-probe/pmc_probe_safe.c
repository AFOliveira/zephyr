/*
 * ET-SoC1 U-mode PMC probe.
 *
 * This runs under the erbium-soc1sim U-mode firmware path.  It records:
 * - direct U-mode reads of hpmcounter3..8 on each launched hart
 * - public WorkerMinion PMC sample syscalls for Shire Cache and Memshire PMCs
 *
 * Results are written to the 16 MiB device buffer passed as a0 by the host
 * launcher, then evicted for host-side dumping.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/syscall.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif

#define PMC_PROBE_MAGIC          0x504D4350u
#define PMC_PROBE_DIRECT_MAGIC   0x44495250u
#define PMC_PROBE_SYSCALL_MAGIC  0x53595350u
#define PMC_PROBE_SUMMARY_MAGIC  0x53554D50u

#define DIRECT_OFFSET            0x0000u
#define SYSCALL_OFFSET           0x4000u
#define SUMMARY_OFFSET           0x8000u
#define SCRATCH_OFFSET           0x10000u
#define SCRATCH_STRIDE           0x20000u
#define SCRATCH_BYTES            0x10000u

#define MAX_HARTS                16u
#define HPM_COUNT                6u
#define SC_BANKS                 4u
#define MS_COUNT                 8u
#define PMC_PER_BLOCK            3u
#define STAGES                   3u

#define BARRIER_FLB              2u
#define BARRIER_FCC              FCC_0

struct direct_record {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t active_harts;
	uint32_t done;
	uint32_t reserved0;
	uint32_t reserved1;
	uint64_t begin[HPM_COUNT];
	uint64_t after_instr[HPM_COUNT];
	uint64_t after_mem[HPM_COUNT];
	uint64_t after_icache[HPM_COUNT];
	uint64_t final[HPM_COUNT];
	uint64_t instr_checksum;
	uint64_t mem_checksum;
	uint64_t icache_checksum;
};

struct syscall_record {
	uint32_t magic;
	uint32_t kind;       /* 0 = SC, 1 = MS */
	uint32_t stage;      /* 0 = before, 1 = after work, 2 = after repeat */
	uint32_t block_id;   /* SC bank or MS id */
	uint32_t pmc_id;     /* 0 = cycle, 1 = PMC0, 2 = PMC1 */
	uint32_t shire_id;
	uint32_t hart_id;
	uint32_t reserved;
	uint64_t value;
};

struct summary_record {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t done_count;
	uint32_t shire_id;
	uint32_t hart0_id;
	uint32_t syscall_records;
	uint32_t sc_banks;
	uint32_t ms_count;
	uint64_t hpm_delta_sum[HPM_COUNT];
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

static inline void probe_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BARRIER_FLB, BARRIER_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

#define HPM_SAFE_READ(counter, value)                \
	do {                                         \
		__asm__ __volatile__(".p2align 4\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     "csrr %0," counter "\n" \
				     : "=r"(value));       \
	} while (0)

static inline uint64_t read_hpm(uint32_t idx)
{
	uint64_t value = 0;

	switch (idx) {
	case 0:
		HPM_SAFE_READ("hpmcounter3", value);
		break;
	case 1:
		HPM_SAFE_READ("hpmcounter4", value);
		break;
	case 2:
		HPM_SAFE_READ("hpmcounter5", value);
		break;
	case 3:
		HPM_SAFE_READ("hpmcounter6", value);
		break;
	case 4:
		HPM_SAFE_READ("hpmcounter7", value);
		break;
	case 5:
		HPM_SAFE_READ("hpmcounter8", value);
		break;
	default:
		break;
	}

	return value;
}

static void read_all_hpm(uint64_t *dst)
{
	for (uint32_t i = 0; i < HPM_COUNT; i++) {
		dst[i] = read_hpm(i);
	}
}

static uint64_t instruction_work(uint32_t hart)
{
	uint64_t x = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)hart << 32);

	for (uint32_t i = 0; i < 200000u; i++) {
		x ^= x << 7;
		x ^= x >> 9;
		x += 0x100000001b3ULL + i + hart;
		if ((x & 0x20u) != 0u) {
			x += x >> 17;
		} else {
			x ^= x << 11;
		}
		__asm__ __volatile__("" : "+r"(x) :: "memory");
	}

	return x;
}

static uint64_t memory_work(uintptr_t base, uint32_t hart)
{
	volatile uint64_t *scratch =
		(volatile uint64_t *)(base + SCRATCH_OFFSET +
				      (uintptr_t)hart * SCRATCH_STRIDE);
	uint64_t acc = 0xC001D00DULL + hart;

	evict((const void *)scratch, SCRATCH_BYTES);

	for (uint32_t pass = 0; pass < 8u; pass++) {
		for (uint32_t i = 0; i < (SCRATCH_BYTES / sizeof(uint64_t)); i += 8u) {
			const uint64_t v = acc + ((uint64_t)pass << 32) + i;
			scratch[i] = v;
			acc ^= scratch[(i + 24u) & ((SCRATCH_BYTES / sizeof(uint64_t)) - 1u)];
			acc += v ^ (acc >> 13);
		}
		evict((const void *)scratch, SCRATCH_BYTES);
	}

	return acc;
}

#define ICACHE_FN(n) \
	static __attribute__((noinline)) uint64_t icache_fn_##n(uint64_t x) \
	{ \
		x += (uint64_t)(0x1000u + (n)); \
		x ^= x << ((n) % 13u + 1u); \
		x += x >> ((n) % 11u + 1u); \
		return x ^ (uint64_t)(0x5a5a0000u + (n)); \
	}

ICACHE_FN(0)
ICACHE_FN(1)
ICACHE_FN(2)
ICACHE_FN(3)
ICACHE_FN(4)
ICACHE_FN(5)
ICACHE_FN(6)
ICACHE_FN(7)
ICACHE_FN(8)
ICACHE_FN(9)
ICACHE_FN(10)
ICACHE_FN(11)
ICACHE_FN(12)
ICACHE_FN(13)
ICACHE_FN(14)
ICACHE_FN(15)

static uint64_t icache_work(uint32_t hart)
{
	uint64_t x = 0x123456789abcdef0ULL ^ hart;

	for (uint32_t i = 0; i < 20000u; i++) {
		switch ((i + hart) & 15u) {
		case 0: x = icache_fn_0(x); break;
		case 1: x = icache_fn_1(x); break;
		case 2: x = icache_fn_2(x); break;
		case 3: x = icache_fn_3(x); break;
		case 4: x = icache_fn_4(x); break;
		case 5: x = icache_fn_5(x); break;
		case 6: x = icache_fn_6(x); break;
		case 7: x = icache_fn_7(x); break;
		case 8: x = icache_fn_8(x); break;
		case 9: x = icache_fn_9(x); break;
		case 10: x = icache_fn_10(x); break;
		case 11: x = icache_fn_11(x); break;
		case 12: x = icache_fn_12(x); break;
		case 13: x = icache_fn_13(x); break;
		case 14: x = icache_fn_14(x); break;
		default: x = icache_fn_15(x); break;
		}
	}

	return x;
}

static void sample_syscalls(uintptr_t base, uint32_t stage)
{
	volatile struct syscall_record *records =
		(volatile struct syscall_record *)(base + SYSCALL_OFFSET);
	uint32_t idx = stage * (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK;
	const uint32_t shire_id = get_shire_id();
	const uint32_t hart_id = get_hart_id();

	for (uint32_t bank = 0; bank < SC_BANKS; bank++) {
		for (uint32_t pmc = 0; pmc < PMC_PER_BLOCK; pmc++) {
			volatile struct syscall_record *r = &records[idx++];

			r->magic = PMC_PROBE_SYSCALL_MAGIC;
			r->kind = 0;
			r->stage = stage;
			r->block_id = bank;
			r->pmc_id = pmc;
			r->shire_id = shire_id;
			r->hart_id = hart_id;
			r->reserved = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_SC_SAMPLE,
						     shire_id, bank, pmc);
		}
	}

	for (uint32_t ms = 0; ms < MS_COUNT; ms++) {
		for (uint32_t pmc = 0; pmc < PMC_PER_BLOCK; pmc++) {
			volatile struct syscall_record *r = &records[idx++];

			r->magic = PMC_PROBE_SYSCALL_MAGIC;
			r->kind = 1;
			r->stage = stage;
			r->block_id = ms;
			r->pmc_id = pmc;
			r->shire_id = shire_id;
			r->hart_id = hart_id;
			r->reserved = 0;
			r->value = (uint64_t)syscall(SYSCALL_PMC_MS_SAMPLE,
						     ms, pmc, 0);
		}
	}

	evict((const void *)(base + SYSCALL_OFFSET),
	      STAGES * (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK *
		      sizeof(struct syscall_record));
}

int main(uintptr_t arg_area)
{
	const uint32_t hart = get_hart_id();
	const uintptr_t base = buffer_base_from_args(arg_area);

	if (hart >= ACTIVE_HARTS || hart >= MAX_HARTS) {
		return 0;
	}

	volatile struct direct_record *direct =
		(volatile struct direct_record *)(base + DIRECT_OFFSET);
	volatile struct direct_record *rec = &direct[hart];

	rec->magic = PMC_PROBE_DIRECT_MAGIC;
	rec->hart_id = hart;
	rec->minion_id = get_minion_id();
	rec->thread_id = get_thread_id();
	rec->active_harts = ACTIVE_HARTS;
	rec->done = 0;
	rec->reserved0 = PMC_PROBE_MAGIC;
	rec->reserved1 = 0;
	evict((const void *)rec, sizeof(*rec));

	if (hart == 0u) {
		sample_syscalls(base, 0);
	}

	probe_barrier();
	read_all_hpm((uint64_t *)rec->begin);
	rec->instr_checksum = instruction_work(hart);
	read_all_hpm((uint64_t *)rec->after_instr);
	rec->mem_checksum = memory_work(base, hart);
	read_all_hpm((uint64_t *)rec->after_mem);
	rec->icache_checksum = icache_work(hart);
	read_all_hpm((uint64_t *)rec->after_icache);
	read_all_hpm((uint64_t *)rec->final);
	rec->done = 1;
	evict((const void *)rec, sizeof(*rec));

	probe_barrier();

	if (hart == 0u) {
		volatile struct summary_record *summary =
			(volatile struct summary_record *)(base + SUMMARY_OFFSET);
		uint32_t done = 0;

		for (uint32_t h = 0; h < ACTIVE_HARTS && h < MAX_HARTS; h++) {
			if (direct[h].done == 1u &&
			    direct[h].magic == PMC_PROBE_DIRECT_MAGIC) {
				done++;
			}
		}

		sample_syscalls(base, 1);
		sample_syscalls(base, 2);

		summary->magic = PMC_PROBE_SUMMARY_MAGIC;
		summary->active_harts = ACTIVE_HARTS;
		summary->done_count = done;
		summary->shire_id = get_shire_id();
		summary->hart0_id = hart;
		summary->syscall_records = STAGES * (SC_BANKS + MS_COUNT) * PMC_PER_BLOCK;
		summary->sc_banks = SC_BANKS;
		summary->ms_count = MS_COUNT;

		for (uint32_t i = 0; i < HPM_COUNT; i++) {
			uint64_t sum = 0;

			for (uint32_t h = 0; h < ACTIVE_HARTS && h < MAX_HARTS; h++) {
				sum += direct[h].final[i] - direct[h].begin[i];
			}
			summary->hpm_delta_sum[i] = sum;
		}

		evict((const void *)summary, sizeof(*summary));
	}

	probe_barrier();
	return 0;
}
