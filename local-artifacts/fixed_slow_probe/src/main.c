/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/kernel.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/isa/hart.h>

extern void *cm_umode_kernel_args;

#define SLOW_PROBE_LOOPS 100000000ULL

static volatile uint64_t sink;

static uint64_t burn_cycles(uint64_t seed)
{
	uint64_t acc = seed | 1U;

	for (uint64_t i = 0; i < SLOW_PROBE_LOOPS; i++) {
		acc ^= i + 0x9e3779b97f4a7c15ULL;
		acc = (acc << 7) | (acc >> 57);
		acc += seed + i;
		__asm__ volatile("" : "+r"(acc) :: "memory");
	}

	sink = acc;
	return acc;
}

static void publish_results(const int32_t *src, size_t count)
{
	void *args = cm_umode_kernel_args;

	if ((args == NULL) || (args == (void *)~(uintptr_t)0)) {
		return;
	}

	uint64_t out_dev = *(volatile uint64_t *)args;
	if (out_dev == 0U) {
		return;
	}

	volatile int32_t *dst = (volatile int32_t *)(uintptr_t)out_dev;

	for (size_t i = 0; i < count; i++) {
		dst[i] = src[i];
	}

	(void)cache_ops_priv_evict_whole_l1_l2();
}

int main(void)
{
	unsigned int hart = get_hart_id();
	unsigned int shire = get_shire_id();
	unsigned int neigh = get_neighborhood_id();
	unsigned int minion = get_minion_id();
	unsigned int thread = get_thread_id();
	uint64_t acc = burn_cycles(((uint64_t)hart << 32) ^ shire);

	int32_t out[15] = {
		(int32_t)shire,
		(int32_t)neigh,
		(int32_t)hart,
		(int32_t)minion,
		(int32_t)thread,
		(int32_t)(SLOW_PROBE_LOOPS & 0xffffffffU),
		(int32_t)(SLOW_PROBE_LOOPS >> 32),
		(int32_t)(acc & 0xffffffffU),
		(int32_t)(acc >> 32),
	};

	publish_results(out, ARRAY_SIZE(out));
	return 0;
}
