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

struct probe_args {
	uint64_t out_dev;
	uint64_t loops;
	uint64_t marker;
};

struct probe_result {
	uint64_t shire;
	uint64_t neighborhood;
	uint64_t hart;
	uint64_t minion;
	uint64_t thread;
	uint64_t loops;
	uint64_t marker;
	uint64_t accumulator;
};

static volatile uint64_t sink;

static uint64_t burn_cycles(uint64_t loops, uint64_t seed)
{
	uint64_t acc = seed | 1U;

	for (uint64_t i = 0; i < loops; i++) {
		acc ^= (i + 0x9e3779b97f4a7c15ULL);
		acc = (acc << 7) | (acc >> 57);
		acc += seed + i;
		__asm__ volatile("" : "+r"(acc) :: "memory");
	}

	sink = acc;
	return acc;
}

static void publish_result(const struct probe_result *result)
{
	void *args = cm_umode_kernel_args;

	if ((args == NULL) || (args == (void *)~(uintptr_t)0)) {
		return;
	}

	const struct probe_args *probe_args = (const struct probe_args *)args;
	if (probe_args->out_dev == 0U) {
		return;
	}

	volatile struct probe_result *dst =
		(volatile struct probe_result *)(uintptr_t)probe_args->out_dev;
	const uint8_t *src = (const uint8_t *)result;
	volatile uint8_t *out = (volatile uint8_t *)dst;

	for (size_t i = 0; i < sizeof(*result); i++) {
		out[i] = src[i];
	}

	(void)cache_ops_priv_evict_whole_l1_l2();
}

int main(void)
{
	struct probe_args local_args = {0};
	void *args = cm_umode_kernel_args;

	if ((args != NULL) && (args != (void *)~(uintptr_t)0)) {
		local_args = *(const struct probe_args *)args;
	}

	unsigned int hart = get_hart_id();
	unsigned int shire = get_shire_id();
	unsigned int neigh = get_neighborhood_id();
	unsigned int minion = get_minion_id();
	unsigned int thread = get_thread_id();
	uint64_t loops = local_args.loops;
	uint64_t marker = local_args.marker;
	uint64_t acc = burn_cycles(loops, marker ^ hart);

	struct probe_result result = {
		.shire = shire,
		.neighborhood = neigh,
		.hart = hart,
		.minion = minion,
		.thread = thread,
		.loops = loops,
		.marker = marker,
		.accumulator = acc,
	};

	publish_result(&result);
	return 0;
}
