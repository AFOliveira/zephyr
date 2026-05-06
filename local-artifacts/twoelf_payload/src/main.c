/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/kernel.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/cacheops-umode.h>

extern void *cm_umode_kernel_args;

#ifndef TWOELF_PAYLOAD_BASE
#define TWOELF_PAYLOAD_BASE 1000
#endif

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
	int32_t out[15];

	et_printf("twoelf-payload: base=%d\n", TWOELF_PAYLOAD_BASE);

	for (int i = 0; i < 15; i++) {
		out[i] = TWOELF_PAYLOAD_BASE + i;
	}

	publish_results(out, ARRAY_SIZE(out));

	return 0;
}
