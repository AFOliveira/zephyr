/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Syscall-sweep M->U->M demo for the Erbium Minion.
 *
 *   1. Zephyr boots in M-mode, prints the banner.
 *   2. aifoundry_launch_umode() mret's into _umode_entry in MRAM.
 *   3. The U-mode blob ecalls every cm-umode U-mode syscall (ids 1..11 minus
 *      the one used to exit, id=8) and stores each return value in the
 *      shared syscall_results[] array.
 *   4. Blob ecalls SYSCALL_RETURN_FROM_KERNEL to unwind to the launcher.
 *   5. Zephyr main() prints the per-syscall return code so the operator can
 *      see which were serviced (ret=0) vs stubbed (ret=-1 / SYSCALL_INVALID_ID).
 */

#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "umode_abi.h"

extern void _umode_entry(uint64_t arg);
extern int64_t syscall_results[10];

static const char *const syscall_labels[10] = {
	"id  1 EVICT_SW           ",
	"id  2 FLUSH_SW           ",
	"id  3 LOCK_SW            ",
	"id  4 UNLOCK_SW          ",
	"id  5 INVALIDATE         ",
	"id  6 EVICT_L1           ",
	"id  7 SHIRE_CACHE_BANK_OP",
	"id  9 PMC_SC_SAMPLE      ",
	"id 10 PMC_MS_SAMPLE      ",
	"id 11 EVICT_WHOLE_L1_L2  ",
};

/* 4 KiB U-mode stack. */
static uint8_t umode_stack[4096] __aligned(16);

int main(void)
{
	printk("syscall sweep: launching U-mode kernel\n");

	erbium_mprot_allow_all();

	void *stack_top = &umode_stack[sizeof(umode_stack)];
	int64_t ret = aifoundry_launch_umode((void *)_umode_entry, stack_top, 0);

	printk("kernel returned: %lld\n", (long long)ret);
	printk("--- per-syscall results ---\n");
	for (int i = 0; i < 10; i++) {
		printk("  %s -> %lld\n", syscall_labels[i],
		       (long long)syscall_results[i]);
	}
	printk("=== done ===\n");

	return 0;
}
