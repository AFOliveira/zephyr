/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal M->U->M round-trip demo for the Erbium Minion.
 *
 *   1. Zephyr boots in M-mode and prints "launching kernel".
 *   2. aifoundry_launch_umode() mret's into _umode_entry in MRAM.
 *   3. The U-mode blob issues two syscalls:
 *        a0 = SYSCALL_CACHE_OPS_INVALIDATE, a1 = 1
 *        a0 = SYSCALL_RETURN_FROM_KERNEL,   a1 = 0
 *   4. The M-mode dispatcher services the first via CSR 0x7d0 and
 *      long-jumps back out of the second, restoring the launcher's context.
 *   5. main() prints "kernel returned: <code>" and "=== done ===".
 */

#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "umode_abi.h"

/* Symbol defined by src/umode_kernel.S. */
extern void _umode_entry(uint64_t arg);

/* 4 KiB U-mode stack.  Aligned for RISC-V ABI (sp = 16-byte aligned). */
static uint8_t umode_stack[4096] __aligned(16);

int main(void)
{
	printk("launching kernel\n");

	/* MPROT is already in its reset-all-zero state (mprot_en = 0, i.e.
	 * enforcement disabled) on emulator boot, but do the write anyway so
	 * the behaviour is deterministic even if a future boot stage pokes it.
	 */
	erbium_mprot_allow_all();

	void *stack_top = &umode_stack[sizeof(umode_stack)];
	int64_t ret = aifoundry_launch_umode((void *)_umode_entry, stack_top, 0);

	printk("kernel returned: %lld\n", (long long)ret);
	printk("=== done ===\n");

	return 0;
}
