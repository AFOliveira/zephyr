/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Phase 1 sanity check for Zephyr running as a cm-umode kernel on
 * ET-SoC1:  emit a greeting via printk and return to the host firmware
 * via SYSCALL_RETURN_FROM_KERNEL.
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>
#include <isa/common/syscall.h>   /* cm-umode ABI */

int main(void)
{
	printk("Hello from Zephyr in U-mode\n");
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);

	/* Should not reach here; MasterMinion tears down the kernel. */
	for (;;) {
	}
	return 0;
}
