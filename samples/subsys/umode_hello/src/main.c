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

#define SYSCALL_RETURN_FROM_KERNEL    8
#define KERNEL_RETURN_SUCCESS         0

static inline void umode_return_from_kernel(int64_t status)
{
	register uint64_t a0 __asm__("a0") = SYSCALL_RETURN_FROM_KERNEL;
	register uint64_t a1 __asm__("a1") = (uint64_t)status;
	register uint64_t a2 __asm__("a2") = KERNEL_RETURN_SUCCESS;

	__asm__ volatile("ecall"
			 : "+r"(a0)
			 : "r"(a1), "r"(a2)
			 : "memory");
}

int main(void)
{
	printk("Hello from Zephyr in U-mode\n");
	umode_return_from_kernel(0);

	/* Should not reach here; MasterMinion tears down the kernel. */
	for (;;) {
	}
	return 0;
}
