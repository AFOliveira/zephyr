/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode "reboot" = SYSCALL_RETURN_FROM_KERNEL.  There is no way for a
 * compute kernel to actually reset the SoC by itself; just return to the
 * host firmware.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#define SYSCALL_RETURN_FROM_KERNEL    8
#define KERNEL_RETURN_SUCCESS         0

void sys_arch_reboot(int type)
{
	ARG_UNUSED(type);

	register uint64_t a0 __asm__("a0") = SYSCALL_RETURN_FROM_KERNEL;
	register uint64_t a1 __asm__("a1") = 0;
	register uint64_t a2 __asm__("a2") = KERNEL_RETURN_SUCCESS;

	__asm__ volatile("ecall"
			 : "+r"(a0)
			 : "r"(a1), "r"(a2)
			 : "memory");

	for (;;) {
	}
}
