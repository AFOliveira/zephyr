/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal fatal-error / halt handlers for Zephyr-in-U-mode.  We cannot
 * touch mcause/mtval, and the only "halt" primitive available to us is
 * SYSCALL_RETURN_FROM_KERNEL.
 */

#include <zephyr/kernel.h>
#include <zephyr/fatal.h>
#include <zephyr/toolchain.h>

#define SYSCALL_RETURN_FROM_KERNEL    8
#define KERNEL_RETURN_SYSTEM_ABORT    2

FUNC_NORETURN void arch_system_halt(unsigned int reason)
{
	ARG_UNUSED(reason);

	register uint64_t a0 __asm__("a0") = SYSCALL_RETURN_FROM_KERNEL;
	register uint64_t a1 __asm__("a1") = reason;
	register uint64_t a2 __asm__("a2") = KERNEL_RETURN_SYSTEM_ABORT;

	__asm__ volatile("ecall"
			 : "+r"(a0)
			 : "r"(a1), "r"(a2)
			 : "memory");

	/* If for any reason the firmware doesn't tear us down, hang. */
	for (;;) {
		__asm__ volatile("nop");
	}
	CODE_UNREACHABLE;
}

/* The generic kernel/fatal.c path calls z_fatal_error() which calls
 * arch_system_halt(); no other arch hooks are strictly required.  Some
 * drivers reference z_riscv_fatal_error() directly, so provide a thin
 * stub that just hands off to the generic path. */
void z_riscv_fatal_error(unsigned int reason, const void *esf)
{
	ARG_UNUSED(esf);
	arch_system_halt(reason);
}
