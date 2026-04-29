/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thread-arch support for Zephyr-in-U-mode.
 */

#include <zephyr/kernel.h>
#include <zephyr/toolchain.h>
#include <erbium/isa/syscall.h>

int arch_coprocessors_disable(struct k_thread *thread)
{
	ARG_UNUSED(thread);
	return -ENOTSUP;
}

#if !defined(CONFIG_MULTITHREADING)
FUNC_NORETURN void z_riscv_umode_switch_to_main_no_multithreading(
	k_thread_entry_t main_func, void *p1, void *p2, void *p3)
{
	main_func(p1, p2, p3);

	register uint64_t a0 asm("a0") = SYSCALL_RETURN_FROM_KERNEL;
	register uint64_t a1 asm("a1") = 0;
	register uint64_t a2 asm("a2") = KERNEL_RETURN_SUCCESS;
	register uint64_t a3 asm("a3") = 0;

	__asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a3) : "memory");

	for (;;) {
	}
}
#endif
