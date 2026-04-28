/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode idle: never issue WFI (that would require M-mode on many
 * RISC-V implementations and in any case stops the hart in the ET-SoC1
 * emulator).  Just pause/nop.
 */

#include <zephyr/kernel.h>

#ifndef CONFIG_ARCH_HAS_CUSTOM_CPU_IDLE
void arch_cpu_idle(void)
{
	/* Zihintpause is recognized as a hint and decodes as a nop on
	 * older CPUs; fall back to a plain nop when unavailable. */
	__asm__ volatile("nop" ::: "memory");
}
#endif

#ifndef CONFIG_ARCH_HAS_CUSTOM_CPU_ATOMIC_IDLE
void arch_cpu_atomic_idle(unsigned int key)
{
	ARG_UNUSED(key);
	__asm__ volatile("nop" ::: "memory");
}
#endif
