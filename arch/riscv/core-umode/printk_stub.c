/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bootstrap printk backend for Zephyr-in-U-mode.
 *
 * cm-umode kernels run without a UART driver (the console hardware is
 * owned by the host M-mode firmware).  As a phase-1 stopgap we drop
 * each character into a small ring in RAM at a fixed, linker-visible
 * symbol.  The region can be dumped by an external tool (or by the
 * simulator / gp-sdk launcher) after the kernel returns.  A later
 * phase will switch this to the et-trace buffer.
 */

#include <stdint.h>
#include <stddef.h>
#include <zephyr/toolchain.h>
#include <isa/common/syscall.h>

#ifndef CONFIG_UMODE_PRINTK_BUFFER_SIZE
#define CONFIG_UMODE_PRINTK_BUFFER_SIZE 2048
#endif

/* Placed in .noinit so it survives BSS zeroing and is easy to locate in
 * the memory map.  The first two words are { write-index, length }. */
struct umode_printk_ring {
	uint32_t write_index;
	uint32_t length;
	uint8_t  data[CONFIG_UMODE_PRINTK_BUFFER_SIZE];
};

__attribute__((section(".noinit"), used))
struct umode_printk_ring __umode_printk_ring;

int arch_printk_char_out(int c)
{
	uint32_t idx = __umode_printk_ring.write_index;

	__umode_printk_ring.data[idx] = (uint8_t)c;
	idx = (idx + 1) % CONFIG_UMODE_PRINTK_BUFFER_SIZE;
	__umode_printk_ring.write_index = idx;
	if (__umode_printk_ring.length < CONFIG_UMODE_PRINTK_BUFFER_SIZE) {
		__umode_printk_ring.length++;
	}

	/* Also echo to the host UART via the CONSOLE_WRITE OSKERN syscall.
	 * The MM handler busy-waits on THR-empty and streams one byte at a
	 * time; slow but reliable for the MVP.  This path is only live
	 * when MULTITHREADING=y (the OSKERN ABI is wired up then). */
#ifdef CONFIG_MULTITHREADING
	uint8_t byte = (uint8_t)c;
	(void)syscall(SYSCALL_OSKERN_CONSOLE_WRITE,
		      (uint64_t)(uintptr_t)&byte, 1U, 0U);
#endif
	return c;
}
