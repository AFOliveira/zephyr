/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * M-mode (Erbium) implementation of the portable AIFoundry kernel
 * runtime declared in <zephyr/aifoundry/runtime.h>.
 *
 * Logging routes through Zephyr's printk → console driver → Shakti
 * UART.  No host param-passing is involved (Erbium boots Zephyr-M
 * directly), so kernel_args is always NULL and publish_results is a
 * silent no-op.  Kernel exit just halts the CPU; if Zephyr's idle
 * thread is configured it will run, otherwise the for-loop spins
 * forever (the kernel never schedules anything else once main has
 * been told to exit).
 */

#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

#include <zephyr/aifoundry/runtime.h>
#include <zephyr/sys/printk.h>
#include <zephyr/kernel.h>

void aifoundry_log(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vprintk(fmt, ap);
	va_end(ap);
}

void *aifoundry_kernel_args(void)
{
	return NULL;
}

void aifoundry_publish_results(const void *src, size_t size)
{
	(void)src;
	(void)size;
}

void aifoundry_kernel_exit(int rc)
{
	(void)rc;
	/* Give any IRQ-driven console TX a chance to drain to the UART
	 * hardware before we hard-disable MIE.  Zephyr's default console
	 * driver mode is interrupt-driven; aifoundry_log calls go into a
	 * small ring buffer that the UART IRQ flushes asynchronously.  If
	 * we cleared MIE immediately the trailing log lines (including
	 * the final "done, mismatches=N") would sit in that buffer
	 * forever.  A short busy-wait costs ~ms in the emulator and is
	 * enough to drain ~hundreds of bytes at any reasonable baud. */
	for (volatile int i = 0; i < 1000000; i++) {
		__asm__ volatile("nop");
	}
	/* Disable M-mode interrupts and spin.  k_cpu_idle() would
	 * re-enable interrupts, which fires the timer ISR — without
	 * MULTITHREADING the dispatcher can't service it and Zephyr
	 * lands in z_riscv_fatal_error → arch_system_halt.  Plain spin
	 * keeps the final PC inside this function so the test
	 * harness reads it as a clean termination. */
	__asm__ volatile("csrci mstatus, 0x8");   /* clear MIE */
	for (;;) {
		__asm__ volatile("nop");
	}
}
