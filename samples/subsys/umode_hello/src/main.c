/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * "gp-sdk kernel on top of Zephyr-U" demo.
 *
 * Zephyr itself runs in U-mode on ET-SoC1 as a cm-umode kernel, launched by
 * MachineMinion (patched to service U-mode syscalls at M directly).  We
 * reproduce the behaviour of test-compute-kernels/src/abort/abort.c inside
 * Zephyr's main() to demonstrate that a gp-sdk-flavored kernel's logic can
 * coexist with the Zephyr runtime in the same U-mode binary.
 *
 * The original abort kernel:
 *
 *   int64_t entry_point(void) {
 *       int temp = 0;
 *       temp++;
 *       et_abort();            // syscall(SYSCALL_RETURN_FROM_KERNEL,
 *                              //         0, KERNEL_RETURN_SELF_ABORT, 0)
 *       return 0;
 *   }
 *
 * Direct translation below, minus the et-common-libs libc dependency.
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>

#include <isa/common/syscall.h>

static __attribute__((noreturn)) void et_abort_inline(void)
{
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SELF_ABORT, 0);
	for (;;) {
		__asm__ __volatile__("wfi");
	}
}

int main(void)
{
	printk("gp-sdk abort-kernel payload running inside Zephyr-U\n");

	/* abort.c body, translated to cm-umode ABI: */
	int temp = 0;

	temp++;

	printk("pre-abort: temp = %d\n", temp);

	et_abort_inline();

	/* Not reached — MachineMinion halts us on SELF_ABORT. */
	return 0;
}
