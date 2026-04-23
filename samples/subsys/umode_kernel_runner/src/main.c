/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * gp-sdk kernel batch runner (M3).
 *
 * Compiles the kernel source file whose path is handed in via
 * -DGPSDK_KERNEL_SRC=<path> as a peer of main.c, then invokes its
 * entry_point() from Zephyr's main() and ecalls RETURN_FROM_KERNEL.
 *
 * The build also compiles a subset of cm-umode's .c sources (non-trace)
 * so that et_abort / et_printf-style helpers the kernels call resolve at
 * link time without needing the full cm-umode static library.
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>

#include <isa/common/syscall.h>

/*
 * Every test-compute-kernels kernel we target in the batch exposes an
 * entry_point.  Signature is one of:
 *    int64_t entry_point(void);
 *    int     entry_point(const Parameters*);
 * For the parameterless variants we declare-and-call directly.
 * Kernels that take a Parameters* are wrapped in a per-kernel shim
 * declared via GPSDK_KERNEL_HAS_PARAMS (not used in phase 1).
 */
extern int64_t entry_point(void);

int main(void)
{
	printk("SWEEP: begin kernel=" GPSDK_KERNEL_NAME "\n");

	int64_t rc = entry_point();

	printk("SWEEP: entry_point returned %lld\n", (long long)rc);
	printk("SWEEP: end\n");

	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);

	for (;;) {
		__asm__ __volatile__("wfi");
	}
	return 0;
}
