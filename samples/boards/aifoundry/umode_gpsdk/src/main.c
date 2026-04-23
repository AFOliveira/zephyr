/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Launch a gp-sdk-built compute kernel (e.g. test-compute-kernels/abort) as
 * the U-mode payload from Zephyr-M on Erbium.  The kernel binary is expected
 * to be loaded into MRAM at GPSDK_KERNEL_ADDR by the emulator (`-file_load
 * 0x40010000,<kernel.bin>`); we just mret into it.
 *
 * The kernel uses the cm-umode ABI:
 *   - Cache ops → SYSCALL_CACHE_OPS_* → Zephyr-M dispatcher services them.
 *   - Exit     → SYSCALL_RETURN_FROM_KERNEL → umode_fault.c longjmp's back.
 *
 *   arg1 in the RETURN_FROM_KERNEL ecall = status (0 = success)
 *   arg2                                = kernel return code (KERNEL_RETURN_*)
 *
 * Our launcher packages the full (status << 0) | (retcode << 8) into the
 * value handed back; `abort` sends status=0, retcode=KERNEL_RETURN_SELF_ABORT=1
 * so a successful abort run prints "retcode = 1 (self-abort)".
 */

#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "umode_abi.h"

/* Where the kernel binary is expected to live in MRAM.  Must match the
 * -file_load <addr>,<path> passed to erbium_emu. */
#ifndef GPSDK_KERNEL_ADDR
#define GPSDK_KERNEL_ADDR 0x40010000UL
#endif

/* 16 KiB U-mode stack for the gp-sdk kernel. */
static uint8_t umode_stack[16 * 1024] __aligned(16);

int main(void)
{
	printk("gp-sdk kernel launcher: entry=0x%lx\n",
	       (unsigned long)GPSDK_KERNEL_ADDR);

	erbium_mprot_allow_all();

	void *entry = (void *)GPSDK_KERNEL_ADDR;
	void *stack_top = &umode_stack[sizeof(umode_stack)];
	int64_t ret = aifoundry_launch_umode(entry, stack_top, 0);

	printk("kernel returned: %lld\n", (long long)ret);
	printk("=== done ===\n");

	return 0;
}
