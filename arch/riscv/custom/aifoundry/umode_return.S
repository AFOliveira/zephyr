/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Long-jump back to the original caller of aifoundry_launch_umode().
 *
 *   void aifoundry_resume_from_kernel_asm(int64_t retcode)
 *
 * This is invoked from the M-mode fault path (umode_fault.c) when U-mode
 * executes `ecall` with a0 = SYSCALL_RETURN_FROM_KERNEL.  The dispatcher has
 * already populated g_umode_launch_ctx.retcode (slot 18) with the code from
 * U-mode's a1; we pass that same value in a0 here and use it as the return
 * value of aifoundry_launch_umode().
 *
 * Because we unwind directly from the trap frame (rather than mret-ing back
 * to U-mode), we:
 *   - restore the saved M-mode sp (abandoning the Zephyr ISR/trap stack
 *     frame -- safe because we are on a single hart and the trap was fully
 *     contained in aifoundry_launch_umode's dynamic scope),
 *   - restore callee-saved registers and ra,
 *   - restore mstatus / mie,
 *   - `ret` to the launcher's caller with a0 holding the kernel retcode.
 */

#include <zephyr/toolchain.h>

	.section .text, "ax", @progbits
	.balign 4
	.globl aifoundry_resume_from_kernel_asm
	.type aifoundry_resume_from_kernel_asm, @function
aifoundry_resume_from_kernel_asm:
	/* a0 already holds the retcode we want to return to our caller. */
	la   t0, g_umode_launch_ctx

	/* Restore mstatus / mie first (so subsequent instructions see the
	 * launcher's interrupt state). */
	ld   t1, 16 * 8(t0)
	csrw mstatus, t1
	ld   t1, 17 * 8(t0)
	csrw mie, t1

	/* Restore callee-saved GPRs. */
	ld   ra,   0 * 8(t0)
	ld   sp,   1 * 8(t0)
	ld   gp,   2 * 8(t0)
	ld   tp,   3 * 8(t0)
	ld   s0,   4 * 8(t0)
	ld   s1,   5 * 8(t0)
	ld   s2,   6 * 8(t0)
	ld   s3,   7 * 8(t0)
	ld   s4,   8 * 8(t0)
	ld   s5,   9 * 8(t0)
	ld   s6,  10 * 8(t0)
	ld   s7,  11 * 8(t0)
	ld   s8,  12 * 8(t0)
	ld   s9,  13 * 8(t0)
	ld   s10, 14 * 8(t0)
	ld   s11, 15 * 8(t0)

	/* a0 still holds the retcode. */
	ret

	.size aifoundry_resume_from_kernel_asm, . - aifoundry_resume_from_kernel_asm
