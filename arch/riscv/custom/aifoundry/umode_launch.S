/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * M-mode -> U-mode launcher for Erbium Minion.
 *
 *   int64_t aifoundry_launch_umode(void *entry, void *stack, uint64_t arg)
 *
 *   a0 = U-mode entry point (MRAM virtual/physical address)
 *   a1 = top of U-mode stack
 *   a2 = argument forwarded to U-mode in a0
 *
 * Behaviour:
 *   1. Save the M-mode caller's context (ra, sp, callee-saved, mstatus) into
 *      the static struct g_umode_launch_ctx so that resume_from_kernel can
 *      long-jump back to the caller.
 *   2. Set mstatus.MPP = U-mode (00), MPIE cleared (U-mode starts with
 *      interrupts disabled for this slice).
 *   3. Set mepc = entry, sp = stack, a0 = arg; mret.
 *
 * Single-hart / single-minion: only one launcher can be active.  The context
 * block is therefore a file-static (.bss) struct with no locking.
 */

#include <zephyr/toolchain.h>

/*
 * Context block layout (doublewords):
 *   [ 0] ra
 *   [ 1] sp
 *   [ 2] gp
 *   [ 3] tp
 *   [ 4] s0
 *   [ 5] s1
 *   [ 6] s2
 *   [ 7] s3
 *   [ 8] s4
 *   [ 9] s5
 *   [10] s6
 *   [11] s7
 *   [12] s8
 *   [13] s9
 *   [14] s10
 *   [15] s11
 *   [16] mstatus
 *   [17] mie
 *   [18] retcode    (populated by the dispatcher on RETURN_FROM_KERNEL)
 */
	.section .bss, "aw", @nobits
	.balign 16
	.globl g_umode_launch_ctx
g_umode_launch_ctx:
	.zero 19 * 8

/* MSTATUS helpers */
#define MSTATUS_MPP_MASK   (3 << 11)
#define MSTATUS_MPP_U      (0 << 11)   /* User mode */
#define MSTATUS_MPIE       (1 << 7)
#define MSTATUS_MIE        (1 << 3)

	.section .text, "ax", @progbits
	.balign 4
	.globl aifoundry_launch_umode
	.type aifoundry_launch_umode, @function
aifoundry_launch_umode:
	/* Save caller context to g_umode_launch_ctx */
	la   t0, g_umode_launch_ctx
	sd   ra,   0 * 8(t0)
	sd   sp,   1 * 8(t0)
	sd   gp,   2 * 8(t0)
	sd   tp,   3 * 8(t0)
	sd   s0,   4 * 8(t0)
	sd   s1,   5 * 8(t0)
	sd   s2,   6 * 8(t0)
	sd   s3,   7 * 8(t0)
	sd   s4,   8 * 8(t0)
	sd   s5,   9 * 8(t0)
	sd   s6,  10 * 8(t0)
	sd   s7,  11 * 8(t0)
	sd   s8,  12 * 8(t0)
	sd   s9,  13 * 8(t0)
	sd   s10, 14 * 8(t0)
	sd   s11, 15 * 8(t0)

	csrr t1, mstatus
	sd   t1,  16 * 8(t0)
	csrr t1, mie
	sd   t1,  17 * 8(t0)

	/* Zero the retcode slot (will be overwritten by dispatcher). */
	sd   zero, 18 * 8(t0)

	/*
	 * Build the mstatus value for the mret into U-mode:
	 *   MPP   = 00    (U-mode)
	 *   MPIE  = 0     (U-mode starts with interrupts disabled)
	 *   MIE   = 0     (M-mode interrupts are *currently* disabled; will
	 *                  be re-enabled when MPIE latches back on trap entry)
	 */
	csrr t1, mstatus
	li   t2, MSTATUS_MPP_MASK
	not  t2, t2
	and  t1, t1, t2      /* clear MPP */
	li   t2, MSTATUS_MPIE
	not  t2, t2
	and  t1, t1, t2      /* clear MPIE */
	li   t2, MSTATUS_MIE
	not  t2, t2
	and  t1, t1, t2      /* clear MIE for the upcoming mret */
	csrw mstatus, t1

	/*
	 * Program mret landing site:
	 *   mepc = U-mode entry
	 *   sp   = U-mode stack
	 *   a0   = arg
	 */
	csrw mepc, a0
	mv   sp, a1
	mv   a0, a2

	/* Make sure no stray registers leak into U-mode. */
	mv   ra, zero
	mv   gp, zero
	mv   tp, zero
	mv   t0, zero
	mv   t1, zero
	mv   t2, zero
	mv   t3, zero
	mv   t4, zero
	mv   t5, zero
	mv   t6, zero
	mv   a1, zero
	mv   a2, zero
	mv   a3, zero
	mv   a4, zero
	mv   a5, zero
	mv   a6, zero
	mv   a7, zero
	mv   s0, zero
	mv   s1, zero
	mv   s2, zero
	mv   s3, zero
	mv   s4, zero
	mv   s5, zero
	mv   s6, zero
	mv   s7, zero
	mv   s8, zero
	mv   s9, zero
	mv   s10, zero
	mv   s11, zero

	mret

	.size aifoundry_launch_umode, . - aifoundry_launch_umode
