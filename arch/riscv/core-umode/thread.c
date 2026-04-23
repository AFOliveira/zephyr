/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thread-arch for Zephyr-in-U-mode.
 *
 * Two modes:
 *
 *   CONFIG_MULTITHREADING=n: the generic kernel init calls bg_thread_main()
 *   (and thereby main()) directly; we only need a handful of stubs.
 *
 *   CONFIG_MULTITHREADING=y: we piggy-back on the upstream z_riscv_switch
 *   assembly (arch/riscv/core/switch.S) which is purely register save /
 *   restore and touches no CSRs -- safe from U-mode.  We only need to
 *   provide a U-mode-safe arch_new_thread that sets up the initial
 *   callee_saved frame.
 */

#include <zephyr/kernel.h>
#include <kernel_internal.h>
#include <zephyr/arch/riscv/csr.h>

int arch_coprocessors_disable(struct k_thread *thread)
{
	ARG_UNUSED(thread);
	return -ENOTSUP;
}

#ifdef CONFIG_MULTITHREADING

/* U-mode thread-start trampoline.
 *
 * arch_new_thread stores the user entry (z_thread_entry) and its args
 * in the thread's callee_saved slots (ra + s0..s3).  When z_riscv_switch
 * restores a new thread for the first time, it ret's to ra, which
 * lands here: we pull the saved entry/args out of s0..s3 and jump to
 * z_thread_entry.
 */
void z_umode_thread_start(void);

__asm__ (
	".globl z_umode_thread_start\n"
	"z_umode_thread_start:\n"
	"    mv a0, s0\n"   /* entry */
	"    mv a1, s1\n"   /* p1 */
	"    mv a2, s2\n"   /* p2 */
	"    mv a3, s3\n"   /* p3 */
	"    call z_thread_entry\n"
	/* z_thread_entry is FUNC_NORETURN; if it returns just loop. */
	"1:  j 1b\n"
);

void arch_new_thread(struct k_thread *thread, k_thread_stack_t *stack,
		     char *stack_ptr, k_thread_entry_t entry,
		     void *p1, void *p2, void *p3)
{
	ARG_UNUSED(stack);

	/* We don't use arch_esf in U-mode (no trap frame); just park the
	 * entry + args in callee-saved slots and have ra point at the
	 * trampoline. */
	thread->callee_saved.sp = (unsigned long)stack_ptr;
	thread->callee_saved.ra = (unsigned long)z_umode_thread_start;
	thread->callee_saved.s0 = (unsigned long)entry;
	thread->callee_saved.s1 = (unsigned long)p1;
	thread->callee_saved.s2 = (unsigned long)p2;
	thread->callee_saved.s3 = (unsigned long)p3;

	thread->switch_handle = thread;
}

#endif /* CONFIG_MULTITHREADING */
