/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tiny U-mode "compute kernel".
 *
 * Enters with sp = top of umode_stack[] (set by aifoundry_launch_umode)
 * and a0 = caller-supplied arg.
 *
 * Performs two syscalls:
 *   - SYSCALL_CACHE_OPS_INVALIDATE (id=5) with a1=1
 *   - SYSCALL_RETURN_FROM_KERNEL   (id=8) with a1=0  (success)
 *
 * The RETURN_FROM_KERNEL ecall never resumes here.  If the M-mode side for
 * some reason does mret back instead of longjmp'ing, we spin in a tight
 * infinite loop to keep the emulator deterministic.
 */

	.section .text, "ax", @progbits
	.balign 4
	.globl _umode_entry
	.type _umode_entry, @function
_umode_entry:
	/* syscall(SYSCALL_CACHE_OPS_INVALIDATE, 1, 0, 0) */
	li   a0, 5        /* SYSCALL_CACHE_OPS_INVALIDATE */
	li   a1, 1
	li   a2, 0
	li   a3, 0
	ecall

	/* syscall(SYSCALL_RETURN_FROM_KERNEL, 0, 0, 0) */
	li   a0, 8        /* SYSCALL_RETURN_FROM_KERNEL */
	li   a1, 0        /* KERNEL_RETURN_SUCCESS */
	li   a2, 0
	li   a3, 0
	ecall

1:	/* Should not be reached. */
	j    1b

	.size _umode_entry, . - _umode_entry
