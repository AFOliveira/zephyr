/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Minimal RISC-V U-mode entry point ("crt0") for Zephyr running as a
 * compute kernel under an external M/S-mode firmware (e.g. ET-SoC1
 * MasterMinion).
 *
 * Modeled on et-platform/gp-sdk/device/sdk/lib/crt0.S.  We never touch
 * M-mode-only CSRs (mstatus, mtvec, mie, ...).  On entry we merely set
 * up the global pointer and the initial stack, zero .bss, then call
 * z_prep_c.  On return we issue an ecall to ask the host firmware to
 * tear down this kernel (SYSCALL_RETURN_FROM_KERNEL == 8).
 */

#include <zephyr/toolchain.h>
#include <zephyr/linker/sections.h>
#include <isa/common/syscall.h>   /* cm-umode ABI: SYSCALL_RETURN_FROM_KERNEL, KERNEL_RETURN_* */

GTEXT(_start)
GTEXT(z_prep_c)
GDATA(z_main_stack)

/* The Zephyr main stack is declared as a K_THREAD_STACK.  Its size is
 * CONFIG_MAIN_STACK_SIZE plus any reserved guard region; we use the
 * compile-time constant from Kconfig here and let the linker sort it
 * out via the stack buffer macros.  For a crt0-style entry we just
 * point sp at the end of the buffer. */

.section .text.init, "ax", @progbits
.globl _start
.type  _start, @function

_start:
	/* When the host (e.g. ET-SoC1 MasterMinion + WorkerMinion DM-API
	 * dispatch) launches a kernel across an entire shire, every minion
	 * srets to this _start in parallel.  Zephyr-U is single-hart only
	 * (no SMP support in the U-mode arch), so non-zero harts must not
	 * touch shared data — they would race with hart-0 on .bss, the
	 * printk ring, scheduler state, etc.  Filter here: any hart with
	 * non-zero ET-Minion `hartid` (custom CSR 0xcd0) immediately ecalls
	 * RETURN_FROM_KERNEL and parks.
	 *
	 * On hosts that launch only a single hart (M+U direct, S-trampoline)
	 * `hartid` is 0 and the check is a single branch — no harm. */
	csrr  t0, 0xcd0          /* ET-Minion hartid */
	beqz  t0, 2f
	li    a0, SYSCALL_RETURN_FROM_KERNEL
	li    a1, 0
	li    a2, KERNEL_RETURN_SUCCESS
	ecall
1:	j 1b
2:
	/* Initialize global pointer (gp) if the linker defined one; under
	 * CONFIG_RISCV_GP=n the symbol is absent and the compiler emits
	 * plain absolute addressing, so we skip this entirely. */
#ifdef CONFIG_RISCV_GP
.option push
.option norelax
	la   gp, __global_pointer$
.option pop
#endif

	/* Point sp at the top of the main stack.  K_THREAD_STACK_BUFFER
	 * is a macro, but the stack symbol itself is laid out as a plain
	 * array by the linker.  We add CONFIG_MAIN_STACK_SIZE at runtime. */
	la   sp, z_main_stack
	li   t0, CONFIG_MAIN_STACK_SIZE
	add  sp, sp, t0
	andi sp, sp, -16          /* RISC-V psABI: 16-byte aligned sp */

	/* Hand off to the C runtime.  z_prep_c zeros .bss, runs the
	 * SYS_INIT levels, and ultimately calls main().  In the
	 * MULTITHREADING=n / RISCV_RUN_IN_UMODE configuration, main()
	 * returns normally; we fall through here and exit. */
	call z_prep_c

	/* z_prep_c is FUNC_NORETURN in the upstream source.  If we ever
	 * do get here, exit cleanly via SYSCALL_RETURN_FROM_KERNEL. */
	mv   a1, a0
	li   a2, KERNEL_RETURN_SUCCESS
	li   a0, SYSCALL_RETURN_FROM_KERNEL
	ecall

1:	j 1b

.size _start, . - _start
