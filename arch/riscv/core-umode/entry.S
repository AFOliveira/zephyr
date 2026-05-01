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
#include <erbium/isa/syscall.h>   /* cm-umode ABI: SYSCALL_RETURN_FROM_KERNEL, KERNEL_RETURN_* */

GTEXT(_boot)
GTEXT(_start)
GTEXT(z_prep_c)
GDATA(__stack_base)
GDATA(cm_umode_kernel_args)
GDATA(cm_umode_kernel_env)

.section .text.init, "ax", @progbits
.globl _boot
.type  _boot, @function
.globl _start
.type  _start, @function

_boot:
_start:
	/* When the host (e.g. ET-SoC1 MasterMinion + WorkerMinion DM-API
	 * dispatch) launches a kernel across an entire shire, every minion
	 * srets to this _start in parallel.  Zephyr-U is single-hart only
	 * (no SMP support in the U-mode arch), so non-zero harts must not
	 * touch shared data — they would race with the selected Zephyr hart on
	 * .bss, the printk ring, scheduler state, etc.  Filter here: every
	 * non-selected ET-Minion `hartid` (custom CSR 0xcd0) immediately
	 * ecalls RETURN_FROM_KERNEL and parks.
	 *
	 * For the four-neighborhood ET-SoC1 flow, the launched hart for shires
	 * 0..3 appears at local-in-shire hart 0, 16, 32, 48 respectively.  Use
	 * the shire id encoded above bit 6 to choose that local hart; otherwise
	 * only shire 0 runs Zephyr C code and the other shires return before
	 * publishing results.
	 *
	 * On hosts that launch only a single hart (M+U direct, S-trampoline)
	 * `hartid` is 0 and the check is a single branch — no harm. */
	csrr  t0, 0xcd0          /* ET-Minion hartid */
	srli  t1, t0, 6          /* shire id */
	andi  t1, t1, 0x3        /* neighborhood id in this 4-shire flow */
	slli  t1, t1, 4          /* selected local hart: 0, 16, 32, 48 */
	andi  t0, t0, 0x3F       /* local-in-shire hart id */
	beq   t0, t1, 2f
	li    a0, SYSCALL_RETURN_FROM_KERNEL
	li    a1, 0
	li    a2, KERNEL_RETURN_SUCCESS
	ecall
1:	j 1b
2:
	/* Preserve the firmware launch ABI for code that needs to publish
	 * results back to the host. */
	la    t1, cm_umode_kernel_args
	sd    a0, 0(t1)
	la    t1, cm_umode_kernel_env
	sd    a1, 0(t1)

	/* Ask the host firmware to invalidate this hart's I-cache and TLBs.
	 * The runtime DMA's the kernel ELF into DRAM, but the minion's I-cache
	 * can still hold lines from a previous kernel on this hart. `fence.i`
	 * is not implemented on ET-Minion, so route this through the cm-umode
	 * syscall ABI. csr_enc=3 means TLB/PTW invalidate plus I-cache
	 * invalidate. */
	li    a0, SYSCALL_CACHE_OPS_INVALIDATE
	li    a1, 3
	li    a2, 0
	li    a3, 0
	ecall

	/* The host firmware is not required to zero argument registers before
	 * entering U-mode; keep Zephyr's C path from observing stale launch
	 * metadata as conventional boot arguments. */
	li    a0, 0
	li    a1, 0
	li    a2, 0
	li    a3, 0

	/* Initialize global pointer (gp) if the linker defined one; under
	 * CONFIG_RISCV_GP=n the symbol is absent and the compiler emits
	 * plain absolute addressing, so we skip this entirely. */
#ifdef CONFIG_RISCV_GP
.option push
.option norelax
	la   gp, __global_pointer$
.option pop
#endif

	/* Point sp at the linker-managed soc1sim stack region. */
	la   sp, __stack_base
	andi sp, sp, -16          /* RISC-V psABI: 16-byte aligned sp */

	/* Hand off to the C runtime.  z_prep_c zeros .bss and enters Zephyr's
	 * common init path.  For MULTITHREADING=n, the U-mode arch hook calls
	 * main() and exits through RETURN_FROM_KERNEL. */
	call z_prep_c

	/* z_prep_c is FUNC_NORETURN.  If we ever do get here, exit cleanly
	 * via SYSCALL_RETURN_FROM_KERNEL. */
	mv   a1, a0
	li   a2, KERNEL_RETURN_SUCCESS
	li   a0, SYSCALL_RETURN_FROM_KERNEL
	ecall

1:	j 1b

.size _start, . - _start
.size _boot, . - _boot
