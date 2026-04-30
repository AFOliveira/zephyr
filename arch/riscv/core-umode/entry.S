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
#include <erbium-soc1sim/isa/syscall.h>   /* cm-umode ABI: SYSCALL_RETURN_FROM_KERNEL, KERNEL_RETURN_* */

GTEXT(_start)
GTEXT(z_prep_c)
GDATA(z_main_stack)
GDATA(cm_umode_kernel_args)
GDATA(cm_umode_kernel_env)

/* The Zephyr main stack is declared as a K_THREAD_STACK.  Its size is
 * CONFIG_MAIN_STACK_SIZE plus any reserved guard region; we use the
 * compile-time constant from Kconfig here and let the linker sort it
 * out via the stack buffer macros.  For a crt0-style entry we just
 * point sp at the end of the buffer. */

.section .text.init, "ax", @progbits
.globl _start
.type  _start, @function

_start:
	/* Hartid filter: only hart-0 enters the boot path; everyone else
	 * RETURN_FROM_KERNELs immediately.  The host launches the kernel
	 * across all minions of a shire; without this filter they race
	 * on .bss / printk ring / scheduler state.  Mask to local-in-shire
	 * (64 harts/shire = 6 bits) so hart 0 of every shire passes the
	 * filter — required for multi-shire launches like the parallel
	 * dual_shire_launcher experiment. */
	csrr  t0, 0xcd0
	andi  t0, t0, 0x3F
	beqz  t0, 2f
	li    a0, SYSCALL_RETURN_FROM_KERNEL
	li    a1, 0
	li    a2, KERNEL_RETURN_SUCCESS
	ecall
1:	j 1b
2:
	/* cm-umode ABI: host firmware sret's with
	 *     a0 = void *                  (launch args / kernel_args ptr)
	 *     a1 = kernel_environment_t *  (shire_mask + freq, see <system/abi.h>)
	 * Stash both into the cm_umode_kernel_{args,env} globals defined in
	 * kernel_runtime_storage.c so the SoC runtime.c can read them.  Must
	 * happen before we scrub a0/a1. */
	la    t1, cm_umode_kernel_args
	sd    a0, 0(t1)
	la    t1, cm_umode_kernel_env
	sd    a1, 0(t1)

	/* Ask the M-mode firmware to invalidate this hart's I-cache and
	 * TLBs.  The runtime DMA's the kernel ELF into DRAM but the
	 * minion's I-cache may still hold lines from a previous kernel
	 * that ran on this hart; without this invalidate, calls into
	 * functions that the I-cache hasn't yet missed-on read stale
	 * bytes from the cached line and trap as illegal instructions.
	 * RISC-V `fence.i` is unimplemented on ET-Minion (verified
	 * empirically: traps with cause 0x1e), so we route through
	 * SYSCALL_CACHE_OPS_INVALIDATE (id=5).  csr_enc=3 means
	 * inval_TLBs_and_PTW=1 (bit 0) | inval_instr_cache=1 (bit 1). */
	li    a0, SYSCALL_CACHE_OPS_INVALIDATE
	li    a1, 3
	li    a2, 0
	li    a3, 0
	ecall
	/* Scrub the RISC-V argument registers.  The host firmware
	 * (M-mode + S-mode dispatch) is not required to zero a0-a3
	 * before sret-ing into U-mode; on real silicon they arrive
	 * with whatever the WorkerMinion DM-API code happened to
	 * leave behind, and any later C code that conventionally
	 * expects a0=hartid / a1=DTB pointer would dereference
	 * garbage.  Sysemu doesn't catch this because its memory
	 * model is permissive — silicon is not. */
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

	/* Point sp at the top of the main stack.  K_THREAD_STACK_BUFFER
	 * is a macro, but the stack symbol itself is laid out as a plain
	 * array by the linker.  We add CONFIG_MAIN_STACK_SIZE at runtime.
	 *
	 * When CONFIG_AIFOUNDRY_PER_SHIRE_STACK=y the single z_main_stack
	 * is carved into one slice per shire so a single multi-shire
	 * kernelLaunch can have lead harts on different shires running
	 * non-trivial workloads without racing on the stack:
	 *
	 *     shire_id = (CSR 0xcd0 >> 6) & 0x1F
	 *     sp       = &z_main_stack[(shire_id + 1) *
	 *                              CONFIG_AIFOUNDRY_PER_SHIRE_STACK_SIZE]
	 *
	 * Each shire's slice is [shire_id*SIZE .. (shire_id+1)*SIZE), and
	 * sp points one past the end of that slice (stacks grow down).  The
	 * existing single-shire path (Kconfig=n, default) is byte-identical
	 * to the pre-patch behavior. */
#ifdef CONFIG_AIFOUNDRY_PER_SHIRE_STACK
	csrr t1, 0xcd0
	srli t1, t1, 6
	andi t1, t1, 0x1F           /* t1 = shire_id */
	addi t1, t1, 1              /* t1 = shire_id + 1 */
	li   t2, CONFIG_AIFOUNDRY_PER_SHIRE_STACK_SIZE
	mul  t1, t1, t2             /* t1 = (shire_id+1) * SIZE */
	la   sp, z_main_stack
	add  sp, sp, t1
#else
	la   sp, z_main_stack
	li   t0, CONFIG_MAIN_STACK_SIZE
	add  sp, sp, t0
#endif
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
