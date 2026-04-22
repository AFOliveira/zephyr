/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hook into Zephyr's RISC-V fault path so that an `ecall` issued from U-mode
 * (mcause == RISCV_EXC_ECALLU == 8) is dispatched to our cm-umode handler
 * instead of escalating to a fatal kernel exception.
 *
 * Integration model:
 *   - Zephyr's isr.S routes all non-ECALLM exceptions through
 *     z_riscv_fault(struct arch_esf *esf) when CONFIG_USERSPACE is OFF
 *     (which we deliberately keep OFF).
 *   - We mark the core z_riscv_fault() as __weak (see arch/riscv/core/fatal.c)
 *     and provide a strong override here.  For everything other than ECALLU
 *     we defer to the generic implementation via the renamed
 *     __aifoundry_default_riscv_fault() symbol.
 *
 * The esf passed in is the register-save frame built by isr.S.  We mutate it
 * in place (a0 = return value, mepc += 4) and return, which causes isr.S to
 * restore the frame and mret back to U-mode -- exactly the ECALLM fast-path
 * behaviour, just applied to a U-mode caller.
 *
 * RETURN_FROM_KERNEL is the exception: there is no U-mode to return to.
 * We call aifoundry_resume_from_kernel_asm() which tears down directly to the
 * M-mode caller of aifoundry_launch_umode().  That function does not return.
 */

#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/arch/exception.h>
#include <zephyr/arch/riscv/csr.h>
#include <zephyr/sys/printk.h>

#include "umode_abi.h"

/* Provided by umode_syscall.c and umode_return.S. */
int64_t aifoundry_umode_dispatch(uint64_t id, uint64_t a1, uint64_t a2, uint64_t a3);
void    aifoundry_resume_from_kernel_asm(int64_t retcode) __attribute__((noreturn));

/* For anything we do not handle we escalate to the same fatal-error sink
 * the generic z_riscv_fault() uses.  Keeping this path here (rather than
 * delegating to the __weak default) avoids needing linker --wrap or a
 * second entry point. */
extern FUNC_NORETURN void z_riscv_fatal_error(unsigned int reason,
					      const struct arch_esf *esf);

/* Match the ecall cause used by isr.S. */
#ifndef RISCV_EXC_ECALLU
#define RISCV_EXC_ECALLU 8
#endif

void z_riscv_fault(struct arch_esf *esf)
{
	unsigned long mcause = csr_read(mcause);
	unsigned long cause  = mcause & CONFIG_RISCV_MCAUSE_EXCEPTION_MASK;

	if (cause == RISCV_EXC_ECALLU) {
		uint64_t id = esf->a0;
		uint64_t a1 = esf->a1;
		uint64_t a2 = esf->a2;
		uint64_t a3 = esf->a3;

		if (id == SYSCALL_RETURN_FROM_KERNEL) {
			/* a1 carries the kernel return code. */
			aifoundry_resume_from_kernel_asm((int64_t)a1);
			/* Unreachable. */
		}

		int64_t ret = aifoundry_umode_dispatch(id, a1, a2, a3);
		esf->a0 = (unsigned long)ret;

		/* Skip past the ecall instruction so U-mode resumes after it. */
		esf->mepc += 4;
		return;
	}

	/* Not something we handle: fall back to Zephyr's fatal path. */
	z_riscv_fatal_error(K_ERR_CPU_EXCEPTION, esf);
}
