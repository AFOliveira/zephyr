/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode IRQ entry, ISR table dispatch, and arch IRQ knob plumbing.
 *
 * IRQ flow (ET-SoC1 cm-umode):
 *   MachineMinion takes the M-mode external interrupt, claims the PLIC
 *   source, saves the interrupted U-mode context, and mrets back into
 *   U-mode at the address registered via SYSCALL_OSKERN_REGISTER_IRQ_HANDLER
 *   (z_riscv_umode_irq_entry below) with a0 = irq_num.  We dispatch
 *   through _sw_isr_table[irq] exactly like the generic Zephyr RISC-V
 *   port would (minus the machine timer special-casing), then issue
 *   SYSCALL_OSKERN_IRQ_COMPLETE to hand control back to MM.  MM restores
 *   the saved U-mode context and mrets.
 *
 * MVP limitation: we do NOT perform preemptive reschedule from inside
 * the upcall.  After the ISR, if a higher-priority thread became
 * runnable (e.g. a k_sleep timeout fired), the actual switch happens on
 * the next cooperative yield (k_sleep, mutex block, ...).  True
 * preemptive context switching from within the upcall requires rewriting
 * the MM-saved context to point at the new thread's stack/PC, which is
 * out of scope for this first landing.  A cooperative demo
 * (k_sleep-driven threads) exercises the timer IRQ path fully.
 */

#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sw_isr_table.h>
#include <zephyr/sys/printk.h>

#include <isa/common/syscall.h>

/* Whether the OSKERN IRQ upcall has been registered with MM. */
static bool z_umode_irq_registered;

void z_riscv_umode_irq_entry(unsigned int irq);

/*
 * U-mode IRQ entry point.  Called by MachineMinion via mret-to-U at the
 * address registered through SYSCALL_OSKERN_REGISTER_IRQ_HANDLER, with
 * a0 = irq_num.  We run the matching SW ISR table entry, then ecall
 * IRQ_COMPLETE to resume the interrupted code.
 *
 * Note: we have no separate IRQ stack here.  We reuse whatever sp was in
 * the trap frame at the moment the M-mode handler mret'd us.  Per the
 * upcall protocol, M preserves the interrupted U-mode sp in its
 * umode_irq_save_ctx and mret's with the U-mode register file intact
 * (except a0 = irq_num); that means `sp` is the interrupted thread's
 * sp.  For the small amount of stack we use here (<256 bytes of ISR
 * dispatch) that's safe provided every Zephyr thread reserves enough
 * margin.  The dedicated IRQ-stack refinement is future work.
 */
void z_riscv_umode_irq_entry(unsigned int irq)
{
	if (IS_ENABLED(CONFIG_GEN_SW_ISR_TABLE)) {
		if (irq < (CONFIG_NUM_IRQS + CONFIG_RISCV_RESERVED_IRQ_ISR_TABLES_OFFSET)) {
			const struct _isr_table_entry *ent =
				&_sw_isr_table[irq];

			if (ent->isr != NULL) {
				ent->isr(ent->arg);
			}
		}
	}

	/* Hand control back to MM, which restores the saved U-mode
	 * context and mrets to the interrupted instruction. */
	(void)syscall(SYSCALL_OSKERN_IRQ_COMPLETE, (uint64_t)irq, 0, 0);

	/* MM never returns here -- if it does, hang rather than UB. */
	for (;;) {
		__asm__ volatile("nop");
	}
}

void arch_irq_enable(unsigned int irq)
{
	/* Lazily register the upcall entry the first time any IRQ is
	 * enabled.  Doing it at SYS_INIT would require a pre-main init
	 * hook that plays nicely with MULTITHREADING; deferring to the
	 * first arch_irq_enable() keeps the wiring in one place. */
	if (!z_umode_irq_registered) {
		(void)syscall(SYSCALL_OSKERN_REGISTER_IRQ_HANDLER,
			      (uint64_t)(uintptr_t)z_riscv_umode_irq_entry,
			      0, 0);
		z_umode_irq_registered = true;
	}

	/* Priority 1 so it beats the default threshold of 0. */
	(void)syscall(SYSCALL_OSKERN_IRQ_ENABLE, (uint64_t)irq, 1U, 0);
}

void arch_irq_disable(unsigned int irq)
{
	(void)syscall(SYSCALL_OSKERN_IRQ_DISABLE, (uint64_t)irq, 0, 0);
}

int arch_irq_is_enabled(unsigned int irq)
{
	/* MM doesn't expose a readback; assume enabled if we've been
	 * asked about it (conservative). */
	ARG_UNUSED(irq);
	return 1;
}

#if defined(CONFIG_RISCV_HAS_PLIC) || defined(CONFIG_RISCV_HAS_CLIC)
void z_riscv_irq_priority_set(unsigned int irq, unsigned int prio, uint32_t flags)
{
	ARG_UNUSED(flags);
	(void)syscall(SYSCALL_OSKERN_IRQ_ENABLE, (uint64_t)irq, (uint64_t)prio, 0);
}
#endif

/* The generated sw_isr_table fills unused slots with z_irq_spurious; the
 * upstream arch/riscv/core/irq_manage.c definition touches mcause/LOG
 * infrastructure we don't pull in here.  A tiny safety stub is enough. */
FUNC_NORETURN void z_irq_spurious(const void *unused)
{
	ARG_UNUSED(unused);
	printk("z_riscv_umode: spurious IRQ\n");
	for (;;) {
		__asm__ volatile("nop");
	}
}

/* TLS is disabled in the U-mode preempt build; the kernel still calls
 * this unconditionally from setup_thread_stack (thread.c:473).  Return
 * 0 to indicate no TLS stack reservation. */
size_t arch_tls_stack_setup(struct k_thread *new_thread, char *stack_ptr)
{
	ARG_UNUSED(new_thread);
	ARG_UNUSED(stack_ptr);
	return 0;
}
