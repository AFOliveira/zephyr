/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief ET-Minion Inter-Processor Interrupt (IPI) driver
 *
 * This driver implements SMP IPIs for Esperanto ET-Minion cores using
 * the IPI_TRIGGER and IPI_TRIGGER_CLEAR registers.
 *
 * Hardware: 8 ET-Minion cores × 2 hardware threads = 16 harts total
 * IPI registers control MSIP (Machine Software Interrupt Pending) for each hart.
 */

#include <zephyr/kernel.h>
#include <zephyr/arch/cpu.h>
#include <zephyr/init.h>
#include <zephyr/irq.h>
#include <zephyr/sys/atomic.h>
#include <zephyr/arch/riscv/irq.h>
#include <kernel_arch_interface.h>
#include <ipi.h>

#ifdef CONFIG_RISCV_ET_MINION_CACHE
#include <zephyr/cache.h>
#endif

/* ET-Minion IPI register addresses (64-bit) - Erbium ESR region */
#define IPI_TRIGGER_BASE       0x0080F40090ULL
#define IPI_TRIGGER_CLEAR_BASE 0x0080F40098ULL

#define IPI_TRIGGER       ((volatile uint64_t *)IPI_TRIGGER_BASE)
#define IPI_TRIGGER_CLEAR ((volatile uint64_t *)IPI_TRIGGER_CLEAR_BASE)

/* IPI reasons - matches Zephyr's IPI framework */
#define IPI_SCHED     0
#define IPI_FPU_FLUSH 1

/* Per-CPU pending IPI flags */
static atomic_t cpu_pending_ipi[CONFIG_MP_MAX_NUM_CPUS];

/**
 * @brief Send directed IPIs to target CPUs
 *
 * @param cpu_bitmap Bitmap of target CPU IDs to interrupt
 */
void arch_sched_directed_ipi(uint32_t cpu_bitmap)
{
	unsigned int key = arch_irq_lock();
	unsigned int id = _current_cpu->id;
	unsigned int num_cpus = arch_num_cpus();
	uint64_t hart_mask = 0;

	/* Build hart mask from CPU bitmap */
	for (unsigned int cpu_id = 0; cpu_id < num_cpus; cpu_id++) {
		if ((cpu_id != id) && _kernel.cpus[cpu_id].arch.online &&
		    (cpu_bitmap & BIT(cpu_id))) {
			/* Mark IPI pending for this CPU */
			atomic_set_bit(&cpu_pending_ipi[cpu_id], IPI_SCHED);

#ifdef CONFIG_RISCV_ET_MINION_CACHE
			/*
			 * Cache coherency: Flush the pending flag to memory so the
			 * target CPU sees the updated value when it reads.
			 * This is critical for non-coherent caches.
			 */
			sys_cache_data_flush_range(&cpu_pending_ipi[cpu_id],
						   sizeof(atomic_t));
#endif

			/* Get hartid for this CPU */
			uint32_t hartid = _kernel.cpus[cpu_id].arch.hartid;
			hart_mask |= (1ULL << hartid);
		}
	}

	/* Trigger MSIP for all target harts atomically */
	if (hart_mask) {
		*IPI_TRIGGER = hart_mask;  /* MMIO write (uncached) */
	}

	arch_irq_unlock(key);
}

#ifdef CONFIG_FPU_SHARING
/**
 * @brief Request FPU flush on target CPU
 *
 * @param cpu Target CPU ID
 */
void arch_flush_fpu_ipi(unsigned int cpu)
{
	atomic_set_bit(&cpu_pending_ipi[cpu], IPI_FPU_FLUSH);

#ifdef CONFIG_RISCV_ET_MINION_CACHE
	/* Flush pending flag to memory for target CPU */
	sys_cache_data_flush_range(&cpu_pending_ipi[cpu], sizeof(atomic_t));
#endif

	uint32_t hartid = _kernel.cpus[cpu].arch.hartid;
	*IPI_TRIGGER = (1ULL << hartid);
}
#endif /* CONFIG_FPU_SHARING */

/**
 * @brief IPI interrupt handler
 *
 * Handles Machine Software Interrupts (MSIP) triggered by IPIs.
 */
static void sched_ipi_handler(const void *unused)
{
	ARG_UNUSED(unused);

	uint32_t hartid = csr_read(mhartid);
	unsigned int cpu_id = _current_cpu->id;

	/* Clear MSIP for this hart */
	*IPI_TRIGGER_CLEAR = (1ULL << hartid);  /* MMIO write (uncached) */

#ifdef CONFIG_RISCV_ET_MINION_CACHE
	/*
	 * Cache coherency: Invalidate the pending flag cache line to force
	 * reading from memory. This ensures we see the value written by the
	 * sending CPU.
	 */
	sys_cache_data_invd_range(&cpu_pending_ipi[cpu_id], sizeof(atomic_t));
#endif

	/* Read and clear pending IPI reasons atomically */
	atomic_val_t pending_ipi = atomic_clear(&cpu_pending_ipi[cpu_id]);

	if (pending_ipi & ATOMIC_MASK(IPI_SCHED)) {
		/* Handle scheduler IPI - invoke thread rescheduling */
		z_sched_ipi();
	}
#ifdef CONFIG_FPU_SHARING
	if (pending_ipi & ATOMIC_MASK(IPI_FPU_FLUSH)) {
		/* Disable IRQs */
		csr_clear(mstatus, MSTATUS_IEN);
		/* Perform the flush */
		arch_flush_local_fpu();
		/*
		 * No need to re-enable IRQs here as long as
		 * this remains the last case.
		 */
	}
#endif /* CONFIG_FPU_SHARING */
}

#ifdef CONFIG_FPU_SHARING
/**
 * @brief Spinlock relaxation with FPU flush handling
 *
 * Make sure there is no pending FPU flush request for this CPU while
 * waiting for a contended spinlock. This prevents deadlock when the lock
 * we need is taken by another CPU that wants its FPU content reinstated
 * while such content is still live in this CPU's FPU.
 */
void arch_spin_relax(void)
{
	atomic_val_t *pending_ipi = &cpu_pending_ipi[_current_cpu->id];

	if (atomic_test_and_clear_bit(pending_ipi, IPI_FPU_FLUSH)) {
		/*
		 * We may not be in IRQ context here hence cannot use
		 * arch_flush_local_fpu() directly.
		 */
		arch_float_disable(_current_cpu->arch.fpu_owner);
	}
}
#endif /* CONFIG_FPU_SHARING */

/**
 * @brief Initialize SMP IPI subsystem
 *
 * Called during early kernel initialization to set up IPI handling.
 *
 * @return 0 on success
 */
int arch_smp_init(void)
{
	/* Register IPI handler for Machine Software Interrupt */
	IRQ_CONNECT(RISCV_IRQ_MSOFT, 0, sched_ipi_handler, NULL, 0);
	irq_enable(RISCV_IRQ_MSOFT);

	return 0;
}
