/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * U-mode RISC-V system timer driver.
 *
 * Uses the `time` CSR (readable from U-mode on standard RISC-V via
 * `rdtime`) to read mtime; arms mtimecmp via the OSKERN_SET_TIMEOUT
 * cm-umode syscall handled by MachineMinion.  The timer IRQ number is
 * taken from DTS; MachineMinion services the M-mode external interrupt,
 * claims the PLIC source, and upcalls into U-mode where Zephyr's IRQ
 * table dispatches to our ISR.
 *
 * Modelled on drivers/timer/riscv_machine_timer.c.
 */

#include <limits.h>

#include <zephyr/init.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/timer/system_timer.h>
#include <zephyr/sys_clock.h>
#include <zephyr/spinlock.h>
#include <zephyr/irq.h>

#include <isa/common/syscall.h>

#define CYC_PER_TICK (uint32_t)(sys_clock_hw_cycles_per_sec() / CONFIG_SYS_CLOCK_TICKS_PER_SEC)

#define cycle_diff_t   unsigned long
#define CYCLE_DIFF_MAX (~(cycle_diff_t)0)

#define CYCLES_MAX_1 ((uint64_t)INT32_MAX * (uint64_t)CYC_PER_TICK)
#define CYCLES_MAX_2 ((uint64_t)CYCLE_DIFF_MAX)
#define CYCLES_MAX_3 MIN(CYCLES_MAX_1, CYCLES_MAX_2)
#define CYCLES_MAX_4 (CYCLES_MAX_3 / 2 + CYCLES_MAX_3 / 4)
#define CYCLES_MAX   (CYCLES_MAX_4 + LSB_GET(CYCLES_MAX_4))

/* The timer IRQ number is supplied via DTS.  DT_HAS_RISCV_UMODE_TIMER
 * is set when the board DT declares a compatible="riscv,umode-timer"
 * node.  Fall back to a sensible default if none is provided. */
#if DT_HAS_COMPAT_STATUS_OKAY(riscv_umode_timer)
#define TIMER_IRQN DT_IRQN(DT_INST(0, riscv_umode_timer))
#else
#define TIMER_IRQN CONFIG_RISCV_UMODE_TIMER_IRQ
#endif

static struct k_spinlock lock;
static uint64_t last_count;
static uint64_t last_ticks;
static uint32_t last_elapsed;

#if defined(CONFIG_TEST)
const int32_t z_sys_timer_irq_for_test = TIMER_IRQN;
#endif

static inline uint64_t u_rdtime(void)
{
	uint64_t v;
	__asm__ volatile("rdtime %0" : "=r"(v));
	return v;
}

static inline void u_set_mtimecmp(uint64_t deadline)
{
	(void)syscall(SYSCALL_OSKERN_SET_TIMEOUT, deadline, 0, 0);
}

static uint64_t mtime(void)
{
	return u_rdtime();
}

static void timer_isr(const void *arg)
{
	ARG_UNUSED(arg);

	k_spinlock_key_t key = k_spin_lock(&lock);

	uint64_t now = mtime();
	uint64_t dcycles = now - last_count;
	uint32_t dticks = (cycle_diff_t)dcycles / CYC_PER_TICK;

	last_count += (cycle_diff_t)dticks * CYC_PER_TICK;
	last_ticks += dticks;
	last_elapsed = 0;

	if (!IS_ENABLED(CONFIG_TICKLESS_KERNEL)) {
		uint64_t next = last_count + CYC_PER_TICK;

		u_set_mtimecmp(next);
	}

	k_spin_unlock(&lock, key);
	sys_clock_announce(dticks);
}

void sys_clock_set_timeout(int32_t ticks, bool idle)
{
	ARG_UNUSED(idle);

	if (!IS_ENABLED(CONFIG_TICKLESS_KERNEL)) {
		return;
	}

	k_spinlock_key_t key = k_spin_lock(&lock);
	uint64_t cyc;

	if (ticks == K_TICKS_FOREVER) {
		cyc = last_count + CYCLES_MAX;
	} else {
		cyc = (last_ticks + last_elapsed + ticks) * CYC_PER_TICK;
		if ((cyc - last_count) > CYCLES_MAX) {
			cyc = last_count + CYCLES_MAX;
		}
	}
	u_set_mtimecmp(cyc);

	k_spin_unlock(&lock, key);
}

uint32_t sys_clock_elapsed(void)
{
	if (!IS_ENABLED(CONFIG_TICKLESS_KERNEL)) {
		return 0;
	}

	k_spinlock_key_t key = k_spin_lock(&lock);
	uint64_t now = mtime();
	uint64_t dcycles = now - last_count;
	uint32_t dticks = (cycle_diff_t)dcycles / CYC_PER_TICK;

	last_elapsed = dticks;
	k_spin_unlock(&lock, key);
	return dticks;
}

uint32_t sys_clock_cycle_get_32(void)
{
	return (uint32_t)mtime();
}

uint64_t sys_clock_cycle_get_64(void)
{
	return mtime();
}

static int sys_clock_driver_init(void)
{
	IRQ_CONNECT(TIMER_IRQN, 0, timer_isr, NULL, 0);
	last_ticks = mtime() / CYC_PER_TICK;
	last_count = last_ticks * CYC_PER_TICK;
	u_set_mtimecmp(last_count + CYC_PER_TICK);
	irq_enable(TIMER_IRQN);
	return 0;
}

SYS_INIT(sys_clock_driver_init, PRE_KERNEL_2, CONFIG_SYSTEM_CLOCK_INIT_PRIORITY);
