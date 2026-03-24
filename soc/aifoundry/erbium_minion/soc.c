/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/init.h>
#include <zephyr/irq.h>
#include <zephyr/sys/util.h>
#include <zephyr/devicetree.h>

/* SystemConfig register: offset 0x08 from sysreg base */
#define SYSREG_SYSCFG_OFF     0x08
#define SYSCFG_UART_ENABLE    BIT(6)

#define SYSREG_NODE DT_NODELABEL(sysreg)
#define SYSREG_BASE DT_REG_ADDR(SYSREG_NODE)

static int erbium_pinmux_init(void)
{
	volatile uint32_t *syscfg =
		(volatile uint32_t *)(SYSREG_BASE + SYSREG_SYSCFG_OFF);

	/* Enable UART pin mux so TX/RX pins are routed to the UART peripheral */
	*syscfg |= SYSCFG_UART_ENABLE;

	return 0;
}

SYS_INIT(erbium_pinmux_init, PRE_KERNEL_1, 0);

/*
 * Custom idle: spin with interrupts enabled instead of WFI.
 *
 * The erbium emulator only advances peripheral clocks (UART, PLIC) while a
 * hart is actively executing instructions.  A real WFI stops the hart, so
 * peripheral interrupts that arrive while sleeping are never delivered.
 * Spinning keeps the hart "active" from the emulator's perspective.
 */
void arch_cpu_idle(void)
{
	irq_unlock(MSTATUS_IEN);
}

void arch_cpu_atomic_idle(unsigned int key)
{
	irq_unlock(key);
}
