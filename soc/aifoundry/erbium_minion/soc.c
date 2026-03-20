/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/init.h>
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
