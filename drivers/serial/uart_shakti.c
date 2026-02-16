/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal Zephyr UART driver for the Shakti UART on Erbium.
 * Poll-mode only (no interrupt support).
 */

#define DT_DRV_COMPAT aifoundry_shakti_uart

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/sys_io.h>

/* Shakti UART register indices and offset computation */
#define SHAKTI_REG(idx)     ((idx) << CONFIG_UART_SHAKTI_REG_SHIFT)
#define SHAKTI_UART_BAUD    SHAKTI_REG(0)
#define SHAKTI_UART_TX_REG  SHAKTI_REG(1)
#define SHAKTI_UART_RCV_REG SHAKTI_REG(2)
#define SHAKTI_UART_STATUS  SHAKTI_REG(3)
#define SHAKTI_UART_CONTROL SHAKTI_REG(5)

/* STATUS register bits */
#define STATUS_TX_EMPTY     BIT(0)
#define STATUS_TX_FULL      BIT(1)
#define STATUS_RX_NOT_EMPTY BIT(2)

struct uart_shakti_config {
	uintptr_t base;
};

static int uart_shakti_poll_in(const struct device *dev, unsigned char *c)
{
	const struct uart_shakti_config *cfg = dev->config;
	uint32_t status = sys_read32(cfg->base + SHAKTI_UART_STATUS);

	if ((status & STATUS_RX_NOT_EMPTY) == 0U) {
		return -1;
	}

	*c = (unsigned char)(sys_read32(cfg->base + SHAKTI_UART_RCV_REG) & 0xffU);
	return 0;
}

static void uart_shakti_poll_out(const struct device *dev, unsigned char c)
{
	const struct uart_shakti_config *cfg = dev->config;

	/* Wait while TX FIFO is full */
	while ((sys_read32(cfg->base + SHAKTI_UART_STATUS) & STATUS_TX_FULL) != 0U) {
		;
	}

	sys_write32((uint32_t)c, cfg->base + SHAKTI_UART_TX_REG);
}

static int uart_shakti_err_check(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 0;
}

static int uart_shakti_init(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 0;
}

static const struct uart_driver_api uart_shakti_driver_api = {
	.poll_in = uart_shakti_poll_in,
	.poll_out = uart_shakti_poll_out,
	.err_check = uart_shakti_err_check,
};

#define UART_SHAKTI_INIT(n)                                                    \
	static const struct uart_shakti_config uart_shakti_cfg_##n = {         \
		.base = DT_INST_REG_ADDR(n),                                  \
	};                                                                     \
	DEVICE_DT_INST_DEFINE(n,                                               \
			      uart_shakti_init,                                \
			      NULL,                                            \
			      NULL,                                            \
			      &uart_shakti_cfg_##n,                            \
			      PRE_KERNEL_1,                                    \
			      CONFIG_SERIAL_INIT_PRIORITY,                     \
			      &uart_shakti_driver_api);

DT_INST_FOREACH_STATUS_OKAY(UART_SHAKTI_INIT)
