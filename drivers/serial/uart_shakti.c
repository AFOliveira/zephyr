/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Zephyr UART driver for the Shakti UART_V3.
 *
 * The V3 TX sequence requires writing the data register followed by
 * baud, delay, interrupt-enable, and receiver-threshold registers
 * for each byte transmitted.
 */

#define DT_DRV_COMPAT aifoundry_shakti_uart

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/sys_io.h>
#include <zephyr/irq.h>

/* Shakti UART register indices and offset computation */
#define SHAKTI_REG(idx)             ((idx) << CONFIG_UART_SHAKTI_REG_SHIFT)
#define SHAKTI_UART_BAUD            SHAKTI_REG(0)
#define SHAKTI_UART_TX_REG          SHAKTI_REG(1)
#define SHAKTI_UART_RCV_REG         SHAKTI_REG(2)
#define SHAKTI_UART_STATUS          SHAKTI_REG(3)
#define SHAKTI_UART_DELAY           SHAKTI_REG(4)
#define SHAKTI_UART_CONTROL         SHAKTI_REG(5)
#define SHAKTI_UART_IEN             SHAKTI_REG(6)
#define SHAKTI_UART_RX_THRESHOLD    SHAKTI_REG(8)

/* STATUS register bits */
#define STATUS_TX_EMPTY     BIT(0)
#define STATUS_TX_FULL      BIT(1)
#define STATUS_RX_NOT_EMPTY BIT(2)
#define STATUS_RX_FULL      BIT(3)

struct uart_shakti_config {
	uintptr_t base;
	uint32_t clock_freq;
	uint32_t baud_rate;
	uint16_t baud_divisor;
};

struct uart_shakti_data {
	uint32_t ien_shadow;
};

/* V3 TX trigger: write config registers after each TX_REG write. */
static void uart_shakti_tx_trigger(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	sys_write32(cfg->baud_divisor, cfg->base + SHAKTI_UART_BAUD);
	sys_write32(0U, cfg->base + SHAKTI_UART_DELAY);
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
	sys_write32(0U, cfg->base + SHAKTI_UART_RX_THRESHOLD);
}

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

	while ((sys_read32(cfg->base + SHAKTI_UART_STATUS) & STATUS_TX_FULL) != 0U) {
		;
	}

	sys_write32((uint32_t)c, cfg->base + SHAKTI_UART_TX_REG);
	uart_shakti_tx_trigger(dev);
}

static int uart_shakti_err_check(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 0;
}

static int uart_shakti_init(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow = 0U;

	if (cfg->baud_rate > 0U) {
		sys_write32(cfg->baud_divisor, cfg->base + SHAKTI_UART_BAUD);
	}

	/* 8N1: charsize=8, parity=none, stopbits=1 */
	sys_write32(8U << 5, cfg->base + SHAKTI_UART_CONTROL);

	sys_write32(0U, cfg->base + SHAKTI_UART_DELAY);
	sys_write32(0U, cfg->base + SHAKTI_UART_IEN);
	sys_write32(0U, cfg->base + SHAKTI_UART_RX_THRESHOLD);

	return 0;
}

static DEVICE_API(uart, uart_shakti_driver_api) = {
	.poll_in = uart_shakti_poll_in,
	.poll_out = uart_shakti_poll_out,
	.err_check = uart_shakti_err_check,
};

#define UART_SHAKTI_DATA(n) static struct uart_shakti_data uart_shakti_data_##n;

#define UART_SHAKTI_INIT(n)                                                    \
	UART_SHAKTI_DATA(n)                                                    \
	static const struct uart_shakti_config uart_shakti_cfg_##n = {         \
		.base = DT_INST_REG_ADDR(n),                                  \
		.clock_freq = DT_INST_PROP(n, clock_frequency),               \
		.baud_rate = DT_INST_PROP(n, current_speed),                  \
		.baud_divisor = (uint16_t)(DT_INST_PROP(n, clock_frequency) / \
			(16U * DT_INST_PROP(n, current_speed))),              \
	};                                                                     \
	DEVICE_DT_INST_DEFINE(n,                                               \
			      uart_shakti_init,                                \
			      NULL,                                            \
			      &uart_shakti_data_##n,                           \
			      &uart_shakti_cfg_##n,                            \
			      PRE_KERNEL_1,                                    \
			      CONFIG_SERIAL_INIT_PRIORITY,                     \
			      &uart_shakti_driver_api);

DT_INST_FOREACH_STATUS_OKAY(UART_SHAKTI_INIT)
