/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Zephyr UART driver for the Shakti UART on Erbium.
 * Supports poll-mode and optional interrupt-driven mode.
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

/* IEN (Interrupt Enable) register bits — mirror STATUS bit positions */
#define IEN_TX_EMPTY        BIT(0)
#define IEN_TX_FULL         BIT(1)
#define IEN_RX_NOT_EMPTY    BIT(2)

#ifdef CONFIG_UART_INTERRUPT_DRIVEN
typedef void (*irq_cfg_func_t)(void);
#endif

struct uart_shakti_config {
	uintptr_t base;
	uint32_t clock_freq;
	uint32_t baud_rate;
	uint16_t baud_divisor;
#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	irq_cfg_func_t irq_cfg_func;
#endif
};

struct uart_shakti_data {
	uint32_t ien_shadow;
#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	uart_irq_callback_user_data_t callback;
	void *cb_data;
#endif
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

static void uart_shakti_tx_trigger(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	/*
	 * V3 TX sequence step 2: write baud, TX delay, interrupt enable,
	 * and receiver threshold registers to trigger transmission.
	 * Use cached values to avoid slow MMIO reads.
	 */
	sys_write32(cfg->baud_divisor, cfg->base + SHAKTI_UART_BAUD);
	sys_write32(0U, cfg->base + SHAKTI_UART_DELAY);
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
	sys_write32(0U, cfg->base + SHAKTI_UART_RX_THRESHOLD);
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

#ifdef CONFIG_UART_INTERRUPT_DRIVEN

static int uart_shakti_fifo_fill(const struct device *dev, const uint8_t *tx_data, int size)
{
	const struct uart_shakti_config *cfg = dev->config;
	int i;

	for (i = 0; i < size; i++) {
		if ((sys_read32(cfg->base + SHAKTI_UART_STATUS) & STATUS_TX_FULL) != 0U) {
			break;
		}
		sys_write32((uint32_t)tx_data[i], cfg->base + SHAKTI_UART_TX_REG);
	}

	return i;
}

static int uart_shakti_fifo_read(const struct device *dev, uint8_t *rx_data, const int size)
{
	const struct uart_shakti_config *cfg = dev->config;
	int i;

	for (i = 0; i < size; i++) {
		if ((sys_read32(cfg->base + SHAKTI_UART_STATUS) & STATUS_RX_NOT_EMPTY) == 0U) {
			break;
		}
		rx_data[i] = (uint8_t)(sys_read32(cfg->base + SHAKTI_UART_RCV_REG) & 0xffU);
	}

	return i;
}

static void uart_shakti_irq_tx_enable(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow |= IEN_TX_FULL;
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
}

static void uart_shakti_irq_tx_disable(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow &= ~IEN_TX_FULL;
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
}

static int uart_shakti_irq_tx_ready(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	uint32_t status = sys_read32(cfg->base + SHAKTI_UART_STATUS);

	return !!(status & STATUS_TX_EMPTY);
}

static int uart_shakti_irq_tx_complete(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	uint32_t status = sys_read32(cfg->base + SHAKTI_UART_STATUS);

	return !!(status & STATUS_TX_EMPTY);
}

static void uart_shakti_irq_rx_enable(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow |= IEN_RX_NOT_EMPTY;
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
}

static void uart_shakti_irq_rx_disable(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow &= ~IEN_RX_NOT_EMPTY;
	sys_write32(data->ien_shadow, cfg->base + SHAKTI_UART_IEN);
}

static int uart_shakti_irq_rx_ready(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	uint32_t status = sys_read32(cfg->base + SHAKTI_UART_STATUS);

	return !!(status & STATUS_RX_NOT_EMPTY);
}

static void uart_shakti_irq_err_enable(const struct device *dev)
{
	ARG_UNUSED(dev);
}

static void uart_shakti_irq_err_disable(const struct device *dev)
{
	ARG_UNUSED(dev);
}

static int uart_shakti_irq_is_pending(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	uint32_t status = sys_read32(cfg->base + SHAKTI_UART_STATUS);
	uint32_t ien = sys_read32(cfg->base + SHAKTI_UART_IEN);

	return !!(status & ien);
}

static int uart_shakti_irq_update(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 1;
}

static void uart_shakti_irq_callback_set(const struct device *dev,
					  uart_irq_callback_user_data_t cb,
					  void *cb_data)
{
	struct uart_shakti_data *data = dev->data;

	data->callback = cb;
	data->cb_data = cb_data;
}

static void uart_shakti_irq_handler(const struct device *dev)
{
	struct uart_shakti_data *data = dev->data;

	if (data->callback) {
		data->callback(dev, data->cb_data);
	}
}

#endif /* CONFIG_UART_INTERRUPT_DRIVEN */

static int uart_shakti_init(const struct device *dev)
{
	const struct uart_shakti_config *cfg = dev->config;
	struct uart_shakti_data *data = dev->data;

	data->ien_shadow = 0U;

	/* Configure baud rate: baud_value = clk_freq / (16 * baud_rate) */
	if (cfg->baud_rate > 0U) {
		sys_write32(cfg->baud_divisor, cfg->base + SHAKTI_UART_BAUD);
	}

	/* Configure 8N1: charsize=8, parity=none(0), stopbits=1(0) */
	sys_write32(8U << 5, cfg->base + SHAKTI_UART_CONTROL);

	/*
	 * V3 TX sequence step 2 (one-time): write baud, delay, IEN,
	 * and rx_threshold to configure the transmitter.
	 */
	sys_write32(0U, cfg->base + SHAKTI_UART_DELAY);
	sys_write32(0U, cfg->base + SHAKTI_UART_IEN);
	sys_write32(0U, cfg->base + SHAKTI_UART_RX_THRESHOLD);

#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	/* Connect and enable the IRQ */
	cfg->irq_cfg_func();
#endif
	return 0;
}

static DEVICE_API(uart, uart_shakti_driver_api) = {
	.poll_in = uart_shakti_poll_in,
	.poll_out = uart_shakti_poll_out,
	.err_check = uart_shakti_err_check,
#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	.fifo_fill = uart_shakti_fifo_fill,
	.fifo_read = uart_shakti_fifo_read,
	.irq_tx_enable = uart_shakti_irq_tx_enable,
	.irq_tx_disable = uart_shakti_irq_tx_disable,
	.irq_tx_ready = uart_shakti_irq_tx_ready,
	.irq_tx_complete = uart_shakti_irq_tx_complete,
	.irq_rx_enable = uart_shakti_irq_rx_enable,
	.irq_rx_disable = uart_shakti_irq_rx_disable,
	.irq_rx_ready = uart_shakti_irq_rx_ready,
	.irq_err_enable = uart_shakti_irq_err_enable,
	.irq_err_disable = uart_shakti_irq_err_disable,
	.irq_is_pending = uart_shakti_irq_is_pending,
	.irq_update = uart_shakti_irq_update,
	.irq_callback_set = uart_shakti_irq_callback_set,
#endif
};

#ifdef CONFIG_UART_INTERRUPT_DRIVEN

#define UART_SHAKTI_IRQ_CFG_FUNC(n)                                            \
	static void uart_shakti_irq_cfg_func_##n(void)                        \
	{                                                                      \
		IRQ_CONNECT(DT_INST_IRQN(n), DT_INST_IRQ(n, priority),        \
			    uart_shakti_irq_handler,                           \
			    DEVICE_DT_INST_GET(n), 0);                         \
		irq_enable(DT_INST_IRQN(n));                                   \
	}

#define UART_SHAKTI_IRQ_CFG_FUNC_INIT(n) .irq_cfg_func = uart_shakti_irq_cfg_func_##n,

#else

#define UART_SHAKTI_IRQ_CFG_FUNC(n)
#define UART_SHAKTI_IRQ_CFG_FUNC_INIT(n)

#endif /* CONFIG_UART_INTERRUPT_DRIVEN */

#define UART_SHAKTI_DATA(n) static struct uart_shakti_data uart_shakti_data_##n;
#define UART_SHAKTI_DATA_REF(n) &uart_shakti_data_##n,

#define UART_SHAKTI_INIT(n)                                                    \
	UART_SHAKTI_IRQ_CFG_FUNC(n)                                            \
	UART_SHAKTI_DATA(n)                                                    \
	static const struct uart_shakti_config uart_shakti_cfg_##n = {         \
		.base = DT_INST_REG_ADDR(n),                                  \
		.clock_freq = DT_INST_PROP(n, clock_frequency),               \
		.baud_rate = DT_INST_PROP(n, current_speed),                  \
		.baud_divisor = (uint16_t)(DT_INST_PROP(n, clock_frequency) / \
			(16U * DT_INST_PROP(n, current_speed))),              \
		UART_SHAKTI_IRQ_CFG_FUNC_INIT(n)                              \
	};                                                                     \
	DEVICE_DT_INST_DEFINE(n,                                               \
			      uart_shakti_init,                                \
			      NULL,                                            \
			      UART_SHAKTI_DATA_REF(n)                          \
			      &uart_shakti_cfg_##n,                            \
			      PRE_KERNEL_1,                                    \
			      CONFIG_SERIAL_INIT_PRIORITY,                     \
			      &uart_shakti_driver_api);

DT_INST_FOREACH_STATUS_OKAY(UART_SHAKTI_INIT)
