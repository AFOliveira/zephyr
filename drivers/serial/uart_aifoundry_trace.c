/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Virtual UART driver that forwards bytes through the portable
 * AIFoundry HAL (aifoundry_log).  Designed to satisfy the contract
 * ArduinoCore-zephyr's `ZephyrSerial` requires (poll_out + IRQ-driven
 * TX path) on targets where there is no real UART available — namely
 * `etsoc1_minion_umode`, where the U-mode kernel cannot access
 * peripheral MMIO and `aifoundry_log` is the only output channel
 * (et-trace ring on silicon, et_dev_trace_decoder on the host).
 *
 * Implementation notes:
 *
 *   - poll_out / fifo_fill simply call aifoundry_log("%c", c) once per
 *     byte.  Each byte is one trace ring entry.  The decoder will
 *     stitch them back into lines.  We considered buffering until \n
 *     but kept it simple for v1; a bigger buffer can come later if the
 *     per-char ring spam annoys.
 *
 *   - The Arduino Serial layer pushes bytes into its TX ring buffer
 *     and then calls uart_irq_tx_enable() expecting a hardware IRQ to
 *     fire later and drain.  We have no IRQs.  Instead, irq_tx_enable
 *     calls back into the registered IRQ callback synchronously,
 *     letting it loop on irq_tx_ready() / fifo_fill() and drain its
 *     own ring buffer immediately.  irq_tx_ready always returns 1
 *     (we can always accept more); irq_update returns 1 once so the
 *     handler runs, then 0 to break out.
 *
 *   - poll_in always returns -1 (no input).  The Arduino sketches we
 *     care about don't read Serial.
 *
 *   - configure() is a no-op: there is no real baud rate to set, but
 *     ZephyrSerial::begin() calls uart_configure() unconditionally.
 *     Returning 0 keeps it happy.
 */

#define DT_DRV_COMPAT aifoundry_umode_trace_uart

#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/util.h>

#include <zephyr/aifoundry/runtime.h>

struct uart_aifoundry_trace_data {
	uart_irq_callback_user_data_t cb;
	void *cb_user_data;
	bool tx_irq_enabled;
	/* irq_update returns 1 the first time after a tx_enable / pseudo-
	 * trigger so the registered handler runs once; subsequent calls
	 * return 0 so its `if (!uart_irq_update(...)) return;` guard
	 * eventually breaks out. */
	bool irq_pending;
};

static int uart_aifoundry_trace_init(const struct device *dev)
{
	(void)dev;
	return 0;
}

static void uart_aifoundry_trace_poll_out(const struct device *dev, unsigned char c)
{
	(void)dev;
	aifoundry_log("%c", (int)c);
}

static int uart_aifoundry_trace_poll_in(const struct device *dev, unsigned char *p_char)
{
	(void)dev;
	(void)p_char;
	return -1;
}

#ifdef CONFIG_UART_USE_RUNTIME_CONFIGURE
static int uart_aifoundry_trace_configure(const struct device *dev,
					   const struct uart_config *cfg)
{
	(void)dev;
	(void)cfg;
	return 0;
}

static int uart_aifoundry_trace_config_get(const struct device *dev,
					    struct uart_config *cfg)
{
	(void)dev;
	*cfg = (struct uart_config){
		.baudrate = 115200,
		.parity = UART_CFG_PARITY_NONE,
		.stop_bits = UART_CFG_STOP_BITS_1,
		.data_bits = UART_CFG_DATA_BITS_8,
		.flow_ctrl = UART_CFG_FLOW_CTRL_NONE,
	};
	return 0;
}
#endif

#ifdef CONFIG_UART_INTERRUPT_DRIVEN

static void uart_aifoundry_trace_drain(const struct device *dev);

static int uart_aifoundry_trace_fifo_fill(const struct device *dev,
					   const uint8_t *tx_data, int len)
{
	(void)dev;
	for (int i = 0; i < len; i++) {
		aifoundry_log("%c", (int)tx_data[i]);
	}
	return len;
}

static int uart_aifoundry_trace_fifo_read(const struct device *dev,
					   uint8_t *rx_data, const int size)
{
	(void)dev;
	(void)rx_data;
	(void)size;
	return 0;
}

static void uart_aifoundry_trace_irq_tx_enable(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;

	data->tx_irq_enabled = true;
	uart_aifoundry_trace_drain(dev);
}

static void uart_aifoundry_trace_irq_tx_disable(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;

	data->tx_irq_enabled = false;
}

static int uart_aifoundry_trace_irq_tx_ready(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;

	return data->tx_irq_enabled ? 1 : 0;
}

static void uart_aifoundry_trace_irq_rx_enable(const struct device *dev)
{
	(void)dev;
}

static void uart_aifoundry_trace_irq_rx_disable(const struct device *dev)
{
	(void)dev;
}

static int uart_aifoundry_trace_irq_tx_complete(const struct device *dev)
{
	(void)dev;
	return 1;
}

static int uart_aifoundry_trace_irq_rx_ready(const struct device *dev)
{
	(void)dev;
	return 0;
}

static void uart_aifoundry_trace_irq_err_enable(const struct device *dev)
{
	(void)dev;
}

static void uart_aifoundry_trace_irq_err_disable(const struct device *dev)
{
	(void)dev;
}

static int uart_aifoundry_trace_irq_is_pending(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;

	return data->irq_pending ? 1 : 0;
}

static int uart_aifoundry_trace_irq_update(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;
	int v = data->irq_pending ? 1 : 0;

	data->irq_pending = false;
	return v;
}

static void uart_aifoundry_trace_irq_callback_set(const struct device *dev,
						   uart_irq_callback_user_data_t cb,
						   void *user_data)
{
	struct uart_aifoundry_trace_data *data = dev->data;

	data->cb = cb;
	data->cb_user_data = user_data;
}

/* Synchronously drain whatever the registered callback has queued in
 * its software ring buffer.  Loop until the callback turns off the
 * TX-ready bit (i.e. its ring is empty and it called irq_tx_disable). */
static void uart_aifoundry_trace_drain(const struct device *dev)
{
	struct uart_aifoundry_trace_data *data = dev->data;
	int safety = 0;

	while (data->cb != NULL && data->tx_irq_enabled) {
		data->irq_pending = true;
		data->cb(dev, data->cb_user_data);
		if (!data->tx_irq_enabled) {
			break;
		}
		if (++safety > 4096) {
			break;
		}
	}
	data->irq_pending = false;
}

#endif /* CONFIG_UART_INTERRUPT_DRIVEN */

static DEVICE_API(uart, uart_aifoundry_trace_api) = {
	.poll_in = uart_aifoundry_trace_poll_in,
	.poll_out = uart_aifoundry_trace_poll_out,
#ifdef CONFIG_UART_USE_RUNTIME_CONFIGURE
	.configure = uart_aifoundry_trace_configure,
	.config_get = uart_aifoundry_trace_config_get,
#endif
#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	.fifo_fill = uart_aifoundry_trace_fifo_fill,
	.fifo_read = uart_aifoundry_trace_fifo_read,
	.irq_tx_enable = uart_aifoundry_trace_irq_tx_enable,
	.irq_tx_disable = uart_aifoundry_trace_irq_tx_disable,
	.irq_tx_ready = uart_aifoundry_trace_irq_tx_ready,
	.irq_rx_enable = uart_aifoundry_trace_irq_rx_enable,
	.irq_rx_disable = uart_aifoundry_trace_irq_rx_disable,
	.irq_tx_complete = uart_aifoundry_trace_irq_tx_complete,
	.irq_rx_ready = uart_aifoundry_trace_irq_rx_ready,
	.irq_err_enable = uart_aifoundry_trace_irq_err_enable,
	.irq_err_disable = uart_aifoundry_trace_irq_err_disable,
	.irq_is_pending = uart_aifoundry_trace_irq_is_pending,
	.irq_update = uart_aifoundry_trace_irq_update,
	.irq_callback_set = uart_aifoundry_trace_irq_callback_set,
#endif
};

#define UART_AIFOUNDRY_TRACE_INIT(n)                                                                  \
	static struct uart_aifoundry_trace_data uart_aifoundry_trace_data_##n;                         \
	DEVICE_DT_INST_DEFINE(n, uart_aifoundry_trace_init, NULL,                                      \
			      &uart_aifoundry_trace_data_##n, NULL,                                    \
			      PRE_KERNEL_1, CONFIG_SERIAL_INIT_PRIORITY,                               \
			      &uart_aifoundry_trace_api);

DT_INST_FOREACH_STATUS_OKAY(UART_AIFOUNDRY_TRACE_INIT)
