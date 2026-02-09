/*
 * Copyright (c) 2025 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal Zephyr UART driver that wraps the ET common DW_apb_uart
 * routines used by the FreeRTOS stack. This preserves the THRE
 * semantics expected there (Mode B: THRE=1 means FIFO full).
 */

#define DT_DRV_COMPAT aifoundry_etsoc_dw_apb_uart

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/sys_io.h>

#include "etsoc/drivers/serial/serial.h"
#include "etsoc_hal/inc/DW_apb_uart.h"

LOG_MODULE_REGISTER(uart_etsoc_dw_apb, CONFIG_UART_LOG_LEVEL);

struct uart_etsoc_dw_apb_config {
	uintptr_t base;
	uint32_t sys_clk_freq;
};

struct uart_etsoc_dw_apb_data {
	struct uart_config uart_cfg;
};

static int uart_etsoc_dw_apb_poll_in(const struct device *dev, unsigned char *c)
{
	const struct uart_etsoc_dw_apb_config *cfg = dev->config;
	uint32_t lsr = sys_read32(cfg->base + UART_LSR_ADDRESS);

	if (UART_LSR_DR_GET(lsr) != 1U) {
		return -1;
	}

	*c = (unsigned char)UART_RBR_RBR_RBR_GET(
		sys_read32(cfg->base + UART_RBR_ADDRESS));
	return 0;
}

static void uart_etsoc_dw_apb_poll_out(const struct device *dev, unsigned char c)
{
	const struct uart_etsoc_dw_apb_config *cfg = dev->config;
	const char ch = (char)c;

	(void)SERIAL_write(cfg->base, &ch, 1);
}

static int uart_etsoc_dw_apb_err_check(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 0;
}

#ifdef CONFIG_UART_USE_RUNTIME_CONFIGURE
static int uart_etsoc_dw_apb_configure(const struct device *dev,
				       const struct uart_config *cfg)
{
	struct uart_etsoc_dw_apb_data *data = dev->data;
	const struct uart_etsoc_dw_apb_config *dev_cfg = dev->config;

	if (cfg->baudrate != 115200 ||
	    cfg->parity != UART_CFG_PARITY_NONE ||
	    cfg->stop_bits != UART_CFG_STOP_BITS_1 ||
	    cfg->data_bits != UART_CFG_DATA_BITS_8 ||
	    cfg->flow_ctrl != UART_CFG_FLOW_CTRL_NONE) {
		return -ENOTSUP;
	}

	(void)SERIAL_init(dev_cfg->base);
	data->uart_cfg = *cfg;
	return 0;
}

static int uart_etsoc_dw_apb_config_get(const struct device *dev,
					struct uart_config *cfg)
{
	const struct uart_etsoc_dw_apb_data *data = dev->data;

	*cfg = data->uart_cfg;
	return 0;
}
#endif

static int uart_etsoc_dw_apb_init(const struct device *dev)
{
	struct uart_etsoc_dw_apb_data *data = dev->data;
	const struct uart_etsoc_dw_apb_config *cfg = dev->config;

	(void)SERIAL_init(cfg->base);

	data->uart_cfg.baudrate = 115200;
	data->uart_cfg.parity = UART_CFG_PARITY_NONE;
	data->uart_cfg.stop_bits = UART_CFG_STOP_BITS_1;
	data->uart_cfg.data_bits = UART_CFG_DATA_BITS_8;
	data->uart_cfg.flow_ctrl = UART_CFG_FLOW_CTRL_NONE;

	return 0;
}

static const struct uart_driver_api uart_etsoc_dw_apb_driver_api = {
	.poll_in = uart_etsoc_dw_apb_poll_in,
	.poll_out = uart_etsoc_dw_apb_poll_out,
	.err_check = uart_etsoc_dw_apb_err_check,
#ifdef CONFIG_UART_USE_RUNTIME_CONFIGURE
	.configure = uart_etsoc_dw_apb_configure,
	.config_get = uart_etsoc_dw_apb_config_get,
#endif
};

#define UART_ETSOC_DW_APB_INIT(n)                                             \
	static struct uart_etsoc_dw_apb_data uart_etsoc_dw_apb_data_##n;     \
	static const struct uart_etsoc_dw_apb_config uart_etsoc_dw_apb_cfg_##n = { \
		.base = DT_INST_REG_ADDR(n),                                  \
		.sys_clk_freq = DT_INST_PROP_OR(n, clock_frequency, 0),      \
	};                                                                   \
	DEVICE_DT_INST_DEFINE(n,                                              \
			      uart_etsoc_dw_apb_init,                           \
			      NULL,                                            \
			      &uart_etsoc_dw_apb_data_##n,                     \
			      &uart_etsoc_dw_apb_cfg_##n,                      \
			      PRE_KERNEL_1,                                    \
			      CONFIG_SERIAL_INIT_PRIORITY,                    \
			      &uart_etsoc_dw_apb_driver_api);

DT_INST_FOREACH_STATUS_OKAY(UART_ETSOC_DW_APB_INIT)
