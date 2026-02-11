/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal Zephyr UART driver for ET-SOC1 DW_apb_uart.
 *
 * This programs the UART for 8N1 and enables FIFO + PTIME so the
 * THRE bit indicates FIFO full (Mode B). The poll_out path waits for
 * THRE == 0 before writing.
 */

#define DT_DRV_COMPAT aifoundry_etsoc_dw_apb_uart

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/sys_io.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(uart_etsoc_dw_apb, CONFIG_UART_LOG_LEVEL);

/* DW_apb_uart register indexes (for reg-shift addressing) */
#define UART_REG_RBR_THR_DLL 0x00
#define UART_REG_IER_DLH     0x01
#define UART_REG_IIR_FCR     0x02
#define UART_REG_LCR         0x03
#define UART_REG_MCR         0x04
#define UART_REG_LSR         0x05

#define UART_DLF_OFFSET 0xC0

/* LSR bits */
#define UART_LSR_DR   BIT(0)
#define UART_LSR_THRE BIT(5)

/* LCR bits */
#define UART_LCR_DLAB  BIT(7)
#define UART_LCR_DLS_8 (BIT(0) | BIT(1))

/* MCR bits */
#define UART_MCR_DTR BIT(0)
#define UART_MCR_RTS BIT(1)

/* FCR bits */
#define UART_FCR_FIFOE  BIT(0)
#define UART_FCR_RFIFOR BIT(1)
#define UART_FCR_XFIFOR BIT(2)

/* IER bits */
#define UART_IER_PTIME BIT(7)

struct uart_etsoc_dw_apb_config {
	uintptr_t base;
	uint32_t sys_clk_freq;
	uint32_t baudrate;
	uint32_t reg_shift;
};

struct uart_etsoc_dw_apb_data {
	struct uart_config uart_cfg;
};

static inline uintptr_t uart_reg(const struct uart_etsoc_dw_apb_config *cfg, uint32_t reg)
{
	return cfg->base + (reg << cfg->reg_shift);
}

static inline uintptr_t uart_dlf_reg(const struct uart_etsoc_dw_apb_config *cfg)
{
	uint32_t reg = UART_DLF_OFFSET >> cfg->reg_shift;

	return cfg->base + (reg << cfg->reg_shift);
}

static void uart_etsoc_dw_apb_set_baud(const struct uart_etsoc_dw_apb_config *cfg,
				       uint32_t baudrate)
{
	uint32_t divisor;
	uint32_t dlf;

	if (cfg->sys_clk_freq == 0U || baudrate == 0U) {
		return;
	}

	/* Divisor latch and fractional divisor (DLF) */
	divisor = cfg->sys_clk_freq / (16U * baudrate);
	dlf = ((((2U * cfg->sys_clk_freq) / baudrate) + 1U) / 2U) - (16U * divisor);

	sys_write32(UART_LCR_DLAB | UART_LCR_DLS_8, uart_reg(cfg, UART_REG_LCR));
	sys_write32(divisor & 0xffU, uart_reg(cfg, UART_REG_RBR_THR_DLL));
	sys_write32((divisor >> 8) & 0xffU, uart_reg(cfg, UART_REG_IER_DLH));
	sys_write32(dlf & 0x0fU, uart_dlf_reg(cfg));
	sys_write32(UART_LCR_DLS_8, uart_reg(cfg, UART_REG_LCR));
}

static int uart_etsoc_dw_apb_poll_in(const struct device *dev, unsigned char *c)
{
	const struct uart_etsoc_dw_apb_config *cfg = dev->config;
	uint32_t lsr = sys_read32(uart_reg(cfg, UART_REG_LSR));

	if ((lsr & UART_LSR_DR) == 0U) {
		return -1;
	}

	*c = (unsigned char)(sys_read32(uart_reg(cfg, UART_REG_RBR_THR_DLL)) & 0xffU);
	return 0;
}

static void uart_etsoc_dw_apb_poll_out(const struct device *dev, unsigned char c)
{
	const struct uart_etsoc_dw_apb_config *cfg = dev->config;

	/* Mode B: THRE==1 means FIFO full. Wait for space. */
	while ((sys_read32(uart_reg(cfg, UART_REG_LSR)) & UART_LSR_THRE) != 0U) {
		;
	}

	sys_write32((uint32_t)c, uart_reg(cfg, UART_REG_RBR_THR_DLL));
}

static int uart_etsoc_dw_apb_err_check(const struct device *dev)
{
	ARG_UNUSED(dev);
	return 0;
}

#ifdef CONFIG_UART_USE_RUNTIME_CONFIGURE
static int uart_etsoc_dw_apb_configure(const struct device *dev, const struct uart_config *cfg)
{
	struct uart_etsoc_dw_apb_data *data = dev->data;
	const struct uart_etsoc_dw_apb_config *dev_cfg = dev->config;

	if (cfg->parity != UART_CFG_PARITY_NONE || cfg->stop_bits != UART_CFG_STOP_BITS_1 ||
	    cfg->data_bits != UART_CFG_DATA_BITS_8 || cfg->flow_ctrl != UART_CFG_FLOW_CTRL_NONE) {
		return -ENOTSUP;
	}

	uart_etsoc_dw_apb_set_baud(dev_cfg, cfg->baudrate);
	data->uart_cfg = *cfg;
	return 0;
}

static int uart_etsoc_dw_apb_config_get(const struct device *dev, struct uart_config *cfg)
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

	/* Basic 8N1 setup and FIFO enable. */
	sys_write32(UART_MCR_DTR | UART_MCR_RTS, uart_reg(cfg, UART_REG_MCR));
	uart_etsoc_dw_apb_set_baud(cfg, cfg->baudrate);
	sys_write32(UART_FCR_FIFOE | UART_FCR_RFIFOR | UART_FCR_XFIFOR,
		    uart_reg(cfg, UART_REG_IIR_FCR));
	/* Enable PTIME to switch THRE meaning to FIFO-full when FIFO is enabled. */
	sys_write32(UART_IER_PTIME, uart_reg(cfg, UART_REG_IER_DLH));

	data->uart_cfg.baudrate = cfg->baudrate;
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

#define UART_ETSOC_DW_APB_INIT(n)                                                                  \
	static struct uart_etsoc_dw_apb_data uart_etsoc_dw_apb_data_##n;                           \
	static const struct uart_etsoc_dw_apb_config uart_etsoc_dw_apb_cfg_##n = {                 \
		.base = DT_INST_REG_ADDR(n),                                                       \
		.sys_clk_freq = DT_INST_PROP_OR(n, clock_frequency, 0),                            \
		.baudrate = DT_INST_PROP_OR(n, current_speed, 115200),                             \
		.reg_shift = DT_INST_PROP_OR(n, reg_shift, 0),                                     \
	};                                                                                         \
	DEVICE_DT_INST_DEFINE(n, uart_etsoc_dw_apb_init, NULL, &uart_etsoc_dw_apb_data_##n,        \
			      &uart_etsoc_dw_apb_cfg_##n, PRE_KERNEL_1,                            \
			      CONFIG_SERIAL_INIT_PRIORITY, &uart_etsoc_dw_apb_driver_api);

DT_INST_FOREACH_STATUS_OKAY(UART_ETSOC_DW_APB_INIT)
