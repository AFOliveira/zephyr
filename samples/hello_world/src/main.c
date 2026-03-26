/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>

static void uart_isr(const struct device *dev, void *user_data)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);

	/* TX_FULL interrupt fired — signal test pass */
	__asm__ volatile(
		"fence\n"
		"lui a7, 0x1FEED\n"
		"csrw 0x8d0, a7\n"
	);
}

int main(void)
{
	const struct device *uart_dev = DEVICE_DT_GET(DT_NODELABEL(uart0));

	if (!device_is_ready(uart_dev)) {
		return -1;
	}

	/* Set up ISR and enable TX_FULL interrupt */
	uart_irq_callback_user_data_set(uart_dev, uart_isr, NULL);
	uart_irq_tx_enable(uart_dev);

	/* Write 16 chars to fill the TX FIFO */
	for (int i = 0; i < 16; i++) {
		uart_poll_out(uart_dev, 'A' + i);
	}

	/* If ISR doesn't fire, spin forever (test will timeout = fail) */
	while (1) {
		;
	}

	return 0;
}
