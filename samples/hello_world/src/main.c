/*
 * Copyright (c) 2012-2014 Wind River Systems, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>

#ifdef CONFIG_UART_INTERRUPT_DRIVEN

static const char tx_msg[] = "Hello IRQ TX!\n";
static volatile int tx_pos;
static volatile bool tx_done;

static void uart_irq_callback(const struct device *dev, void *user_data)
{
	if (uart_irq_tx_ready(dev)) {
		if (tx_pos < (int)strlen(tx_msg)) {
			uart_fifo_fill(dev, (const uint8_t *)&tx_msg[tx_pos], 1);
			tx_pos++;
		} else {
			uart_irq_tx_disable(dev);
			tx_done = true;
		}
	}
}

#endif /* CONFIG_UART_INTERRUPT_DRIVEN */

int main(void)
{
	/* Poll-mode printf still works */
	printf("Poll: Hello World! %s\n", CONFIG_BOARD_TARGET);

#ifdef CONFIG_UART_INTERRUPT_DRIVEN
	const struct device *uart_dev = DEVICE_DT_GET(DT_NODELABEL(uart0));

	if (device_is_ready(uart_dev)) {
		tx_pos = 0;
		tx_done = false;

		uart_irq_callback_user_data_set(uart_dev, uart_irq_callback, NULL);
		uart_irq_tx_enable(uart_dev);

		/* Spin until ISR has sent all bytes */
		while (!tx_done) {
			;
		}
		printf("Poll: IRQ TX done\n");
	}
#endif

#ifdef CONFIG_SOC_ERBIUM_MINION
	__asm__ volatile(
		"fence\n"
		"lui a7, 0x1FEED\n"
		"csrw 0x8d0, a7\n"
	);
#endif

	return 0;
}
