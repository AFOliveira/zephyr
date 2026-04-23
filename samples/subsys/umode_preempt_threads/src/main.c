/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Zephyr-U preempt-threads demo.
 *
 * Two worker threads each print 3 lines with k_sleep(K_MSEC(100)) between
 * them, while the main thread prints periodically.  After ~1 s main exits
 * the kernel via SYSCALL_RETURN_FROM_KERNEL.
 *
 * The printk output goes to the __umode_printk_ring buffer (see
 * arch/riscv/core-umode/printk_stub.c); dump it after simulation ends
 * to see the interleave.
 */

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <isa/common/syscall.h>

#define STACK_SZ 1024
#define PRIO     5

K_THREAD_STACK_DEFINE(worker_a_stack, STACK_SZ);
K_THREAD_STACK_DEFINE(worker_b_stack, STACK_SZ);

static struct k_thread worker_a;
static struct k_thread worker_b;

static void worker(void *p1, void *p2, void *p3)
{
	const char *name = (const char *)p1;
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	for (int i = 0; i < 3; i++) {
		printk("%s tick %d\n", name, i);
		k_sleep(K_MSEC(100));
	}
	printk("%s done\n", name);
}

int main(void)
{
	printk("umode_preempt_threads: start\n");

	k_thread_create(&worker_a, worker_a_stack, STACK_SZ,
			worker, (void *)"A", NULL, NULL,
			PRIO, 0, K_NO_WAIT);
	k_thread_create(&worker_b, worker_b_stack, STACK_SZ,
			worker, (void *)"B", NULL, NULL,
			PRIO, 0, K_NO_WAIT);

	for (int i = 0; i < 10; i++) {
		printk("main %d\n", i);
		k_sleep(K_MSEC(50));
	}

	printk("umode_preempt_threads: main exiting\n");
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);

	for (;;) {
		__asm__ volatile("nop");
	}
	return 0;
}
