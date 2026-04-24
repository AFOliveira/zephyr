/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * emlearn-style inference demo.
 *
 * Runs a small decision-tree classifier (iris_classifier.c — the
 * shape emlearn emits for sklearn DecisionTreeClassifier) over a
 * handful of test vectors, prints each prediction, and exits.
 *
 * Target-specific details:
 *   - etsoc1_minion_umode: Zephyr runs in U-mode under MachineMinion;
 *     "exit" is SYSCALL_RETURN_FROM_KERNEL (the one pre-existing
 *     cm-umode syscall).  No OSKERN additions on the MM side.
 *   - erbium_minion: Zephyr-M runs natively; "exit" is main() return.
 *
 * Either way, the classifier itself and all scaffolding is pure
 * portable C.
 */

#include <stdint.h>
#include <zephyr/sys/printk.h>

#ifdef CONFIG_RISCV_RUN_IN_UMODE
#include <isa/common/syscall.h>
#endif

#include "iris_classifier.h"

/* A few canned inputs spanning the three classes.  Values are
 * integer millimetres (petal_length, petal_width, sepal_length). */
static const int32_t test_vectors[][IRIS_N_FEATURES] = {
	{ 14,  2, 51 },   /* expected: setosa       */
	{ 45, 15, 63 },   /* expected: versicolor   */
	{ 60, 25, 72 },   /* expected: virginica    */
	{ 32, 10, 55 },   /* expected: versicolor   */
	{ 55, 20, 67 },   /* expected: virginica    */
};
#define N_VECTORS (sizeof(test_vectors) / sizeof(test_vectors[0]))

int main(void)
{
	printk("emlearn-app: start (%u test vectors, %u classes)\n",
	       (unsigned)N_VECTORS, (unsigned)IRIS_N_CLASSES);

	for (unsigned i = 0; i < N_VECTORS; i++) {
		const int32_t *v = test_vectors[i];
		int32_t pred = iris_classify(v);

		printk("  [%u] f=(%d, %d, %d) -> class %d (%s)\n",
		       i, (int)v[0], (int)v[1], (int)v[2],
		       (int)pred, iris_class_names[pred]);
	}

	printk("emlearn-app: done, exiting kernel\n");

#ifdef CONFIG_RISCV_RUN_IN_UMODE
	/* Zephyr-U under a host (MachineMinion on ET-SoC1): ecall out.
	 * The only syscall this sample makes — all 5 inference calls
	 * above are pure computation. */
	(void)syscall(SYSCALL_RETURN_FROM_KERNEL, 0, KERNEL_RETURN_SUCCESS, 0);
	for (;;) {
		__asm__ volatile("nop");
	}
#endif
	/* Zephyr-M (e.g. Erbium): main() returns, bg_thread terminates,
	 * kernel idles.  Simulator exits on quiescence. */
	return 0;
}
