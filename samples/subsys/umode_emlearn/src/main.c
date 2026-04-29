/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * emlearn demo — runs three sklearn-trained classifiers (DecisionTree
 * on iris, RandomForest on iris, RandomForest on wine) on canned test
 * vectors.
 *
 * On ET-SoC1 U-mode this uses Vidas' ET surface directly: et_printf()
 * for logs and cache_ops_priv_evict_whole_l1_l2() for host-visible writes.
 * Exit is handled by the platform runtime after main() returns.
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/kernel.h>

#if defined(CONFIG_RISCV_RUN_IN_UMODE)
#include <etsoc/common/utils.h>
#include <etsoc/isa/cacheops-umode.h>
#else
#include <zephyr/sys/printk.h>
#endif

#include "test_vectors_gen.h"
#include "iris_tree.h"
#include "iris_rf.h"
#include "wine_rf.h"

#if defined(CONFIG_RISCV_RUN_IN_UMODE)
extern void *cm_umode_kernel_args;

#define emlearn_log(...) et_printf(__VA_ARGS__)

static void emlearn_publish_results(const void *src, size_t size)
{
	void *args = cm_umode_kernel_args;

	if ((args == NULL) || (args == (void *)~(uintptr_t)0)) {
		return;
	}

	uint64_t out_dev = *(volatile uint64_t *)args;
	if (out_dev == 0U) {
		return;
	}

	const uint8_t *s = (const uint8_t *)src;
	volatile uint8_t *d = (volatile uint8_t *)(uintptr_t)out_dev;

	for (size_t i = 0; i < size; i++) {
		d[i] = s[i];
	}

	(void)cache_ops_priv_evict_whole_l1_l2();
}
#else
#define emlearn_log(...) printk(__VA_ARGS__)

static void emlearn_publish_results(const void *src, size_t size)
{
	ARG_UNUSED(src);
	ARG_UNUSED(size);
}
#endif

int main(void)
{
	int32_t iris_tree_pred[IRIS_N_VECTORS];
	int32_t iris_rf_pred[IRIS_N_VECTORS];
	int32_t wine_rf_pred[WINE_N_VECTORS];

	ARG_UNUSED(iris_expected);
	ARG_UNUSED(wine_expected);

	emlearn_log("emlearn-demo: start\n");
	emlearn_log("emlearn-demo: iris (DecisionTree, 4 features, 5 vectors)\n");
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		iris_tree_pred[i] = iris_tree_predict(iris_test_vectors[i],
						      IRIS_N_FEATURES);
		emlearn_log("  iris-tree[%u] -> %d\n",
			    i, (int)iris_tree_pred[i]);
	}

	emlearn_log("emlearn-demo: iris (RandomForest, 10 trees)\n");
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		iris_rf_pred[i] = iris_rf_predict(iris_test_vectors[i],
						  IRIS_N_FEATURES);
		emlearn_log("  iris-rf[%u]   -> %d\n",
			    i, (int)iris_rf_pred[i]);
	}

	emlearn_log("emlearn-demo: wine (RandomForest, 10 trees, 13 features)\n");
	for (unsigned i = 0; i < WINE_N_VECTORS; i++) {
		wine_rf_pred[i] = wine_rf_predict(wine_test_vectors[i],
						  WINE_N_FEATURES);
		emlearn_log("  wine-rf[%u]   -> %d\n",
			    i, (int)wine_rf_pred[i]);
	}
	emlearn_log("emlearn-demo: done\n");

	/* Pack predictions into the host-allocated result buffer
	 * (silently no-op on M-mode where there's no host). */
	int32_t out_packed[IRIS_N_VECTORS * 2 + WINE_N_VECTORS];
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		out_packed[i] = iris_tree_pred[i];
		out_packed[IRIS_N_VECTORS + i] = iris_rf_pred[i];
	}
	for (unsigned i = 0; i < WINE_N_VECTORS; i++) {
		out_packed[2 * IRIS_N_VECTORS + i] = wine_rf_pred[i];
	}
	emlearn_publish_results(out_packed, sizeof(out_packed));

	return 0;
}
