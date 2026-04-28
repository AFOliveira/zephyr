/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * emlearn demo — runs three sklearn-trained classifiers (DecisionTree
 * on iris, RandomForest on iris, RandomForest on wine) on canned test
 * vectors and verifies each prediction against its host-computed
 * baseline.
 *
 * Same source compiles for:
 *   - etsoc1_minion_umode  (Zephyr-U on cm-umode + production firmware)
 *   - erbium_minion        (Zephyr-M, native UART)
 *
 * All target-specific behavior is hidden behind <aifoundry/runtime.h>:
 *   aifoundry_log               → et-trace on U, printk on M
 *   aifoundry_publish_results() → mallocDevice writeback on U, no-op on M
 *   AIFOUNDRY_KERNEL_EXIT(rc)   → cm_umode ecall on U, return rc on M
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/aifoundry/runtime.h>

#include "test_vectors_gen.h"
#include "iris_tree.h"
#include "iris_rf.h"
#include "wine_rf.h"

__attribute__((noinline))
static void emlearn_mismatch_halt(void)
{
	for (;;) {
		__asm__ volatile("nop");
	}
}

int main(void)
{
	int32_t iris_tree_pred[IRIS_N_VECTORS];
	int32_t iris_rf_pred[IRIS_N_VECTORS];
	int32_t wine_rf_pred[WINE_N_VECTORS];
	int mismatches = 0;

	aifoundry_log("emlearn-demo: start\n");
	aifoundry_log("emlearn-demo: iris (DecisionTree, 4 features, 5 vectors)\n");
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		iris_tree_pred[i] = iris_tree_predict(iris_test_vectors[i],
						      IRIS_N_FEATURES);
		if (iris_tree_pred[i] != iris_expected[i]) {
			mismatches++;
		}
		aifoundry_log("  iris-tree[%u] -> %d (expected %d)\n",
			      i, (int)iris_tree_pred[i], iris_expected[i]);
	}

	aifoundry_log("emlearn-demo: iris (RandomForest, 10 trees)\n");
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		iris_rf_pred[i] = iris_rf_predict(iris_test_vectors[i],
						  IRIS_N_FEATURES);
		if (iris_rf_pred[i] != iris_expected[i]) {
			mismatches++;
		}
		aifoundry_log("  iris-rf[%u]   -> %d (expected %d)\n",
			      i, (int)iris_rf_pred[i], iris_expected[i]);
	}

	aifoundry_log("emlearn-demo: wine (RandomForest, 10 trees, 13 features)\n");
	for (unsigned i = 0; i < WINE_N_VECTORS; i++) {
		wine_rf_pred[i] = wine_rf_predict(wine_test_vectors[i],
						  WINE_N_FEATURES);
		if (wine_rf_pred[i] != wine_expected[i]) {
			mismatches++;
		}
		aifoundry_log("  wine-rf[%u]   -> %d (expected %d)\n",
			      i, (int)wine_rf_pred[i], wine_expected[i]);
	}

	aifoundry_log("emlearn-demo: done, mismatches=%d\n", mismatches);

	if (mismatches != 0) {
		emlearn_mismatch_halt();
	}

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
	aifoundry_publish_results(out_packed, sizeof(out_packed));

	aifoundry_kernel_exit(mismatches);
}
