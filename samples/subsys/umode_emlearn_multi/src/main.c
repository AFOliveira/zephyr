/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * umode_emlearn_multi — single ELF, multi-program, single kernelLaunch.
 *
 * One Zephyr-U ELF runs FOUR different classifiers concurrently on four
 * different ET-SoC1 shires under a SINGLE kernelLaunch with
 * shire_mask = 0xF.  Each shire's lead hart picks its program based on
 * its shire-id (CSR 0xcd0 >> 6) and writes its 5 predictions into its
 * own 60-byte slot of a shared 240-byte host result buffer:
 *
 *     shire 0 → out[0..14]    iris-tree   (iris_tree_predict)
 *     shire 1 → out[15..29]   iris-rf     (iris_rf_predict)
 *     shire 2 → out[30..44]   wine-rf     (wine_rf_predict)
 *     shire 3 → out[45..59]   iris-tree   (re-run for symmetry)
 *
 * Result buffer layout (15 int32 per slot, only first 5 are predictions,
 * remainder is left at zero):
 *
 *   bytes [   0 ..  59) -> shire 0
 *   bytes [  60 .. 119) -> shire 1
 *   bytes [ 120 .. 179) -> shire 2
 *   bytes [ 180 .. 239) -> shire 3
 *
 * Per-shire stack carving is done by entry.S when
 * CONFIG_AIFOUNDRY_PER_SHIRE_STACK=y (set in this sample's prj.conf):
 * each shire gets its own 4096-byte slice of z_main_stack so the four
 * mains do not race on stack writes.  As a fallback, main() also
 * dispatches by shire-id from C — that path is what runs if the entry.S
 * patch is absent (e.g. on an earlier branch).
 */

#include <stdint.h>
#include <stddef.h>

#include <zephyr/aifoundry/runtime.h>

#include "test_vectors_gen.h"
#include "iris_tree.h"
#include "iris_rf.h"
#include "wine_rf.h"

#define SLOT_BYTES   60
#define N_SLOTS      4
#define BUF_BYTES    (SLOT_BYTES * N_SLOTS)
#define SLOT_INTS    (SLOT_BYTES / sizeof(int32_t))   /* 15 */

static inline unsigned int my_shire_id(void)
{
	uint64_t v;
	__asm__ volatile("csrr %0, 0xcd0" : "=r"(v));
	return (unsigned int)((v >> 6) & 0x1F);
}

static int32_t *my_result_slot(void)
{
	void *args = aifoundry_kernel_args();
	if (args == NULL) {
		return NULL;
	}
	uint64_t out_dev = *(volatile uint64_t *)args;
	if (out_dev == 0) {
		return NULL;
	}
	int32_t *base = (int32_t *)(uintptr_t)out_dev;
	return &base[my_shire_id() * SLOT_INTS];
}

/* Shared helper for main_a / main_d (both run iris_tree on the same 5
 * vectors).  Splitting the per-shire entry points keeps the C-level
 * dispatcher in main() trivial and lets the linker/optimizer share or
 * inline as it sees fit. */
static void run_iris_tree(int32_t *out)
{
	for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
		out[i] = iris_tree_predict(iris_test_vectors[i],
					   IRIS_N_FEATURES);
	}
}

void main_a(void)
{
	aifoundry_log("multi: shire 0 (iris-tree) start\n");
	int32_t *out = my_result_slot();
	if (out) {
		run_iris_tree(out);
		aifoundry_log("multi: shire 0 (iris-tree) preds %d %d %d %d %d\n",
			      (int)out[0], (int)out[1], (int)out[2],
			      (int)out[3], (int)out[4]);
	}
}

void main_b(void)
{
	aifoundry_log("multi: shire 1 (iris-rf) start\n");
	int32_t *out = my_result_slot();
	if (out) {
		for (unsigned i = 0; i < IRIS_N_VECTORS; i++) {
			out[i] = iris_rf_predict(iris_test_vectors[i],
						 IRIS_N_FEATURES);
		}
		aifoundry_log("multi: shire 1 (iris-rf) preds %d %d %d %d %d\n",
			      (int)out[0], (int)out[1], (int)out[2],
			      (int)out[3], (int)out[4]);
	}
}

void main_c(void)
{
	aifoundry_log("multi: shire 2 (wine-rf) start\n");
	int32_t *out = my_result_slot();
	if (out) {
		for (unsigned i = 0; i < WINE_N_VECTORS; i++) {
			out[i] = wine_rf_predict(wine_test_vectors[i],
						 WINE_N_FEATURES);
		}
		aifoundry_log("multi: shire 2 (wine-rf) preds %d %d %d %d %d\n",
			      (int)out[0], (int)out[1], (int)out[2],
			      (int)out[3], (int)out[4]);
	}
}

void main_d(void)
{
	aifoundry_log("multi: shire 3 (iris-tree-2) start\n");
	int32_t *out = my_result_slot();
	if (out) {
		run_iris_tree(out);
		aifoundry_log("multi: shire 3 (iris-tree-2) preds %d %d %d %d %d\n",
			      (int)out[0], (int)out[1], (int)out[2],
			      (int)out[3], (int)out[4]);
	}
}

/*
 * C-level dispatcher.  In this POC entry.S only sets up the per-shire
 * stack and still calls into the standard z_prep_c/main() path — there
 * is no asm-level jump table.  We dispatch on shire-id here.  This also
 * means the sample runs (with shared stack) even if entry.S has not
 * been patched yet, which is useful as a regression check.
 */
int main(void)
{
	switch (my_shire_id()) {
	case 0: main_a(); break;
	case 1: main_b(); break;
	case 2: main_c(); break;
	case 3: main_d(); break;
	default:
		aifoundry_log("multi: shire %u (no program assigned)\n",
			      my_shire_id());
		break;
	}
	aifoundry_kernel_exit(0);
}
