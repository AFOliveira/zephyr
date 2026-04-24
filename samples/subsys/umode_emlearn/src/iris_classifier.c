/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hand-written decision tree classifier in the shape emlearn emits:
 *   - pure C, no heap, no floats (integer thresholds)
 *   - nested if/else over feature indices
 *   - each leaf returns the class label directly
 *
 * This is what `emlearn.convert(sklearn_tree).save(..., 'inline')`
 * would produce for a 3-feature iris-like tree.  The thresholds below
 * are illustrative, not fit to real data.
 */

#include "iris_classifier.h"

const char *const iris_class_names[IRIS_N_CLASSES] = {
	"setosa",
	"versicolor",
	"virginica",
};

int32_t iris_classify(const int32_t features[IRIS_N_FEATURES])
{
	/* features[0] = petal_length (mm)
	 * features[1] = petal_width  (mm)
	 * features[2] = sepal_length (mm)
	 */
	if (features[0] < 25) {
		/* Short petals — dominated by setosa. */
		return 0;
	}
	if (features[1] < 17) {
		if (features[0] < 50) {
			return 1; /* versicolor */
		}
		if (features[2] < 62) {
			return 1;
		}
		return 2;         /* virginica */
	}
	if (features[0] < 48) {
		if (features[1] < 16) {
			return 1;
		}
		return 2;
	}
	return 2;
}
