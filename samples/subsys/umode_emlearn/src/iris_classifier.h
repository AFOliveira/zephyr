/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shape of emlearn C output for a DecisionTreeClassifier.  emlearn
 * generates a `classify()` function like the one in iris_classifier.c:
 * nested if/else over integer features with constant-folded
 * thresholds.  Small, branch-predictable, no heap, no runtime deps.
 */

#ifndef IRIS_CLASSIFIER_H
#define IRIS_CLASSIFIER_H

#include <stdint.h>

#define IRIS_N_FEATURES 3
#define IRIS_N_CLASSES  3

/* Features are integer-scaled (e.g. millimetres) to stay away from
 * minimal libc's lack of %f.  Order: petal_length, petal_width,
 * sepal_length. */
int32_t iris_classify(const int32_t features[IRIS_N_FEATURES]);

/* Human-readable class names. */
extern const char *const iris_class_names[IRIS_N_CLASSES];

#endif /* IRIS_CLASSIFIER_H */
