/* Auto-generated from gen_classifiers.py — do not edit. */
#ifndef TEST_VECTORS_GEN_H
#define TEST_VECTORS_GEN_H

/* Iris: 4 features = sepal_len, sepal_w, petal_len, petal_w (cm) */
#define IRIS_N_FEATURES 4
#define IRIS_N_VECTORS  5
static const float iris_test_vectors[IRIS_N_VECTORS][IRIS_N_FEATURES] = {
    { 5.10f, 3.50f, 1.40f, 0.20f },  /* class 0 (setosa) */
    { 7.00f, 3.20f, 4.70f, 1.40f },  /* class 1 (versicolor) */
    { 6.30f, 3.30f, 6.00f, 2.50f },  /* class 2 (virginica) */
    { 5.00f, 3.00f, 1.60f, 0.20f },  /* class 0 (setosa) */
    { 6.60f, 3.00f, 4.40f, 1.40f },  /* class 1 (versicolor) */
};
static const int iris_expected[IRIS_N_VECTORS] = { 0, 1, 2, 0, 1 };

/* Wine: 13 features, 3 classes (cultivars) */
#define WINE_N_FEATURES 13
#define WINE_N_VECTORS  5
static const float wine_test_vectors[WINE_N_VECTORS][WINE_N_FEATURES] = {
    { 14.2300f, 1.7100f, 2.4300f, 15.6000f, 127.0000f, 2.8000f, 3.0600f, 0.2800f, 2.2900f, 5.6400f, 1.0400f, 3.9200f, 1065.0000f },  /* class 0 */
    { 12.3300f, 1.1000f, 2.2800f, 16.0000f, 101.0000f, 2.0500f, 1.0900f, 0.6300f, 0.4100f, 3.2700f, 1.2500f, 1.6700f, 680.0000f },  /* class 1 */
    { 12.8600f, 1.3500f, 2.3200f, 18.0000f, 122.0000f, 1.5100f, 1.2500f, 0.2100f, 0.9400f, 4.1000f, 0.7600f, 1.2900f, 630.0000f },  /* class 2 */
    { 13.5600f, 1.7100f, 2.3100f, 16.2000f, 117.0000f, 3.1500f, 3.2900f, 0.3400f, 2.3400f, 6.1300f, 0.9500f, 3.3800f, 795.0000f },  /* class 0 */
    { 12.0800f, 2.0800f, 1.7000f, 17.5000f, 97.0000f, 2.2300f, 2.1700f, 0.2600f, 1.4000f, 3.3000f, 1.2700f, 2.9600f, 710.0000f },  /* class 1 */
};
static const int wine_expected[WINE_N_VECTORS] = { 0, 1, 2, 0, 1 };

#endif
