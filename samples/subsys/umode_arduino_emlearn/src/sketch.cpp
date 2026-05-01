/*
 * Copyright (c) 2026 AIFoundry
 * SPDX-License-Identifier: Apache-2.0
 *
 * Arduino-style sketch driving an emlearn iris-tree classifier.
 * Same source compiles for two AIFoundry boards:
 *
 *   - erbium_minion        (Zephyr-M, Shakti UART, real `Serial`)
 *   - etsoc1_minion_umode  (Zephyr-U, virtual UART -> et-trace ring)
 *
 * The portability seam is the Arduino API itself: `Serial.print` is
 * a `arduino::ZephyrSerial` instance bound at compile time to whichever
 * UART device the board's variant overlay picks (`zephyr,user::serials`).
 * `delay()` and `millis()` route through ArduinoCore-zephyr's upstream
 * implementation built atop Zephyr kernel APIs.  On the U-mode build
 * those map to k_busy_wait fallbacks (no scheduler, no real clock);
 * see <zephyr/aifoundry/runtime.h> for the cleaner HAL primitives we
 * exposed for non-Arduino code paths.
 *
 * The kernel-launch contract on ET-SoC1 expects each launch to exit
 * via `aifoundry_kernel_exit`.  The `loop()` here runs a fixed number
 * of iterations and then returns the kernel cleanly — overrides Arduino's
 * "loop forever" assumption because the ET-SoC1 launcher kills the
 * kernel after `--kernel_launch_timeout` seconds and we'd lose the
 * exit code.  Iteration count is `CONFIG_AIFOUNDRY_ARDUINO_LOOP_COUNT`.
 */

#include <Arduino.h>

#include <zephyr/aifoundry/runtime.h>
#include <zephyr/kernel.h>

/* iris_tree.h provides static-inline-only definitions plus a single
 * top-level `int32_t iris_tree_predict(...)` — both are valid C++ as
 * written, no `extern "C"` needed. */
#include "iris_tree.h"
#include "test_vectors_gen.h"

#ifndef CONFIG_AIFOUNDRY_ARDUINO_LOOP_COUNT
#define CONFIG_AIFOUNDRY_ARDUINO_LOOP_COUNT 3
#endif

static int g_loop_count;

void setup()
{
	Serial.begin(115200);
	Serial.println("emlearn-arduino: setup");
	g_loop_count = 0;
}

void loop()
{
	for (int i = 0; i < IRIS_N_VECTORS; i++) {
		int32_t pred = iris_tree_predict(iris_test_vectors[i],
						 IRIS_N_FEATURES);
		Serial.print("iris[");
		Serial.print(i);
		Serial.print("] -> ");
		Serial.println(pred);
	}

	g_loop_count++;
	if (g_loop_count >= CONFIG_AIFOUNDRY_ARDUINO_LOOP_COUNT) {
		Serial.println("emlearn-arduino: done");
		aifoundry_kernel_exit(0);
	}
}
