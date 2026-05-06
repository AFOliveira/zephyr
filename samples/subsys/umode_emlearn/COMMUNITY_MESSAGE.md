Exciting update: it is now possible for more people to try Erbium-style workloads through ET-SoC1 boards.

If you do not have an ET-SoC1 board locally, you can still use the AI Foundry servers to run tests, experiment with kernels, and start benchmarking real workloads on shared hardware.

The first demo is intentionally small, but it shows the important piece working: the same emlearn example can run through the AIFoundry/ET HAL path and produce results on ET-SoC1, while also running as native Erbium M-mode code. That means the application code is no longer tied directly to one platform's firmware details.

For this first experiment, the demo runs emlearn classifiers and reports the prediction outputs. It is a starting point, not the final destination, but it gives us a concrete base for validating the HAL integration, comparing behavior across targets, and building more realistic benchmarks from here.
