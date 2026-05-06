# Resident INT8 Tail Silicon Report

- Result: `PASS`
- Step: `3`
- Host argmax: `843` ` And`
- Silicon argmax: `843`
- Kernel wait seconds: `0.0604801`
- Ops/s: `658589387.2529973`
- LN max abs vs host raw-INT8 ref: `1.5e-05`
- Remote work: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/resident-int8-tail/run_20260506-081846_step3_ah16`

This run stages one 16 MiB runtime region plus one 64 MiB raw INT8 weight region and reads the final projection weights from the resident region on ET-SoC1.
