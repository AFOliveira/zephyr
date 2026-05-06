# Resident INT8 Tail Silicon Report

- Result: `PASS`
- Step: `4`
- Host argmax: `523` ` so`
- Silicon argmax: `523`
- Kernel wait seconds: `0.0604378`
- Ops/s: `659050329.4295954`
- LN max abs vs host raw-INT8 ref: `1.1e-05`
- Remote work: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/resident-int8-tail/run_20260506-082143_step4_ah16`

This run stages one 16 MiB runtime region plus one 64 MiB raw INT8 weight region and reads the final projection weights from the resident region on ET-SoC1.
