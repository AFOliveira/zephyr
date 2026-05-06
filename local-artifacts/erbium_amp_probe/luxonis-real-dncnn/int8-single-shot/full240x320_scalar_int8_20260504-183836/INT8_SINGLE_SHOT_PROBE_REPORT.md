# DnCNN3 INT8 Single-Shot Silicon Probe

- Kind: synthetic scalar INT8 full-size DnCNN-shaped probe
- Board host: `esperanto-soc6`
- Remote artifact root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/luxonis-real-dncnn/int8-single-shot/full240x320_scalar_int8_20260504-183836`
- Shape: `240x320x64`, layers `20`, harts `16`
- Ops counted: `102098534400`
- Kernel wait: `263.919000 s`
- Throughput: `0.386856 GOPS`
- Inference rate: `0.003789041 inference/s`
- Summary OK: `True`
- Active mask: `0xffff`, done count: `16`

This is not the real Luxonis INT8 model execution. It is the silicon feasibility probe for the single-shot INT8 path. The current available implementation is scalar INT8, not packed-int VPU, and therefore does not support the 10-20 s estimate.
