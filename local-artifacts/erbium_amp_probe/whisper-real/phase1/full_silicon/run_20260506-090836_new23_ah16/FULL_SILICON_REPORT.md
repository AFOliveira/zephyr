# Full Whisper Silicon Report

- Result: `PASS`
- Scope: log-mel features -> encoder cross-cache -> decoder token IDs on ET-SoC1; host ONNXRuntime is audit only
- Host text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
- Host tokens: `[50257, 50258, 50358, 50362, 843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, 460, 466, 329, 345, 1265, 644, 345, 460, 466, 329, 534, 1499, 50256]`
- Silicon tokens: `[50257, 50258, 50358, 50362, 843, 523, 616, 5891, 3399, 1265, 407, 644, 534, 1499, 460, 466, 329, 345, 1265, 644, 345, 460, 466, 329, 534, 1499, 50256]`
- Encoder wait seconds: `148.186`
- Decoder wait seconds: `79.7042`
- Total kernel wait seconds: `227.8902`
- Generated token/s total kernel: `0.10092579672140356`
- Decoder-only generated token/s: `0.28856697639522133`
- Cross K max abs vs host FP16 oracle: `0.0009765625`
- Cross V max abs vs host FP16 oracle: `0.001953125`
- Remote work: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/full-silicon/run_20260506-090836_new23_ah16`
