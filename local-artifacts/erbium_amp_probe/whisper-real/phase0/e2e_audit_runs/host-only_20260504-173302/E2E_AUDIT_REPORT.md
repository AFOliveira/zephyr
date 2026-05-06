# Whisper E2E Audit Report

- Audio: `local-artifacts/erbium_amp_probe/whisper-real/audio/openai_whisper_jfk_16k.wav`
- Mode: `host-only`
- Host-only text: ` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`

Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder logits projection used for greedy token choice.
It is not yet a native full-graph ONNX executor on ET-SoC1.
