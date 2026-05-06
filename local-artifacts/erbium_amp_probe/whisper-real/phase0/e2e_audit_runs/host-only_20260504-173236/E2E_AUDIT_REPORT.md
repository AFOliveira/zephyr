# Whisper E2E Audit Report

- Audio: `local-artifacts/erbium_amp_probe/whisper-real/audio/wikimedia_desert_16.wav`
- Mode: `host-only`
- Host-only text: ``

Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder logits projection used for greedy token choice.
It is not yet a native full-graph ONNX executor on ET-SoC1.
