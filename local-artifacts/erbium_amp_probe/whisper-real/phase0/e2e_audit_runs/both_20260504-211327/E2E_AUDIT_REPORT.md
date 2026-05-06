# Whisper E2E Audit Report

- Audio: `local-artifacts/erbium_amp_probe/whisper-real/audio/openai_whisper_jfk_16k.wav`
- Mode: `both`
- Host-only text: ` And so my fellow`
- Hybrid text: ` And so my fellow`
- Token sequence match: True
- Text match: True
- Silicon logits allclose: True
- Silicon logits argmax match: True
- Max logits abs diff: 1.1920929e-05
- Silicon logits wait: 0.342201 s

Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder logits projection used for greedy token choice.
It is not yet a native full-graph ONNX executor on ET-SoC1.
