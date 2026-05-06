# Whisper E2E Audit Report

- Audio: `/usr/share/sounds/speech-dispatcher/test.wav`
- Mode: `both`
- Host-only text: ``
- Hybrid text: ``
- Token sequence match: True
- Text match: True
- Silicon logits allclose: True
- Silicon logits argmax match: True
- Max logits abs diff: 1.62124634e-05
- Silicon logits wait: 0.242483 s

Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder logits projection used for greedy token choice.
It is not yet a native full-graph ONNX executor on ET-SoC1.
