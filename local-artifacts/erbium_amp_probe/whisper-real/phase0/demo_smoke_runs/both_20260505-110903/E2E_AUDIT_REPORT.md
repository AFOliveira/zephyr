# Whisper E2E Audit Report

- Audio: `local-artifacts/erbium_amp_probe/whisper-real/audio/openai_whisper_jfk_16k.wav`
- Mode: `both`
- Host-only text: ` And`
- Hybrid text: ` And`
- Token sequence match: True
- Text match: True
- Silicon tail LayerNorm: True
- Silicon logits allclose: not fetched (argmax-only)
- Silicon logits argmax match: True
- Max logits abs diff: not fetched (argmax-only)
- Silicon logits wait: 0.240568 s

Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder LayerNorm and logits argmax used for greedy token choice.
It is not yet a native full-graph ONNX executor on ET-SoC1.
