# Whisper Audio-to-Text Audit Plan

## Goal

Make Whisper Tiny EN auditable against a host-only reference while moving ET-SoC1
onto the inference path incrementally.

The reference is host-only ONNXRuntime using the Luxonis encoder and decoder
ONNX files plus the `openai/whisper-tiny.en` tokenizer/feature extractor.

The implemented hybrid path is:

1. Host reads WAV and builds Whisper log-mel features.
2. Host runs the ONNX encoder and non-logits decoder graph work.
3. ET-SoC1 computes the final decoder logits projection for each greedy decode
   step.
4. Host compares the ET-SoC1 logits against ONNXRuntime logits for that same
   step, uses the silicon argmax for token choice, and decodes text with the
   tokenizer.

This is audio-to-text with ET-SoC1 on the token-decision math path. It is not
yet a native full-graph ONNX executor.

## Audit Gates

- Input reproducibility: record audio hash, feature hash, ONNX hashes, tokenizer
  id, board host, remote artifact root, and active hart count.
- Tensor correctness: every silicon logits tensor must pass
  `np.allclose(rtol=1e-4, atol=1e-4)` against the ONNXRuntime logits tensor.
- Token correctness: silicon logits argmax must match host logits argmax at each
  step.
- End result: host-only and hybrid token sequences and decoded text must match.
- Timing: record host encoder time, host decoder wall time, silicon kernel wait,
  and total hybrid wall time.

## Implemented Commands

Run the host-only and hybrid comparison:

```sh
python3 local-artifacts/erbium_amp_probe/whisper-real/tools/run_whisper_e2e_audit.py \
  --mode both \
  --audio /usr/share/sounds/speech-dispatcher/test.wav \
  --max-new-tokens 4 \
  --active-harts 16 \
  --timeout 300
```

Artifacts are written under:

```text
local-artifacts/erbium_amp_probe/whisper-real/phase0/e2e_audit_runs/
```

The remote ET-SoC1 artifacts stay under:

```text
/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/e2e-audit/
```

## Latest Silicon Result

Run:

```text
local-artifacts/erbium_amp_probe/whisper-real/phase0/e2e_audit_runs/both_20260504-172043
```

Summary:

- Host-only tokens: `[50257, 50258, 50358, 50362, 50256]`
- Hybrid tokens: `[50257, 50258, 50358, 50362, 50256]`
- Host-only text: empty string
- Hybrid text: empty string
- Token sequence match: true
- Text match: true
- Silicon logits allclose: true
- Silicon logits argmax match: true
- Max logits absolute diff: `1.621246337890625e-05`
- Silicon logits kernel wait across 4 decoder steps: `0.24248321 s`
- Active harts: 16

The default local WAV is a smoke-test audio clip that Whisper classifies as no
transcription after the forced prompt. Use a real speech WAV for a semantic demo;
the same harness and audit gates apply.

## Next Expansion

The next step is to replace more host-computed decoder work with ET-SoC1 kernels:

1. Decoder MLP/QKV/dense MatMuls already have audited shape-family kernels.
2. Add LayerNorm, Softmax, GELU, Add, Slice/Concat, and KV-cache update kernels
   into the host-scheduled graph.
3. Move the encoder MatMul/attention families from piecewise audit into the
   same scheduled execution path.
4. Once FP32 host-scheduled full graph passes, repeat the same harness against a
   quantized host-only ONNX reference.
