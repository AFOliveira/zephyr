# Whisper Native On-Chip Plan

Goal: move from the current auditable hybrid path to a full native ET-SoC1
Whisper Tiny EN execution path.

## Current audited path

`tools/run_whisper_e2e_audit.py` is audio-to-text, but not full native:

- Host: WAV decode/resample, log-mel features, ONNX encoder, decoder control
  flow, decoder scalar/non-logits graph work, tokenizer, text decode.
- ET-SoC1: final decoder logits projection for every greedy decode step.
- Audit: every silicon logits vector is compared against the ONNXRuntime
  decoder logits for the same token step before the next token is accepted.

This gives a mathematically comparable silicon path for token selection, but
it is still host-orchestrated.

## Hart split hypothesis

One shire has 8 Minions and 16 harts.  The practical split for native Whisper
is:

- Even harts / thread 0: VPU-heavy MatMul, QKV, MLP, attention-score, attention
  value, and logits projection kernels.
- Odd harts / thread 1: scalar graph work such as LayerNorm reductions,
  Softmax max/sum/normalize, GELU approximation, residual Add/Mul, KV-cache
  bookkeeping, argmax, token control, and device-side validation.

Important correction: our current `flq2`/`fmadd.ps` FP32 VPU kernels have
already run with `ACTIVE_HARTS=16`, so the second hart is not unusable for all
vector work.  The known limitation is around tensor/scratchpad-style work, so
the split above is still the safer scheduler for a native graph executor.

## Silicon smoke test

The first native scheduling test is:

- Source: `../whisper_native_hart_split_smoke.c`
- Runner: `../tools/run_whisper_native_split_smoke.py`
- Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/native-split-smoke`

It runs both roles in one kernel:

- 8 even harts compute a Whisper-shaped FP32 VPU logits MatMul.
- 8 odd harts compute a LayerNorm-shaped scalar reduction and normalization.
- Barriers and explicit evicts are used between producer/consumer phases to
  handle the non-coherent L1D behavior.
- The dump summary checks hart participation, logits argmax vs host reference,
  logits max error, and LayerNorm-shaped max error.

## Next native steps

1. Add device-side argmax to the audited logits kernel so ET-SoC1 returns the
   selected token, not only the logits vector. Implemented locally on
   2026-05-05 as `--device-argmax-only` in `tools/run_whisper_e2e_audit.py`;
   silicon validation is pending Tailscale SSH re-auth to `esperanto-soc6`.
2. Move decoder LayerNorm nodes into the scalar odd-hart path and compare each
   node against ONNXRuntime tensors.
3. Move decoder MLP MatMuls and GELU into the same kernel family.
4. Move self-attention and cross-attention score/value kernels.
5. Persist KV cache in the 16 MiB argument arena instead of round-tripping it
   through host ONNXRuntime.
6. Move encoder blocks after the decoder-step executor is stable.
7. Only then remove host ONNXRuntime from the execution loop and keep it as an
   offline audit reference.

Native success criteria:

- Same token IDs as host ONNXRuntime for a fixed WAV.
- Per-node FP32 allclose for every moved node, or explicit INT8/INT4 audit
  gates if quantized.
- Device-side token/s includes all moved work, not only logits MatMul.
