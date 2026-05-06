# Whisper Native On-Chip Plan

Goal: move from the current auditable hybrid path to a full native ET-SoC1
Whisper Tiny EN execution path.

## Current audited path

`tools/run_whisper_e2e_audit.py` is audio-to-text, but not full native:

- Host: WAV decode/resample, log-mel features, ONNX encoder, decoder control
  flow, decoder scalar/non-logits graph work, tokenizer, text decode.
- ET-SoC1: final decoder LayerNorm plus final logits projection/argmax for
  every greedy decode step.
- Audit: every silicon token choice is checked against the ONNXRuntime decoder
  for the same token step before the next token is accepted.  In argmax-only
  mode the full logits vector is not fetched; the gate is token sequence match,
  text match, LayerNorm error, and per-tile argmax/value agreement.

This gives a mathematically comparable silicon path for token selection, but
it is still host-orchestrated.

The current real demo entry point is:

`../tools/run_whisper_silicon_demo.py`

It runs the validated seven-shire tail path and fails if the ET-SoC1 token
sequence diverges from host ONNXRuntime.

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

## Completed native steps

1. Device-side argmax for logits tiles is implemented and validated on
   `esperanto-soc6`.
2. The final decoder LayerNorm is now computed on ET-SoC1 before logits argmax.
3. All 13 decoder LayerNorm nodes have a real ONNX tensor audit on ET-SoC1 at
   decoder step 3.  Max abs error is `7e-06`.
4. The seven vocab tiles can launch from one ET runtime process across shires
   `0..6`, avoiding seven independent host launcher processes per decode step.
5. The 16-hart split smoke passes: even harts do VPU-heavy work and odd harts
   do scalar LayerNorm-shaped work in one kernel with explicit barriers/cache
   handling.
6. Encoder conv1 and conv2 have real ONNX tensor audits by lowering each Conv
   to an equivalent MatMul.  Both pass against ONNXRuntime on ET-SoC1.
7. Encoder data movement ops have real ONNX tensor audits for Reshape, Concat,
   Unsqueeze, Split, and Transpose.  The large final Concat nodes are tiled and
   pass exactly.
8. Encoder LayerNorm rows have a real ONNX tensor audit for all 9 encoder
   LayerNorm nodes.  All pass; the worst observed tile is about `1.53e-05`.
9. Encoder scalar Add/Mul/Div/Erf/Softmax kernels have partial real ONNX tensor
   coverage through selected node index 52.  The important fix is that long
   encoder Softmax must be row-owned by harts; cross-hart row reduction is not
   reliable for this path.

## Remaining native steps

1. Stop relying on one host launch per tile.  Build a resident or batched graph
   executor that keeps tensors on device between audited kernels.
2. Merge the already audited decoder MatMul families with those scalar nodes so
   one device-resident decoder-step executor can produce the next token without
   host ONNXRuntime in the loop.
3. Persist the self-attention KV cache in the 16 MiB argument arena.  Cross
   attention K/V is too large in FP32, so it needs paging or FP16/INT8 storage.
4. Move encoder blocks after the decoder-step executor is stable.  The encoder
   output K/V cache is about 17.6 MiB in FP32, so full-native encoder execution
   also needs paging or reduced-precision cache storage.
5. Replace host tokenizer/text decode only after token IDs are device-native and
   audited.  Text decode is not performance critical, so this is last.

Native success criteria:

- Same token IDs as host ONNXRuntime for a fixed WAV.
- Per-node FP32 allclose for every moved node, or explicit INT8/INT4 audit
  gates if quantized.
- Device-side token/s includes all moved work, not only logits MatMul.
