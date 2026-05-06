# Whisper E2E and DnCNN Path A/B Silicon Report

Date: 2026-05-04

Board host: `esperanto-soc6`

Allowed remote root:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2`

## Whisper E2E Status

Whisper is complete for the current auditable hybrid path: audio to text runs
end-to-end, and the ET-SoC1 silicon logits projection matches host ONNXRuntime
well enough to produce exactly the same token sequence and final text.

Input audio:
`local-artifacts/erbium_amp_probe/whisper-real/audio/openai_whisper_jfk_16k.wav`

Audit artifact:
`local-artifacts/erbium_amp_probe/whisper-real/phase0/e2e_audit_runs/both_20260504-181738/e2e_audit_report.json`

Output text:
` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`

Key results:

- Token sequence match vs host ONNXRuntime: `true`
- Text match vs host ONNXRuntime: `true`
- Silicon logits allclose at 1e-4: `true`
- Silicon argmax match: `true`
- Max logits abs diff: `1.9073486328125e-05`
- Generated non-prompt tokens: `23`
- Host-only decode wall: `0.42539869697066024 s`
- Host-only tokens/s: `54.066926306514546`
- Hybrid decode wall: `656.4843640880426 s`
- Hybrid wall tokens/s: `0.03503510709192674`
- Silicon logits-only wait: `1.5749789400000003 s`
- Silicon logits-only tokens/s: `14.603369871091735`

Scope note: this is not full native on-chip Whisper. Host still does WAV/log-mel,
encoder, non-logits decoder graph work, tokenization, and text decode. ET-SoC1
does the final decoder logits projection for each generated step.

## DnCNN Path A - Tiled FP32

Path A is complete and audited for the 20-tile 240x320 full image path.

Audit artifact:
`local-artifacts/erbium_amp_probe/luxonis-real-dncnn/tiled-fp32/runs/a3_tiled64_halo20_f9t_pfw_align128_20260504-182946/full_audit_report.json`

Geometry:

- Tiles: `20`
- Halo: `20`
- Input tile shapes: `84x84`, `84x104`, `104x84`, `104x104`, `68x84`, `68x104`
- Full output: `240x320`

Key results:

- All tile audits OK: `true`
- Stitched output allclose vs full ORT: `true`
- Stitched max abs vs full ORT: `8.940696716308594e-07`
- Total kernel wait: `95.02435 s`
- Useful full-image ops: `102098534400`
- Executed tiled ops: `229721702400`
- Useful GOPS: `1.0744460172576817`
- Executed GOPS: `2.4175035388297843`
- Inference/s from kernel wait: `0.0105236192093898`

## DnCNN Path B - INT8 Single-Shot

The host quantization quality gate is plausible, but the real packed-int VPU
single-shot kernel does not exist in this tree yet. I did not find an existing
on-silicon INT8 DnCNN implementation that can be audited against INT8 ORT.

Best host INT8 ORT variant from the existing audit table:

- Variant: `dncnn3-int8-s10`
- Quantization: QDQ Conv, per-channel weights, UINT8 activations, INT8 weights
- PSNR vs FP32 ORT: `40.823520 dB`
- Worst tile PSNR: `40.320462 dB`
- Passes 35 dB PSNR gate: `true`

Silicon probe artifact:
`local-artifacts/erbium_amp_probe/luxonis-real-dncnn/int8-single-shot/full240x320_scalar_int8_20260504-183836/silicon_report.json`

Remote full dump:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/luxonis-real-dncnn/int8-single-shot/full240x320_scalar_int8_20260504-183836/dump.bin`

Probe shape:

- Full single-shot image: `240x320`
- Channels: `64`
- Layers: `20`
- Active harts: `16`
- Counted ops: `102098534400`

Probe results:

- Summary OK: `true`
- Active mask: `0xffff`
- Done count: `16`
- Kernel wait: `263.919 s`
- Throughput: `0.38685556704898094 GOPS`
- Inference/s: `0.00378904133465192`

Scope note: this silicon run is a synthetic scalar INT8 full-size DnCNN-shaped
throughput probe. It is not the real Luxonis INT8 model and it is not packed-int
VPU. The result says the current scalar INT8 path does not support the optimistic
10-20 s estimate; that estimate still depends on writing and auditing a real
packed-int VPU implementation.
