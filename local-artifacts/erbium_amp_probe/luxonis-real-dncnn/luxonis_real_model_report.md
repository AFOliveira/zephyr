# Luxonis Real Model Ingest And DnCNN Silicon Probe

Date: 2026-05-02

Board host: `esperanto-soc6`

Remote root:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`

Local root:
`local-artifacts/erbium_amp_probe/luxonis-real-dncnn`

## What Was Pulled

I used the Luxonis model cards as the source of truth and downloaded both the
RVC deployable packages and the raw ONNX artifacts exposed by the Luxonis model
API.

Downloaded ONNX artifacts are under:

`local-artifacts/erbium_amp_probe/luxonis-models/downloads/onnx`

Key graph inventory:

| Model | Variant | ONNX nodes | Main ops | Notes |
|---|---|---:|---|---|
| DnCNN3 | 240x320 / 480x640 | 40 | 20 Conv, 19 Relu, 1 Sub | Best first Erbium target |
| Depth Anything V2 | ViT-S 252x336 / 420x560 | 1108 | MatMul, Conv, LayerNorm, Softmax, Resize | Needs transformer/importer work |
| Whisper Tiny EN | encoder | 257 | MatMul, LayerNorm, Softmax, Conv | Needs transformer/importer work |
| Whisper Tiny EN | decoder | 476 | MatMul, Slice, Softmax, LayerNorm | Needs KV-cache/runtime work |
| Foundation Stereo | 640x416 / 1280x800 | 37817 / 58447 | Conv, MatMul, ScatterND, dynamic shape ops | Too large for first manual port |

The Luxonis RVC downloads are `.dlc` packages targeting Qualcomm/RVC4, not ET
executables.  They are useful metadata, but they do not run on ET-SOC1 without
a DLC importer/runtime.

## DnCNN3 Real-Weight Kernel

I built a first Erbium kernel using the real Luxonis DnCNN3 weights/topology:

- 20 convolution layers
- 64 hidden channels
- FP32 input/weights/activations
- ONNX behavior: zero padding, ReLU after every hidden conv, final
  `enhanced_image = image - residual`
- hidden weights pre-packed as `{output channel, kernel point, input channel}`
  for Erbium VPU dot products

Main source:

`dncnn3_luxonis_real_vpu.c`

The full 240x320 static ELF was built successfully:

`dncnn3_luxonis_real_static_16h.elf`

Its program segment is about 64 MiB, which proves the host runtime can load a
large BSS-backed real-model image through `loadCode()`.  The current direct
kernel did not finish within 300 seconds, so the first successful silicon probes
use smaller tiles with the same real weights/topology.

## Silicon Results

Successful runs:

| Kernel | Tile | Harts | Wait s | Ops | GOPS | Correctness |
|---|---:|---:|---:|---:|---:|---|
| `dncnn3_luxonis_real_tile1_static_1h.elf` | 1x1 | 1 | 0.006441 | 1,329,408 | 0.206390 | bit-identical to reference |
| `dncnn3_luxonis_real_tile4_static_1h.elf` | 4x4 | 1 | 0.279096 | 21,270,528 | 0.076212 | max abs error 4.73e-7, mean abs error 1.30e-7 |

Timed out:

| Kernel | Tile | Harts | Timeout | Result |
|---|---:|---:|---:|---|
| `dncnn3_luxonis_real_tile4_static_4h.elf` | 4x4 | 4 | 60 s | timeout |
| `dncnn3_luxonis_real_tile16_static_4h.elf` | 16x16 | 4 | 120 s | timeout |
| `dncnn3_luxonis_real_tile64_static_16h.elf` | 64x64 | 16 | 120 s | timeout |
| `dncnn3_luxonis_real_static_16h.elf` | 240x320 | 16 | 300 s | timeout |

The timeout path triggers a host runtime abort bug: after `waitForStream()`
times out and aborts, the launcher often segfaults in runtime/event cleanup.
This is a host-side cleanup issue after timeout, not a board reset.

## Interpretation

This confirms we now used the actual Luxonis model artifacts, not just synthetic
model-shaped kernels.

It also shows the direct manual DnCNN port is not yet the right performance
path.  The reduced DnCNN-shaped VPU kernel was fast because it used 16 channels,
5 layers, fused 3x3 accumulation, and OC2 reuse.  Real DnCNN3 is 64 channels and
20 layers.  The current direct real-weight kernel falls back to scalar boundary
work too often and pays too much per-dot control overhead.

The next useful optimization path is:

1. Use larger tiles with halo, not whole-image per-layer global barriers.
2. Add a real 64-channel fused 3x3x64 VPU accumulator.
3. Compute multiple output channels per input load, more than OC2 if register
   pressure allows.
4. Keep weights resident/prefetched per layer rather than sweeping all packed
   weights blindly.
5. Only after DnCNN is healthy, move to Depth/Whisper; those need MatMul,
   LayerNorm, Softmax, and graph import/codegen work.

## Artifacts

Checksums are in:

`artifact_manifest_sha256.tsv`
