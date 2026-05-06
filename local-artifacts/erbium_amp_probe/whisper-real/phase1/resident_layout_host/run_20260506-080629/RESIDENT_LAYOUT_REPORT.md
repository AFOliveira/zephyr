# Whisper Resident Layout Host Verification

- Bundle: `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/whisper-real/phase1/native_v1_bundles/bundle_20260505-205553`
- Result: `PASS`
- Weight region: 55.036 / 64.000 MiB used
- Runtime region: 15.040 / 16.000 MiB used
- FP32 vs dynamic INT8 token match in bundle: `True`

## Contract

This is the host-side proof for:

```text
16 MiB region + 64 MiB region is enough for:
  INT8 raw weights
  tiled/resident executor
  log-mel -> token IDs
  short/normal decode lengths
```

The verified layout does not embed ONNX files or FP32 weights on chip.
It uses raw INT8 weight blobs and a compact offset/scale manifest.

## Weight Region

| Item | Bytes | MiB |
|---|---:|---:|
| Raw INT8 weight files | 57640199 | 54.970 |
| Compact manifest | 68398 | 0.065 |
| Packed high-water | 57709038 | 55.036 |
| Free | 9399826 | 8.964 |

## Runtime Region

| Allocation | Offset | Bytes | MiB | Notes |
|---|---:|---:|---:|---|
| `control_block` | 0 | 65536 | 0.062 | host-filled run ABI and status |
| `input_mel_fp32` | 65536 | 960000 | 0.916 | 1x80x3000 log-mel input |
| `encoder_hidden_fp32_reusable` | 1025536 | 2304000 | 2.197 | used while building cross cache; reusable as scratch during decode |
| `cross_k_cache_int8` | 3329536 | 2304000 | 2.197 | persistent decoder cross-attention K cache |
| `cross_v_cache_int8` | 5633536 | 2304000 | 2.197 | persistent decoder cross-attention V cache |
| `cross_cache_scales_fp32` | 7937536 | 192 | 0.000 | per-layer/head K/V scales |
| `self_k_cache_fp32` | 7937728 | 1376256 | 1.312 | persistent decoder self-attention K cache |
| `self_v_cache_fp32` | 9313984 | 1376256 | 1.312 | persistent decoder self-attention V cache |
| `decoder_hidden_a_fp32` | 10690240 | 344064 | 0.328 | ping-pong hidden state buffer |
| `decoder_hidden_b_fp32` | 11034304 | 344064 | 0.328 | ping-pong hidden state buffer |
| `mlp_tile_fp32` | 11378368 | 1376256 | 1.312 | tiled MLP expansion buffer |
| `attention_scores_tile_fp32` | 12754624 | 1376256 | 1.312 | cross/self attention source-window tile |
| `matmul_accum_tile_fp32` | 14130880 | 1048576 | 1.000 | generic tiled GEMM accumulation scratch |
| `token_output_u32` | 15179456 | 896 | 0.001 | generated token IDs |
| `summary_status` | 15180352 | 65536 | 0.062 | timing, PMCs, audit status |
| `runtime_guard_slack` | 15245888 | 524288 | 0.500 | reserved guard for alignment and kernel-local metadata |

## Host Checks

- Every raw INT8 weight file was size-checked, SHA-checked, copied into a simulated 64 MiB region, and SHA-checked again from the packed region.
- The 16 MiB runtime arena was allocated with the same block sizes the resident executor needs and fully touched with deterministic patterns.
- The frozen bundle already shows FP32 ONNXRuntime and dynamic INT8 ONNXRuntime produce the same token sequence for the JFK sample.

## Important Caveat

FP32 cross-attention K/V does not fit in this 16 MiB runtime plan.  The passing plan uses compressed cross K/V (`int8`) or an equivalent tiled/recomputed cross-attention path.
