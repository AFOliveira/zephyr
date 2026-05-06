# Whisper block 0 MatMul silicon audit

Scope: real Whisper Tiny En encoder block 0 tensors exported from ONNXRuntime, executed on ET-SoC1 silicon through `esperanto-soc6`, staged under `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2`. Each row is stitched from tiled silicon outputs and compared to the ORT tensor for the same ONNX node.

| Piece | Shape | Active harts | Wait s | GOPS | max_abs vs ORT | allclose 1e-4 | Artifact |
|---|---:|---:|---:|---:|---:|---|---|
| `block0_qkv` | `1500x384x1152` | 16 | 1.772558 | 0.749 | 6.68e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_qkv_full_stitched_o3_unroll_20260504-105253` |
| `block0_attn_score_h0` | `1500x64x1500` | 16 | 0.998731 | 0.288 | 7.63e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h0_full_stitched_o3_unroll_20260504-105916` |
| `block0_attn_score_h1` | `1500x64x1500` | 16 | 0.997210 | 0.289 | 1.14e-05 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h1_full_stitched_o3_unroll_20260504-110356` |
| `block0_attn_score_h2` | `1500x64x1500` | 16 | 1.001705 | 0.288 | 4.77e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h2_full_stitched_o3_unroll_20260504-110720` |
| `block0_attn_score_h3` | `1500x64x1500` | 16 | 1.005231 | 0.287 | 2.38e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h3_full_stitched_o3_unroll_20260504-111034` |
| `block0_attn_score_h4` | `1500x64x1500` | 16 | 0.998239 | 0.289 | 1.53e-05 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h4_full_stitched_o3_unroll_20260504-111339` |
| `block0_attn_score_h5` | `1500x64x1500` | 16 | 0.994292 | 0.290 | 4.77e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_score_h5_full_stitched_o3_unroll_20260504-111709` |
| `block0_attn_value_h0` | `1500x1504x64` | 16 | 0.294600 | 0.980 | 4.29e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h0_full_stitched_o3_unroll_20260504-110235` |
| `block0_attn_value_h1` | `1500x1504x64` | 16 | 0.294301 | 0.981 | 3.1e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h1_full_stitched_o3_unroll_20260504-112023` |
| `block0_attn_value_h2` | `1500x1504x64` | 16 | 0.295181 | 0.978 | 9.54e-07 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h2_full_stitched_o3_unroll_20260504-112101` |
| `block0_attn_value_h3` | `1500x1504x64` | 16 | 0.294733 | 0.980 | 1.43e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h3_full_stitched_o3_unroll_20260504-112204` |
| `block0_attn_value_h4` | `1500x1504x64` | 16 | 0.295194 | 0.978 | 2.15e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h4_full_stitched_o3_unroll_20260504-112242` |
| `block0_attn_value_h5` | `1500x1504x64` | 16 | 0.295218 | 0.978 | 6.68e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_value_h5_full_stitched_o3_unroll_20260504-112321` |
| `block0_attn_out` | `1500x384x384` | 16 | 0.603699 | 0.733 | 2.03e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/block0_attn_out_full_stitched_o3_unroll_20260504-105641` |
| `mlp0` | `1500x384x1536` |  | 2.372748 | 0.746 | 3.81e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/mlp0_full_stitched_o3_unroll_20260504-104251` |
| `mlp2` | `1500x1536x384` | 16 | 1.806589 | 0.979 | 1.43e-06 | True | `local-artifacts/erbium_amp_probe/whisper-real/phase0/runs/mlp2_full_stitched_o3_unroll_20260504-104948` |

Result: 16 block-0 MatMul pieces audited; all pass = True.

Notes:
- This is an end-to-end-in-pieces audit of the dense and attention MatMul pieces in encoder block 0, not yet a complete native Whisper graph executor. Non-MatMul ops are still supplied by ORT/host for the piece boundaries.
- Attention value uses zero-padded K=1504 for the VPU loop; the extra columns/rows are zero, and the output is compared against the unpadded ORT result.
- Some launcher runs still report stream errors after valid output dumps; correctness is gated by per-hart slot completion plus full tensor diff against ORT.
