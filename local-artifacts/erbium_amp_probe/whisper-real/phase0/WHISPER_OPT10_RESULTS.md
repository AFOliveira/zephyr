# Whisper optimization split 10

All runs use real ONNXRuntime-exported tensors, the VPU FP32 MatMul kernel, ET-SoC1 silicon on `esperanto-soc6`, and a full stitched output audit against ORT.

| Variant | Shape | Tile rows | Harts | Wait s | GOPS | max_abs | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| `opt10_mlp2_tr128_ah16` | `1500x1536x384` | 128 | 16 | 1.790357 | 0.988 | 1.43e-06 | True |
| `opt10_mlp2_tr256_ah16` | `1500x1536x384` | 256 | 16 | 1.683841 | 1.051 | 1.43e-06 | True |
| `opt10_mlp2_tr512_ah16` | `1500x1536x384` | 512 | 16 | 1.660121 | 1.066 | 1.43e-06 | True |
| `opt10_mlp2_tr128_ah8` | `1500x1536x384` | 128 | 8 | 3.139395 | 0.564 | 1.43e-06 | True |
| `opt10_mlp2_tr256_ah8` | `1500x1536x384` | 256 | 8 | 3.016251 | 0.587 | 1.43e-06 | True |
| `opt10_mlp2_tr512_ah8` | `1500x1536x384` | 512 | 8 | 3.070468 | 0.576 | 1.43e-06 | True |
| `opt10_qkv_tr256_ah16` | `1500x384x1152` | 256 | 16 | 1.710595 | 0.776 | 6.68e-06 | True |
| `opt10_scoreh0_tr256_ah16` | `1500x64x1500` | 256 | 16 | 0.957264 | 0.301 | 7.63e-06 | True |
| `opt10_scoreh0_tr512_ah16` | `1500x64x1500` | 512 | 16 | 0.947814 | 0.304 | 7.63e-06 | True |
| `opt10_valueh0_tr256_ah16` | `1500x1504x64` | 256 | 16 | 0.270395 | 1.068 | 4.29e-06 | True |

Findings:
- 16 active harts are clearly better than 8 for MLP2; 8 harts roughly halves throughput.
- Larger row tiles improve MLP2 and attention score by reducing launch/pullback overhead.
- Attention score remains the weak shape: `1500x64x1500` only reaches about 0.304 GOPS with this row-tiled kernel. It likely needs column blocking or a different data layout, not just larger row tiles.
- Attention value and MLP2 are the best current paths, both around 1.07 GOPS with larger tiles.
