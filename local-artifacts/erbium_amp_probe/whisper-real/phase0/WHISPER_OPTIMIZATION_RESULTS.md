# Whisper optimization results

Full-node silicon audit reports: 100. Optimization variants: 84. Passing optimization variants: 84.

| Family | Best variant | Shape | Tile rows | GOPS | Wait s | max_abs |
|---|---|---:|---:|---:|---:|---:|
| mlp2 | `opt100_i033_mlp2_fullk_tr768_ah16` | `1500x1536x384` | 768 | 1.275 | 1.387407 | 1.07e-06 |
| mlp0 | `opt100_i029_mlp0_fullk_tr1024_ah16` | `1500x384x1536` | 1024 | 0.898 | 1.971431 | 3.1e-06 |
| qkv | `opt100_i043_qkv_fullk_tr1280_ah16` | `1500x384x1152` | 1280 | 0.900 | 1.475239 | 6.2e-06 |
| attn_out | `optfill_attnout_fullk_tr1024_ah16` | `1500x384x384` | 1024 | 0.963 | 0.459378 | 2.09e-06 |
| score | `opt100_i008_block0_score_h5_fullk_tr1280_ah16` | `1500x64x1500` | 1280 | 0.314 | 0.917543 | 4.77e-06 |
| value | `opt100_i003_block0_value_h4_fullk_tr512_ah16` | `1500x1504x64` | 512 | 1.284 | 0.224890 | 1.91e-06 |

What actually helped:
- `VPU_ACCUM_FULL_K=1` is the main win. It keeps VPU lane accumulators live across the full K loop and reduces horizontal reductions from one per 16-float chunk to one per output.
- 16 active harts beat 8 active harts for the tested MLP2 shape.
- Larger row tiles help until cache/memory pressure pushes back; best tile size is shape-dependent.

What did not help:
- `VPU_DOT_COLS=4` was slower than the 2-column path.
- DnCNN-style `-falign-*` plus `-fno-tree-loop-*` flags were slower for this inline-VPU MatMul code.

Remaining bottleneck:
- Attention score (`1500x64x1500`) remains the weak shape at about 0.314 GOPS. It likely needs a different kernel layout, such as column blocking or a kernel that reuses Q rows across more K/V columns without spilling.
