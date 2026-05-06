# Whisper Encoder LayerNorm Rows Audit

- Graph: `encoder`
- Selected nodes: `9`
- Audited tiles: `27`
- Tile rows: `512`
- All pass: `True`
- Max abs diff: `1.52587891e-05`
- Mean wait per node: `0.146480 s`

| index | node | rows | tiles | max abs | wait s | pass |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `/encoder/blocks.0/attn_ln/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.147398 | True |
| 1 | `/encoder/blocks.0/mlp_ln/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.146269 | True |
| 2 | `/encoder/blocks.1/attn_ln/LayerNormalization` | 1500 | 3 | 7.62939453e-06 | 0.146886 | True |
| 3 | `/encoder/blocks.1/mlp_ln/LayerNormalization` | 1500 | 3 | 6.19888306e-06 | 0.144607 | True |
| 4 | `/encoder/blocks.2/attn_ln/LayerNormalization` | 1500 | 3 | 5.7220459e-06 | 0.146226 | True |
| 5 | `/encoder/blocks.2/mlp_ln/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.146749 | True |
| 6 | `/encoder/blocks.3/attn_ln/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.146662 | True |
| 7 | `/encoder/blocks.3/mlp_ln/LayerNormalization` | 1500 | 3 | 1.52587891e-05 | 0.146802 | True |
| 8 | `/encoder/ln_post/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.146722 | True |
