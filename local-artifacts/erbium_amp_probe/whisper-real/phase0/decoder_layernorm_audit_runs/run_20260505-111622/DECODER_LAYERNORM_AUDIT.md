# Whisper Decoder LayerNorm Audit

- Decoder step: `3`
- Selected LayerNorm nodes: `13`
- All pass: `True`
- Max LayerNorm abs diff: `7e-06`
- Mean silicon wait: `0.000518 s`

| index | node | max abs | wait s | pass |
| ---: | --- | ---: | ---: | --- |
| 0 | `/0/attn_ln/LayerNormalization` | 1e-06 | 0.000550 | True |
| 1 | `/0/cross_attn_ln/LayerNormalization` | 0 | 0.000405 | True |
| 2 | `/0/mlp_ln/LayerNormalization` | 2e-06 | 0.000552 | True |
| 3 | `/1/attn_ln/LayerNormalization` | 1e-06 | 0.000551 | True |
| 4 | `/1/cross_attn_ln/LayerNormalization` | 1e-06 | 0.000552 | True |
| 5 | `/1/mlp_ln/LayerNormalization` | 0 | 0.000545 | True |
| 6 | `/2/attn_ln/LayerNormalization` | 0 | 0.000551 | True |
| 7 | `/2/cross_attn_ln/LayerNormalization` | 5e-06 | 0.000563 | True |
| 8 | `/2/mlp_ln/LayerNormalization` | 1e-06 | 0.000405 | True |
| 9 | `/3/attn_ln/LayerNormalization` | 2e-06 | 0.000552 | True |
| 10 | `/3/cross_attn_ln/LayerNormalization` | 0 | 0.000551 | True |
| 11 | `/3/mlp_ln/LayerNormalization` | 0 | 0.000406 | True |
| 12 | `/ln/LayerNormalization` | 7e-06 | 0.000552 | True |
