# Whisper Decoder LayerNorm Audit

- Decoder step: `3`
- Selected LayerNorm nodes: `1`
- All pass: `True`
- Max LayerNorm abs diff: `1e-06`
- Mean silicon wait: `0.000552 s`

| index | node | max abs | wait s | pass |
| ---: | --- | ---: | ---: | --- |
| 0 | `/0/attn_ln/LayerNormalization` | 1e-06 | 0.000552 | True |
