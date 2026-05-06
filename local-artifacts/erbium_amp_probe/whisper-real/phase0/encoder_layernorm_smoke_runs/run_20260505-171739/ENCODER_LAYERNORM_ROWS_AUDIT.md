# Whisper Encoder LayerNorm Rows Audit

- Graph: `encoder`
- Selected nodes: `1`
- Audited tiles: `3`
- Tile rows: `512`
- All pass: `True`
- Max abs diff: `3.81469727e-06`
- Mean wait per node: `0.147283 s`

| index | node | rows | tiles | max abs | wait s | pass |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `/encoder/blocks.0/attn_ln/LayerNormalization` | 1500 | 3 | 3.81469727e-06 | 0.147283 | True |
