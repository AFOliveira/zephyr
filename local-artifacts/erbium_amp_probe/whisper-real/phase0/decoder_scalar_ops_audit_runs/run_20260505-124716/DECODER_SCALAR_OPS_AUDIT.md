# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `3`
- All pass: `True`
- Max abs diff: `0`
- Mean silicon wait: `0.001554 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/Add` | 384 | 0 | 0.001608 | 0 | True |
| 1 | `Add` | `/0/attn/query/Add` | 384 | 0 | 0.001474 | 0 | True |
| 2 | `Add` | `/0/attn/value/Add` | 384 | 0 | 0.001579 | 0 | True |
