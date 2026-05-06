# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `3`
- All pass: `False`
- Max abs diff: `0`
- Mean silicon wait: `0.001473 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/Add` | 384 | 0 | 0.001478 | 0 | False |
| 1 | `Add` | `/0/attn/query/Add` | 384 | 0 | 0.001477 | 0 | False |
| 2 | `Add` | `/0/attn/value/Add` | 384 | 0 | 0.001464 | 0 | False |
