# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `3`
- All pass: `False`
- Max abs diff: `2.15947318`
- Mean silicon wait: `0.001606 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/Add` | 384 | 0.069568634 | 0.001701 | 0 | False |
| 1 | `Add` | `/0/attn/query/Add` | 384 | 2.15947318 | 0.001411 | 0 | False |
| 2 | `Add` | `/0/attn/value/Add` | 384 | 0.94324708 | 0.001704 | 0 | False |
