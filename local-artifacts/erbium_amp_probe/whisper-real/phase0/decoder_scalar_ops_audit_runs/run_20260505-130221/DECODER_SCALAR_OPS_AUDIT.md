# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `4`
- All pass: `False`
- Max abs diff: `0.996661425`
- Mean silicon wait: `0.000551 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/attn/Softmax` | 224 | 0.60919714 | 0.000551 | 11991 | False |
| 1 | `Softmax` | `/0/attn/Softmax_1` | 224 | 0.675493479 | 0.000551 | 12114 | False |
| 2 | `Softmax` | `/0/attn/Softmax_2` | 224 | 0.996661425 | 0.000552 | 11967 | False |
| 3 | `Softmax` | `/0/attn/Softmax_3` | 224 | 0.701344073 | 0.000550 | 12139 | False |
