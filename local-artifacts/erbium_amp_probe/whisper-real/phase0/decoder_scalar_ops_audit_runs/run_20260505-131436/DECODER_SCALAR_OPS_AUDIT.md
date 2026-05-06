# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `2`
- All pass: `False`
- Max abs diff: `0.000333115458`
- Mean silicon wait: `0.000682 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/cross_attn/Softmax` | 1500 | 4.29097563e-05 | 0.000654 | 26754 | True |
| 1 | `Softmax` | `/0/cross_attn/Softmax_1` | 1500 | 0.000333115458 | 0.000711 | 26719 | False |
