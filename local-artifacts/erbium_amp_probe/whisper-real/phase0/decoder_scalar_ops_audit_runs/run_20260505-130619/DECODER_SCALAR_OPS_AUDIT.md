# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `1`
- All pass: `True`
- Max abs diff: `7.21812248e-05`
- Mean silicon wait: `0.000408 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/attn/Softmax` | 224 | 7.21812248e-05 | 0.000408 | 11882 | True |
