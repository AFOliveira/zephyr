# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `1`
- All pass: `False`
- Max abs diff: `0.39080286`
- Mean silicon wait: `0.001479 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/attn/Softmax` | 224 | 0.39080286 | 0.001479 | 0 | False |
