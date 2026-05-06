# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- All pass: `True`
- Max abs diff: `2.98023224e-07`
- Mean silicon wait: `0.000612 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/attn/Softmax` | 224 | 1.1920929e-07 | 0.000552 | 10933 | True |
| 1 | `Softmax` | `/0/attn/Softmax_1` | 224 | 1.1920929e-07 | 0.000552 | 10928 | True |
| 2 | `Softmax` | `/0/attn/Softmax_2` | 224 | 5.96046448e-08 | 0.000551 | 10819 | True |
| 3 | `Softmax` | `/0/attn/Softmax_3` | 224 | 5.96046448e-08 | 0.000558 | 11064 | True |
| 4 | `Softmax` | `/0/attn/Softmax_4` | 224 | 5.96046448e-08 | 0.000553 | 11069 | True |
| 5 | `Softmax` | `/0/attn/Softmax_5` | 224 | 2.98023224e-07 | 0.000552 | 11094 | True |
| 6 | `Softmax` | `/0/cross_attn/Softmax` | 1500 | 1.11758709e-08 | 0.000652 | 29970 | True |
| 7 | `Softmax` | `/0/cross_attn/Softmax_1` | 1500 | 1.86264515e-08 | 0.000652 | 29971 | True |
| 8 | `Softmax` | `/0/cross_attn/Softmax_2` | 1500 | 5.0291419e-08 | 0.000654 | 29981 | True |
| 9 | `Softmax` | `/0/cross_attn/Softmax_3` | 1500 | 2.79396772e-08 | 0.000655 | 29816 | True |
| 10 | `Softmax` | `/0/cross_attn/Softmax_4` | 1500 | 7.4505806e-09 | 0.000653 | 29979 | True |
| 11 | `Softmax` | `/0/cross_attn/Softmax_5` | 1500 | 2.14204192e-08 | 0.000765 | 29971 | True |
