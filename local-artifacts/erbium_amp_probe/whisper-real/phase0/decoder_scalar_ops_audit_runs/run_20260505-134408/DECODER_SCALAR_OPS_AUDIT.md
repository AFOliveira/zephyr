# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `6`
- All pass: `True`
- Max abs diff: `5.0291419e-08`
- Mean silicon wait: `0.000680 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/0/cross_attn/Softmax` | 1500 | 1.11758709e-08 | 0.000756 | 29884 | True |
| 1 | `Softmax` | `/0/cross_attn/Softmax_1` | 1500 | 1.86264515e-08 | 0.000654 | 29822 | True |
| 2 | `Softmax` | `/0/cross_attn/Softmax_2` | 1500 | 5.0291419e-08 | 0.000649 | 29790 | True |
| 3 | `Softmax` | `/0/cross_attn/Softmax_3` | 1500 | 2.79396772e-08 | 0.000653 | 29825 | True |
| 4 | `Softmax` | `/0/cross_attn/Softmax_4` | 1500 | 7.4505806e-09 | 0.000716 | 29938 | True |
| 5 | `Softmax` | `/0/cross_attn/Softmax_5` | 1500 | 2.14204192e-08 | 0.000654 | 30014 | True |
