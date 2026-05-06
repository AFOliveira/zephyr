# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- All pass: `True`
- Max abs diff: `1.00582838e-07`
- Mean silicon wait: `0.000618 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/1/attn/Softmax` | 224 | 5.96046448e-08 | 0.000553 | 11023 | True |
| 1 | `Softmax` | `/1/attn/Softmax_1` | 224 | 5.96046448e-08 | 0.000552 | 11024 | True |
| 2 | `Softmax` | `/1/attn/Softmax_2` | 224 | 5.96046448e-08 | 0.000552 | 11162 | True |
| 3 | `Softmax` | `/1/attn/Softmax_3` | 224 | 1.00582838e-07 | 0.000551 | 10949 | True |
| 4 | `Softmax` | `/1/attn/Softmax_4` | 224 | 5.96046448e-08 | 0.000552 | 10964 | True |
| 5 | `Softmax` | `/1/attn/Softmax_5` | 224 | 1.49011612e-08 | 0.000405 | 11049 | True |
| 6 | `Softmax` | `/1/cross_attn/Softmax` | 1500 | 1.86264515e-08 | 0.000658 | 30004 | True |
| 7 | `Softmax` | `/1/cross_attn/Softmax_1` | 1500 | 1.14087015e-08 | 0.000716 | 30097 | True |
| 8 | `Softmax` | `/1/cross_attn/Softmax_2` | 1500 | 8.19563866e-08 | 0.000755 | 29858 | True |
| 9 | `Softmax` | `/1/cross_attn/Softmax_3` | 1500 | 3.53902578e-08 | 0.000711 | 29726 | True |
| 10 | `Softmax` | `/1/cross_attn/Softmax_4` | 1500 | 2.14204192e-08 | 0.000653 | 30044 | True |
| 11 | `Softmax` | `/1/cross_attn/Softmax_5` | 1500 | 3.7252903e-08 | 0.000757 | 29900 | True |
