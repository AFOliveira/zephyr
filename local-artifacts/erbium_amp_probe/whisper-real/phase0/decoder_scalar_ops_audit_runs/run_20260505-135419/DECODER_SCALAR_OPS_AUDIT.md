# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- All pass: `True`
- Max abs diff: `1.78813934e-07`
- Mean silicon wait: `0.000612 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/2/attn/Softmax` | 224 | 2.98023224e-08 | 0.000555 | 10900 | True |
| 1 | `Softmax` | `/2/attn/Softmax_1` | 224 | 1.78813934e-07 | 0.000550 | 11025 | True |
| 2 | `Softmax` | `/2/attn/Softmax_2` | 224 | 1.1920929e-07 | 0.000407 | 10894 | True |
| 3 | `Softmax` | `/2/attn/Softmax_3` | 224 | 1.1920929e-07 | 0.000551 | 11119 | True |
| 4 | `Softmax` | `/2/attn/Softmax_4` | 224 | 1.1920929e-07 | 0.000549 | 10898 | True |
| 5 | `Softmax` | `/2/attn/Softmax_5` | 224 | 5.96046448e-08 | 0.000554 | 10948 | True |
| 6 | `Softmax` | `/2/cross_attn/Softmax` | 1500 | 2.14204192e-08 | 0.000703 | 29937 | True |
| 7 | `Softmax` | `/2/cross_attn/Softmax_1` | 1500 | 6.33299351e-08 | 0.000654 | 29914 | True |
| 8 | `Softmax` | `/2/cross_attn/Softmax_2` | 1500 | 5.58793545e-09 | 0.000752 | 29911 | True |
| 9 | `Softmax` | `/2/cross_attn/Softmax_3` | 1500 | 3.53902578e-08 | 0.000709 | 29925 | True |
| 10 | `Softmax` | `/2/cross_attn/Softmax_4` | 1500 | 4.09781933e-08 | 0.000708 | 29942 | True |
| 11 | `Softmax` | `/2/cross_attn/Softmax_5` | 1500 | 4.09781933e-08 | 0.000652 | 29878 | True |
