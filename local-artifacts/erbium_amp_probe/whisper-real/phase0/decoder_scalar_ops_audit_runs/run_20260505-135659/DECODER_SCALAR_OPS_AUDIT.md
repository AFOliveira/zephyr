# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- All pass: `True`
- Max abs diff: `1.11758709e-07`
- Mean silicon wait: `0.000616 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Softmax` | `/3/attn/Softmax` | 224 | 5.96046448e-08 | 0.000552 | 11099 | True |
| 1 | `Softmax` | `/3/attn/Softmax_1` | 224 | 1.11758709e-07 | 0.000549 | 10786 | True |
| 2 | `Softmax` | `/3/attn/Softmax_2` | 224 | 9.31322575e-09 | 0.000552 | 10886 | True |
| 3 | `Softmax` | `/3/attn/Softmax_3` | 224 | 0 | 0.000553 | 10934 | True |
| 4 | `Softmax` | `/3/attn/Softmax_4` | 224 | 5.96046448e-08 | 0.000553 | 10912 | True |
| 5 | `Softmax` | `/3/attn/Softmax_5` | 224 | 7.4505806e-09 | 0.000400 | 11039 | True |
| 6 | `Softmax` | `/3/cross_attn/Softmax` | 1500 | 1.67638063e-08 | 0.000654 | 30025 | True |
| 7 | `Softmax` | `/3/cross_attn/Softmax_1` | 1500 | 5.21540642e-08 | 0.000710 | 29882 | True |
| 8 | `Softmax` | `/3/cross_attn/Softmax_2` | 1500 | 5.0291419e-08 | 0.000742 | 29866 | True |
| 9 | `Softmax` | `/3/cross_attn/Softmax_3` | 1500 | 2.79396772e-08 | 0.000653 | 29945 | True |
| 10 | `Softmax` | `/3/cross_attn/Softmax_4` | 1500 | 3.16649675e-08 | 0.000710 | 30064 | True |
| 11 | `Softmax` | `/3/cross_attn/Softmax_5` | 1500 | 2.98023224e-08 | 0.000763 | 29949 | True |
