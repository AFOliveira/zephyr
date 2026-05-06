# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `8`
- All pass: `True`
- Max abs diff: `9.56654549e-05`
- Mean silicon wait: `0.000686 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Div` | `/0/mlp/1/Div` | 1536 | 4.76837158e-07 | 0.000686 | 12906 | True |
| 1 | `Erf` | `/0/mlp/1/Erf` | 1536 | 9.56654549e-05 | 0.000653 | 25768 | True |
| 2 | `Mul` | `/0/mlp/1/Mul` | 1536 | 0 | 0.000711 | 5806 | True |
| 3 | `Mul` | `/0/mlp/1/Mul_1` | 1536 | 0 | 0.000653 | 5896 | True |
| 4 | `Div` | `/1/mlp/1/Div` | 1536 | 4.76837158e-07 | 0.000710 | 12984 | True |
| 5 | `Erf` | `/1/mlp/1/Erf` | 1536 | 9.53674316e-05 | 0.000712 | 25592 | True |
| 6 | `Mul` | `/1/mlp/1/Mul` | 1536 | 0 | 0.000709 | 5768 | True |
| 7 | `Mul` | `/1/mlp/1/Mul_1` | 1536 | 0 | 0.000654 | 5881 | True |
