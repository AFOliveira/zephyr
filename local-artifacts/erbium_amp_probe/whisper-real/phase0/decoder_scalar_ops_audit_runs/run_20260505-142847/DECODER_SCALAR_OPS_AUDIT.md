# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `8`
- All pass: `True`
- Max abs diff: `1.37090683e-06`
- Mean silicon wait: `0.000661 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Div` | `/2/mlp/1/Div` | 1536 | 2.38418579e-07 | 0.000650 | 12836 | True |
| 1 | `Erf` | `/2/mlp/1/Erf` | 1536 | 1.31130219e-06 | 0.000653 | 30208 | True |
| 2 | `Mul` | `/2/mlp/1/Mul` | 1536 | 0 | 0.000709 | 5877 | True |
| 3 | `Mul` | `/2/mlp/1/Mul_1` | 1536 | 0 | 0.000654 | 5847 | True |
| 4 | `Div` | `/3/mlp/1/Div` | 1536 | 2.38418579e-07 | 0.000651 | 12867 | True |
| 5 | `Erf` | `/3/mlp/1/Erf` | 1536 | 1.37090683e-06 | 0.000658 | 30407 | True |
| 6 | `Mul` | `/3/mlp/1/Mul` | 1536 | 0 | 0.000656 | 5767 | True |
| 7 | `Mul` | `/3/mlp/1/Mul_1` | 1536 | 0 | 0.000654 | 5815 | True |
