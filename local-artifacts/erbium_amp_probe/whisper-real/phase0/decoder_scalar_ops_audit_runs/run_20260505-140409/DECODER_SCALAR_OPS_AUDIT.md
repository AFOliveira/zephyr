# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `20`
- All pass: `True`
- Max abs diff: `0`
- Mean silicon wait: `0.000516 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/1/attn/Add` | 224 | 0 | 0.000405 | 2053 | True |
| 1 | `Add` | `/1/attn/Add_1` | 224 | 0 | 0.000552 | 2050 | True |
| 2 | `Add` | `/1/attn/Add_2` | 224 | 0 | 0.000551 | 2185 | True |
| 3 | `Add` | `/1/attn/Add_3` | 224 | 0 | 0.000552 | 1910 | True |
| 4 | `Add` | `/1/attn/Add_4` | 224 | 0 | 0.000550 | 2032 | True |
| 5 | `Add` | `/1/attn/Add_5` | 224 | 0 | 0.000404 | 2041 | True |
| 6 | `Add` | `/1/attn/out/Add` | 384 | 0 | 0.000550 | 2363 | True |
| 7 | `Add` | `/1/Add` | 384 | 0 | 0.000553 | 2369 | True |
| 8 | `Add` | `/1/cross_attn/query/Add` | 384 | 0 | 0.000546 | 2364 | True |
| 9 | `Add` | `/1/cross_attn/out/Add` | 384 | 0 | 0.000551 | 2351 | True |
| 10 | `Add` | `/1/Add_1` | 384 | 0 | 0.000551 | 2332 | True |
| 11 | `Add` | `/1/mlp/0/Add` | 1536 | 0 | 0.000652 | 5757 | True |
| 12 | `Add` | `/1/mlp/1/Add` | 1536 | 0 | 0.000660 | 5945 | True |
| 13 | `Add` | `/1/mlp/2/Add` | 384 | 0 | 0.000552 | 2362 | True |
| 14 | `Add` | `/1/Add_2` | 384 | 0 | 0.000551 | 2370 | True |
| 15 | `Add` | `/2/attn/query/Add` | 384 | 0 | 0.000407 | 2479 | True |
| 16 | `Add` | `/2/attn/value/Add` | 384 | 0 | 0.000355 | 2341 | True |
| 17 | `Add` | `/2/attn/Add` | 224 | 0 | 0.000417 | 2159 | True |
| 18 | `Add` | `/2/attn/Add_1` | 224 | 0 | 0.000551 | 2209 | True |
| 19 | `Add` | `/2/attn/Add_2` | 224 | 0 | 0.000405 | 2046 | True |
