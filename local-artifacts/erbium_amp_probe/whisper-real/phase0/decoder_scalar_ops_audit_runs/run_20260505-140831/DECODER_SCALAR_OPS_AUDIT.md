# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `20`
- All pass: `True`
- Max abs diff: `0`
- Mean silicon wait: `0.000511 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/2/attn/Add_3` | 224 | 0 | 0.000552 | 2028 | True |
| 1 | `Add` | `/2/attn/Add_4` | 224 | 0 | 0.000551 | 2044 | True |
| 2 | `Add` | `/2/attn/Add_5` | 224 | 0 | 0.000551 | 2047 | True |
| 3 | `Add` | `/2/attn/out/Add` | 384 | 0 | 0.000553 | 2347 | True |
| 4 | `Add` | `/2/Add` | 384 | 0 | 0.000551 | 2366 | True |
| 5 | `Add` | `/2/cross_attn/query/Add` | 384 | 0 | 0.000551 | 2356 | True |
| 6 | `Add` | `/2/cross_attn/out/Add` | 384 | 0 | 0.000405 | 2372 | True |
| 7 | `Add` | `/2/Add_1` | 384 | 0 | 0.000401 | 2382 | True |
| 8 | `Add` | `/2/mlp/0/Add` | 1536 | 0 | 0.000681 | 5606 | True |
| 9 | `Add` | `/2/mlp/1/Add` | 1536 | 0 | 0.000664 | 5593 | True |
| 10 | `Add` | `/2/mlp/2/Add` | 384 | 0 | 0.000552 | 2330 | True |
| 11 | `Add` | `/2/Add_2` | 384 | 0 | 0.000553 | 2471 | True |
| 12 | `Add` | `/3/attn/query/Add` | 384 | 0 | 0.000403 | 2361 | True |
| 13 | `Add` | `/3/attn/value/Add` | 384 | 0 | 0.000552 | 2343 | True |
| 14 | `Add` | `/3/attn/Add` | 224 | 0 | 0.000396 | 2047 | True |
| 15 | `Add` | `/3/attn/Add_1` | 224 | 0 | 0.000401 | 2053 | True |
| 16 | `Add` | `/3/attn/Add_2` | 224 | 0 | 0.000403 | 2026 | True |
| 17 | `Add` | `/3/attn/Add_3` | 224 | 0 | 0.000406 | 2078 | True |
| 18 | `Add` | `/3/attn/Add_4` | 224 | 0 | 0.000548 | 2042 | True |
| 19 | `Add` | `/3/attn/Add_5` | 224 | 0 | 0.000552 | 2054 | True |
