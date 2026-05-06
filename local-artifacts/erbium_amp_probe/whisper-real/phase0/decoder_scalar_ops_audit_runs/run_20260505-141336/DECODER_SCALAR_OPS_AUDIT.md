# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `9`
- All pass: `True`
- Max abs diff: `0`
- Mean silicon wait: `0.000521 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/3/attn/out/Add` | 384 | 0 | 0.000369 | 2361 | True |
| 1 | `Add` | `/3/Add` | 384 | 0 | 0.000404 | 2331 | True |
| 2 | `Add` | `/3/cross_attn/query/Add` | 384 | 0 | 0.000551 | 2331 | True |
| 3 | `Add` | `/3/cross_attn/out/Add` | 384 | 0 | 0.000552 | 2330 | True |
| 4 | `Add` | `/3/Add_1` | 384 | 0 | 0.000552 | 2358 | True |
| 5 | `Add` | `/3/mlp/0/Add` | 1536 | 0 | 0.000654 | 5738 | True |
| 6 | `Add` | `/3/mlp/1/Add` | 1536 | 0 | 0.000653 | 5677 | True |
| 7 | `Add` | `/3/mlp/2/Add` | 384 | 0 | 0.000551 | 2338 | True |
| 8 | `Add` | `/3/Add_2` | 384 | 0 | 0.000403 | 2321 | True |
