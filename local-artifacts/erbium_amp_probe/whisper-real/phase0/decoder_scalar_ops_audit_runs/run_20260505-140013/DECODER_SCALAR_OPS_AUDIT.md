# Whisper Decoder Scalar Ops Audit

- Decoder step: `3`
- Selected nodes: `20`
- All pass: `True`
- Max abs diff: `0`
- Mean silicon wait: `0.000525 s`

| index | op | node | elems | max abs | wait s | cycles | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Add` | `/Add` | 384 | 0 | 0.000551 | 2344 | True |
| 1 | `Add` | `/0/attn/query/Add` | 384 | 0 | 0.000551 | 2335 | True |
| 2 | `Add` | `/0/attn/value/Add` | 384 | 0 | 0.000550 | 2331 | True |
| 3 | `Add` | `/0/attn/Add` | 224 | 0 | 0.000551 | 2026 | True |
| 4 | `Add` | `/0/attn/Add_1` | 224 | 0 | 0.000404 | 2050 | True |
| 5 | `Add` | `/0/attn/Add_2` | 224 | 0 | 0.000552 | 2041 | True |
| 6 | `Add` | `/0/attn/Add_3` | 224 | 0 | 0.000404 | 2052 | True |
| 7 | `Add` | `/0/attn/Add_4` | 224 | 0 | 0.000404 | 2030 | True |
| 8 | `Add` | `/0/attn/Add_5` | 224 | 0 | 0.000551 | 2057 | True |
| 9 | `Add` | `/0/attn/out/Add` | 384 | 0 | 0.000552 | 2218 | True |
| 10 | `Add` | `/0/Add` | 384 | 0 | 0.000552 | 2336 | True |
| 11 | `Add` | `/0/cross_attn/query/Add` | 384 | 0 | 0.000552 | 2338 | True |
| 12 | `Add` | `/0/cross_attn/out/Add` | 384 | 0 | 0.000553 | 2336 | True |
| 13 | `Add` | `/0/Add_1` | 384 | 0 | 0.000552 | 2327 | True |
| 14 | `Add` | `/0/mlp/0/Add` | 1536 | 0 | 0.000653 | 5660 | True |
| 15 | `Add` | `/0/mlp/1/Add` | 1536 | 0 | 0.000653 | 5658 | True |
| 16 | `Add` | `/0/mlp/2/Add` | 384 | 0 | 0.000406 | 2335 | True |
| 17 | `Add` | `/0/Add_2` | 384 | 0 | 0.000552 | 2451 | True |
| 18 | `Add` | `/1/attn/query/Add` | 384 | 0 | 0.000551 | 2344 | True |
| 19 | `Add` | `/1/attn/value/Add` | 384 | 0 | 0.000406 | 2231 | True |
