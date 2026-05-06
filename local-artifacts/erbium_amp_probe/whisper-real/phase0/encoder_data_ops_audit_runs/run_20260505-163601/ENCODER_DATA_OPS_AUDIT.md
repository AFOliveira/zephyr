# Whisper Encoder Data Ops Audit

- Graph: `encoder`
- Decoder step: `n/a`
- Selected nodes: `79`
- Audited nodes: `103`
- Skipped nodes: `0`
- Tile elements: `262144`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.0972086854368932 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Transpose` | `/encoder/Transpose` | 576000 | 576000 | 0 | 0.139634 | True |
| 1 | `Split` | `Split_of_/encoder/blocks.0/attn/query/MatMul_and_/encoder/blocks.0/attn/key/MatMul_and_/encoder/blocks.0/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.135926 | True |
| 1 | `Split` | `Split_of_/encoder/blocks.0/attn/query/MatMul_and_/encoder/blocks.0/attn/key/MatMul_and_/encoder/blocks.0/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136156 | True |
| 1 | `Split` | `Split_of_/encoder/blocks.0/attn/query/MatMul_and_/encoder/blocks.0/attn/key/MatMul_and_/encoder/blocks.0/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136157 | True |
| 2 | `Reshape` | `/encoder/blocks.0/attn/Reshape_1` | 576000 | 576000 | 0 | 0.136312 | True |
| 3 | `Reshape` | `/encoder/blocks.0/attn/Reshape` | 576000 | 576000 | 0 | 0.136236 | True |
| 4 | `Transpose` | `/encoder/blocks.0/attn/Transpose_1` | 576000 | 576000 | 0 | 0.147713 | True |
| 5 | `Reshape` | `/encoder/blocks.0/attn/Reshape_2` | 576000 | 576000 | 0 | 0.136419 | True |
| 6 | `Transpose` | `/encoder/blocks.0/attn/Transpose` | 576000 | 576000 | 0 | 0.136531 | True |
| 7 | `Transpose` | `/encoder/blocks.0/attn/Transpose_2` | 576000 | 576000 | 0 | 0.136404 | True |
| 8 | `Transpose` | `/encoder/blocks.0/attn/Transpose_3` | 576000 | 576000 | 0 | 0.136367 | True |
| 9 | `Reshape` | `/encoder/blocks.0/attn/Reshape_3` | 576000 | 576000 | 0 | 0.136249 | True |
| 10 | `Split` | `Split_of_/encoder/blocks.1/attn/query/MatMul_and_/encoder/blocks.1/attn/key/MatMul_and_/encoder/blocks.1/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136145 | True |
| 10 | `Split` | `Split_of_/encoder/blocks.1/attn/query/MatMul_and_/encoder/blocks.1/attn/key/MatMul_and_/encoder/blocks.1/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136122 | True |
| 10 | `Split` | `Split_of_/encoder/blocks.1/attn/query/MatMul_and_/encoder/blocks.1/attn/key/MatMul_and_/encoder/blocks.1/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136062 | True |
| 11 | `Reshape` | `/encoder/blocks.1/attn/Reshape_1` | 576000 | 576000 | 0 | 0.136211 | True |
| 12 | `Reshape` | `/encoder/blocks.1/attn/Reshape` | 576000 | 576000 | 0 | 0.136389 | True |
| 13 | `Transpose` | `/encoder/blocks.1/attn/Transpose_1` | 576000 | 576000 | 0 | 0.147762 | True |
| 14 | `Reshape` | `/encoder/blocks.1/attn/Reshape_2` | 576000 | 576000 | 0 | 0.136373 | True |
| 15 | `Transpose` | `/encoder/blocks.1/attn/Transpose` | 576000 | 576000 | 0 | 0.136465 | True |
| 16 | `Transpose` | `/encoder/blocks.1/attn/Transpose_2` | 576000 | 576000 | 0 | 0.136455 | True |
| 17 | `Transpose` | `/encoder/blocks.1/attn/Transpose_3` | 576000 | 576000 | 0 | 0.136401 | True |
| 18 | `Reshape` | `/encoder/blocks.1/attn/Reshape_3` | 576000 | 576000 | 0 | 0.136318 | True |
| 19 | `Split` | `Split_of_/encoder/blocks.2/attn/query/MatMul_and_/encoder/blocks.2/attn/key/MatMul_and_/encoder/blocks.2/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.135990 | True |
| 19 | `Split` | `Split_of_/encoder/blocks.2/attn/query/MatMul_and_/encoder/blocks.2/attn/key/MatMul_and_/encoder/blocks.2/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136160 | True |
| 19 | `Split` | `Split_of_/encoder/blocks.2/attn/query/MatMul_and_/encoder/blocks.2/attn/key/MatMul_and_/encoder/blocks.2/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136043 | True |
| 20 | `Reshape` | `/encoder/blocks.2/attn/Reshape_1` | 576000 | 576000 | 0 | 0.136380 | True |
| 21 | `Reshape` | `/encoder/blocks.2/attn/Reshape` | 576000 | 576000 | 0 | 0.136443 | True |
| 22 | `Transpose` | `/encoder/blocks.2/attn/Transpose_1` | 576000 | 576000 | 0 | 0.147712 | True |
| 23 | `Reshape` | `/encoder/blocks.2/attn/Reshape_2` | 576000 | 576000 | 0 | 0.136378 | True |
| 24 | `Transpose` | `/encoder/blocks.2/attn/Transpose` | 576000 | 576000 | 0 | 0.136380 | True |
| 25 | `Transpose` | `/encoder/blocks.2/attn/Transpose_2` | 576000 | 576000 | 0 | 0.136457 | True |
| 26 | `Transpose` | `/encoder/blocks.2/attn/Transpose_3` | 576000 | 576000 | 0 | 0.136329 | True |
| 27 | `Reshape` | `/encoder/blocks.2/attn/Reshape_3` | 576000 | 576000 | 0 | 0.136411 | True |
| 28 | `Split` | `Split_of_/encoder/blocks.3/attn/query/MatMul_and_/encoder/blocks.3/attn/key/MatMul_and_/encoder/blocks.3/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136118 | True |
| 28 | `Split` | `Split_of_/encoder/blocks.3/attn/query/MatMul_and_/encoder/blocks.3/attn/key/MatMul_and_/encoder/blocks.3/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136102 | True |
| 28 | `Split` | `Split_of_/encoder/blocks.3/attn/query/MatMul_and_/encoder/blocks.3/attn/key/MatMul_and_/encoder/blocks.3/attn/value/MatMul` | 576000 | 1728000 | 0 | 0.136071 | True |
| 29 | `Reshape` | `/encoder/blocks.3/attn/Reshape_1` | 576000 | 576000 | 0 | 0.136344 | True |
| 30 | `Reshape` | `/encoder/blocks.3/attn/Reshape` | 576000 | 576000 | 0 | 0.136299 | True |
| 31 | `Transpose` | `/encoder/blocks.3/attn/Transpose_1` | 576000 | 576000 | 0 | 0.147682 | True |
| 32 | `Reshape` | `/encoder/blocks.3/attn/Reshape_2` | 576000 | 576000 | 0 | 0.136427 | True |
| 33 | `Transpose` | `/encoder/blocks.3/attn/Transpose` | 576000 | 576000 | 0 | 0.136323 | True |
| 34 | `Transpose` | `/encoder/blocks.3/attn/Transpose_2` | 576000 | 576000 | 0 | 0.136314 | True |
| 35 | `Transpose` | `/encoder/blocks.3/attn/Transpose_3` | 576000 | 576000 | 0 | 0.136320 | True |
| 36 | `Reshape` | `/encoder/blocks.3/attn/Reshape_3` | 576000 | 576000 | 0 | 0.136292 | True |
| 37 | `Transpose` | `/0/Transpose` | 96000 | 96000 | 0 | 0.023279 | True |
| 38 | `Transpose` | `/0/Transpose_1` | 96000 | 96000 | 0 | 0.023262 | True |
| 39 | `Transpose` | `/0/Transpose_2` | 96000 | 96000 | 0 | 0.023234 | True |
| 40 | `Transpose` | `/0/Transpose_3` | 96000 | 96000 | 0 | 0.023179 | True |
| 41 | `Transpose` | `/0/Transpose_4` | 96000 | 96000 | 0 | 0.023201 | True |
| 42 | `Transpose` | `/0/Transpose_5` | 96000 | 96000 | 0 | 0.023302 | True |
| 43 | `Transpose` | `/1/Transpose` | 96000 | 96000 | 0 | 0.023158 | True |
| 44 | `Transpose` | `/1/Transpose_1` | 96000 | 96000 | 0 | 0.023327 | True |
| 45 | `Transpose` | `/1/Transpose_2` | 96000 | 96000 | 0 | 0.023259 | True |
| 46 | `Transpose` | `/1/Transpose_3` | 96000 | 96000 | 0 | 0.023257 | True |
| 47 | `Transpose` | `/1/Transpose_4` | 96000 | 96000 | 0 | 0.023227 | True |
| 48 | `Transpose` | `/1/Transpose_5` | 96000 | 96000 | 0 | 0.023183 | True |
| 49 | `Transpose` | `/2/Transpose` | 96000 | 96000 | 0 | 0.023242 | True |
| 50 | `Transpose` | `/2/Transpose_1` | 96000 | 96000 | 0 | 0.023268 | True |
| 51 | `Transpose` | `/2/Transpose_2` | 96000 | 96000 | 0 | 0.023165 | True |
| 52 | `Transpose` | `/2/Transpose_3` | 96000 | 96000 | 0 | 0.023230 | True |
| 53 | `Transpose` | `/2/Transpose_4` | 96000 | 96000 | 0 | 0.023255 | True |
| 54 | `Transpose` | `/2/Transpose_5` | 96000 | 96000 | 0 | 0.023145 | True |
| 55 | `Transpose` | `/3/Transpose` | 96000 | 96000 | 0 | 0.023151 | True |
| 56 | `Transpose` | `/3/Transpose_1` | 96000 | 96000 | 0 | 0.023191 | True |
| 57 | `Transpose` | `/3/Transpose_2` | 96000 | 96000 | 0 | 0.023301 | True |
| 58 | `Transpose` | `/3/Transpose_3` | 96000 | 96000 | 0 | 0.023135 | True |
| 59 | `Transpose` | `/3/Transpose_4` | 96000 | 96000 | 0 | 0.023220 | True |
| 60 | `Transpose` | `/3/Transpose_5` | 96000 | 96000 | 0 | 0.023142 | True |
| 61 | `Concat` | `/0/Concat` | 576000 | 576000 | 0 | 0.136381 | True |
| 62 | `Concat` | `/1/Concat` | 576000 | 576000 | 0 | 0.136306 | True |
| 63 | `Concat` | `/2/Concat` | 576000 | 576000 | 0 | 0.136282 | True |
| 64 | `Concat` | `/3/Concat` | 576000 | 576000 | 0 | 0.136519 | True |
| 65 | `Concat` | `/0_1/Concat` | 576000 | 576000 | 0 | 0.136568 | True |
| 66 | `Concat` | `/1_1/Concat` | 576000 | 576000 | 0 | 0.136429 | True |
| 67 | `Concat` | `/2_1/Concat` | 576000 | 576000 | 0 | 0.136491 | True |
| 68 | `Concat` | `/3_1/Concat` | 576000 | 576000 | 0 | 0.136397 | True |
| 69 | `Unsqueeze` | `/Unsqueeze` | 576000 | 576000 | 0 | 0.136359 | True |
| 70 | `Unsqueeze` | `/Unsqueeze_1` | 576000 | 576000 | 0 | 0.136354 | True |
| 71 | `Unsqueeze` | `/Unsqueeze_2` | 576000 | 576000 | 0 | 0.136448 | True |
| 72 | `Unsqueeze` | `/Unsqueeze_3` | 576000 | 576000 | 0 | 0.136270 | True |
| 73 | `Unsqueeze` | `/Unsqueeze_4` | 576000 | 576000 | 0 | 0.136396 | True |
| 74 | `Unsqueeze` | `/Unsqueeze_5` | 576000 | 576000 | 0 | 0.136249 | True |
| 75 | `Unsqueeze` | `/Unsqueeze_6` | 576000 | 576000 | 0 | 0.136254 | True |
| 76 | `Unsqueeze` | `/Unsqueeze_7` | 576000 | 576000 | 0 | 0.136370 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062059 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062090 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062127 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062101 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062156 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062187 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062125 | True |
| 77 | `Concat` | `/Concat` | 262144 | 2304000 | 0 | 0.062238 | True |
| 77 | `Concat` | `/Concat` | 206848 | 2304000 | 0 | 0.048504 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062193 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062144 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062093 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062117 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062260 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062068 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062286 | True |
| 78 | `Concat` | `/Concat_1` | 262144 | 2304000 | 0 | 0.062262 | True |
| 78 | `Concat` | `/Concat_1` | 206848 | 2304000 | 0 | 0.048319 | True |
