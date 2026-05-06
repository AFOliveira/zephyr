# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `4`
- Audited nodes: `12`
- Skipped nodes: `0`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.0005290973333333333 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Split` | `Split_of_/0/attn/query/MatMul_and_/0/attn/key/MatMul_and_/0/attn/value/MatMul` | 384 | 1152 | 0 | 0.000539 | True |
| 0 | `Split` | `Split_of_/0/attn/query/MatMul_and_/0/attn/key/MatMul_and_/0/attn/value/MatMul` | 384 | 1152 | 0 | 0.000404 | True |
| 0 | `Split` | `Split_of_/0/attn/query/MatMul_and_/0/attn/key/MatMul_and_/0/attn/value/MatMul` | 384 | 1152 | 0 | 0.000568 | True |
| 1 | `Split` | `Split_of_/1/attn/query/MatMul_and_/1/attn/key/MatMul_and_/1/attn/value/MatMul` | 384 | 1152 | 0 | 0.000552 | True |
| 1 | `Split` | `Split_of_/1/attn/query/MatMul_and_/1/attn/key/MatMul_and_/1/attn/value/MatMul` | 384 | 1152 | 0 | 0.000552 | True |
| 1 | `Split` | `Split_of_/1/attn/query/MatMul_and_/1/attn/key/MatMul_and_/1/attn/value/MatMul` | 384 | 1152 | 0 | 0.000551 | True |
| 2 | `Split` | `Split_of_/2/attn/query/MatMul_and_/2/attn/key/MatMul_and_/2/attn/value/MatMul` | 384 | 1152 | 0 | 0.000549 | True |
| 2 | `Split` | `Split_of_/2/attn/query/MatMul_and_/2/attn/key/MatMul_and_/2/attn/value/MatMul` | 384 | 1152 | 0 | 0.000551 | True |
| 2 | `Split` | `Split_of_/2/attn/query/MatMul_and_/2/attn/key/MatMul_and_/2/attn/value/MatMul` | 384 | 1152 | 0 | 0.000406 | True |
| 3 | `Split` | `Split_of_/3/attn/query/MatMul_and_/3/attn/key/MatMul_and_/3/attn/value/MatMul` | 384 | 1152 | 0 | 0.000575 | True |
| 3 | `Split` | `Split_of_/3/attn/query/MatMul_and_/3/attn/key/MatMul_and_/3/attn/value/MatMul` | 384 | 1152 | 0 | 0.000551 | True |
| 3 | `Split` | `Split_of_/3/attn/query/MatMul_and_/3/attn/key/MatMul_and_/3/attn/value/MatMul` | 384 | 1152 | 0 | 0.000551 | True |
