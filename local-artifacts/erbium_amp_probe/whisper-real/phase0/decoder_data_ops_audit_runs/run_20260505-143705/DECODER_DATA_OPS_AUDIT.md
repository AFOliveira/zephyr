# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- Audited nodes: `11`
- Skipped nodes: `1`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.007522360909090909 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `Gather` | `/positional_embedding/Gather` | 384 | 86016 | 0 | 0.000542 | True |
| 2 | `Gather` | `/mask/Gather` | 224 | 50176 | 0 | 0.000635 | True |
| 3 | `Reshape` | `/0/attn/Reshape` | 384 | 384 | 0 | 0.000551 | True |
| 4 | `Unsqueeze` | `/0/attn/Unsqueeze` | 384 | 384 | 0 | 0.000544 | True |
| 5 | `Reshape` | `/0/attn/Reshape_1` | 384 | 384 | 0 | 0.000551 | True |
| 6 | `Unsqueeze` | `/0/attn/Unsqueeze_1` | 384 | 384 | 0 | 0.000551 | True |
| 7 | `Reshape` | `/0/attn/Reshape_2` | 384 | 384 | 0 | 0.000440 | True |
| 8 | `Slice` | `/Slice_fused_with_/0/attn/Slice` | 85632 | 344064 | 0 | 0.019674 | True |
| 9 | `Concat` | `/0/attn/Concat` | 86016 | 86016 | 0 | 0.019647 | True |
| 10 | `Slice` | `/Slice_1_fused_with_/0/attn/Slice_1` | 85632 | 344064 | 0 | 0.019834 | True |
| 11 | `Concat` | `/0/attn/Concat_1` | 86016 | 86016 | 0 | 0.019777 | True |

## Skipped

- `Gather` `/token_embedding/Gather`: source too large for arena: 19915776 floats
