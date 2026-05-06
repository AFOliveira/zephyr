# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `2`
- Audited nodes: `2`
- Skipped nodes: `0`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.02237985 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_1` | 96000 | 2304000 | 0 | 0.022403 | True |
| 1 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_2` | 96000 | 2304000 | 0 | 0.022357 | True |
