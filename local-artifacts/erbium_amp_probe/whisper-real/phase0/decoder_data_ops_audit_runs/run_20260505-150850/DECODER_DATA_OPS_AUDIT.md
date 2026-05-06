# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `12`
- Audited nodes: `12`
- Skipped nodes: `0`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.022376058333333334 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_1` | 96000 | 2304000 | 0 | 0.022327 | True |
| 1 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_2` | 96000 | 2304000 | 0 | 0.022332 | True |
| 2 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_4` | 96000 | 2304000 | 0 | 0.022381 | True |
| 3 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_5` | 96000 | 2304000 | 0 | 0.022330 | True |
| 4 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_7` | 96000 | 2304000 | 0 | 0.022332 | True |
| 5 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_8` | 96000 | 2304000 | 0 | 0.022422 | True |
| 6 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_10` | 96000 | 2304000 | 0 | 0.022447 | True |
| 7 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_11` | 96000 | 2304000 | 0 | 0.022450 | True |
| 8 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_13` | 96000 | 2304000 | 0 | 0.022411 | True |
| 9 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_14` | 96000 | 2304000 | 0 | 0.022411 | True |
| 10 | `Slice` | `/Slice_2_fused_with_/0/cross_attn/Slice_16` | 96000 | 2304000 | 0 | 0.022350 | True |
| 11 | `Slice` | `/Slice_3_fused_with_/0/cross_attn/Slice_17` | 96000 | 2304000 | 0 | 0.022320 | True |
