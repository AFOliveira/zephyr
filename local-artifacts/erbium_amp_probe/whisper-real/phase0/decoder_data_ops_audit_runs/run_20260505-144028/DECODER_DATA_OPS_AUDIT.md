# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `60`
- Audited nodes: `48`
- Skipped nodes: `12`
- All audited pass: `True`
- Max abs diff: `0.0`
- Mean silicon wait: `0.003211348145833333 s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `Slice` | `/0/attn/Slice_2` | 64 | 384 | 0 | 0.000550 | True |
| 1 | `Slice` | `/0/attn/Slice_3` | 14336 | 86016 | 0 | 0.003519 | True |
| 2 | `Slice` | `/0/attn/Slice_4` | 14336 | 86016 | 0 | 0.003553 | True |
| 3 | `Slice` | `/0/attn/Slice_5` | 64 | 384 | 0 | 0.000552 | True |
| 4 | `Slice` | `/0/attn/Slice_6` | 14336 | 86016 | 0 | 0.003463 | True |
| 5 | `Slice` | `/0/attn/Slice_7` | 14336 | 86016 | 0 | 0.003577 | True |
| 6 | `Slice` | `/0/attn/Slice_8` | 64 | 384 | 0 | 0.000302 | True |
| 7 | `Slice` | `/0/attn/Slice_9` | 14336 | 86016 | 0 | 0.003499 | True |
| 8 | `Slice` | `/0/attn/Slice_10` | 14336 | 86016 | 0 | 0.003660 | True |
| 9 | `Slice` | `/0/attn/Slice_11` | 64 | 384 | 0 | 0.000551 | True |
| 10 | `Slice` | `/0/attn/Slice_12` | 14336 | 86016 | 0 | 0.003527 | True |
| 11 | `Slice` | `/0/attn/Slice_13` | 14336 | 86016 | 0 | 0.003546 | True |
| 12 | `Slice` | `/0/attn/Slice_14` | 64 | 384 | 0 | 0.000303 | True |
| 13 | `Slice` | `/0/attn/Slice_15` | 14336 | 86016 | 0 | 0.003509 | True |
| 14 | `Slice` | `/0/attn/Slice_16` | 14336 | 86016 | 0 | 0.003496 | True |
| 15 | `Slice` | `/0/attn/Slice_17` | 64 | 384 | 0 | 0.000303 | True |
| 16 | `Slice` | `/0/attn/Slice_18` | 14336 | 86016 | 0 | 0.003488 | True |
| 17 | `Slice` | `/0/attn/Slice_19` | 14336 | 86016 | 0 | 0.003486 | True |
| 18 | `Concat` | `/0/attn/Concat_2` | 384 | 384 | 0 | 0.000406 | True |
| 19 | `Reshape` | `/0/attn/Reshape_3` | 384 | 384 | 0 | 0.000401 | True |
| 20 | `Reshape` | `/0/cross_attn/Reshape` | 384 | 384 | 0 | 0.000405 | True |
| 21 | `Slice` | `/0/cross_attn/Slice` | 64 | 384 | 0 | 0.000361 | True |
| 24 | `Slice` | `/0/cross_attn/Slice_3` | 64 | 384 | 0 | 0.000550 | True |
| 27 | `Slice` | `/0/cross_attn/Slice_6` | 64 | 384 | 0 | 0.000536 | True |
| 30 | `Slice` | `/0/cross_attn/Slice_9` | 64 | 384 | 0 | 0.000302 | True |
| 33 | `Slice` | `/0/cross_attn/Slice_12` | 64 | 384 | 0 | 0.000559 | True |
| 36 | `Slice` | `/0/cross_attn/Slice_15` | 64 | 384 | 0 | 0.000361 | True |
| 39 | `Concat` | `/0/cross_attn/Concat` | 384 | 384 | 0 | 0.000551 | True |
| 40 | `Reshape` | `/0/cross_attn/Reshape_1` | 384 | 384 | 0 | 0.000550 | True |
| 41 | `Reshape` | `/1/attn/Reshape` | 384 | 384 | 0 | 0.000550 | True |
| 42 | `Unsqueeze` | `/1/attn/Unsqueeze` | 384 | 384 | 0 | 0.000551 | True |
| 43 | `Reshape` | `/1/attn/Reshape_1` | 384 | 384 | 0 | 0.000443 | True |
| 44 | `Unsqueeze` | `/1/attn/Unsqueeze_1` | 384 | 384 | 0 | 0.000403 | True |
| 45 | `Reshape` | `/1/attn/Reshape_2` | 384 | 384 | 0 | 0.000483 | True |
| 46 | `Slice` | `/Slice_4_fused_with_/1/attn/Slice` | 85632 | 344064 | 0 | 0.019872 | True |
| 47 | `Concat` | `/1/attn/Concat` | 86016 | 86016 | 0 | 0.019711 | True |
| 48 | `Slice` | `/Slice_5_fused_with_/1/attn/Slice_1` | 85632 | 344064 | 0 | 0.019861 | True |
| 49 | `Concat` | `/1/attn/Concat_1` | 86016 | 86016 | 0 | 0.019600 | True |
| 50 | `Slice` | `/1/attn/Slice_2` | 64 | 384 | 0 | 0.000301 | True |
| 51 | `Slice` | `/1/attn/Slice_3` | 14336 | 86016 | 0 | 0.003514 | True |
| 52 | `Slice` | `/1/attn/Slice_4` | 14336 | 86016 | 0 | 0.003528 | True |
| 53 | `Slice` | `/1/attn/Slice_5` | 64 | 384 | 0 | 0.000303 | True |
| 54 | `Slice` | `/1/attn/Slice_6` | 14336 | 86016 | 0 | 0.003552 | True |
| 55 | `Slice` | `/1/attn/Slice_7` | 14336 | 86016 | 0 | 0.003496 | True |
| 56 | `Slice` | `/1/attn/Slice_8` | 64 | 384 | 0 | 0.000549 | True |
| 57 | `Slice` | `/1/attn/Slice_9` | 14336 | 86016 | 0 | 0.003499 | True |
| 58 | `Slice` | `/1/attn/Slice_10` | 14336 | 86016 | 0 | 0.003507 | True |
| 59 | `Slice` | `/1/attn/Slice_11` | 64 | 384 | 0 | 0.000552 | True |

## Skipped

- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_1`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_2`: source too large for arena: 2304000 floats
- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_4`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_5`: source too large for arena: 2304000 floats
- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_7`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_8`: source too large for arena: 2304000 floats
- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_10`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_11`: source too large for arena: 2304000 floats
- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_13`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_14`: source too large for arena: 2304000 floats
- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_16`: source too large for arena: 2304000 floats
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_17`: source too large for arena: 2304000 floats
