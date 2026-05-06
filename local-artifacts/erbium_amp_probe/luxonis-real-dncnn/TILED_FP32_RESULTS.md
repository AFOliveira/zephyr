# DnCNN3 Tiled FP32 Silicon Results

Board host: `esperanto-soc6`

Remote root:
`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/luxonis-real-dncnn`

Master audit:
`local-artifacts/erbium_amp_probe/luxonis-real-dncnn/audit_master_tiled.tsv`

## Summary

The full 240x320 Luxonis DnCNN3 FP32 path now runs as ORT-audited tiles on
ET-SoC1 silicon.  The final master log contains 101 tile-level hardware runs
with `allclose=1` against the per-tile ORT references, plus 17 full-image
stitched audit rows.

Best geometry:

| Variant | Tiles | Mean Wait s | Min Wait s | Max Wait s | Useful GOPS |
|---|---:|---:|---:|---:|---:|
| `a4_geom120x160_h20_f9t_pfw_align128` | 4 | 50.568423 | 50.564600 | 50.573100 | 2.019018 |

Geometry sweep:

| Variant | Tiles | Wait s | Useful GOPS | Status |
|---|---:|---:|---:|---|
| `a3_tiled64_halo20_f9t_pfw_align128` | 20 | 95.017700 | 1.074521 | pass |
| `a4_geom120x160_h20_f9t_pfw_align128` | 4 | 50.567600 first run, 50.568423 repeat mean | 2.019018 repeat mean | pass |
| `a4_geom128x160_h20_f9t_pfw_align128` | 4 | 52.445600 | 1.946751 | pass |
| `a4_geom80x80_h20_f9t_pfw_align128` | 12 | 75.191630 | 1.357844 | pass |
| `a4_geom96x96_h20_f9t_pfw_align128` | 12 | 75.319910 | 1.355532 | pass |

Accuracy gate:

- Full stitched output vs full ORT FP32: `max_abs=8.940696716308594e-07`
- Mean absolute error: about `8.486065894430794e-08`
- Tolerance: `1e-5`

## Notes

The tile kernels currently produce valid output and valid per-hart slots, but
the summary block is still absent in these argbuf runs and the launcher logs a
stream error.  The audit harness therefore classifies tile rows as
`ok_stream_error` when all 16 slots are done and the tile crop matches ORT.
Full stitched rows are `ok`.

The largest improvement came from tile geometry, not compiler flags: using
four 120x160 output tiles reduces halo recomputation enough to improve useful
full-image throughput from 1.07 GOPS to about 2.02 GOPS.
