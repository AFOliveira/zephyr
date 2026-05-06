# DnCNN Variant Sweep Report

Date: 2026-05-01
Board host: `esperanto-soc6`
Remote artifact root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/dncnn3-pmc`
Local artifact root: `local-artifacts/erbium_amp_probe/dncnn3-pmc`

## Method

Each variant was built for 1, 2, 4, 8, and 16 active harts, then run on
silicon with:

1. `pmc_snapshot_16h.elf` before the DnCNN run.
2. The DnCNN variant ELF.
3. `pmc_snapshot_16h.elf` after the DnCNN run.

The summary was accepted only when the kernel reported the expected magic,
the expected done count, matching slot/output checksums, and, for the original
16-channel model, the original output sum `514785`.

## Results

| Variant | Harts | Valid | Wait s | GOPS | Output sum | SC reads | SC writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| best | 1 | 1 | 4.655900 | 0.10135 | 514785 | 15,282,585 | 1,955,061 |
| best | 2 | 1 | 3.359950 | 0.14044 | 514785 | 12,803,957 | 1,948,800 |
| best | 4 | 1 | 1.771060 | 0.26643 | 514785 | 12,848,059 | 1,955,453 |
| best | 8 | 1 | 0.900719 | 0.52387 | 514785 | 12,884,171 | 1,983,586 |
| best | 16 | 1 | 0.447677 | 1.05402 | 514785 | 13,988,556 | 2,074,941 |
| hidden_unroll | 1 | 1 | 4.708660 | 0.10021 | 514785 | 15,394,709 | 1,970,985 |
| hidden_unroll | 2 | 1 | 3.352440 | 0.14075 | 514785 | 16,786,014 | 2,008,577 |
| hidden_unroll | 4 | 1 | 1.691190 | 0.27901 | 514785 | 16,741,350 | 1,988,432 |
| hidden_unroll | 8 | 1 | 0.865099 | 0.54544 | 514785 | 15,294,448 | 1,971,784 |
| hidden_unroll | 16 | 1 | 0.452458 | 1.04288 | 514785 | 17,332,140 | 3,079,335 |
| skip_final_barrier | 1 | 1 | 4.829370 | 0.09771 | 514785 | 15,274,965 | 1,939,072 |
| skip_final_barrier | 2 | 1 | 3.672140 | 0.12850 | 514785 | 16,139,182 | 2,010,582 |
| skip_final_barrier | 4 | 1 | 1.814770 | 0.26001 | 514785 | 12,903,540 | 1,962,717 |
| skip_final_barrier | 8 | 1 | 0.908638 | 0.51930 | 514785 | 12,887,557 | 1,980,396 |
| skip_final_barrier | 16 | 1 | 0.466575 | 1.01133 | 514785 | 12,907,732 | 1,965,203 |
| no_layer_barriers_unsafe | 1 | 1 | 4.829260 | 0.09771 | 514785 | 15,274,855 | 1,938,938 |
| no_layer_barriers_unsafe | 2 | 1 | 3.624470 | 0.13019 | 514785 | 14,038,816 | 2,051,384 |
| no_layer_barriers_unsafe | 4 | 0 | 1.850950 | 0.25493 | 514681 | 14,247,865 | 1,953,733 |
| no_layer_barriers_unsafe | 8 | 0 | 0.001307 | 0.00000 | 0 | 73,338 | 48,018 |
| no_layer_barriers_unsafe | 16 | 0 | 0.479259 | 0.98456 | 514410 | 14,252,677 | 1,958,535 |
| fused_halo | 1 | 1 | 4.880570 | 0.09668 | 514785 | 9,581,148 | 1,937,089 |
| fused_halo | 2 | 1 | 3.736090 | 0.12630 | 514785 | 10,076,233 | 2,011,235 |
| fused_halo | 4 | 0 | 2.161960 | 0.21826 | 514521 | 11,494,384 | 2,378,265 |
| fused_halo | 8 | 0 | 1.323620 | 0.35649 | 514458 | 13,967,538 | 2,902,476 |
| fused_halo | 16 | 0 | 0.884589 | 0.53342 | 514589 | 18,900,722 | 3,947,525 |
| fused_halo_unroll | 1 | 1 | 4.653450 | 0.10140 | 514785 | 11,939,465 | 1,891,155 |
| fused_halo_unroll | 2 | 1 | 3.446120 | 0.13692 | 514785 | 12,654,746 | 3,206,304 |
| fused_halo_unroll | 4 | 0 | 2.004040 | 0.23545 | 514573 | 17,104,282 | 3,552,889 |
| fused_halo_unroll | 8 | 0 | 1.248450 | 0.37796 | 514544 | 20,795,981 | 4,336,840 |
| fused_halo_unroll | 16 | 0 | 0.837238 | 0.56359 | 514538 | 28,143,158 | 5,900,186 |
| model_ch8 | 1 | 1 | 1.455220 | 0.08431 | 524288 | 6,389,194 | 1,030,310 |
| model_ch8 | 2 | 1 | 0.988524 | 0.12411 | 524288 | 5,973,953 | 1,055,234 |
| model_ch8 | 4 | 1 | 0.502332 | 0.24423 | 524288 | 5,868,176 | 1,037,411 |
| model_ch8 | 8 | 1 | 0.263278 | 0.46598 | 524288 | 5,892,657 | 1,047,690 |
| model_ch8 | 16 | 1 | 0.134848 | 0.90979 | 524288 | 5,888,481 | 1,042,708 |
| model_ch4 | 1 | 1 | 0.411777 | 0.08021 | 524288 | 2,298,902 | 466,812 |
| model_ch4 | 2 | 1 | 0.276103 | 0.11963 | 524288 | 2,147,052 | 452,514 |
| model_ch4 | 4 | 1 | 0.144418 | 0.22871 | 524288 | 2,131,428 | 440,057 |
| model_ch4 | 8 | 1 | 0.075483 | 0.43758 | 524288 | 2,169,000 | 447,987 |
| model_ch4 | 16 | 1 | 0.040080 | 0.82411 | 524288 | 2,118,994 | 438,164 |

## Findings

- The best valid original-model result at 16 harts is still the previous
  `best` variant: `0.447677 s`, `1.05402 GOPS`.
- Manual hidden-loop unrolling is useful at 2, 4, and 8 harts. It is not useful
  at 16 harts, where SC traffic rises to `17.33M` reads and `3.08M` writes.
- Removing only the final layer barrier is valid but slower at every hart count.
- Removing all inter-layer barriers is invalid from 4 harts upward. That confirms
  the non-coherent halo exchange really does need either explicit cache/ordering
  or a different private-scratch tiling design.
- The fused-halo prototype is invalid from 4 harts upward. Recomputing halos
  into shared activation buffers races under the non-coherent cache model.
- Lower-channel model variants run much faster in absolute time. They are not
  the same DnCNN3 shape, so their GOPS are lower, but the result is directionally
  useful: smaller/grouped/depthwise models are a better scalar fit than dense
  `3x3 x 16 x 16` hidden layers.

## Tensor Smoke

I also built and ran a minimal tensor-FMA smoke test through the same launch
path. Hart 0 initializes a `14x16` FP32 matrix, multiplies it by a `16x16`
identity matrix with tensor load/FMA/store, and checks the result.

Result:

```text
magic=0x54534d4b hart=0 minion=0 thread=0 tensor_error=0x0 checksum=25200 expected=25200 done=1
Kernel wait seconds: 0.000550568
```

This proves the tensor instruction path is usable from the launcher. It does
not yet tensorize DnCNN. The next real tensor step is to pack one hidden
convolution as im2col-style `14x16 x 16x16` tiles on hart 0 of each minion, and
keep hart 1 out of tensor instructions or use it only for scalar orchestration.

## Artifacts

- `dncnn3_bench_argbuf.c`: local benchmark source with the compile-time variants.
- `build_dncnn_variants.sh`: local build script for all DnCNN variants.
- `run_dncnn_variant_pmc_suite.sh`: remote silicon run script.
- `parsed/dncnn_variant_sweep_summary.tsv`: compact parsed result table.
- `parsed/dncnn_pmc_results.tsv`: full parsed PMC result table.
- `artifact_manifest_variant_sweep_local.tsv`: local hashes for built sources and ELFs.
- `artifact_manifest_variant_sweep_remote.tsv`: remote hashes for ELFs, dumps, logs, and snapshots.
- `tensor_smoke_argbuf.c` and `tensor_smoke_argbuf.elf`: tensor-FMA smoke test.
- `artifact_manifest_tensor_smoke_local.tsv` and `artifact_manifest_tensor_smoke_remote.tsv`: tensor smoke hashes.
