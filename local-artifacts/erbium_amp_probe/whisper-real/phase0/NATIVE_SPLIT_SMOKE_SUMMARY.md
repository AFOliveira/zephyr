# Whisper native split smoke summary

Date: 2026-05-04

Board: `esperanto-soc6`

Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real/native-split-smoke`

Scope: this validates the native scheduler shape for Whisper on one ET-SoC1
shire. It is not a full Whisper graph executor yet. Even harts run a
Whisper-shaped VPU FP32 logits MatMul and odd harts run LayerNorm-shaped scalar
reduction/normalization in the same kernel, with explicit barriers and cache
evictions.

## Fixes needed

- Used the soc1sim-staged Erbium headers before generic `et-common-libs`
  headers. Without this, the barrier code targeted the wrong FCC/FLB addresses.
- Built with `NUM_HARTS=16` and selected the tested population using
  `ACTIVE_HARTS`. This matches the known-good DnCNN runner shape.
- Removed scalar FP divides from the device path and used compile-time
  reciprocals instead.
- Padded per-hart scalar partial records to 64 bytes. Without this, 8-hart
  LayerNorm-shaped reductions failed from non-coherent cache-line false sharing.

## Passing silicon runs

| Active harts | VPU harts | Scalar harts | VPU mask | Scalar mask | Wait s | Local artifact |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 2 | 1 | 1 | `0x1` | `0x2` | 0.00883826 | `native_split_smoke_runs/run_both_phase0_ah2_20260504-204144` |
| 4 | 2 | 2 | `0x5` | `0xa` | 0.00476738 | `native_split_smoke_runs/run_both_phase0_ah4_20260504-204302` |
| 8 | 4 | 4 | `0x55` | `0xaa` | 0.00276203 | `native_split_smoke_runs/run_both_phase0_ah8_20260504-204424` |
| 16 | 8 | 8 | `0x5555` | `0xaaaa` | 0.00179517 | `native_split_smoke_runs/run_both_phase0_ah16_20260504-204539` |
| 16 + PMCs | 8 | 8 | `0x5555` | `0xaaaa` | 0.00187290 | `native_split_smoke_runs/run_both_phase0_ah16_20260504-204908` |

All passing rows had:

- `logits_argmax_match = true`
- `logits_max_abs = 0.0` at the report scale
- `ln_max_abs = 0.0` at the report scale
- `stream_error = false`
- `kernel_launch_error = false`

## Meaning

The 16-hart run proves we can use all 8 minions in one shire with a practical
Whisper scheduler split: thread 0 harts for VPU-heavy MatMul work and thread 1
harts for scalar graph work. The main integration risk now moves from hart
coordination to memory management and paging of real Whisper tensors.

## 16-hart PMC snapshot

Run: `native_split_smoke_runs/run_both_phase0_ah16_20260504-204908`

| Counter | Firmware event | Delta |
| --- | --- | ---: |
| `hpmcounter3` | cycles | 888160 |
| `hpmcounter4` | retired inst0 | 247430 |
| `hpmcounter5` | retired inst1 | 10899 |
| `hpmcounter6` | L2 miss request | 38519 |
| `hpmcounter7` | minion icache request | 86126 |
| `hpmcounter8` | icache etlink request | 107 |

Optimization readout: the split kernel is dominated by thread-0/VPU-side work
and memory traffic. Thread-1 scalar work is visible but small in this smoke
shape, so the next useful optimizations are larger VPU tiles, fewer tile
launches, reusing staged weights, and eliminating cache-line false sharing in
all cross-hart handoff structures.
