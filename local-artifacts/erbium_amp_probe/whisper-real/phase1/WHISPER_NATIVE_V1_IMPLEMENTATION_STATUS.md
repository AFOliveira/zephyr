# Whisper native-v1 implementation status

Date: 2026-05-05

Board host: `esperanto-soc6`

Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2`

## Implemented

- Added `tools/build_whisper_native_v1_bundle.py`.
  - Freezes the log-mel input, FP32 ONNX references, dynamic INT8 ONNX
    references, raw symmetric INT8 weight bins, model hashes, token references,
    and quantization metadata.
  - Generated bundle:
    `phase1/native_v1_bundles/bundle_20260505-205553/bundle_manifest.json`
  - FP32 and dynamic INT8 ONNX produced the same JFK token sequence and text.

- Added `tools/run_whisper_native_v1_overnight.py`.
  - Runs continuously with a remote lock under the allowed ET-SoC1 artifact
    root.
  - Logs every run to TSV and JSONL.
  - Runs native split smoke plus audited E2E silicon variants.
  - Uses `nohup` background mode when `tmux` is not installed.
  - Does not run reboot, reset, power-cycle, bus-reset, driver reload, or other
    board recovery commands.

- Hardened `tools/run_whisper_e2e_audit.py`.
  - Remote static asset cache now checks exact file sizes instead of only
    non-empty files.
  - Rsync staging/fetching now uses explicit timeouts so a stalled transfer does
    not pin the overnight run indefinitely.

## Silicon validation

Full log-mel-to-token-ID silicon run:

- Added `whisper_resident_encoder_argbuf.c`.
  - Runs Whisper Tiny EN encoder on ET-SoC1 from log-mel features.
  - Uses resident raw-INT8 weights from the 64 MiB weight region.
  - Uses all 16 harts for row-partitioned scalar encoder work.
  - Produces FP16 decoder cross-attention K/V caches in the 16 MiB runtime
    arena.
- Added `tools/run_whisper_full_silicon.py`.
  - Runs encoder kernel, then runs the resident decoder kernel from the encoder
    dump.
  - Host ONNXRuntime is used only to generate the audit oracle, not to execute
    the model path.
- Full 23-generated-token run:
  - Report:
    `phase1/full_silicon/run_20260506-090836_new23_ah16/FULL_SILICON_REPORT.md`
  - Result: pass
  - Scope: log-mel features -> encoder cross-cache -> decoder token IDs on
    ET-SoC1.
  - Host/silicon token IDs match exactly.
  - Output text:
    `And so my fellow Americans ask not what your country can do for you ask what you can do for your country`
  - Encoder wait: 148.186 s
  - Decoder wait: 79.7042 s
  - Total kernel wait: 227.8902 s
  - Full silicon generated-token rate: 0.100926 token/s
  - Decoder-only generated-token rate: 0.288567 token/s
  - Cross-cache audit vs host raw-INT8 FP16 oracle:
    - K max abs: 0.0009765625
    - V max abs: 0.001953125
  - Encoder PMCs:
    - hpmcounter3 cycles-like: 88908887555
    - hpmcounter4 retired-inst0: 64252377643
    - hpmcounter5 retired-inst1: 64568437733
    - hpmcounter6 L2-miss-requests: 17623484414
    - hpmcounter7 minion-icache-requests: 17936796095
    - hpmcounter8 icache-etlink-requests: 213
  - Decoder PMCs:
    - hpmcounter3 cycles-like: 47821019324
    - hpmcounter4 retired-inst0: 8048391777
    - hpmcounter5 retired-inst1: 1244
    - hpmcounter6 L2-miss-requests: 910760327
    - hpmcounter7 minion-icache-requests: 1500998336
    - hpmcounter8 icache-etlink-requests: 204

This is not WAV-to-text fully on chip yet: WAV decode/resampling, log-mel
feature extraction, and tokenizer text rendering remain host-side.  The neural
network path from frozen log-mel input through token IDs is now on ET-SoC1.

Short smoke:

- Path:
  `phase1/overnight_smoke_runs_v2/overnight_20260505-211115/ledger.tsv`
- Native split smoke: pass
- E2E tail run: pass
- Generated tokens: 1
- Token/text match vs ONNXRuntime: true
- Silicon audit pass: true

Overnight run:

- Background PID:
  `phase1/overnight_runs/background_logs/whisper-native-overnight.pid`
- Background log:
  `phase1/overnight_runs/background_logs/whisper-native-overnight_20260505-212045.log`
- Ledger:
  `phase1/overnight_runs/overnight_20260505-212045/ledger.tsv`
- First full E2E row:
  - variant: `tail_parallel_tile8192`
  - pass: true
  - generated tokens: 23
  - token/text match vs ONNXRuntime: true
  - silicon audit pass: true
  - hybrid wall rate: 0.100715 token/s
  - silicon tail rate: 14.557756 token/s

Full scalar/Softmax layer coverage:

- Encoder scalar/Softmax:
  - Report:
    `phase1/remote_native_encoder_scalar_runs/run_20260505-222129/results_full_0_94_softmax3/encoder_scalar_remote_native_report.json`
  - Selected nodes: 95 / 95 expected
  - All pass: true
  - Max abs: 0.000160992
  - Mean silicon wait: 0.446747 s/node
  - Total silicon wait: 42.440948 s
- Decoder scalar/Softmax at decoder step 3:
  - Report:
    `phase1/decoder_scalar_fullstep_runs/run_20260505-224625/decoder_scalar_ops_audit_report.json`
  - Selected nodes: 133 / 133 expected
  - All pass: true
  - Max abs: 0.00000155

Scalar performance iteration:

- 16 active harts beat 8 active harts for the full encoder scalar family:
  - 16 harts, accurate softmax/order-5 exp: 44.575150 s total kernel wait
  - 8 harts, accurate softmax/order-5 exp: 48.908174 s total kernel wait
- Best audited scalar kernel keeps accurate order-5 exp for Erf/GELU, uses
  order-3 exp only inside Softmax, and uses software FP32 division instead of
  `fdiv.s` or reciprocal multiply.
  - 16 harts, Softmax order-3: 42.440948 s total kernel wait
  - Improvement vs accurate baseline: 4.8%
  - Accuracy remains within the per-op audit gates.

## Not complete yet

This is not yet a full resident log-mel-to-token-ID graph executor.  The current
validated silicon path still uses ET-SoC1 for final decoder LayerNorm plus
logits argmax, while host ONNXRuntime performs the encoder and remaining
decoder graph.

The next implementation step is to move from the current tail-only audited path
to a host-scheduled silicon graph executor, then collapse batches of ops into a
resident executor.

## Dynamic-memory launcher status

The board-compatible dynamic-memory launcher is staged at:

`/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/erbium_soc1sim_argbuf_dynmem.remote.board`

Validation:

- Report:
  `phase1/dynmem_probe_runs/run_20260505-215358_off20971520_mem33554432/dynmem_probe_report.json`
- `mem_size`: 32 MiB
- high-offset access: 20 MiB
- active harts: 16 / 16
- result: pass

This removes the old 16 MiB arg-buffer limitation for staged layer audits and
is the transport needed for the next host-scheduled graph executor step.
