# Whisper current results

Date: 2026-05-04

Board: `esperanto-soc6`

Remote root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/whisper-real`

## Full audio-to-text audit

Input WAV: `local-artifacts/erbium_amp_probe/whisper-real/audio/openai_whisper_jfk_16k.wav`

Model: Luxonis Whisper Tiny EN encoder/decoder ONNX exports.

Latest optimized run:
`phase0/e2e_audit_runs/both_20260504-210334/e2e_audit_report.json`

Transcript:

` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`

| Metric | Result |
| --- | ---: |
| Generated non-prompt tokens | 23 |
| Silicon logits steps | 26 |
| Host/silicon token sequence match | true |
| Host/silicon text match | true |
| Silicon logits allclose `1e-4` | true |
| Silicon logits argmax match | true |
| Max logits abs diff vs ONNXRuntime | 1.9073486328125e-05 |
| Silicon logits wait | 1.27618545 s |
| Silicon logits-only token rate | 18.022459 token/s |
| Hybrid host-orchestrated wall token rate | 0.045928 token/s |
| Host-only ONNXRuntime wall token rate | 116.788408 token/s |

This is full audio-to-text and mathematically audited against host ONNXRuntime,
but it is still host-orchestrated. ET-SoC1 computes the final decoder logits
projection for every greedy decode step; host code still performs audio
preprocessing, encoder execution, decoder non-logits graph work, token control,
and tokenizer decode.

## Optimization delta

Previous full run:
`phase0/e2e_audit_runs/both_20260504-181738/e2e_audit_report.json`

The latest run removed the duplicate device-side reference tensor path from the
logits kernel. The host still audits the dumped logits against ONNXRuntime, so
correctness coverage is unchanged.

| Metric | Before | After |
| --- | ---: | ---: |
| Silicon logits wait | 1.57497894 s | 1.27618545 s |
| Silicon logits-only token rate | 14.603370 token/s | 18.022459 token/s |
| Hybrid wall token rate | 0.035035 token/s | 0.045928 token/s |
| Max logits abs diff | 1.9073486328125e-05 | 1.9073486328125e-05 |

The logits kernel improved by about 23.4 percent on silicon wait. Wall time is
still dominated by the current host launch/fetch loop.

Tile-width follow-up:

`phase0/e2e_audit_runs/both_20260504-211327/e2e_audit_report.json`
tested 10240-column logits tiles on the first 4 generated tokens. It passed the
same host-vs-silicon audit, but the silicon wait was effectively flat versus
8192-column tiles: 0.342200706 s for 7 silicon steps, or about 48.89 ms/step.
The 8192-column configuration remains the better default because it is already
staged for the full run and is not measurably slower.

## Native scheduler milestone

Native split smoke:
`phase0/NATIVE_SPLIT_SMOKE_SUMMARY.md`

The all-core scheduler smoke now passes with 16 harts:

- VPU/thread-0 mask: `0x5555`
- scalar/thread-1 mask: `0xaaaa`
- logits argmax match: true
- LayerNorm-shaped audit: true
- kernel wait: 0.00187290 s
- `hpmcounter3` cycles: 888160
- `hpmcounter6` L2 miss requests: 38519

This proves the practical split for a native Whisper executor: use all 8 minions
in one shire, with thread 0 doing VPU-heavy MatMul work and thread 1 doing
scalar graph work. The remaining blocker for full native Whisper is model/state
memory management and graph paging, not hart coordination.

## Argmax-only silicon token choice

2026-05-05 update: the logits kernel now has an argmax-only mode.  In this
mode ET-SoC1 computes each logits tile and returns only the tile argmax summary
instead of dumping the full logits tile for host stitching.  The host still runs
ONNXRuntime as the reference/audit oracle, but token choice comes from silicon
argmax summaries.

Full transcript run:
`phase0/e2e_audit_runs_argmax_only/both_20260505-094529/e2e_audit_report.json`

| Metric | Result |
| --- | ---: |
| Generated non-prompt tokens | 23 |
| Silicon logits steps | 26 |
| Host/silicon token sequence match | true |
| Host/silicon text match | true |
| Silicon argmax audit pass | true |
| Silicon tile active mask | `0xffff` |
| Max tile-argmax value diff vs ONNXRuntime | 1.9073486328125e-05 |
| Silicon logits wait | 1.32065762 s |
| Silicon logits-only token rate | 17.415566 token/s |
| Hybrid host-orchestrated wall token rate | 0.063373 token/s |

Short smoke run:
`phase0/e2e_audit_runs_argmax_only/both_20260505-094122/e2e_audit_report.json`

- Text: ` And so my fellow`
- 7 silicon steps, all argmax audits passed.
- Average silicon wait per logits step: 0.050923494 s.

This is more autonomous than the previous hybrid path because the host no
longer fetches and stitches full logits vectors to pick the token.  It is still
not a full native graph executor: audio preprocessing, encoder, decoder
non-logits nodes, KV cache control, and tokenizer decode remain on the host.
Full-logits allclose is intentionally not available in this mode because those
tiles are not fetched; the correctness gate is token sequence match plus
per-tile argmax/value agreement against ONNXRuntime.

## Final decoder LayerNorm plus logits argmax on silicon

2026-05-05 update: the hybrid path now moves the final decoder LayerNorm onto
ET-SoC1 as well.  The host supplies the ONNXRuntime tensor before
`/ln/LayerNormalization`; ET-SoC1 computes the final LayerNorm, then computes
the vocab logits tile argmax from the silicon-computed normalized vector.  The
host remains the audit oracle and compares both the token sequence and the
device LayerNorm/logits argmax summaries against ONNXRuntime.

Full transcript run:
`phase0/e2e_audit_runs_tail_ln/both_20260505-103351/e2e_audit_report.json`

| Metric | Result |
| --- | ---: |
| Generated non-prompt tokens | 23 |
| Silicon tail steps | 26 |
| Host/silicon token sequence match | true |
| Host/silicon text match | true |
| Silicon tail audit pass | true |
| Silicon tile active mask | `0xffff` |
| Max final LayerNorm abs diff vs ONNXRuntime | 2.2e-05 |
| Max tile-argmax value diff vs ONNXRuntime logits | 1.9073486328125e-05 |
| Silicon tail wait | 1.10474496 s |
| Silicon tail token rate | 20.819285 token/s |
| Hybrid host-orchestrated wall token rate | 0.057767 token/s |

Transcript:

` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`

This is the current most autonomous audited audio-to-text path.  It still uses
host ONNXRuntime for audio preprocessing, encoder execution, the remaining
decoder graph, KV-cache updates, token control, and text decode.  The next
native step is to keep the decoder state and a sequence of graph kernels resident
on ET-SoC1 so we stop paying one host launcher round trip per vocab tile.

## Parallel decoder tail across seven shires

2026-05-05 update: the final decoder LayerNorm plus vocab-tile argmax path can
now launch all seven vocab tiles from one ET runtime process.  Each tile uses a
different shire, so the host no longer starts seven independent launcher
processes per decode step.  This keeps the same audit surface as the sequential
tail path: token sequence match, text match, silicon LayerNorm check, and
per-tile argmax/value agreement against ONNXRuntime.

Full transcript run:
`phase0/e2e_audit_runs_tail_parallel/both_20260505-105350/e2e_audit_report.json`

| Metric | Result |
| --- | ---: |
| Generated non-prompt tokens | 23 |
| Silicon tail steps | 26 |
| Tail shires | `0,1,2,3,4,5,6` |
| Host/silicon token sequence match | true |
| Host/silicon text match | true |
| Silicon tail audit pass | true |
| Silicon tail wait | 1.5881506 s |
| Silicon tail token rate | 14.482254 token/s |
| Hybrid host-orchestrated wall token rate | 0.073919 token/s |

Transcript:

` And so my fellow Americans ask not what your country can do for you ask what you can do for your country`

The parallel launcher improves the host-orchestrated wall rate from 0.057767
token/s to 0.073919 token/s, about 28 percent.  The raw silicon tail wait is
worse than the sequential single-shire path because the seven shires contend for
the same memory traffic, but the single runtime launch/fetch path removes enough
host overhead to improve end-to-end hybrid throughput.

## Demo entry point

The current demo command is:

`local-artifacts/erbium_amp_probe/whisper-real/tools/run_whisper_silicon_demo.py`

Validated full demo run:
`phase0/demo_runs/both_20260505-110311/e2e_audit_report.json`

| Metric | Result |
| --- | ---: |
| Generated non-prompt tokens | 23 |
| Silicon tail steps | 26 |
| Host/silicon token sequence match | true |
| Host/silicon text match | true |
| Silicon tail audit pass | true |
| Silicon tail wait | 1.5882443 s |
| Silicon tail token rate | 14.481399 token/s |
| Hybrid host-orchestrated wall token rate | 0.078111 token/s |

The demo still uses ONNXRuntime on the host as the correctness oracle and for
the graph pieces that are not native yet.  It is the current reproducible
audio-to-text path, not the final full-native executor.

## Decoder LayerNorm node audit

2026-05-05 update: all 13 decoder LayerNorm nodes are now individually audited
on ET-SoC1 with real ONNXRuntime tensors from decoder step 3.

Report:
`phase0/decoder_layernorm_audit_runs/run_20260505-111622/decoder_layernorm_audit_report.json`

| Metric | Result |
| --- | ---: |
| Selected decoder LayerNorm nodes | 13 |
| All nodes pass | true |
| Max LayerNorm abs diff vs ONNXRuntime | 7e-06 |
| Mean silicon wait per LayerNorm | 0.000518 s |

This proves the scalar LayerNorm implementation used by the final tail can be
applied to every decoder block LayerNorm with the node-specific gamma/beta
parameters.  The remaining decoder-native work is now mostly graph plumbing,
Softmax/GELU/residual kernels, attention/cache updates, and combining the
already-audited MatMul families with these scalar nodes.

## Real encoder and decoder pieces already on silicon

Encoder block-0 MatMul audit:
`phase0/WHISPER_BLOCK0_MATMUL_AUDIT.md`

- 16 real Whisper encoder block-0 MatMul pieces were stitched from ET-SoC1
  silicon outputs and compared to ONNXRuntime tensors.
- All listed pieces pass `allclose_1e-4`.
- Covered families include QKV, attention score heads, attention value heads,
  attention output, MLP0, and MLP2.

Decoder token MatMul audit:
`phase0/WHISPER_DECODER_TOKEN_RATE.md`

- Real Whisper decoder per-token MatMul families were exported from
  ONNXRuntime, run on ET-SoC1 silicon, and compared back to their ONNX node
  outputs.
- Covered families include logits, MLP0, MLP2, self-attention QKV/out,
  cross-attention score/value, and self-attention score/value.

These are real ONNX tensor audits, not synthetic shapes. The missing native
piece is a device-resident graph executor that pages the weights/state and
connects these kernels with LayerNorm, softmax, GELU, residuals, KV-cache
updates, token control, and audio feature flow without host ONNXRuntime in the
loop.

## Encoder scalar audit partial sweep

2026-05-05 update: the full encoder scalar sweep was stopped after it became
clear that the harness, not the silicon kernels, was the bottleneck.  The run
was a Python orchestrator driving one stock 16 MiB launcher invocation per real
ONNX tensor tile.  Several large tiles stalled in SSH/rsync for one to two
minutes before retrying, so continuing the all-node sweep was not a useful use
of board time.

Partial report:
`phase0/encoder_scalar_ops_audit_runs/run_20260505-175731/partial_encoder_scalar_ops_summary.json`

| Metric | Result |
| --- | ---: |
| Completed tiles | 131 |
| Covered selected node indexes | 0..52 |
| Unique selected node indexes covered | 53 |
| Add tiles | 38 |
| Div tiles | 12 |
| Erf tiles | 12 |
| Mul tiles | 27 |
| Softmax tiles | 42 |
| Max abs diff vs ONNXRuntime | 1.329183578491211e-05 |

The important kernel result from this sweep is the Softmax fix: long encoder
Softmax rows failed when split across harts with cross-hart reduction, but pass
when each hart owns complete rows.  The next native step should not be another
exhaustive one-launch-per-tile audit.  It should be a resident or batched graph
executor that keeps tensors on device between kernels, with representative
real-ONNX audits for each op and shape class.
