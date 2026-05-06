# TFMA primitive validated on ET-SoC1

The Erbium **TensorFMA** unit — a separate matrix-FMA engine, distinct from
the per-minion VPU `fmadd.ps` — is callable, fast, and produces correct FP32
results. This unblocks the architectural pivot from VPU-bound DnCNN3
(currently 6% of VPU peak, ≈3.5 GOPS) to TFMA-bound (≈100 GOPS peak/shire).

## What "validated" means

Two independent silicon runs on `esperanto-soc4`, shire 0:

### 1. Throughput (`tfma_probe.c`)

A bisect-friendly probe (`PROBE_STAGE` 0…3) staged each setup step before
issuing N back-to-back `tensor_fma` calls and timing via `hpmcounter3`.
Hart 0 only; arows=acols=16, bcols=16, type=fp32.

| iters | cycles | cycles / TFMA |
|---:|---:|---:|
| 1,000 | 546,019 | 546.02 |
| 5,000 | 2,730,215 | 546.04 |
| 10,000 | 5,460,017 | 546.00 |

**Bit-exact 546 cycles per TFMA dispatch** (one minion). Each dispatch does
arows × acols × bcols = 16 × 16 × 16 = **4,096 MACs**. So:

- **7.50 MACs/cycle/minion** vs VPU's 4 MACs/cycle peak = **1.88×**
- 8 minions × 833 MHz = **50 GMACs/s = 100 GOPS** TFMA peak per shire

### 2. Correctness (`tfma_matmul_audit.c`)

Real 16×16 fp32 matmul: A and B random (np.random.seed=0xC0FFEE), C = A @ B.
Linked into the ELF as `_binary_A_bin_*` / `_binary_B_bin_*`, copied to
heap-resident buffers, `tensor_load`-ed into scratchpad lines 0..15 (A)
and 16..31 (B), `tensor_fma` (first_pass=1, type=fp32), `tensor_store` to
DRAM, `evict` for host visibility.

Host comparison vs `(A @ B)` computed in numpy:

| metric | result |
|---|---:|
| max_abs (TFMA vs host) | **9.537e-07** |
| mean_abs | 1.305e-07 |
| pass @ 1e-5 audit | **True** |
| any nan/inf | False |

Sub-ULP differences vs numpy come from TFMA's fused mul-add (single rounding)
vs numpy's separate mul-then-add. Same numerical floor (≈8.94e-7) we see in
the existing dncnn3 audits. **Within the campaign's audit gate.**

## Critical engineering details (not in headers)

These tripped up the first 5 build iterations — documented here so the next
TFMA kernel doesn't repeat:

1. **`csrr cycle` (CSR 0xC00) traps in U-mode.** Use `hpmcounter3` (0xC03)
   instead — the `dncnn3-pmc/pmc_snapshot` kernel proves it works.
2. **Set `NUM_HARTS=16` in the build, even if only hart 0 does work.**
   `NUM_HARTS` controls the linker's per-hart stack reservation. With
   `NUM_HARTS=1`, harts 1..15 spawn anyway and their stacks land inside
   `.text`, corrupting code.
3. **The launcher passes a non-zero `arg_area` pointing to the actual buffer
   base.** Use the `buffer_base_from_args(arg_area)` pattern from
   `dncnn3-pmc/pmc_snapshot.c` — `heap0_end - 16MB` is *not* the right
   computation when `arg_area != 0`.
4. **`tensor_store` cols=4 advances `src` by `srcinc * 2` per row.** The
   inner loop advances `src` after every odd col (`col & 1` true). For
   16-fp32-per-row store with TFMA's 2-FREGs-per-row layout, set
   `reg_stride=0` (srcinc=1), not `reg_stride=1`.
5. **Field encodings: arows = field+1, acols = field+1, bcols = (field+1)*4.**
   The header signatures don't make this obvious. arows=`0xF` ⇒ 16 rows,
   bcols=`3` ⇒ 16 columns.
6. **Always `evict` the destination before exit.** L1D is non-coherent;
   without `evict()` + `WAIT_CACHEOPS`, small writes stay in L1D and never
   reach DRAM, so `--dump_after` returns zeros.
7. **The "Error on kernel launch: 2" warning is normal launcher chatter** —
   the working untiled DnCNN3 also produces it. Don't chase that.

## DnCNN3 with TFMA — projection

| metric | VPU untiled (current best) | TFMA realistic | TFMA peak |
|---|---:|---:|---:|
| Throughput / shire | 3.45 GOPS | 30–70 GOPS (30–70% eff) | 100 GOPS |
| Time / inference | 30 s | **1.5–3.4 s** | 1.0 s |
| Single-shire fps | 0.034 | 0.3–0.7 | 1.0 |
| 32-shire fps | ~1 | **10–22** | 32 |

Realistic eff < 100% because: 36 TFMA dispatches accumulating per spatial
batch (acols=16 chunks across IC*K*K=576), scratchpad ping-pong for
weight + activation chunks, sync overhead between dispatches.

## Next steps (not done in this session)

- **Task C: TFMA-based 3×3 conv kernel for DnCNN3** (1–3 days). Im2col-style
  data marshaling, weight pre-pack, per-shire multi-hart split, 4 OC chunks
  × 36 IC*K*K accumulation chunks per pixel batch. Audit at 1e-5 vs ORT FP32.
- **Task D: FP16 TFMA path** (depends on C). Use `tfma_type_fp16` (FP16 in,
  FP32 acc), re-baseline audit against ORT-FP16 reference at ~1e-3 tolerance.
  Expected: another ~2× from packed-half operand bandwidth.

## Files

- `tfma_probe.c` — staged throughput probe (PROBE_STAGE 0..3)
- `tfma_correctness.c` — all-ones matmul (256/256 cells == 16.0)
- `tfma_matmul_audit.c` — random A, B vs host A@B (max_abs 9.5e-7)
- `gen_tfma_inputs.py` — host-side reference generation
- `/tmp/tfma_probe/*.elf` — built ELFs (1-hart probes)
- `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/runs/tfma_*` — soc4 run dirs
