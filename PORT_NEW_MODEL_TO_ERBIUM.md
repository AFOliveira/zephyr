# Porting a New Model to Erbium / ET-SoC1 — fresh-agent guide

A single document that combines everything we've learned porting models
to a single shire of Erbium ETSOC1 silicon. Performance-focused. Aimed
at someone new to the platform with a model they want to run.

This consolidates:
- The DnCNN3 optimization journey (47× speedup from a scalar baseline)
- The Whisper Tiny EN optimization journey (4× end-to-end, plus quantization-fidelity findings)
- The general build/stage/run flow

Device host: **`esperanto-soc3`** (ssh and rsync there). All silicon
artifacts live under `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/`.

---

## TL;DR — the 3-minute version

1. **Get your model into ONNX, inventory it.** Note the dominant ops, total MACs, weight size.
2. **Quantize it (PTQ)** to INT8 per-channel for the bulk-compute layers; keep FP32 at I/O boundaries (first/last conv, embeddings, layer norms, biases).
3. **Repack weights** into the layout TFMA expects: `[L][oc_tile][tap][oc_in_tile][ic]` for conv, equivalent for matmul.
4. **Write the kernel**: row-stripe across 8 minions, TFMA INT8 for hot loops, VPU `fmadd.ps` for the FP-shaped marshalling between layers.
5. **Build, stage to soc3, run, audit** vs ORT FP32. Iterate until max_abs < 5e-2.
6. **Measure with PMC** to find the actual bottleneck. Don't optimize on intuition — measure.
7. **Optimize in this order**: TFMA hardware path → VPU marshalling → padded layout → layer fusion / parallel splits → cooperative loads → eventually multi-shire.

What you're optimizing **is almost always one of**: per-channel weight memory layout, B-side activation packing, marshalling FP chain, or cross-hart synchronization. Memory bandwidth has never been the bottleneck on this hardware for any model we've ported.

---

## §1. Hardware in one page

```
ET-SoC1 chip
└── 32 shires (use 1 unless multi-shire is unblocked)
    └── 8 minions per shire
        ├── T0/T1 (SMT siblings — L1D may be split between them)
        ├── L1D 4 KB (non-coherent across minions)
        ├── L1 SCP 3 KB (explicit-addressed scratchpad)
        ├── 32 FREGs (256-bit each, used by VPU + TFMA)
        ├── TenA, TenB, TenC (matrix registers used by TFMA dispatch)
        └── TFMA unit — matrix multiply engine
            ├── FP32 path: 16×16×16 tile, 7.5 MACs/cyc/minion
            ├── FP16 path: 16×32×16 tile, 15 MACs/cyc/minion
            └── INT8 path: 16×64×16 tile, 51.5 MACs/cyc/minion ← typically the right choice
└── 4 MB L2 SCP per shire (shire-shared, explicit-addressed; underused so far)
└── ~17 GB DRAM (PCIe-addressable from host)
```

Key facts that shape every kernel:

- **64-byte cache line.** Native unit for all DMA, `tensor_load`, alignment.
- **No transparent L2 D-cache** on real Erbium silicon. cacheops resolve to "memory" (DRAM). Verified empirically: PMC shows DRAM at 1 MB/s during compute (≈99.99% idle), so the working set effectively fits in L1D + scratchpad.
- **L1D is non-coherent across minions.** Cross-hart sharing requires explicit `evict + WAIT_CACHEOPS + FENCE` discipline.
- **`tensor_load` silently rounds addresses to 64-byte boundaries.** Anything you pass it must be naturally 64-byte aligned; linker-placed binary blobs are not.
- **`tensor_fma(tenc_loc=1)` writes int32 results into the FP register file** without GCC seeing it. You need an explicit FREG clobber list around it.
- **L1D between SMT siblings (T0/T1)** can be split (disjoint cache sets) — naive shared-memory communication between T0 and T1 of one minion does not work.

Read `/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/erbium_quickstart.md` for the absolute minimum to launch any ELF. The rest of this guide assumes you've done that.

---

## §2. End-to-end model porting flow

Below is the path I'd take starting from a fresh ONNX. Each step has a
"what worked" and "what to skip" entry from the past two ports.

### Step 1 — Inventory the ONNX

Goal: understand where the work is.

```python
# inventory the model
import onnx
m = onnx.load("model.onnx")
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(ops)                                    # what op types
print(len(m.graph.node))                       # how many nodes
init_size = sum(len(i.raw_data) for i in m.graph.initializer)
print(f"weights: {init_size / (1<<20):.1f} MiB")
```

Decide your **hot path** — the op type / layer that has the most MACs.
Examples from our work:
- DnCNN3: 99% of MACs in 18 hidden 3×3 conv layers (64→64). Optimization
  effort goes there.
- Whisper Tiny EN encoder: MatMul, LayerNorm, Softmax. Conv1/Conv2 are
  not the bottleneck.
- Whisper Tiny EN decoder: cross-attention (~50%), final argmax (~10%),
  MLP, self-attention — multiple roughly-equal hot regions, each
  worth parallelizing.

### Step 2 — Build a host-side reference

Before any silicon work, get an FP32 ORT reference output for your test
input. **This is your ground truth for the entire optimization
journey.** Without it you're flying blind.

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
out = sess.run(None, {sess.get_inputs()[0].name: input_arr})[0]
out.tofile("ort_reference.bin")
```

For Whisper-style multi-stage models (encoder → decoder), audit at
each stage boundary. Don't compose all errors silently.

### Step 3 — Quantization plan (mixed-precision PTQ)

The pattern that worked for both DnCNN3 and Whisper:

| layer kind | quantize? | scheme |
|---|---|---|
| Bulk MatMul / Conv (the hot path) | **Yes, INT8** | Per-channel symmetric, weight scale = `max(|W[oc]|) / 127` |
| Activations between hot layers | **Yes, INT8** | Per-tensor symmetric from calibration set |
| Embeddings | Selectively | Token embedding for Whisper needed **per-dimension** scales (not per-tensor) — this was a critical fidelity finding |
| First conv / last conv (image models) | **No, FP32** | Boundary precision matters for the visible output |
| LayerNorm weights/biases | **No, FP32** | Tiny in size, sensitive |
| Biases (everywhere) | **Treat carefully** | Whisper showed encoder bias quantization caused wording errors |
| Mask / attention constants | **No, FP32** | Quantizing these triggered EOS collapse on Whisper |

**Calibration set**: as small as 32 augmented copies works for image
denoising (DnCNN3). For Whisper-class models you want a *real
distribution* — Whisper's "raw all-tensor INT8" approach failed on the
Hawking clip even though it worked on JFK, because the calibration was
not representative.

**Rule of thumb**: only quantize MatMul/Conv weights and (post-ReLU)
activations. Leave everything else FP32 unless you've measured it does
not matter. This matches what ORT dynamic-INT8 does and gave the cleanest
audit pass on both ports.

### Step 4 — Repack weights for the hardware

The TFMA expects a specific in-memory layout for its A and B operands.
For a conv with shape `[OC, IC, ky, kx]` from ONNX, the TFMA-friendly
A-pack is:

```
weight_pack[L][oc_tile][tap][oc_in_tile][ic]
  L           = layer index
  oc_tile     = OC / 16 (TFMA arows = 16)
  tap         = ky * 3 + kx for 3×3 conv (= 9 dispatches per output tile)
  oc_in_tile  = 0..15
  ic          = 0..63 for INT8 (TFMA acols = 64 for INT8)
```

A single `tensor_load` of 16 lines × 64 bytes = 1024 contiguous bytes
fills SCP[0..15] with one (oc_tile, tap)'s worth of weights. We pre-pack
once on the host, ship the packed blob with the ELF (or DMA it
separately).

**Forget about INT4** — sysemu confirms TFMA only has fp32/fp16/int8
hardware paths (`tfma_type_rsvd_2/4/5/6/7` all throw illegal-type). INT4
unpacking in software costs more than it saves.

### Step 5 — Write the kernel

The skeleton that worked for both DnCNN3 (image conv) and Whisper
(transformer matmul):

```c
int main(uintptr_t arg_area)
{
    const uint32_t raw_hart = get_hart_id();
    if (raw_hart >= 16u || (raw_hart & 1u) != 0u) return 0;  /* T1 idle */
    const uint32_t hart_id = raw_hart >> 1;        /* 0..7 */
    setM0MaskFF();

    uint8_t *base = (uint8_t *)((uintptr_t)heap0_end - 64u * 1024u * 1024u);
    /* Compute this hart's stripe of the output */
    const uint32_t stripe_start = (TOTAL_ROWS * hart_id) / 8u;
    const uint32_t stripe_end   = (TOTAL_ROWS * (hart_id + 1u)) / 8u;

    /* 1. cooperative one-time init: zero buffers, copy weights to
     *    64-byte-aligned region (tensor_load alignment requirement) */
    cooperative_zero_init();
    cooperative_copy_to_aligned_buffer();
    bench_barrier();

    /* 2. boundary-precision input layer (FP32) */
    if (is_t0) conv_first_fp32(...);
    bench_barrier();

    /* 3. the hot loop — usually a stack of TFMA INT8 layers */
    for (L = 0; L < HIDDEN_LAYERS; L++) {
        /* halo invalidation if this layer reads cross-hart rows */
        evict_halo_rows();
        WAIT_CACHEOPS;

        run_layer_via_tfma_int8(...);

        /* push our writes to DRAM so other harts read fresh data next layer */
        evict_my_stripe();
        WAIT_CACHEOPS;
        bench_barrier();
    }

    /* 4. boundary-precision output layer (FP32) */
    if (is_t0) conv_final_fp32(...);
    bench_barrier();

    /* 5. dump output, signal done */
    if (raw_hart == 0u) write_summary();
    return 0;
}
```

Strategy details that matter:

- **Row-stripe partition** (each minion owns N/8 rows) is the simplest
  way to use 8 minions. For DnCNN3 this was clearly best. For Whisper
  encoder/decoder, more granular splits work better — see §5 for the
  parallel-decoder pattern.
- **Mixed precision**: scalar FP32 at the boundaries, TFMA INT8 in the
  middle.
- **VPU `fmadd.ps`** for the FP-shaped marshalling between INT8 layers
  (dequant → bias → ReLU → requantize). 8-lane SIMD, 3× faster than
  scalar FP.
- **Padding** boundaries (e.g., 1-pixel halo for a 3×3 conv) so the
  inner B-pack code has no bounds checks. Cheap (1.5% memory bloat),
  meaningful win.

### Step 6 — Build, stage, run

See `erbium_quickstart.md` for the absolute minimum. For development
iterations we use:

```bash
# Build the kernel
bash build_my_model.sh   # ~few seconds

# Stage everything once on soc3
scp /tmp/dncnn_tfma/my_kernel.elf root@esperanto-soc3:/root/afonso/.../erbium-amp-probe/stream_data/
scp inputs/*.bin root@esperanto-soc3:/root/afonso/.../erbium-amp-probe/stream_data/

# Iterate — runs entirely on soc3, NO ssh per frame:
ssh root@esperanto-soc3 \
  '/root/afonso/.../erbium-amp-probe/onsoc_iterator.sh 3'
```

Lesson learned: **always run the iterator on the device host itself,
not from your laptop.** ssh+scp+rsync per frame adds 30-60s overhead
that has nothing to do with the kernel. Per-frame on-host: **~2.6s**.
Per-frame from laptop: 60s.

### Step 7 — Audit + iterate

```python
import numpy as np
ref = np.fromfile("ort_reference.bin", dtype=np.float32).reshape(...)
out = np.frombuffer(open("dump.bin","rb").read()[OUTPUT_OFFSET:OUTPUT_OFFSET+SIZE],
                   dtype=np.float32).reshape(...)
print(f"max_abs = {np.abs(out - ref).max():.4e}   mean_abs = {np.abs(out - ref).mean():.4e}")
```

Pass gates that proved sufficient for our two models:
- **DnCNN3 image denoising**: max_abs ≤ 5e-2 vs FP32 ORT (visually invisible)
- **Whisper Tiny EN**: token-IDs **exact match** vs host raw-INT8 audit (any token mismatch = fail)

Different models need different gates. Image regression: max_abs or PSNR. Discrete-output models: exact match on token / class IDs. Build the gate before claiming pass.

---

## §3. Performance optimization playbook

The lever ranking below is empirical — what actually moved wall time on
the two models we've ported. Try in order. Stop when you hit your
performance target.

### 3.1 — TFMA INT8 hardware dispatch (always do this first)

**What**: Replace scalar int8×int8→int32 inner loops with
`tensor_fma(type=3, ...)`.

**Speedup observed**:
- DnCNN3 inner conv: 13.4 → 0.044 cyc/MAC = **305× tighter on compute**
- Wall time: 106.7 s → 4.0 s = **26.6×**

**Setup**: Per output tile (16 OC × 16 spatial), 9 dispatches for 3×3
conv (one per ky×kx), `first_pass=1` only on tap 0, `tenc_loc=1` only
on tap 8. For matmul, similar pattern with one or more dispatches per
output tile.

**Verdict**: ✅ Essential. If you're doing conv/matmul on Erbium and
you're not using TFMA, you're leaving 7-30× on the table.

### 3.2 — VPU `fmadd.ps` for FP-shaped post-processing

**What**: Use the 8-lane packed FP32 SIMD for any per-output FP math
between TFMA dispatches: dequant, bias, activation (ReLU/GELU), requant.

**Speedup observed**:
- DnCNN3 marshalling: 100 ms/layer → 32 ms/layer (**3.1× on that block**)
- Saved 1.24 s total wall time (from 4.0 s → 2.77 s)

**Recipe** (DnCNN3 dequant chain):
```c
fcvt.ps.pw f0, f0    // int32 → fp32 (8 lanes)
fmadd.ps  f0, f0, f4, f5    // f0 = f0 * combined_scale + combined_bias
fmax.ps   f0, f0, f7    // ReLU at 0
fmin.ps   f0, f0, f6    // clamp at int8 max (127)
fcvt.pw.ps f0, f0, rne    // round-nearest-even, fp32 → int32
```

**Verdict**: ✅ Worth doing. Free leverage if you're already doing FP math
between INT8 layers.

**Rejected variant**: VPU `fscw.ps` scatter for cross-stride stores. Same
speed as scalar — silicon issues 16 individual cache-line stores either
way. Don't bother.

### 3.3 — Padded buffers + branch-free packing

**What**: Pad activation buffers with zero halos so the inner pack/unpack
loop has no bounds checks. Drop `volatile` from staging buffers so the
compiler can fuse byte stores into wider stores.

**Speedup observed (DnCNN3)**: 2.77 s → 2.43 s wall time. Pack_B per
layer: 62 ms → 47 ms.

**Cost**: ~1.5% memory increase for 1-pixel halo on a 240×320 image.
Trivial.

**Verdict**: ✅ Easy win. Should be considered any time a hot loop has
boundary-conditional accesses.

### 3.4 — Parallel decoder strategies (matters most for transformers)

For models with parallelizable internal structure (e.g., independent
attention heads, vocab projection), splitting work across harts can give
larger wins than data-side optimizations.

**Whisper Tiny EN decoder** (what worked, in order applied):
| split | result | leverage |
|---|---:|---|
| Parallel vocab projection / argmax | 79.7 → 39.7 s | 2.0× — vocab scan is naturally parallel |
| Parallel cross-attention heads | 39.7 → 28.2 s | 1.4× — split across heads |
| Parallel MLP (FC1/GELU/FC2 per output block) | 28.2 → 17.8 s | 1.6× — output-block parallelism |
| Parallel self-attention (QKV+heads+output) | 16.2 → 11.0 s | 1.5× |
| Parallel cross query/output projections | 11.0 → 9.0 s | 1.2× |

Cumulative: **decoder went 79.7 s → 9.0 s = 8.86× from parallel splits alone.**

**Whisper encoder** (data-locality-driven gains):
| change | result | leverage |
|---|---:|---|
| Row-major INT8 matvec | 149 → 116 s | 1.3× — better dense-weight access |
| Attention V locality (contiguous per-token V) | 116 → 64 s | 1.8× — biggest single fix |
| Cross-cache locality | 64 → 61 s | small |
| Attention batch-2 (reuse K/V row loads) | 60 → 52 s | 1.2× |
| Dense batch-2 (reuse INT8 weight rows) | 52 → 47 s | 1.1× |

**Verdict**: ✅ Single biggest performance lever for transformer-class
models. Look at your decoder and ask "which loops are independent across
output dimensions?" — those can be split.

**Important caveat**: The hart count you pick matters. Whisper found
**16 active harts** beats 8 or 12. DnCNN3 used 8 (T0 only). Don't
assume one or the other; benchmark.

### 3.5 — Layer fusion / locality

**What**: Reduce intermediate writes between layers when possible.

DnCNN3 doesn't benefit (DRAM is already at 0.001% utilization). Whisper
benefits from data-flow locality fixes (attention V, cross-cache,
batch-2 reuse). Generally:
- If your PMC shows DRAM is at >50% bandwidth, locality fixes matter.
- If DRAM is essentially idle, locality fixes are unlikely to help; focus on compute.

### 3.6 — Cooperative weight loads (`tensor_coop`)

**What**: All 8 minions broadcast the same weight load. Reduces L2 read
pressure for shared A operands.

**Status**: Pattern proven in `coop_tl_tfma_fc.c`, not used in our
production kernels (we ran out of session budget on DnCNN3, didn't
profile-justify it on Whisper). Estimated 5-10% speedup.

**Verdict**: ⏸️ Worth trying, especially on weight-reused workloads.

### 3.7 — SMT T1 producer pattern

**What**: Activate T1 sibling thread on each minion, have it pre-pack
B-side data while T0 runs the TFMA dispatches.

**Status**: Attempted on DnCNN3, **deadlocked due to L1D split-mode
between SMT siblings** on real silicon. Eviction-based sync didn't
unblock it; needs Esperanto guidance on the right primitive.

Whisper used 16 harts (= T0 + T1 of each minion) but with NO producer-
consumer between siblings — both threads do independent work. Apparently
that pattern works.

**Verdict**: ⏸️ Use 16 active harts (T0 + T1) for transformer-class
parallel work. Avoid SMT producer-consumer between siblings without
talking to Esperanto.

### 3.8 — Multi-shire scaling (32×, currently policy-blocked)

**What**: `--shire-mask 0xFFFFFFFF` to spread work across all 32 shires.

**Status**: Single-shire constraint per `CLAUDE.md`. When the kernel is
ready, the scaling should be ~linear up to 32× (modulo cross-shire
sync overhead).

For DnCNN3: 2.43 s → ~76 ms = **13 fps**.
For Whisper: 55.7 s → ~1.7 s end-to-end.

**Verdict**: 🚧 Eventual ceiling-breaker. Plan for it but don't block on it.

### 3.9 — INT4 weights — DON'T (no HW path)

We checked. TFMA only has fp32/fp16/int8 paths. INT4 in software requires
unpack at dispatch time = pure overhead on a CPU-bound kernel. **Skip.**

If a future Erbium silicon adds an INT4 TFMA path, revisit.

### 3.10 — Block-sparse activations / TFMA zero-skip

We measured the actual quartet-zero rate on natural ReLU activations:
**2.89% average**. Even with perfect TFMA zero-skip, ceiling is
**~0.4% wall savings**. Not worth retraining for. **Skip.**

---

## §4. Silent gotchas — every one of these will silently corrupt output

These are the bugs that DON'T crash the kernel, just produce wrong
numbers. Each one cost real debugging time on past ports.

### 4.1 `tensor_load` rounds addresses to 64-byte boundaries

**Symptom**: TFMA dispatches give garbage results; per-tile dump shows
weights were read 60 bytes earlier than intended.

**Cause**: `tensor_load` encodes `addr & 0xFFFFFFFFFFC0`. Linker-placed
`_binary_*_start` symbols typically land at 4-byte aligned addresses.

**Fix**: Copy linker blobs to a `__attribute__((aligned(64)))` static
buffer at runtime. Cooperative across 8 harts costs ~1.6 ms.

```c
static int8_t static_a_pack[A_PACK_BYTES] __attribute__((aligned(64)));

if (raw_hart == 0u) {
    /* one-time copy from linker blob to aligned buffer */
    for (uint32_t i = 0; i < A_PACK_BYTES; i++)
        static_a_pack[i] = WEIGHT_BLOB[i];
    evict(static_a_pack, A_PACK_BYTES);
    WAIT_CACHEOPS;
    FENCE;
}
bench_barrier();
```

### 4.2 `tensor_fma(tenc_loc=1)` clobbers FREGs invisibly to GCC

**Symptom**: First tile of layer 0 is bit-exact correct, subsequent
tiles diverge. The output gets progressively worse across layers.

**Cause**: `tenc_loc=1` writes int32 results into the FP register file.
The asm in `tensors.h` doesn't declare this in its clobber list. GCC
hoists FP scalars into FREGs that TFMA silently overwrites.

**Fix**: Explicit FREG clobber barrier after every `tensor_store` (or
after `tensor_fma` with `tenc_loc=1`):
```c
__asm__ __volatile__("" ::: "memory",
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
    "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
    "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
```

### 4.3 BSS is not auto-zeroed in this U-mode runtime

**Symptom**: Halo regions or first-iteration values are random.

**Cause**: The U-mode runtime doesn't zero BSS at startup.

**Fix**: Cooperative zero-init across all harts at kernel start. Use
`uint64_t` writes for speed.

### 4.4 `fdiv.s` can hang in U-mode

**Symptom**: Kernel runs the timeout, launcher segfaults.

**Cause**: Some FP divides hang in this U-mode runtime path.

**Fix**: Precompute reciprocals on the host, link as `.rodata`. Or
rewrite `x / k` as `x * (1.0f/k)` where `1.0f/k` is link-time-constant.

### 4.5 Compiler can DCE writes that go through asm-only readers

**Symptom**: Pack-B writes don't survive to memory; tensor_load reads
old data.

**Cause**: If you write to a buffer in C, then `tensor_load` reads it
via a CSR-write asm with no memory clobber, GCC may delete the writes
as dead code.

**Fix**: Either declare the staging buffer `volatile`, or add explicit
`__asm__ __volatile__("" ::: "memory")` between the writes and the asm.

### 4.6 SMT siblings (T0/T1) may have split L1D

**Symptom**: T0 → T1 communication via shared `volatile uint32_t flag`
hangs because the writes never become visible.

**Cause**: In split-mode L1D (the default), T0 and T1 use disjoint
cache sets. Naive shared memory doesn't propagate.

**Fix**: Don't do producer-consumer between SMT siblings until Esperanto
documents the right primitive. Have both threads do independent work
that uses different memory regions. Whisper uses 16 active harts this
way successfully.

### 4.7 Cross-minion writes need explicit eviction

**Symptom**: Hart B reads stale data that hart A wrote one barrier ago.

**Cause**: L1D is non-coherent across minions. Hart A's writes are in
A's L1D until evicted. Hart B's reads hit B's L1D first.

**Fix**: 
```c
/* Hart A writes */
my_data[i] = ...;
__asm__ __volatile__("" ::: "memory");
FENCE;
evict(my_data, sizeof(my_data));    /* push A's L1D → DRAM */
WAIT_CACHEOPS;
bench_barrier();

/* Hart B before reading */
evict(my_data, sizeof(my_data));    /* invalidate B's L1D */
WAIT_CACHEOPS;
FENCE;
read_value = my_data[i];            /* fetch from DRAM */
```

### 4.8 Always end with eviction on data the host needs

**Symptom**: `dump.bin` shows zeros where the kernel definitely wrote
real values.

**Cause**: Writes still in L1D. The post-run DMA reads from DRAM.

**Fix**: Before kernel exits, `evict + WAIT_CACHEOPS + FENCE` on
every region you want the host to see.

---

## §5. Lessons learned (model-specific)

### From DnCNN3 (image conv, 47× speedup, 2.43 s/frame)

- **TFMA INT8 alone delivered 26.6× speedup**. Everything else combined was 1.7×. Get TFMA working first; perfect everything else later.
- **Pack_B (NHWC → TFMA-friendly layout) became the new bottleneck** after TFMA. Tried twice to eliminate it (HW interleave-load, k0-major layout), both blocked by alignment math. Worth ~3-5× more if a future Esperanto firmware adds an aligned-stride load primitive.
- **PMC measurements drove every optimization decision.** Without them I'd have chased memory optimizations that the same measurements showed couldn't help.
- **Two silent silicon bugs cost ~3 days each.** Linker alignment, FREG clobber. Documented them as gotchas above so future ports don't repeat.

### From Whisper Tiny EN (transformer, 4× speedup, 55.7 s end-to-end)

- **Parallel decoder splits delivered 8.86× alone.** Cross-attention heads, MLP output blocks, vocab projection, self-attention QKV+heads — every independent dimension is fair game.
- **Encoder data-locality fixes mattered (1.8× from V-locality alone).** Whisper has more cache pressure than DnCNN3 because the model is bigger; locality work paid off here in a way it didn't on DnCNN3.
- **Quantization fidelity is hard to predict.** Raw-INT8 worked on JFK, collapsed to immediate EOS on Hawking. Per-dimension token-embedding scales were the key fix.
- **Don't quantize blindly.** ORT dynamic-INT8 only quantizes MatMul weights. Naively quantizing all float initializers (LayerNorm params, biases, embeddings, masks) caused EOS collapse. The cheap-and-correct rule: quantize MatMul/Conv weights and post-activation tensors, leave everything else FP32.
- **Conv front-end is fidelity-sensitive.** Even though it's not the runtime bottleneck, conv quantization noise propagated through the whole encoder and changed transcribed words. Consider keeping convs FP32 even if they're INT8-clean on quick eval.

### Rejected optimizations (verified non-useful)

| Tried | Result | Why |
|---|---|---|
| INT4 weights | No HW path | Sysemu confirms only fp32/fp16/int8 |
| Block-sparse activations / TFMA zero-skip | 0.4% wall savings ceiling | Quartet-zero rate is only 2.89% on natural activations |
| VPU `fscw.ps` scatter for B-pack | No speedup vs scalar | Silicon issues 16 stores either way |
| 8 active harts (Whisper) | Slower than 16 | More harts win for transformer parallel splits |
| 12 active harts (Whisper) | Slower than 16 | Same |
| Encoder dense batch-4 (Whisper) | Slower than batch-2 | Register/cache pressure outweighs reuse |
| K0-outer pack loop reorder (DnCNN3) | 1.8× slower | Read pressure on source cache lines beat write coalescing |

---

## §6. Reproduction — all the commands

### Build kernel + repack weights

```bash
DIR=/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn

# regenerate weight blobs (runs once)
python3 $DIR/tools/quantize_dncnn_perchannel.py
python3 $DIR/tools/pack_tfma_int8_a.py

# build
bash $DIR/build_tfma_int8_v2.sh
# produces /tmp/dncnn_tfma/int8_tfma_v2.elf
```

### Stage everything to soc3 (one time)

```bash
DIR=/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn
SOC=root@esperanto-soc3
PARENT=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe

ssh $SOC "mkdir -p $PARENT/stream_data"
scp /tmp/dncnn_tfma/int8_tfma_v2_stream.elf $SOC:$PARENT/stream_data/kernel.elf
for f in $DIR/dncnn3_stream_input_*.bin $DIR/dncnn3_other_input_*.bin; do
    scp $f $SOC:$PARENT/stream_data/
done
```

### Run the iterator on soc3 directly (fast — ~3s/frame)

```bash
ssh root@esperanto-soc3 \
  '/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/onsoc_iterator.sh 3'
```

The iterator script is in
`/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn/onsoc_iterator.sh`
on the dev side; copy to soc3 once.

### Pull dumps + audit

```bash
LATEST=$(ssh root@esperanto-soc3 'ls -td /root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/runs/onsoc_iter_* | head -1')
LOCAL=/tmp/$(basename $LATEST)
mkdir -p $LOCAL
rsync -aqz root@esperanto-soc3:$LATEST/ $LOCAL/
# audit each dump_*.bin against its ORT reference (see DnCNN3 audit script)
```

### Visualize results

```bash
python3 $DIR/tools/build_viz_html.py
# opens /tmp/dncnn_visualization.html in any browser
# 6 test images, input/output/diff/slider, embedded as base64 PNG
```

---

## §7. File index — where to find things

### From the DnCNN3 port (`local-artifacts/erbium_amp_probe/luxonis-real-dncnn/`)

| file | purpose |
|---|---|
| `dncnn3_tfma_int8_v2.c` | the production kernel (TFMA INT8 + VPU marshalling + padded layout) |
| `tools/quantize_dncnn_perchannel.py` | per-channel INT8 PTQ from ONNX |
| `tools/pack_tfma_int8_a.py` | repack INT8 weights into TFMA layout |
| `tools/build_viz_html.py` | self-contained HTML visualization (input/output/diff per image) |
| `tools/render_dncnn.py` | render single input/output pair as PNG |
| `build_tfma_int8_v2.sh` | full build (kernel + linked .o blobs) |
| `run_int8_tfma_v2.sh` | stage + flock-locked single run |
| `onsoc_iterator.sh` | on-device-host iterator (run on soc3 directly) |
| `optimizations.md` | full DnCNN3 optimization journal |
| `erbium_quickstart.md` | minimum-viable launch guide |

### From the Whisper port

| file | purpose |
|---|---|
| `/home/afonso/zephyr/optimizations_new.md` | Whisper Tiny EN optimization log (parallel splits, fidelity findings) |
| `local-artifacts/erbium_amp_probe/whisper-real/phase1/...` | Whisper kernels and run history |

### Cross-cutting

| file | purpose |
|---|---|
| `local-artifacts/erbium_amp_probe/optimization-kb/optimization_knowledge_base.md` | aggregate knowledge base across all 5 models we've touched |
| `/home/afonso/et-platform-vidas/sw-sysemu/insns/tensors.cpp` | TFMA semantics (read this when in doubt about a dispatch) |
| `/home/afonso/et-platform-vidas/sw-sysemu/insns/packed_*.cpp` | full VPU instruction list (`.ps` for FP, `.pi` for INT32) |
| `/home/afonso/et-platform-vidas/et-common-libs/build-headers/erbium-soc1sim-staged-include/erbium/isa/` | the user-mode headers (`hart.h`, `cacheops-umode.h`, `tensors.h`, `barriers.h`, `utils.h`) |

---

## §8. Checklist before you claim a port works

- [ ] Audit max_abs (or token-exact-match) passes against FP32 ORT on at least 3 distinct test inputs
- [ ] Reproduces bit-exact across 5+ runs
- [ ] Tested with `-O2` *and* `-O3` builds
- [ ] All `tensor_load` source addresses 64-byte aligned (verify with `nm | grep` after build)
- [ ] All `tensor_fma(tenc_loc=1)` followed by FP-clobber asm
- [ ] All static buffers explicitly zero-init'd at kernel start (BSS isn't auto-zeroed)
- [ ] No `fdiv.s` instructions in the hot path (precompute reciprocals)
- [ ] All cross-hart shared data has explicit `evict + WAIT_CACHEOPS + FENCE` discipline
- [ ] Output region evicted before kernel returns
- [ ] T1 harts (odd raw_hart_id) either return early *or* participate in barriers consistently — not in between
- [ ] Audited at every model boundary (encoder out, decoder mid-step, etc.) for multi-stage models — don't compose all errors silently
- [ ] PMC capture done on the final config so you know where time goes (and what's left to optimize)

---

## §9. What this guide doesn't cover (yet)

- **Real mid-run DMA streaming** with kernel polling and a custom libetrt host program. We sketched the scaffolding (kernel-side `DNCNN_STREAMING` flag), the host program is the missing piece. ~1-2 days of work to build properly.
- **Multi-shire scaling**. The kernels we have should drop in trivially (just change `shire_mask`), but we haven't validated that the cross-shire barriers work as advertised.
- **Models other than image-denoising and short-clip ASR**. Different op shapes (e.g., diffusion's huge MLPs, vision transformers' large attention) might need different tile/parallel strategies.
- **Real-data calibration sets**. Both ports used small synthetic calibration. For a production deployment with arbitrary user data, you want a much larger and more representative calibration set.
- **PSNR / SSIM / WER metrics**. We use max_abs and exact-token-match. For a real product you'd want perceptual / task-specific metrics.

---

## §10. Recommended reading order for a new agent

1. `erbium_quickstart.md` — 10 minutes, get a hello-world running
2. This document, §1 (hardware) — 5 minutes
3. This document, §4 (gotchas) — 10 minutes, **read this before writing your first kernel**
4. `dncnn3_tfma_int8_v2.c` source — 30 minutes, see a real kernel
5. This document, §3 (optimization playbook) — when you're ready to optimize
6. `optimizations.md` (DnCNN3 full journal) — when you want the long form
7. `optimizations_new.md` (Whisper) — when porting a transformer

By that point you'll have a sense of the platform. The rest is your model's specific shape.

---

## TL;DR (again, after all that)

- **TFMA INT8** for hot conv/matmul. Always. ~7-30× over scalar.
- **Mixed precision**: keep FP32 at I/O boundaries, embeddings, layer-norms, masks, biases (Whisper showed bias quant breaks fidelity).
- **VPU `fmadd.ps`** for the FP-shaped marshalling. ~3× over scalar.
- **Padded buffers + branch-free pack** in inner loops.
- **Parallel decoder splits** (cross-attention heads, MLP output blocks, vocab projection) for transformers. Up to ~9× from this category alone.
- **Run iterator on the device host directly**, not from your laptop. 60s → 3s per frame.
- **Watch out for**: `tensor_load` 64-byte rounding, FREG clobber by TFMA, BSS not zeroed, `fdiv.s` hangs, DCE on asm-only readers, SMT split L1D.
- **PMC-driven decisions only.** Don't optimize on intuition.
- **Audit at every stage.** Don't compose silent errors.

Multi-shire (32×) is the eventual ceiling-breaker. SMT producer-consumer
between siblings is currently blocked. Everything else is a smaller
multiplier on top of the basics above.

Welcome to Erbium. Build something good.
