# Run an ELF on Erbium (ET-SoC1) — quickstart

Three steps: build, stage to socx, launch. That's it.

---

## 1. Build

```bash
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc
INC=/home/afonso/et-platform-vidas
ROOT=/home/afonso/zephyr/local-artifacts/erbium_amp_probe

"$GCC" \
  -O2 -nostdlib \
  -march=rv64imfc -mabi=lp64f -mcmodel=medany \
  -fno-zero-initialized-in-bss -ffunction-sections -fdata-sections \
  -I"$INC/et-common-libs/build-headers/erbium-soc1sim-staged-include" \
  -I"$INC/hal/platform/erbium/include" \
  -I"$INC/hal/platform/etsoc/include" \
  -I"$INC/et-common-libs/include" \
  -Wl,--gc-sections -Wl,--no-warn-rwx-segments \
  -Wl,--defsym=region0_size=0x04000000 \
  -T "$INC/et-common-libs/share/erbium-soc1sim/erbium.ld" \
  -o /tmp/my_kernel.elf \
  my_kernel.c \
  "$ROOT/hart-report/hart_report_crt.S" \
  "$INC/erbium-examples/runtime/erbium-soc1sim/layout.c"
```

If you have raw data files (weights, input image), wrap each into a `.o`
and add to the link:

```bash
OBJCOPY=/home/afonso/et/bin/riscv64-unknown-elf-objcopy
"$OBJCOPY" -I binary -O elf64-littleriscv -B riscv \
  --rename-section .data=.rodata,alloc,load,readonly,data,contents \
  my_data.bin my_data.o
# then add my_data.o to the gcc command above
```

In the kernel, reference the data via:
```c
extern const unsigned char _binary_my_data_bin_start[];
```

---

## 2. Stage to the device host

```bash
scp /tmp/my_kernel.elf root@esperanto-socx:/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/
```

---

## 3. Launch it

```bash
ssh root@esperanto-socx << 'REMOTE'
BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT="$BASE/erbium-amp-probe"
export LD_LIBRARY_PATH="$BASE:$PARENT"

mkdir -p /tmp/myrun && cd /tmp/myrun

flock -x -w 600 /var/lock/etsoc-shire0.lock \
  "$PARENT/erbium_soc1sim_argbuf" \
    --elf-load "$PARENT/my_kernel.elf" \
    --shire 0 \
    --file_load 0x0,"$PARENT/zero64k.bin" \
    --dump_after dump.bin \
    --timeout 60

grep "Kernel wait seconds" run.log 2>/dev/null
ls -la dump.bin
REMOTE
```

**Launcher flags**:
- `--elf-load <file>` — your kernel
- `--shire 0` — pin to shire 0 (project policy)
- `--file_load <addr>,<file>` — DMA `<file>` to device DRAM offset `<addr>`. Repeat for multiple files.
- `--dump_after <file>` — DMA the heap region (16 MB) back to host file after the kernel completes
- `--timeout <secs>` — kill the kernel if it runs longer

**The flock is required** so two users don't collide on shire 0.

---

## 4. Read the output

```bash
scp root@esperanto-socx:/tmp/myrun/dump.bin /tmp/dump.bin

python3 << 'EOF'
import numpy as np
data = open("/tmp/dump.bin","rb").read()
# read whatever your kernel wrote at whatever offset, e.g.:
# val = int.from_bytes(data[0:4], "little")
# img = np.frombuffer(data[0x300000:0x300000+240*320*4], dtype=np.float32).reshape(240,320)
EOF
```

The dump is the first **16 MB** of the kernel's heap region, byte-for-byte
as the kernel left it.

---

## Minimum-viable kernel (`my_kernel.c`)

```c
#include <stdint.h>
#include "erbium/isa/hart.h"
#include "erbium/isa/cacheops-umode.h"

extern char heap0_end[];

int main(uintptr_t arg_area)
{
    if (get_hart_id() != 0u) return 0;   /* only hart 0 writes */

    uint8_t *base = (uint8_t *)((uintptr_t)heap0_end - 64u * 1024u * 1024u);
    volatile uint32_t *out = (volatile uint32_t *)base;
    out[0] = 0xC0FFEEu;
    out[1] = 42u;

    /* push our writes to DRAM so --dump_after picks them up */
    evict((const void *)out, 8u);
    WAIT_CACHEOPS;
    FENCE;
    return 0;
}
```

Reading back: `dump[0:8]` should be `EE FF C0 00 2A 00 00 00`.

---

## Critical gotchas (silent failure modes — every one of these will produce wrong output with no error)

1. **`tensor_load` rounds addresses to 64-byte boundaries.** Anything you
   pass it must be 64-byte aligned. Linker-placed `_binary_*_start`
   symbols typically are *not*. Copy to a `__attribute__((aligned(64)))`
   buffer first.

2. **`tensor_fma(tenc_loc=1)` clobbers FP registers** but the asm doesn't
   declare it. Add after every `tensor_store`:
   ```c
   __asm__ __volatile__("" ::: "memory",
       "f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11",
       "f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22",
       "f23","f24","f25","f26","f27","f28","f29","f30","f31");
   ```

3. **BSS is not zeroed.** Static buffers contain garbage at startup.
   Explicitly zero anything that needs to start at zero.

4. **`fdiv.s` can hang.** Replace `x / k` with `x * (1.0f/k)` (the
   reciprocal computed at host or at link time as a constant).

5. **L1D is non-coherent across minions** and may be split between SMT
   siblings (T0/T1 of one minion). Cross-hart shared data needs explicit
   `evict + WAIT_CACHEOPS + FENCE` discipline on writers, and the same
   on readers before the read.

6. **Always end with `evict + WAIT_CACHEOPS + FENCE`** on anything you
   want the host to see in `dump.bin`. Otherwise it stays in L1D and
   the post-run DMA reads stale DRAM.

---

## When it goes wrong

- **Launcher segfaults with `SIGSEGV(11)`** in run.log: kernel timed out;
  it's a known launcher bug. Look at the kernel for an infinite loop
  / deadlock, not the launcher.
- **`dump.bin` is correct except for one wrong region**: probably missed
  an `evict` after the writes to that region.
- **`dump.bin` is all garbage but the kernel said `Kernel wait seconds:
  X`**: the kernel completed but produced wrong output. Bisect by
  writing checkpoint values to known offsets at every phase.

---

## Reference: existing scripts in this repo

| script | what it does |
|---|---|
| `build_tfma_int8_v2.sh` | full build (compiles kernel + links many `.o` blobs) |
| `run_int8_tfma_v2.sh` | full stage + flock-locked run + rsync dump back |
| `tools/render_dncnn.py` | render a 240×320 FP32 dump as PNG |

Read those for the production-grade version of build + run.
