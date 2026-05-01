# Arduino API on ET-SoC1 / Erbium

Status: **plan, not implemented.** Updated after a web research pass on
`zephyrproject-rtos/ArduinoCore-zephyr` ground-truth.

## Goal

Get the Arduino Core API (the `zephyrproject-rtos/ArduinoCore-zephyr`
upstream module + `arduino/ArduinoCore-API`) running on our two
AIFoundry boards:

  - `erbium_minion`         — Zephyr-M, Shakti UART, native peripherals
  - `etsoc1_minion_umode`   — Zephyr-U on ET-SoC1, no UART, no peripherals

…with the same Arduino sketch source compiling for both. Zero
target-conditional `#ifdef`s in user sketches.

## Upstream ground truth (verified by reading the repo)

The active project is `github.com/zephyrproject-rtos/ArduinoCore-zephyr`
(not the GSoC-2022 archive — same content, but Zephyr-blessed home).
Twelve variants already shipped: `arduino_nano_33_ble`,
`arduino_nano_33_iot`, `beagleconnect_freedom`, `nrf52840dk_nrf52840`,
`pocketbeagle_2_am6254_a53`, `rpi_pico`, `native_sim`, plus a few
others. Adding a 13th (or 14th) variant is the standard contribution
shape.

Key implementation facts, all from reading the upstream tree:

1. **`cores/arduino/main.cpp` is 14 lines**:
   ```cpp
   int main(void) {
     setup();
     for (;;) {
       loop();
       if (arduino::serialEventRun) arduino::serialEventRun();
     }
     return 0;
   }
   ```
   Zephyr's `main()` becomes the Arduino superloop — no adapter we have to write.

2. **`cores/arduino/zephyrSerial.cpp` is monolithic** — one
   `arduino::ZephyrSerial` class hardcoded to talk to a Zephyr UART
   device via `uart_configure`, `uart_irq_callback_user_data_set`,
   `uart_irq_rx_enable`. **Variants don't override Serial.** They just
   pick which UART device to bind to (via DT `chosen { zephyr,console = &foo; }`).

3. **A "variant" is just two files**: `variant.h` + `pins_arduino.h`.
   The latter maps Arduino pin numbers to Zephyr DT GPIO references.
   That's it — no Serial code, no peripheral drivers.

4. **`zephyrCommon.cpp` does GPIO** via DT macros. Per-board pin tables
   are materialized at compile time from the `zephyr,user` DT node's
   `digital-pin-gpios` property.

This means **Arduino on Zephyr's contract is: provide a UART device (any
Zephyr UART device) and a pin map; Arduino-Core-zephyr handles the rest.**

## How this changes our approach

Knowing point 2 above (Serial assumes a Zephyr UART device), the
clean way to get Arduino on U-mode is **not** to subclass `Print` —
it's to provide a *fake Zephyr UART driver* that backs `uart_poll_out`
with `aifoundry_log`. Then upstream `ZephyrSerial` works unmodified;
the `Serial.print` bytes end up in the et-trace ring without
Arduino-Core-zephyr knowing or caring.

This is the same trick `native_sim`'s variant uses — a Zephyr UART
driver backed by stdin/stdout instead of a real MMIO peripheral.

## The HAL-everything principle

Per-target divergence in the Arduino API maps cleanly onto our existing
AIFoundry HAL surface (`<zephyr/aifoundry/runtime.h>`). **Any Arduino
primitive whose implementation differs by board should route through
the HAL.** Anything board-agnostic (the Arduino API headers themselves,
`setup()`, `loop()`) doesn't need to.

Concretely, every Arduino API call falls into one of three buckets:

| Arduino primitive | Bucket | Routing |
|---|---|---|
| `Serial.print` / `Serial.write` | per-target | **fake UART driver → `aifoundry_log`** (HAL) |
| `delay(ms)` / `delayMicroseconds(us)` | per-target | **HAL: `aifoundry_delay_ms`/`_us`** (M-mode → k_busy_wait, U-mode → software loop) |
| `millis()` / `micros()` | per-target | **HAL: `aifoundry_uptime_ms`/`_us`** (M-mode → k_uptime_get_32, U-mode → software counter) |
| `Serial.read` / `Serial.available` | per-target | fake UART's `poll_in` returns -1 today; HAL future hook for host→device mailbox if/when we want one |
| `setup()` / `loop()` / `Print` / `Stream` | board-agnostic | upstream Arduino-Core-zephyr, untouched |
| `String`, `random`, `parseInt`, `map`, `constrain`, etc. | board-agnostic | upstream, untouched |
| `pinMode` / `digitalWrite` / `digitalRead` | per-target | upstream zephyrCommon.cpp on top of Zephyr's GPIO API; **stub at variant level on U-mode** (no GPIO access). HAL doesn't gain a primitive — there's no useful M-mode/U-mode commonality to abstract. |
| `analogRead` / `analogWrite` / `Wire` / `SPI` | per-target | same — no HAL primitive; not in scope for v1. |
| `attachInterrupt` | per-target | no HAL primitive; M-mode could (later); U-mode can't. Not v1. |

So the HAL grows by **three new entry points** to cover the time-related
Arduino primitives. Output goes through `aifoundry_log` (already exists).
Input is a stub for now. Peripherals are stubbed at the Zephyr-driver
layer (no GPIO devices defined in U-mode DT → upstream `zephyrCommon.cpp`
walks zero pins → digitalWrite is a no-op).

### New HAL surface to add to `<zephyr/aifoundry/runtime.h>`

```c
/* Block for at least `ms` milliseconds before returning. */
void     aifoundry_delay_ms(uint32_t ms);

/* Microsecond-resolution variant.  May round up to ms granularity
 * on targets without a finer-grained clock source. */
void     aifoundry_delay_us(uint32_t us);

/* Elapsed milliseconds since kernel entry.  May return 0 always on
 * targets without a usable clock source — sketches that don't
 * read elapsed time are unaffected. */
uint32_t aifoundry_uptime_ms(void);
uint32_t aifoundry_uptime_us(void);
```

### Per-target implementations

| Function | `erbium_minion` (M-mode) | `etsoc1_minion_umode` (U-mode) |
|---|---|---|
| `aifoundry_delay_ms(ms)` | `k_busy_wait(ms * 1000)` | software loop calibrated against the busy-wait cycles we already use in `aifoundry_kernel_exit` |
| `aifoundry_delay_us(us)` | `k_busy_wait(us)` | scaled software loop; rounds up |
| `aifoundry_uptime_ms()` | `k_uptime_get_32()` | software counter incremented inside `delay_ms`/`delay_us`. Returns 0 if user code never called delay. |
| `aifoundry_uptime_us()` | `k_uptime_get_32() * 1000` (rough) | same software counter, scaled |

The U-mode software counter is a deliberate compromise: precise only
when the sketch actually uses `delay()`. Sketches doing pure compute
without `delay` will see `millis() == 0` always. Acceptable for the
v1 demo; documented limitation. A future v2 could plug et-trace's
timestamp field as a real clock source (currently stubbed in
`arch/riscv/core-umode/trace_umode_glue.c` as `et_trace_zero_timestamp`).

### Wiring HAL into Arduino's delay/millis

Arduino-Core-zephyr's upstream `delay()` / `millis()` implementations
live in its `cores/arduino/` (likely `time_macros.h` and a `.cpp`).
We override them at the variant level — the standard Arduino-Core
escape hatch — by providing our own `wiring.cpp` (or whatever the
exact override file is named upstream; verified in phase 1) for
the AIFoundry variants that calls `aifoundry_delay_ms` /
`aifoundry_uptime_ms` directly. Same source for both variants;
the per-target divergence is fully inside the HAL.

## Architecture

```
   user sketch (setup() + loop(), Arduino-style C++)
                              │
                              ↓
   ArduinoCore-API headers (arduino/ArduinoCore-API)
                              │
                              ↓
   ArduinoCore-zephyr glue (cores/arduino/zephyrSerial.cpp etc.)
                              │
                              ↓ (calls Zephyr UART API)
                              ↓
   Zephyr UART driver layer
                              │
   ┌──────────────────────────┴──────────────────────────┐
   ↓                                                     ↓
   uart_shakti (M-mode, exists)              uart_aifoundry_trace (NEW)
   → MMIO at 0x02004000                      → uart_poll_out body:
   → erbium_emu UART model                       aifoundry_log("%c", c)
   → -uart_tx_file                           → Trace_Format_String_V
                                             → et-trace ring
                                             → host's et_dev_trace_decoder
```

**The Arduino layer is byte-identical on both targets.** Per-target
divergence is one Zephyr UART driver, hidden behind the standard UART
device interface. We don't fork ArduinoCore-zephyr at all.

## Files to add

### Wired via west.yml (existing upstream module)

```yaml
- name: Arduino-Core-Zephyr
  path: modules/lib/Arduino-Zephyr-API
  revision: main
  url: https://github.com/zephyrproject-rtos/ArduinoCore-zephyr
```

(Use the active `zephyrproject-rtos/ArduinoCore-zephyr` URL, not the
historical `gsoc-2022-arduino-core` URL — same content, more durable.)

### Modified HAL surface

```
include/zephyr/aifoundry/runtime.h          ← +4 declarations:
                                                aifoundry_delay_ms / _us
                                                aifoundry_uptime_ms / _us

soc/aifoundry/etsoc1_minion_umode/runtime.c ← +4 impls (sw counter for U-mode)
soc/aifoundry/erbium_minion/runtime.c       ← +4 impls (k_busy_wait + k_uptime)
```

### New files we own (in this branch)

```
drivers/serial/uart_aifoundry_trace.c       ← NEW Zephyr UART driver, et-trace backed.
                                              uart_poll_out / uart_fifo_fill callbacks
                                              call aifoundry_log() directly.
                                              ~80 lines including the DT binding.

dts/bindings/serial/aifoundry,umode-trace-uart.yaml
                                            ← DT compatible string for the new driver.
                                              "Virtual UART that writes to the cm-umode
                                              et-trace ring via aifoundry_log."

modules/lib/Arduino-Zephyr-API/variants/erbium_minion/
  ├── variant.h                             ← per Arduino-Core-zephyr's variant docs
  └── pins_arduino.h                        ← Arduino pin → Erbium GPIO map (or stubs)

modules/lib/Arduino-Zephyr-API/variants/etsoc1_minion_umode/
  ├── variant.h
  └── pins_arduino.h                        ← all "pins" stub out (no GPIO on U-mode);
                                              Serial maps to the trace UART.

samples/subsys/umode_arduino_emlearn/
├── PLAN.md                                 ← this doc
├── CMakeLists.txt
├── prj.conf                                ← CONFIG_ARDUINO=y, CONFIG_CPP=y, etc.
├── boards/                                 ← per-board overlays if needed
│   ├── erbium_minion.overlay               ← chosen { zephyr,console = &uart0; }
│   └── etsoc1_minion_umode.overlay         ← chosen { zephyr,console = &trace_uart; }
└── src/
    ├── sketch.cpp                          ← Arduino-style: setup() + loop() running emlearn
    ├── iris_tree.h                         ← copy of existing
    └── test_vectors_gen.h
```

### Optional: variant-level wiring.cpp

If Arduino-Core-zephyr's upstream `delay`/`millis` implementations
are board-overridable (verified in phase 1), each variant directory
gets a small `wiring.cpp`:

```cpp
// modules/lib/Arduino-Zephyr-API/variants/erbium_minion/wiring.cpp
//                                       (and same file in etsoc1_minion_umode/)
#include <zephyr/aifoundry/runtime.h>
extern "C" {
    void delay(unsigned long ms)              { aifoundry_delay_ms((uint32_t)ms); }
    void delayMicroseconds(unsigned int us)   { aifoundry_delay_us((uint32_t)us); }
    unsigned long millis(void)                { return aifoundry_uptime_ms(); }
    unsigned long micros(void)                { return aifoundry_uptime_us(); }
}
```

Same source in both variant directories. Per-target divergence is
fully inside the HAL implementations. If upstream's defaults aren't
overridable per-variant, we'd need to either patch upstream (would
upstream a clean per-variant hook) or accept the upstream behavior
(Zephyr `k_busy_wait`-based; works fine on M-mode, would block forever
on U-mode without a clock).

### Modified files

- **`west.yml`** — adds the ArduinoCore-zephyr module entry.
- **No changes to** `arch/riscv/core-umode/`, board defconfigs, or
  the AIFoundry HAL surface (`<zephyr/aifoundry/runtime.h>`,
  `soc/aifoundry/<soc>/runtime.c`). The new path slots in beneath the
  HAL — Arduino → Zephyr UART → fake-uart-driver → `aifoundry_log` →
  per-SoC HAL impl. The existing HAL path is unchanged.

## Implementation phases

### Phase 0 — extend the HAL (~30 min)

Add the four new entry points to `<zephyr/aifoundry/runtime.h>` and
implement them in both `soc/aifoundry/<soc>/runtime.c` files. This is
purely additive — no existing samples break. Cross-test by rebuilding
`umode_emlearn` (existing M-mode build), `umode_emlearn_iris_rf`,
`umode_emlearn_wine_rf` to confirm nothing regresses.

Exit criteria: HAL surface has `aifoundry_delay_ms/_us` and
`aifoundry_uptime_ms/_us`; existing samples still build and run.

### Phase 1 — wire the upstream module (~30 min)

1. Add Arduino-Core-Zephyr to `west.yml`.
2. `west update` — verify `modules/lib/Arduino-Zephyr-API/` exists.
3. Run upstream `install.sh` (or symlink `arduino/ArduinoCore-API/api`
   into `cores/arduino/`).
4. Smoke-test: build Arduino-Core-Zephyr's own `samples/basic/blink`
   for `qemu_riscv64` (or `native_sim`). Confirms the module integrates
   cleanly before we touch our boards.

Exit criteria: a stock Arduino sample builds for a stock Zephyr board.

### Phase 2 — fake-UART driver for et-trace (~1 h)

1. Write `drivers/serial/uart_aifoundry_trace.c`. Skeleton:
   ```c
   static int uart_aifoundry_trace_init(const struct device *dev) { return 0; }
   static void uart_aifoundry_trace_poll_out(const struct device *dev, unsigned char c) {
       char buf[2] = { (char)c, 0 };
       aifoundry_log("%s", buf);
   }
   static int uart_aifoundry_trace_poll_in(const struct device *dev, unsigned char *c) {
       return -1;   /* no input */
   }
   static const struct uart_driver_api uart_aifoundry_trace_api = {
       .poll_out = uart_aifoundry_trace_poll_out,
       .poll_in  = uart_aifoundry_trace_poll_in,
       /* configure() / err_check() / IRQ ops not implemented;
        * Zephyr's UART API tolerates partial driver impls when only
        * polled output is needed. */
   };
   #define UART_AIFOUNDRY_TRACE_INIT(n) \
       DEVICE_DT_INST_DEFINE(n, uart_aifoundry_trace_init, NULL, NULL, NULL, \
                             PRE_KERNEL_1, CONFIG_SERIAL_INIT_PRIORITY, \
                             &uart_aifoundry_trace_api);
   DT_INST_FOREACH_STATUS_OKAY(UART_AIFOUNDRY_TRACE_INIT)
   ```
2. Add `dts/bindings/serial/aifoundry,umode-trace-uart.yaml`:
   ```yaml
   compatible: "aifoundry,umode-trace-uart"
   include: uart-controller.yaml
   ```
3. Add a DT node in our U-mode board overlay:
   ```dts
   trace_uart: trace_uart {
       compatible = "aifoundry,umode-trace-uart";
   };
   chosen { zephyr,console = &trace_uart; };
   ```
4. Plumb a Kconfig — `CONFIG_UART_AIFOUNDRY_TRACE` — gating the driver.
5. Verify: a tiny test that does `printk("hello\n")` on the U-mode
   board ends up in the et-trace ring (decoded post-launch).

Exit criteria: `printk` on U-mode goes through this fake UART → et-trace,
visible after `et_dev_trace_decoder`.

### Phase 3 — board variants (~45 min)

1. **`variants/erbium_minion/`**: clone `arduino_nano_33_iot` or
   `beagleconnect_freedom` as a starting template. Adjust `pins_arduino.h`
   to point Serial at our existing `&uart0` (Shakti UART at `0x02004000`).
   Stub out GPIO pin maps for now (return errors / no-ops); fill in if
   we ever expose Erbium GPIOs in DTS.
2. **`variants/etsoc1_minion_umode/`**: same shape, but `Serial` maps
   to `&trace_uart`; all other pins stubbed.

Exit criteria: `samples/basic/blink` from upstream Arduino-Core-zephyr
builds for both our boards (it'll just print to Serial; the digitalWrite
calls are no-ops on U-mode and TBD on M-mode).

### Phase 4 — emlearn sketch (~30 min)

1. `samples/subsys/umode_arduino_emlearn/src/sketch.cpp`:
   ```cpp
   #include <Arduino.h>
   #include "iris_tree.h"
   #include "test_vectors_gen.h"

   void setup() {
       Serial.begin(115200);
       Serial.println("emlearn-arduino: setup");
   }

   void loop() {
       for (int i = 0; i < IRIS_N_VECTORS; i++) {
           int32_t pred = iris_tree_predict(iris_test_vectors[i], IRIS_N_FEATURES);
           Serial.print("iris[");
           Serial.print(i);
           Serial.print("] -> ");
           Serial.println(pred);
       }
       delay(1000);
   }
   ```
2. `prj.conf`: `CONFIG_CPP=y`, `CONFIG_ARDUINO=y` (or whatever the
   module's enable Kconfig is named — verify in phase 1), our usual
   U-mode constraints inherited from the board defconfig.
3. CMakeLists.txt: pull in the module's CMake glue + this sketch.

### Phase 5 — build + run (~1 h)

1. `west build -b erbium_minion samples/subsys/umode_arduino_emlearn`
   and run on `erbium_emu` with `-uart_tx_file`. Expect `Serial.println`
   output on the captured UART.
2. `west build -b etsoc1_minion_umode samples/subsys/umode_arduino_emlearn`,
   then run via `basic_launcher --device_type=sysemu` (and later silicon).
   Decode the et-trace ring with `et_dev_trace_decoder` — same lines
   should appear there.

Exit criteria: same Arduino sketch source produces the same logical
output on both targets, surfaced through different physical channels.

## Reference variants to copy from

When writing our two new variants, the most useful reference points
in the upstream tree:

| Variant we're writing | Closest upstream reference | Why |
|---|---|---|
| `erbium_minion` | `beagleconnect_freedom`, `arduino_nano_33_iot` | Real MCU, real UART, real (eventually) GPIOs. BeagleConnect Freedom is what kgiori specifically cited. |
| `etsoc1_minion_umode` | `native_sim` | Both have a "no real peripherals; UART is virtual" model. native_sim backs UART with stdin/stdout; we back it with et-trace. |

Particularly worth reading before phase 3:
- `variants/beagleconnect_freedom/pins_arduino.h` — concrete RISC-V Arduino pin map.
- `variants/native_sim/native_sim.overlay` — concrete "virtual everything" DT pattern.
- `documentation/variants.md` — official "how to add a new variant" guide.

## Risks and unknowns

1. **C++ on Zephyr-U**: Arduino-Core-zephyr is C++. Our `etsoc1_minion_umode`
   builds today are C-only. Adding `CONFIG_CPP=y` plus the right libstdc++
   subset should work (the SDK bundles libstdc++) but should be verified
   in phase 1's smoke test.

2. **`String` class wants a heap.** Many Arduino libraries use `String`.
   Either turn on a small `CONFIG_HEAP_MEM_POOL_SIZE=16384` in our board
   defconfigs (we already worked out how this fits in 16 MiB SRAM, see
   `etsoc1_run_notes.md`), or avoid `String` in v1 sketches.

3. **`millis()` on U-mode**: `rdcycle` traps because `mcounteren=0`.
   Three options ranked by complexity:
   - Stub to 0 (acceptable for sketches that don't read elapsed time).
   - Software counter incremented inside `delay()` only — coarse but cheap.
   - Back with et-trace's timestamp field (currently stubbed in
     `arch/riscv/core-umode/trace_umode_glue.c`); requires picking a
     non-`rdcycle` time source on the chip.

4. **`Serial.read` / RX path**: U-mode has no inbound channel. Our
   fake UART's `poll_in` returns `-1`. Sketches that block on `Serial.read`
   would hang. Acceptable for the v1 demo; document the limitation.

5. **Per-character `aifoundry_log("%c", c)`**: each call is one
   et-trace ring entry. `Serial.print("hello")` ends up as 5 entries.
   Visible as 5 lines in the decoded trace. Workaround in v2: buffer
   in `uart_poll_out` until newline, then emit one ring entry per line.
   Defer if it's not annoying enough to block the demo.

6. **Verifying the upstream module's enable Kconfig name**: the README
   doesn't say. Phase 1's smoke test will tell us (probably `CONFIG_ARDUINO`,
   but possibly `CONFIG_ARDUINO_API`).

## What this is NOT (deliberately)

- **Not a fork of Arduino-Core-zephyr.** We add two variants + one Zephyr
  UART driver. Everything else stays upstream. Zero patches to send.
- **Not a Linux/POSIX target like `pocketbeagle_2_am6254_a53`.** That
  variant is for Cortex-A53 + Linux. Our target is bare Zephyr.
- **Not a full GPIO/I²C/SPI port for U-mode.** U-mode can't touch
  peripherals. Those APIs stub out; the demo's value is `Serial` +
  compute.
- **Not MicroBlocks itself.** Once Arduino works, MicroBlocks's
  firmware sketch should build for our boards using the existing
  Arduino-Core-zephyr setup. Separate follow-on, doesn't change this
  plan.

## What we're FIRST at (no prior art)

- **Arduino API on a U-mode-only RISC-V target with no peripheral
  access.** Closest precedent (`native_sim`) is still a Linux process
  with full peripheral simulation.
- **Arduino's `Serial` backed by a trace ring instead of a UART.** The
  "fake UART driver" trick is precedent (used by `native_sim`); the
  trace-ring backing is new.

These are additive — they don't conflict with upstream — so they're
shippable as our own contribution if we want to upstream them later.

## Recommended first move

Phase 1 only: wire the module into `west.yml`, build the upstream
`samples/basic/blink` for `qemu_riscv64` to confirm everything else
on our laptop works. **30 minutes max.** If anything breaks (C++
toolchain, west fetch, missing dep), we find out before we've written
any of our own code.

Then phase 2 (fake UART driver) once phase 1 is green.
