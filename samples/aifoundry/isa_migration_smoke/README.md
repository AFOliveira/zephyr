# ISA migration smoke (strict contract)

Covers the nine approved custom-0/custom-1 migration targets from
`harness/config/encoding_plan.yaml`.

`main()` does not execute raw custom instructions. Compile-time encoder checks
live in `src/aif_encode_verify.c`. Canonical NEW `.word` emission for all nine
targets lives in `src/aif_compile_only_emitters.c`, is linker-retained via
`aif_compile_only_retain_table`, and is not called from `main()`.

After `west build`, record non-execution proof with:

```sh
riscv64-zephyr-elf-objdump -d build/zephyr/zephyr.elf | rg 'main>|aif_co_emit'
```

Canonical NEW words emitted (see `AIF_NEW_WORD_*` in `xaifet.h`):
`0x0005500b`, `0x0005502b`, `0x0400802b`, `0x0400902b`, `0x1c20a02b`,
`0x0020802b`, `0x80c5e52b`, `0x80c5f52b`, `0x02a5c52b`.

With `CONFIG_AIF_ET_ISA_MIGRATION_PATCHED_ASM=y`, `src/aif_mnemonic_emitters.S`
is assembled by the patched Phase 2 `as-new` (`rv64i_xaifet`, `-mabi=lp64f`)
and verified with patched `objdump` during the build (`build/aif_patched_binutils_verify.log`).
Stock Zephyr SDK still links C sources; patched objdump on the final ELF shows
`aif.*` mnemonics under `aif_mn_emit_*`.

## Build

```sh
export ZEPHYR_BASE=/path/to/zephyr-worktree
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.4
cd "${ZEPHYR_BASE}"
west init -l .   # once per worktree

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build -p always -b erbium_minion samples/aifoundry/isa_migration_smoke
```

Build output: `build/zephyr/zephyr.elf` (entry `0x40000200` for `erbium_minion`).

## Runtime (`erbium_emu`)

`erbium_minion` links for the Erbium minion memory map. Runtime boot evidence uses
the C++ **Erbium system emulator** (`erbium_emu`), not ET-SoC1 `sys_emu` and not
full-system Verilator RTL.

### Prerequisites

- Completed `west build` (above).
- `erbium_emu` built under `repos/et-platform/sw-sysemu/build/erbium_emu`
  (Conan/`cmake` build of `sw-sysemu`; see `repos/et-platform/sw-sysemu/tests/erbium/README.md`).

### Boot command

From the Zephyr worktree (or repository root with paths adjusted):

```sh
ERBIUM_EMU=repos/et-platform/sw-sysemu/build/erbium_emu
ELF=repos/zephyr/build/zephyr/zephyr.elf
UART_LOG=/tmp/isa_migration_smoke_uart.txt

rm -f "${UART_LOG}"
"${ERBIUM_EMU}" \
  -elf_load "${ELF}" \
  -reset_pc 0x40000200 \
  -uart_tx_file "${UART_LOG}" \
  -max_cycles 20000000

cat "${UART_LOG}"
```

`-reset_pc 0x40000200` matches the ELF entry / linker script for `erbium_minion`.
Do not pass ET-SoC1 `-shires` flags to `erbium_emu`.

### Expected UART output

Verified on `esperanto-soc6` (2026-06-04, infrastructure task
`zephyr-runtime-infra-isa-migration-20260604`):

```text
*** Booting Zephyr OS build 76e134655cec ***
isa-migration-smoke: strict compile-only PASS (no raw insn in main)
  expected packb x10=0x2211 bitmixb x10=0xf895 summit x10=0xaa18
```

This proves Zephyr boots far enough for the sample `printk` path. It does **not**
prove execution of the nine raw custom instructions in `main()` — those live in
linker-retained compile-only objects (`aif_compile_only_emitters.c`,
`aif_mnemonic_emitters.S`) that `main()` does not call.

### General Erbium Zephyr pattern

`samples/subsys/umode_emlearn/README.md` documents the same `erbium_emu`
`-elf_load` / `-reset_pc 0x40000200` / `-uart_tx_file` flow for another
`erbium_minion` sample.

## Why `sys_emu` is not valid here

`boards/aifoundry/erbium_minion/board.cmake` defaults the `et-emu` runner to
`erbium_emu`, not ET-SoC1 `sys_emu`. The ET-SoC1 emulator models shires, a
different reset/IO map, and minion topology.

A prior infrastructure attempt booted with:

```sh
repos/et-platform/sw-sysemu/build/sys_emu \
  -elf_load repos/zephyr/build/zephyr/zephyr.elf \
  -reset_pc 0x40000200 -shires 0x1 -max_cycles 50000000
```

That run exited with trap recursion at PC `0x0` before any sample output — it is
**not** acceptable runtime proof for `erbium_minion` / `isa_migration_smoke`.

The Zephyr `etemu` west runner (`scripts/west_commands/runners/etemu.py`) always
passes `-shires` (default `0x400000000` for ET-SoC1-style boards). Even when
`ERBIUM_EMU` is set, the runner still targets the `sys_emu` CLI shape. Prefer the
direct `erbium_emu` invocation above over `west build -t simulate` until a
board-specific runner omits shires for Erbium.

## Why `west build -t simulate` does not work

Board `erbium_minion.yaml` declares `simulation: et-emu`, and `board.cmake` wires
the `etemu` runner, but for this sample the CMake/ninja graph does not expose a
`simulate` target:

```sh
west build -t simulate -b erbium_minion samples/aifoundry/isa_migration_smoke
# ninja: error: unknown target 'simulate'
```

Use the manual `erbium_emu` command in **Runtime** until Zephyr integration adds a
working `simulate` target for this board/sample.

## Full-system Verilator (not available)

No Verilator target in this monorepo loads and boots `zephyr.elf` for
`erbium_minion`. Repository search (2026-06-04) found:

| Area | What exists | Why it is not Zephyr full-system |
|------|-------------|----------------------------------|
| `repos/core-et` | `intpipe_decode` DV, `rtlcosim` migration tests, FPGA Verilator demos | Decode or fixed testbenches; no `-elf_load`, no Zephyr memory map |
| `repos/et-platform/sw-sysemu` | `erbium_emu` / `sys_emu` C++ emulators | Functional models, not cycle-accurate SoC Verilator |
| `repos/et-soc1-rtl/dv/cosim` | Legacy cosim infrastructure | Not wired to Zephyr ELF boot here |
| `harness/` `phase3-runtime.sh verilator` | core-et decoder/lint/migration cosim | Not OS boot |

**Decoder-level evidence cannot substitute** for Zephyr runtime proof: passing
`intpipe_decode` tests or `minion_frontend_intpipe_decode_migration` rtlcosim
proves decode-row correctness on fixed words, not linker layout at `0x40000200`,
M-mode boot, drivers, or UART printk from `isa_migration_smoke`.

A future full-system Verilator path would need at least:

1. Verilator model of Erbium minion + MRAM + UART at Zephyr-linked addresses
2. C++ testbench ELF loader matching `erbium_minion` map and entry `0x40000200`
3. UART capture equivalent to `-uart_tx_file`
4. Zephyr west/CMake runner integration (not only `et-emu` → `sys_emu`)
5. CI/harness build of that target

## What ISA migration releases may claim

| Evidence type | Allowed claim for this sample |
|---------------|-------------------------------|
| `west build` + objdump / patched-binutils log | Patched assembler/objdump saw migration mnemonics/words in ELF objects |
| `erbium_emu` boot (above) | Zephyr OS boot + sample `printk` on compile-only contract |
| Compile-only emitters retained, not called from `main` | Non-execution of raw custom instructions in the runtime path exercised |
| core-et decode DV / migration rtlcosim | Decode-row correctness only — **not** Zephyr OS boot |
| Full-system Verilator Zephyr boot | **Blocked** until the missing stack above exists |

Production ISA migration tasks must not treat Zephyr `west build` alone as runtime
proof when a sample could execute custom instructions; for this sample, runtime
proof is the `erbium_emu` UART line plus explicit non-execution of emitters.
