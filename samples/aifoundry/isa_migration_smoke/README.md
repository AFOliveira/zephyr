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
