# ISA migration smoke (strict contract)

Covers the nine approved custom-0/custom-1 migration targets from
`harness/config/encoding_plan.yaml`.

`main()` performs compile-time `BUILD_ASSERT` word checks only. Raw `.word`
emission for all nine targets lives in `src/aif_compile_only_emitters.c`, is
linker-retained via `aif_compile_only_retain_table`, and is not called from
`main()`. After `west build`, record non-execution proof with:

```sh
riscv64-zephyr-elf-objdump -d build/zephyr/zephyr.elf | rg 'main>|aif_co_emit'
```

## Build

```sh
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.4
cd "${ZEPHYR_BASE}"

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build -p always -b erbium_minion samples/aifoundry/isa_migration_smoke
```
