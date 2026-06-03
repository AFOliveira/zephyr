# ISA migration smoke (AIFoundry custom-0/custom-1)

Builds against `include/zephyr/arch/riscv/aif/xaifet.h` and exercises the nine
approved migration targets from `harness/config/encoding_plan.yaml`.

```sh
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.4
cd "${ZEPHYR_BASE}"

# Preferred ET-SoC1 U-mode board (requires ET_PLATFORM_ROOT):
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build -p always -b etsoc1_minion_umode \
  samples/aifoundry/isa_migration_smoke \
  -- -DET_PLATFORM_ROOT=/path/to/et-platform

# Erbium M-mode (no ET platform dependency):
ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build -p always -b erbium_minion \
  samples/aifoundry/isa_migration_smoke
```
