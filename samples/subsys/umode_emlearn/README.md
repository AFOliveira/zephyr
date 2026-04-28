# umode_emlearn

Portable emlearn inference demo. Same `main.c` builds and runs on both
the ET-SoC1 U-mode payload (under MachineMinion + WorkerMinion DM-API)
and the native Erbium M-mode build, via the AIFoundry kernel HAL
(`<zephyr/aifoundry/runtime.h>`).

## Build

```sh
# ET-SoC1 (U-mode under MachineMinion + DM-API)
cd /home/afonso/zephyr-erbium-uart && \
  ZEPHYR_BASE=$(pwd) \
  ZEPHYR_SDK_INSTALL_DIR=/home/afonso/toolchains/zephyr-sdk-0.17.4 \
  west build --build-dir build-etsoc1 -b etsoc1_minion_umode \
             samples/subsys/umode_emlearn -p auto

# Erbium (M-mode native)
cd /home/afonso/zephyr-erbium-uart && \
  ZEPHYR_BASE=$(pwd) \
  ZEPHYR_SDK_INSTALL_DIR=/home/afonso/toolchains/zephyr-sdk-0.17.4 \
  west build --build-dir build-erbium -b erbium_minion \
             samples/subsys/umode_emlearn -p auto
```
