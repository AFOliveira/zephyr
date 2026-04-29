# umode_emlearn

Portable emlearn inference demo. The ET-SoC1 U-mode build uses the ET
Platform surface directly: `et_printf()` for trace logging, the U-mode
cache-op helper before returning results, and the Zephyr U-mode runtime
for launch completion after `main()` returns.

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

## Generated classifier headers

`src/iris_tree.h`, `src/iris_rf.h`, and `src/wine_rf.h` are generated
emlearn classifier headers committed with the sample so the normal Zephyr
build does not need Python or sklearn.  For regeneration, use upstream
emlearn's bundled example as the reference flow:

```sh
modules/lib/emlearn/examples/classifiers.py
```

That example uses the same inline C header generation path as these
classifiers.  Adapt its iris and wine dataset setup to regenerate this
sample's DecisionTree and RandomForest headers.

## ET-SoC1 per-launch memory

The ET-SoC1 U-mode board reserves a 16 MiB region 0.  The Zephyr linker
wrapper exposes the unused tail of that region as the NOLOAD `.heap0`
area, so the ELF program header's `p_memsz` spans the full 16 MiB while
the file remains small.  Loading the same ELF into multiple runtime slots
therefore gives each launch its own physical 16 MiB heap/fake-MRAM window
without the application computing a shire-local offset.
