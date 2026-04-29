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
