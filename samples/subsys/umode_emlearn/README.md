# umode_emlearn

Portable emlearn inference demo. The ET-SoC1 U-mode build uses the ET
Platform surface directly: `et_printf()` for trace logging, the U-mode
cache-op helper before returning results, and the Zephyr U-mode runtime
for launch completion after `main()` returns.

## Build Locally

```sh
# ET-SoC1 (U-mode under MachineMinion + DM-API)
cd /home/afonso/zephyr

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/home/afonso/toolchains/zephyr-sdk-0.17.4 \
west build \
  --build-dir build-etsoc1-lp64f-test \
  -b etsoc1_minion_umode \
  samples/subsys/umode_emlearn \
  -p always \
  -- -DET_PLATFORM_ROOT=/home/afonso/et-platform-vidas

# Erbium (M-mode native)
cd /home/afonso/zephyr

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
ZEPHYR_SDK_INSTALL_DIR=/home/afonso/toolchains/zephyr-sdk-0.17.4 \
west build \
  --build-dir build-erbium-emlearn-test \
  -b erbium_minion \
  samples/subsys/umode_emlearn \
  -p always
```

The ET-SoC1 build output is:

```text
build-etsoc1-lp64f-test/zephyr/zephyr.elf
```

## Run On ET-SoC1

The ET-SoC1 launchers run on the board host, not inside Zephyr.  The
commands below assume the desired ELF is already present in the launcher
bundle directory.

```sh
ssh root@esperanto-soc4
cd /root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
```

Single-shire smoke run:

```sh
LD_LIBRARY_PATH=$PWD ./basic_launcher \
  --kernel_path=zephyr_halified_emlearn_16m.elf \
  --device_type=silicon \
  --shire_mask=0x1 \
  --kernel_launch_timeout=30 \
  --num_launches=1
```

Repeated single-shire run:

```sh
LD_LIBRARY_PATH=$PWD ./basic_launcher \
  --kernel_path=zephyr_halified_emlearn_16m.elf \
  --device_type=silicon \
  --shire_mask=0x1 \
  --kernel_launch_timeout=30 \
  --num_launches=5
```

Four-shire concurrent run:

```sh
LD_LIBRARY_PATH=$PWD ./quad_shire_launcher \
  --kernel_path_0=zephyr_halified_emlearn_16m.elf \
  --kernel_path_1=zephyr_halified_emlearn_16m.elf \
  --kernel_path_2=zephyr_halified_emlearn_16m.elf \
  --kernel_path_3=zephyr_halified_emlearn_16m.elf \
  --device_type=silicon \
  --kernel_launch_timeout=30
```

The same launcher commands can use the ET-SoC1 system emulator by
switching the device type:

```sh
LD_LIBRARY_PATH=$PWD ./quad_shire_launcher \
  --kernel_path_0=zephyr_halified_emlearn_16m.elf \
  --kernel_path_1=zephyr_halified_emlearn_16m.elf \
  --kernel_path_2=zephyr_halified_emlearn_16m.elf \
  --kernel_path_3=zephyr_halified_emlearn_16m.elf \
  --device_type=sysemu \
  --kernel_launch_timeout=120
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

`quad_shire_launcher` is gp-sdk host infrastructure.  It calls
`loadKernel()` four times, which delegates ELF allocation to
`IRuntime::loadCode()`.  The runtime allocates device DRAM from its host
memory manager based on the ELF program headers, relocates each copy, and
then launches the four kernel IDs on shire masks `0x1`, `0x2`, `0x4`,
and `0x8`.
