# umode_emlearn

Portable emlearn inference demo.  The same sample source runs as
ET-SoC1 U-mode and as native Erbium M-mode because the target-specific
parts are behind the AIFoundry/ET HAL surfaces.  The ET-SoC1 U-mode
build uses `et_printf()` for trace logging, the U-mode cache-op helper
before returning results, and the Zephyr U-mode runtime for launch
completion after `main()` returns.

## Build Locally

```sh
# ET-SoC1 (U-mode under MachineMinion + DM-API)
export ZEPHYR_ROOT=/path/to/zephyr
export ET_PLATFORM_ROOT=/path/to/et-platform
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.4

cd "${ZEPHYR_ROOT}"

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build \
  --build-dir build-etsoc1-lp64f-test \
  -b etsoc1_minion_umode \
  samples/subsys/umode_emlearn \
  -p always \
  -- -DET_PLATFORM_ROOT="${ET_PLATFORM_ROOT}"

# Erbium (M-mode native)
cd "${ZEPHYR_ROOT}"

ZEPHYR_TOOLCHAIN_VARIANT=zephyr \
west build \
  --build-dir build-erbium-emlearn-test \
  -b erbium_minion \
  samples/subsys/umode_emlearn \
  -p always
```

The build outputs are:

```text
build-etsoc1-lp64f-test/zephyr/zephyr.elf
build-erbium-emlearn-test/zephyr/zephyr.elf
```

For release asset packaging and a fully pinned source rebuild flow, see
`RELEASE_GUIDE.md`.

## Run On ET-SoC1

The ET-SoC1 launcher runs on the board host, not inside Zephyr.  The
commands below use the stock gp-sdk `basic_launcher` and assume the
desired ELF is already present in the launcher bundle directory.

```sh
ssh root@esperanto-soc4
cd /root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
```

Single-kernel smoke run on shire 0:

```sh
LD_LIBRARY_PATH=$PWD ./basic_launcher \
  --kernel_path=zephyr_halified_emlearn_16m.elf \
  --device_type=silicon \
  --shire_mask=0x1 \
  --kernel_launch_timeout=30 \
  --num_launches=1
```

Repeated run of the same loaded kernel:

```sh
LD_LIBRARY_PATH=$PWD ./basic_launcher \
  --kernel_path=zephyr_halified_emlearn_16m.elf \
  --device_type=silicon \
  --shire_mask=0x1 \
  --kernel_launch_timeout=30 \
  --num_launches=5
```

The same stock launcher can use the ET-SoC1 system emulator by switching
the device type:

```sh
LD_LIBRARY_PATH=$PWD ./basic_launcher \
  --kernel_path=zephyr_halified_emlearn_16m.elf \
  --device_type=sysemu \
  --shire_mask=0x1 \
  --kernel_launch_timeout=120 \
  --num_launches=1
```

## Run On Erbium

The Erbium build is the same emlearn source built for `erbium_minion`
and run in native M-mode through `erbium_emu`.

```sh
export ET_PLATFORM_ROOT=/path/to/et-platform
export ERBIUM_EMU="${ET_PLATFORM_ROOT}/build-emu/erbium_emu"
export ERBIUM_UART=/tmp/erbium_emlearn_uart.txt

rm -f "${ERBIUM_UART}"
"${ERBIUM_EMU}" \
  -reset_pc 0x40000200 \
  -single_thread \
  -max_cycles 500000000 \
  -elf_load build-erbium-emlearn-test/zephyr/zephyr.elf \
  -uart_tx_file "${ERBIUM_UART}"

cat "${ERBIUM_UART}"
```

Expected UART output includes:

```text
emlearn-demo: start
emlearn-demo: iris (DecisionTree, 4 features, 5 vectors)
  iris-tree[0] -> 0
  iris-tree[1] -> 1
  iris-tree[2] -> 2
emlearn-demo: done
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
the file remains small.

`basic_launcher` calls `loadKernel()`, which delegates ELF allocation to
`IRuntime::loadCode()`.  The runtime allocates device DRAM from its host
memory manager based on the ELF program headers, relocates the loaded
copy, and then launches that kernel ID on the requested shire mask.
