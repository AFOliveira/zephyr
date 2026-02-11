# AIFoundry ET-SOC1 SP - Running on the System Emulator

## Prerequisites

### Zephyr SDK

Follow the [Zephyr Getting Started Guide](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)
to install the Zephyr SDK and set up the west workspace.

Set the SDK path:

```bash
export ZEPHYR_SDK_INSTALL_DIR=/path/to/zephyr-sdk-0.17.4
```

### Emulator Binary

Build `sys_emu` from the `et-platform` repository:

```bash
make -C /path/to/et-platform/sw-sysemu/build -j$(nproc)
```

Set the emulator path:

```bash
export SYS_EMU=/path/to/et-platform/sw-sysemu/build/sys_emu
```

## Quick Start: Synchronization Demo

The `samples/synchronization` sample runs two threads that alternate printing
greeting messages using semaphores.

### Build and Run

```bash
west build -b etsoc1_sp/etsoc1_sp samples/synchronization -p \
    -- -DEXTRA_CONF_FILE=$(pwd)/boards/aifoundry/etsoc1_sp/etsoc1_sp_fast_emu.conf
west build -t run
```

The `etsoc1_sp_fast_emu.conf` overlay speeds up timer-based sleeps so the demo
produces output in a reasonable number of emulated cycles.

### Run Directly

```bash
sys_emu \
    -elf_load build/zephyr/zephyr.elf \
    -shires 0x400000000 \
    -mins_dis \
    -sp_reset_pc 0x40400000 \
    -max_cycles 50000000
```

### Expected Output

```
*** Booting Zephyr OS build v1.13.0-rc2-... ***
thread_a: Hello World from cpu 0 on etsoc1_sp!
thread_b: Hello World from cpu 0 on etsoc1_sp!
thread_a: Hello World from cpu 0 on etsoc1_sp!
thread_b: Hello World from cpu 0 on etsoc1_sp!
...
```

## Other Samples

Any Zephyr sample that uses the UART console works the same way:

```bash
west build -b etsoc1_sp/etsoc1_sp samples/hello_world -p
west build -t run
```

## sys_emu Options Reference

| Option | Description |
|---|---|
| `-elf_load <path>` | Load ELF file (repeatable) |
| `-sp_reset_pc <addr>` | SP reset program counter |
| `-mins_dis` | Disable minions |
| `-shires <mask>` | Shire configuration mask |
| `-max_cycles <n>` | Stop after N cycles |
| `-spio_uart0_tx_file <path>` | Redirect UART0 TX to file |
| `-spio_uart0_rx_file <path>` | Feed UART0 RX from file |
| `-gdb` | Start GDB stub |
