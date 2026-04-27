# Minimal S-mode trampoline (`stramp.S` + `stramp.ld`)

Bridges between **stock upstream MachineMinion** (which mrets to S at
`FW_WORKER_SMODE_ENTRY = 0x8000C00000`) and a U-mode kernel like
Zephyr-U, *without* loading MasterMinion + WorkerMinion + the gp-sdk
host launcher.

When MM mrets to S, `stramp` does the absolute minimum:
- distinguishes cold-entry (`sscratch == 0`) from re-entry via trap
- on cold entry, points `sepc` at the kernel's link-time entry, sets
  `sstatus.SPP = 0` (next mode = U), `sret`s
- on trap re-entry (`scause == 8`, ecall-from-U), interprets
  `SYSCALL_RETURN_FROM_KERNEL` as "halt" and any other syscall as
  "no-op stub: a0=0, sepc+=4, sret"

That's it — no command polling, no DM-API, no fleet-management state.

## Build

    riscv64-zephyr-elf-gcc -x assembler-with-cpp -c \
        -mabi=lp64 -march=rv64imac_zicsr -mcmodel=medany \
        -DZEPHYR_ENTRY=0x8006335940ULL \
        stramp.S -o stramp.o

    riscv64-zephyr-elf-gcc -mabi=lp64 -march=rv64imac_zicsr -mcmodel=medany \
        -T stramp.ld -nostdlib -Wl,--build-id=none stramp.o -o stramp.elf

`ZEPHYR_ENTRY` must be the value of `_start` in the Zephyr ELF you'll
load alongside.  Re-build whenever the entry shifts.

## Run

    sys_emu \
      -elf_load /path/to/stock/MachineMinion.elf \
      -elf_load stramp.elf \
      -elf_load /path/to/zephyr.elf \
      -reset_pc 0x8000001000 \
      -max_cycles 5000000 \
      -sp_dis -shires 0x1 -single_thread

## When to use which path

- **M+U direct (`feat/machineminion-umode-direct`)**: bring-up,
  fastest dev loop, but uses a patched MM.
- **M+S+U with `stramp` (this)**: stock MM unchanged, no full
  fleet-management firmware, lightweight S-mode shim.
- **M+S+U with full DM-API (basic_launcher + MasterMinion +
  WorkerMinion)**: production path; uses the same ELF this sample
  ships, plus the post-link padder under `../tools/`.
