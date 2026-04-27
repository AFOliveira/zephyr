# DM-API launcher compatibility tools

When running this sample through the **stock gp-sdk DM-API launcher**
(`basic_launcher` from et-platform/gp-sdk), the on-device runtime
(`libetrt.so`) needs the kernel ELF in a specific shape:

1. **First `LOAD` segment must start at file offset 0x1000.**
   The runtime's `relocateSection()` has a hardcoded
   `BASE_OFFSET = 0x1000` assumption.  Zephyr's default link (with
   `-Wl,-N` / NMAGIC) packs the segment right after the headers
   (offset 0xb0).  Pad with `elf_pad.py`:
       python3 elf_pad.py build/zephyr/zephyr.elf zephyr_padded.elf

2. **R_RISCV_64 relocations must be preserved.**  The runtime
   fixes up absolute references after relocating the image.  Zephyr
   strips them by default; this sample's CMakeLists.txt opts back in
   via `-Wl,--emit-relocs`.

Neither tweak is needed for the M+U direct-launch path or the S-mode
trampoline path — those don't go through the DM-API loader.

For the M+S+U production path the sequence is:
    west build -b etsoc1_minion_umode samples/subsys/umode_emlearn
    python3 samples/subsys/umode_emlearn/tools/elf_pad.py \
            build/zephyr/zephyr.elf zephyr_padded.elf
    basic_launcher -k zephyr_padded.elf -d sysemu -m 0x1
