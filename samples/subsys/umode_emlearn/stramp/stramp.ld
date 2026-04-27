OUTPUT_ARCH(riscv)
ENTRY(_stramp_start)
SECTIONS {
    . = 0x8000C00000;
    .text : { *(.text.init) *(.text) }
}
