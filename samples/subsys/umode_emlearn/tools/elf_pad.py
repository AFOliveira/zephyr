#!/usr/bin/env python3
"""
Pad the first LOAD segment of an RISC-V ELF64 so it starts at file offset
0x1000.  The Esperanto runtime's relocateSection() assumes BASE_OFFSET=0x1000
between the ELF's segment physical address and its location in the file
buffer; Zephyr's default link places the first LOAD at offset 0xb0 (right
after ELF header + 2 program headers), which breaks the math.
"""
import struct, sys, pathlib

path = pathlib.Path(sys.argv[1])
out  = pathlib.Path(sys.argv[2])
data = bytearray(path.read_bytes())

if data[:4] != b'\x7fELF' or data[4] != 2:
    sys.exit("not ELF64")
e_phoff,    = struct.unpack_from('<Q', data, 0x20)
e_shoff,    = struct.unpack_from('<Q', data, 0x28)
e_phentsize,= struct.unpack_from('<H', data, 0x36)
e_phnum,    = struct.unpack_from('<H', data, 0x38)
e_shentsize,= struct.unpack_from('<H', data, 0x3a)
e_shnum,    = struct.unpack_from('<H', data, 0x3c)

ph_base = e_phoff
first_load_idx = None
for i in range(e_phnum):
    off = ph_base + i * e_phentsize
    p_type, = struct.unpack_from('<I', data, off)
    if p_type == 1:
        first_load_idx = i
        break
if first_load_idx is None:
    sys.exit("no LOAD segment")

ph = ph_base + first_load_idx * e_phentsize
p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = \
    struct.unpack_from('<IIQQQQQQ', data, ph)
print(f"first LOAD: offset={hex(p_offset)} filesz={hex(p_filesz)} align={hex(p_align)}")

TARGET = 0x1000
if p_offset == TARGET:
    print("already at 0x1000; nothing to do")
    out.write_bytes(data)
    sys.exit(0)
if p_offset > TARGET:
    sys.exit(f"can't shrink offset {hex(p_offset)} -> {hex(TARGET)}")

pad = TARGET - p_offset
new = bytearray(data[:p_offset] + b'\x00'*pad + data[p_offset:])

for i in range(e_phnum):
    off = ph_base + i * e_phentsize
    p_type_i, p_flags_i, p_offset_i = struct.unpack_from('<IIQ', new, off)
    if p_offset_i >= p_offset and p_type_i != 0:
        struct.pack_into('<Q', new, off + 8, p_offset_i + pad)
if e_shoff >= p_offset:
    struct.pack_into('<Q', new, 0x28, e_shoff + pad)
new_shoff, = struct.unpack_from('<Q', new, 0x28)
for i in range(e_shnum):
    off = new_shoff + i * e_shentsize
    sh_offset, = struct.unpack_from('<Q', new, off + 0x18)
    if sh_offset >= p_offset:
        struct.pack_into('<Q', new, off + 0x18, sh_offset + pad)

out.write_bytes(new)
check = bytearray(out.read_bytes())
p_offset_new, = struct.unpack_from('<Q', check, ph + 8)
print(f"after pad: first LOAD offset = {hex(p_offset_new)}")
