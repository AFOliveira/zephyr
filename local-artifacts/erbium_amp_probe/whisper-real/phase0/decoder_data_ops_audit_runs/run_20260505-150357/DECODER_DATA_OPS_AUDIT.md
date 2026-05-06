# Whisper Decoder Data Ops Audit

- Decoder step: `3`
- Selected nodes: `2`
- Audited nodes: `0`
- Skipped nodes: `2`
- All audited pass: `False`
- Max abs diff: `None`
- Mean silicon wait: `None s`

| index | op | node | elems | src elems | max abs | wait s | pass |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |

## Skipped

- `Slice` `/Slice_2_fused_with_/0/cross_attn/Slice_1`: Command '['ssh', '-o', 'BatchMode=yes', '-o', 'NumberOfPasswordPrompts=0', '-o', 'PreferredAuthentications=publickey', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=12', '-o', 'ConnectTimeout=10', 'root@esperanto-soc6', 'bash', '-s']' returned non-zero exit status 1.
- `Slice` `/Slice_3_fused_with_/0/cross_attn/Slice_2`: Command '['ssh', '-o', 'BatchMode=yes', '-o', 'NumberOfPasswordPrompts=0', '-o', 'PreferredAuthentications=publickey', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=12', '-o', 'ConnectTimeout=10', 'root@esperanto-soc6', 'bash', '-s']' returned non-zero exit status 1.
