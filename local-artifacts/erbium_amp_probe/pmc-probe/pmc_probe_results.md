# U-mode PMC probe results

Date: 2026-05-01
Board host: `esperanto-soc6`
Remote artifact root: `/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/pmc-probe`
Local artifact root: `local-artifacts/erbium_amp_probe/pmc-probe`

## Direct U-mode CSR reads

Usable direct counters:

- `hpmcounter3` (`0xc03`): readable and increasing; firmware maps it to cycles.
- `hpmcounter4` (`0xc04`): readable and increasing; firmware maps it to retired inst0.
- `hpmcounter5` (`0xc05`): readable and increasing when thread-1 work is present; firmware maps it to retired inst1.
- `hpmcounter6` (`0xc06`): readable and increasing; firmware maps it to L2 miss requests.
- `hpmcounter7` (`0xc07`): readable and increasing; firmware maps it to minion icache requests.
- `hpmcounter8` (`0xc08`): readable and increasing slowly; firmware maps it to icache etlink requests.

Not usable from U-mode in the current firmware:

- `cycle` (`0xc00`)
- `time` (`0xc01`)
- `instret` (`0xc02`)
- `hpmcounter9..hpmcounter31` (`0xc09..0xc1f`)

Those negative-control ELFs wrote their start marker, faulted on the CSR read,
and produced runtime stream-error logs. The launcher process currently still
prints success for those runs, so classification uses both the dump marker and
the runtime log.

## Public U-mode PMC syscalls

Valid through `SYSCALL_PMC_SC_SAMPLE`:

- Shire Cache banks `0..3`
- PMC IDs `0..2`
- `0`: cycle counter
- `1`: default PMC0, configured as L2 reads
- `2`: default PMC1, configured as L2 writes

Valid through `SYSCALL_PMC_MS_SAMPLE`:

- Memshires `0..7`
- PMC IDs `0..2`
- `0`: cycle counter
- `1`: default PMC0, configured as mesh reads
- `2`: default PMC1, configured as mesh writes

Not exposed through the public single-sample syscalls:

- `pmc_id=3` (`PMU_SC_ALL` / `PMU_MS_ALL`) returns `0xffffffffffffffff`.
- The internal sample-all syscalls still are not public U-mode ABI.

## Key parsed artifacts

- `parsed/csr_access_classification.tsv`: direct CSR accessibility result.
- `parsed/safe_direct_hpm.tsv`: 16-hart direct `hpmcounter3..8` samples.
- `parsed/safe_syscall_pmc.tsv`: SC/MS syscall samples before and after work.
- `parsed/syscall_edges.tsv`: confirms `pmc_id=3` is not public sample-all.
- `artifact_manifest_local.tsv`: local files and hashes.
- `artifact_manifest_remote.tsv`: remote files, logs, dumps, and hashes.
