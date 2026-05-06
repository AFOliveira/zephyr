#!/usr/bin/env bash
# Stage + run dncnn3 int8_tfma_v2 on esperanto-soc4 under shire-0 lock.
set -uo pipefail

LABEL="${LABEL:-int8_tfma_v2}"
TIMEOUT="${TIMEOUT:-180}"
ELF_SRC="${ELF_SRC:-/tmp/dncnn_tfma/int8_tfma_v2.elf}"

BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT="$BASE/erbium-amp-probe"
LAUNCH="$PARENT/erbium_soc1sim_argbuf"
ZERO64K="$PARENT/zero64k.bin"

[ -f "$ELF_SRC" ] || { echo "missing local $ELF_SRC"; exit 2; }

STAMP=$(date -u +%Y%m%d-%H%M%SZ)
RUN_REL_DIR="runs/${LABEL}-${STAMP}"
RUN_DIR="$BASE/$RUN_REL_DIR"

# Stage ELF + create run dir on soc4
ssh root@esperanto-soc4 "mkdir -p '$RUN_DIR' '$PARENT'" || exit 3
scp -q "$ELF_SRC" "root@esperanto-soc4:$PARENT/${LABEL}.elf" || exit 4

# Run under lock
ssh root@esperanto-soc4 "
set -uo pipefail
export LD_LIBRARY_PATH='$BASE:$PARENT:\${LD_LIBRARY_PATH:-}'
LOCK=/var/lock/etsoc-shire0.lock
(
  exec 9>\"\$LOCK\"
  flock -x -w 600 9 || { echo 'flock timeout'; exit 3; }
  cd '$RUN_DIR' || exit 4
  '$LAUNCH' --elf-load '$PARENT/${LABEL}.elf' --shire 0 \\
    --file_load 0x0,'$ZERO64K' \\
    --dump_after dump.bin \\
    --timeout $TIMEOUT > run.log 2>&1
  rc=\$?
  wait_s=\$(grep -oE 'Kernel wait seconds: [0-9.]+' run.log | awk '{print \$NF}')
  echo \"rc=\$rc wait_s=\${wait_s:-?} dump=\$(stat -c%s dump.bin 2>/dev/null || echo 0)B\"
  exit \$rc
)
" 2>&1
RC=$?

echo "Run dir on soc4: $RUN_DIR"
mkdir -p "/tmp/dncnn_tfma/${LABEL}-${STAMP}"
rsync -aqz "root@esperanto-soc4:${RUN_DIR}/" "/tmp/dncnn_tfma/${LABEL}-${STAMP}/" 2>/dev/null
echo "Local copy: /tmp/dncnn_tfma/${LABEL}-${STAMP}/"
exit $RC
