#!/usr/bin/env bash
# One-shot: build → stage → run → pull → audit → append to master TSV.
# Usage: smt_run_batch.sh batch_N.json
set -euo pipefail

BATCH_JSON="${1:?need batch JSON}"
DIR=/home/afonso/zephyr/local-artifacts/erbium_amp_probe/luxonis-real-dncnn
ROOT=/home/afonso/zephyr/local-artifacts/erbium_amp_probe
INC_BASE=/home/afonso/et-platform-vidas
GCC=/home/afonso/et/bin/riscv64-unknown-elf-gcc
MASTER=/home/afonso/etsoc1-luxonis-experiments/results/audit_master_smt.tsv
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
BATCH_NAME="$(basename "$BATCH_JSON" .json)"
OUT=/tmp/smt_${BATCH_NAME}
LOCAL_RES=/tmp/smt_${BATCH_NAME}_results
REMOTE_DIR=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/erbium-amp-probe/smt_${BATCH_NAME}
mkdir -p "$OUT" "$LOCAL_RES"

echo "=== build ==="
python3 - "$BATCH_JSON" "$OUT" "$DIR" "$ROOT" "$INC_BASE" "$GCC" <<'PY'
import json, sys, subprocess, shlex, os
batch_json, out_dir, dir_, root_, inc, gcc = sys.argv[1:7]
specs = json.load(open(batch_json))
common = [
    "-O3", "-funroll-loops", "-falign-functions=128", "-falign-loops=128",
    "-fno-tree-loop-distribute-patterns", "-fno-tree-loop-vectorize",
    "-march=rv64imfc", "-mabi=lp64f", "-mcmodel=medany", "-nostdlib",
    "-fno-zero-initialized-in-bss", "-ffunction-sections", "-fdata-sections",
    f"-I{inc}/et-common-libs/build-headers/erbium-soc1sim-staged-include",
    f"-I{inc}/hal/platform/erbium/include",
    f"-I{inc}/hal/platform/etsoc/include",
    f"-I{inc}/et-common-libs/include",
    "-Wl,--gc-sections", "-Wl,--no-warn-rwx-segments", "-Wl,--emit-relocs",
    "-T", f"{inc}/et-common-libs/share/erbium-soc1sim/erbium.ld",
    "-DDNCNN_STATIC_BLOBS", "-DDNCNN_TILE64_BLOBS", "-DIMG_W=64", "-DIMG_H=64",
    "-DDNCNN_VPU_FUSED9TAP=1", "-DDNCNN_VPU_PREFETCH_READ_WINDOW=1",
]
deps = [
    f"{dir_}/dncnn3_luxonis_real_vpu.c",
    f"{dir_}/dncnn3_luxonis_64x64_input_f32.o",
    f"{dir_}/dncnn3_luxonis_64x64_ort_output_f32.o",
    f"{dir_}/dncnn3_luxonis_240x320_weights_packed_f32.o",
    f"{root_}/hart-report/hart_report_crt.S",
    f"{inc}/erbium-examples/runtime/erbium-soc1sim/layout.c",
]
ok = 0
for s in specs:
    name = s["variant"]
    flags = s.get("flags", [])
    macros = s.get("macros", [])
    harts = s.get("harts", 16)
    passes = s.get("passes", 1)
    extra = list(flags) + [f"-D{m}" for m in macros]
    extra += [f"-DACTIVE_HARTS={harts}u", f"-DNUM_HARTS={harts}", f"-DDNCNN_PASSES={passes}u"]
    elf = f"{out_dir}/{name}.elf"
    cmd = [gcc, *common, *extra, "-o", elf, *deps]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  BUILD FAIL {name}:\n    {res.stderr.strip()[-200:]}")
    else:
        print(f"  built {name}.elf")
        ok += 1
print(f"  total built ok: {ok}/{len(specs)}")
PY

echo
echo "=== stage to soc4 ==="
ssh root@esperanto-soc4 "mkdir -p $REMOTE_DIR" >/dev/null
rsync -aq "$OUT/" "root@esperanto-soc4:$REMOTE_DIR/"

echo
echo "=== run on silicon ==="
RUN_SCRIPT=/tmp/run_${BATCH_NAME}.sh
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -uo pipefail
BASE=/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2
PARENT=\$BASE/erbium-amp-probe
DIR=$REMOTE_DIR
for elf in \$DIR/*.elf; do
    name=\$(basename \$elf .elf)
    \$BASE/board_run.sh --elf "\$elf" --shire 0 \\
        --file-load 0x0,\$PARENT/zero64k.bin \\
        --agent smt-${BATCH_NAME} --purpose "\$name" \\
        --run-id "smt-${BATCH_NAME}-\$name-${STAMP}" \\
        --timeout 180 --lock-timeout 30 || true
done
EOF
scp -q "$RUN_SCRIPT" "root@esperanto-soc4:$REMOTE_DIR/run.sh"
ssh root@esperanto-soc4 "bash $REMOTE_DIR/run.sh" 2>&1 | grep -E "===|ok rc|FAIL"

echo
echo "=== pull and tabulate ==="
for d in $(ssh root@esperanto-soc4 "ls -d /root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2/runs/smt-${BATCH_NAME}-* 2>/dev/null | grep ${STAMP}"); do
    rsync -aq "root@esperanto-soc4:$d/" "$LOCAL_RES/$(basename $d)/"
done

# Append to master TSV
python3 - "$BATCH_JSON" "$LOCAL_RES" "$STAMP" "$MASTER" <<'PY'
import json, struct, glob, os, sys
batch_json, local_res, stamp, master = sys.argv[1:5]
SUMMARY = struct.Struct("<16I")
DNCNN_MAGIC = 0xD3C11004

specs = {s["variant"]: s for s in json.load(open(batch_json))}
batch_name = os.path.basename(batch_json).replace(".json","")

cols = ["exp_id","batch","variant","tile","harts","passes","wait_s","gops",
        "inferences_per_s","ops","allclose","bit_exact","max_abs","mean_abs",
        "output_hash","reference_hash","done_count","magic_ok","status",
        "macros","run_id"]

new_rows = []
for spec_name, spec in specs.items():
    rd = None
    for d in glob.glob(f"{local_res}/smt-*-{spec_name}-{stamp}"):
        rd = d; break
    if not rd or not os.path.exists(f"{rd}/manifest.json"):
        new_rows.append([spec["exp_id"], batch_name, spec_name, 64,
                         spec.get("harts",16), spec.get("passes",1),
                         "", "", "", 0, "False","False","","","-","-",0,
                         "False","build_or_run_fail",
                         ",".join(spec.get("macros",[])),""])
        continue
    man = json.load(open(f"{rd}/manifest.json"))
    if not os.path.exists(f"{rd}/dump.bin"):
        new_rows.append([spec["exp_id"], batch_name, spec_name, 64,
                         spec.get("harts",16), spec.get("passes",1),
                         f"{man.get('kernel_wait_s') or 0:.6f}","","",0,
                         "False","False","","","-","-",0,"False",
                         f"no_dump:{man.get('status','?')}",
                         ",".join(spec.get("macros",[])),os.path.basename(rd)])
        continue
    with open(f"{rd}/dump.bin","rb") as f:
        f.seek(0x1000); s = SUMMARY.unpack(f.read(64))
    ops = s[11] | (s[12]<<32)
    wait_s = man.get("kernel_wait_s") or 0
    gops = ops/wait_s/1e9 if wait_s else 0
    inf_s = 1/wait_s if wait_s else 0
    ok = (s[0]==DNCNN_MAGIC and s[8]==s[1] and s[13]/1e9 <= 1e-5)
    new_rows.append([spec["exp_id"], batch_name, spec_name, 64,
                     s[1], s[2], f"{wait_s:.6f}", f"{gops:.4f}",
                     f"{inf_s:.4f}", ops, str(ok), str(s[9]==s[10]),
                     f"{s[13]/1e9:.3e}", f"{s[14]/1e9:.3e}",
                     f"0x{s[9]:08x}", f"0x{s[10]:08x}", s[8],
                     str(s[0]==DNCNN_MAGIC),
                     "ok" if ok else ("wrong_output" if s[0]==DNCNN_MAGIC else "no_magic"),
                     ",".join(spec.get("macros",[])), os.path.basename(rd)])

# Ensure header exists; otherwise create
if not os.path.exists(master):
    with open(master,"w") as f:
        f.write("\t".join(cols) + "\n")
with open(master,"a") as f:
    for r in new_rows:
        f.write("\t".join(str(x) for x in r) + "\n")

# Print this-batch leaderboard (sorted by GOPS desc among allclose)
print(f"\n=== {batch_name} leaderboard ({len(new_rows)} rows) ===")
print(f"{'exp':>4s} {'variant':28s} {'h':>3s} {'p':>3s} {'wait_s':>8s} {'GOPS':>7s} {'inf/s':>7s} {'max_abs':>10s} status")
for r in sorted(new_rows, key=lambda r: -float(r[7] or 0)):
    g = f"{float(r[7]):>7.3f}" if r[7] else "      -"
    i = f"{float(r[8]):>7.3f}" if r[8] else "      -"
    w = f"{float(r[6]):>8.4f}" if r[6] else "       -"
    print(f"{r[0]:>4d} {r[2]:28s} {r[4]:>3d} {r[5]:>3d} {w} {g} {i} {r[12] if r[12] else '          ':>10s} {r[18]}")
PY

echo
echo "=== global top 10 (across all batches) ==="
python3 - <<PY
import csv
master = "$MASTER"
rows = list(csv.DictReader(open(master), delimiter="\t"))
ok = [r for r in rows if r["allclose"]=="True"]
ok.sort(key=lambda r: -float(r["gops"] or 0))
print(f"{'rank':>4s} {'exp':>4s} {'variant':28s} {'batch':22s} {'GOPS':>7s} {'inf/s':>7s}")
for i,r in enumerate(ok[:10],1):
    print(f"{i:>4d} {r['exp_id']:>4s} {r['variant']:28s} {r['batch']:22s} {float(r['gops']):>7.3f} {float(r['inferences_per_s']):>7.3f}")
print(f"\ntotal: {len(rows)} rows  ({len(ok)} pass audit)")
PY
