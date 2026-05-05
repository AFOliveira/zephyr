#!/usr/bin/env python3
"""Auditable Whisper Tiny EN host-only vs ET-SoC1 hybrid runner.

The hybrid mode keeps audio preprocessing, graph orchestration, non-MatMul
decoder work, tokenization, and final text decoding on the host.  ET-SoC1
computes the final decoder logits projection used for greedy token choice.
That is intentionally narrower than a native on-chip ONNX executor, but it is
audio-to-text, comparable against host-only, and every silicon logits tensor is
checked against the ONNXRuntime logits tensor for the same token step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import gcd
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AMP_ROOT = ROOT.parent
PHASE0 = ROOT / "phase0"
MODELS = ROOT.parent / "luxonis-models" / "downloads" / "onnx" / "extracted"
DEFAULT_ENCODER = MODELS / "whisper_tiny_en_encoder_qaihub.onnx" / "whisper_tiny_en_encoder_qaihub.onnx"
DEFAULT_DECODER = MODELS / "whisper_tiny_en_decoder_qaihub.onnx" / "whisper_tiny_en_decoder_qaihub.onnx"
DEFAULT_AUDIO = Path("/usr/share/sounds/speech-dispatcher/test.wav")

SRC = ROOT / "whisper_decoder_colmatmul_vpu_argbuf.c"
CRT = AMP_ROOT / "hart-report" / "hart_report_crt.S"

GCC = os.environ.get("GCC", "/home/afonso/et/bin/riscv64-unknown-elf-gcc")
OBJCOPY = os.environ.get("OBJCOPY", "/home/afonso/et/bin/riscv64-unknown-elf-objcopy")
LAYOUT = Path(os.environ.get(
    "LAYOUT", "/home/afonso/et-platform-vidas/erbium-examples/runtime/erbium-soc1sim/layout.c"))
LD_SCRIPT = Path(os.environ.get(
    "LD_SCRIPT", "/home/afonso/et-platform-vidas/et-common-libs/share/erbium-soc1sim/erbium.ld"))
SOC1SIM_STAGED_INCLUDE = Path(os.environ.get(
    "SOC1SIM_STAGED_INCLUDE",
    "/home/afonso/zephyr/build-etsoc1-halified-emlearn/aifoundry/erbium-soc1sim-staged-include"))

REMOTE_HOST = "root@esperanto-soc6"
REMOTE_BASE = "/root/afonso/zephyr-u-mode/zephyr-emlearn-silicon-v2"
REMOTE_PARENT = f"{REMOTE_BASE}/erbium-amp-probe"
REMOTE_ROOT = f"{REMOTE_PARENT}/whisper-real/e2e-audit"
SSH_OPTS = shlex.split(os.environ.get(
    "SSH_OPTS",
    "-o BatchMode=yes -o NumberOfPasswordPrompts=0 "
    "-o PreferredAuthentications=publickey -o ServerAliveInterval=10 "
    "-o ServerAliveCountMax=12 -o ConnectTimeout=10"))
SSH_CMD = ["ssh", *SSH_OPTS, REMOTE_HOST]
RSYNC_RSH = "ssh " + shlex.join(SSH_OPTS)

MAGIC = 0x57444C47
ALIGN = 0x10000

BASE_FLAGS = [
    "-march=rv64imfc",
    "-mabi=lp64f",
    "-mcmodel=medany",
    "-nostdlib",
    "-fno-zero-initialized-in-bss",
    "-ffunction-sections",
    "-fdata-sections",
    f"-I{SOC1SIM_STAGED_INCLUDE}",
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-I/home/afonso/et-platform-vidas/hal/platform/erbium/include",
    "-I/home/afonso/et-platform-vidas/hal/platform/etsoc/include",
    "-I/home/afonso/et-platform-vidas/et-common-libs/include",
    "-Wl,--gc-sections",
    "-Wl,--no-warn-rwx-segments",
    "-Wl,--emit-relocs",
    "-T",
    str(LD_SCRIPT),
]


def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def run_retry(cmd: list[str], attempts: int = 5, delay_s: float = 5.0,
              **kwargs: Any) -> subprocess.CompletedProcess:
    last: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None

    for attempt in range(1, attempts + 1):
        try:
            return run(cmd, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last = exc
            print(f"command failed attempt {attempt}/{attempts}: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(delay_s)

    assert last is not None
    raise last


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def align_up(v: int, align: int = ALIGN) -> int:
    return (v + align - 1) & ~(align - 1)


def align_cols(v: int) -> int:
    return (v + 15) & ~15


def add_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {
        vi.name: vi
        for vi in list(inferred.graph.value_info)
        + list(inferred.graph.output)
        + list(inferred.graph.input)
    }
    existing = {o.name for o in model.graph.output}
    for name in names:
        if name in existing:
            continue
        vi = known.get(name)
        if vi is None:
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        model.graph.output.append(vi)
        existing.add(name)
    return model


def make_session(model_path: Path, outputs: list[str] | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    if outputs is None:
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    model = onnx.load(model_path)
    patched = add_outputs(model, outputs)
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.close()
    onnx.save(patched, tmp.name)
    return ort.InferenceSession(tmp.name, sess_options=opts, providers=["CPUExecutionProvider"])


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    return int(sr), data


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    common = gcd(src_sr, dst_sr)
    up = dst_sr // common
    down = src_sr // common
    out = resample_poly(audio, up, down).astype(np.float32)
    return np.ascontiguousarray(out)


def prepare_features(audio_path: Path, run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    original_sr, original_audio = read_audio(audio_path)
    audio = resample_audio(original_audio, original_sr, 16000)
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    features = extractor(audio, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
    features.astype("<f4").tofile(run_dir / "input_features_1x80x3000.bin")
    return features, {
        "audio_path": str(audio_path),
        "audio_sha256": sha256_path(audio_path),
        "original_sample_rate": original_sr,
        "sample_rate": 16000,
        "original_samples": int(original_audio.shape[0]),
        "samples": int(audio.shape[0]),
        "feature_shape": list(features.shape),
        "feature_sha256": sha256_array(features),
    }


def decoder_prompt(tokenizer: WhisperTokenizer) -> list[int]:
    return [
        tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        tokenizer.convert_tokens_to_ids("<|en|>"),
        tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
    ]


def layout_for(k_dim: int, n_cols: int, active_harts: int) -> dict[str, int]:
    out_offset = 0x4000
    out_len = n_cols * 4
    tmp_offset = align_up(out_offset + out_len)
    tmp_len = active_harts * align_cols(n_cols) * 4
    act_offset = align_up(tmp_offset + tmp_len)
    act_len = k_dim * 4
    wt_offset = align_up(act_offset + act_len)
    wt_len = n_cols * k_dim * 4
    ref_offset = align_up(wt_offset + wt_len)
    ref_len = n_cols * 4
    end = ref_offset + ref_len
    if end > 16 * 1024 * 1024:
        raise SystemExit(f"argument buffer layout exceeds 16 MiB: 0x{end:x}")
    return {
        "out_offset": out_offset,
        "tmp_offset": tmp_offset,
        "act_offset": act_offset,
        "wt_offset": wt_offset,
        "ref_offset": ref_offset,
        "end": end,
    }


class SiliconLogitsRunner:
    def __init__(self, run_dir: Path, weight: np.ndarray, active_harts: int,
                 timeout: int, tile_cols: int, device_argmax_only: bool = False) -> None:
        self.run_dir = run_dir
        self.local = run_dir / "silicon_logits"
        self.local.mkdir(parents=True, exist_ok=True)
        self.active_harts = active_harts
        self.timeout = timeout
        self.device_argmax_only = device_argmax_only
        self.k_dim, self.n_total = weight.shape
        if self.k_dim % 16 != 0:
            raise SystemExit(f"logits K must be divisible by 16, got {self.k_dim}")
        if tile_cols <= 0 or tile_cols % 2:
            raise SystemExit("--tile-cols must be a positive even integer")
        self.tile_cols = tile_cols
        self.tiles: list[tuple[int, int, int]] = []
        for col0 in range(0, self.n_total, self.tile_cols):
            col1 = min(col0 + self.tile_cols, self.n_total)
            if (col1 - col0) % 2:
                raise SystemExit(f"odd logits tile cols {col0}:{col1}")
            self.tiles.append((len(self.tiles), col0, col1))
        self.weight_t = weight.T.astype(np.float32).copy()
        self.remote = f"{REMOTE_ROOT}/{run_dir.name}"
        self.remote_static = (
            f"{REMOTE_ROOT}/static_logits_k{self.k_dim}_n{self.n_total}"
            f"_h{self.active_harts}_tile{self.tile_cols}_auditref0"
            f"_argmax{int(self.device_argmax_only)}"
        )
        self._build_and_stage_static_assets()

    def _build_elf(self, bdir: Path, n_cols: int) -> Path:
        mem = layout_for(self.k_dim, n_cols, self.active_harts)
        elf = bdir / f"logits_colmatmul_n{n_cols}.elf"
        cmd = [
            GCC,
            "-O3",
            "-funroll-loops",
            "-DINPUTS_PRELOADED=1u",
            "-DAUDIT_REF=0u",
            f"-DARGMAX_ONLY={int(self.device_argmax_only)}u",
            *BASE_FLAGS,
            f"-DNUM_HARTS={self.active_harts}",
            f"-DACTIVE_HARTS={self.active_harts}u",
            f"-DK_DIM={self.k_dim}u",
            f"-DN_COLS={n_cols}u",
            f"-DOUT_OFFSET=0x{mem['out_offset']:x}u",
            f"-DTMP_OFFSET=0x{mem['tmp_offset']:x}u",
            f"-DACT_OFFSET=0x{mem['act_offset']:x}u",
            f"-DWT_OFFSET=0x{mem['wt_offset']:x}u",
            f"-DREF_OFFSET=0x{mem['ref_offset']:x}u",
            "-o",
            str(elf),
            str(SRC),
            str(CRT),
            str(LAYOUT),
        ]
        run(cmd)
        return elf

    def _build_and_stage_static_assets(self) -> None:
        print("checking esperanto-soc6 SSH access", flush=True)
        run_retry([*SSH_CMD, "true"], timeout=30)
        run_retry([*SSH_CMD, f"mkdir -p {shlex.quote(self.remote)} {shlex.quote(self.remote_static)}"],
                  timeout=30)
        widths = sorted({col1 - col0 for _, col0, col1 in self.tiles})
        self.elf_for_width: dict[int, Path] = {}
        build = self.local / "build"
        build.mkdir(parents=True, exist_ok=True)
        for width in widths:
            wdir = build / f"n{width}"
            wdir.mkdir(parents=True, exist_ok=True)
            self.elf_for_width[width] = self._build_elf(wdir, width)
        weight_paths = []
        for tile_id, col0, col1 in self.tiles:
            path = self.local / f"weight_t{tile_id:02d}_{col0}_{col1}.bin"
            self.weight_t[col0:col1].astype("<f4").tofile(path)
            weight_paths.append(path)
        static_names = [p.name for p in self.elf_for_width.values()] + [p.name for p in weight_paths]
        static_check = " && ".join(f"test -s {shlex.quote(name)}" for name in static_names)
        have_static = subprocess.run(
            [*SSH_CMD, f"cd {shlex.quote(self.remote_static)} && {static_check}"],
            text=True,
        ).returncode == 0
        if not have_static:
            run_retry(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
                       *(str(p) for p in self.elf_for_width.values()),
                       *(str(p) for p in weight_paths),
                       f"{REMOTE_HOST}:{self.remote_static}/"],
                      attempts=5, delay_s=10.0)

        link_cmd = "set -e; " + "; ".join(
            f"ln -sf {shlex.quote(self.remote_static + '/' + name)} {shlex.quote(self.remote + '/' + name)}"
            for name in static_names
        )
        run_retry([*SSH_CMD, link_cmd], timeout=30)

    def run_logits(self, step: int, act: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        step_dir = self.local / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        act = np.asarray(act, dtype=np.float32).reshape(self.k_dim)
        ref = np.asarray(ref, dtype=np.float32).reshape(self.n_total)
        act_path = step_dir / "act.bin"
        act.astype("<f4").tofile(act_path)
        run(["rsync", "-e", RSYNC_RSH, "-aq", "--partial", "--inplace",
             str(act_path), f"{REMOTE_HOST}:{self.remote}/"])

        script = f"""set -euo pipefail
BASE={REMOTE_BASE}
PARENT={REMOTE_PARENT}
WORK={self.remote}
LAUNCH=$PARENT/erbium_soc1sim_argbuf
ZERO=$PARENT/zero2m.bin
cd "$WORK"
export LD_LIBRARY_PATH=$BASE:$PARENT:${{LD_LIBRARY_PATH:-}}
"""
        for tile_id, col0, col1 in self.tiles:
            width = col1 - col0
            mem = layout_for(self.k_dim, width, self.active_harts)
            elf = self.elf_for_width[width].name
            out_extract = ""
            if not self.device_argmax_only:
                out_extract = (
                    f'Path("out_s{step:03d}_t{tile_id:02d}.bin").write_bytes('
                    f"data[0x{mem['out_offset']:x}:0x{mem['out_offset']:x} + {width} * 4])"
                )
            script += f"""
echo "=== step {step} tile {tile_id:02d} cols {col0}:{col1} ==="
"$LAUNCH" --elf-load ./{elf} --shire 0 --file_load 0x0,$ZERO \\
  --file_load 0x{mem['act_offset']:x},act.bin \\
  --file_load 0x{mem['wt_offset']:x},weight_t{tile_id:02d}_{col0}_{col1}.bin \\
  --dump_after dump_s{step:03d}_t{tile_id:02d}.bin --timeout {self.timeout} \\
  > run_s{step:03d}_t{tile_id:02d}.log 2>&1
grep -oE 'Kernel wait seconds: [0-9.]+' run_s{step:03d}_t{tile_id:02d}.log | tail -1 || true
python3 - <<'PY'
import json, re, struct
from pathlib import Path
data = Path("dump_s{step:03d}_t{tile_id:02d}.bin").read_bytes()
{out_extract}
slot_struct = struct.Struct("<16I")
done = 0
active_mask = 0
for h in range({self.active_harts}):
    f = slot_struct.unpack_from(data, h * 64)
    if f[0] == {MAGIC} and f[8] == 1:
        done += 1
        active_mask |= 1 << f[1]
summary = struct.unpack_from("<16I", data, 0x1000)
argmax_value = struct.unpack("<f", struct.pack("<I", summary[14]))[0]
text = Path("run_s{step:03d}_t{tile_id:02d}.log").read_text(errors="ignore")
m = re.search(r"Kernel wait seconds:\\s*([0-9.eE+-]+)", text)
Path("summary_s{step:03d}_t{tile_id:02d}.json").write_text(json.dumps({{
    "step": {step}, "tile_id": {tile_id}, "col0": {col0}, "col1": {col1},
    "wait_s": float(m.group(1)) if m else None,
    "slot_done_count": done, "slot_active_mask": active_mask,
    "summary_magic": summary[0],
    "summary_output_hash": summary[7],
    "summary_argmax_local_col": summary[13],
    "summary_argmax_global_col": {col0} + summary[13],
    "summary_argmax_value_bits": summary[14],
    "summary_argmax_value": argmax_value,
    "summary_argmax_only": bool(summary[15]),
    "stream_error": "Stream error" in text,
    "kernel_launch_error": "Error on kernel launch" in text
}}, indent=2) + "\\n")
PY
"""
        run([*SSH_CMD, "bash", "-s"], input=script)
        fetch_paths = []
        for tile_id, _, _ in self.tiles:
            fetch_paths.extend([
                f"{REMOTE_HOST}:{self.remote}/run_s{step:03d}_t{tile_id:02d}.log",
                f"{REMOTE_HOST}:{self.remote}/summary_s{step:03d}_t{tile_id:02d}.json",
            ])
            if not self.device_argmax_only:
                fetch_paths.append(f"{REMOTE_HOST}:{self.remote}/out_s{step:03d}_t{tile_id:02d}.bin")
        run(["rsync", "-e", RSYNC_RSH, "-aq", *fetch_paths, str(step_dir) + "/"])

        stitched = None if self.device_argmax_only else np.zeros((self.n_total,), dtype=np.float32)
        tile_reports = []
        total_wait = 0.0
        tile_ok = True
        best_global_col = 0
        best_value = -np.inf
        for tile_id, col0, col1 in self.tiles:
            summary = json.loads((step_dir / f"summary_s{step:03d}_t{tile_id:02d}.json").read_text())
            wait = summary["wait_s"] or 0.0
            total_wait += wait
            local_ref = ref[col0:col1]
            local_ref_argmax = int(local_ref.argmax())
            local_ref_best = float(local_ref[local_ref_argmax])
            local_argmax = int(summary["summary_argmax_local_col"])
            local_argmax_ok = 0 <= local_argmax < (col1 - col0)
            global_argmax = col0 + local_argmax if local_argmax_ok else -1
            summary_value = float(summary["summary_argmax_value"])
            value_ref_at_summary = (
                float(local_ref[local_argmax]) if local_argmax_ok else float("nan")
            )
            summary_value_abs = (
                abs(summary_value - value_ref_at_summary)
                if local_argmax_ok else float("inf")
            )
            tile_argmax_match = local_argmax_ok and local_argmax == local_ref_argmax
            summary_ok = (
                summary["summary_magic"] == MAGIC
                and summary["slot_done_count"] == self.active_harts
                and not summary["stream_error"]
                and not summary["kernel_launch_error"]
            )

            if self.device_argmax_only:
                ok = summary_ok and tile_argmax_match and summary_value_abs <= 1e-4
                max_abs: float | None = None
                mean_abs: float | None = None
            else:
                assert stitched is not None
                out = np.fromfile(step_dir / f"out_s{step:03d}_t{tile_id:02d}.bin", dtype=np.float32)
                stitched[col0:col1] = out
                diff = np.abs(out - local_ref)
                local_out_argmax = int(out.argmax())
                local_out_best = float(out[local_out_argmax])
                summary_matches_output = (
                    local_argmax_ok
                    and local_argmax == local_out_argmax
                    and abs(summary_value - local_out_best) <= 1e-4
                )
                ok = summary_ok and summary_matches_output and float(diff.max()) <= 1e-4
                max_abs = float(diff.max())
                mean_abs = float(diff.mean())

            tile_ok = tile_ok and ok
            if local_argmax_ok and (
                summary_value > best_value
                or (summary_value == best_value and global_argmax < best_global_col)
            ):
                best_value = summary_value
                best_global_col = global_argmax
            tile_reports.append({
                "tile_id": tile_id,
                "col0": col0,
                "col1": col1,
                "wait_s": wait,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "host_tile_argmax_local": local_ref_argmax,
                "host_tile_argmax_global": col0 + local_ref_argmax,
                "host_tile_argmax_value": local_ref_best,
                "device_tile_argmax_local": local_argmax,
                "device_tile_argmax_global": global_argmax,
                "device_tile_argmax_value": summary_value,
                "device_tile_argmax_value_abs_diff": summary_value_abs,
                "tile_argmax_match": tile_argmax_match,
                "slot_done_count": summary["slot_done_count"],
                "active_mask": hex(summary["slot_active_mask"]),
                "summary_present": summary["summary_magic"] == MAGIC,
                "summary_argmax_only": summary["summary_argmax_only"],
                "summary_output_hash": summary["summary_output_hash"],
                "stream_error": summary["stream_error"],
                "status": "ok" if ok else "fail",
            })

        if self.device_argmax_only:
            argmax_silicon = int(best_global_col)
            argmax_host = int(ref.argmax())
            max_abs = None
            mean_abs = None
            p99_abs = None
            allclose_1e4 = None
        else:
            assert stitched is not None
            diff = np.abs(stitched - ref)
            argmax_silicon = int(stitched.argmax())
            argmax_host = int(ref.argmax())
            max_abs = float(diff.max())
            mean_abs = float(diff.mean())
            p99_abs = float(np.percentile(diff, 99))
            allclose_1e4 = bool(np.allclose(stitched, ref, rtol=1e-4, atol=1e-4))

        report = {
            "step": step,
            "device_argmax_only": self.device_argmax_only,
            "logits_full_audit_available": not self.device_argmax_only,
            "total_wait_s": total_wait,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "p99_abs": p99_abs,
            "allclose_1e4": allclose_1e4,
            "tile_ok": bool(tile_ok),
            "argmax_silicon": argmax_silicon,
            "argmax_host": argmax_host,
            "argmax_match": bool(argmax_silicon == argmax_host),
            "device_argmax_value": float(best_value),
            "tiles": tile_reports,
        }
        report["audit_pass"] = bool(
            report["tile_ok"]
            and report["argmax_match"]
            and (self.device_argmax_only or report["allclose_1e4"])
        )
        if stitched is not None:
            stitched.astype("<f4").tofile(step_dir / "stitched_logits.bin")
        (step_dir / "silicon_logits_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["audit_pass"]:
            raise SystemExit(f"silicon logits audit failed at step {step}: {step_dir}")
        return stitched, report


def decode_loop(
    mode: str,
    run_dir: Path,
    decoder: ort.InferenceSession,
    tokenizer: WhisperTokenizer,
    k_cross: np.ndarray,
    v_cross: np.ndarray,
    max_new_tokens: int,
    silicon: SiliconLogitsRunner | None,
) -> dict[str, Any]:
    prompt = decoder_prompt(tokenizer)
    generated = [prompt[0]]
    forced_after_sot = prompt[1:]
    k_self = np.zeros((4, 6, 64, 224), dtype=np.float32)
    v_self = np.zeros((4, 6, 224, 64), dtype=np.float32)
    steps = []
    silicon_wait = 0.0
    t0 = time.perf_counter()
    for step in range(max_new_tokens + len(forced_after_sot)):
        current = generated[-1]
        inputs = {
            "x": np.asarray([[current]], dtype=np.int32),
            "index": np.asarray([[step]], dtype=np.int32),
            "k_cache_cross": k_cross,
            "v_cache_cross": v_cross,
            "k_cache_self": k_self,
            "v_cache_self": v_self,
        }
        act, host_logits, k_next, v_next = decoder.run(
            ["/ln/LayerNormalization_output_0", "logits", "k_cache", "v_cache"], inputs
        )
        host_logits_1d = np.asarray(host_logits, dtype=np.float32).reshape(-1)
        host_argmax = int(host_logits_1d.argmax())
        silicon_report = None
        if mode == "hybrid-silicon":
            assert silicon is not None
            silicon_logits, silicon_report = silicon.run_logits(
                step, np.asarray(act, dtype=np.float32).reshape(-1), host_logits_1d
            )
            if silicon_logits is not None:
                chosen_argmax = int(silicon_logits.argmax())
            else:
                chosen_argmax = int(silicon_report["argmax_silicon"])
            silicon_wait += silicon_report["total_wait_s"]
        else:
            chosen_argmax = host_argmax

        if step < len(forced_after_sot):
            next_id = forced_after_sot[step]
            forced = True
        else:
            next_id = chosen_argmax
            forced = False

        generated.append(next_id)
        steps.append({
            "step": step,
            "input_token": int(current),
            "input_text": tokenizer.decode([current]),
            "host_argmax": host_argmax,
            "host_argmax_text": tokenizer.decode([host_argmax]),
            "chosen_argmax": chosen_argmax,
            "chosen_argmax_text": tokenizer.decode([chosen_argmax]),
            "next_token": int(next_id),
            "next_text": tokenizer.decode([next_id]),
            "forced": forced,
            "silicon": silicon_report,
        })
        k_self = np.asarray(k_next, dtype=np.float32)
        v_self = np.asarray(v_next, dtype=np.float32)
        if not forced and next_id == tokenizer.eos_token_id:
            break
    total_s = time.perf_counter() - t0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "mode": mode,
        "tokens": [int(x) for x in generated],
        "token_text": [tokenizer.decode([int(x)]) for x in generated],
        "text": text,
        "steps": steps,
        "decode_wall_s": total_s,
        "silicon_logits_wait_s": silicon_wait,
        "generated_nonprompt_tokens": max(0, len(generated) - len(prompt)),
        "tokens_per_s_wall": (max(0, len(generated) - len(prompt)) / total_s) if total_s else 0.0,
        "tokens_per_s_silicon_logits_only": (max(0, len(generated) - len(prompt)) / silicon_wait) if silicon_wait else None,
    }


def compare_runs(host: dict[str, Any], hybrid: dict[str, Any] | None) -> dict[str, Any]:
    if hybrid is None:
        return {"available": False}
    host_tokens = host["tokens"]
    hybrid_tokens = hybrid["tokens"]
    limit = min(len(host_tokens), len(hybrid_tokens))
    mismatches = [
        {"index": i, "host": host_tokens[i], "hybrid": hybrid_tokens[i]}
        for i in range(limit)
        if host_tokens[i] != hybrid_tokens[i]
    ]
    if len(host_tokens) != len(hybrid_tokens):
        mismatches.append({
            "index": limit,
            "host_tail": host_tokens[limit:],
            "hybrid_tail": hybrid_tokens[limit:],
        })
    silicon_steps = [s["silicon"] for s in hybrid["steps"] if s.get("silicon")]
    allclose_values = [s.get("allclose_1e4") for s in silicon_steps]
    full_logits_audit_available = bool(silicon_steps) and all(v is not None for v in allclose_values)
    max_abs_values = [s.get("max_abs") for s in silicon_steps if s.get("max_abs") is not None]
    return {
        "available": True,
        "token_sequence_match": not mismatches,
        "text_match": host["text"] == hybrid["text"],
        "mismatches": mismatches,
        "silicon_steps": len(silicon_steps),
        "full_logits_audit_available": full_logits_audit_available,
        "all_silicon_logits_allclose_1e4": (
            all(v is True for v in allclose_values)
            if full_logits_audit_available else None
        ),
        "all_silicon_argmax_match": all(s["argmax_match"] for s in silicon_steps),
        "all_silicon_audit_pass": all(s.get("audit_pass", False) for s in silicon_steps),
        "max_logits_abs": max(max_abs_values) if max_abs_values else None,
        "total_silicon_logits_wait_s": sum((s["total_wait_s"] for s in silicon_steps), 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["host-only", "hybrid-silicon", "both"], default="both")
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--out-dir", type=Path, default=PHASE0 / "e2e_audit_runs")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--active-harts", type=int, default=16)
    ap.add_argument("--tile-cols", type=int, default=8192)
    ap.add_argument("--device-argmax-only", action="store_true",
                    help="Return per-tile device argmax summaries instead of fetching full logits tiles")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    args = ap.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"audio file not found: {args.audio}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / f"{args.mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
    features, input_meta = prepare_features(args.audio, run_dir)
    encoder = make_session(args.encoder)
    decoder = make_session(args.decoder, ["/ln/LayerNormalization_output_0"])
    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(args.decoder).graph.initializer}
    logits_weight = init["onnx::MatMul_3194"].astype(np.float32)

    t_enc0 = time.perf_counter()
    k_cross, v_cross = encoder.run(["k_cache_cross", "v_cache_cross"], {"audio": features})
    encoder_wall_s = time.perf_counter() - t_enc0
    k_cross = np.asarray(k_cross, dtype=np.float32)
    v_cross = np.asarray(v_cross, dtype=np.float32)
    k_cross.astype("<f4").tofile(run_dir / "host_k_cache_cross.bin")
    v_cross.astype("<f4").tofile(run_dir / "host_v_cache_cross.bin")

    host_result = None
    hybrid_result = None
    if args.mode in ("host-only", "both"):
        host_result = decode_loop(
            "host-only", run_dir, decoder, tokenizer, k_cross, v_cross,
            args.max_new_tokens, None,
        )
    if args.mode in ("hybrid-silicon", "both"):
        silicon = SiliconLogitsRunner(
            run_dir, logits_weight, args.active_harts, args.timeout,
            args.tile_cols, args.device_argmax_only,
        )
        hybrid_result = decode_loop(
            "hybrid-silicon", run_dir, decoder, tokenizer, k_cross, v_cross,
            args.max_new_tokens, silicon,
        )
        if host_result is None:
            host_result = decode_loop(
                "host-only", run_dir, decoder, tokenizer, k_cross, v_cross,
                args.max_new_tokens, None,
            )

    assert host_result is not None
    comparison = compare_runs(host_result, hybrid_result)
    report = {
        "timestamp": stamp,
        "scope": (
            "Host-only ONNXRuntime vs host-scheduled ET-SoC1 hybrid. "
            "Hybrid uses ET-SoC1 for final decoder logits projection and host for preprocessing, "
            "encoder, non-logits decoder graph work, tokenization, and text decode."
        ),
        "input": input_meta,
        "model": {
            "encoder": str(args.encoder),
            "encoder_sha256": sha256_path(args.encoder),
            "decoder": str(args.decoder),
            "decoder_sha256": sha256_path(args.decoder),
            "tokenizer": "openai/whisper-tiny.en",
        },
        "environment": {
            "board_host": "esperanto-soc6",
            "remote_root": REMOTE_BASE,
            "remote_work": f"{REMOTE_ROOT}/{run_dir.name}",
            "active_harts": args.active_harts,
            "tile_cols": args.tile_cols,
            "device_argmax_only": args.device_argmax_only,
        },
        "encoder": {
            "host_wall_s": encoder_wall_s,
            "k_cache_cross_shape": list(k_cross.shape),
            "v_cache_cross_shape": list(v_cross.shape),
            "k_cache_cross_sha256": sha256_array(k_cross),
            "v_cache_cross_sha256": sha256_array(v_cross),
        },
        "host_only": host_result,
        "hybrid_silicon": hybrid_result,
        "comparison": comparison,
    }
    (run_dir / "e2e_audit_report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Whisper E2E Audit Report",
        "",
        f"- Audio: `{args.audio}`",
        f"- Mode: `{args.mode}`",
        f"- Host-only text: `{host_result['text']}`",
    ]
    if hybrid_result is not None:
        max_abs_text = (
            f"{comparison['max_logits_abs']:.9g}"
            if comparison["max_logits_abs"] is not None else "not fetched (argmax-only)"
        )
        allclose_text = (
            str(comparison["all_silicon_logits_allclose_1e4"])
            if comparison["all_silicon_logits_allclose_1e4"] is not None
            else "not fetched (argmax-only)"
        )
        lines.extend([
            f"- Hybrid text: `{hybrid_result['text']}`",
            f"- Token sequence match: {comparison['token_sequence_match']}",
            f"- Text match: {comparison['text_match']}",
            f"- Silicon logits allclose: {allclose_text}",
            f"- Silicon logits argmax match: {comparison['all_silicon_argmax_match']}",
            f"- Max logits abs diff: {max_abs_text}",
            f"- Silicon logits wait: {comparison['total_silicon_logits_wait_s']:.6f} s",
        ])
    lines.extend([
        "",
        "Scope: hybrid-silicon is audio-to-text with ET-SoC1 computing the final decoder logits projection used for greedy token choice.",
        "It is not yet a native full-graph ONNX executor on ET-SoC1.",
    ])
    (run_dir / "E2E_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n")

    tsv = PHASE0 / "whisper_e2e_audit.tsv"
    new = not tsv.exists()
    with tsv.open("a", newline="") as f:
        cols = [
            "timestamp", "mode", "audio", "host_text", "hybrid_text",
            "token_sequence_match", "text_match", "silicon_logits_allclose",
            "silicon_argmax_match", "max_logits_abs", "silicon_wait_s",
            "run_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow({
            "timestamp": stamp,
            "mode": args.mode,
            "audio": str(args.audio),
            "host_text": host_result["text"],
            "hybrid_text": hybrid_result["text"] if hybrid_result else "",
            "token_sequence_match": comparison.get("token_sequence_match", ""),
            "text_match": comparison.get("text_match", ""),
            "silicon_logits_allclose": comparison.get("all_silicon_logits_allclose_1e4", ""),
            "silicon_argmax_match": comparison.get("all_silicon_argmax_match", ""),
            "max_logits_abs": comparison.get("max_logits_abs", ""),
            "silicon_wait_s": comparison.get("total_silicon_logits_wait_s", ""),
            "run_dir": str(run_dir),
        })

    print(json.dumps(report, indent=2))
    if hybrid_result is not None and not (
        comparison["token_sequence_match"]
        and comparison["text_match"]
        and comparison["all_silicon_audit_pass"]
        and comparison["all_silicon_argmax_match"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
