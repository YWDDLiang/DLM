#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-cleanup-checkpoints}"
DRY_RUN="${DRY_RUN:-0}"
CANCEL_RAW_MINI="${CANCEL_RAW_MINI:-1}"
export RUN_ID DRY_RUN

KEEP_LOWLR5="runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final"
KEEP_SFTBEST="runs/20260521_211500-final07-refined-seal1/outputs/sft_refined_seal1/final"

cd "${PROJECT_ROOT}"
NOTE_DIR="runs/${RUN_ID}/notes"
LOG_DIR="runs/${RUN_ID}/logs"
mkdir -p "${NOTE_DIR}" "${LOG_DIR}"

if [[ "${CANCEL_RAW_MINI}" == "1" ]]; then
  mapfile -t RAW_MINI_JOBS < <(squeue -u "${USER}" -h -o "%i %j" | awk '$2 == "rawminifx" {print $1}')
  printf '%s\n' "${RAW_MINI_JOBS[@]}" > "${NOTE_DIR}/cancelled_rawmini_jobs.txt"
  if [[ "${#RAW_MINI_JOBS[@]}" -gt 0 ]]; then
    scancel "${RAW_MINI_JOBS[@]}" || true
  fi
fi

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(".").resolve()
run_id = os.environ["RUN_ID"]
dry_run = os.environ.get("DRY_RUN", "0") == "1"
note_dir = Path("runs") / run_id / "notes"
keep = {
    (root / "runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final").resolve(),
    (root / "runs/20260521_211500-final07-refined-seal1/outputs/sft_refined_seal1/final").resolve(),
}
weight_names = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
}

def has_weight_file(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and (child.name in weight_names or child.name.startswith("pytorch_model-") or child.name.endswith(".safetensors")):
            return True
    return False

candidates = []
for path in sorted(Path("runs").rglob("*")):
    if not path.is_dir():
        continue
    if path.name != "final" and not path.name.startswith("step-") and path.name != "checkpoints":
        continue
    if path.name == "checkpoints":
        continue
    resolved = path.resolve()
    if resolved in keep:
        continue
    if has_weight_file(path):
        candidates.append(path)

manifest = []
for path in candidates:
    size = 0
    files = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            try:
                size += child.stat().st_size
            except OSError:
                pass
    manifest.append({"path": str(path), "bytes": size, "files": files})

removed = []
for item in manifest:
    path = Path(item["path"])
    if not dry_run:
        shutil.rmtree(path)
    removed.append(item)

payload = {
    "dry_run": dry_run,
    "keep": [str(path.relative_to(root)) for path in sorted(keep)],
    "candidate_count": len(manifest),
    "candidate_bytes": sum(item["bytes"] for item in manifest),
    "removed": removed,
}
note_dir.mkdir(parents=True, exist_ok=True)
(note_dir / "checkpoint_cleanup_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
