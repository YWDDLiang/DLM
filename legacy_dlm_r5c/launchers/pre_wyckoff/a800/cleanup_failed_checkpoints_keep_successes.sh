#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-cleanup-failed-checkpoints}"
DRY_RUN="${DRY_RUN:-1}"

KEEP_LOWLR5="${KEEP_LOWLR5:-runs/20260515_101500-sft-low-lr-5epoch/outputs/llada_sft_low_lr_5epoch/final}"
KEEP_SFTBEST="${KEEP_SFTBEST:-runs/20260521_211500-final07-refined-seal1/outputs/sft_refined_seal1/final}"
KEEP_R2="${KEEP_R2:-runs/20260527_semalign_selfimprove_r2/outputs/stage_b/final}"

cd "${PROJECT_ROOT}"
NOTE_DIR="runs/${RUN_ID}/notes"
mkdir -p "${NOTE_DIR}"
export DRY_RUN KEEP_LOWLR5 KEEP_SFTBEST KEEP_R2 NOTE_DIR

python - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(".").resolve()
dry_run = os.environ.get("DRY_RUN", "1") == "1"
note_dir = Path(os.environ["NOTE_DIR"])
keep_paths = {
    (root / os.environ["KEEP_LOWLR5"]).resolve(),
    (root / os.environ["KEEP_SFTBEST"]).resolve(),
    (root / os.environ["KEEP_R2"]).resolve(),
}
weight_names = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
}


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def has_direct_weight(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_file():
            continue
        if child.name in weight_names:
            return True
        if child.name.startswith("pytorch_model-") and child.name.endswith(".bin"):
            return True
        if child.name.endswith(".safetensors"):
            return True
    return False


candidate_dirs: list[Path] = []
for path in sorted(Path("runs").rglob("*")):
    if not path.is_dir():
        continue
    if path.name != "final" and not path.name.startswith("step-"):
        continue
    resolved = path.resolve()
    if any(resolved == keep or is_inside(resolved, keep) for keep in keep_paths):
        continue
    if has_direct_weight(path):
        candidate_dirs.append(path)

manifest = []
for path in candidate_dirs:
    size = 0
    files = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            try:
                size += child.stat().st_size
            except OSError:
                pass
    manifest.append({"path": str(path), "bytes": int(size), "files": int(files)})

removed = []
for item in manifest:
    path = Path(item["path"])
    if not dry_run:
        shutil.rmtree(path)
    removed.append(item)

payload = {
    "dry_run": dry_run,
    "keep": [str(path.relative_to(root)) for path in sorted(keep_paths)],
    "candidate_count": len(manifest),
    "candidate_bytes": sum(item["bytes"] for item in manifest),
    "removed": removed,
    "policy": "delete only checkpoint/final directories with direct model weight files; preserve logs, notes, samples, metrics",
}
note_dir.mkdir(parents=True, exist_ok=True)
(note_dir / "failed_checkpoint_cleanup_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
