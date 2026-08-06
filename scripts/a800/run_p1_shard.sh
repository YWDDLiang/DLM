#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 1 ] || [[ ! "$1" =~ ^[0-7]$ ]]; then
  echo "Usage: $0 SHARD_INDEX  # SHARD_INDEX must be 0..7" >&2
  exit 2
fi

SHARD_INDEX="$1"
SHARD_COUNT=8
PROJECT_ROOT="${PROJECT_ROOT:-/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion}"
SOURCE_ROOT="${MP20_SOURCE_ROOT:-${PROJECT_ROOT}/reference/crysllmgen/data/mp_20}"
DESTINATION_ROOT="${WQ_P1_ROOT:-${PROJECT_ROOT}/data/wqcodiff/p1_v3}"

cd "${PROJECT_ROOT}"
mkdir -p "${DESTINATION_ROOT}"
for split in train val test; do
  csv="${SOURCE_ROOT}/${split}.csv"
  output="${DESTINATION_ROOT}/${split}.part-$(printf '%03d' "${SHARD_INDEX}")-of-008.jsonl"
  summary="${output}.summary.json"
  if [ ! -f "${csv}" ]; then
    echo "Missing frozen MP20 input: ${csv}" >&2
    exit 3
  fi
  if [ -e "${output}" ] || [ -e "${summary}" ]; then
    echo "Immutable P1 shard output already exists: ${output}" >&2
    exit 4
  fi
done

for split in train val test; do
  csv="${SOURCE_ROOT}/${split}.csv"
  output="${DESTINATION_ROOT}/${split}.part-$(printf '%03d' "${SHARD_INDEX}")-of-008.jsonl"
  python -m crystal_dlm.wqcodiff \
    --protocol configs/experiments/wyckoff_codiffusion/protocol_v3.yaml \
    preprocess \
    --csv "${csv}" \
    --split "${split}" \
    --output "${output}" \
    --shard-index "${SHARD_INDEX}" \
    --shard-count "${SHARD_COUNT}"
done

python - "${DESTINATION_ROOT}" "${SHARD_INDEX}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
index = int(sys.argv[2])
files = []
for split in ("train", "val", "test"):
    path = root / f"{split}.part-{index:03d}-of-008.jsonl"
    summary = path.with_suffix(path.suffix + ".summary.json")
    files.append({
        "split": split,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary": str(summary),
        "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    })
payload = {
    "schema": "wqcodiff_p1_shard_manifest_v1",
    "shard_index": index,
    "shard_count": 8,
    "files": files,
}
destination = root / f"shard-{index:03d}-of-008.manifest.json"
with destination.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
