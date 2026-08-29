#!/usr/bin/env bash
#SBATCH --job-name=h1a2-d3po-min
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-d3po-min-%j.out

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE="${H1A2_D3PO_BASE_CHECKPOINT:?export H1A2_D3PO_BASE_CHECKPOINT}"
readonly PAIRS="${H1A2_D3PO_PAIR_DATA:?export H1A2_D3PO_PAIR_DATA}"
readonly RUN_DIR="${ROOT}/runs/d3po_256_min_${SLURM_JOB_ID}"
readonly BASE_CONFIG_SHA="8101ee2a917dd1b08d5ef5d90472207a01161a6bcd2b03c78f9e037e756e6300"
readonly BASE_MODEL_SHA="6ea3c2a633706968e4b3e3cf77e98e46399c23e1568333522283472634553ecb"
readonly PAIR_MANIFEST_SHA="90028189c1f3631e6d86a875713f037aa556df583824ff127ffd630050e751b1"
readonly TRAIN_SHA="103f672bac29d913141f3e927efa3abb70767be79fbace87e0a31c14dcce4320"
readonly VALIDATION_SHA="a175d68907f4d9450478ae1fcce6ce0b822017f71806855a9615a36613b6d05e"

for path in \
  "${MODEL}" \
  "${BASE}/adapter_config.json" \
  "${BASE}/adapter_model.safetensors" \
  "${PAIRS}/_SUCCESS" \
  "${PAIRS}/D3PO_PAIR_MANIFEST.json" \
  "${PAIRS}/train.jsonl" \
  "${PAIRS}/validation.jsonl"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ "$(basename -- "${BASE}")" == "step-696" ]] || {
  log "D3PO reference must be ctv_minimal_base step-696: ${BASE}"
  exit 2
}

assert_sha256() {
  local expected="$1"
  local path="$2"
  local observed
  observed="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${observed}" == "${expected}" ]] || {
    log "Frozen input hash mismatch: ${path}"
    exit 2
  }
}

assert_sha256 "${BASE_CONFIG_SHA}" "${BASE}/adapter_config.json"
assert_sha256 "${BASE_MODEL_SHA}" "${BASE}/adapter_model.safetensors"
assert_sha256 "${PAIR_MANIFEST_SHA}" "${PAIRS}/D3PO_PAIR_MANIFEST.json"
assert_sha256 "${TRAIN_SHA}" "${PAIRS}/train.jsonl"
assert_sha256 "${VALIDATION_SHA}" "${PAIRS}/validation.jsonl"
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || {
  log "D3PO-256-Min requires exactly 8 CPUs"
  exit 2
}

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	shared_noise_soft_d3po_256_min
initialization	${BASE}
policy_reference	one_backbone_two_named_copies
training_seeds	81017,81018 sequential
optimizer_updates_per_seed	348
microbatch_pairs	1
sequences_per_pair	2
gradient_accumulation	16
learning_rate	5e-6
beta	0.1
lora_rank	8
lora_alpha	32
lora_dropout	0
winner_anchor_weight	0.2
representation	dynamic exact 7+4N
masked_fields	lattice,angles,XYZ only
checkpoint_selection	none; step348 only
gpu	1x A800
cpus	8
EOF

sha256sum \
  "${BASE}/adapter_config.json" \
  "${BASE}/adapter_model.safetensors" \
  "${PAIRS}/D3PO_PAIR_MANIFEST.json" \
  "${PAIRS}/train.jsonl" \
  "${PAIRS}/validation.jsonl" \
  >"${RUN_DIR}/inputs.sha256"

# Mandatory independent seeds run serially.  There is never a second process,
# backbone, GPU job, checkpoint candidate, or adaptive retry in this wrapper.
for seed in 81017 81018; do
  readonly_seed_dir="${RUN_DIR}/seed${seed}"
  [[ ! -e "${readonly_seed_dir}" ]] || {
    log "Refusing to overwrite seed output: ${readonly_seed_dir}"
    exit 2
  }
  log "Starting frozen D3PO seed ${seed}"
  srun --ntasks=1 --cpus-per-task=8 \
    python "${ROOT}/src/scripts/llada_d3po.py" \
      --model-path "${MODEL}" \
      --checkpoint-path "${BASE}" \
      --data-dir "${PAIRS}" \
      --output-dir "${readonly_seed_dir}" \
      --seed "${seed}"
  [[ -f "${readonly_seed_dir}/_SUCCESS" ]] || missing "${readonly_seed_dir}/_SUCCESS"
  [[ -f "${readonly_seed_dir}/D3PO_TRAIN_MANIFEST.json" ]] || \
    missing "${readonly_seed_dir}/D3PO_TRAIN_MANIFEST.json"
  adapter_count="$(find "${readonly_seed_dir}/checkpoints/step-348" -type f -name adapter_model.safetensors | wc -l)"
  [[ "${adapter_count}" == "1" ]] || {
    log "Seed ${seed} did not save exactly one step348 policy adapter"
    exit 2
  }
  log "Completed frozen D3PO seed ${seed}"
done

python - "${RUN_DIR}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
seeds = {}
for seed in (81017, 81018):
    path = root / f"seed{seed}" / "D3PO_TRAIN_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success" or payload.get("optimizer_updates") != 348:
        raise SystemExit(f"seed{seed} manifest is not a complete step348 terminal")
    seeds[str(seed)] = {
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint": payload.get("checkpoint"),
    }
summary = {
    "schema": "h1a2_shared_noise_soft_d3po_two_seed_run_v1",
    "status": "success",
    "sequential": True,
    "gpu_count": 1,
    "cpus": 8,
    "seeds": seeds,
    "checkpoint_selection": False,
}
(root / "D3PO_TWO_SEED_MANIFEST.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

touch "${RUN_DIR}/_SUCCESS"
log "Both mandatory D3PO seeds completed serially: ${RUN_DIR}"
