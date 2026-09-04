#!/usr/bin/env bash
#SBATCH --job-name=spad-closure-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=spad-closure-canary-%j.out
#SBATCH --error=spad-closure-canary-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT to the deployed closure branch}"
readonly DATA="${ROOT}/data/spad_basin_closure_sft_v1_20260904"
readonly BASE="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${SOURCE}/src/scripts/llada_sft.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_ce_canary_${SLURM_JOB_ID}"
readonly CANARY_DATA="${RUN}/canary_data"
readonly SEED=88017

mkdir "${RUN}"
mkdir "${RUN}/logs"
mkdir "${CANARY_DATA}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${DATA}/_SUCCESS" "${DATA}/train.jsonl" \
  "${DATA}/val.jsonl" "${DATA}/vocab_tokens.txt" \
  "${DATA}/manifest.json" "${BASE}/adapter_model.safetensors" "${TRAINER}"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${DATA}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l <"${DATA}/val.jsonl")" -eq 9047 ]]

"${PYTHON}" - "${DATA}" "${CANARY_DATA}" <<'PY'
import json
import pathlib
import shutil
import sys

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
expected = {"cell_sequential_component", "reverse_species_block_component"}
categories = (
    ("cell_first", lambda row: row["spad_mask_class"] == "cell_sequential_component" and row["closure"]["cell_component_index"] == 0),
    ("cell_last", lambda row: row["spad_mask_class"] == "cell_sequential_component" and row["closure"]["cell_component_index"] == 5),
    ("coordinate_x", lambda row: row["spad_mask_class"] == "reverse_species_block_component" and row["closure"]["coordinate_component_index"] == 0),
    ("coordinate_y", lambda row: row["spad_mask_class"] == "reverse_species_block_component" and row["closure"]["coordinate_component_index"] == 1),
    ("coordinate_z", lambda row: row["spad_mask_class"] == "reverse_species_block_component" and row["closure"]["coordinate_component_index"] == 2),
    ("multi_site_block", lambda row: row["spad_mask_class"] == "reverse_species_block_component" and len(row["closure"]["block_slot_indices"]) > 1),
)
for split in ("train", "val"):
    selected = {}
    used_sources = set()
    with (source / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_idx = int(row["source_row_idx"])
            for category, predicate in categories:
                if category in selected or source_idx in used_sources:
                    continue
                if predicate(row):
                    payload = dict(row)
                    payload["canary_category"] = category
                    selected[category] = payload
                    used_sources.add(source_idx)
                    break
            if len(selected) == len(categories):
                break
    assert set(selected) == {name for name, _predicate in categories}
    with (output / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
        for category, _predicate in categories:
            handle.write(json.dumps(selected[category], sort_keys=True) + "\n")
shutil.copy2(source / "vocab_tokens.txt", output / "vocab_tokens.txt")
(output / "manifest.json").write_text(
    json.dumps(
        {
            "schema": "spad_basin_closure_canary_data_v1",
            "source_data": str(source),
            "rows_per_split": len(categories),
            "mask_classes": sorted(expected),
            "selected_categories": [name for name, _predicate in categories],
            "outcomes_read": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(output / "_SUCCESS").touch()
PY
[[ "$(wc -l <"${CANARY_DATA}/train.jsonl")" -eq 6 ]]
[[ "$(wc -l <"${CANARY_DATA}/val.jsonl")" -eq 6 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master-port 35918 \
  --nproc_per_node=1 "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --data-dir "${CANARY_DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --require-paired-source \
  --require-rollout-masks \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 6 \
  --batch-size 1 \
  --grad-accum 1 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 1 \
  --min-lr-ratio 0.1 \
  --save-steps 6 \
  --skip-final-alias \
  --eval-steps 6 \
  --eval-max-batches 6 \
  --logging-steps 1 \
  --dataloader-num-workers 1 \
  --fail-on-truncation \
  --seed "${SEED}" \
  --data-seed 20260904 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
  --counterfactual-grounding-weight 0.0 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-6"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ ! -e "${OUTPUT}/final" ]]

"${PYTHON}" - "${OUTPUT}" "${BASE}" "${CANARY_DATA}" "${RUN}" <<'PY'
import json
import math
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
base = pathlib.Path(sys.argv[2]).resolve(strict=True)
data = pathlib.Path(sys.argv[3])
run = pathlib.Path(sys.argv[4])
config = json.load((output / "run_config.json").open(encoding="utf-8"))
assert pathlib.Path(config["checkpoint_path"]).resolve(strict=True) == base
assert config["world_size"] == 1
assert config["max_train_steps"] == 6
assert config["require_paired_source"] is True
assert config["require_rollout_masks"] is True
events = [
    json.loads(line)
    for line in (output / "training_log.jsonl").open(encoding="utf-8")
    if line.strip()
]
start = next(row for row in events if row.get("event") == "start")
assert start["paired_source"]["train_rows"] == 6
assert start["paired_source"]["validation_rows"] == 6
assert start["rollout_masks"]["train_rows"] == 6
assert start["rollout_masks"]["validation_rows"] == 6
train = [row for row in events if row.get("event") == "train"]
assert [int(row["step"]) for row in train] == [1, 2, 3, 4, 5, 6]
assert all(
    all(math.isfinite(float(row[key])) for key in ("loss", "task_loss", "grad_norm", "lr"))
    for row in train
)
assert all(float(row["grad_norm"]) > 0.0 for row in train)
assert any(row.get("event") == "eval" and int(row["step"]) == 6 for row in events)
manifest = json.load((data / "manifest.json").open(encoding="utf-8"))
classes = {name: 1 for name in manifest["mask_classes"]}
expected = {"cell_sequential_component", "reverse_species_block_component"}
assert set(classes) == expected
assert all(int(classes[name]) > 0 for name in expected)
assert set(manifest["selected_categories"]) == {
    "cell_first", "cell_last", "coordinate_x", "coordinate_y",
    "coordinate_z", "multi_site_block",
}
report = {
    "schema": "spad_basin_closure_ce_canary_v1",
    "status": "success",
    "optimizer_updates": 6,
    "finite_loss_and_gradient": True,
    "paired_source_rows": 6,
    "rollout_mask_rows": 6,
    "mask_classes": classes,
    "checkpoint_publishable": False,
    "scientific_result": False,
}
(run / "CANARY_ONLY.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

[[ ! -e "${CHECKPOINT}/spad_basin_closure_capability.json" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
