#!/usr/bin/env bash
#SBATCH --job-name=spad-closure-ce
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=spad-closure-ce-%j.out
#SBATCH --error=spad-closure-ce-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?export SSCD_SOURCE_ROOT to the deployed closure branch}"
readonly DATA="${ROOT}/data/spad_basin_closure_sft_v1_20260904"
readonly BASE="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${SOURCE}/src/scripts/llada_sft.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_ce_${SLURM_JOB_ID}"
readonly SEED=88017

mkdir "${RUN}"
mkdir "${RUN}/logs"
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
  "${BASE}/adapter_model.safetensors" "${TRAINER}"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${DATA}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l <"${DATA}/val.jsonl")" -eq 9047 ]]

"${PYTHON}" - "${DATA}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["schema"] == "spad_basin_closure_sft_manifest_v1"
assert manifest["schedule"] == "cell_then_reverse_llama_species_blocks_v1"
assert manifest["splits"]["train"]["rows"] == 27136
assert manifest["splits"]["val"]["rows"] == 9047
assert manifest["exact_N_elements_never_masked"] is True
assert manifest["outcomes_read"] is False
PY

IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 4 ]]
readonly DEVICES="${allocated[0]},${allocated[1]},${allocated[2]},${allocated[3]}"

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	spad_basin_closure_ce_train_v1
initialization	${BASE}
teacher	full_MP20_clean_teacher
schedule	cell_then_reverse_llama_species_blocks_v1
train_rows	27136
validation_rows	9047
updates	1696
epochs	1
world_size	4
effective_batch	16
learning_rate	5e-6_cosine
seed	${SEED}
lora	r8_alpha32_dropout0.05_qkv_ff_up
paired_source	required
rollout_masks	required
eligible_checkpoint	step-1696_only
checkpoint_selection	none
EOF

CUDA_VISIBLE_DEVICES="${DEVICES}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master-port 35917 \
  --nproc_per_node=4 "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --require-paired-source \
  --require-rollout-masks \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 1696 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 25 \
  --min-lr-ratio 0.1 \
  --save-steps 1696 \
  --skip-final-alias \
  --eval-steps 1696 \
  --eval-max-batches 50 \
  --logging-steps 10 \
  --dataloader-num-workers 2 \
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
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-1696"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoints < <(
  find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'step-*' -printf '%f\n' | sort
)
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-1696 ]]

"${PYTHON}" - "${OUTPUT}" "${BASE}" "${CHECKPOINT}" "${DATA}" "${SEED}" <<'PY'
import json
import hashlib
import math
import pathlib
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output = pathlib.Path(sys.argv[1])
base = str(pathlib.Path(sys.argv[2]).resolve(strict=True))
checkpoint = pathlib.Path(sys.argv[3]).resolve(strict=True)
data = pathlib.Path(sys.argv[4])
seed = int(sys.argv[5])
config = json.load((output / "run_config.json").open(encoding="utf-8"))
assert str(pathlib.Path(config["checkpoint_path"]).resolve(strict=True)) == base
assert config["world_size"] == 4
assert config["epochs"] == 1 and config["max_train_steps"] == 1696
assert config["batch_size"] == 1 and config["grad_accum"] == 4
assert config["lr"] == 5e-6 and config["seed"] == seed
assert config["require_paired_source"] is True
assert config["require_rollout_masks"] is True
events = [
    json.loads(line)
    for line in (output / "training_log.jsonl").open(encoding="utf-8")
    if line.strip()
]
start = next(row for row in events if row.get("event") == "start")
assert start["paired_source"]["train_rows"] == 27136
assert start["paired_source"]["validation_rows"] == 9047
assert start["rollout_masks"]["train_rows"] == 27136
assert start["rollout_masks"]["validation_rows"] == 9047
train = [row for row in events if row.get("event") == "train"]
assert train and int(train[-1]["step"]) >= 1690
assert any(row.get("event") == "eval" and int(row["step"]) == 1696 for row in events)
assert all(
    all(math.isfinite(float(row[key])) for key in ("loss", "task_loss", "grad_norm", "lr"))
    for row in train
)
adapter = json.load((checkpoint / "adapter_config.json").open(encoding="utf-8"))
assert int(adapter["r"]) == 8 and int(adapter["lora_alpha"]) == 32
assert float(adapter["lora_dropout"]) == 0.05
assert set(adapter["target_modules"]) == {
    "q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"
}
manifest = json.load((data / "manifest.json").open(encoding="utf-8"))
assert set(manifest["splits"]["train"]["mask_classes"]) == {
    "cell_sequential_component", "reverse_species_block_component"
}
adapter_model = checkpoint / "adapter_model.safetensors"
assert adapter_model.is_file()
adapter_model_sha256 = sha256_file(adapter_model)
capability = {
    "schema": "spad_basin_closure_capability_v1",
    "checkpoint_path": str(checkpoint),
    "initialization_checkpoint_path": base,
    "spad_cell_closure_trained": True,
    "spad_species_block_closure_trained": True,
    "closure_schedule_version": "cell_then_reverse_llama_species_blocks_v1",
    "train_rows": 27136,
    "validation_rows": 9047,
    "optimizer_updates": 1696,
    "training_seed": seed,
    "checkpoint_selection": "none_step1696_only",
    "adapter_model_sha256": adapter_model_sha256,
}
(checkpoint / "spad_basin_closure_capability.json").write_text(
    json.dumps(capability, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"${PYTHON}" - "${CHECKPOINT}/spad_basin_closure_capability.json" "${CHECKPOINT}" <<'PY'
import json
import hashlib
import pathlib
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


path = pathlib.Path(sys.argv[1])
checkpoint = pathlib.Path(sys.argv[2]).resolve(strict=True)
capability = json.load(path.open(encoding="utf-8"))
assert pathlib.Path(capability["checkpoint_path"]) == checkpoint
assert capability["spad_cell_closure_trained"] is True
assert capability["spad_species_block_closure_trained"] is True
assert capability["closure_schedule_version"] == "cell_then_reverse_llama_species_blocks_v1"
assert capability["schema"] == "spad_basin_closure_capability_v1"
adapter = checkpoint / "adapter_model.safetensors"
assert adapter.is_file()
assert sha256_file(adapter) == capability["adapter_model_sha256"]
PY

printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
printf '%s\n' "${CHECKPOINT}/spad_basin_closure_capability.json" >"${RUN}/CAPABILITY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
