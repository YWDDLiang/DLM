#!/usr/bin/env bash
#SBATCH --job-name=rollout-matched-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=02:00:00
#SBATCH --output=rollout-matched-train-%j.out
#SBATCH --error=rollout-matched-train-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${ROLLOUT_TRAIN_BASE_POLICY:?}"
readonly DATA="${ROLLOUT_TRAIN_DATA:?}"
readonly MASTER_PORT="${ROLLOUT_TRAIN_MASTER_PORT:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${CODE}/src/scripts/llada_sft.py"
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/rollout_matched_pilot_train_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'set +e; touch "${RUN}/_FAILED"; exit 1' ERR
[[ -f "${DATA}/_SUCCESS" ]]
[[ "$(wc -l < "${DATA}/train.jsonl")" -eq 384 ]]
[[ "$(wc -l < "${DATA}/val.jsonl")" -eq 128 ]]
[[ -f "${BASE}/adapter_model.safetensors" ]]
[[ -f "${BASE}/periodic_relation_adapter.pt" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	rollout_matched_active_group_training_v1
initialization	${BASE}
train_transitions	384
validation_transitions	128
training_seed	81017
updates	128
world_size	2
effective_batch	16
learning_rate	1e-6_cosine_warmup10
trainable	LoRA_plus_periodic_relation
model_mask	current_and_future_real_rollout_masks
CE_supervision	current_active_group_only
schedule	exact_axis_unchanged
checkpoint_selection	none_step128_only
planner_or_inference_change	false
EOF

"${TORCHRUN}" --standalone --master-port "${MASTER_PORT}" --nproc_per_node=2 \
  "${TRAINER}" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --periodic-relation-checkpoint "${BASE}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --require-paired-source \
  --require-rollout-masks \
  --skip-data-vocab-resize \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 20 \
  --max-train-steps 128 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 1e-6 \
  --lr-scheduler cosine \
  --warmup-steps 10 \
  --min-lr-ratio 0.1 \
  --save-steps 128 \
  --skip-final-alias \
  --eval-steps 128 \
  --eval-max-batches 64 \
  --position-diagnostics-steps 0 \
  --logging-steps 10 \
  --dataloader-num-workers 4 \
  --fail-on-truncation \
  --seed 81017 \
  --data-seed 20260903 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
  --counterfactual-grounding-weight 0.0 \
  --periodic-metric-weight 0.0 \
  --periodic-pair-rdf-weight 0.0 \
  --periodic-overlap-weight 0.0 \
  --periodic-coordination-weight 0.0 \
  --periodic-exact-triclinic-pbc \
  --periodic-image-radius 2 \
  --periodic-species-margin-scale 0.55 \
  --periodic-species-margin-floor 0.60 \
  --periodic-species-margin-ceiling 1.40 \
  --periodic-overlap-tail-temperature 0.10 \
  --periodic-overlap-tail-mix 0.0 \
  --periodic-relation-rank 64 \
  --periodic-relation-circular-min-resultant 0.25 \
  --periodic-relation-uncertainty-floor 0.25 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-128"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ -f "${CHECKPOINT}/periodic_relation_adapter.pt" ]]
[[ -f "${CHECKPOINT}/periodic_relation_config.json" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoints < <(
  find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'step-*' -printf '%f\n' | sort
)
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-128 ]]

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
config = json.load((root / "run_config.json").open())
assert config["require_paired_source"] is True
assert config["require_rollout_masks"] is True
assert config["max_train_steps"] == 128
events = [json.loads(line) for line in (root / "training_log.jsonl").open() if line.strip()]
start = next(row for row in events if row.get("event") == "start")
assert start["paired_source"]["train_rows"] == 384
assert start["rollout_masks"]["train_rows"] == 384
assert start["rollout_masks"]["validation_rows"] == 128
train = [row for row in events if row.get("event") == "train"]
assert train
assert any(row.get("event") == "eval" and int(row["step"]) == 128 for row in events)
for row in train:
    for key in (
        "loss", "task_loss", "grad_norm", "periodic_relation_grad_norm",
        "periodic_relation_output_grad_norm",
    ):
        assert math.isfinite(float(row[key])), (key, row[key])
assert max(float(row["periodic_relation_grad_norm"]) for row in train) > 0.0
PY

printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
