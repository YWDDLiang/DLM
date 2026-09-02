#!/usr/bin/env bash
#SBATCH --job-name=force-microstudent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=02:00:00
#SBATCH --output=force-microstudent-%j.out
#SBATCH --error=force-microstudent-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${FORCE_MICRO_BASE_POLICY:?}"
readonly MASTER_PORT="${FORCE_MICRO_MASTER_PORT:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/projected_force_microstudent_384_128_v1_20260902"
readonly TRAINER="${CODE}/src/scripts/llada_sft.py"
readonly RUN="${ROOT}/runs/projected_force_microstudent_${SLURM_JOB_ID}"
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

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

[[ -f "${DATA}/_SUCCESS" && -f "${DATA}/train.jsonl" && -f "${DATA}/val.jsonl" ]]
[[ "$(wc -l < "${DATA}/train.jsonl")" -eq 384 ]]
[[ "$(wc -l < "${DATA}/val.jsonl")" -eq 128 ]]
[[ -f "${BASE}/adapter_model.safetensors" ]]
[[ -f "${BASE}/periodic_relation_adapter.pt" ]]
[[ -f "${BASE}/periodic_relation_config.json" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	projected_force_microstudent_v1
initialization	${BASE}
train_rows	384
holdout_rows	128
base_split	48_train_16_holdout_no_overlap
training_seed	81017
optimizer_steps	128
world_size	2
effective_batch	16
learning_rate	1e-6_cosine_warmup10
trainable	periodic_relation_residual_only
paired_transition	unmasked_source_masked_target
hard_validity	project0.55A_accept0.50A
species_prior	soft_clamp_0.55_radius_sum_0.60A_1.40A
loss_weights	CE_plus_metric0.1_pairRDF0.1_overlap0.2_coordination0.05
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
  --data-seed 20260902 \
  --no-lora \
  --counterfactual-grounding-weight 0.0 \
  --periodic-metric-weight 0.1 \
  --periodic-pair-rdf-weight 0.1 \
  --periodic-overlap-weight 0.2 \
  --periodic-coordination-weight 0.05 \
  --periodic-exact-triclinic-pbc \
  --periodic-image-radius 2 \
  --periodic-species-margin-scale 0.55 \
  --periodic-species-margin-floor 0.60 \
  --periodic-species-margin-ceiling 1.40 \
  --periodic-overlap-tail-temperature 0.10 \
  --periodic-overlap-tail-mix 0.50 \
  --periodic-relation-rank 64 \
  --periodic-relation-only \
  --periodic-relation-circular-min-resultant 0.25 \
  --periodic-relation-uncertainty-floor 0.25 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-128"
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
assert config["periodic_relation_only"] is True
assert config["max_train_steps"] == 128
events = [json.loads(line) for line in (root / "training_log.jsonl").open() if line.strip()]
start = next(row for row in events if row.get("event") == "start")
assert start["paired_source"]["train_rows"] == 384
assert start["paired_source"]["validation_rows"] == 128
train = [row for row in events if row.get("event") == "train"]
assert train
assert any(row.get("event") == "eval" and int(row["step"]) == 128 for row in events)
for row in train:
    for key in (
        "loss", "task_loss", "grad_norm", "periodic_relation_grad_norm",
        "periodic_relation_output_grad_norm", "periodic_geometry_metric_loss",
        "periodic_geometry_pair_rdf_loss", "periodic_geometry_overlap_loss",
        "periodic_geometry_coordination_loss",
    ):
        assert math.isfinite(float(row[key])), (key, row[key])
assert max(float(row["periodic_relation_grad_norm"]) for row in train) > 0.0
PY

printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
