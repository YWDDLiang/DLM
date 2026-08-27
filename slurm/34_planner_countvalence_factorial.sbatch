#!/usr/bin/env bash
#SBATCH --job-name=h1a2-plan-cv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-plan-cv-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLANNER_ROOT="${H1A2_PLANNER_ROOT:?export H1A2_PLANNER_ROOT}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_PLANNER_MODEL_PATH:?export H1A2_PLANNER_MODEL_PATH}"
readonly P0="${H1A2_PLANNER_P0_CHECKPOINT:?export H1A2_PLANNER_P0_CHECKPOINT}"
readonly COUNTFIELDS_DATA="${H1A2_COUNTFIELDS_DATA:?export H1A2_COUNTFIELDS_DATA}"
readonly COUNTVALENCE_DATA="${H1A2_COUNTVALENCE_DATA:?export H1A2_COUNTVALENCE_DATA}"
readonly AUDIT="${H1A2_COUNTVALENCE_AUDIT:?export H1A2_COUNTVALENCE_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/planner_countvalence_factorial_${SLURM_JOB_ID}"

for path in "${MODEL}" "${P0}" "${COUNTFIELDS_DATA}" "${COUNTVALENCE_DATA}" "${AUDIT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${AUDIT}/_SUCCESS" ]] || missing "${AUDIT}/_SUCCESS"
python - "${AUDIT}/PLANNER_COUNTVALENCE_AUDIT.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("gate", {}).get("candidate_training_authorized") is not True:
    raise SystemExit("count-valence coverage gate did not authorize training")
PY
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
planner_initialization	${P0}
model	${MODEL}
control_representation	countfields_without_valence
candidate_representation	countvalence_crysvcd_plus_versioned_extension
same_teacher_rows	true
same_optimizer_contract	true
train_epochs	1.0
batch_size	1
gradient_accumulation	8
learning_rate	2e-6
planner_seeds	17,18
sampling_seeds	17017,18018
samples_per_cell	256
temperature	0.9
top_p	0.95
top_k	50
public_output	h1_rich_plan_v1
rl	false
rerank	false
repair	false
p0_control_seed17	${PLANNER_ROOT}/runs/difficulty_planner_meta_guard_v4_34822/plans_control_seed17
p0_control_seed18	${PLANNER_ROOT}/runs/difficulty_planner_meta_guard_v4_34822/plans_control_seed18
EOF

run_cell() {
  local gpu="$1" representation="$2" planner_seed="$3" sampling_seed="$4" data="$5" output="$6"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/llama_formula_sft.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${P0}" \
    --data-dir "${data}" \
    --output-dir "${output}/train" \
    --max-length 768 \
    --epochs 1.0 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-6 \
    --weight-decay 0.0 \
    --warmup-steps 100 \
    --logging-steps 20 \
    --eval-steps 500 \
    --eval-max-batches 50 \
    --seed "${planner_seed}" \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --gradient-checkpointing

  [[ -f "${output}/train/final/adapter_model.safetensors" ]] || \
    missing "${output}/train/final/adapter_model.safetensors"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_llama_count_plans.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${output}/train/final" \
    --output-dir "${output}/plans" \
    --representation "${representation}" \
    --num-samples 256 \
    --batch-size 4 \
    --max-new-tokens 192 \
    --temperature 0.9 \
    --top-p 0.95 \
    --top-k 50 \
    --seed "${sampling_seed}" \
    --max-atoms 20
  [[ -f "${output}/plans/_SUCCESS" ]] || missing "${output}/plans/_SUCCESS"
  touch "${output}/_SUCCESS"
}

run_cell 0 countfields 17 17017 "${COUNTFIELDS_DATA}" "${RUN_DIR}/countfields_seed17" &
pid0=$!
run_cell 1 countvalence 17 17017 "${COUNTVALENCE_DATA}" "${RUN_DIR}/countvalence_seed17" &
pid1=$!
run_cell 2 countfields 18 18018 "${COUNTFIELDS_DATA}" "${RUN_DIR}/countfields_seed18" &
pid2=$!
run_cell 3 countvalence 18 18018 "${COUNTVALENCE_DATA}" "${RUN_DIR}/countvalence_seed18" &
pid3=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

for cell in countfields_seed17 countvalence_seed17 countfields_seed18 countvalence_seed18; do
  [[ -f "${RUN_DIR}/${cell}/_SUCCESS" ]] || missing "${RUN_DIR}/${cell}/_SUCCESS"
done
touch "${RUN_DIR}/_SUCCESS"
log "Planner count-valence factorial completed: ${RUN_DIR}"
