#!/usr/bin/env bash
#SBATCH --job-name=h1a2-cond-sched
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:8
#SBATCH --mem=700G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-cond-sched-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLANS="${H1A2_L6_PLANS:?export H1A2_L6_PLANS}"
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_PUBLIC_DLM_CHECKPOINT:?export H1A2_PUBLIC_DLM_CHECKPOINT}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly RUN_DIR="${ROOT}/runs/dlm_condition_schedule_l6_${SLURM_JOB_ID}"
for path in "${PLANS}" "${MODEL}" "${CHECKPOINT}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]] || missing "${CHECKPOINT}/adapter_model.safetensors"
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
plans	${PLANS}
attempts_per_cell	256
checkpoint	${CHECKPOINT}
factor_conditioning	full_plan_state_vs_hard_anchor_only
factor_schedule	exact_axis_vs_joint_coordinates
primary_candidate	hard_joint
seeds	17,18
dlm_sampling_seeds	17117,18117
refiner_seeds	27117,28117
temperature	0.7
refiner	${REFINER}
refiner_steps	800
stability_condition	none
cfg_scale	0
rl	false
rerank	false
EOF

run_cell() {
  local gpu="$1" seed_label="$2" arm="$3" prompt_style="$4" schedule="$5" dlm_seed="$6" refiner_seed="$7"
  local output="${RUN_DIR}/seed${seed_label}/${arm}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${CHECKPOINT}" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" \
    --num-samples 256 \
    --batch-size 1 \
    --preserve-prompt-sample-idx \
    --body-prompt-style "${prompt_style}" \
    --seed "${dlm_seed}" \
    --seed-by-sample-index \
    --temperature 0.7 \
    --cfg-scale 0 \
    --generation-schedule "${schedule}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed "${refiner_seed}" \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 256
  touch "${output}/_SUCCESS"
}

run_cell 0 17 full_axis full_plan_state exact-plan 17117 27117 & p0=$!
run_cell 1 17 full_joint full_plan_state joint-coordinates 17117 27117 & p1=$!
run_cell 2 17 hard_axis hard_anchor_only exact-plan 17117 27117 & p2=$!
run_cell 3 17 hard_joint hard_anchor_only joint-coordinates 17117 27117 & p3=$!
run_cell 4 18 full_axis full_plan_state exact-plan 18117 28117 & p4=$!
run_cell 5 18 full_joint full_plan_state joint-coordinates 18117 28117 & p5=$!
run_cell 6 18 hard_axis hard_anchor_only exact-plan 18117 28117 & p6=$!
run_cell 7 18 hard_joint hard_anchor_only joint-coordinates 18117 28117 & p7=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
wait "${p4}"; wait "${p5}"; wait "${p6}"; wait "${p7}"

for seed in 17 18; do
  for arm in full_axis full_joint hard_axis hard_joint; do
    [[ -f "${RUN_DIR}/seed${seed}/${arm}/_SUCCESS" ]] || missing "${RUN_DIR}/seed${seed}/${arm}/_SUCCESS"
  done
done
touch "${RUN_DIR}/_SUCCESS"
log "Condition/schedule L6 generation completed: ${RUN_DIR}"
