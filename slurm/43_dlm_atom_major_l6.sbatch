#!/usr/bin/env bash
#SBATCH --job-name=h1a2-atom-major
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=h1a2-atom-major-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLANS="${H1A2_L6_PLANS:?export H1A2_L6_PLANS}"
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_PUBLIC_DLM_CHECKPOINT:?export H1A2_PUBLIC_DLM_CHECKPOINT}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly RUN_DIR="${ROOT}/runs/dlm_atom_major_l6_${SLURM_JOB_ID}"
for path in "${PLANS}" "${MODEL}" "${CHECKPOINT}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]] || missing "${CHECKPOINT}/adapter_model.safetensors"
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
trigger	joint-coordinate body gate failed before stability evaluation
plans	${PLANS}
attempts_per_cell	256
checkpoint	${CHECKPOINT}
conditioning	full_plan_state,hard_anchor_only
schedule	atom-major X-Y-Z tuples
axis_controls	reused from condition-schedule L6
seeds	17,18
dlm_sampling_seeds	17117,18117
refiner_seeds	27117,28117
temperature	0.7
refiner_steps	800
stability_condition	none
rl	false
rerank	false
EOF

run_cell() {
  local gpu="$1" seed_label="$2" arm="$3" prompt_style="$4" dlm_seed="$5" refiner_seed="$6"
  local output="${RUN_DIR}/seed${seed_label}/${arm}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" --checkpoint-path "${CHECKPOINT}" \
    --output-dir "${output}/body" --crysllmgen-dir "${CRYS}" \
    --prompt-jsonl "${PLANS}" --num-samples 256 --batch-size 1 \
    --preserve-prompt-sample-idx --body-prompt-style "${prompt_style}" \
    --seed "${dlm_seed}" --seed-by-sample-index --temperature 0.7 \
    --cfg-scale 0 --generation-schedule atom-major
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${refiner_seed}" --seed-by-sample-index \
    --diff-steps 800 --max-proposals 256
  touch "${output}/_SUCCESS"
}

run_cell 0 17 full_atom full_plan_state 17117 27117 & p0=$!
run_cell 1 17 hard_atom hard_anchor_only 17117 27117 & p1=$!
run_cell 2 18 full_atom full_plan_state 18117 28117 & p2=$!
run_cell 3 18 hard_atom hard_anchor_only 18117 28117 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

for seed in 17 18; do
  for arm in full_atom hard_atom; do
    [[ -f "${RUN_DIR}/seed${seed}/${arm}/_SUCCESS" ]] || missing "seed${seed}/${arm}"
  done
done
touch "${RUN_DIR}/_SUCCESS"
log "Atom-major L6 generation completed: ${RUN_DIR}"
