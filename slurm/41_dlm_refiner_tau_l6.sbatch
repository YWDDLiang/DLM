#!/usr/bin/env bash
#SBATCH --job-name=h1a2-refiner-tau
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=h1a2-refiner-tau-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY_RUN="${H1A2_CONDITION_SCHEDULE_RUN:?export H1A2_CONDITION_SCHEDULE_RUN}"
readonly SOURCE_ARM="${H1A2_SELECTED_CONDITION_ARM:?export H1A2_SELECTED_CONDITION_ARM}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
source "${ROOT}/slurm/_common.sh"
activate_environment

case "${SOURCE_ARM}" in full_axis|hard_joint) ;; *) missing "unsupported source arm ${SOURCE_ARM}" ;; esac
readonly RUN_DIR="${ROOT}/runs/dlm_refiner_tau_l6_${SLURM_JOB_ID}"
for path in "${BODY_RUN}/_SUCCESS" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
body_run	${BODY_RUN}
source_arm	${SOURCE_ARM}
seeds	17,18
new_timesteps	200,500
reused_timesteps	0,800
refiner	${REFINER}
refiner_seeds	27117,28117
sample_idx_pairing	true
checkpoint_training	false
rl	false
rerank	false
EOF

run_cell() {
  local gpu="$1" seed="$2" tau="$3" refiner_seed="$4"
  local body="${BODY_RUN}/seed${seed}/${SOURCE_ARM}/body"
  local output="${RUN_DIR}/seed${seed}/tau${tau}/refine"
  [[ -f "${body}/proposal_graphs.pt" ]] || missing "${body}/proposal_graphs.pt"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${body}/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${output}" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${refiner_seed}" --seed-by-sample-index \
    --diff-steps "${tau}" --max-proposals 256
  touch "${RUN_DIR}/seed${seed}/tau${tau}/_SUCCESS"
}

run_cell 0 17 200 27117 & p0=$!
run_cell 1 17 500 27117 & p1=$!
run_cell 2 18 200 28117 & p2=$!
run_cell 3 18 500 28117 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

for seed in 17 18; do
  for tau in 200 500; do
    [[ -f "${RUN_DIR}/seed${seed}/tau${tau}/_SUCCESS" ]] || missing "seed${seed}/tau${tau}"
  done
done
touch "${RUN_DIR}/_SUCCESS"
log "Refiner timestep L6 cells completed: ${RUN_DIR}"
