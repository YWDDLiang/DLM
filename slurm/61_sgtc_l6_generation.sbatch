#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sgtc-l6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-sgtc-l6-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE="${H1A2_SGTC_BASE_CHECKPOINT:?export H1A2_SGTC_BASE_CHECKPOINT}"
readonly G0="${H1A2_SGTC_G0_CHECKPOINT:?export H1A2_SGTC_G0_CHECKPOINT}"
readonly G1="${H1A2_SGTC_G1_CHECKPOINT:?export H1A2_SGTC_G1_CHECKPOINT}"
readonly PLANS="${H1A2_SGTC_L6_PLANS:?export H1A2_SGTC_L6_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/sgtc_l6_${SLURM_JOB_ID}"
for path in "${MODEL}" "${BASE}" "${G0}" "${G1}" "${PLANS}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for checkpoint in "${BASE}" "${G0}" "${G1}"; do
  [[ -f "${checkpoint}/adapter_model.safetensors" ]] || missing "${checkpoint}/adapter_model.safetensors"
done
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
method	SGTC-DLM-v1 matched L6
plans	${PLANS}
attempts_per_cell	256
arms	base,g0_all,g1_strict
seeds	17,18
dlm_seeds	91117,92117
refiner_seeds	101117,102117
temperature	0.7
schedule	exact_axis
refiner_tau	800
rerank_or_replacement	false
EOF

run_cell() {
  local gpu="$1" seed_label="$2" arm="$3" checkpoint="$4" dlm_seed="$5" refiner_seed="$6"
  local output="${RUN_DIR}/seed${seed_label}/${arm}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --prompt-jsonl "${PLANS}" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --num-samples 256 \
    --temperature 0.7 \
    --seed "${dlm_seed}"
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

run_cell 0 17 base "${BASE}" 91117 101117 & p0=$!
run_cell 1 17 g0_all "${G0}" 91117 101117 & p1=$!
run_cell 2 17 g1_strict "${G1}" 91117 101117 & p2=$!
run_cell 3 18 base "${BASE}" 92117 102117 & p3=$!
run_cell 4 18 g0_all "${G0}" 92117 102117 & p4=$!
run_cell 5 18 g1_strict "${G1}" 92117 102117 & p5=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"; wait "${p4}"; wait "${p5}"

for seed in 17 18; do
  for arm in base g0_all g1_strict; do
    [[ -f "${RUN_DIR}/seed${seed}/${arm}/_SUCCESS" ]] || missing "${RUN_DIR}/seed${seed}/${arm}/_SUCCESS"
  done
done
touch "${RUN_DIR}/_SUCCESS"
log "SGTC matched L6 generation finished: ${RUN_DIR}"
