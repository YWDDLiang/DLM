#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sgtc-l7
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:3
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-sgtc-l7-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE="${H1A2_SGTC_BASE_CHECKPOINT:?export H1A2_SGTC_BASE_CHECKPOINT}"
readonly G0="${H1A2_SGTC_G0_CHECKPOINT:?export H1A2_SGTC_G0_CHECKPOINT}"
readonly G1="${H1A2_SGTC_G1_CHECKPOINT:?export H1A2_SGTC_G1_CHECKPOINT}"
readonly PLANS="${H1A2_SGTC_L7_PLANS:?export H1A2_SGTC_L7_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/sgtc_l7_${SLURM_JOB_ID}"
for path in "${MODEL}" "${BASE}" "${G0}" "${G1}" "${PLANS}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for checkpoint in "${BASE}" "${G0}" "${G1}"; do
  [[ -f "${checkpoint}/adapter_model.safetensors" ]] || missing "${checkpoint}/adapter_model.safetensors"
done
[[ "$(wc -l <"${PLANS}")" -eq 1000 ]] || missing "frozen requested-1000 L7 Plans"
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field\tvalue
method\tSGTC-DLM-v1 matched L7
plans\t${PLANS}
plans_sha256\t$(sha256sum "${PLANS}" | cut -d' ' -f1)
attempts_per_cell\t1000
arms\tbase,g0_all,g1_strict
seed\t18
dlm_seed\t92117
refiner_seed\t102117
temperature\t0.7
schedule\texact_axis
refiner_tau\t800
rerank_or_replacement\tfalse
EOF

run_cell() {
  local gpu="$1" arm="$2" checkpoint="$3"
  local output="${RUN_DIR}/seed18/${arm}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" \
    --checkpoint-path "${checkpoint}" \
    --prompt-jsonl "${PLANS}" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --num-samples 1000 \
    --temperature 0.7 \
    --seed 92117
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" \
    --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" \
    --batch-size 1 \
    --seed 102117 \
    --seed-by-sample-index \
    --diff-steps 800 \
    --max-proposals 1000
  touch "${output}/_SUCCESS"
}

run_cell 0 base "${BASE}" & p0=$!
run_cell 1 g0_all "${G0}" & p1=$!
run_cell 2 g1_strict "${G1}" & p2=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"

for arm in base g0_all g1_strict; do
  [[ -f "${RUN_DIR}/seed18/${arm}/_SUCCESS" ]] || missing "${RUN_DIR}/seed18/${arm}/_SUCCESS"
done
touch "${RUN_DIR}/_SUCCESS"
log "SGTC matched L7 generation finished: ${RUN_DIR}"
