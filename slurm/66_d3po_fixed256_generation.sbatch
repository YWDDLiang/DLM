#!/usr/bin/env bash
#SBATCH --job-name=h1a2-d3po-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-d3po-gen-%j.out

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE="${H1A2_D3PO_BASE_CHECKPOINT:?export H1A2_D3PO_BASE_CHECKPOINT}"
readonly SEED_A="${H1A2_D3PO_SEED81017_CHECKPOINT:?export H1A2_D3PO_SEED81017_CHECKPOINT}"
readonly SEED_B="${H1A2_D3PO_SEED81018_CHECKPOINT:?export H1A2_D3PO_SEED81018_CHECKPOINT}"
readonly TRAIN_RUN="${H1A2_D3PO_TRAIN_RUN:?export H1A2_D3PO_TRAIN_RUN}"
readonly PLANS="${H1A2_D3PO_TEST_PLANS:?export H1A2_D3PO_TEST_PLANS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/d3po_fixed256_generation_${SLURM_JOB_ID}"
readonly TEST_PLANS_SHA="21a20c8eca10c30953f486ee00301a872e3c32b853bb0acbe187be2f9d94d3f5"

for path in \
  "${MODEL}" "${BASE}" "${SEED_A}" "${SEED_B}" \
  "${TRAIN_RUN}/_SUCCESS" "${PLANS}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
for checkpoint in "${BASE}" "${SEED_A}" "${SEED_B}"; do
  [[ -f "${checkpoint}/adapter_model.safetensors" ]] || \
    missing "${checkpoint}/adapter_model.safetensors"
  [[ -f "${checkpoint}/adapter_config.json" ]] || \
    missing "${checkpoint}/adapter_config.json"
done
[[ "$(wc -l <"${PLANS}")" -eq 256 ]] || missing "frozen D3PO test256 Plans"
[[ "$(sha256sum "${PLANS}" | awk '{print $1}')" == "${TEST_PLANS_SHA}" ]] || {
  log "Frozen D3PO certified test Plan hash mismatch"
  exit 2
}
mkdir -p "${RUN_DIR}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" \
    >"${RUN_DIR}/ENGINEERING_FAILURE.tsv"
  touch "${RUN_DIR}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
method	shared-noise masked-D3PO fixed256 evaluation
train_run	${TRAIN_RUN}
plans	${PLANS}
plans_sha256	$(sha256sum "${PLANS}" | awk '{print $1}')
attempts_per_cell	256
arms	base,d3po_seed81017,d3po_seed81018
streams	17,18
dlm_seeds	91117,92117
refiner_seeds	101117,102117
temperature	0.7
schedule	exact_axis
refiner_tau	800
rerank_or_replacement	false
checkpoint_or_seed_selection	false
EOF
sha256sum \
  "${BASE}/adapter_config.json" "${BASE}/adapter_model.safetensors" \
  "${SEED_A}/adapter_config.json" "${SEED_A}/adapter_model.safetensors" \
  "${SEED_B}/adapter_config.json" "${SEED_B}/adapter_model.safetensors" \
  "${PLANS}" >"${RUN_DIR}/inputs.sha256"

run_cell() {
  local gpu="$1" stream="$2" arm="$3" checkpoint="$4" dlm_seed="$5" refiner_seed="$6"
  local output="${RUN_DIR}/stream${stream}/${arm}"
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
run_cell 1 17 d3po_seed81017 "${SEED_A}" 91117 101117 & p1=$!
run_cell 2 17 d3po_seed81018 "${SEED_B}" 91117 101117 & p2=$!
run_cell 3 18 base "${BASE}" 92117 102117 & p3=$!
run_cell 4 18 d3po_seed81017 "${SEED_A}" 92117 102117 & p4=$!
run_cell 5 18 d3po_seed81018 "${SEED_B}" 92117 102117 & p5=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"
wait "${p3}"; wait "${p4}"; wait "${p5}"

for stream in 17 18; do
  for arm in base d3po_seed81017 d3po_seed81018; do
    [[ -f "${RUN_DIR}/stream${stream}/${arm}/_SUCCESS" ]] || \
      missing "${RUN_DIR}/stream${stream}/${arm}/_SUCCESS"
  done
done
find "${RUN_DIR}" -type f ! -name 'OUTPUTS.sha256' -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN_DIR}/OUTPUTS.sha256"
trap - ERR
touch "${RUN_DIR}/_SUCCESS"
log "D3PO fixed256 six-cell generation/refinement finished: ${RUN_DIR}"
