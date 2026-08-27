#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ccfd-f1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=420G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-ccfd-f1-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL="${H1A2_PLANNER_MODEL_PATH:?export H1A2_PLANNER_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_PLANNER_CHECKPOINT:?export H1A2_PLANNER_CHECKPOINT}"
readonly NUM_SAMPLES="${H1A2_CCFD_NUM_SAMPLES:-1000}"
readonly RUN_KIND="${H1A2_CCFD_RUN_KIND:-full}"
source "${ROOT}/slurm/_common.sh"
activate_environment

for path in "${MODEL}" "${CHECKPOINT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
case "${RUN_KIND}" in pilot|full) ;; *) missing "unsupported run kind ${RUN_KIND}" ;; esac
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || missing "invalid sample count ${NUM_SAMPLES}"

readonly RUN_DIR="${ROOT}/runs/ccfd_phase1_${RUN_KIND}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
model	${MODEL}
checkpoint	${CHECKPOINT}
weights_changed	false
tokenizer_resized	false
prompt_style	h1_rich_plan_formula_prefill_v1
arms	f0_free,f1_ccfd
seeds	17,18
requested_per_cell	${NUM_SAMPLES}
temperature	0.9
top_p	0.95
top_k	50
max_atoms	20
repair	false
replacement	false
rerank	false
rl	false
EOF

run_cell() {
  local gpu="$1" arm="$2" seed="$3" mode="$4"
  local output="${RUN_DIR}/${arm}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/src/scripts/sample_llama_h1_formula_plans.py" \
    --model-path "${MODEL}" --checkpoint-path "${CHECKPOINT}" \
    --output-dir "${output}" --num-samples "${NUM_SAMPLES}" --batch-size 4 \
    --max-new-tokens 96 --temperature 0.9 --top-p 0.95 --top-k 50 \
    --max-atoms 20 --prompt-style h1_rich_plan_formula_prefill_v1 \
    --ccfd-mode "${mode}" --no-include-sample-id --seed "${seed}" \
    --do-sample --stop-after-plan-marker --truncate-after-plan-marker
  python - "${output}/sample_metrics.json" "${output}/run_config.json" \
    "${NUM_SAMPLES}" "${mode}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
if int(metrics.get("requested_samples", -1)) != int(sys.argv[3]):
    raise SystemExit("requested denominator changed")
if str(config.get("ccfd_mode")) != sys.argv[4]:
    raise SystemExit("CCFD arm identity changed")
if config.get("weights_changed_by_ccfd") is not False or config.get("tokenizer_resized_by_ccfd") is not False:
    raise SystemExit("same-checkpoint/tokenizer contract changed")
PY
  touch "${output}/_SUCCESS"
}

run_cell 0 f0 17 off & p0=$!
run_cell 1 f1 17 phase0_assignment & p1=$!
run_cell 2 f0 18 off & p2=$!
run_cell 3 f1 18 phase0_assignment & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for cell in f0_seed17 f1_seed17 f0_seed18 f1_seed18; do
  [[ -f "${RUN_DIR}/${cell}/_SUCCESS" ]] || missing "${cell}"
done
touch "${RUN_DIR}/_SUCCESS"
log "CCFD Phase-1 ${RUN_KIND} completed: ${RUN_DIR}"
