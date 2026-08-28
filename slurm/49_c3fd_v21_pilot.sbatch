#!/usr/bin/env bash
#SBATCH --job-name=h1a2-c3fd21-p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-c3fd21-p-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly DATA="${H1A2_C3FD_V21_DATA:?export H1A2_C3FD_V21_DATA}"
readonly CONTEXT="${H1A2_C3FD_CONTEXT:?export H1A2_C3FD_CONTEXT}"
readonly CONTROL_RUN="${H1A2_C3FD_CONTROL_RUN:?export H1A2_C3FD_CONTROL_RUN}"
readonly TRAIN_JSONL="${H1A2_C3FD_TRAIN_JSONL:?export H1A2_C3FD_TRAIN_JSONL}"
readonly V2_DIAGNOSTIC="${H1A2_C3FD_V2_DIAGNOSTIC:?export H1A2_C3FD_V2_DIAGNOSTIC}"
readonly STEP1_AUDIT="${H1A2_C3FD_V21_STEP1_AUDIT:?export H1A2_C3FD_V21_STEP1_AUDIT}"
readonly STEP2_AUDIT="${H1A2_C3FD_V21_STEP2_AUDIT:?export H1A2_C3FD_V21_STEP2_AUDIT}"
readonly STEP3_AUDIT="${H1A2_C3FD_V21_STEP3_AUDIT:?export H1A2_C3FD_V21_STEP3_AUDIT}"
readonly STEP4_AUDIT="${H1A2_C3FD_V21_STEP4_AUDIT:?export H1A2_C3FD_V21_STEP4_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/c3fd_v21_pilot_${SLURM_JOB_ID}"

for path in "${DATA}/_SUCCESS" "${CONTEXT}" "${CONTROL_RUN}" "${TRAIN_JSONL}" "${V2_DIAGNOSTIC}" "${STEP1_AUDIT}" "${STEP2_AUDIT}" "${STEP3_AUDIT}" "${STEP4_AUDIT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
python - "${STEP1_AUDIT}" "${STEP2_AUDIT}" "${STEP3_AUDIT}" "${STEP4_AUDIT}" <<'PY'
import json,sys
for path,key in zip(sys.argv[1:],('step1_pass','step2_pass','step3_pass','step4_pass')):
    x=json.load(open(path,encoding='utf-8'))
    if x.get('gate',{}).get(key) is not True:
        raise SystemExit(f'{key} not authorized: {path}')
PY
mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	requested256_pilot
control	${CONTROL_RUN}/f0_seed17,${CONTROL_RUN}/f0_seed18 first256
candidate	C3FD_v2.1
family_N_arity	train_all_rows_plus_train_only_supported_strata
semantic_supervision	benchmark_compatible_only
ledger_features	remaining_atoms,net_charge,remaining_species,branch
exact_arity	true
species_top_k	0
top_p	0.95
temperature	validation_LBFGS_per_head
global_pair_prior_weight	0
seeds	17,18
requested_per_seed	256
repair	false
replacement	false
rerank	false
rl	false
EOF

train_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/train_c3fd_planner.py" \
    --data-dir "${DATA}" --context-path "${CONTEXT}" \
    --output-dir "${RUN_DIR}/train_seed${seed}" --seed "${seed}" \
    --semantic-size 256 --decoder-layers 2 --decoder-heads 8 --dropout 0.05 \
    --epochs 10 --batch-size 256 --lr 1e-3 --weight-decay 1e-4
}
train_seed 0 17 & t0=$!
train_seed 1 18 & t1=$!
wait "${t0}"
wait "${t1}"

sample_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/sample_c3fd_plans.py" \
    --checkpoint "${RUN_DIR}/train_seed${seed}/checkpoint.pt" --data-dir "${DATA}" \
    --output-dir "${RUN_DIR}/c3_seed${seed}" --num-samples 256 --start-index 0 \
    --seed "${seed}" --temperature 0.9 --top-p 0.95 --top-k 0 \
    --pair-prior-weight 0 --max-species 7
}
sample_seed 2 17 & s0=$!
sample_seed 3 18 & s1=$!
wait "${s0}"
wait "${s1}"

python "${ROOT}/scripts/finalize_c3fd_v21_pilot.py" \
  --control-run "${CONTROL_RUN}" --candidate-run "${RUN_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" --v2-diagnostic "${V2_DIAGNOSTIC}" \
  --requested 256 --output-dir "${RUN_DIR}/final"

for path in "${RUN_DIR}/train_seed17/_SUCCESS" "${RUN_DIR}/train_seed18/_SUCCESS" "${RUN_DIR}/c3_seed17/_SUCCESS" "${RUN_DIR}/c3_seed18/_SUCCESS" "${RUN_DIR}/final/_SUCCESS"; do
  [[ -f "${path}" ]] || missing "${path}"
done
touch "${RUN_DIR}/_SUCCESS"
log "C3FD-v2.1 requested256 pilot completed: ${RUN_DIR}"
