#!/usr/bin/env bash
#SBATCH --job-name=h1a2-c3fd25-1k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-c3fd25-1k-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly DATA="${H1A2_C3FD_V21_DATA:?export H1A2_C3FD_V21_DATA}"
readonly SOURCE_RUN="${H1A2_C3FD_V25_SOURCE_RUN:?export H1A2_C3FD_V25_SOURCE_RUN}"
readonly CONTROL_RUN="${H1A2_C3FD_CONTROL_RUN:?export H1A2_C3FD_CONTROL_RUN}"
readonly TRAIN_JSONL="${H1A2_C3FD_TRAIN_JSONL:?export H1A2_C3FD_TRAIN_JSONL}"
readonly V2_DIAGNOSTIC="${H1A2_C3FD_V2_DIAGNOSTIC:?export H1A2_C3FD_V2_DIAGNOSTIC}"
readonly RUN_DIR="${ROOT}/runs/c3fd_v25_requested1000_${SLURM_JOB_ID}"

for path in "${DATA}/_SUCCESS" "${SOURCE_RUN}/_SUCCESS" "${CONTROL_RUN}" "${TRAIN_JSONL}" "${V2_DIAGNOSTIC}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
python - "${SOURCE_RUN}/final/C3FD_V25_PILOT_FINAL.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('gates',{}).get('step3_pass') is not True:
    raise SystemExit('requested256 source did not pass every promotion gate')
PY

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	requested1000_confirmation
source_indices	${SOURCE_RUN}/c3_seed17,18 sample_idx0-255
new_indices	256-999 exactly once
candidate	C3FD_v2.5_pauling_bitset
checkpoints	${SOURCE_RUN}/train_seed17,18/checkpoint.pt
seeds	17,18
parallel_shards	seed17:256-627,628-999;seed18:256-627,628-999
repair	false
replacement	false
backtracking	false
rerank	false
rl	false
EOF

sample_shard() {
  local gpu="$1" seed="$2" name="$3" start="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/sample_c3fd_plans.py" \
    --checkpoint "${SOURCE_RUN}/train_seed${seed}/checkpoint.pt" --data-dir "${DATA}" \
    --output-dir "${RUN_DIR}/${name}_seed${seed}" --num-samples 372 \
    --start-index "${start}" --seed "${seed}" --temperature 0.9 --top-p 0.95 \
    --top-k 0 --pair-prior-weight 0 --max-species 7 \
    --reachability-mode pauling_bitset
}

sample_shard 0 17 tail_a 256 & p0=$!
sample_shard 1 17 tail_b 628 & p1=$!
sample_shard 2 18 tail_a 256 & p2=$!
sample_shard 3 18 tail_b 628 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

for seed in 17 18; do
  python "${ROOT}/scripts/assemble_c3fd_sample_segments.py" \
    --segment "${SOURCE_RUN}/c3_seed${seed}" \
    --segment "${RUN_DIR}/tail_a_seed${seed}" \
    --segment "${RUN_DIR}/tail_b_seed${seed}" \
    --output-dir "${RUN_DIR}/c3_seed${seed}" --requested 1000 --seed "${seed}" \
    --reachability-mode pauling_bitset
done

python "${ROOT}/scripts/finalize_c3fd_v22_pilot.py" \
  --control-run "${CONTROL_RUN}" --candidate-run "${RUN_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" --v2-diagnostic "${V2_DIAGNOSTIC}" \
  --requested 1000 --candidate-arm c3fd_v25 \
  --output-stem C3FD_V25_REQUESTED1000_FINAL \
  --schema h1a2_c3fd_v25_requested1000_final_v1 \
  --title "C³FD-v2.5 requested-1000 confirmation" \
  --gate-name step4_pass --output-dir "${RUN_DIR}/final"

for path in "${RUN_DIR}/tail_a_seed17/_SUCCESS" "${RUN_DIR}/tail_b_seed17/_SUCCESS" "${RUN_DIR}/tail_a_seed18/_SUCCESS" "${RUN_DIR}/tail_b_seed18/_SUCCESS" "${RUN_DIR}/c3_seed17/_SUCCESS" "${RUN_DIR}/c3_seed18/_SUCCESS" "${RUN_DIR}/final/_SUCCESS"; do
  [[ -f "${path}" ]] || missing "${path}"
done
touch "${RUN_DIR}/_SUCCESS"
log "C3FD-v2.5 requested1000 confirmation completed: ${RUN_DIR}"
