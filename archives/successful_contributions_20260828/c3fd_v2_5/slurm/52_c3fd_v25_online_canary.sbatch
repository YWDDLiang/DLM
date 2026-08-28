#!/usr/bin/env bash
#SBATCH --job-name=h1a2-c3fd25-c
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-c3fd25-c-%j.out

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
readonly TEACHER_AUDIT="${H1A2_C3FD_V25_TEACHER_AUDIT:?export H1A2_C3FD_V25_TEACHER_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/c3fd_v25_online_canary_${SLURM_JOB_ID}"

for path in "${DATA}/_SUCCESS" "${CONTEXT}" "${CONTROL_RUN}" "${TRAIN_JSONL}" "${V2_DIAGNOSTIC}" "${TEACHER_AUDIT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
python - "${TEACHER_AUDIT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('gate',{}).get('teacher_gate_pass') is not True:
    raise SystemExit(f'v2.5 teacher-witness audit not authorized: {sys.argv[1]}')
PY

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	requested32_canary_then_requested256
control	${CONTROL_RUN}/f0_seed17,${CONTROL_RUN}/f0_seed18 first256
candidate	C3FD_v2.5_pauling_bitset
single_change	constructive_family_charge_arity_pauling_bitset_mask
training_contract	identical_to_v2.1
canary_indices	0-31
tail_indices	32-255
canary_gate	32/32 parsed and independent certificate per seed; zero failures; <=600s
species_top_k	0
top_p	0.95
global_pair_prior_weight	0
seeds	17,18
repair	false
replacement	false
backtracking	false
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

sample_segment() {
  local gpu="$1" seed="$2" name="$3" count="$4" start="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/sample_c3fd_plans.py" \
    --checkpoint "${RUN_DIR}/train_seed${seed}/checkpoint.pt" --data-dir "${DATA}" \
    --output-dir "${RUN_DIR}/${name}_seed${seed}" --num-samples "${count}" \
    --start-index "${start}" --seed "${seed}" --temperature 0.9 --top-p 0.95 \
    --top-k 0 --pair-prior-weight 0 --max-species 7 \
    --reachability-mode pauling_bitset
}

sample_segment 2 17 canary 32 0 & c0=$!
sample_segment 3 18 canary 32 0 & c1=$!
wait "${c0}"
wait "${c1}"

python - "${RUN_DIR}/canary_seed17/sample_metrics.json" "${RUN_DIR}/canary_seed18/sample_metrics.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    x=json.load(open(path,encoding='utf-8'))
    gates={
        'requested32': x.get('requested_samples') == 32,
        'global_indices_0_31': x.get('start_index') == 0 and x.get('end_index_exclusive') == 32,
        'parsed32': x.get('parsed_samples') == 32,
        'certificate32': x.get('all_request_benchmark_comp_valid') == 32,
        'zero_failures': not x.get('failures'),
        'zero_semantic_dead_end': x.get('failures',{}).get('ValueError:semantic_dead_end',0) == 0,
        'mode': x.get('reachability_mode') == 'pauling_bitset',
        'runtime': float(x.get('elapsed_sec',1e99)) <= 600.0,
    }
    if not all(gates.values()):
        raise SystemExit(f'canary failed {path}: {gates}')
print('C3FD_V25_CANARY_PASS')
PY

sample_segment 2 17 tail 224 32 & r0=$!
sample_segment 3 18 tail 224 32 & r1=$!
wait "${r0}"
wait "${r1}"

for seed in 17 18; do
  python "${ROOT}/scripts/assemble_c3fd_sample_segments.py" \
    --segment "${RUN_DIR}/canary_seed${seed}" \
    --segment "${RUN_DIR}/tail_seed${seed}" \
    --output-dir "${RUN_DIR}/c3_seed${seed}" --requested 256 --seed "${seed}" \
    --reachability-mode pauling_bitset
done

python "${ROOT}/scripts/finalize_c3fd_v22_pilot.py" \
  --control-run "${CONTROL_RUN}" --candidate-run "${RUN_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" --v2-diagnostic "${V2_DIAGNOSTIC}" \
  --requested 256 --candidate-arm c3fd_v25 \
  --output-stem C3FD_V25_PILOT_FINAL \
  --schema h1a2_c3fd_v25_pilot_final_v1 \
  --title "C³FD-v2.5 requested-256 online-canary pilot" \
  --output-dir "${RUN_DIR}/final"

for path in "${RUN_DIR}/train_seed17/_SUCCESS" "${RUN_DIR}/train_seed18/_SUCCESS" "${RUN_DIR}/canary_seed17/_SUCCESS" "${RUN_DIR}/canary_seed18/_SUCCESS" "${RUN_DIR}/tail_seed17/_SUCCESS" "${RUN_DIR}/tail_seed18/_SUCCESS" "${RUN_DIR}/c3_seed17/_SUCCESS" "${RUN_DIR}/c3_seed18/_SUCCESS" "${RUN_DIR}/final/_SUCCESS"; do
  [[ -f "${path}" ]] || missing "${path}"
done
touch "${RUN_DIR}/_SUCCESS"
log "C3FD-v2.5 online-canary pilot completed: ${RUN_DIR}"
