#!/usr/bin/env bash
#SBATCH --job-name=rollout-state-capture
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=rollout-state-capture-%j.out
#SBATCH --error=rollout-state-capture-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${ROLLOUT_PILOT_BASE_POLICY:?}"
readonly PLANS="${ROLLOUT_PILOT_PLANS:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/rollout_matched_capture_${SLURM_JOB_ID}"

mkdir "${RUN}"
trap 'set +e; touch "${RUN}/_FAILED"; exit 1' ERR
[[ "$(wc -l < "${PLANS}")" -eq 128 ]]
[[ -f "${BASE}/adapter_model.safetensors" ]]
[[ -f "${BASE}/periodic_relation_adapter.pt" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	rollout_matched_capture_v1
plans	${PLANS}
base	${BASE}
plans_count	128
stages_per_plan	4
expected_transitions	512
schedule	exact_axis
seed	95117
temperature	0.7
retry_replacement	false
outcomes_read	false
EOF

env CUDA_VISIBLE_DEVICES="${allocated[0]}" PYTHONPATH="${CODE}/src" \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${CODE}/scripts/capture_rollout_matched_states.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${BASE}" \
  --periodic-relation-checkpoint "${BASE}" \
  --periodic-relation-rank 64 \
  --plans-jsonl "${PLANS}" \
  --output-dir "${RUN}/data" \
  --seed 95117 \
  --temperature 0.7 \
  >"${RUN}/capture.out" 2>"${RUN}/capture.err"

[[ -f "${RUN}/data/_SUCCESS" ]]
[[ "$(wc -l < "${RUN}/data/transitions.jsonl")" -eq 512 ]]
[[ "$(wc -l < "${RUN}/data/train.jsonl")" -eq 384 ]]
[[ "$(wc -l < "${RUN}/data/val.jsonl")" -eq 128 ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
