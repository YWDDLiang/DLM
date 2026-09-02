#!/usr/bin/env bash
#SBATCH --job-name=spad-pointer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=spad-pointer-%j.out
#SBATCH --error=spad-pointer-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly PLANNER="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly DATA="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly VOCAB="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json"
readonly RUN="${ROOT}/runs/spad_species_pointer_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/train_seed86017"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for path in "${PLANNER}/_SUCCESS" "${PLANNER}/typed_residual_config.json" \
  "${PLANNER}/typed_residual_state.pt" \
  "${PLANNER}/llama_adapter/adapter_model.safetensors" \
  "${DATA}/_SUCCESS" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${C3FD}" "${VOCAB}" \
  "${SOURCE}/src/scripts/train_spad_species_pointer.py"; do
  [[ -f "${path}" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/contract.tsv" <<EOF
field\tvalue
method\tPlan-conditioned Llama species pointer
teacher\tMP20 periodic maximum-contact-tree
train_rows\t24558
epochs\t1
seed\t86017
batch_size\t128
composition_mutation\tfalse
planner_llama_trainable\tfalse
checkpoint_selection\tnone
EOF

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/train_spad_species_pointer.py" \
  --llama-model "${MODEL}" --planner-final "${PLANNER}" \
  --pointer-data-dir "${DATA}" --c3fd-checkpoint "${C3FD}" \
  --vocabulary "${VOCAB}" --output-dir "${OUTPUT}" \
  --seed 86017 --batch-size 128 --lr 1e-3 --warmup-steps 20 \
  --num-workers 4 >"${RUN}/train.out" 2>"${RUN}/train.err"

[[ -f "${OUTPUT}/_SUCCESS" && -f "${OUTPUT}/pointer_state.pt" ]]
"${PYTHON}" - "${OUTPUT}/train_metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['status']=='complete' and m['epochs_completed']==1
assert m['composition_candidate_set_preserved'] is True
assert m['planner_llama_frozen_verified'] is True
assert m['validation']['loss'] >= 0.0
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"
