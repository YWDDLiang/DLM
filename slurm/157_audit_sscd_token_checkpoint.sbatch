#!/usr/bin/env bash
#SBATCH --job-name=sscd-token-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=sscd-token-audit-%j.out
#SBATCH --error=sscd-token-audit-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly CHECKPOINT="${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392"
readonly TEACHER="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly MP20="${PROJECT}/reference/crysllmgen/data/mp_20"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/sscd_token_checkpoint_audit_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/audit"

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

for path in \
  "${SOURCE}/scripts/audit_sscd_token_checkpoint.py" \
  "${CHECKPOINT}/tokenizer.json" \
  "${CHECKPOINT}/adapter_model.safetensors" \
  "${MP20}/train.csv" "${MP20}/val.csv" \
  "${TEACHER}/train.jsonl" "${TEACHER}/val.jsonl"; do
  [[ -f "${path}" ]]
done

cat >"${RUN}/contract.tsv" <<EOF
field\tvalue
scope\tfull MP20 train/val + deployed DLM tokenizer/checkpoint + teacher contexts
gpu\tfalse
workers\t16
max_length\t382
mutation\tnone
EOF

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/audit_sscd_token_checkpoint.py" \
  --tokenizer-path "${CHECKPOINT}" \
  --adapter-safetensors "${CHECKPOINT}/adapter_model.safetensors" \
  --mp20 "train=${MP20}/train.csv" \
  --mp20 "val=${MP20}/val.csv" \
  --context-jsonl "teacher_train=${TEACHER}/train.jsonl" \
  --context-jsonl "teacher_val=${TEACHER}/val.jsonl" \
  --max-length 382 --workers 16 --output-dir "${OUTPUT}" \
  >"${RUN}/audit.out" 2>"${RUN}/audit.err"

[[ -f "${OUTPUT}/SSCD_TOKEN_CHECKPOINT_AUDIT.json" ]]
[[ -f "${OUTPUT}/_SUCCESS" || -f "${OUTPUT}/_AUDIT_FINDINGS" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
