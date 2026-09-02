#!/usr/bin/env bash
#SBATCH --job-name=spad-pointer-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=spad-pointer-data-%j.out
#SBATCH --error=spad-pointer-data-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly FUSED="${ROOT}/data/c3fd_llama_fused_typed_v4_20260831"
readonly TEACHER="${ROOT}/data/c3fd_native_teacher_sft_canonical_v2_20260902"
readonly VOCAB="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json"
readonly MP20="${PROJECT}/reference/crysllmgen/data/mp_20"
readonly OUTPUT="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly RUN="${ROOT}/runs/build_spad_species_pointer_data_${SLURM_JOB_ID}"
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

[[ ! -e "${OUTPUT}" ]]
for path in "${SOURCE}/scripts/build_spad_species_pointer_data.py" \
  "${FUSED}/train.jsonl" "${FUSED}/val.jsonl" \
  "${TEACHER}/train.jsonl" "${TEACHER}/val.jsonl" \
  "${VOCAB}" "${MP20}/train.csv" "${MP20}/val.csv"; do
  [[ -f "${path}" ]]
done

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/build_spad_species_pointer_data.py" \
  --fused-data-dir "${FUSED}" --teacher-data-dir "${TEACHER}" \
  --mp20-dir "${MP20}" --vocabulary "${VOCAB}" \
  --output-dir "${OUTPUT}" --workers 16 \
  >"${RUN}/build.out" 2>"${RUN}/build.err"

[[ -f "${OUTPUT}/_SUCCESS" && -f "${OUTPUT}/manifest.json" ]]
"${PYTHON}" - "${OUTPUT}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.load(open(root/'manifest.json'))
assert manifest['splits']['train']['rows'] == 24558
assert manifest['splits']['val']['rows'] == 8158
assert sum(1 for line in open(root/'train.jsonl') if line.strip()) == 24558
assert sum(1 for line in open(root/'val.jsonl') if line.strip()) == 8158
assert manifest['energy_hull_or_generated_outcomes_read'] is False
assert manifest['composition_mutation'] is False
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"
