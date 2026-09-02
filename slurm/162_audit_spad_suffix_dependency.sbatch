#!/usr/bin/env bash
#SBATCH --job-name=spad-suffix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=140G
#SBATCH --time=00:30:00
#SBATCH --output=spad-suffix-%j.out
#SBATCH --error=spad-suffix-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly RUN="${ROOT}/runs/spad_suffix_dependency_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
[[ "${#allocated[@]}" -eq 1 ]]
env CUDA_VISIBLE_DEVICES="${allocated[0]}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/scripts/audit_spad_suffix_dependency.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --checkpoint-path "${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392" \
  --pointer-data "${ROOT}/data/spad_species_pointer_v1_20260903/train.jsonl" \
  --teacher-data "${ROOT}/data/c3fd_native_teacher_sft_canonical_v2_20260902/train.jsonl" \
  --output "${RUN}/SUFFIX_DEPENDENCY.json" >"${RUN}/audit.out" 2>"${RUN}/audit.err"
[[ -f "${RUN}/SUFFIX_DEPENDENCY.json" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
