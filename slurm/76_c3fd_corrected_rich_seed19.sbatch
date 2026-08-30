#!/usr/bin/env bash
#SBATCH --job-name=h1a2-c3fd-rich19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=h1a2-c3fd-rich19-%j.out
#SBATCH --error=h1a2-c3fd-rich19-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly CHECKPOINT="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly RUN="${ROOT}/runs/c3fd_corrected_rich_seed19_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CONTRACT_SHA=3a96295b40d3f112d61c86737311db818857460e7e5b9d1dcccecbb1f7065371
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly CHECKPOINT_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly SAMPLER_SHA=0fca6923b8600f7d94c2af1c0409038f515d481f572ec55a4861691055c1a7f3

mkdir -p "${ROOT}/runs/contracts"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

mkdir "${CONTRACT_DIR}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\t%s\n' 'h1a2_c3fd_corrected_rich_seed19_v1'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'planner_checkpoint_seed\t17\n'
  printf 'sampling_seed\t19\n'
  printf 'requested\t1000\n'
  printf 'temperature\t0.9\n'
  printf 'top_p\t0.95\n'
  printf 'top_k\t0\n'
  printf 'reachability\tpauling_bitset\n'
  printf 'spacegroup_mode\tindependent_head\n'
  printf 'outcomes_read\tfalse\n'
  printf 'gpu_jobs_used\t0\n'
} >"${RUN}/scientific_contract.tsv"

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')" == "${VOCAB_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/sample_c3fd_plans.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
sha256sum "${CHECKPOINT}" "${DATA}/vocabulary.json" \
  "${ROOT}/scripts/sample_c3fd_plans.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${ROOT}/scripts/sample_c3fd_plans.py" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA}" \
  --output-dir "${OUTPUT}" \
  --num-samples 1000 \
  --start-index 0 \
  --seed 19 \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 0 \
  --pair-prior-weight 0 \
  --max-species 7 \
  --reachability-mode pauling_bitset \
  --spacegroup-mode independent_head \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
[[ "$(wc -l <"${OUTPUT}/raw_generations.jsonl")" -eq 1000 ]] || exit 3
[[ "$(wc -l <"${OUTPUT}/plans_for_dlm.jsonl")" -eq 1000 ]] || exit 3
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
