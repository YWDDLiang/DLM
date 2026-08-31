#!/usr/bin/env bash
#SBATCH --job-name=c3fd-prosp21
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=c3fd-prosp21-%j.out
#SBATCH --error=c3fd-prosp21-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly CHECKPOINT="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly RUN="${ROOT}/runs/c3fd_prospective_seed21_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CHECKPOINT_SHA=87c1673f709c14488848196e3f466b0f4797cc3511108a07b6d498ca96d17d53
readonly VOCAB_SHA=3268dccfb8562a2936f9a81d928f547b4f75c2471d4b383c2e36bebc7b41a6e3
readonly SAMPLER_SHA=0fca6923b8600f7d94c2af1c0409038f515d481f572ec55a4861691055c1a7f3

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

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${CHECKPOINT_SHA}" ]]
[[ "$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')" == "${VOCAB_SHA}" ]]
[[ "$(sha256sum "${ROOT}/scripts/sample_c3fd_plans.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_prospective_seed21_source_v1
planner_checkpoint_seed	17
sampling_seed	21
requested	1000
temperature	0.9
top_p	0.95
spacegroup_mode	independent_head
selection_or_outcomes	false
device	cpu
EOF
sha256sum "${CHECKPOINT}" "${DATA}/vocabulary.json" \
  "${ROOT}/scripts/sample_c3fd_plans.py" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${ROOT}/scripts/sample_c3fd_plans.py" \
  --checkpoint "${CHECKPOINT}" --data-dir "${DATA}" --output-dir "${OUTPUT}" \
  --num-samples 1000 --start-index 0 --seed 21 --temperature 0.9 --top-p 0.95 \
  --top-k 0 --pair-prior-weight 0 --max-species 7 \
  --reachability-mode pauling_bitset --spacegroup-mode independent_head \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"

[[ -f "${OUTPUT}/_SUCCESS" ]]
[[ "$(wc -l <"${OUTPUT}/raw_generations.jsonl")" -eq 1000 ]]
[[ "$(wc -l <"${OUTPUT}/plans_for_dlm.jsonl")" -eq 1000 ]]
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
