#!/usr/bin/env bash
#SBATCH --job-name=spad-e-labels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=spad-e-labels-%j.out
#SBATCH --error=spad-e-labels-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_spade_b14c457}"
readonly TEACHER_SOURCE="${SPAD_ENERGY_TEACHER_SOURCE:?}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_energy_labels_${SLURM_JOB_ID}"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

[[ -f "${TEACHER_SOURCE}/backfill_actions/_SUCCESS" ]]
[[ -f "${TEACHER_SOURCE}/model494_tau800/refinement_metrics.json" ]]
[[ "$(wc -l <"${TEACHER_SOURCE}/backfill_actions/candidate_actions.jsonl")" -eq 8192 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/label_spad_energy_actions.py" \
  --candidate-actions "${TEACHER_SOURCE}/backfill_actions/candidate_actions.jsonl" \
  --refinement-metrics "${TEACHER_SOURCE}/model494_tau800/refinement_metrics.json" \
  --output-dir "${RUN}/labels" --device cuda:0 \
  >"${RUN}/logs/labels.out" 2>"${RUN}/logs/labels.err"

[[ -f "${RUN}/labels/_SUCCESS" ]]
[[ "$(wc -l <"${RUN}/labels/groups.jsonl")" -eq 2048 ]]
printf '%s\n' "${TEACHER_SOURCE}" >"${RUN}/TEACHER_SOURCE.txt"
touch "${RUN}/_SUCCESS"
trap - ERR
