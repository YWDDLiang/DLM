#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=spad-prosp-freeze-%j.out
#SBATCH --error=spad-prosp-freeze-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly SOURCE_RUN="${SPAD_PROSPECTIVE_SOURCE_RUN:?}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_freeze_${SLURM_JOB_ID}"
readonly COHORT="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${SOURCE_RUN}/_SUCCESS" && ! -e "${COHORT}" ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${SOURCE}/scripts/freeze_spad_prospective_plans.py" \
  --source-records "${SOURCE_RUN}/plans/raw_generations.jsonl" \
  --source-plans "${SOURCE_RUN}/plans/plans_for_dlm.jsonl" \
  --output-dir "${COHORT}" --requested 256 --planner-sampling-seed 23 \
  --minimum-comp-valid 0.95 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

"${PYTHON}" - "${COHORT}/manifest.json" "${RUN}" <<'PY'
import json
import pathlib
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8")); run = pathlib.Path(sys.argv[2])
assert manifest["requested"] == 256
assert manifest["composition_valid_rate"] >= 0.95
assert manifest["selection"] == "none_all_requested_ordinals_retained"
assert manifest["replacement"] is False
assert manifest["planner_resampled_after_freeze"] is False
assert manifest["outcomes_read"] is False
(run / "FREEZE_FINAL.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
