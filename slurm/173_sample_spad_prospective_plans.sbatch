#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp-freeze
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=spad-prosp-freeze-%j.out
#SBATCH --error=spad-prosp-freeze-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly SOURCE_RUN="${SPAD_PROSPECTIVE_SOURCE_RUN:?}"
readonly SOURCE_PLANS="${SOURCE_RUN}/plans/plans_for_dlm.jsonl"
readonly MP20_TRAIN=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length/train.jsonl
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_freeze_${SLURM_JOB_ID}"
readonly COHORT="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${SOURCE_RUN}/_SUCCESS" && -f "${SOURCE_PLANS}" && ! -e "${COHORT}" ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${SOURCE}/scripts/freeze_spad_prospective_plans.py" \
  --source-plans "${SOURCE_PLANS}" --mp20-train "${MP20_TRAIN}" \
  --exclude-cohort-root "${ROOT}/cohorts" --output-dir "${COHORT}" \
  --count 256 --planner-sampling-seed 23 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

"${PYTHON}" - "${COHORT}" "${RUN}" <<'PY'
import json
import pathlib
import sys
cohort = pathlib.Path(sys.argv[1]); run = pathlib.Path(sys.argv[2])
manifest = json.load(open(cohort / "manifest.json", encoding="utf-8"))
assert manifest["selected"] == 256
assert manifest["unique_exact"] == 256
assert manifest["outcomes_read"] is False
assert manifest["planner_resampled_after_freeze"] is False
assert all(manifest["gates"].values())
summary = {
    "schema": "spad_prospective_freeze_v2",
    "selected": 256,
    "unique_chemsys": manifest["unique_chemsys"],
    "exclusions": manifest["exclusions"],
    "cohort": str(cohort),
    "actual_plans_frozen": True,
}
(run / "FREEZE_FINAL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
