#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp23
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=spad-prosp23-%j.out
#SBATCH --error=spad-prosp23-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly CHECKPOINT="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly MP20_TRAIN=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/data/dlm_sft/mp_20_r5_exact_length/train.jsonl
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_seed23_freeze_${SLURM_JOB_ID}"
readonly COHORT="${ROOT}/cohorts/spad_prospective_seed23_256_v1_20260903"
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ ! -e "${COHORT}" ]]

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  "${PYTHON}" "${SOURCE}/scripts/sample_c3fd_plans.py" \
  --checkpoint "${CHECKPOINT}" --data-dir "${DATA}" \
  --output-dir "${RUN}/source_plans" --num-samples 1000 \
  --start-index 0 --seed 23 --temperature 0.9 --top-p 0.95 --top-k 0 \
  --pair-prior-weight 0 --max-species 7 --reachability-mode pauling_bitset \
  --spacegroup-mode independent_head \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"
[[ -f "${RUN}/source_plans/_SUCCESS" ]]
[[ "$(wc -l <"${RUN}/source_plans/plans_for_dlm.jsonl")" -eq 1000 ]]

"${PYTHON}" "${SOURCE}/scripts/freeze_c3fd_native_prospective_cohort.py" \
  --source-plans "${RUN}/source_plans/plans_for_dlm.jsonl" \
  --mp20-train "${MP20_TRAIN}" --exclude-cohort-root "${ROOT}/cohorts" \
  --output-dir "${COHORT}" --count 256 --planner-sampling-seed 23 \
  >"${RUN}/freeze.out" 2>"${RUN}/freeze.err"

"${PYTHON}" - "${COHORT}" "${RUN}" <<'PY'
import json
import pathlib
import sys

cohort = pathlib.Path(sys.argv[1])
run = pathlib.Path(sys.argv[2])
manifest = json.load(open(cohort / "manifest.json", encoding="utf-8"))
assert manifest["selected"] == 256
assert manifest["planner_sampling_seed"] == 23
assert manifest["outcomes_read"] is False
assert all(manifest["gates"].values())
rows = [json.loads(line) for line in open(cohort / "ledger.jsonl", encoding="utf-8")]
assert len(rows) == 256
assert len({tuple(sorted(zip(row["plan_state"]["elements"], row["plan_state"]["counts"]))) for row in rows}) == 256
summary = {
    "schema": "spad_prospective_freeze_v1",
    "source_seed": 23,
    "source_rows": 1000,
    "selected": 256,
    "outcomes_read": False,
    "cohort": str(cohort),
}
(run / "FREEZE_FINAL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"

