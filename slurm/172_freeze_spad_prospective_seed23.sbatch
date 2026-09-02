#!/usr/bin/env bash
#SBATCH --job-name=spad-prosp-plan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=140G
#SBATCH --time=03:00:00
#SBATCH --output=spad-prosp-plan-%j.out
#SBATCH --error=spad-prosp-plan-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly PLANNER="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly POINTER="${ROOT}/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_prospective_seed23_source_${SLURM_JOB_ID}"
readonly OUTPUT="${RUN}/plans"
mkdir "${RUN}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

"${PYTHON}" - "${RUN}/ordinal_ledger.jsonl" <<'PY'
import json
import sys
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    for sample_idx in range(1000):
        handle.write(json.dumps({"sample_idx": sample_idx}) + "\n")
PY

c3fd_sha="$(sha256sum "${C3FD}" | awk '{print $1}')"
vocab_sha="$(sha256sum "${DATA}/vocabulary.json" | awk '{print $1}')"
ledger_sha="$(sha256sum "${RUN}/ordinal_ledger.jsonl" | awk '{print $1}')"
typed_config_sha="$(sha256sum "${PLANNER}/typed_residual_config.json" | awk '{print $1}')"
typed_state_sha="$(sha256sum "${PLANNER}/typed_residual_state.pt" | awk '{print $1}')"
adapter_config_sha="$(sha256sum "${PLANNER}/llama_adapter/adapter_config.json" | awk '{print $1}')"
adapter_model_sha="$(sha256sum "${PLANNER}/llama_adapter/adapter_model.safetensors" | awk '{print $1}')"

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/sample_c3fd_llama_typed_planner.py" \
  --c3fd-checkpoint "${C3FD}" --data-dir "${DATA}" --llama-model "${MODEL}" \
  --fused-planner-final "${PLANNER}" --species-pointer-state "${POINTER}" \
  --source-ledger "${RUN}/ordinal_ledger.jsonl" --output-dir "${OUTPUT}" \
  --expected-c3fd-sha256 "${c3fd_sha}" --expected-vocabulary-sha256 "${vocab_sha}" \
  --expected-source-ledger-sha256 "${ledger_sha}" \
  --expected-typed-config-sha256 "${typed_config_sha}" \
  --expected-typed-state-sha256 "${typed_state_sha}" \
  --expected-adapter-config-sha256 "${adapter_config_sha}" \
  --expected-adapter-model-sha256 "${adapter_model_sha}" \
  --requested 1000 --expected-requested 1000 --seed 23 --expected-seed 23 \
  --temperature 0.9 --top-p 0.95 --top-k 0 --max-species 7 \
  --minimum-comp-valid 0.95 \
  >"${RUN}/sampling.out" 2>"${RUN}/sampling.err"

"${PYTHON}" - "${OUTPUT}" "${RUN}" <<'PY'
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1]); run = pathlib.Path(sys.argv[2])
metrics = json.load(open(out / "sample_metrics.json", encoding="utf-8"))
rows = [json.loads(line) for line in open(out / "plans_for_dlm.jsonl", encoding="utf-8")]
assert metrics["requested_samples"] == 1000
assert metrics["one_trajectory_per_ordinal"] is True
assert metrics["comp_valid_rate_requested_denominator"] >= 0.95
assert len(rows) == metrics["parsed_samples"]
assert all(row["species_program_source"] == "planner_llama_pointer" for row in rows)
summary = {
    "schema": "spad_prospective_actual_plan_source_v1",
    "seed": 23,
    "requested": 1000,
    "parsed": len(rows),
    "composition_valid_rate": metrics["comp_valid_rate_requested_denominator"],
    "one_trajectory_per_ordinal": True,
    "outcomes_read": False,
}
(run / "PLAN_SOURCE_FINAL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
