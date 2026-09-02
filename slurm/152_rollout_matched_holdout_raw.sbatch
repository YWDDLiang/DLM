#!/usr/bin/env bash
#SBATCH --job-name=rollout-holdout-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=04:00:00
#SBATCH --output=rollout-holdout-raw-%j.out
#SBATCH --error=rollout-holdout-raw-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${ROLLOUT_HOLDOUT_BASE_POLICY:?}"
readonly CANDIDATE="${ROLLOUT_HOLDOUT_CANDIDATE_POLICY:?}"
readonly PLANS="${ROLLOUT_HOLDOUT_PLANS:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly SNAPSHOT="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly SAMPLER="${CODE}/src/scripts/sample_sgtc_l6.py"
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly DIRECT="${CODE}/scripts/run_direct_validity_fast.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/rollout_matched_holdout_raw_${SLURM_JOB_ID}"

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

[[ -f "$(dirname "${PLANS}")/_SUCCESS" && "$(wc -l < "${PLANS}")" -eq 128 ]]
for policy in "${BASE}" "${CANDIDATE}"; do
  [[ -f "${policy}/adapter_model.safetensors" ]]
  [[ -f "${policy}/periodic_relation_adapter.pt" ]]
  [[ -f "${policy}/periodic_relation_config.json" ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	rollout_matched_holdout_raw_v1
plans	${PLANS}
base	${BASE}
candidate	${CANDIDATE}
requested_per_arm	128
dlm_seed	96117
temperature	0.7
schedule	exact_axis
trajectory_per_plan	1
planner_resampled	false
retry_filter_replacement_rerank_best_of_n	false
model494	false
chgnet	false_until_raw_gate_passes
EOF

run_body() {
  local arm="$1" policy="$2" device="$3" output="${RUN}/${1}"
  mkdir "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" PYTHONPATH="${CODE}/src" \
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \
    "${PYTHON}" "${SAMPLER}" \
    --model-path "${MODEL}" \
    --checkpoint-path "${policy}" \
    --periodic-relation-checkpoint "${policy}" \
    --periodic-relation-rank 64 \
    --prompt-jsonl "${PLANS}" \
    --output-dir "${output}/body" \
    --crysllmgen-dir "${CRYS}" \
    --num-samples 128 \
    --expected-denominator 128 \
    --temperature 0.7 \
    --seed 96117 \
    >"${output}/body.out" 2>"${output}/body.err"
  [[ -f "${output}/body/_SUCCESS" ]]
}
run_body BASE "${BASE}" "${allocated[0]}" & base_pid=$!
run_body CANDIDATE "${CANDIDATE}" "${allocated[1]}" & candidate_pid=$!
wait "${base_pid}"
wait "${candidate_pid}"

run_direct() {
  local arm="$1" output="${RUN}/${1}" arm_label=candidate
  [[ "${arm}" == BASE ]] && arm_label=control
  "${PYTHON}" "${ASSEMBLER}" \
    --body-dir "${output}/body" \
    --output-dir "${output}/generation" \
    --seed 20 \
    --dlm-seed 96117 \
    --denominator 128 \
    --source-arm full_axis \
    --arm-label "${arm_label}"
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${output}/generation/generation.jsonl" \
    --snapshot-root "${SNAPSHOT}" \
    --output-dir "${output}/direct" \
    --expected-denominator 128 \
    >"${output}/direct.out" 2>"${output}/direct.err"
  [[ -f "${output}/direct/_SUCCESS" ]]
}
run_direct BASE & base_direct_pid=$!
run_direct CANDIDATE & candidate_direct_pid=$!
wait "${base_direct_pid}"
wait "${candidate_direct_pid}"

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
reports = {
    arm: json.load((root / arm / "direct/report.json").open())
    for arm in ("BASE", "CANDIDATE")
}
attempts = {
    arm: {
        int(row["ordinal"]): row
        for row in (
            json.loads(line)
            for line in (root / arm / "direct/attempt_metrics.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    for arm in ("BASE", "CANDIDATE")
}
assert set(attempts["BASE"]) == set(attempts["CANDIDATE"]) == set(range(128))
for report in reports.values():
    assert report["attempts"] == 128
    assert report["retry_or_replacement_used"] is False

def counts(report):
    return {
        "body": int(report["generation_succeeded"]),
        "composition_valid": int(report["comp_valid_count"]),
        "structure_valid": int(report["struct_valid_count"]),
        "direct": int(report["valid_count"]),
    }

result = {
    "schema": "rollout_matched_holdout_raw_final_v1",
    "requested": 128,
    "arms": {arm: counts(report) for arm, report in reports.items()},
}
base = result["arms"]["BASE"]
candidate = result["arms"]["CANDIDATE"]
invalid_to_valid = sum(
    not attempts["BASE"][i]["valid"] and attempts["CANDIDATE"][i]["valid"]
    for i in range(128)
)
valid_to_invalid = sum(
    attempts["BASE"][i]["valid"] and not attempts["CANDIDATE"][i]["valid"]
    for i in range(128)
)
result.update(
    direct_delta_count=candidate["direct"] - base["direct"],
    invalid_to_valid=invalid_to_valid,
    valid_to_invalid=valid_to_invalid,
    net_invalid_to_valid=invalid_to_valid - valid_to_invalid,
)
result["body_within_one_percentage_point"] = candidate["body"] >= base["body"] - 1
result["composition_within_one_percentage_point"] = (
    candidate["composition_valid"] >= base["composition_valid"] - 1
)
result["supports_raw_chgnet"] = bool(
    result["body_within_one_percentage_point"]
    and result["composition_within_one_percentage_point"]
    and result["direct_delta_count"] >= 4
    and result["net_invalid_to_valid"] >= 4
)
(root / "HOLDOUT_RAW_FINAL.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
