#!/usr/bin/env bash
#SBATCH --job-name=compact-v2-g2-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=170G
#SBATCH --time=05:00:00
#SBATCH --output=compact-v2-g2-raw-%j.out
#SBATCH --error=compact-v2-g2-raw-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export immutable deployed code root}"
readonly CODE_COMMIT="${H1A2_CODE_COMMIT:?export deployed git commit}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/compact_v2_periodic_relation_g2_39107/train/checkpoints/step-348"
readonly PLANS="${ROOT}/runs/c3fd_llama_typed_plans_39088/plans/plans_for_dlm.jsonl"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly BODY_SAMPLER="${CODE_ROOT}/src/scripts/sample_sgtc_l6.py"
readonly CTV_ROLLOUT="${CODE_ROOT}/src/crystal_dlm/ctv_rollout.py"
readonly LLADA_GENERATION="${CODE_ROOT}/src/crystal_dlm/llada_generation.py"
readonly RELATION_RUNTIME="${CODE_ROOT}/src/crystal_dlm/periodic_relation_runtime.py"
readonly RELATION_ADAPTER="${CODE_ROOT}/src/crystal_dlm/periodic_relation_adapter.py"
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly BASE_REPORT="${ROOT}/runs/fused_planner_direct_only_39098/direct/report.json"
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_${SLURM_JOB_ID}"
readonly BODY_PLANS="${RUN}/plans_for_body.jsonl"

readonly PLANS_SHA=f7002caa9f5dcfb3bdb962028673e51da108d58854519a145ceedfa3109728ec
readonly POLICY_SHA=153998f0fe9196cbd4ad879068dc609d9ccd674234582bad7315a4bccb77bff4
readonly POLICY_CONFIG_SHA=61ff08995286cf7679fb2e494064892a778a925becd2d55300a70e354f8ff240
readonly RELATION_STATE_SHA=c0c057a29580476ddb9854c1bb8e90027a353291a493f43ad8ca0aaf0c7969cd
readonly RELATION_CONFIG_SHA=0db88e018554c45c986271c1537dc19d7b13f4c6d4e16f46c37453515f94972c
readonly BODY_SAMPLER_SHA=b1749f1701af67e70578bd73a169bbfa4902c0416cee41d01d22193e8d29fb92
readonly CTV_ROLLOUT_SHA=6c2923963037d2a76664ec3722102018f310b5ce38146e0d8d8df15cc727ebfd
readonly LLADA_GENERATION_SHA=427243edf454d2157b232b439223e3cd4a4fab8353b8a4746c3861c6f7ae3d28
readonly RELATION_RUNTIME_SHA=6d071dc8e5a8e83c3a6842318f84a13c2d9a50be9456c8bb83a1b147a3f7e1c3
readonly RELATION_ADAPTER_SHA=d5bd2706b1027f3122f33b0c361210aabf098609433cb9bb48063d22d433aa13
readonly ASSEMBLER_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly BASE_REPORT_SHA=b5881e2999c936fd255972a4d5418cce4d4dcc8325da2666c60b5ef6add0b190

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

actual_commit="$(git --git-dir="${CODE_ROOT}/.git" --work-tree="${CODE_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${CODE_COMMIT}" ]] || exit 2
for spec in \
  "${PLANS_SHA}:${PLANS}" \
  "${POLICY_SHA}:${POLICY}/adapter_model.safetensors" \
  "${POLICY_CONFIG_SHA}:${POLICY}/adapter_config.json" \
  "${RELATION_STATE_SHA}:${POLICY}/periodic_relation_adapter.pt" \
  "${RELATION_CONFIG_SHA}:${POLICY}/periodic_relation_config.json" \
  "${BODY_SAMPLER_SHA}:${BODY_SAMPLER}" \
  "${CTV_ROLLOUT_SHA}:${CTV_ROLLOUT}" \
  "${LLADA_GENERATION_SHA}:${LLADA_GENERATION}" \
  "${RELATION_RUNTIME_SHA}:${RELATION_RUNTIME}" \
  "${RELATION_ADAPTER_SHA}:${RELATION_ADAPTER}" \
  "${ASSEMBLER_SHA}:${ASSEMBLER}" \
  "${DIRECT_SHA}:${DIRECT}" \
  "${GT_SHA}:${GT_CSV}" \
  "${BASE_REPORT_SHA}:${BASE_REPORT}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
[[ -f "${ROOT}/runs/compact_v2_periodic_relation_g2_39107/_SUCCESS" ]]
[[ -f "$(dirname "${PLANS}")/_SUCCESS" ]]
[[ "$(wc -l <"${PLANS}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_raw_screen_v1
base_report	${BASE_REPORT}
base_body_comp_valid_struct_direct	248,248,106,106
policy	${POLICY}
code_commit	${CODE_COMMIT}
plans	${PLANS}
stream	17
dlm_seed	91117
requested	256
temperature	0.7
schedule	exact_axis
trajectory_per_plan	1
periodic_relation_rank	64
model494	false_in_this_raw_stage_but_mandatory_next
chgnet	false_in_this_raw_stage_but_mandatory_next
full_endpoint_required	true_regardless_raw_direction
official_query	false
retry_filter_replacement_rerank_best_of_n	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${PLANS}" \
  "${POLICY}/adapter_model.safetensors" "${POLICY}/adapter_config.json" \
  "${POLICY}/periodic_relation_adapter.pt" "${POLICY}/periodic_relation_config.json" \
  "${BODY_SAMPLER}" "${CTV_ROLLOUT}" "${LLADA_GENERATION}" \
  "${RELATION_RUNTIME}" "${RELATION_ADAPTER}" "${ASSEMBLER}" \
  "${DIRECT}" "${GT_CSV}" "${BASE_REPORT}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" - "${PLANS}" "${BODY_PLANS}" <<'PY'
import json, pathlib, sys
root = pathlib.Path.cwd() / "src"; sys.path.insert(0, str(root))
from crystal_dlm.composition_identity import identity_from_plan_state, identity_text
source = pathlib.Path(sys.argv[1]); output = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in source.open() if line.strip()]
assert len(rows) == 256 and [int(row["sample_idx"]) for row in rows] == list(range(256))
with output.open("x", encoding="utf-8", newline="\n") as handle:
    for row in rows:
        row = dict(row)
        row["reduced_composition_identity"] = identity_text(identity_from_plan_state(row["plan_state"]))
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
PY

env CUDA_VISIBLE_DEVICES="${allocated[0]}" PYTHONPATH="${CODE_ROOT}/src" \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${BODY_SAMPLER}" --model-path "${MODEL}" \
  --checkpoint-path "${POLICY}" --periodic-relation-checkpoint "${POLICY}" \
  --periodic-relation-rank 64 --prompt-jsonl "${BODY_PLANS}" \
  --output-dir "${RUN}/body" --crysllmgen-dir "${CRYS}" \
  --num-samples 256 --temperature 0.7 --seed 91117 \
  >"${RUN}/body.out" 2>"${RUN}/body.err"
[[ -f "${RUN}/body/_SUCCESS" ]]

"${PYTHON}" "${ASSEMBLER}" --body-dir "${RUN}/body" \
  --output-dir "${RUN}/evaluation/generation" --seed 17 --dlm-seed 91117 \
  --denominator 256 --source-arm full_axis --arm-label candidate

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=8 \
  "${PYTHON}" "${DIRECT}" \
  --generation-jsonl "${RUN}/evaluation/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${RUN}/evaluation/direct" --eval-model-name mp20 \
  >"${RUN}/direct.out" 2>"${RUN}/direct.err"
[[ -f "${RUN}/evaluation/direct/report.json" ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); report = json.load((root / "evaluation/direct/report.json").open())
result = {
    "schema": "compact_v2_periodic_relation_g2_raw_final_v1",
    "requested": 256,
    "body_parsed": int(report["generation_succeeded"]),
    "comp_valid": int(report["comp_valid_count"]),
    "struct_valid": int(report["struct_valid_count"]),
    "direct_joint": int(report["valid_count"]),
    "base": {"body_parsed": 248, "comp_valid": 248, "struct_valid": 106, "direct_joint": 106},
    "full_model494_CHGNet_required": True,
}
assert report["denominator"] == 256 and report["attempts"] == 256
assert report["retry_or_replacement_used"] is False
(root / "G2_RAW_FINAL.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
