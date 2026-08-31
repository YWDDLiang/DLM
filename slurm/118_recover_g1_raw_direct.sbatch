#!/usr/bin/env bash
#SBATCH --job-name=g1-raw-direct-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=g1-raw-direct-recover-%j.out
#SBATCH --error=g1-raw-direct-recover-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${ROOT}/runs/compact_v2_periodic_geometry_g1_raw_39105"
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_REPORT="${ROOT}/runs/fused_planner_direct_only_39098/direct/report.json"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/compact_v2_periodic_geometry_g1_raw_recovery_${SLURM_JOB_ID}"

readonly ATTEMPTS_SHA=f6792e81f319106a7351936d3581c1a8c4aaede5288e9d23d854c29fd76004f5
readonly MANIFEST_SHA=a44630f1a08699d4b57b15531cbb27fb3c7274cddb4740b7fddbbb719a688b19
readonly GRAPHS_SHA=9bae1d1aaa5c8aed48314f970fa4baaa00e9afa2b83145aa38cce459e4fd380c
readonly PLANS_SHA=f5c5676d52bbd2e53f80c312c3a8c2475582e4c2c71ee12b13bb3db17de8d22d
readonly ASSEMBLER_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly BASE_REPORT_SHA=b5881e2999c936fd255972a4d5418cce4d4dcc8325da2666c60b5ef6add0b190

mkdir "${RUN}"
mkdir -p "${RUN}/evaluation/generation" "${RUN}/evaluation/direct"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for spec in \
  "${ATTEMPTS_SHA}:${SOURCE}/body/raw_generations.jsonl" \
  "${MANIFEST_SHA}:${SOURCE}/body/SGTC_BODY_MANIFEST.json" \
  "${GRAPHS_SHA}:${SOURCE}/body/proposal_graphs.pt" \
  "${PLANS_SHA}:${SOURCE}/plans_for_body.jsonl" \
  "${ASSEMBLER_SHA}:${ASSEMBLER}" \
  "${DIRECT_SHA}:${DIRECT}" \
  "${GT_SHA}:${GT_CSV}" \
  "${BASE_REPORT_SHA}:${BASE_REPORT}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
[[ -f "${SOURCE}/body/_SUCCESS" ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_geometry_g1_raw_recovery_v1
source_body_job	39105
source_body_parsed	245/256
body_rerun	false
science_parameters_changed	false
engineering_fix	assembler_arm_label_candidate
raw_gate	comp_valid_ge244_body_ge246_direct_gt106
model494	false
chgnet	false
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${SOURCE}/body/raw_generations.jsonl" \
  "${SOURCE}/body/SGTC_BODY_MANIFEST.json" "${SOURCE}/body/proposal_graphs.pt" \
  "${SOURCE}/plans_for_body.jsonl" "${ASSEMBLER}" "${DIRECT}" "${GT_CSV}" \
  "${BASE_REPORT}" >"${RUN}/INPUTS.sha256"

"${PYTHON}" "${ASSEMBLER}" --body-dir "${SOURCE}/body" \
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
    "schema": "compact_v2_periodic_geometry_g1_raw_final_v1",
    "requested": 256,
    "body_parsed": int(report["generation_succeeded"]),
    "comp_valid": int(report["comp_valid_count"]),
    "struct_valid": int(report["struct_valid_count"]),
    "direct_joint": int(report["valid_count"]),
    "base": {"body_parsed": 248, "comp_valid": 248, "struct_valid": 106, "direct_joint": 106},
    "body_reused_from_job": 39105,
    "model494": False,
    "chgnet": False,
}
result["raw_gate_pass"] = result["comp_valid"] >= 244 and result["body_parsed"] >= 246 and result["direct_joint"] > 106
assert report["denominator"] == 256 and report["attempts"] == 256
assert report["retry_or_replacement_used"] is False
(root / "G1_RAW_FINAL.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
