#!/usr/bin/env bash
#SBATCH --job-name=g2-raw-direct-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=g2-raw-direct-recover-%j.out
#SBATCH --error=g2-raw-direct-recover-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_39112"
readonly GENERATION="${SOURCE}/evaluation/generation/generation.jsonl"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_recovery_${SLURM_JOB_ID}"

readonly GENERATION_SHA=44b61f6696886f786e75b4cc73ddae4c6739f31b7d23bbeef54ba39e704855c1
readonly REPORT_SHA=f9a094148e7e853711ea7e746a8d1f53963fb250affcfc141b054f37d2241127
readonly BODY_SHA=eb8d1ee2ee764ed6b6858670b919d40a0c53610e759c4f289b18ad9db07770a5
readonly BODY_MANIFEST_SHA=955572fee48c3b91de7720027d1243cb98514350b54a930f8cb489045519bd3a
readonly GRAPHS_SHA=80b5ac332ddd14e885dfaaedd9311c0c7b24283c27de45ac61f24becb95ee367
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c

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

for spec in \
  "${GENERATION_SHA}:${GENERATION}" \
  "${REPORT_SHA}:${SOURCE}/evaluation/generation/generation_report.json" \
  "${BODY_SHA}:${SOURCE}/body/raw_generations.jsonl" \
  "${BODY_MANIFEST_SHA}:${SOURCE}/body/SGTC_BODY_MANIFEST.json" \
  "${GRAPHS_SHA}:${SOURCE}/body/proposal_graphs.pt" \
  "${DIRECT_SHA}:${DIRECT}" \
  "${GT_SHA}:${GT_CSV}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
[[ -f "${SOURCE}/body/_SUCCESS" && -f "${SOURCE}/evaluation/generation/_SUCCESS" ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_raw_recovery_v1
source_body_job	39112
source_body_parsed	248/256
body_rerun	false
science_parameters_changed	false
engineering_fix	frozen_metric_threads8_worker1
full_model494_CHGNet_required	true
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${GENERATION}" \
  "${SOURCE}/evaluation/generation/generation_report.json" \
  "${SOURCE}/body/raw_generations.jsonl" "${SOURCE}/body/SGTC_BODY_MANIFEST.json" \
  "${SOURCE}/body/proposal_graphs.pt" "${DIRECT}" "${GT_CSV}" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${GENERATION}" \
  --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${RUN}/direct" --eval-model-name mp20 \
  >"${RUN}/direct.out" 2>"${RUN}/direct.err"
[[ -f "${RUN}/direct/report.json" ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); report = json.load((root / "direct/report.json").open())
result = {
    "schema": "compact_v2_periodic_relation_g2_raw_final_v1",
    "requested": 256,
    "body_parsed": int(report["generation_succeeded"]),
    "comp_valid": int(report["comp_valid_count"]),
    "struct_valid": int(report["struct_valid_count"]),
    "direct_joint": int(report["valid_count"]),
    "base": {"body_parsed": 248, "comp_valid": 248, "struct_valid": 106, "direct_joint": 106},
    "body_source_job": 39112,
    "full_model494_CHGNet_required": True,
}
assert report["denominator"] == "all_generation_attempts" and report["attempts"] == 256
assert report["retry_or_replacement_used"] is False
(root / "G2_RAW_FINAL.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
