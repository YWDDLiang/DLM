#!/usr/bin/env bash
#SBATCH --job-name=g2-model494
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=g2-model494-%j.out
#SBATCH --error=g2-model494-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly SOURCE="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_39112"
readonly GRAPHS="${SOURCE}/body/proposal_graphs.pt"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly REFINER_SCRIPT="${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_model494_${SLURM_JOB_ID}"

readonly GRAPHS_SHA=80b5ac332ddd14e885dfaaedd9311c0c7b24283c27de45ac61f24becb95ee367
readonly BODY_MANIFEST_SHA=955572fee48c3b91de7720027d1243cb98514350b54a930f8cb489045519bd3a
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498

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
  "${GRAPHS_SHA}:${GRAPHS}" \
  "${BODY_MANIFEST_SHA}:${SOURCE}/body/SGTC_BODY_MANIFEST.json" \
  "${REFINER_SHA}:${REFINER}" \
  "${REFINER_SCRIPT_SHA}:${REFINER_SCRIPT}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
[[ -f "${SOURCE}/body/_SUCCESS" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_model494_v1
source_body_job	39112
source_parsed	248/256
body_rerun	false
model494_sha256	${REFINER_SHA}
tau	800
refiner_seed	101117
seed_by_sample_idx	true
batch_size	1
max_proposals	256
retry_rerank_replacement	false
official_query	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${GRAPHS}" \
  "${SOURCE}/body/SGTC_BODY_MANIFEST.json" "${REFINER}" "${REFINER_SCRIPT}" \
  >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=8 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${REFINER_SCRIPT}" --proposal-graphs "${GRAPHS}" \
  --checkpoint "${REFINER}" --output-dir "${RUN}/refine" \
  --crysllmgen-dir "${CRYS}" --batch-size 1 --seed 101117 \
  --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
  >"${RUN}/refine.out" 2>"${RUN}/refine.err"

readonly METRICS="${RUN}/refine/refinement_metrics.json"
[[ -f "${METRICS}" ]] || exit 3
output_file="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${METRICS}")"
[[ -f "${output_file}" ]] || exit 3
printf '%s\n' "${output_file}" >"${RUN}/REFINED_OUTPUT_PATH"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
