#!/usr/bin/env bash
#SBATCH --job-name=g2-full-offline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=g2-full-offline-%j.out
#SBATCH --error=g2-full-offline-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly BODY_RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_39112"
readonly RAW_RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_raw_recovery_39114"
readonly REFINE_RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_model494_39115"
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_full_eval_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly BASE_RELAX_CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime"
readonly ASSEMBLE_REFINED="${ROOT}/scripts/assemble_grounding_repeat.py"
readonly FULL_EVAL="${EVAL_RUNTIME}/run_full_reconstructed_eval.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly RAW_GENERATION="${BODY_RUN}/evaluation/generation/generation.jsonl"
readonly REFINED_PT="${REFINE_RUN}/refine/dlm_refined_mp_248.pt"
readonly RAW_DIRECT="${RAW_RUN}/direct/report.json"

readonly RAW_GENERATION_SHA=44b61f6696886f786e75b4cc73ddae4c6739f31b7d23bbeef54ba39e704855c1
readonly REFINED_PT_SHA=97fccd799cc62d7243446a8e999004611ee13c61b000c21d17bcdd73e1b51149
readonly REFINE_METRICS_SHA=f9aa1436a1d53c714a526fb6fa7cf5c9a7aaa568bc5e76e80589fe467bdf7c0d
readonly RAW_DIRECT_SHA=490c3040e8b421bb8a48125178fac797870cb9d65537898c7574f7ba7146002a
readonly ASSEMBLE_REFINED_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly FULL_EVAL_SHA=632c7116faa3738ab1aeadfd7d33d02f21e7b349f41ee874f2a46c753fd200a5
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71

mkdir "${RUN}"
mkdir -p "${RUN}/logs" "${RUN}/raw/direct"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for spec in \
  "${RAW_GENERATION_SHA}:${RAW_GENERATION}" \
  "${REFINED_PT_SHA}:${REFINED_PT}" \
  "${REFINE_METRICS_SHA}:${REFINE_RUN}/refine/refinement_metrics.json" \
  "${RAW_DIRECT_SHA}:${RAW_DIRECT}" \
  "${ASSEMBLE_REFINED_SHA}:${ASSEMBLE_REFINED}" \
  "${FULL_EVAL_SHA}:${FULL_EVAL}" \
  "${CONFIG_SHA}:${V3_SOURCE}/CONFIG.json" \
  "${DIRECT_SHA}:${DIRECT}" \
  "${GT_SHA}:${GT_CSV}" \
  "${CACHE_SHA}:${BASE_RELAX_CACHE}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 2
done
[[ -f "${RAW_RUN}/_RAW_SUCCESS" && -f "${REFINE_RUN}/_SUCCESS" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_full_offline_v1
source_body_job	39112
source_raw_direct_job	39114
source_model494_job	39115
requested	256
raw_generation_reused	true
raw_direct_reused	true
refined_assembly	model494_tau800_seed101117
evaluation_order	raw_and_refined_parallel_after_raw_terminal
chgnet	raw_and_refined
official_query	false
missing_policy	preserve_sample_idx
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${RAW_GENERATION}" \
  "${REFINED_PT}" "${REFINE_RUN}/refine/refinement_metrics.json" "${RAW_DIRECT}" \
  "${ASSEMBLE_REFINED}" "${FULL_EVAL}" "${V3_SOURCE}/CONFIG.json" \
  "${DIRECT}" "${GT_CSV}" "${BASE_RELAX_CACHE}" >"${RUN}/INPUTS.sha256"

cp "${RAW_DIRECT}" "${RUN}/raw/direct/report.json"
cp "${RAW_RUN}/direct/attempt_metrics.jsonl" "${RUN}/raw/direct/attempt_metrics.jsonl"
"${PYTHON}" "${ASSEMBLE_REFINED}" --body-dir "${BODY_RUN}/body" \
  --refine-dir "${REFINE_RUN}/refine" --output-dir "${RUN}/refined/generation" \
  --arm candidate --repeat 1718 --dlm-seed 91117 --refiner-seed 101117 \
  --denominator 256

cp --reflink=auto "${BASE_RELAX_CACHE}" "${RUN}/raw_relax_cache.jsonl"
cp --reflink=auto "${BASE_RELAX_CACHE}" "${RUN}/refined_relax_cache.jsonl"

env OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
  "${PYTHON}" "${DIRECT}" \
  --generation-jsonl "${RUN}/refined/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${RUN}/refined/direct" --eval-model-name mp20 \
  >"${RUN}/logs/refined-direct.out" 2>"${RUN}/logs/refined-direct.err" & direct_pid=$!

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
  PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
  "${PYTHON}" "${FULL_EVAL}" --arm candidate --repeat 0 \
  --generation-jsonl "${RAW_GENERATION}" --config "${V3_SOURCE}/CONFIG.json" \
  --r03e-root "${R03E}" --working-relax-cache "${RUN}/raw_relax_cache.jsonl" \
  --output-dir "${RUN}/raw/full_reconstructed" \
  >"${RUN}/logs/raw-full.out" 2>"${RUN}/logs/raw-full.err" & raw_pid=$!

env CUDA_VISIBLE_DEVICES="${allocated[1]}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
  PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
  "${PYTHON}" "${FULL_EVAL}" --arm candidate --repeat 0 \
  --generation-jsonl "${RUN}/refined/generation/generation.jsonl" \
  --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
  --working-relax-cache "${RUN}/refined_relax_cache.jsonl" \
  --output-dir "${RUN}/refined/full_reconstructed" \
  >"${RUN}/logs/refined-full.out" 2>"${RUN}/logs/refined-full.err" & refined_pid=$!

wait "${direct_pid}"
wait "${raw_pid}"
wait "${refined_pid}"
[[ -f "${RUN}/refined/direct/report.json" ]]
[[ -f "${RUN}/raw/full_reconstructed/_SUCCESS" ]]
[[ -f "${RUN}/refined/full_reconstructed/_SUCCESS" ]]

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"
