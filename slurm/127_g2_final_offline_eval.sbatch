#!/usr/bin/env bash
#SBATCH --job-name=g2-final-offline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=420G
#SBATCH --time=06:00:00
#SBATCH --output=g2-final-offline-%j.out
#SBATCH --error=g2-final-offline-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly GEN="${ROOT}/runs/compact_v2_periodic_relation_g2_final_generation_39137"
readonly RUN="${ROOT}/runs/compact_v2_periodic_relation_g2_final_eval_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RAW_ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly REFINED_ASSEMBLER="${ROOT}/scripts/assemble_grounding_repeat.py"
readonly FULL="${ROOT}/eval_runtime/run_full_reconstructed_eval.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"

readonly BASE_BODY_SHA=a5a920cdf9235ab93e98ee09a5693cb92c67644797c6c2aefd688298b4272caa
readonly BASE_MANIFEST_SHA=f1b7e5b70e87b42a04edb1b358380cd94e9a3fe7d05fc51c3361bb165c064c5e
readonly BASE_REFINED_SHA=a306c73bbaa4ef11894a34707cfe582ada02aac3b162ce4412339abacd6fa7fc
readonly G2_BODY_SHA=7201490ce224aa1b38e76eb7bb479301e4333f11c72b6e488a8d441148ba0488
readonly G2_MANIFEST_SHA=363115968264985fe25a624273616e83ff8e6260cef1b80b4621aeeacb9d7592
readonly G2_REFINED_SHA=5721666d8b1d6a92fd1a0d51eeb151ceadcd38e779ce1afc9fd329433ea04244
readonly RAW_ASSEMBLER_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly REFINED_ASSEMBLER_SHA=e4bd9c4eb8c659aaf3fe49f96899e5805a892771826b605a08dab528919e48d4
readonly FULL_SHA=632c7116faa3738ab1aeadfd7d33d02f21e7b349f41ee874f2a46c753fd200a5
readonly CONFIG_SHA=a79a574209b9ebc4432778dc89ca65b5c18d337d7d7f20408a607f4ab25004ab
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c
readonly CACHE_SHA=9d29489fcf61544ed2420fee7f78c6d59a0dca4bc55f989223bb7000f4961c71

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc=$1; line=$2; command=$3; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$command" >"${RUN}/ENGINEERING_FAILURE.${BASHPID}.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
for spec in "${BASE_BODY_SHA}:${GEN}/arm_BASE/body/raw_generations.jsonl" "${BASE_MANIFEST_SHA}:${GEN}/arm_BASE/body/SGTC_BODY_MANIFEST.json" \
 "${BASE_REFINED_SHA}:${GEN}/arm_BASE/refine/dlm_refined_mp_251.pt" "${G2_BODY_SHA}:${GEN}/arm_G2/body/raw_generations.jsonl" \
 "${G2_MANIFEST_SHA}:${GEN}/arm_G2/body/SGTC_BODY_MANIFEST.json" "${G2_REFINED_SHA}:${GEN}/arm_G2/refine/dlm_refined_mp_252.pt" \
 "${RAW_ASSEMBLER_SHA}:${RAW_ASSEMBLER}" "${REFINED_ASSEMBLER_SHA}:${REFINED_ASSEMBLER}" "${FULL_SHA}:${FULL}" \
 "${CONFIG_SHA}:${V3_SOURCE}/CONFIG.json" "${DIRECT_SHA}:${DIRECT}" "${GT_SHA}:${GT_CSV}" "${CACHE_SHA}:${CACHE}"; do expected="${spec%%:*}";path="${spec#*:}";[[ -f "${path}" && "$(sha256sum "${path}"|awk '{print $1}')" == "${expected}" ]];done
[[ -f "${GEN}/_SUCCESS" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no GPUs}"; [[ "${#gpu[@]}" -eq 4 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	compact_v2_periodic_relation_g2_final_offline_v1
generation_job	39137
arms	BASE,G2
endpoints	raw,refined
requested_per_cell	256
raw_first	true
chgnet	all_four_cells
official_query	false
missing_policy	preserve_sample_idx
selection_retry_topup	false
EOF
sha256sum "${RUN}/scientific_contract.tsv" "${GEN}/arm_BASE/body/raw_generations.jsonl" \
 "${GEN}/arm_BASE/refine/dlm_refined_mp_251.pt" "${GEN}/arm_G2/body/raw_generations.jsonl" \
 "${GEN}/arm_G2/refine/dlm_refined_mp_252.pt" "${RAW_ASSEMBLER}" "${REFINED_ASSEMBLER}" \
 "${FULL}" "${V3_SOURCE}/CONFIG.json" "${DIRECT}" "${GT_CSV}" "${CACHE}" >"${RUN}/INPUTS.sha256"

assemble(){
 local arm="$1" label="$2" body="${GEN}/arm_${1}/body" refine="${GEN}/arm_${1}/refine"
 "${PYTHON}" "${RAW_ASSEMBLER}" --body-dir "${body}" --output-dir "${RUN}/${arm}/raw/generation" \
  --seed 17 --dlm-seed 91117 --denominator 256 --source-arm full_axis --arm-label "${label}"
 "${PYTHON}" "${REFINED_ASSEMBLER}" --body-dir "${body}" --refine-dir "${refine}" \
  --output-dir "${RUN}/${arm}/refined/generation" --arm "${label}" --repeat 0 \
  --dlm-seed 91117 --refiner-seed 101117 --denominator 256
}
assemble BASE control
assemble G2 candidate

run_cell(){
 local slot arm endpoint label cell
 slot="$1"; arm="$2"; endpoint="$3"; label="$4"; cell="${RUN}/${arm}/${endpoint}"
 mkdir "${cell}/evaluation"; cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
  "${PYTHON}" "${DIRECT}" --generation-jsonl "${cell}/generation/generation.jsonl" --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" --output-dir "${cell}/evaluation/direct" \
  --eval-model-name mp20 >"${RUN}/logs/${arm}-${endpoint}-direct.out" 2>"${RUN}/logs/${arm}-${endpoint}-direct.err" & dpid=$!
 env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${ROOT}/eval_runtime:${R03E}/runtime" "${PYTHON}" "${FULL}" \
  --arm "${label}" --repeat 0 --generation-jsonl "${cell}/generation/generation.jsonl" \
  --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" --working-relax-cache "${cell}/relax_cache.jsonl" \
  --output-dir "${cell}/evaluation/full_reconstructed" >"${RUN}/logs/${arm}-${endpoint}-full.out" 2>"${RUN}/logs/${arm}-${endpoint}-full.err"
 wait "${dpid}"
 [[ -f "${cell}/evaluation/direct/report.json" && -f "${cell}/evaluation/full_reconstructed/_SUCCESS" ]]
}
run_cell 0 BASE raw control & p0=$!
run_cell 1 BASE refined control & p1=$!
run_cell 2 G2 raw candidate & p2=$!
run_cell 3 G2 refined candidate & p3=$!
wait "${p0}";wait "${p1}";wait "${p2}";wait "${p3}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"
