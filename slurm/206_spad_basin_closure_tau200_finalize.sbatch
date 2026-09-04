#!/usr/bin/env bash
#SBATCH --job-name=spad-t200-final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=spad-t200-final-%j.out
#SBATCH --error=spad-t200-final-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly ARRAY_RUN="${SPAD_TAU200_RELAX_ARRAY_RUN:?}"
readonly TAU_RUN="${ROOT}/runs/spad_basin_closure_tau200_s17_39732"
readonly GENERATION="${TAU_RUN}/generation/generation.jsonl"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly OFFICIAL="${ROOT}/runs/spad_basin_closure_official_20260904_v1/official_mp_cache"
readonly BASELINE="${ROOT}/runs/spad_basin_closure_official_final_20260904_v1/attempt_results_official.jsonl"
readonly RAW_SCREEN="${ROOT}/runs/spad_basin_closure_native_raw_screen_39710/screen/RAW_SCREEN_FINAL.json"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_tau200_final_${SLURM_JOB_ID}"
readonly CELL="${RUN}/evaluation"
readonly FINAL="${RUN}/official"

mkdir "${RUN}"; mkdir "${RUN}/logs"; mkdir "${CELL}"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${TAU_RUN}/_SUCCESS" "${ARRAY_RUN}/_SUCCESS_rank0" "${ARRAY_RUN}/_SUCCESS_rank1" \
 "${OFFICIAL}/completion_SUCCESS" "${BASELINE}" "${RAW_SCREEN}"; do [[ -f "$p" ]]; done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#allocated[@]}" -eq 1 ]]
cp --reflink=auto "${CACHE}" "${CELL}/relax_cache.jsonl"
cat "${ARRAY_RUN}/relax_cache_shards/cache_rank0.jsonl" >>"${CELL}/relax_cache.jsonl"
cat "${ARRAY_RUN}/relax_cache_shards/cache_rank1.jsonl" >>"${CELL}/relax_cache.jsonl"

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 H1_ACTIVE_DENOMINATOR=256 \
 PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" "${PYTHON}" "${FULL}" \
 --arm candidate --repeat 0 --generation-jsonl "${GENERATION}" \
 --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
 --working-relax-cache "${CELL}/relax_cache.jsonl" \
 --output-dir "${CELL}/full_reconstructed" >"${RUN}/logs/full.out" 2>"${RUN}/logs/full.err"

env H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime" "${PYTHON}" \
 "${SOURCE}/scripts/finalize_spad_basin_closure_official.py" \
 --preofficial-labels "${CELL}/full_reconstructed/attempt_labels_preofficial.jsonl" \
 --generation-jsonl "${GENERATION}" --official-cache-dir "${OFFICIAL}" \
 --baseline-official-attempts "${BASELINE}" --raw-screen-json "${RAW_SCREEN}" \
 --raw-screen-arm closure_ce --endpoint closure_ce_model494_tau200_stream17 \
 --model494 --model494-tau 200 --output-dir "${FINAL}" \
 >"${RUN}/logs/official.out" 2>"${RUN}/logs/official.err"

trap - ERR
touch "${RUN}/_SUCCESS"
