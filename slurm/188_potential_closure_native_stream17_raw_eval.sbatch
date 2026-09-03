#!/usr/bin/env bash
#SBATCH --job-name=pcdlm-native-raw17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=pcdlm-native-raw17-%j.out
#SBATCH --error=pcdlm-native-raw17-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly TRAIN_GENERATION_RUN="${POTENTIAL_CLOSURE_NATIVE_RUN:?}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly SNAPSHOT="${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/potential_closure_native_s17_raw_eval_${SLURM_JOB_ID}"
readonly BS_GENERATION="${ROOT}/runs/spad_prospective_generation_s17_39537/BS/stream17/raw_generation/generation.jsonl"
readonly CONTROL_GENERATION="${TRAIN_GENERATION_RUN}/closure_control/stream17/raw_generation/generation.jsonl"
readonly POTENTIAL_GENERATION="${TRAIN_GENERATION_RUN}/potential_closed/stream17/raw_generation/generation.jsonl"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

[[ -f "${TRAIN_GENERATION_RUN}/_SUCCESS" ]]
for generation in "${BS_GENERATION}" "${CONTROL_GENERATION}" "${POTENTIAL_GENERATION}"; do
  [[ -f "${generation}" ]]
  [[ "$(wc -l <"${generation}")" -eq 256 ]]
done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

run_direct() {
  local arm="$1" generation="$2"
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
    "${SOURCE}/scripts/run_crysllmgen_validity_fast.py" \
    --generation-jsonl "${generation}" --snapshot-root "${SNAPSHOT}" \
    --output-dir "${RUN}/direct/${arm}" \
    >"${RUN}/logs/${arm}_direct.out" 2>"${RUN}/logs/${arm}_direct.err"
}

run_direct BS "${BS_GENERATION}" & bs_direct_pid="$!"
run_direct closure_control "${CONTROL_GENERATION}" & control_direct_pid="$!"
run_direct potential_closed "${POTENTIAL_GENERATION}" & potential_direct_pid="$!"

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/scripts/evaluate_potential_closure_native_raw.py" evaluate-chgnet \
  --bs-generation "${BS_GENERATION}" \
  --control-generation "${CONTROL_GENERATION}" \
  --potential-generation "${POTENTIAL_GENERATION}" \
  --output-dir "${RUN}/chgnet" --denominator 256 \
  --device cuda:0 --batch-size 16 \
  >"${RUN}/logs/chgnet.out" 2>"${RUN}/logs/chgnet.err"

direct_status=0
wait "${bs_direct_pid}" || direct_status=1
wait "${control_direct_pid}" || direct_status=1
wait "${potential_direct_pid}" || direct_status=1
(( direct_status == 0 ))

"${PYTHON}" "${SOURCE}/scripts/evaluate_potential_closure_native_raw.py" finalize \
  --bs-generation "${BS_GENERATION}" \
  --control-generation "${CONTROL_GENERATION}" \
  --potential-generation "${POTENTIAL_GENERATION}" \
  --bs-direct-dir "${RUN}/direct/BS" \
  --control-direct-dir "${RUN}/direct/closure_control" \
  --potential-direct-dir "${RUN}/direct/potential_closed" \
  --chgnet-dir "${RUN}/chgnet" --output-dir "${RUN}/final" \
  --denominator 256 --bootstrap-seed 20260904 --bootstrap-replicates 10000 \
  >"${RUN}/logs/finalize.out" 2>"${RUN}/logs/finalize.err"

[[ -f "${RUN}/final/NATIVE_RAW_FINAL.json" ]]
[[ -f "${RUN}/final/_SUCCESS" ]]
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
