#!/usr/bin/env bash
#SBATCH --job-name=plan1200-offline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=420G
#SBATCH --time=06:00:00
#SBATCH --output=plan1200-offline-%j.out
#SBATCH --error=plan1200-offline-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly RUN="${ROOT}/runs/plan1200_tau800_tau900_offline_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EVAL_RUNTIME="${ROOT}/eval_runtime_plan1200_20260902_v1"
readonly FULL="${EVAL_RUNTIME}/run_full_reconstructed_eval.py"
readonly TAU800="${ROOT}/runs/plan1200_tau800_corrected_assembly_20260902_v2"
readonly TAU900="${ROOT}/runs/plan1200_tau900_corrected_assembly_20260902_v2"

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${TAU800}/_SUCCESS" && -f "${TAU900}/_SUCCESS" ]]
[[ "$(wc -l < "${TAU800}/main1000/generation.jsonl")" -eq 1000 ]]
[[ "$(wc -l < "${TAU800}/remainder139/generation.jsonl")" -eq 139 ]]
[[ "$(wc -l < "${TAU900}/main1000/generation.jsonl")" -eq 1000 ]]
[[ "$(wc -l < "${TAU900}/remainder139/generation.jsonl")" -eq 139 ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose GPUs}"
[[ "${#gpu[@]}" -eq 4 ]]

cat >"${RUN}/contract.tsv" <<EOF
field	value
schema	plan1200_tau800_tau900_offline_v1
endpoints	tau800,tau900
splits	main1000,remainder139
denominators	1000,139
invalid_cif_excluded	20
main_topup	17
direct	skipped
official_query	false
EOF

run_cell() {
  local slot="$1" steps="$2" split="$3" denominator="$4"
  local source_dir="${ROOT}/runs/plan1200_tau${steps}_corrected_assembly_20260902_v2"
  local cell="${RUN}/tau${steps}/${split}"
  mkdir -p "${cell}"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${gpu[${slot}]}" \
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 H1_ACTIVE_DENOMINATOR="${denominator}" \
    PYTHONPATH="${EVAL_RUNTIME}:${R03E}/runtime" \
    "${PYTHON}" "${FULL}" --arm candidate --repeat 0 \
    --generation-jsonl "${source_dir}/${split}/generation.jsonl" \
    --config "${SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/relax_cache.jsonl" \
    --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/tau${steps}-${split}.out" \
    2>"${RUN}/logs/tau${steps}-${split}.err"
  [[ -f "${cell}/full_reconstructed/_SUCCESS" ]]
}

run_cell 0 800 main1000 1000 & p0=$!
run_cell 1 800 remainder139 139 & p1=$!
run_cell 2 900 main1000 1000 & p2=$!
run_cell 3 900 remainder139 139 & p3=$!
wait "${p0}"
wait "${p1}"
wait "${p2}"
wait "${p3}"

trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"
