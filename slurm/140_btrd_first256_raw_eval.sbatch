#!/usr/bin/env bash
#SBATCH --job-name=btrd-first256-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=05:00:00
#SBATCH --output=btrd-first256-eval-%j.out
#SBATCH --error=btrd-first256-eval-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export root}"
readonly CODE_ROOT="${H1A2_CODE_ROOT:?export code root}"
readonly BASE="${ROOT}/runs/g2_known1200_baseline_fast_39183"
readonly BTRD="${BTRD_EVAL_CANDIDATE_RUN:-${ROOT}/runs/btrd_known1200_fast_39188}"
readonly SPLIT="${ROOT}/cohorts/btrd_eval_official_known_dev256_confirm903_v1_20260902"
readonly RUN_PREFIX="${BTRD_EVAL_RUN_PREFIX:-btrd_first256_raw_eval}"
readonly RUN="${ROOT}/runs/${RUN_PREFIX}_${SLURM_JOB_ID}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly EXTRACT="${CODE_ROOT}/scripts/freeze_btrd_first256_generations.py"
readonly FAST="${CODE_ROOT}/scripts/run_direct_validity_fast.py"
readonly FULL="${ROOT}/eval_runtime/run_full_reconstructed_eval.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

mkdir "${RUN}"; mkdir "${RUN}/logs"
on_error(){ rc="$1"; line="$2"; cmd="$3"; set +e; printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' "$rc" "$line" "$cmd" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"; }
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
[[ -f "${BASE}/_SUCCESS" && -f "${BTRD}/_SUCCESS" && -f "${SPLIT}/_SUCCESS" ]]
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"; [[ "${#gpu[@]}" -eq 2 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	btrd_first256_raw_eval_v1
arms	BASE,BTRD-tau200
requested_per_arm	256
body_rerun	false
candidate_model494	false
direct	fast_validity_only
chgnet	raw_both_arms
official_query	false_cached_reference_only
EOF

PYTHONPATH="${CODE_ROOT}/src" "${PYTHON}" "${EXTRACT}" \
 --baseline-run "${BASE}" --candidate-run "${BTRD}" \
 --plans-jsonl "${SPLIT}/development256/plans_for_dlm.jsonl" \
 --output-dir "${RUN}/matched"

run_cell(){
 local slot="$1" arm="$2" label="$3"
 local cell="${RUN}/${arm}"
 mkdir "${cell}"; mkdir "${cell}/evaluation"
 cp --reflink=auto "${RUN}/matched/${arm}/generation.jsonl" "${cell}/generation.jsonl"
 cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
 env SLURM_JOB_ID="${SLURM_JOB_ID}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${PYTHON}" "${FAST}" --generation-jsonl "${cell}/generation.jsonl" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${cell}/evaluation/direct" --expected-denominator 256 \
  >"${RUN}/logs/${arm}-direct.out" 2>"${RUN}/logs/${arm}-direct.err" & dpid=$!
 env CUDA_VISIBLE_DEVICES="${gpu[$slot]}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${ROOT}/eval_runtime:${R03E}/runtime" \
  "${PYTHON}" "${FULL}" --arm "${label}" --repeat 0 \
  --generation-jsonl "${cell}/generation.jsonl" --config "${V3_SOURCE}/CONFIG.json" \
  --r03e-root "${R03E}" --working-relax-cache "${cell}/relax_cache.jsonl" \
  --output-dir "${cell}/evaluation/full_reconstructed" \
  >"${RUN}/logs/${arm}-full.out" 2>"${RUN}/logs/${arm}-full.err"
 wait "${dpid}"
 [[ -f "${cell}/evaluation/direct/_SUCCESS" && -f "${cell}/evaluation/full_reconstructed/_SUCCESS" ]]
}
run_cell 0 BASE control & p0=$!
run_cell 1 BTRD candidate & p1=$!
wait "${p0}"; wait "${p1}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_OFFLINE_SUCCESS"
printf '%s\n' "${RUN}"
