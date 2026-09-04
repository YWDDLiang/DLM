#!/usr/bin/env bash
#SBATCH --job-name=spad-tau-eval2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=spad-tau-eval2-%j.out
#SBATCH --error=spad-tau-eval2-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly GEN_RUN="${ROOT}/runs/spad_basin_closure_tau_calibration_39737"
readonly TAU800_RUN="${ROOT}/runs/spad_basin_closure_tau800_s17_39747"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_SOURCE="${PROJECT}/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly CACHE="${PROJECT}/data/a100_eval_sun_cache/chgnet_relax_cache_global.jsonl"
readonly RESUMABLE="${PROJECT}/reference/a100_eval_sun/eval_sun_resumable.py"
readonly FULL="${SOURCE}/eval_runtime/run_full_reconstructed_eval.py"
readonly OFFICIAL="${ROOT}/runs/spad_basin_closure_official_20260904_v1/official_mp_cache"
readonly BASELINE="${ROOT}/runs/spad_basin_closure_official_final_20260904_v1/attempt_results_official.jsonl"
readonly RAW_SCREEN="${ROOT}/runs/spad_basin_closure_native_raw_screen_39710/screen/RAW_SCREEN_FINAL.json"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_tau_calibration_eval2_${SLURM_JOB_ID}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${GEN_RUN}/_SUCCESS" "${TAU800_RUN}/_SUCCESS" "${OFFICIAL}/completion_SUCCESS" "${BASELINE}" "${RAW_SCREEN}"; do [[ -f "$p" ]]; done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#gpu[@]}" -eq 1 ]]

generation_for_tau() {
  local tau="$1"
  [[ "$tau" == 800 ]] && printf '%s\n' "${TAU800_RUN}/generation/generation.jsonl" || printf '%s\n' "${GEN_RUN}/tau${tau}/generation/generation.jsonl"
}

run_shard() {
  local slot="$1" tau="$2" rank="$3"
  local cell="${RUN}/tau${tau}"
  env CUDA_VISIBLE_DEVICES="${gpu[$slot]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
   MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
   "${PYTHON}" "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
   --generation-jsonl "$(generation_for_tau "$tau")" \
   --generated-pt "${cell}/input/all_attempts.pt" --input-manifest "${cell}/input/input_manifest.json" \
   --base-cache "${CACHE}" --frozen-resumable-py "${RESUMABLE}" \
   --output-jsonl "${cell}/shards/cache_rank${rank}.jsonl" \
   --report-json "${cell}/shards/report_rank${rank}.json" --denominator 256 \
   --shard-rank "$rank" --shard-count 1 --device cuda \
   >"${RUN}/logs/tau${tau}_rank${rank}.out" 2>"${RUN}/logs/tau${tau}_rank${rank}.err"
}

for tau in 400 600 800; do
  generation="$(generation_for_tau "$tau")"; cell="${RUN}/tau${tau}"
  mkdir -p "${cell}/input" "${cell}/shards"
  env CUDA_VISIBLE_DEVICES= PYTHONPATH="${R03E}/runtime" "${PYTHON}" - "$generation" "${cell}/input" <<'PY'
import pathlib,sys
from crystal_dlm.wqcodiff.crysllmgen import a100_sun
m=a100_sun.prepare_a100_input(generation_jsonl=pathlib.Path(sys.argv[1]), output_pt=pathlib.Path(sys.argv[2])/'all_attempts.pt', output_manifest=pathlib.Path(sys.argv[2])/'input_manifest.json', expected_attempts=256)
assert m['total_attempts']==256
PY
  run_shard 0 "$tau" 0
  mkdir "${cell}/evaluation"
  cp --reflink=auto "${CACHE}" "${cell}/evaluation/relax_cache.jsonl"
  cat "${cell}/shards/cache_rank0.jsonl" >>"${cell}/evaluation/relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${gpu[0]}" H1_ACTIVE_DENOMINATOR=256 OPENBLAS_NUM_THREADS=1 \
   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
   PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" "${PYTHON}" "${FULL}" \
   --arm candidate --repeat 0 --generation-jsonl "$generation" --config "${V3_SOURCE}/CONFIG.json" \
   --r03e-root "${R03E}" --working-relax-cache "${cell}/evaluation/relax_cache.jsonl" \
   --output-dir "${cell}/evaluation/full_reconstructed" \
   >"${RUN}/logs/tau${tau}_full.out" 2>"${RUN}/logs/tau${tau}_full.err"
  env H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime" "${PYTHON}" \
   "${SOURCE}/scripts/finalize_spad_basin_closure_official.py" \
   --preofficial-labels "${cell}/evaluation/full_reconstructed/attempt_labels_preofficial.jsonl" \
   --generation-jsonl "$generation" --official-cache-dir "${OFFICIAL}" \
   --baseline-official-attempts "${BASELINE}" --raw-screen-json "${RAW_SCREEN}" \
   --endpoint "closure_ce_model494_tau${tau}_stream17" --model494 --model494-tau "$tau" \
   --output-dir "${cell}/official" >"${RUN}/logs/tau${tau}_official.out" 2>"${RUN}/logs/tau${tau}_official.err"
done

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for tau in (400,600,800):
 q=json.load(open(r/f'tau{tau}'/'official'/'BASIN_CLOSURE_OFFICIAL_FINAL.json'))
 cells.append({'tau':tau,'counts':q['counts'],'rates':q['rates_all_attempts']})
(r/'CALIBRATION_FINAL.json').write_text(json.dumps({'schema':'spad_low_noise_tau_calibration_final_v1','development_only':True,'cells':cells,'direct_run':False,'automatic_selection':False},indent=2,sort_keys=True)+'\n')
PY
trap - ERR; touch "${RUN}/_SUCCESS"
