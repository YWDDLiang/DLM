#!/usr/bin/env bash
#SBATCH --job-name=spad-tau-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=spad-tau-eval-%j.out
#SBATCH --error=spad-tau-eval-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly GEN_RUN="${SPAD_TAU_CALIBRATION_RUN:?}"
readonly TAU800_RUN="${SPAD_TAU800_RUN:-}"
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
readonly RUN="${ROOT}/runs/spad_basin_closure_tau_calibration_eval_${SLURM_JOB_ID}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${GEN_RUN}/_SUCCESS" "${OFFICIAL}/completion_SUCCESS" "${BASELINE}" "${RAW_SCREEN}"; do [[ -f "$p" ]]; done
taus=(400 600)
if [[ -n "${TAU800_RUN}" ]]; then
  [[ -f "${TAU800_RUN}/_SUCCESS" ]]
  taus+=(800)
fi
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#gpu[@]}" -eq 4 ]]

generation_for_tau() {
  local tau="$1"
  if [[ "$tau" == 800 ]]; then
    printf '%s\n' "${TAU800_RUN}/generation/generation.jsonl"
  else
    printf '%s\n' "${GEN_RUN}/tau${tau}/generation/generation.jsonl"
  fi
}

for tau in "${taus[@]}"; do
  generation="$(generation_for_tau "$tau")"; input="${RUN}/tau${tau}/input"
  mkdir -p "${input}" "${RUN}/tau${tau}/shards"
  env CUDA_VISIBLE_DEVICES= PYTHONPATH="${R03E}/runtime" "${PYTHON}" - "$generation" "$input" <<'PY'
import pathlib,sys
from crystal_dlm.wqcodiff.crysllmgen import a100_sun
m=a100_sun.prepare_a100_input(generation_jsonl=pathlib.Path(sys.argv[1]),
 output_pt=pathlib.Path(sys.argv[2])/'all_attempts.pt',
 output_manifest=pathlib.Path(sys.argv[2])/'input_manifest.json',expected_attempts=256)
assert m['reconstructed_structures']==256
PY
done

run_shard() {
  local slot="$1" tau="$2" rank="$3"
  local root="${RUN}/tau${tau}"
  env CUDA_VISIBLE_DEVICES="${gpu[$slot]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/scripts/relax_spad_basin_closure_shard.py" \
    --generation-jsonl "$(generation_for_tau "$tau")" \
    --generated-pt "${root}/input/all_attempts.pt" --input-manifest "${root}/input/input_manifest.json" \
    --base-cache "${CACHE}" --frozen-resumable-py "${RESUMABLE}" \
    --output-jsonl "${root}/shards/cache_rank${rank}.jsonl" \
    --report-json "${root}/shards/report_rank${rank}.json" --denominator 256 \
    --shard-rank "${rank}" --shard-count 2 --device cuda \
    >"${RUN}/logs/tau${tau}_rank${rank}.out" 2>"${RUN}/logs/tau${tau}_rank${rank}.err"
}
run_shard 0 400 0 & p0=$!; run_shard 1 400 1 & p1=$!
run_shard 2 600 0 & p2=$!; run_shard 3 600 1 & p3=$!
status=0; for p in "$p0" "$p1" "$p2" "$p3"; do wait "$p" || status=$?; done; (( status == 0 ))
if [[ -n "${TAU800_RUN}" ]]; then
  run_shard 0 800 0 & p4=$!; run_shard 1 800 1 & p5=$!
  status=0; wait "$p4" || status=$?; wait "$p5" || status=$?; (( status == 0 ))
fi

for tau in "${taus[@]}"; do
  cell="${RUN}/tau${tau}/evaluation"; mkdir "${cell}"
  cp --reflink=auto "${CACHE}" "${cell}/relax_cache.jsonl"
  cat "${RUN}/tau${tau}/shards/cache_rank0.jsonl" >>"${cell}/relax_cache.jsonl"
  cat "${RUN}/tau${tau}/shards/cache_rank1.jsonl" >>"${cell}/relax_cache.jsonl"
  env CUDA_VISIBLE_DEVICES="${gpu[0]}" H1_ACTIVE_DENOMINATOR=256 OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTHONPATH="${SOURCE}/eval_runtime:${R03E}/runtime" "${PYTHON}" "${FULL}" \
    --arm candidate --repeat 0 --generation-jsonl "$(generation_for_tau "$tau")" \
    --config "${V3_SOURCE}/CONFIG.json" --r03e-root "${R03E}" \
    --working-relax-cache "${cell}/relax_cache.jsonl" --output-dir "${cell}/full_reconstructed" \
    >"${RUN}/logs/tau${tau}_full.out" 2>"${RUN}/logs/tau${tau}_full.err"
  env H1_ACTIVE_DENOMINATOR=256 PYTHONPATH="${SOURCE}/eval_runtime" "${PYTHON}" \
    "${SOURCE}/scripts/finalize_spad_basin_closure_official.py" \
    --preofficial-labels "${cell}/full_reconstructed/attempt_labels_preofficial.jsonl" \
    --generation-jsonl "$(generation_for_tau "$tau")" \
    --official-cache-dir "${OFFICIAL}" --baseline-official-attempts "${BASELINE}" \
    --raw-screen-json "${RAW_SCREEN}" --endpoint "closure_ce_model494_tau${tau}_stream17" \
    --model494 --model494-tau "${tau}" --output-dir "${RUN}/tau${tau}/official" \
    >"${RUN}/logs/tau${tau}_official.out" 2>"${RUN}/logs/tau${tau}_official.err"
done

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for tau in sorted(int(p.name[3:]) for p in r.glob('tau*') if (p/'official').is_dir()):
 q=json.load(open(r/f'tau{tau}'/'official'/'BASIN_CLOSURE_OFFICIAL_FINAL.json'))
 cells.append({'tau':tau,'counts':q['counts'],'rates':q['rates_all_attempts']})
(r/'CALIBRATION_FINAL.json').write_text(json.dumps({'schema':'spad_low_noise_tau_calibration_final_v1','development_only':True,'cells':cells,'direct_run':False},indent=2,sort_keys=True)+'\n')
PY
trap - ERR; touch "${RUN}/_SUCCESS"
