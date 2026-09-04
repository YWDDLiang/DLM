#!/usr/bin/env bash
#SBATCH --job-name=spad-tau-cal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=spad-tau-cal-%j.out
#SBATCH --error=spad-tau-cal-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly NATIVE_RUN="${ROOT}/runs/spad_basin_closure_native_s17_39704"
readonly BODY="${NATIVE_RUN}/closure_ce/stream17/body"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_tau_calibration_${SLURM_JOB_ID}"

mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for p in "${NATIVE_RUN}/_SUCCESS" "${BODY}/_SUCCESS" "${BODY}/proposal_graphs.pt" "${REFINER}"; do [[ -f "$p" ]]; done
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ "${#gpu[@]}" -eq 2 ]]

run_tau() {
  local slot="$1" tau="$2"
  local out="${RUN}/tau${tau}"
  mkdir "${out}"
  env CUDA_VISIBLE_DEVICES="${gpu[$slot]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${BODY}/proposal_graphs.pt" --checkpoint "${REFINER}" \
    --output-dir "${out}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed 101117 --seed-by-sample-index --diff-steps "${tau}" --max-proposals 256 \
    >"${RUN}/logs/tau${tau}.out" 2>"${RUN}/logs/tau${tau}.err"
  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${BODY}" --refine-dir "${out}/refine" --output-dir "${out}/generation" \
    --arm candidate --repeat 0 --dlm-seed 91117 --refiner-seed 101117 --denominator 256
  [[ "$(wc -l <"${out}/generation/generation.jsonl")" -eq 256 ]]
  touch "${out}/_SUCCESS"
}

run_tau 0 400 & p0=$!
run_tau 1 600 & p1=$!
status=0; wait "$p0" || status=$?; wait "$p1" || status=$?; (( status == 0 ))
"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1])
(r/'CALIBRATION_GENERATION_FINAL.json').write_text(json.dumps({
 'schema':'spad_low_noise_tau_calibration_generation_v1','taus':[400,600],
 'attempts_per_tau':256,'development_only':True,'seed_by_sample_index':True,
 'selection_retry_replacement':False,'direct_run':False},indent=2,sort_keys=True)+'\n')
PY
trap - ERR; touch "${RUN}/_SUCCESS"
