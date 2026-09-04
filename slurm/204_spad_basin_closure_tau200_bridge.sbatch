#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-tau200
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=spad-basin-tau200-%j.out
#SBATCH --error=spad-basin-tau200-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly NATIVE_RUN="${ROOT}/runs/spad_basin_closure_native_s17_39704"
readonly BODY="${NATIVE_RUN}/closure_ce/stream17/body"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_closure_tau200_s17_${SLURM_JOB_ID}"
readonly DLM_SEED=91117
readonly REFINER_SEED=101117

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for path in "${NATIVE_RUN}/_SUCCESS" "${BODY}/_SUCCESS" \
  "${BODY}/proposal_graphs.pt" "${REFINER}"; do [[ -f "${path}" ]]; done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${BODY}/proposal_graphs.pt" --checkpoint "${REFINER}" \
  --output-dir "${RUN}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
  --seed "${REFINER_SEED}" --seed-by-sample-index --diff-steps 200 \
  --max-proposals 256 >"${RUN}/logs/refine.out" 2>"${RUN}/logs/refine.err"
[[ -f "${RUN}/refine/refinement_metrics.json" ]]

"${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
  --body-dir "${BODY}" --refine-dir "${RUN}/refine" \
  --output-dir "${RUN}/generation" --arm candidate --repeat 0 \
  --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" --denominator 256
[[ "$(wc -l <"${RUN}/generation/generation.jsonl")" -eq 256 ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
metrics=json.load((root/'refine'/'refinement_metrics.json').open())
rows=[json.loads(line) for line in (root/'generation'/'generation.jsonl').open() if line.strip()]
assert len(rows)==256
report={
 'schema':'spad_basin_closure_tau200_bridge_v1','tau':200,'stream':17,
 'requested':256,'generation_rows':256,
 'refined_reported':metrics.get('num_output_structures',metrics.get('output_count')),
 'seed_by_sample_index':True,'one_plan_one_trajectory':True,
 'selection_retry_replacement':False,'bridge_role':'fixed_low_noise_system_fallback',
}
(root/'REFINEMENT_FINAL.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
PY
trap - ERR
touch "${RUN}/_SUCCESS"
