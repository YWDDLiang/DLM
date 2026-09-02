#!/usr/bin/env bash
#SBATCH --job-name=spad-model494
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=spad-model494-%j.out
#SBATCH --error=spad-model494-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly BC_BODY_RUN="${SPAD_BC_BODY_RUN:?}"
readonly BR_BODY_RUN="${SPAD_BR_BODY_RUN:?}"
readonly BS_BODY_RUN="${SPAD_BS_BODY_RUN:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_model494_${SLURM_JOB_ID}"
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 4 ]]

body_dir() {
  local arm="$1" stream="$2"
  case "${arm}" in
    BC) printf '%s\n' "${BC_BODY_RUN}/BC/stream${stream}/body" ;;
    BR) printf '%s\n' "${BR_BODY_RUN}/BR/stream${stream}/body" ;;
    BS) printf '%s\n' "${BS_BODY_RUN}/BS/stream${stream}/body" ;;
    *) return 2 ;;
  esac
}

run_refine() {
  local arm="$1" stream="$2" gpu="$3" body output seed
  [[ "${stream}" == 17 ]] && seed=101117 || seed=102117
  body="$(body_dir "${arm}" "${stream}")"; output="${RUN}/${arm}/stream${stream}"
  [[ -f "${body}/_SUCCESS" && -f "${body}/proposal_graphs.pt" ]]
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${body}/proposal_graphs.pt" --checkpoint "${REFINER}" \
    --output-dir "${output}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${seed}" --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${RUN}/logs/${arm}_${stream}.out" 2>"${RUN}/logs/${arm}_${stream}.err"
  local metrics result
  metrics="${output}/refine/refinement_metrics.json"; [[ -f "${metrics}" ]]
  result="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "${metrics}")"
  [[ -f "${result}" ]]
  "${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
    --body-dir "${body}" --refine-dir "${output}/refine" \
    --output-dir "${output}/generation" --arm candidate --repeat 0 \
    --dlm-seed "$([[ "${stream}" == 17 ]] && echo 91117 || echo 92117)" \
    --refiner-seed "${seed}" --denominator 256
  touch "${output}/_SUCCESS"
}

run_queue() {
  local gpu="$1"; shift
  while [[ "$#" -gt 0 ]]; do
    run_refine "$1" "$2" "${gpu}"; shift 2
  done
}
run_queue "${allocated[0]}" BS 17 BC 17 & p0=$!
run_queue "${allocated[1]}" BS 18 BC 18 & p1=$!
run_queue "${allocated[2]}" BR 17 & p2=$!
run_queue "${allocated[3]}" BR 18 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for arm in ('BC','BR','BS'):
 for s in (17,18):
  m=json.load(open(r/arm/f'stream{s}'/'refine/refinement_metrics.json'))
  cells.append({'arm':arm,'stream':s,'requested':256,
                'refined_reported':m.get('num_output_structures',m.get('output_count')),
                'output_file':m['output_file']})
(r/'REFINEMENT_FINAL.json').write_text(json.dumps({'cells':cells,'tau':800},indent=2,sort_keys=True)+'\n')
print(json.dumps(cells,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
