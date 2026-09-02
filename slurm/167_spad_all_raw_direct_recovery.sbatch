#!/usr/bin/env bash
#SBATCH --job-name=spad-all-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=spad-all-direct-%j.out
#SBATCH --error=spad-all-direct-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly B0_RUN="${SPAD_B0_BODY_RUN:?}"
readonly BC_RUN="${SPAD_CANONICAL_BODY_RUN:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly ASSEMBLER="${SOURCE}/scripts/assemble_raw_body_repeat.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_all_raw_direct_${SLURM_JOB_ID}"
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

source_body() {
  local arm="$1" stream="$2"
  case "${arm}" in
    B0) printf '%s\n' "${B0_RUN}/BC/stream${stream}/body" ;;
    BC) printf '%s\n' "${BC_RUN}/BC/stream${stream}/body" ;;
    BP|BR) printf '%s\n' "${B0_RUN}/${arm}/stream${stream}/body" ;;
    *) return 2 ;;
  esac
}

run_cell() {
  local arm="$1" stream="$2" seed body cell
  [[ "${stream}" == 17 ]] && seed=91117 || seed=92117
  body="$(source_body "${arm}" "${stream}")"
  cell="${RUN}/${arm}/stream${stream}"; mkdir -p "${cell}"
  [[ -f "${body}/_SUCCESS" && -f "${body}/sample_metrics.json" ]]
  "${PYTHON}" "${ASSEMBLER}" --body-dir "${body}" \
    --output-dir "${cell}/generation" --seed "${stream}" --dlm-seed "${seed}" \
    --denominator 256 --source-arm full_axis --arm-label candidate
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${cell}/generation/generation.jsonl" --gt-csv "${GT}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell}/direct" --eval-model-name mp20 \
    >"${RUN}/logs/${arm}_${stream}.out" 2>"${RUN}/logs/${arm}_${stream}.err"
  [[ -f "${cell}/direct/report.json" ]]
}

pids=()
for arm in B0 BC BP BR; do
  for stream in 17 18; do
    run_cell "${arm}" "${stream}" & pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${PYTHON}" - "${RUN}" "${B0_RUN}" "${BC_RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); sources={'B0':pathlib.Path(sys.argv[2]),'BC':pathlib.Path(sys.argv[3])}
cells=[]
for arm in ('B0','BC','BP','BR'):
 for stream in (17,18):
  if arm=='B0': body=sources['B0']/'BC'/f'stream{stream}'/'body/sample_metrics.json'
  elif arm=='BC': body=sources['BC']/'BC'/f'stream{stream}'/'body/sample_metrics.json'
  else: body=sources['B0']/arm/f'stream{stream}'/'body/sample_metrics.json'
  b=json.load(open(body)); d=json.load(open(root/arm/f'stream{stream}'/'direct/report.json'))
  cells.append({'arm':arm,'stream':stream,'requested':256,'body_parsed':b['parse_success'],
   'graphs':b['graph_success'],'composition_valid':d['comp_valid_count'],
   'structure_valid':d['struct_valid_count'],'direct_joint':d['valid_count']})
result={'schema':'spad_all_raw_direct_v1','cells':cells,'model494':False,
        'chgnet':False,'selection_retry_replacement':False}
(root/'RAW_DIRECT_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
