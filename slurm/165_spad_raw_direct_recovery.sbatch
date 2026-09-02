#!/usr/bin/env bash
#SBATCH --job-name=spad-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=spad-direct-%j.out
#SBATCH --error=spad-direct-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly BODY_RUN="${SPAD_BODY_RUN:?export completed SPAD body run}"
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_raw_direct_recovery_${SLURM_JOB_ID}"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR

run_cell() {
  local arm="$1" stream="$2" source="${BODY_RUN}/${arm}/stream${stream}"
  local output="${RUN}/${arm}/stream${stream}/direct"
  [[ -f "${source}/body/_SUCCESS" && -f "${source}/generation/_SUCCESS" ]]
  mkdir -p "$(dirname "${output}")"
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=4 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${source}/generation/generation.jsonl" \
    --gt-csv "${GT}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${output}" --eval-model-name mp20 \
    >"${RUN}/logs/${arm}_${stream}.out" 2>"${RUN}/logs/${arm}_${stream}.err"
  [[ -f "${output}/report.json" ]]
}

pids=()
for arm in BC BP BR; do
  for stream in 17 18; do
    run_cell "${arm}" "${stream}" & pids+=("$!")
  done
done
for pid in "${pids[@]}"; do wait "${pid}"; done

"${PYTHON}" - "${RUN}" "${BODY_RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); body_root=pathlib.Path(sys.argv[2]); cells=[]
for arm in ('BC','BP','BR'):
 for stream in (17,18):
  body=json.load(open(body_root/arm/f'stream{stream}'/'body/sample_metrics.json'))
  direct=json.load(open(root/arm/f'stream{stream}'/'direct/report.json'))
  cells.append({'arm':arm,'stream':stream,'requested':256,
   'body_parsed':int(body['parse_success']),'graphs':int(body['graph_success']),
   'composition_valid':int(direct['comp_valid_count']),
   'structure_valid':int(direct['struct_valid_count']),
   'direct_joint':int(direct['valid_count'])})
result={'schema':'spad_decoder_raw_final_v1','source_body_run':str(body_root),
        'cells':cells,'model494':False,'chgnet':False,
        'selection_retry_replacement':False}
(root/'RAW_DIRECT_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
