#!/usr/bin/env bash
#SBATCH --job-name=spad-canonical
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=spad-canonical-%j.out
#SBATCH --error=spad-canonical-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly PLAN_DIR="${SPAD_PLAN_DIR:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_canonical_schedule_raw_${SLURM_JOB_ID}"
readonly PLANS="${RUN}/canonical_program_plans.jsonl"
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

"${PYTHON}" - "${PLAN_DIR}/plans_for_dlm.jsonl" "${PLANS}" <<'PY'
import json,pathlib,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==256
with open(sys.argv[2],'x') as out:
 for row in rows:
  row=dict(row); elements=list(row['plan_state']['elements'])
  row['species_program']=elements
  row['species_program_indices']=list(range(len(elements)))
  row['species_program_source']='canonical_schedule_control'
  out.write(json.dumps(row,sort_keys=True)+'\n')
PY

run_cell() {
  local stream="$1" gpu="$2" seed cell
  [[ "${stream}" == 17 ]] && seed=91117 || seed=92117
  cell="${RUN}/BC/stream${stream}"; mkdir -p "${cell}"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
    --prompt-jsonl "${PLANS}" --output-dir "${cell}/body" \
    --crysllmgen-dir "${CRYS}" --num-samples 256 --batch-size 8 \
    --temperature 0.7 --seed "${seed}" --preserve-prompt-sample-idx \
    --generation-schedule spad --canonicalize-periodic-alias \
    --pbc-min-distance-mask >"${RUN}/logs/BC_${stream}_body.out" \
    2>"${RUN}/logs/BC_${stream}_body.err"
  [[ -f "${cell}/body/_SUCCESS" ]]
  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body" --output-dir "${cell}/generation" \
    --seed "${stream}" --dlm-seed "${seed}" --denominator 256 \
    --source-arm full_axis --arm-label candidate
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=1 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${cell}/generation/generation.jsonl" --gt-csv "${GT}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell}/direct" --eval-model-name mp20 \
    >"${RUN}/logs/BC_${stream}_direct.out" 2>"${RUN}/logs/BC_${stream}_direct.err"
  [[ -f "${cell}/direct/report.json" ]]
}
run_cell 17 "${allocated[0]}" & p0=$!
run_cell 18 "${allocated[1]}" & p1=$!
wait "${p0}"; wait "${p1}"

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for s in (17,18):
 b=json.load(open(r/'BC'/f'stream{s}'/'body/sample_metrics.json'))
 d=json.load(open(r/'BC'/f'stream{s}'/'direct/report.json'))
 cells.append({'arm':'BC','stream':s,'requested':256,'body_parsed':b['parse_success'],
  'graphs':b['graph_success'],'composition_valid':d['comp_valid_count'],
  'structure_valid':d['struct_valid_count'],'direct_joint':d['valid_count']})
(r/'RAW_DIRECT_FINAL.json').write_text(json.dumps({'schema':'spad_canonical_raw_v1','cells':cells},indent=2,sort_keys=True)+'\n')
print(json.dumps(cells,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
