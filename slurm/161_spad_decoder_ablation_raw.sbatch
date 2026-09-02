#!/usr/bin/env bash
#SBATCH --job-name=spad-raw
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=spad-raw-%j.out
#SBATCH --error=spad-raw-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLAN_DIR="${SPAD_PLAN_DIR:?export completed SPAD plan directory}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392"
readonly PLANS="${PLAN_DIR}/plans_for_dlm.jsonl"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly GT="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly SAMPLER="${SOURCE}/src/scripts/sample_llada_r5_exact_length.py"
readonly ASSEMBLER="${SOURCE}/scripts/assemble_raw_body_repeat.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_decoder_raw_${SLURM_JOB_ID}"

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

for path in "${PLANS}" "${POLICY}/adapter_model.safetensors" \
  "${POLICY}/tokenizer.json" "${SAMPLER}" "${ASSEMBLER}" "${DIRECT}" "${GT}"; do
  [[ -f "${path}" ]]
done
[[ -f "${PLAN_DIR}/_SUCCESS" ]]
[[ "$(wc -l <"${PLANS}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 4 ]]

cat >"${RUN}/contract.tsv" <<EOF
field\tvalue
plans\t${PLANS}
requested_per_cell\t256
cells\tBC,BP,BR x streams17,18
common_geometry\t000/100 aggregation + 0.5A 125-image PBC support
BC\tcanonical exact-plan order
BP\tLlama-pointer anchor-first order
BR\tBP plus suffix-visible anchor backfill
model494\tfalse
CHGNet\tfalse pending raw result
selection_retry_replacement\tfalse
EOF

stream_seed() {
  [[ "$1" == 17 ]] && printf '91117\n' || printf '92117\n'
}

run_direct() {
  local arm="$1" stream="$2" cell="${RUN}/${arm}/stream${stream}"
  env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CRYSLLMGEN_METRICS_NUM_CPUS=4 \
    "${PYTHON}" "${DIRECT}" \
    --generation-jsonl "${cell}/generation/generation.jsonl" \
    --gt-csv "${GT}" \
    --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
    --output-dir "${cell}/direct" --eval-model-name mp20 \
    >"${RUN}/logs/${arm}_${stream}_direct.out" \
    2>"${RUN}/logs/${arm}_${stream}_direct.err"
  [[ -f "${cell}/direct/report.json" ]]
}

run_body() {
  local arm="$1" stream="$2" gpu="$3" schedule="$4" backfill="$5"
  local seed cell
  seed="$(stream_seed "${stream}")"
  cell="${RUN}/${arm}/stream${stream}"
  mkdir -p "${cell}"
  local -a extra=(
    --generation-schedule "${schedule}"
    --canonicalize-periodic-alias
    --pbc-min-distance-mask
  )
  [[ "${backfill}" == true ]] && extra+=(--spad-backfill)
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SAMPLER}" --model-path "${MODEL}" \
    --checkpoint-path "${POLICY}" --prompt-jsonl "${PLANS}" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" \
    --num-samples 256 --batch-size 8 --temperature 0.7 --seed "${seed}" \
    --preserve-prompt-sample-idx "${extra[@]}" \
    >"${RUN}/logs/${arm}_${stream}_body.out" \
    2>"${RUN}/logs/${arm}_${stream}_body.err"
  [[ -f "${cell}/body/_SUCCESS" ]]
  "${PYTHON}" "${ASSEMBLER}" --body-dir "${cell}/body" \
    --output-dir "${cell}/generation" --seed "${stream}" --dlm-seed "${seed}" \
    --denominator 256 --source-arm full_axis --arm-label candidate
  [[ -f "${cell}/generation/_SUCCESS" ]]
}

run_queue() {
  local gpu="$1"; shift
  local -a direct_pids=()
  while [[ "$#" -gt 0 ]]; do
    local arm="$1" stream="$2" schedule="$3" backfill="$4"; shift 4
    run_body "${arm}" "${stream}" "${gpu}" "${schedule}" "${backfill}"
    run_direct "${arm}" "${stream}" & direct_pids+=("$!")
  done
  local pid
  for pid in "${direct_pids[@]}"; do wait "${pid}"; done
}

run_queue "${allocated[0]}" BC 17 exact-plan false BC 18 exact-plan false & q0=$!
run_queue "${allocated[1]}" BP 17 spad false BP 18 spad false & q1=$!
run_queue "${allocated[2]}" BR 17 spad true & q2=$!
run_queue "${allocated[3]}" BR 18 spad true & q3=$!
wait "${q0}"; wait "${q1}"; wait "${q2}"; wait "${q3}"

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cells=[]
for arm in ('BC','BP','BR'):
 for stream in (17,18):
  cell=root/arm/f'stream{stream}'
  body=json.load(open(cell/'body/sample_metrics.json'))
  direct=json.load(open(cell/'direct/report.json'))
  cells.append({'arm':arm,'stream':stream,'requested':256,
   'body_parsed':int(body['parse_success']),'graphs':int(body['graph_success']),
   'composition_valid':int(direct['comp_valid_count']),
   'structure_valid':int(direct['struct_valid_count']),
   'direct_joint':int(direct['valid_count'])})
assert len(cells)==6 and all(x['requested']==256 for x in cells)
result={'schema':'spad_decoder_raw_final_v1','cells':cells,'model494':False,
        'chgnet':False,'selection_retry_replacement':False}
(root/'RAW_DIRECT_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY

trap - ERR
touch "${RUN}/_RAW_SUCCESS"
printf '%s\n' "${RUN}"
