#!/usr/bin/env bash
#SBATCH --job-name=spad-bs-body
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=spad-bs-body-%j.out
#SBATCH --error=spad-bs-body-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly POLICY="${SPAD_POLICY_PATH:?export completed BS checkpoint}"
readonly PLAN_DIR="${SPAD_PLAN_DIR:?export frozen SPAD plan directory}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_bs_body_${SLURM_JOB_ID}"
mkdir "${RUN}"; mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${POLICY}/adapter_model.safetensors" ]]
[[ -f "${PLAN_DIR}/plans_for_dlm.jsonl" && -f "${PLAN_DIR}/_SUCCESS" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]

run_stream() {
  local stream="$1" gpu="$2" seed cell
  [[ "${stream}" == 17 ]] && seed=91117 || seed=92117
  cell="${RUN}/BS/stream${stream}"; mkdir -p "${cell}"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
    "${PYTHON}" "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
    --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
    --prompt-jsonl "${PLAN_DIR}/plans_for_dlm.jsonl" \
    --output-dir "${cell}/body" --crysllmgen-dir "${CRYS}" \
    --num-samples 256 --batch-size 8 --temperature 0.7 --seed "${seed}" \
    --preserve-prompt-sample-idx --generation-schedule spad --spad-backfill \
    --canonicalize-periodic-alias --pbc-min-distance-mask \
    >"${RUN}/logs/BS_${stream}.out" 2>"${RUN}/logs/BS_${stream}.err"
  [[ -f "${cell}/body/_SUCCESS" ]]
  "${PYTHON}" "${SOURCE}/scripts/assemble_raw_body_repeat.py" \
    --body-dir "${cell}/body" --output-dir "${cell}/generation" \
    --seed "${stream}" --dlm-seed "${seed}" --denominator 256 \
    --source-arm full_axis --arm-label candidate
}
run_stream 17 "${allocated[0]}" & p0=$!
run_stream 18 "${allocated[1]}" & p1=$!
wait "${p0}"; wait "${p1}"
"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); cells=[]
for s in (17,18):
 x=json.load(open(r/'BS'/f'stream{s}'/'body/sample_metrics.json'))
 cells.append({'arm':'BS','stream':s,'requested':256,'parsed':x['parse_success'],
               'plan_match':x['plan_match_success'],'graphs':x['graph_success']})
(r/'BODY_FINAL.json').write_text(json.dumps({'cells':cells},indent=2,sort_keys=True)+'\n')
print(json.dumps(cells,sort_keys=True))
PY
trap - ERR
touch "${RUN}/_SUCCESS"
