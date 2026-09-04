#!/usr/bin/env bash
#SBATCH --job-name=spad-basin-pgen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=spad-basin-pgen-%j.out
#SBATCH --error=spad-basin-pgen-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly COHORT="${ROOT}/cohorts/spad_basin_preflight_train128_v1_20260904"
readonly TRAIN_RUN="${ROOT}/runs/spad_basin_closure_ce_39700"
readonly POLICY="$(<"${TRAIN_RUN}/POLICY_PATH")"
readonly CAPABILITY="$(<"${TRAIN_RUN}/CAPABILITY_PATH")"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_basin_preflight_generation_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
for path in "${COHORT}/_SUCCESS" "${COHORT}/plans_for_dlm.jsonl" \
  "${POLICY}/adapter_model.safetensors" "${CAPABILITY}"; do [[ -f "${path}" ]]; done
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 1 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  --model-path "${MODEL}" --checkpoint-path "${POLICY}" \
  --prompt-jsonl "${COHORT}/plans_for_dlm.jsonl" --output-dir "${RUN}/body" \
  --crysllmgen-dir "${CRYS}" --num-samples 128 --batch-size 8 \
  --temperature 0.7 --seed 99217 --preserve-prompt-sample-idx \
  --no-repeat-prompt-records --generation-schedule spad \
  --canonicalize-periodic-alias --pbc-min-distance-mask --spad-basin-closure \
  --spad-basin-closure-capability-json "${CAPABILITY}" \
  >"${RUN}/logs/body.out" 2>"${RUN}/logs/body.err"

[[ -f "${RUN}/body/_SUCCESS" ]]
[[ "$(wc -l <"${RUN}/body/raw_generations.jsonl")" -eq 128 ]]
"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
rows=[json.loads(line) for line in (root/'body'/'raw_generations.jsonl').open() if line.strip()]
metrics=json.load((root/'body'/'sample_metrics.json').open())
assert len(rows)==128
assert metrics['parse_success']==metrics['plan_match_success']==metrics['graph_success']==128
assert all(row.get('spad_basin_closure') is True for row in rows)
assert all(row['spad_basin_closure_metadata']['final_geometry_supported'] is True for row in rows)
report={
 'schema':'spad_basin_preflight_generation_v1','rows':128,'parsed':128,
 'seed':99217,'temperature':0.7,'outcomes_read':False,'model494_or_energy_used':False,
 'one_plan_one_trajectory':True,'selection_retry_replacement':False,
}
(root/'GENERATION_FINAL.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
PY
trap - ERR
touch "${RUN}/_SUCCESS"
