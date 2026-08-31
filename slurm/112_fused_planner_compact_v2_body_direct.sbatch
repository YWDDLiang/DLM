#!/usr/bin/env bash
#SBATCH --job-name=fused-v2-body-direct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=fused-v2-body-direct-%j.out
#SBATCH --error=fused-v2-body-direct-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly POLICY="${ROOT}/runs/c3fd_native_teacher_sft_38703/train_seed82017/checkpoints/step-3392"
readonly PLANS="${ROOT}/runs/c3fd_llama_typed_plans_39088/plans/plans_for_dlm.jsonl"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly V3_RUN="${PROJECT}/runs/20260814_h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3"
readonly R03E="${V3_RUN}/frozen/best/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1"
readonly GT_CSV="${PROJECT}/reference/crysllmgen/data/mp_20/test.csv"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly BODY_SAMPLER="${ROOT}/src/scripts/sample_sgtc_l6.py"
readonly ASSEMBLER="${ROOT}/scripts/assemble_raw_body_repeat.py"
readonly DIRECT="${R03E}/runtime/scripts/a800/run_crysllmgen_metrics.py"
readonly RUN="${ROOT}/runs/fused_planner_compact_v2_body_direct_${SLURM_JOB_ID}"

readonly PLANS_SHA=f7002caa9f5dcfb3bdb962028673e51da108d58854519a145ceedfa3109728ec
readonly POLICY_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20
readonly POLICY_CONFIG_SHA=61ff08995286cf7679fb2e494064892a778a925becd2d55300a70e354f8ff240
readonly BODY_SAMPLER_SHA=e2f5a84f02983828d5551bef15269f20e2dcea8dbcca87ae45ed7a9254a9395b
readonly ASSEMBLER_SHA=8e9e9380eb2b2ca9e833faacb242e04ad0b34f897db69f5c3fca3cce3557d6ec
readonly DIRECT_SHA=556d4e14c96a7a89c96eaf7c07fda04c0f399526f733de0e8cbbd08d1f24e7a4
readonly GT_SHA=b4a2ae467ccab230a1b726778bf1539e3e89ecccaa6925a7c0eeeced46966c7c

mkdir "${RUN}"
mkdir -p "${RUN}/body" "${RUN}/evaluation/generation" "${RUN}/evaluation/direct"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'exit_code\t%s\nline\t%s\ncommand\t%s\n' "${rc}" "${line}" "${command}" \
    >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

for spec in \
  "${PLANS_SHA}:${PLANS}" \
  "${POLICY_SHA}:${POLICY}/adapter_model.safetensors" \
  "${POLICY_CONFIG_SHA}:${POLICY}/adapter_config.json" \
  "${BODY_SAMPLER_SHA}:${BODY_SAMPLER}" \
  "${ASSEMBLER_SHA}:${ASSEMBLER}" \
  "${DIRECT_SHA}:${DIRECT}" \
  "${GT_SHA}:${GT_CSV}"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done
[[ -f "$(dirname "${PLANS}")/_SUCCESS" ]]
[[ "$(wc -l <"${PLANS}")" -eq 256 ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU}"
[[ "${#allocated[@]}" -eq 1 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	fused_planner_compact_v2_body_direct_v1
planner_job	39088
planner_comp_valid	256/256
dlm_checkpoint	compact_v2_job38703_seed82017_step3392
stream	17
dlm_seed	91117
requested	256
temperature	0.7
schedule	exact_axis
trajectory_per_plan	1
model494	false
chgnet	false
official_query	false
retry_filter_replacement_rerank_best_of_n	false
EOF
sha256sum "${PLANS}" "${POLICY}/adapter_model.safetensors" \
  "${POLICY}/adapter_config.json" "${BODY_SAMPLER}" "${ASSEMBLER}" \
  "${DIRECT}" "${GT_CSV}" >"${RUN}/INPUTS.sha256"

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=4 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  "${PYTHON}" "${BODY_SAMPLER}" --model-path "${MODEL}" \
  --checkpoint-path "${POLICY}" --prompt-jsonl "${PLANS}" \
  --output-dir "${RUN}/body" --crysllmgen-dir "${CRYS}" \
  --num-samples 256 --temperature 0.7 --seed 91117 \
  >"${RUN}/body.out" 2>"${RUN}/body.err"
[[ -f "${RUN}/body/_SUCCESS" ]]

"${PYTHON}" "${ASSEMBLER}" --body-dir "${RUN}/body" \
  --output-dir "${RUN}/evaluation/generation" --seed 17 --dlm-seed 91117 \
  --denominator 256 --source-arm full_axis --arm-label candidate

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 CRYSLLMGEN_METRICS_NUM_CPUS=8 \
  "${PYTHON}" "${DIRECT}" \
  --generation-jsonl "${RUN}/evaluation/generation/generation.jsonl" \
  --gt-csv "${GT_CSV}" \
  --snapshot-root "${R03E}/runtime/crystal_dlm/wqcodiff/crysllmgen/upstream" \
  --output-dir "${RUN}/evaluation/direct" --eval-model-name mp20 \
  >"${RUN}/direct.out" 2>"${RUN}/direct.err"
[[ -f "${RUN}/evaluation/direct/report.json" ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); report=json.load(open(root/'evaluation/direct/report.json'))
result={
 'schema':'fused_planner_compact_v2_body_direct_final_v1',
 'requested':256,
 'generation_succeeded':int(report['generation_succeeded']),
 'comp_valid':int(report['comp_valid_count']),
 'struct_valid':int(report['struct_valid_count']),
 'direct_joint':int(report['valid_count']),
 'retry_or_replacement_used':bool(report['retry_or_replacement_used']),
}
assert report['denominator']==256 and report['attempts']==256
assert result['retry_or_replacement_used'] is False
(root/'BODY_DIRECT_FINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
PY

find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
