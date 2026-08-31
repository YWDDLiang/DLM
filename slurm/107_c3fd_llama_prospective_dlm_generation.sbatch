#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-dlm-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=c3fd-llama-dlm-gen-%j.out
#SBATCH --error=c3fd-llama-dlm-gen-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly PLAN_RUN="${C3FD_LLAMA_PLAN_RUN:?export completed prospective Plan run}"
readonly FREEZE_RUN="${C3FD_LLAMA_FREEZE_RUN:?export completed prospective freeze run}"
readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly RICH_DLM="${REPO}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/c3fd_llama_prospective_dlm_generation_${SLURM_JOB_ID}"
readonly RICH_ADAPTER_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA=e2f5a84f02983828d5551bef15269f20e2dcea8dbcca87ae45ed7a9254a9395b
readonly SGTC_CORE_SHA=eea5e1fb52eafcbd456651ede728978d94d2545161db679013df81cbbd5a2561
readonly REFINER_SCRIPT_SHA=c509d1a631fee084247b17a16197ce2445ae7b76f614c2bdcbd3cea2535e3498

mkdir "${RUN}"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${PLAN_RUN}/_SUCCESS" && -f "${FREEZE_RUN}/_SUCCESS" ]]
for route in F M; do
  [[ -f "${PLAN_RUN}/route_${route}/body_plans_fixed256.jsonl" ]]
  [[ "$(wc -l <"${PLAN_RUN}/route_${route}/body_plans_fixed256.jsonl")" -eq 256 ]]
done
for spec in \
  "${RICH_ADAPTER_SHA}:${RICH_DLM}/adapter_model.safetensors" \
  "${REFINER_SHA}:${REFINER}" \
  "${SAMPLER_SHA}:${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${SGTC_CORE_SHA}:${ROOT}/src/crystal_dlm/sgtc_sampling.py" \
  "${REFINER_SCRIPT_SHA}:${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py"; do
  expected="${spec%%:*}"; path="${spec#*:}"
  [[ -f "${path}" && "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
done

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_prospective_dlm_generation_v1
routes	F,M
streams	17,18
denominator	256
planner_failures	preserved_without_DLM_call
dlm	old_H1A2_rich_DLM
temperature	0.7
schedule	exact_axis
refiner	model494_tau800
one_trajectory	true
retry_replacement_rerank	false
official_query	false
EOF

run_cell() {
  local slot="$1" route="$2" stream="$3"
  local device dlm_seed refiner_seed output metrics output_file
  device="${allocated_gpus[${slot}]}"
  if [[ "${stream}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117; else dlm_seed=92117; refiner_seed=102117; fi
  output="${RUN}/stream${stream}/${route}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" --checkpoint-path "${RICH_DLM}" \
    --prompt-jsonl "${PLAN_RUN}/route_${route}/body_plans_fixed256.jsonl" \
    --output-dir "${output}/body" --crysllmgen-dir "${CRYS}" \
    --num-samples 256 --temperature 0.7 --seed "${dlm_seed}" \
    --allow-missing-plans >"${output}/body.out" 2>"${output}/body.err"
  [[ -f "${output}/body/_SUCCESS" ]]
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" \
    --checkpoint "${REFINER}" --output-dir "${output}/refine" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${refiner_seed}" \
    --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${output}/refine.out" 2>"${output}/refine.err"
  metrics="${output}/refine/refinement_metrics.json"
  [[ -f "${metrics}" ]]
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]]
  touch "${output}/refine/_SUCCESS" "${output}/_SUCCESS"
}

run_cell 0 F 17 & p0=$!
run_cell 1 M 17 & p1=$!
run_cell 2 F 18 & p2=$!
run_cell 3 M 18 & p3=$!
rc=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || rc=$?; done
[[ "${rc}" -eq 0 ]]

for stream in 17 18; do for route in F M; do [[ -f "${RUN}/stream${stream}/${route}/_SUCCESS" ]]; done; done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
