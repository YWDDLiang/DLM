#!/usr/bin/env bash
#SBATCH --job-name=c3fd-h1a2-fusion-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=600G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-h1a2-fusion-gen-%j.out
#SBATCH --error=c3fd-h1a2-fusion-gen-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly H1A2_DLM="${REPO}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
readonly SFT_RUN="${ROOT}/runs/c3fd_native_teacher_sft_38703"
readonly POLICY17="${SFT_RUN}/train_seed82017/checkpoints/step-3392"
readonly POLICY18="${SFT_RUN}/train_seed82018/checkpoints/step-3392"
readonly COHORT="${ROOT}/cohorts/c3fd_h1a2_fusion_prospective_seed20_256_v1_20260831"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly RUN="${ROOT}/runs/c3fd_h1a2_fusion_prospective_generation_${SLURM_JOB_ID}"
readonly MANIFEST_SHA=4e612fcdcf76e3b5ac1f0750e4ec773877d58dd6f75e2a217cb73798cf71811e
readonly H1A2_PLANS_SHA=bbadae4ca2ae164a9aa12ccb29a9ce52dbf6acad98037fb2e757098d2ec01cf8
readonly V2_PLANS_SHA=9700462198c3bc5ab5252cd200e55b9fc7dc13e41ac9bd6f740e330c6a3af2a9
readonly LEDGER_SHA=5189fb8e19958425c3c33871b95fb5046ba147fc5b9bd3a30dad9ef2982f35df
readonly H1A2_CONFIG_SHA=2b69a62cc62fdb89ecc8178a6a6fdd6bc03cac702d7bcef2dd282ea7a0bca868
readonly H1A2_MODEL_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d
readonly POLICY_CONFIG_SHA=61ff08995286cf7679fb2e494064892a778a925becd2d55300a70e354f8ff240
readonly POLICY17_SHA=06cd5465d1c12dc70c695a12602a4bd99f6c5561f2dc44daa431aadeef962b20
readonly POLICY18_SHA=7f22580a78b4823b44ee841a25cce9fd39541bb0a2192171b0adb4788725526c
readonly REFINER_SHA=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
readonly SAMPLER_SHA=51e5bc0303778df9382052b2cae65d9922c74efd4136482788bd2d2ea0c44ee8
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

for path in "${MODEL}" "${H1A2_DLM}" "${SFT_RUN}/_SUCCESS" "${COHORT}/_SUCCESS" \
  "${POLICY17}" "${POLICY18}" "${REFINER}" "${CRYS}"; do
  [[ -e "${path}" ]] || exit 2
done
[[ "$(sha256sum "${COHORT}/manifest.json" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/H1A2_FULL.jsonl" | awk '{print $1}')" == "${H1A2_PLANS_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/C3FD_V2.jsonl" | awk '{print $1}')" == "${V2_PLANS_SHA}" ]] || exit 2
[[ "$(sha256sum "${COHORT}/ledger.jsonl" | awk '{print $1}')" == "${LEDGER_SHA}" ]] || exit 2
[[ "$(sha256sum "${H1A2_DLM}/adapter_config.json" | awk '{print $1}')" == "${H1A2_CONFIG_SHA}" ]] || exit 2
[[ "$(sha256sum "${H1A2_DLM}/adapter_model.safetensors" | awk '{print $1}')" == "${H1A2_MODEL_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY17}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY17_SHA}" ]] || exit 2
[[ "$(sha256sum "${POLICY18}/adapter_model.safetensors" | awk '{print $1}')" == "${POLICY18_SHA}" ]] || exit 2
for config in "${POLICY17}/adapter_config.json" "${POLICY18}/adapter_config.json"; do
  [[ "$(sha256sum "${config}" | awk '{print $1}')" == "${POLICY_CONFIG_SHA}" ]] || exit 2
done
[[ "$(sha256sum "${REFINER}" | awk '{print $1}')" == "${REFINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/sample_sgtc_l6.py" | awk '{print $1}')" == "${SAMPLER_SHA}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" | awk '{print $1}')" == "${REFINER_SCRIPT_SHA}" ]] || exit 2

{
  printf 'field\tvalue\n'
  printf 'schema\tc3fd_h1a2_fusion_prospective_generation_v1\n'
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'cohort_manifest\t%s\n' "${MANIFEST_SHA}"
  printf 'arms\th1a2_fusion,v2_seed82017,v2_seed82018\n'
  printf 'streams\t17,18\n'
  printf 'h1a2_fusion_prompt\tcanonical_full_H1A2_rich_JSON\n'
  printf 'h1a2_fusion_dlm\tlegacy_H1A2_rich_checkpoint\n'
  printf 'v2_prompts\tC3FD_NATIVE_PLAN_V2\n'
  printf 'alignment_training\tfalse_stopped_by_user\n'
  printf 'dlm_seeds\t91117,92117\n'
  printf 'refiner_seeds\t101117,102117\n'
  printf 'temperature\t0.7\n'
  printf 'schedule\texact_axis\n'
  printf 'refiner_tau\t800\n'
  printf 'denominator\t256\n'
  printf 'attempts_per_plan\t1\n'
  printf 'retry_rerank_replacement\tfalse\n'
  printf 'official_query\tdeferred_until_offline_union\n'
} >"${RUN}/scientific_contract.tsv"
sha256sum "${COHORT}/manifest.json" "${COHORT}/H1A2_FULL.jsonl" \
  "${COHORT}/C3FD_V2.jsonl" "${COHORT}/ledger.jsonl" \
  "${H1A2_DLM}/adapter_config.json" "${H1A2_DLM}/adapter_model.safetensors" \
  "${POLICY17}/adapter_config.json" "${POLICY17}/adapter_model.safetensors" \
  "${POLICY18}/adapter_config.json" "${POLICY18}/adapter_model.safetensors" \
  "${REFINER}" "${ROOT}/src/scripts/sample_sgtc_l6.py" \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" >"${RUN}/INPUTS.sha256"

readonly ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
IFS=',' read -r -a GPU_DEVICES <<<"${ALLOCATED_GPU_LIST}"
[[ "${#GPU_DEVICES[@]}" -eq 6 ]] || exit 2

run_cell() {
  local slot="$1" stream="$2" arm="$3" checkpoint="$4" plans="$5"
  local device dlm_seed refiner_seed output metrics output_file
  device="${GPU_DEVICES[${slot}]}"
  if [[ "${stream}" == 17 ]]; then dlm_seed=91117; refiner_seed=101117; else dlm_seed=92117; refiner_seed=102117; fi
  output="${RUN}/stream${stream}/${arm}"
  mkdir -p "${output}"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/sample_sgtc_l6.py" \
    --model-path "${MODEL}" --checkpoint-path "${checkpoint}" --prompt-jsonl "${plans}" \
    --output-dir "${output}/body" --crysllmgen-dir "${CRYS}" --num-samples 256 \
    --temperature 0.7 --seed "${dlm_seed}" >"${output}/body.out" 2>"${output}/body.err"
  env CUDA_VISIBLE_DEVICES="${device}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    python "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${output}/body/proposal_graphs.pt" --checkpoint "${REFINER}" \
    --output-dir "${output}/refine" --crysllmgen-dir "${CRYS}" --batch-size 1 \
    --seed "${refiner_seed}" --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${output}/refine.out" 2>"${output}/refine.err"
  [[ -f "${output}/body/_SUCCESS" ]] || exit 3
  metrics="${output}/refine/refinement_metrics.json"; [[ -f "${metrics}" ]] || exit 3
  output_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["output_file"])' "${metrics}")"
  [[ -f "${output_file}" ]] || exit 3
  touch "${output}/refine/_SUCCESS" "${output}/_SUCCESS"
}

run_cell 0 17 h1a2_fusion "${H1A2_DLM}" "${COHORT}/H1A2_FULL.jsonl" & p0=$!
run_cell 1 17 v2_seed82017 "${POLICY17}" "${COHORT}/C3FD_V2.jsonl" & p1=$!
run_cell 2 17 v2_seed82018 "${POLICY18}" "${COHORT}/C3FD_V2.jsonl" & p2=$!
run_cell 3 18 h1a2_fusion "${H1A2_DLM}" "${COHORT}/H1A2_FULL.jsonl" & p3=$!
run_cell 4 18 v2_seed82017 "${POLICY17}" "${COHORT}/C3FD_V2.jsonl" & p4=$!
run_cell 5 18 v2_seed82018 "${POLICY18}" "${COHORT}/C3FD_V2.jsonl" & p5=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"; wait "${p4}"; wait "${p5}"
for stream in 17 18; do for arm in h1a2_fusion v2_seed82017 v2_seed82018; do [[ -f "${RUN}/stream${stream}/${arm}/_SUCCESS" ]] || exit 3; done; done
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
