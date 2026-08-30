#!/usr/bin/env bash
#SBATCH --job-name=h1a2-rich-assets
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=h1a2-rich-assets-%j.out
#SBATCH --error=h1a2-rich-assets-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly REPO=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
readonly RICH_DLM="${REPO}/runs/20260529_212834-r5c-exactlen-256/outputs/r5c_exact_sft/final"
readonly MINIMAL_DLM="${ROOT}/runs/ctv_minimal_base_36898/train/checkpoints/step-696"
readonly DLM_BASE=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly MODEL494=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly MP20="${REPO}/reference/crysllmgen/data/mp_20"
readonly C3FD_DATA="${ROOT}/data/c3fd_semantic_v21_step1b_20260828"
readonly C3FD17="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly C3FD18="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed18/checkpoint.pt"
readonly OUTPUT="${ROOT}/data/rich_dlm_live_asset_manifest_v1_20260830"
readonly PREPARING="${OUTPUT}.preparing"
readonly RUN="${ROOT}/runs/rich_dlm_live_asset_manifest_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly CONTRACT_SHA=c329f4a9b7c1dcbb08b9b5d861e7dfc8e65fcfdca8955416ad9b9931f5e81ad3
readonly CONTRACT_DIR="${ROOT}/runs/contracts/${CONTRACT_SHA}"
readonly SCRIPT_SHA=620bc34b4e61f7164c830d2303fe4ee4305eca55192979a93daef2b9d75462a6
readonly RICH_ADAPTER_SHA=5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d

mkdir -p "${ROOT}/runs/contracts"
mkdir "${RUN}"

on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  if [[ -d "${PREPARING}" ]]; then
    printf '%s\n' "${SLURM_JOB_ID}" >"${PREPARING}/FAILED_JOB_ID"
    mv "${PREPARING}" "${OUTPUT}.FAILED.${SLURM_JOB_ID}"
  fi
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

mkdir "${CONTRACT_DIR}"
mkdir "${PREPARING}"
printf '%s\n' "${SLURM_JOB_ID}" >"${CONTRACT_DIR}/job_id"
printf '%s\n' "${CONTRACT_SHA}" >"${CONTRACT_DIR}/contract_sha256"
{
  printf 'field\tvalue\n'
  printf 'schema\t%s\n' 'h1a2_rich_dlm_live_asset_manifest_v1'
  printf 'contract_sha256\t%s\n' "${CONTRACT_SHA}"
  printf 'job_id\t%s\n' "${SLURM_JOB_ID}"
  printf 'script_sha256\t%s\n' "${SCRIPT_SHA}"
  printf 'rich_adapter_expected_sha256\t%s\n' "${RICH_ADAPTER_SHA}"
  printf 'workers\t4\n'
  printf 'gpu_jobs_used\t0\n'
  printf 'outcomes_read\tfalse\n'
} >"${RUN}/scientific_contract.tsv"

[[ ! -e "${OUTPUT}" ]] || exit 2
[[ "$(sha256sum "${ROOT}/scripts/audit_live_assets.py" | awk '{print $1}')" == "${SCRIPT_SHA}" ]] || exit 2

sha256sum "${ROOT}/scripts/audit_live_assets.py" >"${RUN}/CODE.sha256"

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${PYTHON}" "${ROOT}/scripts/audit_live_assets.py" \
  --asset "rich_dlm=${RICH_DLM}" \
  --asset "minimal_dlm=${MINIMAL_DLM}" \
  --asset "dlm_base=${DLM_BASE}" \
  --asset "model494=${MODEL494}" \
  --asset "mp20=${MP20}" \
  --asset "c3fd_checkpoint17=${C3FD17}" \
  --asset "c3fd_checkpoint18=${C3FD18}" \
  --asset "c3fd_data=${C3FD_DATA}" \
  --asset "body_sampler=${ROOT}/src/scripts/sample_llada_r5_exact_length.py" \
  --asset "rich_parser=${ROOT}/src/crystal_dlm/r5_plan_body.py" \
  --asset "c3fd_sampler=${ROOT}/scripts/sample_c3fd_plans.py" \
  --asset "refiner=${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --asset "evaluator=${ROOT}/eval_runtime/run_full_reconstructed_eval.py" \
  --expected-file "rich_dlm:adapter_model.safetensors=${RICH_ADAPTER_SHA}" \
  --workers 4 \
  --output-dir "${OUTPUT}" \
  >"${RUN}/audit.out" 2>"${RUN}/audit.err"

[[ -f "${OUTPUT}/_SUCCESS" ]] || exit 3
rmdir "${PREPARING}"
find "${RUN}" -type f ! -name OUTPUTS.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${OUTPUT}"
