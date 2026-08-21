#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly SLURM_DIR="${REPO_ROOT}/slurm"
readonly ACTION="${CHECKPOINT_ACTION:-train}"
readonly RESAMPLE="${RESAMPLE_PLANS:-false}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to submit the quick workflow" >&2
  exit 2
fi

submit_id() {
  sbatch --parsable "$@" | cut -d';' -f1
}

dependencies=()

for entry in \
  "dlm:checkpoints/dlm/adapter_model.safetensors:20_train_dlm.sbatch" \
  "diffusion:checkpoints/diffusion/model_494.pt:25_train_diffusion.sbatch"; do
  IFS=: read -r label relative trainer <<<"${entry}"
  if [[ -f "${REPO_ROOT}/${relative}" ]]; then
    continue
  fi
  if [[ "${ACTION}" == "download" ]]; then
    "${SCRIPT_DIR}/download_checkpoints.sh"
  else
    echo "${label} checkpoint is missing; defaulting to training."
    dependencies+=("$(submit_id "${SLURM_DIR}/${trainer}")")
  fi
done

plans="${REPO_ROOT}/data/plans/r03_parsed_256.jsonl"
if [[ "${RESAMPLE,,}" == "true" || "${RESAMPLE}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/checkpoints/planner/adapter_model.safetensors" ]]; then
    plan_job="$(PLAN_SAMPLE_MODE=quick submit_id --export=ALL,PLAN_SAMPLE_MODE=quick "${SLURM_DIR}/30_sample_plans.sbatch")"
    dependencies+=("${plan_job}")
    plans="${REPO_ROOT}/runs/quick_256x4/plans/plans_for_dlm.jsonl"
    echo "Planner checkpoint found; resampling 256 Plans with seed 17029."
  else
    echo "Planner checkpoint is missing; falling back to frozen 256 Plans."
  fi
fi

if [[ ! -f "${plans}" && "${plans}" == *"data/plans/r03_parsed_256.jsonl" ]]; then
  echo "Frozen Plan placeholder is not populated: ${plans}" >&2
  exit 2
fi

dependency_arg=()
if (( ${#dependencies[@]} > 0 )); then
  joined="$(IFS=:; echo "${dependencies[*]}")"
  dependency_arg=("--dependency=afterok:${joined}")
fi

array_job="$(submit_id "${dependency_arg[@]}" --export="ALL,H1A2_PLANS_JSONL=${plans},SAFE_AXIS=true" "${SLURM_DIR}/70_quick_256x4.sbatch")"
report_job="$(submit_id --dependency="afterok:${array_job}" "${SLURM_DIR}/80_assemble_quick.sbatch")"

printf 'Submitted quick workflow: repeats=%s report=%s\n' "${array_job}" "${report_job}"

