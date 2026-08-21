#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly SLURM_DIR="${REPO_ROOT}/slurm"
readonly ACTION="${CHECKPOINT_ACTION:-train}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to submit the H1-A2 workflow" >&2
  exit 2
fi

submit_id() {
  sbatch --parsable "$@" | cut -d';' -f1
}

dependencies=()

if [[ ! -f "${REPO_ROOT}/checkpoints/dlm/adapter_model.safetensors" ]]; then
  if [[ "${ACTION}" == "download" ]]; then
    "${SCRIPT_DIR}/download_checkpoints.sh"
  else
    dependencies+=("$(submit_id "${SLURM_DIR}/20_train_dlm.sbatch")")
  fi
fi

if [[ ! -f "${REPO_ROOT}/checkpoints/diffusion/model_494.pt" ]]; then
  if [[ "${ACTION}" == "download" ]]; then
    "${SCRIPT_DIR}/download_checkpoints.sh"
  else
    dependencies+=("$(submit_id "${SLURM_DIR}/25_train_diffusion.sbatch")")
  fi
fi

planner_dependency=""
if [[ -f "${REPO_ROOT}/checkpoints/planner/adapter_model.safetensors" ]]; then
  planner_dependency="$(submit_id "${SLURM_DIR}/30_sample_plans.sbatch")"
  dependencies+=("${planner_dependency}")
elif [[ ! -f "${REPO_ROOT}/data/plans/h1a2_parsed_1186.jsonl" ]]; then
  echo "Planner checkpoint and fallback H1-A2 Plan file are both missing." >&2
  echo "See docs/PLACEHOLDER_ASSETS.md." >&2
  exit 2
else
  echo "Planner checkpoint is missing; using frozen H1-A2 Plans."
fi

dependency_arg=()
if (( ${#dependencies[@]} > 0 )); then
  joined="$(IFS=:; echo "${dependencies[*]}")"
  dependency_arg=("--dependency=afterok:${joined}")
fi

body_job="$(submit_id "${dependency_arg[@]}" "${SLURM_DIR}/40_generate_body.sbatch")"
refine_job="$(submit_id --dependency="afterok:${body_job}" "${SLURM_DIR}/50_refine_diffusion.sbatch")"
eval_job="$(submit_id --dependency="afterok:${refine_job}" "${SLURM_DIR}/60_evaluate.sbatch")"

printf 'Submitted H1-A2 workflow: body=%s refine=%s evaluate=%s\n' \
  "${body_job}" "${refine_job}" "${eval_job}"

