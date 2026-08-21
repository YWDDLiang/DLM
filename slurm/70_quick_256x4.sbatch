#!/usr/bin/env bash
#SBATCH --job-name=h1a2-quick
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=h1a2-quick-%A_%a.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly PLANS="${H1A2_PLANS_JSONL:-${REPO_ROOT}/data/plans/r03_parsed_256.jsonl}"
readonly LEDGER="${REPO_ROOT}/data/plans/r03_seed_ledger_256.jsonl"
[[ -f "${PLANS}" ]] || missing "${PLANS}"
[[ -f "${LEDGER}" ]] || missing "${LEDGER}"
[[ -f "${REPO_ROOT}/checkpoints/dlm/adapter_model.safetensors" ]] || missing "checkpoints/dlm/adapter_model.safetensors"
[[ -f "${REPO_ROOT}/checkpoints/diffusion/model_494.pt" ]] || missing "checkpoints/diffusion/model_494.pt"

log "Repeat ${SLURM_ARRAY_TASK_ID}: Safe-axis=true, frozen Plans, frozen ordinal ledger"
log "The streamlined quick runner is awaiting the actual ledger and A800 environment assets."
exit 2

