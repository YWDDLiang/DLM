#!/usr/bin/env bash
#SBATCH --job-name=btrd-known1159
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=520G
#SBATCH --time=05:00:00
#SBATCH --output=btrd-known1159-%j.out
#SBATCH --error=btrd-known1159-%j.err

set -Eeuo pipefail

export KNOWN1200_POLICY="${BTRD_POLICY:?export completed BTRD step512 checkpoint}"
export KNOWN1200_POLICY_SHA256="${BTRD_POLICY_SHA256:?pin frozen Compact-V2 LoRA}"
export KNOWN1200_RELATION_SHA256="${BTRD_RELATION_SHA256:?pin BTRD residual}"
export KNOWN1200_RELATION_CONFIG_SHA256="${BTRD_RELATION_CONFIG_SHA256:?pin BTRD residual config}"
export KNOWN1200_METHOD_LABEL=BTRD-tau200-step512
export KNOWN1200_RUN_PREFIX=btrd_known1200_fast

exec bash "${H1A2_CODE_ROOT:?export deployed code root}/slurm/136_generate_known1200_baseline_fast.sbatch"
