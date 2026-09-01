#!/usr/bin/env bash
#SBATCH --job-name=btrd800t-residual
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=260G
#SBATCH --time=06:00:00
#SBATCH --output=btrd800t-residual-%j.out
#SBATCH --error=btrd800t-residual-%j.err

set -Eeuo pipefail
export BTRD_METHOD_LABEL=BTRD-800T
export BTRD_TRAIN_RUN_PREFIX=btrd800t_residual
export BTRD_TAIL_TEMPERATURE=0.10
export BTRD_TAIL_MIX=0.50
exec bash "${H1A2_CODE_ROOT:?export code root}/slurm/138_train_btrd_residual.sbatch"
