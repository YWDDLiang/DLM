#!/bin/bash
#SBATCH --job-name=h1-cf-a256-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_exact_identity_copy_repair_v6/logs/%j_assemble256.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_exact_identity_copy_repair_v6/logs/%j_assemble256.err

set -Eeuo pipefail
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_exact_identity_copy_repair_v6"
SOURCE_ROOT="${RUN_ROOT}/source"
export STAGE=256
: "${EXPECTED_CANDIDATES:?missing passing raw64 candidates}"
exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"
