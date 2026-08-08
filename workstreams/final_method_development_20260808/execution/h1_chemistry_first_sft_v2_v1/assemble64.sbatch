#!/bin/bash
#SBATCH --job-name=h1-cf-a64-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5/logs/%j_assemble64.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5/logs/%j_assemble64.err

set -Eeuo pipefail
PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="${PROJECT_ROOT}/runs/20260808_h1_chemistry_first_sft_v2_smact_split_v2_source_gate_path_repair_v5"
SOURCE_ROOT="${RUN_ROOT}/source"
export STAGE=64
export EXPECTED_CANDIDATES=sft_v2,sft_v2_c
exec bash "${SOURCE_ROOT}/workstreams/final_method_development_20260808/execution/h1_chemistry_first_sft_v2_v1/assemble_stage.sbatch"
