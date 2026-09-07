#!/usr/bin/env bash
#SBATCH --job-name=r03-p0-pointer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -Eeuo pipefail
SOURCE="${R03_SOURCE_ROOT:?}"
RUNROOT="${R03_RUN_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
G="$PROJECT/workstreams/proposal_realization_candidates_20260826/grounding"
RUN="$RUNROOT/pointer_$SLURM_JOB_ID"
mkdir "$RUN"
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; exit "$rc"' ERR
"$PY" "$SOURCE/operations/r03_c3fd_main_20260907/prepare_assets.py" --output-dir "$RUN/assets" >"$RUN/assets.out" 2>"$RUN/assets.err"
"$PY" "$SOURCE/src/scripts/train_r03_control_pointer.py" \
 --llama-model /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B \
 --planner-checkpoint "$PROJECT/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final" \
 --teacher-data-dir "$G/data/spad_basin_closure_sft_v1_20260904" \
 --pointer-data-dir "$G/data/spad_species_pointer_v1_20260903" \
 --mp20-dir "$PROJECT/reference/crysllmgen/data/mp_20" \
 --output-dir "$RUN/train" --prompt-style h1_rich_plan_v1 --no-include-sample-id \
 --prompt-text-file "$RUN/assets/P0_NATIVE_PROMPT.txt" --contact-workers 4 \
 >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
touch "$RUN/_SUCCESS"
