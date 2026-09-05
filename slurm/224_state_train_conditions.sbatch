#!/usr/bin/env bash
#SBATCH --job-name=spad-state-conditions
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=spad-state-conditions-%j.out
#SBATCH --error=spad-state-conditions-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_state_train_conditions_${SLURM_JOB_ID}"
mkdir "$RUN"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$SOURCE/src/scripts/prepare_state_train_conditions.py" \
  --pointer-data "$ROOT/data/spad_species_pointer_v1_20260903/train.jsonl" \
  --sft-data "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --c3fd-checkpoint "$ROOT/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt" \
  --vocabulary "$ROOT/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json" \
  --llama-model /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B \
  --planner-final "$ROOT/runs/c3fd_llama_typed_planner_39051/train_seed85017/final" \
  --pointer-state "$ROOT/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt" \
  --output-dir "$RUN/conditions" --count 1024 --batch-size 128 --seed 20260905 \
  >"$RUN/prepare.out" 2>"$RUN/prepare.err"
test -f "$RUN/conditions/_SUCCESS"
touch "$RUN/_SUCCESS"
