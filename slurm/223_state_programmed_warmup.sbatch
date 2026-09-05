#!/usr/bin/env bash
#SBATCH --job-name=spad-state-warmup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=280G
#SBATCH --time=05:00:00
#SBATCH --output=spad-state-warmup-%j.out
#SBATCH --error=spad-state-warmup-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?set H1A2_CANDIDATE_ROOT}"
SOURCE="${SSCD_SOURCE_ROOT:?set SSCD_SOURCE_ROOT}"
PREFLIGHT="${STATE_PREFLIGHT_RUN:?set STATE_PREFLIGHT_RUN}"
test -f "$PREFLIGHT/_SUCCESS"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
POLICY="$(<"$ROOT/runs/spad_basin_closure_ce_39700/POLICY_PATH")"
RUN="$ROOT/runs/spad_state_warmup_${SLURM_JOB_ID}"
mkdir "$RUN"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$SOURCE/src/scripts/train_state_conditioned_spad.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --checkpoint-path "$POLICY" \
  --data-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --output-dir "$RUN/train" --expected-rows 27136 --epochs 1 \
  --batch-size 2 --effective-batch 16 --num-workers 2 \
  --learning-rate 1e-5 --warmup-steps 100 --seed 20260905 \
  >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
