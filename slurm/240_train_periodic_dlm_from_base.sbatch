#!/usr/bin/env bash
#SBATCH --job-name=periodic-base-dlm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=periodic-base-dlm-%j.out
#SBATCH --error=periodic-base-dlm-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
EXPERIMENT="${PERIODIC_REPAIR_ROOT:?}/from_original_llada_v1"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
mkdir -p "$EXPERIMENT/runs"
RUN="$EXPERIMENT/runs/train_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
export PYTHONPATH="$RUN/code/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$RUN/code/src/scripts/train_periodic_dlm_from_base.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --train-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --val-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/val.jsonl" \
  --output-dir "$RUN/train" --seed 82017 --data-seed 20260515 \
  >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
