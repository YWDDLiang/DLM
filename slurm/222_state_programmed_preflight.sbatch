#!/usr/bin/env bash
#SBATCH --job-name=spad-state-preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=spad-state-preflight-%j.out
#SBATCH --error=spad-state-preflight-%j.err
set -Eeuo pipefail

ROOT="${H1A2_CANDIDATE_ROOT:?set H1A2_CANDIDATE_ROOT}"
SOURCE="${SSCD_SOURCE_ROOT:?set SSCD_SOURCE_ROOT}"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
POLICY="$(<"$ROOT/runs/spad_basin_closure_ce_39700/POLICY_PATH")"
RUN="$ROOT/runs/spad_state_preflight_${SLURM_JOB_ID}"
mkdir "$RUN"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
export PYTHONPATH="$SOURCE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$SOURCE/src/scripts/preflight_state_programmed_spad.py" \
  --model-path "$MODEL" --checkpoint-path "$POLICY" \
  --data-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --output-dir "$RUN/check" --steps 5 --seed 20260905 \
  >"$RUN/preflight.out" 2>"$RUN/preflight.err"
test -f "$RUN/check/_SUCCESS"
touch "$RUN/_SUCCESS"
