#!/usr/bin/env bash
#SBATCH --job-name=spad-state-labels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=spad-state-labels-%j.out
#SBATCH --error=spad-state-labels-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
INPUT="${SPAD_PATHS_JSONL:?}"
test -f "$INPUT"
RUN="$ROOT/runs/spad_state_labels_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
git -C "$SOURCE" rev-parse HEAD >"$RUN/CODE_COMMIT"
git -C "$SOURCE" archive HEAD | tar -x -C "$RUN/code"
export PYTHONPATH="$RUN/code/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" "$RUN/code/scripts/label_programmed_paths.py" --input-jsonl "$INPUT" --output-dir "$RUN/labels" \
  --gpu-count 2 --workers-per-gpu 2 >"$RUN/label.out" 2>"$RUN/label.err"
test -f "$RUN/labels/_SUCCESS"
touch "$RUN/_SUCCESS"
