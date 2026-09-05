#!/usr/bin/env bash
#SBATCH --job-name=spad-state-terminal-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=spad-state-terminal-audit-%j.out
#SBATCH --error=spad-state-terminal-audit-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_state_terminal_audit_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
export PYTHONPATH="$RUN/code/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" "$RUN/code/scripts/audit_programmed_terminal_consistency.py" \
  --labels-jsonl "$ROOT/runs/spad_state_labels_39873/labels/labels.jsonl" "$ROOT/runs/spad_state_labels_39885/labels/labels.jsonl" \
  --output-dir "$RUN/audit" --gpu-count 4 >"$RUN/audit.out" 2>"$RUN/audit.err"
test -f "$RUN/audit/_SUCCESS"
touch "$RUN/_SUCCESS"
