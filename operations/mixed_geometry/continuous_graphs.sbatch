#!/usr/bin/env bash
#SBATCH --job-name=dlm-float-graph-check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=dlm-float-graph-check-%j.out
#SBATCH --error=dlm-float-graph-check-%j.err
set -Eeuo pipefail
RUN="${MIXED_ROOT:?}/runs/continuous_graphs_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${MIXED_CODE_COMMIT:?}")" = "$MIXED_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$MIXED_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXED_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
export PYTHONPATH="$RUN/code/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${MIXED_PYTHON:?}" "$RUN/code/operations/mixed_geometry/verify_continuous_graphs.py" \
  --identity "${MIXED_VAL_IDENTITY:?}" --crysllmgen-dir "${MIXED_CRYSLLMGEN:?}" \
  --output-dir "$RUN/verification" >"$RUN/verification.out" 2>"$RUN/verification.err"
test -f "$RUN/verification/_SUCCESS"
touch "$RUN/_SUCCESS"
