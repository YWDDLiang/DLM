#!/usr/bin/env bash
#SBATCH --job-name=dlm-g-failure-diagnosis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=dlm-g-failure-diagnosis-%j.out
#SBATCH --error=dlm-g-failure-diagnosis-%j.err
set -Eeuo pipefail
RUN="${MIXED_ROOT:?}/runs/failure_diagnostic_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${DIAG_CODE_COMMIT:?}")" = "$DIAG_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$DIAG_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$DIAG_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY="${MIXED_PYTHON:?}"
SOURCE="$RUN/code/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907"
"$PY" "$SOURCE/zero_validation_baseline.py" --train-dir "${MIXED_TRAIN_RUN:?}/train" \
  --output-dir "$RUN/zero-baseline" >"$RUN/zero-baseline.out" 2>"$RUN/zero-baseline.err"
"$PY" "$SOURCE/audit_actual_output_chain.py" --run-dir "${MIXED_EVAL_RUN:?}" \
  --crysllmgen-dir "${MIXED_CRYSLLMGEN:?}" --code-root "$MIXED_TRAIN_RUN/code" \
  --output-dir "$RUN/output-chain" --cpu-threads 4 >"$RUN/output-chain.out" 2>"$RUN/output-chain.err"
test -f "$RUN/zero-baseline/ZERO_VALIDATION_BASELINE.json"
test -f "$RUN/output-chain/_SUCCESS"
touch "$RUN/_SUCCESS"
