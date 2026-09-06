#!/usr/bin/env bash
#SBATCH --job-name=dlm-frozen-audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=320G
#SBATCH --time=01:00:00
#SBATCH --output=dlm-frozen-audit-%j.out
#SBATCH --error=dlm-frozen-audit-%j.err
set -Eeuo pipefail
RUN="${AUDIT_ROOT:?}/runs/probe_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "${AUDIT_MODEL_SOURCE:?}" && git rev-parse HEAD) >"$RUN/MODEL_CODE_COMMIT"
(cd "$AUDIT_MODEL_SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
TOOL_REL=docs/v3_scientific_audit_20260906/tools/diagnose_v2_conditioning.py
mkdir -p "$(dirname "$RUN/code/$TOOL_REL")"
(cd "$AUDIT_MODEL_SOURCE" && git show "${AUDIT_TOOLS_COMMIT:?}:$TOOL_REL") >"$RUN/code/$TOOL_REL"
printf '%s\n' "$AUDIT_TOOLS_COMMIT" >"$RUN/AUDIT_CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
export PYTHONPATH="$RUN/code/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"${AUDIT_PYTHON:?}" -m torch.distributed.run --standalone --nproc_per_node="${AUDIT_GPUS:-4}" \
  "$RUN/code/$TOOL_REL" --model-path "${AUDIT_MODEL_PATH:?}" \
  --checkpoint-path "${AUDIT_CHECKPOINT:?}" --prepared-val "${AUDIT_PREPARED_VAL:?}" \
  --original-val "${AUDIT_ORIGINAL_VAL:?}" --output-dir "$RUN/diagnostic" \
  --source-count 100 --selection-seed 202609061 --state-seed 20260906 \
  --state-epoch 1 --batch-size 2 --max-length 382 --cpu-threads 2 \
  >"$RUN/probe.out" 2>"$RUN/probe.err"
test -f "$RUN/diagnostic/_SUCCESS"
touch "$RUN/_SUCCESS"
