#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_mpcomplete_v1"
RUN="$ROOT/runs/20260803_h1_body_safeaxis_refined_repeats4_mpcomplete_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
: "${H1_R03F_SOURCE_SHA256:?R03F source manifest SHA256 is required}"

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_NAME:-}" ]]; then
  echo "ERROR: R03F preflight is A800 login-node-only" >&2
  exit 2
fi
if [[ -e "$RUN" ]]; then
  echo "ERROR: fixed R03F run identity already exists" >&2
  exit 3
fi

cd "$ROOT"
test "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = \
  "$H1_R03F_SOURCE_SHA256"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/complete_sun.py" \
  --config "$SOURCE/config.json" \
  --source-dir "$SOURCE" \
  --project-root "$ROOT" \
  --run-root "$RUN" \
  --execution-manifest-sha256 "$H1_R03F_SOURCE_SHA256" \
  --preflight-only
