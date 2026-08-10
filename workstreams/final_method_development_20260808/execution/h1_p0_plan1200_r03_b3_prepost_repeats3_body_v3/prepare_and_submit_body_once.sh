#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
SOURCE="$RUN_ROOT/body_source"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -d "$SOURCE"
test -f "$RUN_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/mp_cache/completion_SUCCESS"
test -f "$RUN_ROOT/mp_cache/completion_manifest.json"
test ! -e "$RUN_ROOT/status/body_preflight_report.json"
test ! -e "$RUN_ROOT/status/body_preparation_SUCCESS"
test ! -e "$RUN_ROOT/status/body_submission.lock"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/preflight_body.py" \
  --config "$SOURCE/CONFIG.json" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$SOURCE_SHA" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/status/body_preflight_report.json"
sha256sum "$RUN_ROOT/status/body_preflight_report.json" \
  > "$RUN_ROOT/status/body_preflight_report.sha256"
touch "$RUN_ROOT/status/body_preparation_SUCCESS"
bash "$SOURCE/submit_body_once.sh"
