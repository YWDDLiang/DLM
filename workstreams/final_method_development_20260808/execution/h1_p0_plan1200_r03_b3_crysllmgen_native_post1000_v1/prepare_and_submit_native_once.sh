#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
BODY_SOURCE="$RUN_ROOT/body_source"
SOURCE="$RUN_ROOT/native1000_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -d "$SOURCE"
test -f "$RUN_ROOT/status/native1000_inputs_SUCCESS"
test -f "$RUN_ROOT/native_mp_cache/completion_SUCCESS"
test -f "$RUN_ROOT/status/body_submission_record.json"
test ! -e "$RUN_ROOT/status/native1000_preflight_report.json"
test ! -e "$RUN_ROOT/status/native1000_preparation_SUCCESS"
test ! -e "$RUN_ROOT/status/native1000_submission.lock"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE:$BODY_SOURCE:$R03D/runtime:$R03D"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

"$PYTHON" "$SOURCE/preflight_native.py" \
  --config "$SOURCE/CONFIG.json" \
  --body-source-dir "$BODY_SOURCE" \
  --native-source-dir "$SOURCE" \
  --native-source-manifest-sha256 "$SOURCE_SHA" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/status/native1000_preflight_report.json"
sha256sum "$RUN_ROOT/status/native1000_preflight_report.json" \
  > "$RUN_ROOT/status/native1000_preflight_report.sha256"
touch "$RUN_ROOT/status/native1000_preparation_SUCCESS"
bash "$SOURCE/submit_native_once.sh"
