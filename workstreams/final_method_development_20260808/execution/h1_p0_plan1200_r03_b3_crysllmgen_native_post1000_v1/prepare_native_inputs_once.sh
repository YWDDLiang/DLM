#!/bin/bash
set -Eeuo pipefail
umask 077

PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_execmode_repair_v3"
BODY_SOURCE="$RUN_ROOT/body_source"
NATIVE_SOURCE="$RUN_ROOT/native1000_source"
PARALLEL_SOURCE="$RUN_ROOT/mp_parallel_source"
R03D="$PROJECT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test -d "$NATIVE_SOURCE"
test -d "$BODY_SOURCE"
test -d "$PARALLEL_SOURCE"
test -f "$RUN_ROOT/status/body_submission_record.json"
test -f "$RUN_ROOT/mp_cache/completion_SUCCESS"
test ! -e "$RUN_ROOT/status/native1000_inputs_SUCCESS"
test ! -e "$RUN_ROOT/status/native_mp_cache_audit.json"
test ! -e "$RUN_ROOT/native_mp_cache"
(cd "$NATIVE_SOURCE" && sha256sum -c SOURCE_SHA256.txt)
SOURCE_SHA="$(sha256sum "$NATIVE_SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

cd "$PROJECT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$NATIVE_SOURCE:$BODY_SOURCE:$PARALLEL_SOURCE:$R03D/runtime:$R03D"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

for repeat in 0 1 2; do
  candidate_root="$RUN_ROOT/repeats/$repeat/crysllmgen_native_candidates"
  test ! -e "$candidate_root"
  "$PYTHON" "$NATIVE_SOURCE/freeze_candidate_pools.py" \
    --repeat "$repeat" \
    --planner-dir "$RUN_ROOT/repeats/$repeat/planner1200" \
    --v3-cohort-dir "$RUN_ROOT/repeats/$repeat/cohort" \
    --output-dir "$candidate_root"
done

"$PYTHON" "$NATIVE_SOURCE/native_mp_cache.py" audit \
  --body-config "$BODY_SOURCE/CONFIG.json" \
  --body-source-dir "$BODY_SOURCE" \
  --native-source-dir "$NATIVE_SOURCE" \
  --native-source-manifest-sha256 "$SOURCE_SHA" \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/status/native_mp_cache_audit.json"
sha256sum "$RUN_ROOT/status/native_mp_cache_audit.json" \
  > "$RUN_ROOT/status/native_mp_cache_audit.sha256"
touch "$RUN_ROOT/status/native1000_inputs_SUCCESS"
"$PYTHON" - "$RUN_ROOT/status/native_mp_cache_audit.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print({
    "status": d["status"],
    "wanted_chemsys_count": d["wanted_chemsys_count"],
    "cached_chemsys_count": d["cached_chemsys_count"],
    "missing_chemsys_count": d["missing_chemsys_count"],
})
PY
