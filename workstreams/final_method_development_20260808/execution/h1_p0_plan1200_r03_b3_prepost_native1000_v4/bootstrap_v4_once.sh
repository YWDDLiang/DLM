#!/bin/bash
set -Eeuo pipefail
umask 077

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: bootstrap_v4_once.sh SOURCE_GIT_COMMIT" >&2
  exit 2
fi

SOURCE_GIT_COMMIT="$1"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ID=20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4
RUN_ROOT="$PROJECT/runs/$RUN_ID"
PREPARING="$PROJECT/runs/.$RUN_ID.preparing.$$"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

test ! -e "$RUN_ROOT"
test ! -e "$PREPARING"
test -f "$SELF_DIR/INPUT_IMPORT_CONTRACT.json"
test -f "$SELF_DIR/import_v3_inputs.py"

on_exit() {
  rc=$?
  set +e
  if [[ "$rc" -ne 0 ]] && [[ -d "$PREPARING" ]]; then
    mv "$PREPARING" "$PROJECT/runs/.$RUN_ID.FAILED.$$"
  fi
  return "$rc"
}
trap on_exit EXIT

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

"$PYTHON" "$SELF_DIR/import_v3_inputs.py" \
  --package-root "$SELF_DIR" \
  --preparing-root "$PREPARING" \
  --final-root "$RUN_ROOT" \
  --source-git-commit "$SOURCE_GIT_COMMIT"

mv "$PREPARING" "$RUN_ROOT"
trap - EXIT
sha256sum "$RUN_ROOT/status/v4_input_import_report.json" \
  > "$RUN_ROOT/status/v4_input_import_report.sha256"
echo "V4_RUN_ROOT=$RUN_ROOT"
echo "STAGE v4_input_import_complete"
