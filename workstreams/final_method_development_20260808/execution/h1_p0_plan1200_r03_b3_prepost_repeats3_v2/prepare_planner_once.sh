#!/bin/bash
set -Eeuo pipefail
umask 077

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: prepare_planner_once.sh SOURCE_GIT_COMMIT" >&2
  exit 2
fi

SOURCE_GIT_COMMIT="$1"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_ROOT="$(cd "$SELF_DIR/../../../.." && pwd)"
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN_ID=20260811_h1_p0_plan1200_r03_b3_prepost_repeats3_import_repair_v2
EXECUTION_NAME=h1_p0_plan1200_r03_b3_prepost_repeats3_v2
RUN_ROOT="$PROJECT/runs/$RUN_ID"
TMP_ROOT="$PROJECT/runs/.$RUN_ID.preparing.$$"
SOURCE_STAGE="$TMP_ROOT/planner_source"

cleanup() {
  rc=$?
  if [[ "$rc" -ne 0 ]] && [[ -d "$TMP_ROOT" ]]; then
    mv "$TMP_ROOT" "${TMP_ROOT}.FAILED"
  fi
}
trap cleanup EXIT

test ! -e "$RUN_ROOT"
test ! -e "$TMP_ROOT"
test -d "$ARCHIVE_ROOT/crystal_dlm"
test -f "$ARCHIVE_ROOT/scripts/__init__.py"
test -f "$ARCHIVE_ROOT/scripts/sample_llama_h1_formula_plans.py"
test -f "$ARCHIVE_ROOT/scripts/sample_llada_dynamic_crystals.py"

mkdir -p "$SOURCE_STAGE/scripts" \
  "$SOURCE_STAGE/workstreams/final_method_development_20260808/execution" \
  "$TMP_ROOT/logs" "$TMP_ROOT/status" "$TMP_ROOT/repeats"
cp -a "$ARCHIVE_ROOT/crystal_dlm" "$SOURCE_STAGE/"
cp "$ARCHIVE_ROOT/scripts/__init__.py" "$SOURCE_STAGE/scripts/"
cp "$ARCHIVE_ROOT/scripts/sample_llama_h1_formula_plans.py" "$SOURCE_STAGE/scripts/"
cp "$ARCHIVE_ROOT/scripts/sample_llada_dynamic_crystals.py" "$SOURCE_STAGE/scripts/"
cp -a "$SELF_DIR" \
  "$SOURCE_STAGE/workstreams/final_method_development_20260808/execution/"

if find "$SOURCE_STAGE" -type d -name __pycache__ -print -quit | grep -q . \
  || find "$SOURCE_STAGE" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print -quit | grep -q .; then
  echo "source archive contains Python cache artifacts" >&2
  exit 3
fi
(
  cd "$SOURCE_STAGE"
  find . -type f ! -name SOURCE_SHA256.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sed 's#  \./#  #' > SOURCE_SHA256.txt
  sha256sum -c SOURCE_SHA256.txt
)
SOURCE_SHA="$(sha256sum "$SOURCE_STAGE/SOURCE_SHA256.txt" | cut -d' ' -f1)"

for script in "$SOURCE_STAGE/workstreams/final_method_development_20260808/execution/$EXECUTION_NAME"/*.sh \
  "$SOURCE_STAGE/workstreams/final_method_development_20260808/execution/$EXECUTION_NAME"/*.sbatch; do
  bash -n "$script"
done
"$PYTHON" - "$SOURCE_STAGE" <<'PY'
import ast
from pathlib import Path
import sys

for path in sorted(Path(sys.argv[1]).rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
PYTHONPATH="$SOURCE_STAGE" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" - "$SOURCE_STAGE" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).resolve()
import scripts
from scripts.sample_llada_dynamic_crystals import init_distributed, rank_path
from scripts.sample_llama_h1_formula_plans import merge_distributed_outputs, main

expected_package = (source / "scripts/__init__.py").resolve()
observed_package = Path(scripts.__file__).resolve()
if observed_package != expected_package:
    raise SystemExit(
        f"repository scripts package was shadowed: expected={expected_package} "
        f"observed={observed_package}"
    )
for symbol in (init_distributed, rank_path, merge_distributed_outputs, main):
    if not callable(symbol):
        raise SystemExit(f"planner import preflight found non-callable symbol: {symbol!r}")
print({"planner_import_preflight": "pass", "scripts_package": str(observed_package)})
PY
if grep -R --include='*.sbatch' -E -q '^#SBATCH[[:space:]]+--partition=.*gpu_long' "$SOURCE_STAGE"; then
  echo "gpu_long is forbidden" >&2
  exit 3
fi
if find "$SOURCE_STAGE" -type f ! -name prepare_planner_once.sh -print0 \
  | xargs -0 grep -E -q 'MP_API_KEY=|PMG_MAPI_KEY=|MAPI_KEY='; then
  echo "credential serialization is forbidden" >&2
  exit 3
fi

test -d /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
P0="$PROJECT/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final"
test "$(sha256sum "$P0/adapter_model.safetensors" | cut -d' ' -f1)" = 65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a

printf '%s\n' "$SOURCE_GIT_COMMIT" > "$TMP_ROOT/status/planner_source_git_commit.txt"
printf '%s  planner_source/SOURCE_SHA256.txt\n' "$SOURCE_SHA" \
  > "$TMP_ROOT/status/planner_source_manifest.sha256"
touch "$TMP_ROOT/status/preparation_SUCCESS"
mv "$TMP_ROOT" "$RUN_ROOT"
trap - EXIT
bash "$RUN_ROOT/planner_source/workstreams/final_method_development_20260808/execution/$EXECUTION_NAME/submit_planner_once.sh"
