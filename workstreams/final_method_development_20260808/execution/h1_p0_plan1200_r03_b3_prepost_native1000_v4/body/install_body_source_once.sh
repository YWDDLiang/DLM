#!/bin/bash
set -Eeuo pipefail
umask 077

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: install_body_source_once.sh SOURCE_GIT_COMMIT" >&2
  exit 2
fi

SOURCE_GIT_COMMIT="$1"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN_ROOT="$PROJECT/runs/20260811_h1_p0_plan1200_r03_b3_prepost_native1000_cohort_contract_repair_v4"
SOURCE="$RUN_ROOT/body_source"
TMP="$RUN_ROOT/.body_source.preparing.$$"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python

cleanup() {
  rc=$?
  if [[ "$rc" -ne 0 ]] && [[ -d "$TMP" ]]; then
    mv "$TMP" "$RUN_ROOT/.body_source.FAILED.$$"
  fi
}
trap cleanup EXIT

test -d "$RUN_ROOT"
test -f "$RUN_ROOT/status/v4_input_import_SUCCESS"
test -f "$RUN_ROOT/status/planner_assembly_SUCCESS"
test -f "$RUN_ROOT/planner_terminal_report.json"
test ! -e "$SOURCE"
test ! -e "$TMP"
test ! -e "$RUN_ROOT/status/body_source_git_commit.txt"
mkdir "$TMP"
cp -a "$SELF_DIR/." "$TMP/"

if find "$TMP" -type d -name __pycache__ -print -quit | grep -q . \
  || find "$TMP" -type f \( -name '*.pyc' -o -name '*.pyo' \) -print -quit | grep -q .; then
  echo "body source contains Python cache artifacts" >&2
  exit 3
fi
(
  cd "$TMP"
  find . -type f ! -name SOURCE_SHA256.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sed 's#  \./#  #' > SOURCE_SHA256.txt
  sha256sum -c SOURCE_SHA256.txt
)

for script in "$TMP"/*.sh "$TMP"/*.sbatch; do
  bash -n "$script"
done
"$PYTHON" - "$TMP" <<'PY'
import ast
from pathlib import Path
import sys

for path in sorted(Path(sys.argv[1]).glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY
if grep -R --include='*.sbatch' -E -q '^#SBATCH[[:space:]]+--partition=.*gpu_long' "$TMP"; then
  echo "gpu_long is forbidden" >&2
  exit 3
fi
if grep -R -E -q '(MP_API_KEY|PMG_MAPI_KEY|MAPI_KEY)[[:space:]]*=[[:space:]]*[^"'"'"'[:space:]]' "$TMP"; then
  echo "credential serialization is forbidden" >&2
  exit 3
fi

printf '%s\n' "$SOURCE_GIT_COMMIT" > "$RUN_ROOT/status/body_source_git_commit.txt"
mv "$TMP" "$SOURCE"
trap - EXIT
SOURCE_SHA="$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | cut -d' ' -f1)"
printf '%s  body_source/SOURCE_SHA256.txt\n' "$SOURCE_SHA" \
  > "$RUN_ROOT/status/body_source_manifest.sha256"
echo "body_source_manifest_sha256=$SOURCE_SHA"
