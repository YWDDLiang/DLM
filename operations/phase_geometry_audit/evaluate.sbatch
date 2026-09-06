#!/usr/bin/env bash
#SBATCH --job-name=dlm-k4k8-phase-geometry
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=dlm-k4k8-phase-geometry-%j.out
#SBATCH --error=dlm-k4k8-phase-geometry-%j.err
set -Eeuo pipefail
RUN="${PHASE_DIAGNOSTIC_ROOT:?}/run_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${PHASE_OBJECT_SOURCE:?}" && git rev-parse "${PHASE_CODE_COMMIT:?}")" = "$PHASE_CODE_COMMIT"
(cd "$PHASE_OBJECT_SOURCE" && git archive "$PHASE_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$PHASE_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${PHASE_SOURCE_MANIFEST:?}" "$RUN/SOURCE_MANIFEST.json"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PHASE_PYTHON:?}"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src"
for method in k4 k8; do
  policy="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policies"][sys.argv[2]]["path"])' "$RUN/SOURCE_MANIFEST.json" "$method")"
  source="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["construction_sources"][sys.argv[2]]["path"])' "$RUN/SOURCE_MANIFEST.json" "$method")"
  source_sha="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["construction_sources"][sys.argv[2]]["sha256"])' "$RUN/SOURCE_MANIFEST.json" "$method")"
  "$PY" "$CODE/scripts/audit_legacy_phase_geometry.py" --paths-jsonl "$source" --checkpoint-path "$policy" --method "$method" --source-sha256 "$source_sha" --expected-requests 256 --output-dir "$RUN/$method" >"$RUN/$method.out" 2>"$RUN/$method.err"
  test -f "$RUN/$method/_SUCCESS"
done
"$PY" - "$RUN" <<'PY'
import json,pathlib,sys,hashlib
r=pathlib.Path(sys.argv[1])
o={"methods":{k:json.loads((r/k/"SUMMARY.json").read_text()) for k in ("k4","k8")},
   "code_commit":(r/"CODE_COMMIT").read_text().strip(),"source_manifest_sha256":hashlib.sha256((r/"SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
   "model_forwards":0,"energy_evaluations":0,"optimizer_steps":0,"per_case_stage_selection":False}
(r/"SUMMARY.json").write_text(json.dumps(o,indent=2)+"\n")
PY
touch "$RUN/_SUCCESS"

