#!/usr/bin/env bash
#SBATCH --job-name=dlm-cif-identity
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=dlm-cif-identity-%j.out
#SBATCH --error=dlm-cif-identity-%j.err
set -Eeuo pipefail
RUN="${AUDIT_ROOT:?}/runs/source_identity_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "${AUDIT_MODEL_SOURCE:?}" && git rev-parse HEAD) >"$RUN/MODEL_CODE_COMMIT"
test "$(cat "$RUN/MODEL_CODE_COMMIT")" = "${AUDIT_MODEL_COMMIT:?}"
(cd "$AUDIT_MODEL_SOURCE" && git archive "$AUDIT_MODEL_COMMIT") | tar -x -C "$RUN/code"
TOOL_REL=docs/v3_scientific_audit_20260906/tools/audit_continuous_source_identity.py
mkdir -p "$(dirname "$RUN/code/$TOOL_REL")"
(cd "$AUDIT_MODEL_SOURCE" && git show "${AUDIT_TOOLS_COMMIT:?}:$TOOL_REL") >"$RUN/code/$TOOL_REL"
printf '%s\n' "$AUDIT_TOOLS_COMMIT" >"$RUN/AUDIT_CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${AUDIT_SOURCE_CONFIG:?}" "$RUN/INPUT_CONFIG.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SOURCE_ID_RUN="$RUN" SOURCE_ID_TOOL="$TOOL_REL"
"${AUDIT_PYTHON:?}" - <<'PY'
import json, os, subprocess, sys
from pathlib import Path

root = Path(os.environ["SOURCE_ID_RUN"])
config = json.loads((root / "INPUT_CONFIG.json").read_text())
results = []
for split in config["splits"]:
    name = split["name"]
    arguments = [
        sys.executable, str(root / "code" / os.environ["SOURCE_ID_TOOL"]),
        "--project-root", str(root / "code"),
        "--source-csv", split["source_csv"],
        "--training-jsonl", split["training_jsonl"],
        "--output-dir", str(root / name),
        "--split", name,
        "--expected-sources", str(split["expected_sources"]),
        "--source-csv-sha256", split["source_csv_sha256"],
        "--training-sha256", split["training_sha256"],
        "--workers", str(config.get("workers", 6)),
    ]
    with (root / (name + ".out")).open("w") as out, (root / (name + ".err")).open("w") as err:
        outcome = subprocess.run(arguments, stdout=out, stderr=err)
    summary_path = root / name / "SUMMARY.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    results.append({"split": name, "exit_code": outcome.returncode, "summary": summary})
    print(json.dumps({"split": name, "exit_code": outcome.returncode, "status": summary.get("status"),
                      "counts": summary.get("counts")}), flush=True)
complete = all(r["summary"].get("audit_complete") is True and r["exit_code"] in (0, 2) for r in results)
passed = complete and all(r["summary"].get("original_cif_identity_gate_passed") is True for r in results)
(root / "SUMMARY.json").write_text(json.dumps({"audit_complete": complete,
    "original_cif_identity_gate_passed": passed, "splits": results}, ensure_ascii=False, indent=2) + "\n")
if not complete:
    raise SystemExit(1)
(root / "_SUCCESS").write_text("identity audit completed; inspect the identity gate in SUMMARY.json\n")
if not passed:
    (root / "_IDENTITY_ISSUES").write_text("All sources reported; identity gate not passed.\n")
PY
