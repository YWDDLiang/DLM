#!/usr/bin/env bash
#SBATCH --job-name=dlm-k4-k8-cooperative-endpoint
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=dlm-k4-k8-cooperative-endpoint-%j.out
#SBATCH --error=dlm-k4-k8-cooperative-endpoint-%j.err
set -Eeuo pipefail
ROOT="${COOPERATIVE_CANDIDATE_ROOT:?}"
RUN="${COOPERATIVE_ROOT:?}/runs/eval_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${COOPERATIVE_OBJECT_SOURCE:?}" && git rev-parse "${COOPERATIVE_CODE_COMMIT:?}")" = "$COOPERATIVE_CODE_COMMIT"
(cd "$COOPERATIVE_OBJECT_SOURCE" && git archive "$COOPERATIVE_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$COOPERATIVE_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${COOPERATIVE_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
PY="${COOPERATIVE_PYTHON:?}"
GPUS=6
CRYS="${COOPERATIVE_CRYSLLMGEN:?}"
CONFIG="${COOPERATIVE_FROZEN_CONFIG:?}"
CACHE="${COOPERATIVE_OFFICIAL_CACHE:?}"
CONDITIONS="${COOPERATIVE_CONDITIONS:?}"
INPUTS="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_prefix_inputs.json"
SPEC="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_cooperative/candidate_spec.json"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$RUN/LAUNCH_MANIFEST.json" "$COOPERATIVE_CODE_COMMIT" "$RUN/ENTRY.sbatch" "$SPEC" <<'PY'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m["code_commit"] == sys.argv[2]
assert m["entry_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
assert m["spec_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest()
for name, entry in m["frozen_inputs"].items():
    assert hashlib.sha256(pathlib.Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"], name
for name, entry in m["policies"].items():
    policy = pathlib.Path(entry["path"])
    train = json.loads((policy.parent.parent/"TRAIN_FINAL.json").read_text())
    assert train["eligible_policy"] is True and pathlib.Path(train["policy_path"]).resolve() == policy.resolve()
    for filename, expected in entry["files_sha256"].items():
        assert hashlib.sha256((policy/filename).read_bytes()).hexdigest() == expected, (name, filename)
PY
label_evaluate() {
  local directory="$1" stage="$2" endpoint="$3"
  "$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$directory/$stage/paths.jsonl" \
    --output-dir "$directory/$stage-labels" --purpose evaluation --gpu-count "$GPUS" --workers-per-gpu 2 \
    >"$directory/$stage-label.out" 2>"$directory/$stage-label.err"
  "$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$directory/$stage/paths.jsonl" \
    --labels-jsonl "$directory/$stage-labels/labels.jsonl" --frozen-config "$CONFIG" --official-cache "$CACHE" \
    --output-dir "$directory/$stage-evaluation" --expected-requests 256 --endpoint "$endpoint" \
    --cohort-role fixed_development --policy-stage final >"$directory/$stage-evaluate.out" 2>"$directory/$stage-evaluate.err"
}
compare() {
  local directory="$1" stage="$2" reference="$3" paths="$4"
  "$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" --reference-eval-dir "$reference" \
    --method-eval-dir "$directory/$stage-evaluation" --reference-paths-jsonl "$paths" \
    --method-paths-jsonl "$directory/$stage/paths.jsonl" --output-dir "$directory/$stage-vs-original" --comparison-purpose final \
    >"$directory/$stage-compare.out" 2>"$directory/$stage-compare.err"
}
for method in k4 k8; do
  directory="$RUN/$method"
  mkdir "$directory"
  policy="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policies"][sys.argv[2]]["path"])' "$RUN/LAUNCH_MANIFEST.json" "$method")"
  source="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cooperative_sources"][sys.argv[2]]["path"])' "$RUN/LAUNCH_MANIFEST.json" "$method")"
  source_sha="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cooperative_sources"][sys.argv[2]]["sha256"])' "$RUN/LAUNCH_MANIFEST.json" "$method")"
  "$PY" "$CODE/scripts/extract_legacy_cooperative_endpoint.py" --paths-jsonl "$source" --checkpoint-path "$policy" \
    --method "$method" --output-dir "$directory/sample" --expected-requests 256 --source-sha256 "$source_sha" \
    >"$directory/extraction.out" 2>"$directory/extraction.err"
  test -f "$directory/sample/_SUCCESS"
  "$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$directory/sample/paths.jsonl" \
    --native-source body --output-dir "$directory/native" --crysllmgen-dir "$CRYS" \
    >"$directory/native-artifacts.out" 2>"$directory/native-artifacts.err"
  label_evaluate "$directory" native native
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "$directory/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
    --crysllmgen-dir "$CRYS" --output-dir "$directory/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
    --diff-steps 800 --num-evals 1 --max-proposals 256 >"$directory/refine.out" 2>"$directory/refine.err"
  payload="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$directory/refine/refinement_metrics.json")"
  "$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$directory/native/paths.jsonl" \
    --native-source body --refined-pt "$payload" --output-dir "$directory/tau800" \
    >"$directory/tau800-artifacts.out" 2>"$directory/tau800-artifacts.err"
  label_evaluate "$directory" tau800 tau800
  if [[ "$method" == k4 ]]; then
    original_native="$ROOT/runs/spad_state_eval_method_39910"
    original_tau="$ROOT/runs/spad_state_tau800_39910"
  else
    original_native="$ROOT/runs/spad_state_eval_method_39945"
    original_tau="$ROOT/runs/spad_state_tau800_39948"
  fi
  compare "$directory" native "$original_native/evaluation" "$original_native/native/paths.jsonl"
  compare "$directory" tau800 "$original_tau/evaluation" "$original_tau/tau800/paths.jsonl"
  "$PY" - "$directory" "$method" "$policy" "$SPEC" <<'PY'
import hashlib, json, pathlib, sys
run = pathlib.Path(sys.argv[1])
counts = {}
for name in ("native", "tau800"):
    d = run/f"{name}-evaluation"
    assert (d/"_SUCCESS").is_file(), name
    counts[name] = json.loads((d/"EVALUATION_FINAL.json").read_text())["counts"]
    assert counts[name]["requests"] == 256
qualified = [name for name, c in counts.items() if c["strict_sun"] >= 26 and c["meta_sun"] >= 128]
report = {"method": sys.argv[2]+"_cooperative_endpoint_v1", "policy_path": sys.argv[3], "requests": 256,
          "spec_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest(),
          "counts": counts, "qualified_endpoints": qualified, "supplement_1000_required": bool(qualified),
          "qualification_rule": "same endpoint strict_sun>=26 AND meta_sun>=128; all256",
          "outcome_selection": False, "inference_mlip": False, "new_optimizer_steps": 0,
          "new_model_forwards": 0, "endpoint_selection": "same cooperative end for every successfully constructed request"}
(run/"EVAL_FINAL.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
PY
  touch "$directory/_SUCCESS"
done
"$PY" - "$RUN" <<'PY'
import json, pathlib, sys
r = pathlib.Path(sys.argv[1])
results = {}
for name in ("k4", "k8"):
    assert (r/name/"_SUCCESS").is_file(), name
    results[name] = json.loads((r/name/"EVAL_FINAL.json").read_text())
(r/"EVAL_FINAL.json").write_text(json.dumps({"methods": results, "same_fixed_spec": True, "no_per_request_stage_selection": True}, indent=2)+"\n")
PY
touch "$RUN/_SUCCESS"
