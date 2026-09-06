#!/usr/bin/env bash
#SBATCH --job-name=dlm-k4-k8-soft-contact
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=dlm-k4-k8-soft-contact-%j.out
#SBATCH --error=dlm-k4-k8-soft-contact-%j.err
set -Eeuo pipefail
ROOT="${CONTACT_CANDIDATE_ROOT:?}"
RUN="${CONTACT_ROOT:?}/runs/eval_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${CONTACT_OBJECT_SOURCE:?}" && git rev-parse "${CONTACT_CODE_COMMIT:?}")" = "$CONTACT_CODE_COMMIT"
(cd "$CONTACT_OBJECT_SOURCE" && git archive "$CONTACT_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$CONTACT_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${CONTACT_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
PY="${CONTACT_PYTHON:?}"
GPUS=6
CRYS="${CONTACT_CRYSLLMGEN:?}"
CONFIG="${CONTACT_FROZEN_CONFIG:?}"
CACHE="${CONTACT_OFFICIAL_CACHE:?}"
CONDITIONS="${CONTACT_CONDITIONS:?}"
INPUTS="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_prefix_inputs.json"
SPEC="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_physics/candidate_spec.json"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$RUN/LAUNCH_MANIFEST.json" "$CONTACT_CODE_COMMIT" "$RUN/ENTRY.sbatch" "$SPEC" <<'PY'
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
sample() {
  local policy="$1" stop="$2" directory="$3" world="$4"
  local common=(--model-path "${CONTACT_MODEL_PATH:?}" --checkpoint-path "$policy" --conditions "$CONDITIONS"
    --condition-start 0 --condition-stop "$stop" --seed 20260905 --collection-round 0 --world-size "$world"
    --candidates 1 --batch-size 4 --legacy-layout-world-size 2 --temperature .7 --purpose evaluation
    --short-contact-spec "$SPEC" --output-dir "$directory")
  local pids=() rank pid
  for ((rank=0; rank<world; rank++)); do
    LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_short_contact_paths.py" \
      "${common[@]}" >"${directory}.rank${rank}.out" 2>"${directory}.rank${rank}.err" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$CODE/src/scripts/sample_short_contact_paths.py" "${common[@]}" --merge-only \
    >"${directory}.merge.out" 2>"${directory}.merge.err"
}
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
  mkdir "$directory" "$directory/pilot"
  policy="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policies"][sys.argv[2]]["path"])' "$RUN/LAUNCH_MANIFEST.json" "$method")"
  LOCAL_RANK=0 RANK=0 "$PY" "$CODE/src/scripts/probe_short_contact_prefixes.py" --method "$method" \
    --model-path "$CONTACT_MODEL_PATH" --input-manifest "$INPUTS" --short-contact-spec "$SPEC" \
    --output-dir "$directory/prefix-probe" --reference-world-size 2 --batch-size 4 --first-conditions 4 \
    --replay-tolerance 1e-6 >"$directory/prefix-probe.out" 2>"$directory/prefix-probe.err"
  test -f "$directory/prefix-probe/_SUCCESS"
  sample "$policy" 4 "$directory/pilot/sample" 2
  LOCAL_RANK=0 RANK=0 "$PY" "$CODE/src/scripts/sample_short_contact_paths.py" \
    --model-path "$CONTACT_MODEL_PATH" --checkpoint-path "$policy" --conditions "$CONDITIONS" \
    --output-dir "$directory/pilot/replay" --condition-start 0 --condition-stop 4 --world-size 1 \
    --seed 20260905 --collection-round 0 --candidates 1 --batch-size 4 --legacy-layout-world-size 2 \
    --temperature .7 --purpose evaluation --short-contact-spec "$SPEC" \
    --replay-jsonl "$directory/pilot/sample/paths.jsonl" --replay-tolerance 1e-6 \
    >"$directory/pilot/replay.out" 2>"$directory/pilot/replay.err"
  test -f "$directory/pilot/replay/_SUCCESS"
  sample "$policy" 256 "$directory/sample" "$GPUS"
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
report = {"method": sys.argv[2]+"_short_contact_v1", "policy_path": sys.argv[3], "requests": 256,
          "spec_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest(),
          "counts": counts, "qualified_endpoints": qualified, "supplement_1000_required": bool(qualified),
          "qualification_rule": "same endpoint strict_sun>=26 AND meta_sun>=128; all256",
          "outcome_selection": False, "inference_mlip": False, "new_optimizer_steps": 0}
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
(r/"EVAL_FINAL.json").write_text(json.dumps({"methods": results, "same_fixed_spec": True}, indent=2)+"\n")
PY
touch "$RUN/_SUCCESS"
