#!/usr/bin/env bash
#SBATCH --job-name=dlm-k4-k8-probability-mixture
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=dlm-k4-k8-mixture-%j.out
#SBATCH --error=dlm-k4-k8-mixture-%j.err
set -Eeuo pipefail
ROOT="${MIXTURE_CANDIDATE_ROOT:?}"
RUN="${MIXTURE_ROOT:?}/runs/eval_${SLURM_JOB_ID}"
mkdir -p "$RUN/code" "$RUN/pilot"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXTURE_OBJECT_SOURCE:?}" && git rev-parse "${MIXTURE_CODE_COMMIT:?}")" = "$MIXTURE_CODE_COMMIT"
(cd "$MIXTURE_OBJECT_SOURCE" && git archive "$MIXTURE_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXTURE_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${MIXTURE_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
PY="${MIXTURE_PYTHON:?}"
GPUS=6
CRYS="${MIXTURE_CRYSLLMGEN:?}"
CONFIG="${MIXTURE_FROZEN_CONFIG:?}"
CACHE="${MIXTURE_OFFICIAL_CACHE:?}"
CONDITIONS="${MIXTURE_CONDITIONS:?}"
EVIDENCE="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907"
INPUTS="$EVIDENCE/k4k8_prefix_inputs.json"
SPEC="$EVIDENCE/k4k8_mixture/candidate_spec.json"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$RUN/LAUNCH_MANIFEST.json" "$MIXTURE_CODE_COMMIT" "$RUN/ENTRY.sbatch" "$SPEC" <<'PY'
import hashlib, json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m["code_commit"] == sys.argv[2]
assert m["entry_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
assert m["spec_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest()
assert m["weights"] == [.5, .5] and m["temperature"] == .7 and m["contact_enabled"] is False
for name, entry in m["frozen_inputs"].items():
    assert hashlib.sha256(pathlib.Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"], name
for name, entry in m["policies"].items():
    policy = pathlib.Path(entry["path"])
    train = json.loads((policy.parent.parent/"TRAIN_FINAL.json").read_text())
    assert train["eligible_policy"] is True and pathlib.Path(train["policy_path"]).resolve() == policy.resolve()
    for filename, expected in entry["files_sha256"].items():
        assert hashlib.sha256((policy/filename).read_bytes()).hexdigest() == expected, (name, filename)
PY
K4="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policies"]["k4"]["path"])' "$RUN/LAUNCH_MANIFEST.json")"
K8="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policies"]["k8"]["path"])' "$RUN/LAUNCH_MANIFEST.json")"
sample() {
  local stop="$1" directory="$2" world="$3"
  local common=(--model-path "${MIXTURE_MODEL_PATH:?}" --checkpoint-path "$K4" --mixture-peer-checkpoint "$K8"
    --conditions "$CONDITIONS" --condition-start 0 --condition-stop "$stop" --seed 20260905 --collection-round 0
    --world-size "$world" --candidates 1 --batch-size 4 --legacy-layout-world-size 2 --temperature .7
    --purpose evaluation --output-dir "$directory")
  local pids=() rank pid
  for ((rank=0; rank<world; rank++)); do
    LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_probability_mixture_paths.py" \
      "${common[@]}" >"${directory}.rank${rank}.out" 2>"${directory}.rank${rank}.err" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$CODE/src/scripts/sample_probability_mixture_paths.py" "${common[@]}" --merge-only \
    >"${directory}.merge.out" 2>"${directory}.merge.err"
}
label_evaluate() {
  local stage="$1" endpoint="$2"
  "$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/$stage/paths.jsonl" \
    --output-dir "$RUN/$stage-labels" --purpose evaluation --gpu-count "$GPUS" --workers-per-gpu 2 \
    >"$RUN/$stage-label.out" 2>"$RUN/$stage-label.err"
  "$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/$stage/paths.jsonl" \
    --labels-jsonl "$RUN/$stage-labels/labels.jsonl" --frozen-config "$CONFIG" --official-cache "$CACHE" \
    --output-dir "$RUN/$stage-evaluation" --expected-requests 256 --endpoint "$endpoint" \
    --cohort-role fixed_development --policy-stage final >"$RUN/$stage-evaluate.out" 2>"$RUN/$stage-evaluate.err"
}
LOCAL_RANK=0 RANK=0 "$PY" "$EVIDENCE/k4k8_mixture/probe_same_state_mixture.py" \
  --source-root "$CODE" --model-path "$MIXTURE_MODEL_PATH" --input-manifest "$INPUTS" \
  --output-dir "$RUN/prefix-probe" >"$RUN/prefix-probe.out" 2>"$RUN/prefix-probe.err"
test -f "$RUN/prefix-probe/_SUCCESS"
"$PY" - "$RUN/prefix-probe/PROBE.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["status"] == "PASS" and p["state_count"] == 8 and p["actual_model_forwards"] == 16
assert p["distribution_difference_above_numerical_floor_1e_8"] is True
assert p["energy_evaluations"] == 0 and p["optimizer_steps"] == 0 and p["contact_enabled"] is False
PY
sample 4 "$RUN/pilot/sample" 2
LOCAL_RANK=0 RANK=0 "$PY" "$CODE/src/scripts/sample_probability_mixture_paths.py" \
  --model-path "$MIXTURE_MODEL_PATH" --checkpoint-path "$K4" --mixture-peer-checkpoint "$K8" \
  --conditions "$CONDITIONS" --output-dir "$RUN/pilot/replay" --condition-start 0 --condition-stop 4 --world-size 1 \
  --seed 20260905 --collection-round 0 --candidates 1 --batch-size 4 --legacy-layout-world-size 2 \
  --temperature .7 --purpose evaluation --replay-jsonl "$RUN/pilot/sample/paths.jsonl" --replay-tolerance 1e-6 \
  >"$RUN/pilot/replay.out" 2>"$RUN/pilot/replay.err"
test -f "$RUN/pilot/replay/_SUCCESS"
sample 256 "$RUN/sample" "$GPUS"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --native-source body --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" \
  >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
label_evaluate native native
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
  --diff-steps 800 --num-evals 1 --max-proposals 256 >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --native-source body --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" >"$RUN/tau800-artifacts.out" 2>"$RUN/tau800-artifacts.err"
label_evaluate tau800 tau800
for reference in k4 k8; do
  if [[ "$reference" == k4 ]]; then
    original_native="$ROOT/runs/spad_state_eval_method_39910"
    original_tau="$ROOT/runs/spad_state_tau800_39910"
  else
    original_native="$ROOT/runs/spad_state_eval_method_39945"
    original_tau="$ROOT/runs/spad_state_tau800_39948"
  fi
  for stage in native tau800; do
    if [[ "$stage" == native ]]; then
      original="$original_native"
    else
      original="$original_tau"
    fi
    "$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" --reference-eval-dir "$original/evaluation" \
      --method-eval-dir "$RUN/$stage-evaluation" --reference-paths-jsonl "$original/$stage/paths.jsonl" \
      --method-paths-jsonl "$RUN/$stage/paths.jsonl" --output-dir "$RUN/$stage-vs-$reference" --comparison-purpose final \
      >"$RUN/$stage-vs-$reference.out" 2>"$RUN/$stage-vs-$reference.err"
  done
done
"$PY" - "$RUN" "$K4" "$K8" "$SPEC" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
counts = {}
for name in ("native", "tau800"):
    d = r/f"{name}-evaluation"
    assert (d/"_SUCCESS").is_file(), name
    counts[name] = json.loads((d/"EVALUATION_FINAL.json").read_text())["counts"]
    assert counts[name]["requests"] == 256
qualified = [name for name, c in counts.items() if c["strict_sun"] >= 26 and c["meta_sun"] >= 128]
report = {"method": "k4k8_equal_same_state_probability_mixture_v1", "policy_paths": {"k4": sys.argv[2], "k8": sys.argv[3]},
          "requests": 256, "weights": [.5, .5], "temperature": .7, "counts": counts,
          "spec_sha256": hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest(),
          "qualified_endpoints": qualified, "supplement_1000_required": bool(qualified),
          "qualification_rule": "same endpoint strict_sun>=26 AND meta_sun>=128; all256",
          "outcome_selection": False, "inference_mlip": False, "new_optimizer_steps": 0,
          "model_forwards_per_action": 2, "contact_enabled": False}
(r/"EVAL_FINAL.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
PY
touch "$RUN/_SUCCESS"
