#!/usr/bin/env bash
#SBATCH --job-name=dlm-token-construction-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=02:00:00
#SBATCH --output=dlm-token-construction-eval-%j.out
#SBATCH --error=dlm-token-construction-eval-%j.err
set -Eeuo pipefail
ROOT="${MIXED_CANDIDATE_ROOT:?}"
TRAIN_RUN="${MIXED_TRAIN_RUN:?}"
test -f "$TRAIN_RUN/_SUCCESS"
POLICY="$(<"$TRAIN_RUN/POLICY_PATH")"
RUN="${MIXED_ROOT:?}/runs/token_eval_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${MIXED_CODE_COMMIT:?}")" = "$MIXED_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$MIXED_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXED_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${TOKEN_EVAL_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
PY="${MIXED_PYTHON:?}"
GPUS=6
CRYS="${MIXED_CRYSLLMGEN:?}"
CONFIG="${MIXED_FROZEN_CONFIG:?}"
CACHE="${MIXED_OFFICIAL_CACHE:?}"
CONDITIONS="${MIXED_CONDITIONS:?}"
V2="$ROOT/experiments/periodic_self_repair_20260906/v2_from_original_llada/runs/eval_39998"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$RUN/LAUNCH_MANIFEST.json" "$MIXED_CODE_COMMIT" "$CONDITIONS" "$POLICY" "$RUN/ENTRY.sbatch" <<'PY'
import hashlib, json, pathlib, sys
from crystal_dlm.mixed_geometry_sampling import validate_final_policy
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m["code_commit"] == sys.argv[2]
assert m["conditions_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
assert pathlib.Path(m["policy_path"]).resolve() == pathlib.Path(sys.argv[4]).resolve()
assert m["entry_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest()
for name, record in m["frozen_inputs"].items():
    assert hashlib.sha256(pathlib.Path(record["path"]).read_bytes()).hexdigest() == record["sha256"], name
validate_final_policy(sys.argv[4])
PY
# This final diagnostic cannot choose or modify T samples. A failure is preserved.
if timeout 600 "$PY" "$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/trace_frozen_geometry.py" \
  --source-root "$TRAIN_RUN/code" --model-path "$MIXED_MODEL_PATH" --checkpoint-path "$POLICY" \
  --native-paths "$MIXED_ROOT/runs/eval_40066/sample/native/paths.jsonl" --output-dir "$RUN/g-trace" --device cuda:0 \
  >"$RUN/g-trace.out" 2>"$RUN/g-trace.err"; then
  printf '%s\n' completed >"$RUN/G_TRACE_STATUS"
else
  trace_rc=$?
  printf 'diagnostic_failed exit_code=%s\n' "$trace_rc" >"$RUN/G_TRACE_STATUS"
fi
sample() {
  local stop="$1" directory="$2" world="$3"
  local common=(--model-path "${MIXED_MODEL_PATH:?}" --checkpoint-path "$POLICY" --conditions "$CONDITIONS"
    --condition-start 0 --condition-stop "$stop" --seed 20260905 --collection-round 0 --world-size "$world"
    --batch-size 4 --sampling-layout-world-size 2 --temperature .7 --output-dir "$directory")
  local pids=() rank pid
  for ((rank=0; rank<world; rank++)); do
    LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_mixed_token_paths.py" \
      "${common[@]}" >"${directory}.rank${rank}.out" 2>"${directory}.rank${rank}.err" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$CODE/src/scripts/sample_mixed_token_paths.py" "${common[@]}" --merge-only \
    >"${directory}.merge.out" 2>"${directory}.merge.err"
}
mkdir "$RUN/pilot"
sample 4 "$RUN/pilot/sample" 2
LOCAL_RANK=0 RANK=0 "$PY" "$CODE/src/scripts/sample_mixed_token_paths.py" \
  --model-path "$MIXED_MODEL_PATH" --checkpoint-path "$POLICY" --conditions "$CONDITIONS" \
  --output-dir "$RUN/pilot/replay" --condition-start 0 --condition-stop 4 --world-size 1 --rank 0 \
  --replay-jsonl "$RUN/pilot/sample/native/paths.jsonl" --replay-tolerance 1e-6 \
  >"$RUN/pilot/replay.out" 2>"$RUN/pilot/replay.err"
test -f "$RUN/pilot/replay/_SUCCESS"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "PASS"' "$RUN/pilot/replay/REPLAY.json"
sample 256 "$RUN/sample" "$GPUS"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/native/paths.jsonl" \
  --native-source body --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" \
  >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
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
refine() {
  local native="$1" refined="$2" payload
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "$RUN/$native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
    --crysllmgen-dir "$CRYS" --output-dir "$RUN/$refined-refine" --batch-size 1 --seed-from-graph-field refiner_seed \
    --diff-steps 800 --num-evals 1 --max-proposals 256 >"$RUN/$refined-refine.out" 2>"$RUN/$refined-refine.err"
  payload="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/$refined-refine/refinement_metrics.json")"
  "$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/$native/paths.jsonl" \
    --native-source body --refined-pt "$payload" --output-dir "$RUN/$refined" \
    >"$RUN/$refined-artifacts.out" 2>"$RUN/$refined-artifacts.err"
  label_evaluate "$refined" tau800
}
label_evaluate native native
refine native tau800
# The old construction endpoint and raw labels are reused only after exact binding.
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$V2/construction-traces/paths.jsonl" \
  --native-source body --output-dir "$RUN/v2-construction" --crysllmgen-dir "$CRYS" \
  >"$RUN/v2-construction-artifacts.out" 2>"$RUN/v2-construction-artifacts.err"
"$PY" "$CODE/operations/mixed_geometry/verify_construction_reexport.py" \
  --original-paths "$V2/construction/paths.jsonl" --original-labels "$V2/construction-labels/labels.jsonl" \
  --exported-paths "$RUN/v2-construction/paths.jsonl" --output "$RUN/V2_CONSTRUCTION_BINDING.json" \
  >"$RUN/v2-construction-binding.out" 2>"$RUN/v2-construction-binding.err"
refine v2-construction v2-construction-tau800
compare() {
  local ref_eval="$1" ref_paths="$2" method="$3" name="$4"
  "$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" --reference-eval-dir "$ref_eval" \
    --method-eval-dir "$RUN/$method-evaluation" --reference-paths-jsonl "$ref_paths" \
    --method-paths-jsonl "$RUN/$method/paths.jsonl" --output-dir "$RUN/$name" --comparison-purpose final \
    >"$RUN/$name.out" 2>"$RUN/$name.err"
}
compare "$V2/construction-evaluation" "$V2/construction/paths.jsonl" native native-vs-v2-construction
compare "$RUN/v2-construction-tau800-evaluation" "$RUN/v2-construction-tau800/paths.jsonl" tau800 tau800-vs-v2-construction
compare "$ROOT/runs/spad_state_tau800_39910/evaluation" "$ROOT/runs/spad_state_tau800_39910/tau800/paths.jsonl" tau800 tau800-vs-k4
compare "$ROOT/runs/spad_state_tau800_39948/evaluation" "$ROOT/runs/spad_state_tau800_39948/tau800/paths.jsonl" tau800 tau800-vs-k8
"$PY" - "$RUN" "$POLICY" "$V2" <<'PY'
import json, pathlib, sys
run, v2 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3])
directories = {name: run/f"{name}-evaluation" for name in ("native", "tau800", "v2-construction-tau800")}
directories["v2-construction-native"] = v2/"construction-evaluation"
results = {}
for name, directory in directories.items():
    assert (directory/"_SUCCESS").is_file(), name
    report = json.loads((directory/"EVALUATION_FINAL.json").read_text())
    assert report["counts"]["requests"] == 256, name
    results[name] = report["counts"]
qualified = [name for name, counts in results.items() if counts["strict_sun"] >= 26 and counts["meta_sun"] >= 128]
report = {"method": "mixed_token_construction", "training_method": "mixed_geometry_h_p33",
          "policy_path": sys.argv[2], "requests": 256, "counts": results,
          "qualified_endpoints": qualified, "supplement_1000_required": bool(qualified),
          "qualification_rule": "same endpoint strict_sun>=26 AND meta_sun>=128; all256",
          "outcome_selection": False, "pilot_used_for_quality_selection": False,
          "inference_mlip": False, "geometry_forward_calls": 0, "new_optimizer_steps": 0}
(run/"EVAL_FINAL.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
PY
touch "$RUN/_SUCCESS"
