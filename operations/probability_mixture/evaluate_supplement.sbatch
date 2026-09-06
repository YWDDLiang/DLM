#!/usr/bin/env bash
#SBATCH --job-name=dlm-probability-mixture-supplement
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=03:00:00
#SBATCH --output=dlm-probability-mixture-supplement-%j.out
#SBATCH --error=dlm-probability-mixture-supplement-%j.err
set -Eeuo pipefail
LOCK="${MIXTURE_METHOD_LOCK:?}"
SOURCE="${MIXTURE_OBJECT_SOURCE:?}"
CODE_COMMIT="${MIXTURE_CODE_COMMIT:?}"
PY="${MIXTURE_PYTHON:?}"
COHORT="${MIXTURE_SUPPLEMENT_COHORT:?}"
CACHE="${MIXTURE_SUPPLEMENT_CACHE:?}"
CONFIG="${MIXTURE_FROZEN_CONFIG:?}"
CRYS="${MIXTURE_CRYSLLMGEN:?}"
GPUS=6
test -f "$COHORT/_SUCCESS"
test "$(cd "$SOURCE" && git rev-parse "$CODE_COMMIT")" = "$CODE_COMMIT"
K4="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_paths"]["k4"])' "$LOCK")"
K8="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_paths"]["k8"])' "$LOCK")"
METHOD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["method"])' "$LOCK")"
test "$METHOD" = k4k8_equal_same_state_probability_mixture_v1
RUN="${MIXTURE_ROOT:?}/runs/supplement_${METHOD}_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git archive "$CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "$LOCK" "$RUN/METHOD_LOCK.json"
CODE="$RUN/code"
SPEC="$CODE/docs/v3_scientific_audit_20260906/evidence_v3_failure_20260907/k4k8_mixture/candidate_spec.json"
CONDITIONS="$COHORT/cohort/plans_for_dlm.jsonl"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$LOCK" "$CODE_COMMIT" "$SPEC" "$CONDITIONS" "$RUN/ENTRY.sbatch" <<'PY'
import hashlib, json, pathlib, sys
lock = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert lock["method_frozen"] is True and lock["qualified_for_1000"] is True
assert lock["code_commit"] == sys.argv[2]
assert lock["spec_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
assert lock["conditions_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[4]).read_bytes()).hexdigest()
assert lock["entry_sha256"] == hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest()
assert (lock["planner_seed"], lock["dlm_seed"], lock["cohort_requested"], lock["parser_only_target"]) == (27, 20260906, 1200, 1000)
assert lock["weights"] == [.5, .5] and lock["temperature"] == .7 and lock["contact_enabled"] is False
for name, entry in lock["policies"].items():
    policy = pathlib.Path(entry["path"])
    assert str(policy) == lock["policy_paths"][name]
    train = json.loads((policy.parent.parent/"TRAIN_FINAL.json").read_text())
    assert train["eligible_policy"] is True and pathlib.Path(train["policy_path"]).resolve() == policy.resolve()
    for filename, digest in entry["files_sha256"].items():
        assert hashlib.sha256((policy/filename).read_bytes()).hexdigest() == digest, (name, filename)
development = pathlib.Path(lock["development_final_path"])
assert (development.parent/"_SUCCESS").is_file()
assert hashlib.sha256(development.read_bytes()).hexdigest() == lock["development_final_sha256"]
report = json.loads(development.read_text())
assert report["method"] == lock["method"] and report["policy_paths"] == lock["policy_paths"]
assert report["spec_sha256"] == lock["spec_sha256"] and report["requests"] == 256
assert any(c["requests"] == 256 and c["strict_sun"] >= 26 and c["meta_sun"] >= 128 for c in report["counts"].values())
for name, item in lock["frozen_inputs"].items():
    assert hashlib.sha256(pathlib.Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], name
PY
"$PY" "$CODE/operations/supplement_1000/check_hull_coverage.py" --plans "$CONDITIONS" --cache "$CACHE" \
  >"$RUN/HULL_COVERAGE_BEFORE_SAMPLING.json"
STOP="$("$PY" -c 'import sys; print(sum(bool(x.strip()) for x in open(sys.argv[1])))' "$CONDITIONS")"
common=(--model-path "${MIXTURE_MODEL_PATH:?}" --checkpoint-path "$K4" --mixture-peer-checkpoint "$K8" --conditions "$CONDITIONS"
  --condition-stop "$STOP" --candidates 1 --batch-size 4 --world-size "$GPUS" --sampling-layout-world-size 4
  --purpose evaluation --seed 20260906 --temperature .7 --output-dir "$RUN/sample")
pids=()
for ((rank=0; rank<GPUS; rank++)); do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_probability_mixture_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_probability_mixture_paths.py" "${common[@]}" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/select_programmed_path_cohort.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --planner-ledger "$COHORT/cohort/ledger.jsonl" --target 1000 --output-dir "$RUN/selection" \
  >"$RUN/selection.out" 2>"$RUN/selection.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/selection/all_requests.jsonl" \
  --native-source body --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" \
  >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
label_evaluate() {
  local stage="$1" endpoint="$2"
  "$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/$stage/paths.jsonl" \
    --output-dir "$RUN/$stage-labels" --purpose evaluation --gpu-count "$GPUS" --workers-per-gpu 2 \
    >"$RUN/$stage-label.out" 2>"$RUN/$stage-label.err"
  "$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/$stage/paths.jsonl" \
    --labels-jsonl "$RUN/$stage-labels/labels.jsonl" --frozen-config "$CONFIG" --official-cache "$CACHE" \
    --selection-json "$RUN/selection/SELECTION_FINAL.json" --output-dir "$RUN/$stage-evaluation" \
    --expected-requests 1200 --endpoint "$endpoint" --cohort-role independent_main --policy-stage final \
    >"$RUN/$stage-evaluate.out" 2>"$RUN/$stage-evaluate.err"
}
label_evaluate native native
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
  --diff-steps 800 --num-evals 1 --max-proposals 1200 >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --native-source body --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" >"$RUN/tau800-artifacts.out" 2>"$RUN/tau800-artifacts.err"
label_evaluate tau800 tau800
test -f "$RUN/native-evaluation/_SUCCESS"
test -f "$RUN/tau800-evaluation/_SUCCESS"
"$PY" - "$RUN" <<'PY'
import json, pathlib, sys
r=pathlib.Path(sys.argv[1])
lock=json.loads((r/"METHOD_LOCK.json").read_text())
endpoints={stage:json.loads((r/f"{stage}-evaluation/EVALUATION_FINAL.json").read_text()) for stage in ("native","tau800")}
assert all(e["counts"]["requests"]==1200 for e in endpoints.values())
(r/"EVAL_FINAL.json").write_text(json.dumps({"method":lock["method"],"policy_paths":lock["policy_paths"],
    "method_lock":lock,"all_requests":{s:e["counts"] for s,e in endpoints.items()},
    "conditional_1000":{s:e["conditional_1000"] for s,e in endpoints.items()},
    "selection_before_energy":True,"same_selection_for_both_endpoints":True},indent=2)+"\n")
PY
touch "$RUN/_SUCCESS"
