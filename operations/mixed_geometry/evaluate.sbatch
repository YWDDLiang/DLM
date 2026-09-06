#!/usr/bin/env bash
#SBATCH --job-name=dlm-mixed-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=03:00:00
#SBATCH --output=dlm-mixed-eval-%j.out
#SBATCH --error=dlm-mixed-eval-%j.err
set -Eeuo pipefail
ROOT="${MIXED_CANDIDATE_ROOT:?}"
TRAIN_RUN="${MIXED_TRAIN_RUN:?}"
test -f "$TRAIN_RUN/_SUCCESS"
POLICY="$(<"$TRAIN_RUN/POLICY_PATH")"
RUN="${MIXED_ROOT:?}/runs/eval_${SLURM_JOB_ID}"
mkdir -p "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test "$(cd "${MIXED_OBJECT_SOURCE:?}" && git rev-parse "${MIXED_CODE_COMMIT:?}")" = "$MIXED_CODE_COMMIT"
(cd "$MIXED_OBJECT_SOURCE" && git archive "$MIXED_CODE_COMMIT") | tar -x -C "$RUN/code"
printf '%s\n' "$MIXED_CODE_COMMIT" >"$RUN/CODE_COMMIT"
cp "${BASH_SOURCE[0]}" "$RUN/ENTRY.sbatch"
cp "${MIXED_EVAL_MANIFEST:?}" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
PY="${MIXED_PYTHON:?}"
GPUS="${MIXED_GPUS:-6}"
CRYS="${MIXED_CRYSLLMGEN:?}"
CONFIG="${MIXED_FROZEN_CONFIG:?}"
CACHE="${MIXED_OFFICIAL_CACHE:?}"
CONDITIONS="${MIXED_CONDITIONS:?}"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PY" - "$RUN/LAUNCH_MANIFEST.json" "$MIXED_CODE_COMMIT" "$CONDITIONS" "$POLICY" <<'PY'
import hashlib, json, pathlib, sys
from crystal_dlm.mixed_geometry_sampling import validate_final_policy
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest["code_commit"] != sys.argv[2] or manifest["conditions_sha256"] != hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest():
    raise ValueError("evaluation code or frozen conditions changed")
if pathlib.Path(manifest["policy_path"]).resolve() != pathlib.Path(sys.argv[4]).resolve():
    raise ValueError("evaluation policy differs from its launch manifest")
validate_final_policy(sys.argv[4])
PY
common=(--model-path "${MIXED_MODEL_PATH:?}" --checkpoint-path "$POLICY" --conditions "$CONDITIONS"
  --condition-start 0 --condition-stop 256 --seed 20260905 --collection-round 0 --world-size "$GPUS"
  --batch-size 4 --sampling-layout-world-size 2 --output-dir "$RUN/sample")
pids=()
for ((rank=0; rank<GPUS; rank++)); do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_mixed_geometry_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_mixed_geometry_paths.py" "${common[@]}" --merge-only \
  >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/native/paths.jsonl" \
  --native-source structure --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" \
  >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/quantized/paths.jsonl" \
  --native-source body --output-dir "$RUN/quantized" >"$RUN/quantized-artifacts.out" 2>"$RUN/quantized-artifacts.err"
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
compare() {
  local reference_eval="$1" reference_paths="$2" stage="$3" name="$4"
  "$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" --reference-eval-dir "$reference_eval" \
    --method-eval-dir "$RUN/$stage-evaluation" --reference-paths-jsonl "$reference_paths" \
    --method-paths-jsonl "$RUN/$stage/paths.jsonl" --output-dir "$RUN/$name" --comparison-purpose final \
    >"$RUN/$name.out" 2>"$RUN/$name.err"
  "$PY" "$CODE/scripts/analyze_programmed_path_regressions.py" --reference-eval-dir "$reference_eval" \
    --method-eval-dir "$RUN/$stage-evaluation" --reference-paths-jsonl "$reference_paths" \
    --method-paths-jsonl "$RUN/$stage/paths.jsonl" --output-dir "$RUN/$name-analysis" \
    >"$RUN/$name-analysis.out" 2>"$RUN/$name-analysis.err"
}
label_evaluate native native
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
  --diff-steps 800 --num-evals 1 --max-proposals 256 >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --native-source structure --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" \
  >"$RUN/tau800-artifacts.out" 2>"$RUN/tau800-artifacts.err"
label_evaluate tau800 tau800
label_evaluate quantized native
compare "$ROOT/runs/spad_state_reference_verification_39893/native-evaluation" \
  "$ROOT/runs/spad_state_eval_reference_39884/native/paths.jsonl" native native-vs-reference
compare "$ROOT/runs/spad_state_reference_verification_39893/tau800-evaluation" \
  "$ROOT/runs/spad_state_tau800_39890/tau800/paths.jsonl" tau800 tau800-vs-reference
V2="$ROOT/experiments/periodic_self_repair_20260906/v2_from_original_llada/runs/eval_39998"
compare "$V2/native-evaluation" "$V2/native/paths.jsonl" native native-vs-v2
compare "$V2/tau800-evaluation" "$V2/tau800/paths.jsonl" tau800 tau800-vs-v2
compare "$RUN/native-evaluation" "$RUN/native/paths.jsonl" quantized quantization-effect
"$PY" - "$RUN" "$POLICY" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
results = {}
for name in ("native", "tau800", "quantized"):
    directory = run / f"{name}-evaluation"
    if not (directory / "_SUCCESS").is_file():
        raise ValueError("an evaluation endpoint is incomplete")
    report = json.loads((directory / "EVALUATION_FINAL.json").read_text())
    if report["counts"]["requests"] != 256:
        raise ValueError("fixed evaluation denominator changed")
    results[name] = report["counts"]
qualified = [name for name in ("native", "tau800")
             if results[name]["strict_sun"] >= 26 and results[name]["meta_sun"] >= 128]
final = {"method": "mixed_geometry_h_p33", "policy_path": sys.argv[2], "requests": 256,
         "primary": ["native", "tau800"], "secondary": "quantized", "counts": results,
         "qualified_primary_endpoints": qualified, "supplement_1000_required": bool(qualified),
         "qualification_rule": "same primary endpoint strict_sun>=26 AND meta_sun>=128; all 256 requests",
         "selection_by_outcome": False}
(run / "EVAL_FINAL.json").write_text(json.dumps(final, indent=2) + "\n")
print(json.dumps(final), flush=True)
PY
touch "$RUN/_SUCCESS"
