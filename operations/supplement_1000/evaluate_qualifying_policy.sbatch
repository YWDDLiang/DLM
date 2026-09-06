#!/usr/bin/env bash
#SBATCH --job-name=periodic-supp1000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=periodic-supp1000-%j.out
#SBATCH --error=periodic-supp1000-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
COHORT_RUN="${SPAD_MAIN_COHORT_RUN:?}"
WORLD="${SUPPLEMENT_WORLD_SIZE:-2}"
[[ "$WORLD" =~ ^(2|4|6)$ ]]
SUPPLEMENT="${SUPPLEMENT_ROOT:?}"
VARIANT="${SUPPLEMENT_VARIANT:?}"
[[ "$VARIANT" =~ ^[a-z0-9_]+$ ]]
LOCK="${SUPPLEMENT_METHOD_LOCK:?}"
test "$(cd "$SOURCE" && git rev-parse HEAD)" = 2e904c260bafb6750c9a5dbb0d920c4ff8c3a868
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
test -f "$COHORT_RUN/_SUCCESS"
test -f "$LOCK"
POLICY=$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["method_frozen"] is True; assert r["planner_seed"]==27 and r["dlm_seed"]==20260906; print(r["policy_path"])' "$LOCK")
"$PY" - "$POLICY" "$LOCK" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]).resolve()
lock=json.load(open(sys.argv[2]))
report=json.loads((p.parent.parent/"TRAIN_FINAL.json").read_text())
assert report["eligible_policy"] is True
assert pathlib.Path(report["policy_path"]).resolve()==p
assert lock["qualified_for_1000"] is True
assert lock["qualification_rule"] in ("final_sun_goal", "posttrain_admission")
assert lock["cohort_requested"]==1200 and lock["parser_only_target"]==1000
PY
mkdir -p "$SUPPLEMENT/runs"
RUN="$SUPPLEMENT/runs/${VARIANT}_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
cp "$LOCK" "$RUN/METHOD_LOCK.json"
cp "${BASH_SOURCE[0]}" "$RUN/OPERATIONS_ENTRY.sbatch"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
CONDITIONS="$COHORT_RUN/cohort/plans_for_dlm.jsonl"
STOP=$("$PY" -c 'import sys; print(sum(bool(l.strip()) for l in open(sys.argv[1])))' "$CONDITIONS")
extra=()
if test -f "$POLICY/raw_periodic_initialization.json"; then extra+=(--full-cell-repair); fi
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-stop "$STOP" --candidates 1 --batch-size 4 --world-size "$WORLD" --sampling-layout-world-size 4 "${extra[@]}"
  --purpose evaluation --seed 20260906 --output-dir "$RUN/sample")
pids=()
for ((rank=0; rank<WORLD; rank++)); do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/select_programmed_path_cohort.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --planner-ledger "$COHORT_RUN/cohort/ledger.jsonl" --target 1000 --output-dir "$RUN/selection" \
  >"$RUN/selection.out" 2>"$RUN/selection.err"
# Freeze the parser-only prefix BEFORE any energy call. Evaluate the full 1200
# requests too, so the conditional denominator never hides requested failures.
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/selection/all_requests.jsonl" \
  --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/native/paths.jsonl" \
  --output-dir "$RUN/native-labels" --purpose evaluation --gpu-count "$WORLD" --workers-per-gpu 2 \
  >"$RUN/native-label.out" 2>"$RUN/native-label.err"
"$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --labels-jsonl "$RUN/native-labels/labels.jsonl" --frozen-config "$CONFIG" \
  --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
  --selection-json "$RUN/selection/SELECTION_FINAL.json" --output-dir "$RUN/native-evaluation" \
  --expected-requests 1200 --endpoint native --cohort-role independent_main --policy-stage final >"$RUN/native-evaluate.out" 2>"$RUN/native-evaluate.err"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
  --diff-steps 800 --num-evals 1 --max-proposals 1200 >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" >"$RUN/tau800-artifacts.out" 2>"$RUN/tau800-artifacts.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/tau800/paths.jsonl" \
  --output-dir "$RUN/tau800-labels" --purpose evaluation --gpu-count "$WORLD" --workers-per-gpu 2 \
  >"$RUN/tau800-label.out" 2>"$RUN/tau800-label.err"
"$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/tau800/paths.jsonl" \
  --labels-jsonl "$RUN/tau800-labels/labels.jsonl" --frozen-config "$CONFIG" \
  --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
  --selection-json "$RUN/selection/SELECTION_FINAL.json" --output-dir "$RUN/tau800-evaluation" \
  --expected-requests 1200 --endpoint tau800 --cohort-role independent_main --policy-stage final >"$RUN/tau800-evaluate.out" 2>"$RUN/tau800-evaluate.err"
test -f "$RUN/native-evaluation/_SUCCESS"
test -f "$RUN/tau800-evaluation/_SUCCESS"
touch "$RUN/_SUCCESS"
