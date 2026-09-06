#!/usr/bin/env bash
#SBATCH --job-name=periodic-v2-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=periodic-v2-eval-%j.out
#SBATCH --error=periodic-v2-eval-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
EXPERIMENT="${PERIODIC_V2_ROOT:?}"
TRAIN_RUN="${PERIODIC_V2_TRAIN_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
test -f "$TRAIN_RUN/_SUCCESS"
test "$(cd "$SOURCE" && git rev-parse HEAD)" = "$(<"$TRAIN_RUN/CODE_COMMIT")"
POLICY="$(<"$TRAIN_RUN/POLICY_PATH")"
"$PY" - "$TRAIN_RUN/train/TRAIN_FINAL.json" "$POLICY" <<'PY'
import json, pathlib, sys
r = json.load(open(sys.argv[1]))
p = pathlib.Path(sys.argv[2]).resolve()
assert r['eligible_policy'] is True and r['updates'] == 4524
assert r['method'] == 'periodic_dlm_v2_from_original_llada'
assert r['old_dlm_weights_loaded'] is False and r['K4_K8_data_read'] is False
assert pathlib.Path(r['policy_path']).resolve() == p
assert (p / 'periodic_v2_config.json').is_file()
PY
RUN="$EXPERIMENT/runs/eval_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
CONDITIONS="$ROOT/cohorts/spad_prospective_seed23_256_v1_20260903/plans_for_dlm.jsonl"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-stop 256 --candidates 1 --batch-size 4 --world-size 6
  --sampling-layout-world-size 2 --purpose evaluation --seed 20260905 --full-cell-repair --output-dir "$RUN/sample")
pids=()
for rank in 0 1 2 3 4 5; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only \
  >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" >"$RUN/native-artifacts.out" 2>"$RUN/native-artifacts.err"
label_evaluate() {
  local stage="$1" endpoint="$2"
  "$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/$stage/paths.jsonl" \
    --output-dir "$RUN/$stage-labels" --purpose evaluation --gpu-count 6 --workers-per-gpu 2 \
    >"$RUN/$stage-label.out" 2>"$RUN/$stage-label.err"
  "$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/$stage/paths.jsonl" \
    --labels-jsonl "$RUN/$stage-labels/labels.jsonl" --frozen-config "$CONFIG" \
    --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
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
"$PY" -m torch.distributed.run --standalone --nproc_per_node=6 "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "$RUN/native/proposal_graphs.pt" --checkpoint "$CRYS/out/mp_20/22042026/203930/model_494.pt" \
  --crysllmgen-dir "$CRYS" --output-dir "$RUN/refine" --batch-size 1 --seed-from-graph-field refiner_seed \
  --diff-steps 800 --num-evals 1 --max-proposals 256 >"$RUN/refine.out" 2>"$RUN/refine.err"
PAYLOAD="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_file"])' "$RUN/refine/refinement_metrics.json")"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --refined-pt "$PAYLOAD" --output-dir "$RUN/tau800" >"$RUN/tau800-artifacts.out" 2>"$RUN/tau800-artifacts.err"
label_evaluate tau800 tau800
compare "$ROOT/runs/spad_state_reference_verification_39893/native-evaluation" \
  "$ROOT/runs/spad_state_eval_reference_39884/native/paths.jsonl" native native-vs-reference
compare "$ROOT/runs/spad_state_reference_verification_39893/tau800-evaluation" \
  "$ROOT/runs/spad_state_tau800_39890/tau800/paths.jsonl" tau800 tau800-vs-reference
RAW_BASE="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1/runs"
RAW_NATIVE="$RAW_BASE/eval_native_$(<"$ROOT/runs/RAW_BASE_NATIVE_EVAL_JOB")"
RAW_TAU="$RAW_BASE/eval_tau800_$(<"$ROOT/runs/RAW_BASE_TAU800_EVAL_JOB")"
compare "$RAW_NATIVE/evaluation" "$RAW_NATIVE/native/paths.jsonl" native native-vs-raw-v1
compare "$RAW_TAU/evaluation" "$RAW_TAU/tau800/paths.jsonl" tau800 tau800-vs-raw-v1
# The pre-repair endpoint is taken from the same attempted trace, never sampled
# again or chosen using its energy. Keep failed constructions in the denominator.
"$PY" "$CODE/scripts/extract_periodic_construction_endpoint.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --checkpoint-path "$POLICY" --output-dir "$RUN/construction-traces" \
  >"$RUN/construction-extract.out" 2>"$RUN/construction-extract.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/construction-traces/paths.jsonl" \
  --output-dir "$RUN/construction" >"$RUN/construction-artifacts.out" 2>"$RUN/construction-artifacts.err"
label_evaluate construction native
compare "$RUN/construction-evaluation" "$RUN/construction/paths.jsonl" native repair-effect
test -f "$RUN/native-evaluation/_SUCCESS"
test -f "$RUN/tau800-evaluation/_SUCCESS"
test -f "$RUN/repair-effect/_SUCCESS"
touch "$RUN/_SUCCESS"
