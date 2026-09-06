#!/usr/bin/env bash
#SBATCH --job-name=periodic-dlm-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=periodic-dlm-eval-%j.out
#SBATCH --error=periodic-dlm-eval-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
TRAIN_RUN="${PERIODIC_RAW_TRAIN_RUN:?}"
RAW_STAGE="${PERIODIC_RAW_STAGE:-base}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
PROJECT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
CONFIG="$PROJECT/workstreams/final_method_development_20260808/execution/h1a2_epoch2_exactplan1200_h1a2_r03_refine800_fullsun1000_v3/CONFIG.json"
CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1"
test -f "$TRAIN_RUN/_SUCCESS"
POLICY="$(<"$TRAIN_RUN/POLICY_PATH")"
"$PY" - "$TRAIN_RUN/train/TRAIN_FINAL.json" "$POLICY" "$RAW_STAGE" <<'PY'
import json, pathlib, sys
report=json.load(open(sys.argv[1]))
policy=pathlib.Path(sys.argv[2]).resolve()
stage=sys.argv[3]
assert report["eligible_policy"] is True
assert pathlib.Path(report["policy_path"]).resolve() == policy
assert (policy/"raw_periodic_initialization.json").is_file()
if stage == "base":
    assert report["method"] == "periodic_dlm_from_original_llada_v1"
    assert report["old_dlm_weights_loaded"] is False
    assert report["K4_K8_data_read"] is False
elif stage in ("k4", "k8"):
    assert report["method"] == "periodic_raw_k4_k8_path_posttraining_v1"
    assert report["raw_periodic_posttraining"] is True
    assert report["legacy_K4_K8_paths_read"] is False
    assert report["collection_round"] == {"k4": 0, "k8": 1}[stage]
else:
    raise AssertionError(stage)
PY
RUN="$EXPERIMENT/runs/eval_native_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
CONDITIONS="$ROOT/cohorts/spad_prospective_seed23_256_v1_20260903/plans_for_dlm.jsonl"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
  --checkpoint-path "$POLICY" --conditions "$CONDITIONS" --condition-stop 256
  --candidates 1 --batch-size 4 --world-size 2 --purpose evaluation
  --seed 20260905 --full-cell-repair --output-dir "$RUN/sample")
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only \
  >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/export_programmed_path_artifacts.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/native" --crysllmgen-dir "$CRYS" >"$RUN/artifacts.out" 2>"$RUN/artifacts.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/native/paths.jsonl" \
  --output-dir "$RUN/labels" --purpose evaluation --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" "$CODE/scripts/evaluate_programmed_paths.py" --paths-jsonl "$RUN/native/paths.jsonl" \
  --labels-jsonl "$RUN/labels/labels.jsonl" --frozen-config "$CONFIG" \
  --official-cache "$ROOT/runs/spad_basin_closure_official_20260904_v1/official_mp_cache" \
  --output-dir "$RUN/evaluation" --expected-requests 256 --endpoint native \
  --cohort-role fixed_development --policy-stage final >"$RUN/evaluate.out" 2>"$RUN/evaluate.err"
"$PY" "$CODE/scripts/compare_programmed_path_evaluations.py" \
  --reference-eval-dir "$ROOT/runs/spad_state_reference_verification_39893/native-evaluation" \
  --method-eval-dir "$RUN/evaluation" \
  --reference-paths-jsonl "$ROOT/runs/spad_state_eval_reference_39884/native/paths.jsonl" \
  --method-paths-jsonl "$RUN/native/paths.jsonl" --output-dir "$RUN/comparison" \
  --comparison-purpose final >"$RUN/compare.out" 2>"$RUN/compare.err"
test -f "$RUN/evaluation/_SUCCESS"
test -f "$RUN/comparison/_SUCCESS"
touch "$RUN/_SUCCESS"
