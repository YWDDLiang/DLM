#!/usr/bin/env bash
#SBATCH --job-name=periodic-self-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=periodic-self-data-%j.out
#SBATCH --error=periodic-self-data-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
TRAIN_RUN="${PERIODIC_RAW_TRAIN_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1"
CONDITIONS="$ROOT/runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl"
test -f "$TRAIN_RUN/_SUCCESS"
test -f "$ROOT/runs/spad_state_train_conditions_39857/_SUCCESS"
POLICY="$(<"$TRAIN_RUN/POLICY_PATH")"
"$PY" - "$TRAIN_RUN/train/TRAIN_FINAL.json" "$POLICY" "$CONDITIONS" <<'PY'
import json, pathlib, sys
report=json.load(open(sys.argv[1]))
policy=pathlib.Path(sys.argv[2]).resolve()
conditions=[json.loads(line) for line in open(sys.argv[3]) if line.strip()]
assert report["method"] == "periodic_dlm_from_original_llada_v1"
assert report["eligible_policy"] is True
assert pathlib.Path(report["policy_path"]).resolve() == policy
assert report["old_dlm_weights_loaded"] is False and report["K4_K8_data_read"] is False
assert (policy/"raw_periodic_initialization.json").is_file()
assert len(conditions) == 1024 and len({str(row["group_id"]) for row in conditions}) == 1024
assert all(row["source_split"] == "train" and row["outcomes_read"] is False for row in conditions)
PY
RUN="$EXPERIMENT/self_improvement/runs/collect_${SLURM_JOB_ID}"
mkdir -p "$EXPERIMENT/self_improvement/runs"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
  --checkpoint-path "$POLICY" --conditions "$CONDITIONS" --condition-stop 1024
  --candidates 4 --batch-size 4 --world-size 2 --purpose train
  --collection-round 0 --seed 20260907 --full-cell-repair --output-dir "$RUN/sample")
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only \
  >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/parent-labels" --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/parent-label.out" 2>"$RUN/parent-label.err"
"$PY" "$CODE/scripts/build_terminal_repair_dataset.py" --mode prepare \
  --parent-paths "$RUN/sample/paths.jsonl" --labels "$RUN/parent-labels/labels.jsonl" \
  --tokenizer "$POLICY" --expected-requests 4096 --output "$RUN/targets" \
  >"$RUN/targets.out" 2>"$RUN/targets.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/targets/target_paths.jsonl" \
  --output-dir "$RUN/target-labels" --gpu-count 2 --workers-per-gpu 2 \
  >"$RUN/target-label.out" 2>"$RUN/target-label.err"
"$PY" "$CODE/scripts/build_terminal_repair_dataset.py" --mode finalize \
  --prepared "$RUN/targets" --labels "$RUN/target-labels/labels.jsonl" \
  --expected-requests 4096 --output "$RUN/data" >"$RUN/finalize.out" 2>"$RUN/finalize.err"
test -f "$RUN/data/_SUCCESS"
touch "$RUN/_SUCCESS"
