#!/usr/bin/env bash
#SBATCH --job-name=spad-state-production-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=spad-state-production-check-%j.out
#SBATCH --error=spad-state-production-check-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_state_production_check_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
git -C "$SOURCE" rev-parse HEAD >"$RUN/CODE_COMMIT"
git -C "$SOURCE" archive HEAD | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
POLICY="$ROOT/runs/spad_state_warmup_39853/train/checkpoints/step-1696"
CONDITIONS="$ROOT/runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl"
test -f "$ROOT/runs/spad_state_warmup_39853/_SUCCESS"
test -f "$ROOT/runs/spad_state_train_conditions_39857/_SUCCESS"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-stop 2 --candidates 1 --batch-size 1 --world-size 2)
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" --output-dir "$RUN/sample" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" \
  --output-dir "$RUN/sample" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
pids=()
for rank in 0 1; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" --output-dir "$RUN/replay.rank${rank}" --replay-jsonl "$RUN/sample/records.rank${rank}.jsonl" \
    >"$RUN/replay.rank${rank}.out" 2>"$RUN/replay.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/labels" --gpu-count 2 --workers-per-gpu 1 >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); bad={"worker_error","evaluation_error","energy_protocol_mismatch","optimizer_stop_unverified"}; assert p["completed"]==2; assert not bad.intersection(p["statuses"]),p; print(p)' "$RUN/labels/LABEL_FINAL.json" >"$RUN/protocol.out"
touch "$RUN/_SUCCESS"
