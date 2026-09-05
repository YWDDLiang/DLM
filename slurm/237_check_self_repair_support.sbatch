#!/usr/bin/env bash
#SBATCH --job-name=spad-self-repair-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --output=spad-self-repair-check-%j.out
#SBATCH --error=spad-self-repair-check-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_self_repair_check_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
POLICY="$ROOT/runs/spad_state_path_train_39892/train/checkpoints/step-1020"
CONDITIONS="$ROOT/runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl"
PARENTS="$ROOT/runs/spad_state_paths_39872/sample/paths.jsonl"
LABELS="$ROOT/data/spad_state_labels_round0_consistent_20260906_v2/labels.jsonl"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-start 0 --condition-stop 128
  --repair-roots-jsonl "$PARENTS" --collection-round 2 --seed 20260905
  --candidates 4 --batch-size 4 --world-size 4 --output-dir "$RUN/sample")
pids=()
for rank in 0 1 2 3; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
"$PY" "$CODE/scripts/label_programmed_paths.py" --input-jsonl "$RUN/sample/paths.jsonl" \
  --output-dir "$RUN/labels" --purpose train --gpu-count 4 --workers-per-gpu 2 >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" "$CODE/scripts/diagnose_self_repair_support.py" --paths-jsonl "$RUN/sample/paths.jsonl" \
  --labels-jsonl "$RUN/labels/labels.jsonl" --parent-labels-jsonl "$LABELS" --expected-roots 128 \
  --output-dir "$RUN/support" >"$RUN/support.out" 2>"$RUN/support.err"
touch "$RUN/_SUCCESS"
