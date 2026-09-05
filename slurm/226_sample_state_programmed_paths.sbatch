#!/usr/bin/env bash
#SBATCH --job-name=spad-state-paths
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=05:00:00
#SBATCH --output=spad-state-paths-%j.out
#SBATCH --error=spad-state-paths-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
test -f "${SPAD_PRODUCTION_CHECK_RUN:?}/_SUCCESS"
RUN="$ROOT/runs/spad_state_paths_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
POLICY="${SPAD_POLICY_PATH:-$ROOT/runs/spad_state_warmup_39853/train/checkpoints/step-1696}"
CONDITIONS="$ROOT/runs/spad_state_train_conditions_39857/conditions/plans_for_dlm.jsonl"
common=(--model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct --checkpoint-path "$POLICY"
  --conditions "$CONDITIONS" --condition-start "${SPAD_CONDITION_START:-0}" --condition-stop "${SPAD_CONDITION_STOP:-128}"
  --collection-round "${SPAD_COLLECTION_ROUND:-0}" --candidates 4 --batch-size 4 --world-size 4 --output-dir "$RUN/sample")
pids=()
for rank in 0 1 2 3; do
  LOCAL_RANK="$rank" RANK="$rank" "$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" \
    "${common[@]}" >"$RUN/sample.rank${rank}.out" 2>"$RUN/sample.rank${rank}.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" "$CODE/src/scripts/sample_state_programmed_paths.py" "${common[@]}" --merge-only >"$RUN/merge.out" 2>"$RUN/merge.err"
test -f "$RUN/sample/_SUCCESS"
touch "$RUN/_SUCCESS"
