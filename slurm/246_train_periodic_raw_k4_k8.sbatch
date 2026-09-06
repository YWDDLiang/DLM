#!/usr/bin/env bash
#SBATCH --job-name=periodic-path-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=periodic-path-train-%j.out
#SBATCH --error=periodic-path-train-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
ROUND="${PERIODIC_POSTTRAIN_ROUND:?}"
COLLECT="${PERIODIC_POSTTRAIN_COLLECT_RUN:?}"
POLICY="${PERIODIC_POSTTRAIN_POLICY:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
EXPERIMENT="$ROOT/experiments/periodic_self_repair_20260906/from_original_llada_v1"
test -f "$COLLECT/_SUCCESS"
test -f "$COLLECT/teacher/_SUCCESS"
test -f "$POLICY/raw_periodic_initialization.json"
extra=()
if [[ "$ROUND" == 1 ]]; then
  STATE="${PERIODIC_RAW_K4_POST_STATE:?}"
  test -f "$STATE"
  extra+=(--optimizer-state "$STATE")
elif [[ "$ROUND" != 0 ]]; then
  exit 2
fi
RUN="$EXPERIMENT/path_posttraining/runs/train_r${ROUND}_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
export PYTHONPATH="$RUN/code/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$RUN/code/src/scripts/train_programmed_path_policy.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --checkpoint-path "$POLICY" --teacher-json "$COLLECT/teacher/teacher.json" \
  --ce-data-jsonl "$ROOT/data/spad_basin_closure_sft_v1_20260904/train.jsonl" \
  --output-dir "$RUN/train" --batch-size 2 --effective-batch 16 \
  --passes 2 --learning-rate 5e-6 --warmup-updates 100 --schedule-updates 7680 \
  --seed 20260905 "${extra[@]}" >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
