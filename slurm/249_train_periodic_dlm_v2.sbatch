#!/usr/bin/env bash
#SBATCH --job-name=periodic-v2-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:6
#SBATCH --mem=440G
#SBATCH --time=12:00:00
#SBATCH --output=periodic-v2-train-%j.out
#SBATCH --error=periodic-v2-train-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${PERIODIC_REPAIR_SOURCE:?}"
EXPERIMENT="${PERIODIC_V2_ROOT:?}"
MANIFEST="${PERIODIC_V2_LAUNCH_MANIFEST:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
"$PY" "$SOURCE/scripts/check_periodic_v2_queue.py" --root "$ROOT" --source "$SOURCE" \
  --experiment "$EXPERIMENT" --manifest "$MANIFEST" --allow-job-id "$SLURM_JOB_ID"
mkdir -p "$EXPERIMENT/runs"
RUN="$EXPERIMENT/runs/train_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
cp "$MANIFEST" "$RUN/LAUNCH_MANIFEST.json"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
"$PY" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$CODE/src/scripts/prepare_periodic_v2_plans.py" \
  --pointer-data-dir "$ROOT/data/spad_species_pointer_v1_20260903" \
  --sft-data-dir "$ROOT/data/spad_basin_closure_sft_v1_20260904" \
  --c3fd-checkpoint "$ROOT/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt" \
  --vocabulary "$ROOT/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json" \
  --llama-model /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B \
  --planner-final "$ROOT/runs/c3fd_llama_typed_planner_39051/train_seed85017/final" \
  --pointer-state "$ROOT/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt" \
  --output-dir "$RUN/plans" --batch-size 64 --seed 20260906 \
  >"$RUN/prepare.out" 2>"$RUN/prepare.err"
test -f "$RUN/plans/_SUCCESS"
"$PY" -m torch.distributed.run --standalone --nproc_per_node=6 \
  "$CODE/src/scripts/train_periodic_dlm_v2.py" \
  --model-path /public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct \
  --data-dir "$RUN/plans" --output-dir "$RUN/train" --seed 82017 --data-seed 20260906 \
  >"$RUN/train.out" 2>"$RUN/train.err"
test -f "$RUN/train/_SUCCESS"
cp "$RUN/train/POLICY_PATH" "$RUN/POLICY_PATH"
touch "$RUN/_SUCCESS"
