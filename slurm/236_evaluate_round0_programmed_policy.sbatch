#!/usr/bin/env bash
#SBATCH --job-name=spad-state-round0-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=180G
#SBATCH --time=03:00:00
#SBATCH --output=spad-state-round0-eval-%j.out
#SBATCH --error=spad-state-round0-eval-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
POLICY="${SPAD_ROUND0_POLICY_PATH:?}"
REFERENCE="${SPAD_MATCHED_REFERENCE_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_state_round0_eval_${SLURM_JOB_ID}"
mkdir "$RUN"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
test -f "$REFERENCE/native-evaluation/_SUCCESS"
test -f "$REFERENCE/tau800-evaluation/_SUCCESS"
export SPAD_EVAL_ROLE=method SPAD_EVAL_STAGE=round0_diagnostic SPAD_EVAL_POLICY_PATH="$POLICY"
# Both complete workflows execute sequentially inside this one two-GPU allocation.
bash "$SOURCE/slurm/230_evaluate_state_programmed_paths.sbatch"
NATIVE="$ROOT/runs/spad_state_eval_method_${SLURM_JOB_ID}"
export SPAD_NATIVE_EVAL_RUN="$NATIVE"
bash "$SOURCE/slurm/231_refine_state_programmed_paths.sbatch"
TAU="$ROOT/runs/spad_state_tau800_${SLURM_JOB_ID}"
NATIVE_REFERENCE=$(cat "$REFERENCE/NATIVE_REFERENCE_RUN")
TAU_REFERENCE=$(cat "$REFERENCE/TAU_REFERENCE_RUN")
"$PY" "$NATIVE/code/scripts/compare_programmed_path_evaluations.py" \
  --reference-eval-dir "$REFERENCE/native-evaluation" --method-eval-dir "$NATIVE/evaluation" \
  --reference-paths-jsonl "$NATIVE_REFERENCE/native/paths.jsonl" --method-paths-jsonl "$NATIVE/native/paths.jsonl" \
  --comparison-purpose round0_diagnostic --output-dir "$RUN/native-comparison" >"$RUN/native-comparison.out" 2>"$RUN/native-comparison.err"
"$PY" "$TAU/code/scripts/compare_programmed_path_evaluations.py" \
  --reference-eval-dir "$REFERENCE/tau800-evaluation" --method-eval-dir "$TAU/evaluation" \
  --reference-paths-jsonl "$TAU_REFERENCE/tau800/paths.jsonl" --method-paths-jsonl "$TAU/tau800/paths.jsonl" \
  --comparison-purpose round0_diagnostic --output-dir "$RUN/tau800-comparison" >"$RUN/tau800-comparison.out" 2>"$RUN/tau800-comparison.err"
printf '%s\n' "$POLICY" >"$RUN/POLICY_PATH"
printf '%s\n' round0_diagnostic >"$RUN/EVALUATION_STAGE"
touch "$RUN/_SUCCESS"
