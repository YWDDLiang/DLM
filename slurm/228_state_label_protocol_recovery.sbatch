#!/usr/bin/env bash
#SBATCH --job-name=spad-state-label-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=spad-state-label-check-%j.out
#SBATCH --error=spad-state-label-check-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
CHECK="${SPAD_GENERATION_CHECK_RUN:?}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
test -f "$CHECK/sample/_SUCCESS"
test -f "$CHECK/replay.rank0/_SUCCESS"
test -f "$CHECK/replay.rank1/_SUCCESS"
RUN="$ROOT/runs/spad_state_label_check_${SLURM_JOB_ID}"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
printf '%s\n' "$CHECK" >"$RUN/GENERATION_AND_REPLAY_RUN"
export PYTHONPATH="$RUN/code/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" "$RUN/code/scripts/label_programmed_paths.py" --input-jsonl "$CHECK/sample/paths.jsonl" \
  --output-dir "$RUN/labels" --gpu-count 2 --workers-per-gpu 1 >"$RUN/label.out" 2>"$RUN/label.err"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); bad={"worker_error","evaluation_error","energy_protocol_mismatch","optimizer_stop_unverified"}; assert p["completed"]==2; assert not bad.intersection(p["statuses"]),p; print(p)' "$RUN/labels/LABEL_FINAL.json" >"$RUN/protocol.out"
touch "$RUN/_SUCCESS"
