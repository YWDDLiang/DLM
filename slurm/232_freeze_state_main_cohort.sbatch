#!/usr/bin/env bash
#SBATCH --job-name=spad-state-main-plans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=spad-state-main-plans-%j.out
#SBATCH --error=spad-state-main-plans-%j.err
set -Eeuo pipefail
ROOT="${H1A2_CANDIDATE_ROOT:?}"
SOURCE="${SSCD_SOURCE_ROOT:?}"
LOCK="${SPAD_METHOD_LOCK_JSON:?freeze the final method before main sampling}"
PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
RUN="$ROOT/runs/spad_state_main_plans_${SLURM_JOB_ID}"
"$PY" -c 'import json,pathlib,sys; r=json.load(open(sys.argv[1])); assert r["method_frozen"] is True; assert r["planner_seed"]==27; assert r["dlm_seed"]==20260906; p=pathlib.Path(r["policy_path"]); f=json.loads((p.parent.parent/"TRAIN_FINAL.json").read_text()); assert f["eligible_policy"] is True' "$LOCK"
mkdir "$RUN" "$RUN/code"
trap 'rc=$?; printf "exit_code=%s line=%s\n" "$rc" "$LINENO" >"$RUN/FAILURE.txt"; touch "$RUN/_FAILED"; exit "$rc"' ERR
(cd "$SOURCE" && git rev-parse HEAD) >"$RUN/CODE_COMMIT"
(cd "$SOURCE" && git archive HEAD) | tar -x -C "$RUN/code"
CODE="$RUN/code"
export PYTHONPATH="$CODE/src" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
DATA="$ROOT/data/c3fd_semantic_v21_step1b_20260828"
C3FD="$ROOT/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
PLANNER="$ROOT/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
POINTER="$ROOT/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt"
"$PY" -c 'import json,sys; f=open(sys.argv[1],"x"); f.writelines(json.dumps({"sample_idx":i})+"\n" for i in range(1200)); f.close()' "$RUN/ordinal_ledger.jsonl"
checksum() { sha256sum "$1" | awk '{print $1}'; }
"$PY" "$CODE/src/scripts/sample_c3fd_llama_typed_planner.py" \
  --c3fd-checkpoint "$C3FD" --data-dir "$DATA" --llama-model /public/home/jiaosz/ywliang/models/Meta-Llama-3-8B \
  --fused-planner-final "$PLANNER" --species-pointer-state "$POINTER" \
  --source-ledger "$RUN/ordinal_ledger.jsonl" --output-dir "$RUN/plans" \
  --expected-c3fd-sha256 "$(checksum "$C3FD")" --expected-vocabulary-sha256 "$(checksum "$DATA/vocabulary.json")" \
  --expected-source-ledger-sha256 "$(checksum "$RUN/ordinal_ledger.jsonl")" \
  --expected-typed-config-sha256 "$(checksum "$PLANNER/typed_residual_config.json")" \
  --expected-typed-state-sha256 "$(checksum "$PLANNER/typed_residual_state.pt")" \
  --expected-adapter-config-sha256 "$(checksum "$PLANNER/llama_adapter/adapter_config.json")" \
  --expected-adapter-model-sha256 "$(checksum "$PLANNER/llama_adapter/adapter_model.safetensors")" \
  --requested 1200 --expected-requested 1200 --seed 27 --expected-seed 27 \
  --temperature .9 --top-p .95 --top-k 0 --max-species 7 --minimum-comp-valid 0. \
  >"$RUN/planner.out" 2>"$RUN/planner.err"
"$PY" "$CODE/scripts/freeze_spad_prospective_plans.py" --source-records "$RUN/plans/raw_generations.jsonl" \
  --source-plans "$RUN/plans/plans_for_dlm.jsonl" --output-dir "$RUN/cohort" \
  --requested 1200 --planner-sampling-seed 27 --minimum-comp-valid 0. >"$RUN/freeze.out" 2>"$RUN/freeze.err"
cp "$LOCK" "$RUN/METHOD_LOCK.json"
test -f "$RUN/cohort/_SUCCESS"
touch "$RUN/_SUCCESS"
