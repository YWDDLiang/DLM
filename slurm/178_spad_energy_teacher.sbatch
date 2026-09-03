#!/usr/bin/env bash
#SBATCH --job-name=spad-e-teacher
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=spad-e-teacher-%j.out
#SBATCH --error=spad-e-teacher-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly COHORT="${ROOT}/cohorts/spad_energy_train2048_v1_20260903"
readonly LLAMA=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly LLADA=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PLANNER="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly POINTER="${ROOT}/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt"
readonly POINTER_DATA="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly VOCAB="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json"
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_energy_teacher_${SLURM_JOB_ID}"
readonly PROGRAMS="${RUN}/programs"
readonly PREDICTOR="${RUN}/predictor_body"
readonly ACTIONS="${RUN}/backfill_actions"
readonly REFINE="${RUN}/model494_tau800"
readonly LABELS="${RUN}/labels"
mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 4 ]]
[[ -f "${COHORT}/_SUCCESS" ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/export_spad_energy_pointer_programs.py" \
  --cohort-dir "${COHORT}" --llama-model "${LLAMA}" --planner-final "${PLANNER}" \
  --pointer-state "${POINTER}" --c3fd-checkpoint "${C3FD}" --vocabulary "${VOCAB}" \
  --output-dir "${PROGRAMS}" --batch-size 128 \
  >"${RUN}/logs/programs.out" 2>"${RUN}/logs/programs.err"
[[ -f "${PROGRAMS}/_SUCCESS" && "$(wc -l <"${PROGRAMS}/plans_for_dlm.jsonl")" -eq 2048 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master_port=29678 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  --model-path "${LLADA}" --checkpoint-path "${BS_POLICY}" \
  --prompt-jsonl "${PROGRAMS}/plans_for_dlm.jsonl" --output-dir "${PREDICTOR}" \
  --crysllmgen-dir "${CRYS}" --num-samples 2048 --batch-size 8 \
  --temperature 0.7 --seed 95017 --preserve-prompt-sample-idx \
  --no-repeat-prompt-records --generation-schedule spad \
  --canonicalize-periodic-alias --pbc-min-distance-mask \
  >"${RUN}/logs/predictor.out" 2>"${RUN}/logs/predictor.err"
[[ -f "${PREDICTOR}/_SUCCESS" && "$(wc -l <"${PREDICTOR}/raw_generations.jsonl")" -eq 2048 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master_port=29679 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/sample_spad_energy_backfill_actions.py" \
  --model-path "${LLADA}" --checkpoint-path "${BS_POLICY}" \
  --predictor-body "${PREDICTOR}/raw_generations.jsonl" \
  --output-dir "${ACTIONS}" --crysllmgen-dir "${CRYS}" \
  --batch-size 8 --seed 96017 \
  >"${RUN}/logs/actions.out" 2>"${RUN}/logs/actions.err"
[[ -f "${ACTIONS}/_SUCCESS" && "$(wc -l <"${ACTIONS}/candidate_actions.jsonl")" -eq 8192 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master_port=29680 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${ACTIONS}/proposal_graphs.pt" --checkpoint "${REFINER}" \
  --output-dir "${REFINE}" --crysllmgen-dir "${CRYS}" --batch-size 1 \
  --seed 97017 --seed-by-sample-index --diff-steps 800 --max-proposals 8192 \
  >"${RUN}/logs/refine.out" 2>"${RUN}/logs/refine.err"
[[ -f "${REFINE}/refinement_metrics.json" ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/src/scripts/label_spad_energy_actions.py" \
  --candidate-actions "${ACTIONS}/candidate_actions.jsonl" \
  --refinement-metrics "${REFINE}/refinement_metrics.json" \
  --output-dir "${LABELS}" --device cuda \
  >"${RUN}/logs/labels.out" 2>"${RUN}/logs/labels.err"
[[ -f "${LABELS}/_SUCCESS" && "$(wc -l <"${LABELS}/groups.jsonl")" -eq 2048 ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
programs = json.load(open(root / "programs/manifest.json"))
predictor = json.load(open(root / "predictor_body/sample_metrics.json"))
actions = json.load(open(root / "backfill_actions/sample_metrics.json"))
refine = json.load(open(root / "model494_tau800/refinement_metrics.json"))
labels = json.load(open(root / "labels/manifest.json"))
report = {
    "schema": "spad_energy_teacher_run_v1",
    "groups": 2048,
    "K": 4,
    "programs": programs["rows"],
    "predictor_parsed": predictor["parse_success"],
    "predictor_graphs": predictor["graph_success"],
    "valid_actions": actions["valid_actions"],
    "valid_noops": actions["valid_noops"],
    "model494_endpoints": int(refine["assigned_proposals"]),
    "terminal_energy_known": labels["terminal_energy_known"],
    "trainable_groups": labels["trainable_groups"],
    "validity_is_lexicographic": True,
    "mandatory_noop": True,
    "official_or_test_outcomes_read": False,
    "inference_time_critic": False,
}
(root / "TEACHER_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

trap - ERR
touch "${RUN}/_SUCCESS"
