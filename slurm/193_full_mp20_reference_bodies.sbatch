#!/usr/bin/env bash
#SBATCH --job-name=mp20-ref-bodies
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=18:00:00
#SBATCH --output=mp20-ref-bodies-%j.out
#SBATCH --error=mp20-ref-bodies-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly SFT="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly POINTER_DATA="${ROOT}/data/spad_species_pointer_v1_20260903"
readonly LLAMA=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly LLADA=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly PLANNER="${ROOT}/runs/c3fd_llama_typed_planner_39051/train_seed85017/final"
readonly POINTER="${ROOT}/runs/spad_species_pointer_39511/train_seed86017/pointer_state.pt"
readonly C3FD="${ROOT}/runs/c3fd_v25_online_canary_36608/train_seed17/checkpoint.pt"
readonly VOCAB="${ROOT}/data/c3fd_semantic_v21_step1b_20260828/vocabulary.json"
readonly BS_POLICY="${ROOT}/runs/spad_schedule_dlm_39520/train/checkpoints/step-1696"
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/full_mp20_reference_bodies_${SLURM_JOB_ID}"
readonly PROGRAMS="${RUN}/programs"
readonly LEDGER="${RUN}/transaction_sources.jsonl"
readonly BODY="${RUN}/reference_body"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 4 ]]
for path in "${SFT}/_SUCCESS" "${SFT}/train.jsonl" \
  "${POINTER_DATA}/train.jsonl" "${POINTER}" "${C3FD}" "${VOCAB}" \
  "${PLANNER}/typed_residual_state.pt" "${BS_POLICY}/adapter_model.safetensors"; do
  [[ -f "${path}" ]]
done
[[ "$(wc -l <"${SFT}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l <"${POINTER_DATA}/train.jsonl")" -eq 24558 ]]

env CUDA_VISIBLE_DEVICES="${allocated[0]}" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
  "${SOURCE}/src/scripts/export_full_mp20_pointer_programs.py" \
  --sft-data-dir "${SFT}" --pointer-data-dir "${POINTER_DATA}" \
  --llama-model "${LLAMA}" --planner-final "${PLANNER}" \
  --pointer-state "${POINTER}" --c3fd-checkpoint "${C3FD}" \
  --vocabulary "${VOCAB}" --output-dir "${PROGRAMS}" --batch-size 128 \
  >"${RUN}/logs/programs.out" 2>"${RUN}/logs/programs.err"
[[ -f "${PROGRAMS}/_SUCCESS" ]]
[[ "$(wc -l <"${PROGRAMS}/plans_for_dlm.jsonl")" -eq 27136 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" "${SOURCE}/scripts/freeze_full_mp20_transaction_sources.py" \
  --sft-train "${SFT}/train.jsonl" \
  --plans-for-dlm "${PROGRAMS}/plans_for_dlm.jsonl" \
  --output-jsonl "${LEDGER}" --common-random-seed 20260904 --formal \
  >"${RUN}/logs/ledger.out" 2>"${RUN}/logs/ledger.err"
[[ "$(wc -l <"${LEDGER}")" -eq 27136 ]]

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 PYTHONPATH="${SOURCE}/src" \
  "${PYTHON}" -m torch.distributed.run --standalone --master_port=29693 \
  --nproc_per_node=4 "${SOURCE}/src/scripts/sample_llada_r5_exact_length.py" \
  --model-path "${LLADA}" --checkpoint-path "${BS_POLICY}" \
  --prompt-jsonl "${PROGRAMS}/plans_for_dlm.jsonl" --output-dir "${BODY}" \
  --crysllmgen-dir "${CRYS}" --num-samples 27136 --batch-size 8 \
  --temperature 0.7 --seed 99117 --preserve-prompt-sample-idx \
  --no-repeat-prompt-records --generation-schedule spad \
  --canonicalize-periodic-alias --pbc-min-distance-mask \
  >"${RUN}/logs/body.out" 2>"${RUN}/logs/body.err"
[[ -f "${BODY}/_SUCCESS" ]]
[[ "$(wc -l <"${BODY}/raw_generations.jsonl")" -eq 27136 ]]

"${PYTHON}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
programs = json.load(open(root / "programs" / "manifest.json"))
body = json.load(open(root / "reference_body" / "sample_metrics.json"))
assert programs["rows"] == 27136
assert programs["pointer_decoded_rows"] == 24558
assert programs["canonical_missing_pointer_rows"] == 2578
assert body["requested_samples"] == 27136
report = {
    "schema": "full_mp20_reference_body_run_v1",
    "sources": 27136,
    "pointer_programs": programs["pointer_decoded_rows"],
    "canonical_fallback_programs": programs["canonical_missing_pointer_rows"],
    "body_decoded": body["decoded_samples"],
    "body_parsed": body["parse_success"],
    "body_plan_match": body["plan_match_success"],
    "body_graphs": body["graph_success"],
    "failed_sources_retained": 27136 - body["parse_success"],
    "outcomes_read": False,
    "model494_or_energy_used": False,
}
(root / "REFERENCE_BODY_FINAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
