#!/usr/bin/env bash
#SBATCH --job-name=spad-bs-tau800-s17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=spad-bs-tau800-s17-%j.out
#SBATCH --error=spad-bs-tau800-s17-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:?}"
readonly BS_BODY_RUN="${SPAD_BS_BODY_RUN:?}"
readonly REFINER=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_bs_tau800_s17_${SLURM_JOB_ID}"
readonly BODY="${BS_BODY_RUN}/BS/stream17/body"
readonly DLM_SEED=91117
readonly REFINER_SEED=101117

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\ncommand\t%s\n" "$rc" "$LINENO" "$BASH_COMMAND" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${BODY}/_SUCCESS" ]]
[[ -f "${BODY}/proposal_graphs.pt" ]]
[[ -f "${REFINER}" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ "${#allocated[@]}" -eq 4 ]]

"${PYTHON}" "${SOURCE}/scripts/shard_proposal_graphs.py" \
  --proposal-graphs "${BODY}/proposal_graphs.pt" \
  --output-dir "${RUN}/proposal_shards" --shard-count 8

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  gpu="${allocated[$((rank % 4))]}"
  worker="${RUN}/workers/worker-${rank}"
  mkdir -p "${worker}"
  env CUDA_VISIBLE_DEVICES="${gpu}" OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    PYTHONPATH="${SOURCE}/src" "${PYTHON}" \
    "${SOURCE}/src/scripts/refine_dlm_with_crysllmgen.py" \
    --proposal-graphs "${RUN}/proposal_shards/shard-$(printf '%03d' "${rank}").pt" \
    --checkpoint "${REFINER}" --output-dir "${worker}" \
    --crysllmgen-dir "${CRYS}" --batch-size 1 --seed "${REFINER_SEED}" \
    --seed-by-sample-index --diff-steps 800 --max-proposals 256 \
    >"${RUN}/logs/worker${rank}.out" 2>"${RUN}/logs/worker${rank}.err" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status="$?"; done
(( status == 0 ))

merge_args=()
for rank in 0 1 2 3 4 5 6 7; do
  merge_args+=(--worker-dir "${RUN}/workers/worker-${rank}")
done
"${PYTHON}" "${SOURCE}/scripts/merge_refinement_shards.py" \
  --proposal-graphs "${BODY}/proposal_graphs.pt" "${merge_args[@]}" \
  --output-dir "${RUN}/refine" --diff-steps 800

"${PYTHON}" "${SOURCE}/scripts/assemble_grounding_repeat.py" \
  --body-dir "${BODY}" --refine-dir "${RUN}/refine" \
  --output-dir "${RUN}/generation" --arm candidate --repeat 0 \
  --dlm-seed "${DLM_SEED}" --refiner-seed "${REFINER_SEED}" \
  --denominator 256
[[ "$(wc -l <"${RUN}/generation/generation.jsonl")" -eq 256 ]]

"${PYTHON}" - "${RUN}" "${BS_BODY_RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
metrics = json.load(open(root / "refine" / "refinement_metrics.json"))
(root / "REFINEMENT_FINAL.json").write_text(json.dumps({
    "schema": "spad_bs_tau800_stream17_v1",
    "body_run": str(pathlib.Path(sys.argv[2]).resolve()),
    "stream": 17,
    "tau": 800,
    "seed_by_sample_index": True,
    "workers": 8,
    "requested": 256,
    "reported": metrics.get("num_output_structures"),
}, indent=2, sort_keys=True) + "\n")
PY

trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
