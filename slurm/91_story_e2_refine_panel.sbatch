#!/usr/bin/env bash
#SBATCH --job-name=h1a2-story-e2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=h1a2-story-e2-%j.out

set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_environment

readonly E1_DIR="${REPO_ROOT}/runs/story_e1"
readonly CONTRACT_DIR="${E1_DIR}/contracts"
readonly BODY_DIR="${E1_DIR}/body"
readonly PREP_DIR="${REPO_ROOT}/runs/story_e2/prepared"
readonly OUTPUT_DIR="${REPO_ROOT}/runs/story_e2/refined"
readonly CHECKPOINT="${REPO_ROOT}/checkpoints/diffusion/model_494.pt"
readonly CHGNET_MODEL="${REPO_ROOT}/checkpoints/chgnet/chgnet_0.3.0_e29f68s314m37.pth.tar"
readonly CRYSLLMGEN_DIR="${REPO_ROOT}/src/vendor/crysllmgen"

[[ -f "${BODY_DIR}/proposal_graphs.pt" ]] || missing "${BODY_DIR}/proposal_graphs.pt"
[[ -f "${BODY_DIR}/accepted_tasks.jsonl" ]] || missing "${BODY_DIR}/accepted_tasks.jsonl"
[[ -f "${CONTRACT_DIR}/e1_tasks.jsonl" ]] || missing "${CONTRACT_DIR}/e1_tasks.jsonl"
[[ -f "${CONTRACT_DIR}/e2_contract.jsonl" ]] || missing "${CONTRACT_DIR}/e2_contract.jsonl"
[[ -f "${CHECKPOINT}" ]] || missing "${CHECKPOINT}"
[[ -f "${CHGNET_MODEL}" ]] || missing "${CHGNET_MODEL}"

python "${REPO_ROOT}/scripts/prepare_story_refine_panel.py" \
  --proposal-graphs "${BODY_DIR}/proposal_graphs.pt" \
  --accepted-tasks "${BODY_DIR}/accepted_tasks.jsonl" \
  --e1-tasks "${CONTRACT_DIR}/e1_tasks.jsonl" \
  --e2-contract "${CONTRACT_DIR}/e2_contract.jsonl" \
  --output-dir "${PREP_DIR}" --seed 20260822

python "${REPO_ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${PREP_DIR}/e2_proposal_graphs.pt" \
  --metadata-jsonl "${PREP_DIR}/e2_selected_metadata.jsonl" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --crysllmgen-dir "${CRYSLLMGEN_DIR}" \
  --batch-size 1 --timesteps 1000 --diff-steps 800 \
  --num-evals 1 --run-type train --max-proposals 0

[[ -f "${OUTPUT_DIR}/refinement_metrics.json" ]] || missing "${OUTPUT_DIR}/refinement_metrics.json"
refined_file="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_file"])' "${OUTPUT_DIR}/refinement_metrics.json")"
[[ -f "${refined_file}" ]] || missing "${refined_file}"

python "${REPO_ROOT}/scripts/build_refiner_pairs.py" \
  --proposal-graphs "${PREP_DIR}/e2_proposal_graphs.pt" \
  --refined "${refined_file}" \
  --metadata-jsonl "${PREP_DIR}/e2_selected_metadata.jsonl" \
  --attempt-ledger "${PREP_DIR}/e2_attempt_ledger.jsonl" \
  --chgnet-model "${CHGNET_MODEL}" --chgnet-device cuda \
  --output "${REPO_ROOT}/runs/story_e2/refiner_pairs.jsonl"

python "${REPO_ROOT}/scripts/audit_refiner.py" \
  --input "${REPO_ROOT}/runs/story_e2/refiner_pairs.jsonl" \
  --output "${REPO_ROOT}/runs/story_e2/refiner_attribution.json"

log "E2 complete: ${OUTPUT_DIR}; requested denominator remains 192 via ${PREP_DIR}/e2_attempt_ledger.jsonl"
