#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-formal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-ctv-formal-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly SPLIT="${H1A2_CTV_BRANCH_SPLIT:?set train or validation}"
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE_RUN="${H1A2_CTV_BASE_RUN:?export H1A2_CTV_BASE_RUN}"
readonly CHECKPOINT="${BASE_RUN}/train/checkpoints/step-696"
readonly COHORTS="${H1A2_CTV_CERTIFIED_COHORTS:?export H1A2_CTV_CERTIFIED_COHORTS}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"

case "${SPLIT}" in
  train)
    readonly PLANS="${COHORTS}/CTV_BRANCH_TRAIN_PLANS.jsonl"
    readonly EXPECTED_PLANS=128
    readonly CONTINUATIONS=1
    readonly MODE=branch_train
    readonly SEED=62017
    readonly REFINE_SEED=72017
    ;;
  validation)
    readonly PLANS="${COHORTS}/CTV_BRANCH_VAL_PLANS.jsonl"
    readonly EXPECTED_PLANS=32
    readonly CONTINUATIONS=2
    readonly MODE=branch_validation
    readonly SEED=63017
    readonly REFINE_SEED=73017
    ;;
  *)
    missing "H1A2_CTV_BRANCH_SPLIT must be train or validation"
    ;;
esac

readonly RUN_DIR="${ROOT}/runs/ctv_branch_${SPLIT}_${SLURM_JOB_ID}"
for path in \
  "${MODEL}" \
  "${BASE_RUN}/_SUCCESS" \
  "${CHECKPOINT}/adapter_model.safetensors" \
  "${COHORTS}/_SUCCESS" \
  "${PLANS}" \
  "${REFINER}" \
  "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	formal_CTV_Branch_rollout
split	${SPLIT}
plans	${EXPECTED_PLANS}
milestones	0.60,0.80
actions_per_state	8
continuations_per_action	${CONTINUATIONS}
branches	$((EXPECTED_PLANS * 2 * 8 * CONTINUATIONS))
base	${CHECKPOINT}
temperature	0.7
schedule	exact_axis
action_rule	argmax_plus_distinct_cdf_projection_v2
refiner	${REFINER}
refiner_tau	800
refiner_noise	common_within_state_continuation
terminal_rerank	false
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30857 \
  --nproc_per_node=4 "${ROOT}/src/scripts/sample_ctv_branch_canary.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}/branch" \
  --crysllmgen-dir "${CRYS}" \
  --canary-jsonl "${PLANS}" \
  --expected-plans "${EXPECTED_PLANS}" \
  --continuations-per-action "${CONTINUATIONS}" \
  --run-mode "${MODE}" \
  --temperature 0.7 \
  --seed "${SEED}"

[[ -f "${RUN_DIR}/branch/_DLM_SUCCESS" ]] || missing "branch/_DLM_SUCCESS"
python - "${RUN_DIR}/branch/CTV_BRANCH_ROLLOUT_MANIFEST.json" \
  "$((EXPECTED_PLANS * 2 * 8 * CONTINUATIONS))" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('layout',{}).get('rows') != int(sys.argv[2]):
    raise SystemExit('formal Branch ledger is incomplete')
if x.get('engineering_only') is not False:
    raise SystemExit('formal Branch rollout is mislabeled')
PY

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30858 \
  --nproc_per_node=4 "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${RUN_DIR}/branch/proposal_graphs.pt" \
  --checkpoint "${REFINER}" \
  --output-dir "${RUN_DIR}/refine" \
  --crysllmgen-dir "${CRYS}" \
  --batch-size 1 \
  --seed "${REFINE_SEED}" \
  --seed-from-graph-field ctv_refiner_seed \
  --diff-steps 800 \
  --max-proposals "$((EXPECTED_PLANS * 2 * 8 * CONTINUATIONS))"

touch "${RUN_DIR}/_SUCCESS"
log "CTV formal ${SPLIT} Branch rollout completed: ${RUN_DIR}"
