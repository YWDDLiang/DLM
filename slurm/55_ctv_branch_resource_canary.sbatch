#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-canary
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-ctv-canary-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly BASE_RUN="${H1A2_CTV_BASE_RUN:?export H1A2_CTV_BASE_RUN}"
readonly CHECKPOINT="${BASE_RUN}/train/checkpoints/step-696"
readonly PLANS="${H1A2_CTV_CANARY_PLANS:?export H1A2_CTV_CANARY_PLANS}"
readonly IDENTITY_MANIFEST="${H1A2_CTV_IDENTITY_MANIFEST:?export H1A2_CTV_IDENTITY_MANIFEST}"
if [[ -z "${H1A2_CTV_CERTIFIED_COHORTS_MANIFEST:-}" ]]; then
  missing "H1A2_CTV_CERTIFIED_COHORTS_MANIFEST"
fi
readonly CERTIFIED_MANIFEST="${H1A2_CTV_CERTIFIED_COHORTS_MANIFEST}"
readonly MINIMAL_AUDIT="${H1A2_CTV_MINIMAL_AUDIT:?export H1A2_CTV_MINIMAL_AUDIT}"
readonly MATTERSIM_GATE_A="${H1A2_MATTERSIM_GATE_A:?export H1A2_MATTERSIM_GATE_A}"
readonly REFINER="${H1A2_REFINER_PATH:?export H1A2_REFINER_PATH}"
readonly CRYS="${H1A2_CRYSLLMGEN_DIR:?export H1A2_CRYSLLMGEN_DIR}"
readonly RUN_DIR="${ROOT}/runs/ctv_branch_canary_${SLURM_JOB_ID}"

for path in \
  "${MODEL}" \
  "${BASE_RUN}/_SUCCESS" \
  "${CHECKPOINT}/adapter_model.safetensors" \
  "${PLANS}" \
  "${IDENTITY_MANIFEST}" \
  "${CERTIFIED_MANIFEST}" \
  "${MINIMAL_AUDIT}" \
  "${MATTERSIM_GATE_A}" \
  "${REFINER}" \
  "${CRYS}"; do
  [[ -e "${path}" ]] || missing "${path}"
done

python - \
  "${IDENTITY_MANIFEST}" \
  "${CERTIFIED_MANIFEST}" \
  "${MINIMAL_AUDIT}" \
  "${MATTERSIM_GATE_A}" \
  "${PLANS}" <<'PY'
import hashlib,json,sys
identity,certified,minimal,mattersim=(
    json.load(open(path,encoding='utf-8')) for path in sys.argv[1:5]
)
if identity.get('gate',{}).get('identity_freeze_authorized') is not True:
    raise SystemExit('CTV reduced-identity freeze is not authorized')
if certified.get('gate',{}).get('ctv_certified_cohorts_authorized') is not True:
    raise SystemExit('CTV certified cohorts are not authorized')
expected=certified.get('files',{}).get('CTV_BRANCH_CANARY_PLANS.jsonl',{}).get('output_sha256')
actual=hashlib.sha256(open(sys.argv[5],'rb').read()).hexdigest()
if not expected or expected != actual:
    raise SystemExit('CTV canary cohort hash changed')
if minimal.get('gate',{}).get('minimal_spec_data_authorized') is not True:
    raise SystemExit('CTV minimal-spec audit is not authorized')
if mattersim.get('gates',{}).get('critic_training_authorized') is not True:
    raise SystemExit('MatterSim Gate A did not authorize the CTV resource canary')
PY

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	CTV_resource_canary_only
plans	8
milestones	0.60,0.80
actions_per_state	8
continuations_per_action	2
terminal_completions	256
checkpoint	${CHECKPOINT}
visible_condition	C3FD-certified minimal composition only
temperature	0.7
generation_schedule	exact_axis
refiner	${REFINER}
refiner_tau	800
method_adaptation_from_outcomes	forbidden
energy_or_rank_conclusion	forbidden
formal_branch_Q	not_authorized
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30855 --nproc_per_node=4 \
  "${ROOT}/src/scripts/sample_ctv_branch_canary.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}/branch" \
  --crysllmgen-dir "${CRYS}" \
  --canary-jsonl "${PLANS}" \
  --temperature 0.7 \
  --seed 61017

[[ -f "${RUN_DIR}/branch/_DLM_SUCCESS" ]] || missing "${RUN_DIR}/branch/_DLM_SUCCESS"
python - "${RUN_DIR}/branch/CTV_RESOURCE_CANARY_MANIFEST.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('layout',{}).get('rows') != 256 or x.get('states') != 16:
    raise SystemExit('CTV resource canary ledger is incomplete')
if x.get('engineering_only') is not True:
    raise SystemExit('CTV resource canary lost its engineering-only marker')
PY

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30856 --nproc_per_node=4 \
  "${ROOT}/src/scripts/refine_dlm_with_crysllmgen.py" \
  --proposal-graphs "${RUN_DIR}/branch/proposal_graphs.pt" \
  --checkpoint "${REFINER}" \
  --output-dir "${RUN_DIR}/refine" \
  --crysllmgen-dir "${CRYS}" \
  --batch-size 1 \
  --seed 71017 \
  --seed-by-sample-index \
  --diff-steps 800 \
  --max-proposals 256

touch "${RUN_DIR}/_SUCCESS"
log "CTV 256-completion resource canary finished: ${RUN_DIR}"
