#!/usr/bin/env bash
#SBATCH --job-name=h1a2-ctv-base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-ctv-base-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly DATA="${H1A2_CTV_MINIMAL_DATA:?export H1A2_CTV_MINIMAL_DATA}"
readonly AUDIT="${H1A2_CTV_MINIMAL_AUDIT:?export H1A2_CTV_MINIMAL_AUDIT}"
readonly RESUME="${H1A2_RESUME_CHECKPOINT:?export H1A2_RESUME_CHECKPOINT}"
readonly RUN_DIR="${ROOT}/runs/ctv_minimal_base_${SLURM_JOB_ID}"

for path in "${MODEL}" "${DATA}/_SUCCESS" "${AUDIT}" "${RESUME}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${RESUME}/adapter_model.safetensors" ]] || missing "${RESUME}/adapter_model.safetensors"
python - "${AUDIT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('gate',{}).get('minimal_spec_data_authorized') is not True:
    raise SystemExit('minimal-spec data audit did not authorize training')
PY

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
stage	minimal_spec_base_freeze
initialization	${RESUME}
visible_condition	formula,N,elements/counts,family,charge certificate
soft_rich_fields	absent
stability_or_energy_prompt	absent
additional_optimizer_steps	696
checkpoint_selection	none
world_size	4
batch_size_per_rank	1
gradient_accumulation	4
effective_batch	16
learning_rate	1e-5
lr_scheduler	cosine
safe_axis_schedule	frozen downstream
planner	C3FD-v2.5 frozen
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30854 --nproc_per_node=4 \
  "${ROOT}/src/scripts/llada_sft.py" \
  --model-path "${MODEL}" --checkpoint-path "${RESUME}" \
  --data-dir "${DATA}" --output-dir "${RUN_DIR}/train" \
  --representation dynamic_v1 --max-length 382 --answer-token-count 87 \
  --epochs 1 --max-train-steps 696 --batch-size 1 --grad-accum 4 \
  --lr 1e-5 --lr-scheduler cosine --warmup-steps 50 --min-lr-ratio 0.1 \
  --save-steps 696 --eval-steps 348 --eval-max-batches 50 \
  --seed 51017 --counterfactual-grounding-weight 0.0

readonly FINAL="${RUN_DIR}/train/checkpoints/step-696"
[[ -f "${FINAL}/adapter_model.safetensors" ]] || missing "${FINAL}/adapter_model.safetensors"
sha256sum "${FINAL}/adapter_model.safetensors" >"${RUN_DIR}/selected_checkpoint.sha256"
sha256sum "${DATA}/train.jsonl" "${DATA}/val.jsonl" "${DATA}/test.jsonl" >"${RUN_DIR}/data.sha256"
touch "${RUN_DIR}/_SUCCESS"
log "CTV minimal-spec base frozen: ${RUN_DIR}"
