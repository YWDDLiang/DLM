#!/usr/bin/env bash
#SBATCH --job-name=h1a2-sgtc-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --output=h1a2-sgtc-train-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly ARM="${H1A2_SGTC_ARM:?set g0_all or g1_strict}"
case "${ARM}" in
  g0_all|g1_strict) ;;
  *) printf 'invalid SGTC arm %s\n' "${ARM}" >&2; exit 2 ;;
esac
readonly MODEL="${H1A2_DLM_MODEL_PATH:?export H1A2_DLM_MODEL_PATH}"
readonly CHECKPOINT="${H1A2_SGTC_BASE_CHECKPOINT:?export H1A2_SGTC_BASE_CHECKPOINT}"
readonly DATA="${H1A2_SGTC_DATA:?export H1A2_SGTC_DATA}"
readonly AUDIT="${H1A2_SGTC_AUDIT:?export H1A2_SGTC_AUDIT}"
readonly RUN_DIR="${ROOT}/runs/sgtc_${ARM}_${SLURM_JOB_ID}"

for path in \
  "${MODEL}" \
  "${CHECKPOINT}/adapter_model.safetensors" \
  "${DATA}/train.jsonl" \
  "${DATA}/val.jsonl" \
  "${DATA}/vocab_tokens.txt" \
  "${AUDIT}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
python - "${AUDIT}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
if x.get('sgtc_data_authorized') is not True:
    raise SystemExit('SGTC data audit is not authorized')
PY

mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
method	SGTC-DLM-v1
arm	${ARM}
initialization	${CHECKPOINT}
visible_condition	C3FD-certified minimal composition only
stability_or_energy_prompt	absent
mask_policy	dynamic_geometry_only
supervised_tokens	lattice_lengths,lattice_angles,XYZ
anchored_unsupervised_tokens	N,elements
additional_optimizer_steps	348
checkpoint_selection	none
seed	81017
effective_batch	16
learning_rate	5e-6
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master-port 30860 --nproc_per_node=4 \
  "${ROOT}/src/scripts/llada_sft.py" \
  --model-path "${MODEL}" \
  --checkpoint-path "${CHECKPOINT}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN_DIR}/train" \
  --representation dynamic_v1 \
  --dynamic-geometry-only \
  --max-length 382 \
  --answer-token-count 87 \
  --epochs 1 \
  --max-train-steps 348 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --lr-scheduler cosine \
  --warmup-steps 25 \
  --min-lr-ratio 0.1 \
  --save-steps 348 \
  --eval-steps 174 \
  --eval-max-batches 50 \
  --seed 81017 \
  --atom-count-loss-weight 0 \
  --nonempty-slot-loss-weight 0 \
  --dynamic-lattice-length-loss-weight 1 \
  --dynamic-lattice-angle-loss-weight 1 \
  --dynamic-coord-loss-weight 1 \
  --counterfactual-grounding-weight 0

readonly FINAL="${RUN_DIR}/train/checkpoints/step-348"
[[ -f "${FINAL}/adapter_model.safetensors" ]] || missing "${FINAL}/adapter_model.safetensors"
touch "${RUN_DIR}/_SUCCESS"
log "SGTC ${ARM} training finished: ${RUN_DIR}"
