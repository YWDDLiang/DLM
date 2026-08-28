#!/usr/bin/env bash
#SBATCH --job-name=h1a2-c3fd-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=360G
#SBATCH --time=18:00:00
#SBATCH --output=h1a2-c3fd-v2-%j.out

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

readonly MODEL="${H1A2_PLANNER_MODEL_PATH:?export H1A2_PLANNER_MODEL_PATH}"
readonly P0="${H1A2_PLANNER_P0_CHECKPOINT:?export H1A2_PLANNER_P0_CHECKPOINT}"
readonly DATA="${H1A2_C3FD_DATA:?export H1A2_C3FD_DATA}"
readonly CONTROL_RUN="${H1A2_C3FD_CONTROL_RUN:?export H1A2_C3FD_CONTROL_RUN}"
readonly TRAIN_JSONL="${H1A2_C3FD_TRAIN_JSONL:?export H1A2_C3FD_TRAIN_JSONL}"
readonly RUN_DIR="${ROOT}/runs/c3fd_planner_v2_${SLURM_JOB_ID}"

for path in "${MODEL}" "${P0}" "${DATA}" "${CONTROL_RUN}" "${TRAIN_JSONL}"; do
  [[ -e "${path}" ]] || missing "${path}"
done
[[ -f "${DATA}/_SUCCESS" ]] || missing "${DATA}/_SUCCESS"
python - "${DATA}/manifest.json" <<'PY'
import json
import sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
if x.get("gate", {}).get("planner_training_data_authorized") is not True:
    raise SystemExit("C3FD data gate did not authorize training")
PY
mkdir -p "${RUN_DIR}"
cat >"${RUN_DIR}/scientific_contract.tsv" <<EOF
field	value
control	${CONTROL_RUN}/f0_seed17,${CONTROL_RUN}/f0_seed18
candidate	full_C3FD_v2
external_crysvcd_comparison	false
formula_bpe	false
planner_backbone	${P0}
backbone_weights_frozen	true
typed_heads	N,element-valence,count,rich-properties
physics_embedding	aufbau_ionic_features_v1
pair_prior	train_only_soft_weight_0.25
hard_constraints	locked_N,atom_charge_reachability,benchmark_EOS
seeds	17,18
epochs	10
batch_size	256
learning_rate	1e-3
requested_per_seed	1000
temperature	0.9
top_p	0.95
top_k	50
repair	false
replacement	false
rerank	false
rl	false
stability_condition	false
EOF

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/extract_c3fd_planner_context.py" \
  --model-path "${MODEL}" --checkpoint-path "${P0}" \
  --output "${RUN_DIR}/frozen_context.pt"
[[ -f "${RUN_DIR}/frozen_context.pt" ]] || missing "${RUN_DIR}/frozen_context.pt"

train_seed() {
  local gpu="$1" seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/train_c3fd_planner.py" \
    --data-dir "${DATA}" --context-path "${RUN_DIR}/frozen_context.pt" \
    --output-dir "${RUN_DIR}/train_seed${seed}" --seed "${seed}" \
    --semantic-size 256 --decoder-layers 2 --decoder-heads 8 --dropout 0.05 \
    --epochs 10 --batch-size 256 --lr 1e-3 --weight-decay 1e-4
  [[ -f "${RUN_DIR}/train_seed${seed}/_SUCCESS" ]] || missing "train seed ${seed}"
}

train_seed 0 17 & p0=$!
train_seed 1 18 & p1=$!
wait "${p0}"
wait "${p1}"

sample_shard() {
  local gpu="$1" seed="$2" start="$3" count="$4" name="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT}/scripts/sample_c3fd_plans.py" \
    --checkpoint "${RUN_DIR}/train_seed${seed}/checkpoint.pt" --data-dir "${DATA}" \
    --output-dir "${RUN_DIR}/${name}" --num-samples "${count}" --start-index "${start}" \
    --seed "${seed}" --temperature 0.9 --top-p 0.95 --top-k 50 \
    --pair-prior-weight 0.25 --max-species 7
  [[ -f "${RUN_DIR}/${name}/_SUCCESS" ]] || missing "sample shard ${name}"
}

sample_shard 0 17 0 500 c3_seed17_shard0 & s0=$!
sample_shard 1 17 500 500 c3_seed17_shard1 & s1=$!
sample_shard 2 18 0 500 c3_seed18_shard0 & s2=$!
sample_shard 3 18 500 500 c3_seed18_shard1 & s3=$!
wait "${s0}"
wait "${s1}"
wait "${s2}"
wait "${s3}"

for seed in 17 18; do
  python "${ROOT}/scripts/assemble_c3fd_plan_shards.py" \
    --shard "${RUN_DIR}/c3_seed${seed}_shard0" \
    --shard "${RUN_DIR}/c3_seed${seed}_shard1" \
    --output-dir "${RUN_DIR}/c3_seed${seed}" --requested 1000
done

python "${ROOT}/scripts/finalize_c3fd_planner.py" \
  --control-run "${CONTROL_RUN}" --candidate-run "${RUN_DIR}" \
  --train-jsonl "${TRAIN_JSONL}" --requested 1000 --output-dir "${RUN_DIR}/final"

for path in "${RUN_DIR}/c3_seed17/_SUCCESS" "${RUN_DIR}/c3_seed18/_SUCCESS" "${RUN_DIR}/final/_SUCCESS"; do
  [[ -f "${path}" ]] || missing "${path}"
done
touch "${RUN_DIR}/_SUCCESS"
log "C3FD-v2 Planner completed: ${RUN_DIR}"
