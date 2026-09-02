#!/usr/bin/env bash
#SBATCH --job-name=spad-dlm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=05:00:00
#SBATCH --output=spad-dlm-%j.out
#SBATCH --error=spad-dlm-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly SOURCE="${SSCD_SOURCE_ROOT:-/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https}"
readonly DATA="${ROOT}/data/spad_schedule_sft_v1_20260903"
readonly BASE="${ROOT}/runs/canonical_compact_v2_dlm_39282/train/checkpoints/step-3392"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${SOURCE}/src/scripts/llada_sft.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly RUN="${ROOT}/runs/spad_schedule_dlm_${SLURM_JOB_ID}"

mkdir "${RUN}"
mkdir "${RUN}/logs"
trap 'rc=$?; printf "exit_code\t%s\nline\t%s\n" "$rc" "$LINENO" >"${RUN}/ENGINEERING_FAILURE.tsv"; touch "${RUN}/_FAILED"; exit "$rc"' ERR
[[ -f "${DATA}/_SUCCESS" && -f "${DATA}/vocab_tokens.txt" ]]
[[ "$(wc -l <"${DATA}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l <"${DATA}/val.jsonl")" -eq 9047 ]]
[[ -f "${BASE}/adapter_model.safetensors" ]]
IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 4 ]]
readonly DEVICES="${allocated[0]},${allocated[1]},${allocated[2]},${allocated[3]}"

cat >"${RUN}/contract.tsv" <<EOF
field\tvalue
initialization\tcanonical Compact-V2 step3392
teacher\tfull MP20 teacher Plan/body
mask_mix\tdeterministic random geometry, SPAD predictor, suffix-visible correction
train_rows\t27136
updates\t1696
effective_batch\t16
seed\t87017
learning_rate\t5e-6 cosine
energy_hull_chgnet_model494\tfalse
checkpoint_selection\tnone
EOF

CUDA_VISIBLE_DEVICES="${DEVICES}" PYTHONPATH="${SOURCE}/src" \
  torchrun --standalone --master-port 35817 --nproc_per_node=4 "${TRAINER}" \
  --model-path "${MODEL}" --checkpoint-path "${BASE}" \
  --data-dir "${DATA}" --output-dir "${RUN}/train" \
  --representation dynamic_v1 --dynamic-geometry-only \
  --require-paired-source --require-rollout-masks \
  --max-length 382 --answer-token-count 87 \
  --epochs 1 --max-train-steps 1696 --batch-size 1 --grad-accum 4 \
  --lr 5e-6 --lr-scheduler cosine --warmup-steps 25 --min-lr-ratio 0.1 \
  --save-steps 1696 --skip-final-alias --eval-steps 1696 \
  --eval-max-batches 50 --logging-steps 10 --dataloader-num-workers 4 \
  --fail-on-truncation --seed 87017 --data-seed 20260903 \
  --lora-rank 8 --lora-alpha 32 --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
  --counterfactual-grounding-weight 0.0 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-1696"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoints < <(find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -printf '%f\n')
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-1696 ]]
"${PYTHON}" - "${OUTPUT}" "${BASE}" <<'PY'
import json,math,pathlib,sys
out=pathlib.Path(sys.argv[1]); base=str(pathlib.Path(sys.argv[2]))
c=json.load(open(out/'run_config.json'))
assert c['checkpoint_path']==base and c['world_size']==4
assert c['max_train_steps']==1696 and c['seed']==87017
events=[json.loads(x) for x in open(out/'training_log.jsonl') if x.strip()]
train=[x for x in events if x.get('event')=='train']
assert train and int(train[-1]['step'])>=1690
assert any(x.get('event')=='eval' and int(x['step'])==1696 for x in events)
assert all(all(math.isfinite(float(x[k])) for k in ('loss','task_loss','grad_norm','lr')) for x in train)
start=next(x for x in events if x.get('event')=='start')
assert start['paired_source']['train_rows']==27136
assert start['rollout_masks']['train_rows']==27136
PY
printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
