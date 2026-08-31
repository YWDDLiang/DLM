#!/usr/bin/env bash
#SBATCH --job-name=c3fd-llama-rich
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-llama-rich-%j.out
#SBATCH --error=c3fd-llama-rich-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/Meta-Llama-3-8B
readonly CHECKPOINT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260603_034533-h1a2-epoch2-3-fullmetrics/outputs/h1a2_epoch2_llama_rich_sft/final
readonly DATA="${ROOT}/data/c3fd_llama_rich_expander_v1_20260831"
readonly TRAINER="${ROOT}/src/scripts/llama_formula_sft.py"
readonly EXPANDER="${ROOT}/src/crystal_dlm/c3fd_rich_expander.py"
readonly RUN="${ROOT}/runs/c3fd_llama_rich_expanders_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TRAINER_SHA=e61031edb32b55ff0f774cfb8a3d0d6c0c66f85b21fbe2ee6e34a81f1f87479e
readonly EXPANDER_SHA=a6a824118728c07ea6179613a0d295181ebb783471f80b0eb603f9a77da83007

mkdir "${RUN}"
mkdir "${RUN}/logs"
on_error() {
  local rc="$1" line="$2" command="$3"
  set +e
  printf 'field\tvalue\nexit_code\t%s\nline\t%s\ncommand\t%s\n' \
    "${rc}" "${line}" "${command}" >"${RUN}/ENGINEERING_FAILURE.tsv"
  touch "${RUN}/_FAILED"
  exit "${rc}"
}
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

[[ -f "${DATA}/_SUCCESS" ]]
for route in F M; do
  [[ -f "${DATA}/${route}/train.jsonl" && -f "${DATA}/${route}/val.jsonl" ]]
done
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ -f "${CHECKPOINT}/adapter_config.json" ]]
[[ "$(sha256sum "${TRAINER}" | awk '{print $1}')" == "${TRAINER_SHA}" ]]
[[ "$(sha256sum "${EXPANDER}" | awk '{print $1}')" == "${EXPANDER_SHA}" ]]

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]]

data_manifest_sha="$(sha256sum "${DATA}/manifest.json" | awk '{print $1}')"
cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_llama_rich_expander_training_v1
routes	F_formula_only,M_learned_soft_prefix
initialization	historical_H1A2_epoch2_rich_Planner_adapter
data_manifest_sha256	${data_manifest_sha}
training_seeds	F:84017,84018;M:84117,84118
source_epochs	1
batch_size	2
gradient_accumulation	8
effective_batch	16
updates_expected	1696
learning_rate	2e-5
soft_prefix	M_only_K4_two_layer_MLP_hidden1024
eligible_checkpoint	final_only
seed_checkpoint_selection	none
outcomes_read	false
EOF

sha256sum "${DATA}/manifest.json" "${DATA}/F/train.jsonl" "${DATA}/F/val.jsonl" \
  "${DATA}/M/train.jsonl" "${DATA}/M/val.jsonl" \
  "${CHECKPOINT}/adapter_model.safetensors" "${CHECKPOINT}/adapter_config.json" \
  "${TRAINER}" "${EXPANDER}" >"${RUN}/INPUTS.sha256"

train_one() {
  local route="$1" seed="$2" gpu="$3"
  local output="${RUN}/train_${route}_seed${seed}"
  local prefix_length=0
  if [[ "${route}" == M ]]; then
    prefix_length=4
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "${PYTHON}" "${TRAINER}" \
    --model-path "${MODEL}" --checkpoint-path "${CHECKPOINT}" \
    --data-dir "${DATA}/${route}" --output-dir "${output}" \
    --max-length 768 --epochs 1 --batch-size 2 --grad-accum 8 \
    --lr 2e-5 --weight-decay 0 --warmup-steps 100 \
    --logging-steps 20 --eval-steps 1696 --eval-max-batches 50 \
    --seed "${seed}" --soft-prefix-length "${prefix_length}" \
    --soft-prefix-projector-hidden 1024 \
    >"${RUN}/logs/train_${route}_seed${seed}.out" \
    2>"${RUN}/logs/train_${route}_seed${seed}.err"
}

train_one F 84017 "${allocated_gpus[0]}" & p0=$!
train_one F 84018 "${allocated_gpus[1]}" & p1=$!
train_one M 84117 "${allocated_gpus[2]}" & p2=$!
train_one M 84118 "${allocated_gpus[3]}" & p3=$!
rc=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  wait "${pid}" || rc=$?
done
[[ "${rc}" -eq 0 ]]

for spec in F:84017 F:84018 M:84117 M:84118; do
  route="${spec%%:*}"
  seed="${spec#*:}"
  output="${RUN}/train_${route}_seed${seed}"
  final="${output}/final"
  [[ -f "${final}/adapter_model.safetensors" ]]
  "${PYTHON}" - "${output}" "${route}" "${seed}" <<'PY'
import json,math,pathlib,sys
output=pathlib.Path(sys.argv[1]); route=sys.argv[2]; seed=int(sys.argv[3])
config=json.load((output/'train_config.json').open())
metrics=json.load((output/'train_metrics.json').open())
assert config['epochs']==1.0 and config['batch_size']==2 and config['grad_accum']==8
assert config['total_updates']==1696 and config['seed']==seed
assert metrics['global_step']==1696 and math.isfinite(float(metrics['final_eval_loss']))
prefix=config['soft_prefix']
if route=='F':
    assert prefix is None
    assert not (output/'final/soft_prefix_projector.pt').exists()
else:
    assert prefix['prefix_length']==4 and prefix['projector_hidden_dim']==1024
    assert (output/'final/soft_prefix_projector.pt').is_file()
    assert (output/'final/soft_prefix_projector_config.json').is_file()
PY
  sha256sum "${final}/adapter_model.safetensors" >"${RUN}/${route}_seed${seed}_adapter.sha256"
  if [[ "${route}" == M ]]; then
    sha256sum "${final}/soft_prefix_projector.pt" >"${RUN}/${route}_seed${seed}_projector.sha256"
  fi
done

find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors \
  ! -name soft_prefix_projector.pt -print0 | sort -z \
  | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
