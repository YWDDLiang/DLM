#!/usr/bin/env bash
#SBATCH --job-name=c3fd-native-sft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=c3fd-native-sft-%j.out
#SBATCH --error=c3fd-native-sft-%j.err

set -Eeuo pipefail

readonly ROOT="${H1A2_CANDIDATE_ROOT:?export H1A2_CANDIDATE_ROOT}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly DATA="${ROOT}/data/c3fd_native_teacher_sft_v1_20260831"
readonly TRAINER="${ROOT}/src/scripts/llada_sft.py"
readonly LR_HELPER="${ROOT}/src/crystal_dlm/two_stage_lr.py"
readonly TOKEN_AUDIT="${ROOT}/runs/c3fd_native_teacher_token_audit_38699/audit/TOKENIZATION_AUDIT.json"
readonly RUN="${ROOT}/runs/c3fd_native_teacher_sft_${SLURM_JOB_ID}"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly MODEL_TREE_SHA=812a0ef72b2b2290b19b8beac32fd8ba0f4d32f308268f0c7914c4efd8755037
readonly DATA_MANIFEST_SHA=b77a1d0126d19576b2a6e381fe177de792b230cab2eca58e2c5c8097ff23ae06
readonly DATA_TRAIN_SHA=f61cd177f252ce44ba15669a187a5f6fed22e2d3f50218088124fd0b5b836068
readonly DATA_VAL_SHA=bea7d8e67476bcdde545f13971cb87ea1e2931954ff8948410dc21ce38e509e1
readonly TRAINER_SHA=9c4296110708b69eec5d855c0133b13909e5446505e3361f86ff9bdcde63421a
readonly LR_HELPER_SHA=e7a8be58b8b0e822e9b50394e22f98f1d250fdc60d23a0f8e0153fa4847c2bd7
readonly TOKEN_AUDIT_SHA=acb07572be689a9eaecab39fd74c934f2f1b7439595e569cf54963d1c0eccff0
readonly SEEDS=(82017 82018)

export H1A2_CONDA_ENV="${H1A2_CONDA_ENV:-diff_meets_diff}"
source "${ROOT}/slurm/_common.sh"
activate_environment

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

[[ -f "${DATA}/_SUCCESS" ]] || exit 2
[[ "$(sha256sum "${DATA}/manifest.json" | awk '{print $1}')" == "${DATA_MANIFEST_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/train.jsonl" | awk '{print $1}')" == "${DATA_TRAIN_SHA}" ]] || exit 2
[[ "$(sha256sum "${DATA}/val.jsonl" | awk '{print $1}')" == "${DATA_VAL_SHA}" ]] || exit 2
[[ "$(sha256sum "${TRAINER}" | awk '{print $1}')" == "${TRAINER_SHA}" ]] || exit 2
[[ "$(sha256sum "${LR_HELPER}" | awk '{print $1}')" == "${LR_HELPER_SHA}" ]] || exit 2
[[ "$(sha256sum "${TOKEN_AUDIT}" | awk '{print $1}')" == "${TOKEN_AUDIT_SHA}" ]] || exit 2
actual_model_tree_sha="$(find "${MODEL}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "${actual_model_tree_sha}" == "${MODEL_TREE_SHA}" ]] || exit 2

IFS=',' read -r -a allocated_gpus <<<"${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
[[ "${#allocated_gpus[@]}" -eq 4 ]] || exit 2
readonly DEVICES_A="${allocated_gpus[0]},${allocated_gpus[1]}"
readonly DEVICES_B="${allocated_gpus[2]},${allocated_gpus[3]}"

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	c3fd_native_teacher_sft_v1
initialization	fresh_LoRA_from_shared_LLaDA_8B
model_tree_sha256	${MODEL_TREE_SHA}
data_manifest_sha256	${DATA_MANIFEST_SHA}
tokenization_audit_sha256	${TOKEN_AUDIT_SHA}
train_rows	27136
validation_rows	9047
training_prompt	ground_truth_C3FD_NATIVE_PLAN_V2_teacher_rich_JSON_only
predicted_plans_in_sft	false
training_seeds	82017,82018
world_size_per_seed	2
batch_size_per_rank	1
gradient_accumulation	8
effective_batch_per_seed	16
epoch1_updates	1696
epoch1_lr	5e-5
epoch1_warmup	100
epoch1_min_lr_ratio	0.2
epoch2_updates	1696
epoch2_lr	1e-5
epoch2_warmup	100
epoch2_min_lr_ratio	0.1
total_updates	3392
eligible_checkpoint	step-3392_only
checkpoint_selection	none
lora	r8_alpha32_dropout0.05_qkv_ff_up
EOF
sha256sum "${RUN}/scientific_contract.tsv" >"${RUN}/scientific_contract.sha256"
sha256sum "${DATA}/manifest.json" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${TRAINER}" "${LR_HELPER}" "${TOKEN_AUDIT}" >"${RUN}/INPUTS.sha256"

train_seed() {
  local seed="$1" devices="$2" port="$3"
  local output="${RUN}/train_seed${seed}"
  CUDA_VISIBLE_DEVICES="${devices}" torchrun --standalone --master-port "${port}" \
    --nproc_per_node=2 "${TRAINER}" \
    --model-path "${MODEL}" \
    --data-dir "${DATA}" \
    --output-dir "${output}" \
    --representation dynamic_v1 \
    --max-length 382 \
    --answer-token-count 87 \
    --epochs 2 \
    --max-train-steps 3392 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 5e-5 \
    --lr-scheduler cosine \
    --warmup-steps 100 \
    --min-lr-ratio 0.2 \
    --lr-stage-boundary 1696 \
    --lr-stage2 1e-5 \
    --stage2-warmup-steps 100 \
    --stage2-min-lr-ratio 0.1 \
    --save-steps 3392 \
    --skip-final-alias \
    --eval-steps 1696 \
    --eval-max-batches 50 \
    --logging-steps 10 \
    --dataloader-num-workers 4 \
    --fail-on-truncation \
    --seed "${seed}" \
    --data-seed 20260515 \
    --lora-rank 8 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
    --counterfactual-grounding-weight 0.0 \
    >"${RUN}/logs/train_seed${seed}.out" \
    2>"${RUN}/logs/train_seed${seed}.err"
}

train_seed 82017 "${DEVICES_A}" 31817 &
pid_a=$!
train_seed 82018 "${DEVICES_B}" 31818 &
pid_b=$!
if wait "${pid_a}"; then rc_a=0; else rc_a=$?; fi
if wait "${pid_b}"; then rc_b=0; else rc_b=$?; fi
[[ "${rc_a}" -eq 0 && "${rc_b}" -eq 0 ]] || false

for seed in "${SEEDS[@]}"; do
  output="${RUN}/train_seed${seed}"
  checkpoint="${output}/checkpoints/step-3392"
  [[ -f "${checkpoint}/adapter_model.safetensors" ]] || exit 3
  [[ ! -e "${output}/final" ]] || exit 3
  mapfile -t checkpoint_dirs < <(find "${output}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -printf '%f\n' | sort)
  [[ "${#checkpoint_dirs[@]}" -eq 1 && "${checkpoint_dirs[0]}" == step-3392 ]] || exit 3
  "${PYTHON}" - "${output}" "${seed}" <<'PY'
import json, math, pathlib, sys
output = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
config = json.load((output / "run_config.json").open())
assert config["checkpoint_path"] is None
assert config["epochs"] == 2 and config["max_train_steps"] == 3392
assert config["lr"] == 5e-5 and config["lr_stage2"] == 1e-5
assert config["lr_stage_boundary"] == 1696
assert config["seed"] == seed and config["world_size"] == 2
assert config["data_seed"] == 20260515 and config["skip_final_alias"] is True
canary = json.load((output / "step0_lora_equality_canary.json").open())
assert canary["fresh_lora"] is True and canary["exact_zero_delta"] is True
events = [json.loads(line) for line in (output / "training_log.jsonl").open() if line.strip()]
train = [row for row in events if row.get("event") == "train"]
evaluated = {int(row["step"]) for row in events if row.get("event") == "eval"}
assert train and {1696, 3392} <= evaluated
for row in train:
    assert all(math.isfinite(float(row[key])) for key in ("loss", "task_loss", "grad_norm", "lr"))
    assert float(row["grad_norm"]) >= 0.0 and float(row["lr"]) > 0.0
adapter = json.load((output / "checkpoints" / "step-3392" / "adapter_config.json").open())
assert int(adapter["r"]) == 8 and int(adapter["lora_alpha"]) == 32
assert float(adapter["lora_dropout"]) == 0.05
assert set(adapter["target_modules"]) == {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}
PY
  sha256sum "${checkpoint}/adapter_model.safetensors" >"${RUN}/seed${seed}_step3392_adapter.sha256"
  printf '%s\n' "${checkpoint}" >"${RUN}/seed${seed}_POLICY_PATH"
done

find "${RUN}" -type f ! -name OUTPUTS.sha256 ! -name adapter_model.safetensors -print0 \
  | sort -z | xargs -0 sha256sum >"${RUN}/OUTPUTS.sha256"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
