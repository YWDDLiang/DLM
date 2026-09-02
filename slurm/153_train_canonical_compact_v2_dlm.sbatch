#!/usr/bin/env bash
#SBATCH --job-name=canonical-v2-dlm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:2
#SBATCH --mem=280G
#SBATCH --time=06:00:00
#SBATCH --output=canonical-v2-dlm-%j.out
#SBATCH --error=canonical-v2-dlm-%j.err

set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly DATA="${CANONICAL_SFT_DATA:?}"
readonly TOKEN_AUDIT="${CANONICAL_TOKEN_AUDIT:?}"
readonly MODEL=/public/home/jiaosz/ywliang/models/LLaDA-8B-Instruct
readonly TRAINER="${ROOT}/src/scripts/llada_sft.py"
readonly PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly SEED=82017
readonly RUN="${ROOT}/runs/canonical_compact_v2_dlm_${SLURM_JOB_ID}"

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

[[ -f "${DATA}/_SUCCESS" && -f "${DATA}/vocab_tokens.txt" ]]
[[ "$(wc -l < "${DATA}/train.jsonl")" -eq 27136 ]]
[[ "$(wc -l < "${DATA}/val.jsonl")" -eq 9047 ]]
[[ -f "${TOKEN_AUDIT}" ]]
"${PYTHON}" - "${DATA}/manifest.json" "${TOKEN_AUDIT}" <<'PY'
import json, sys
data=json.load(open(sys.argv[1],encoding="utf-8"))
audit=json.load(open(sys.argv[2],encoding="utf-8"))
assert data["schema"] == "c3fd_native_teacher_sft_canonical_site_v1"
assert data["site_order"] == "plan_expanded_stable_v1"
assert data["train"]["rows"] == 27136 and data["val"]["rows"] == 9047
assert data["train"]["dropped_rows"] == data["val"]["dropped_rows"] == 0
assert data["train"]["prompt_changed_rows"] == data["val"]["prompt_changed_rows"] == 0
assert data["outcomes_read"] is False and data["energy_or_hull_fields_added"] is False
assert audit["splits"]["train"]["rows"] == 27136
assert audit["splits"]["val"]["rows"] == 9047
assert audit["splits"]["train"]["over_limit"] == 0
assert audit["splits"]["val"]["over_limit"] == 0
assert audit["splits"]["train"]["prompt_answer_suffix_mismatches"] == 0
assert audit["splits"]["val"]["prompt_answer_suffix_mismatches"] == 0
assert all(audit["gate"].values()) and audit["outcomes_read"] is False
PY

IFS=',' read -r -a allocated <<<"${CUDA_VISIBLE_DEVICES:?no allocated GPUs}"
[[ "${#allocated[@]}" -eq 2 ]]
readonly DEVICES="${allocated[0]},${allocated[1]}"

cat >"${RUN}/scientific_contract.tsv" <<EOF
field	value
schema	canonical_compact_v2_dlm_v1
initialization	fresh_LoRA_from_shared_LLaDA_8B
teacher	original_MP20_only
site_order	plan_expanded_stable_v1
physical_data_change	site_permutation_only
train_rows	27136
validation_rows	9047
training_seed	${SEED}
world_size	2
effective_batch	16
stage1_updates	1696
stage1_lr	5e-5
stage2_updates	1696
stage2_lr	1e-5
total_updates	3392
eligible_checkpoint	step-3392_only
checkpoint_selection	none
lora	r8_alpha32_dropout0.05_qkv_ff_up
EOF

CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun --standalone --master-port 32817 \
  --nproc_per_node=2 "${TRAINER}" \
  --model-path "${MODEL}" \
  --data-dir "${DATA}" \
  --output-dir "${RUN}/train" \
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
  --seed "${SEED}" \
  --data-seed 20260515 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,ff_proj,up_proj \
  --counterfactual-grounding-weight 0.0 \
  >"${RUN}/logs/train.out" 2>"${RUN}/logs/train.err"

readonly OUTPUT="${RUN}/train"
readonly CHECKPOINT="${OUTPUT}/checkpoints/step-3392"
[[ -f "${CHECKPOINT}/adapter_model.safetensors" ]]
[[ ! -e "${OUTPUT}/final" ]]
mapfile -t checkpoints < <(
  find "${OUTPUT}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
    -name 'step-*' -printf '%f\n' | sort
)
[[ "${#checkpoints[@]}" -eq 1 && "${checkpoints[0]}" == step-3392 ]]
"${PYTHON}" - "${OUTPUT}" "${SEED}" <<'PY'
import json, math, pathlib, sys
output=pathlib.Path(sys.argv[1]); seed=int(sys.argv[2])
config=json.load((output/"run_config.json").open())
assert config["checkpoint_path"] is None
assert config["epochs"] == 2 and config["max_train_steps"] == 3392
assert config["lr"] == 5e-5 and config["lr_stage2"] == 1e-5
assert config["lr_stage_boundary"] == 1696
assert config["seed"] == seed and config["world_size"] == 2
canary=json.load((output/"step0_lora_equality_canary.json").open())
assert canary["fresh_lora"] is True and canary["exact_zero_delta"] is True
events=[json.loads(line) for line in (output/"training_log.jsonl").open() if line.strip()]
train=[row for row in events if row.get("event") == "train"]
evaluated={int(row["step"]) for row in events if row.get("event") == "eval"}
assert train and {1696,3392} <= evaluated
for row in train:
    assert all(math.isfinite(float(row[key])) for key in ("loss","task_loss","grad_norm","lr"))
adapter=json.load((output/"checkpoints/step-3392/adapter_config.json").open())
assert int(adapter["r"]) == 8 and int(adapter["lora_alpha"]) == 32
assert float(adapter["lora_dropout"]) == 0.05
assert set(adapter["target_modules"]) == {"q_proj","k_proj","v_proj","ff_proj","up_proj"}
PY

printf '%s\n' "${CHECKPOINT}" >"${RUN}/POLICY_PATH"
trap - ERR
touch "${RUN}/_SUCCESS"
printf '%s\n' "${RUN}"
