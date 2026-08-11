#!/bin/bash
#SBATCH --job-name=h1-r03-r256-sun-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2/logs/%x-%A_%a.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2/logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$ROOT/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2"
SOURCE="$RUN/source"
RUNTIME="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis_refined_repeats4_v1/runtime"
R03E="$ROOT/runs/20260802_h1_body_safeaxis_refined_repeats4_v1"
REPEAT="${SLURM_ARRAY_TASK_ID:?array task ID is required}"
REPEAT_ROOT="$RUN/repeats/$REPEAT"
OUTPUT="$REPEAT_ROOT/current_sun"
: "${H1_R03_REPLAY_SOURCE_SHA256:?source manifest SHA is required}"

mkdir -p "$REPEAT_ROOT"
on_exit() {
  code=$?
  printf '%s\n' "$code" > "$REPEAT_ROOT/exit_code"
  if [[ "$code" -ne 0 ]]; then
    touch "$REPEAT_ROOT/_FAILURE"
  fi
}
trap on_exit EXIT

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1_R03_REPLAY_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ "$REPEAT" -lt 0 || "$REPEAT" -gt 3 ]] \
  || [[ -e "$OUTPUT" ]]; then
  echo "replay identity/resource contract changed" >&2
  exit 2
fi
test -f "$RUN/mp_cache/completion_SUCCESS"

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$RUNTIME:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

readarray -t IDENTITIES < <(python - "$SOURCE/CONFIG.json" "$RUN/mp_cache/completion_manifest.json" "$REPEAT" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
repeat = int(sys.argv[3])
spec = next(row for row in config["historical"]["repeats"] if int(row["repeat"]) == repeat)
print(manifest["completed_mp_hull_cache"]["sha256"])
print(spec["frozen_relax_cache_sha256"])
PY
)
CACHE_SHA="${IDENTITIES[0]}"
RELAX_SHA="${IDENTITIES[1]}"

python - <<'PY'
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1 or "A800" not in names[0]:
    raise RuntimeError(f"expected exactly one A800, observed {names}")
print({"gpu": names, "torch": torch.__version__})
PY

echo "STAGE repeat=$REPEAT current_exact_sun_frozen_refined256"
python "$SOURCE/run_current_sun_replay.py" \
  --runner "$RUNTIME/scripts/a800/run_crysllmgen_a100_sun.py" \
  --runner-sha256 e952795c32efe1be328165d99389ae3934a6698d29027569376bbb984cf788de \
  --module-sha256 61c8e13c194028c4ab514902827575de41a2d06e502824b8355fcf3530d7f968 \
  --expected-mp-cache-sha256 "$CACHE_SHA" \
  --expected-relax-cache-sha256 "$RELAX_SHA" \
  -- \
  --generation-jsonl "$R03E/repeats/$REPEAT/arms/candidate/generation/generation.jsonl" \
  --expected-attempts 256 \
  --eval-sun-py reference/a100_eval_sun/eval_sun.py \
  --eval-sun-resumable-py reference/a100_eval_sun/eval_sun_resumable.py \
  --train-csv reference/crysllmgen/data/mp_20/train.csv \
  --training-index-cache reference/crysllmgen/data/mp_20/.cache/train.csv.3a814f7b7bf29b1a.training_index.pkl \
  --mp-hull-cache "$RUN/mp_cache/completed_mp_hull_cache.jsonl" \
  --chgnet-relax-cache "$R03E/repeats/$REPEAT/arms/candidate/evaluation/r5c_a100_sun/working_chgnet_relax_cache.jsonl" \
  --chgnet-model-asset /public/home/jiaosz/ywliang/models/wqcodiff/chgnet-0.3.0.pth.tar \
  --chgnet-runtime-checkpoint /public/home/jiaosz/miniconda3/envs/diff_meets_diff/lib/python3.10/site-packages/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar \
  --base-source-bundle-sha256 6b0dd8298c9a423712a3965b00f8aeae9c06c824a37fbf1d94dbbf94aebcf15f \
  --execution-patch-sha256 "$H1_R03_REPLAY_SOURCE_SHA256" \
  --output-dir "$OUTPUT"

echo "STAGE repeat=$REPEAT validate_isolated_replay"
python "$SOURCE/validate_repeat.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03_REPLAY_SOURCE_SHA256" \
  --run-root "$RUN" \
  --repeat "$REPEAT" \
  --output-dir "$OUTPUT"
test -f "$OUTPUT/_SUCCESS"
touch "$REPEAT_ROOT/_SUCCESS"
echo "STAGE repeat=$REPEAT complete"
