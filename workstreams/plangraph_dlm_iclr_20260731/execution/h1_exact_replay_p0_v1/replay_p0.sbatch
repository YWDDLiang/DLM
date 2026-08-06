#!/bin/bash
#SBATCH --job-name=h1-exact-replay-p0-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=runs/20260802_h1_exact_replay_p0_v1/logs/%x-%j.out
#SBATCH --error=runs/20260802_h1_exact_replay_p0_v1/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_exact_replay_p0_v1"
FROZEN_SOURCE="$ROOT/workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1"
FROZEN_RUN="$ROOT/runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1"
RUN="$ROOT/runs/20260802_h1_exact_replay_p0_v1"
: "${H1_EXACT_REPLAY_SOURCE_SHA256:?new source manifest SHA256 is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1_EXACT_REPLAY_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "H1 replay resource or source identity changed" >&2
  exit 2
fi

cd "$ROOT"
if [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" != "$H1_EXACT_REPLAY_SOURCE_SHA256" ]]; then
  echo "H1 replay source manifest identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT:$FROZEN_SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONHASHSEED=0

mkdir -p "$RUN/replay/P0"
python "$FROZEN_SOURCE/run_paired_body_refine.py" \
  --arm P0 \
  --config "$FROZEN_SOURCE/config.json" \
  --source-dir "$FROZEN_SOURCE" \
  --project-root "$ROOT" \
  --data-dir "$FROZEN_RUN/data" \
  --output-dir "$RUN/replay/P0/generation" \
  --execution-manifest-sha256 f4a4e94a6df1acb10e8c6ad8d0b712448371d1dcea79027f6c2b07b8a3303d34

python "$SOURCE/compare_replay.py" \
  --reference-dir "$FROZEN_RUN/arms/P0/generation" \
  --replay-dir "$RUN/replay/P0/generation" \
  --output "$RUN/replay_report.json"

touch "$RUN/_SUCCESS"
