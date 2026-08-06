#!/bin/bash
#SBATCH --job-name=h1-body-safeaxis256-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis256_v1/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260802_h1_body_safeaxis256_v1/logs/%x-%j.err

set -Eeuo pipefail
umask 077

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
SOURCE="$ROOT/workstreams/plangraph_dlm_iclr_20260731/execution/h1_body_safeaxis256_v1"
RUN="$ROOT/runs/20260802_h1_body_safeaxis256_v1"
: "${H1_BODY_SAFEAXIS256_SOURCE_SHA256:?source manifest SHA256 is required}"

if [[ "${SLURM_JOB_PARTITION:-}" != gpu ]] \
  || [[ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]] \
  || [[ ! "$H1_BODY_SAFEAXIS256_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "safeaxis256 resource or source identity changed" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" != "$H1_BODY_SAFEAXIS256_SOURCE_SHA256" ]]; then
  echo "safeaxis256 source manifest identity changed" >&2
  exit 2
fi
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256.txt)

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$SOURCE/runtime:$SOURCE"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

test ! -e "$RUN/screen"
test ! -e "$RUN/terminal_report.json"

python "$SOURCE/run_schedule256.py" \
  --config "$SOURCE/CONFIG.json" \
  --output-dir "$RUN/screen"

python "$SOURCE/assemble_schedule256.py" \
  --config "$SOURCE/CONFIG.json" \
  --candidate-dir "$RUN/screen" \
  --output "$RUN/terminal_report.json"

touch "$RUN/_SUCCESS"
