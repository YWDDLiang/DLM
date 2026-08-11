#!/bin/bash
#SBATCH --job-name=h1-r03-r256-asm-v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2/logs/%x-%j.err

set -Eeuo pipefail
umask 077
ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN="$ROOT/runs/20260811_h1_r03_refined256_current_sun_cache_replay_env_repair_v2"
SOURCE="$RUN/source"
: "${H1_R03_REPLAY_SOURCE_SHA256:?source manifest SHA is required}"

cd "$ROOT"
source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH="$SOURCE"
export PYTHONDONTWRITEBYTECODE=1
unset MP_API_KEY PMG_MAPI_KEY MAPI_KEY

for repeat in 0 1 2 3; do
  test -f "$RUN/repeats/$repeat/_SUCCESS"
  test "$(cat "$RUN/repeats/$repeat/exit_code")" = 0
done
echo "STAGE assemble current-cache replay"
python "$SOURCE/assemble.py" \
  --source-dir "$SOURCE" \
  --source-manifest-sha256 "$H1_R03_REPLAY_SOURCE_SHA256" \
  --run-root "$RUN"
test -f "$RUN/status/assembly_SUCCESS"
test -f "$RUN/_SUCCESS"
echo "STAGE assembly complete"
