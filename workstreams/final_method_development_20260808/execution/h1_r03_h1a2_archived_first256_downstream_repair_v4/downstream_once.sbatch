#!/bin/bash
#SBATCH --job-name=h1-r03-h1a2-down-v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_r03_h1a2_archived_first256_downstream_repair_v4/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260812_h1_r03_h1a2_archived_first256_downstream_repair_v4/logs/%x-%j.err

set -Eeuo pipefail
V3_SOURCE=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/workstreams/final_method_development_20260808/execution/h1_r03_h1a2_archived_first256_downstream_repair_v3
: "${ARCHIVED_DOWNSTREAM_V4_SOURCE_SHA256:?v4 source manifest SHA is required}"
test "$(sha256sum "$V3_SOURCE/SOURCE_SHA256.txt" | awk '{print $1}')" = c84e6044978db5a8df96a71268246e15489605bb275c492bb8e9b7eaa9dc5f35
(cd "$V3_SOURCE" && sha256sum -c SOURCE_SHA256.txt >/dev/null)
export ARCHIVED_DOWNSTREAM_V3_SOURCE_SHA256="$ARCHIVED_DOWNSTREAM_V4_SOURCE_SHA256"
exec /bin/bash <(sed \
  's/h1_r03_h1a2_archived_first256_downstream_repair_v3/h1_r03_h1a2_archived_first256_downstream_repair_v4/g' \
  "$V3_SOURCE/downstream_once.sbatch")
