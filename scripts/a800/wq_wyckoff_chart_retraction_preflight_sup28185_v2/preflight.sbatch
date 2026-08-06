#!/bin/bash
#SBATCH --job-name=wq-chart-ret-wtb32-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.out
#SBATCH --error=runs/20260720_0401-crysllmgen-wq-final-v3/logs/%x-%j.err

set -Eeuo pipefail

ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
RUN=runs/20260720_0401-crysllmgen-wq-final-v3
CONTRACT=configs/experiments/wyckoff_codiffusion/wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
CONTRACT_SHA256=518e44cc1a94334f8232ee54f4199a3c01436c0768defb9e61e2628a27324a6a
RUNNER=scripts/a800/run_wq_wyckoff_chart_retraction_preflight_sup28185_v2.py
RUNNER_SHA256=63925aa0b877914b35240e55459026cf10e660665a14304a427c20253ef57a35
LEGACY_RUNNER=scripts/a800/run_wq_wyckoff_tangent_bridge_preflight_v1.py
LEGACY_RUNNER_SHA256=edddded5bbf3aca16b47bfd3e060a374f60e22ff3473e852e88a8c72b7cec3c3
TANGENT=crystal_dlm/wqcodiff/crysllmgen/tangent_bridge.py
TANGENT_SHA256=127e3c707b1bf79f2fc44d97bccecd8a6de3cb39b8e989a2806c5c7b377bfbaf
RUNTIME=crystal_dlm/wqcodiff/runtime.py
RUNTIME_SHA256=8b5ba104ee1be25ff7f8a14b703193b33920bfd71abd52b1ba1e0d082e909ea4
AUTHORIZATION=diagnostics/authorization_records/wq_wyckoff_chart_retraction_preflight_sup28185_v2.json
AUTHORIZATION_SHA256=ba1acd31cda69a4388dca420796dd7bee7e7419239ce3b5405668e42bc6e8ce1
SOURCE="$RUN/outputs/wq_parent_csp_sun256_v1/generation.jsonl"
SOURCE_SHA256=b6eb7f80a29da699407d8d19bbedeb2d657f5d7940cd767d6d71aecb6c58a598
CHECKPOINT=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
CHECKPOINT_SHA256=573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e
U_AUDIT=runs/remote_audit/20260726_wq_schedule_correct_bridge_parity_sup28054_v1/terminal_audit.json
U_AUDIT_SHA256=4b77a10d632c33b53b2208d49db19f542081a7fd91d52f038ebe7e5280e2cf41
U_SELECTION="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/selection_manifest.json"
U_SELECTION_SHA256=547c521c96e3375ffc0665d07c9104d605d14cef2e79c92159339eefb5065cc1
U_ATTEMPTS="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/attempts.jsonl"
U_ATTEMPTS_SHA256=888caaba6683d1b953bf6a7f5a42d9da3d2b432df7a3b1b810b525328b555e94
U_TERMINAL="$RUN/outputs/wq_schedule_correct_bridge_parity_sup28054_v1/terminal_report.json"
U_TERMINAL_SHA256=730dc0f005676f08d160b37006f1d2d9ababddef5e6ebeb0b200f129e062032e
FAILURE_AUDIT=runs/remote_audit/20260726_wq_wyckoff_tangent_bridge_preflight_v1/terminal_audit_job28185.json
FAILURE_AUDIT_SHA256=2f686b881479f12b4abdc4c0ece217947c5aeb99c072f6119137e97887905f22
OUTPUT="$RUN/outputs/wq_wyckoff_chart_retraction_preflight_sup28185_v2"

: "${WTB32_V2_PATCH_SHA256:?submission must export WTB32_V2_PATCH_SHA256}"
: "${WTB32_V2_AUTHORIZATION_SHA256:?submission must export WTB32_V2_AUTHORIZATION_SHA256}"
if [[ ! "$WTB32_V2_PATCH_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [[ ! "$WTB32_V2_AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]] \
  || [ "$WTB32_V2_AUTHORIZATION_SHA256" != "$AUTHORIZATION_SHA256" ] \
  || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
  || [ "${SLURM_CPUS_PER_TASK:-0}" -ne 8 ]; then
  echo "WTB-32 v2 identity/resource contract changed" >&2
  exit 2
fi

cd "$ROOT"
test -f ".artifacts/source_sync/authorized_patch_${WTB32_V2_PATCH_SHA256}.json"
test "$(sha256sum "$CONTRACT" | awk '{print $1}')" = "$CONTRACT_SHA256"
test "$(sha256sum "$RUNNER" | awk '{print $1}')" = "$RUNNER_SHA256"
test "$(sha256sum "$LEGACY_RUNNER" | awk '{print $1}')" = "$LEGACY_RUNNER_SHA256"
test "$(sha256sum "$TANGENT" | awk '{print $1}')" = "$TANGENT_SHA256"
test "$(sha256sum "$RUNTIME" | awk '{print $1}')" = "$RUNTIME_SHA256"
test "$(sha256sum "$AUTHORIZATION" | awk '{print $1}')" = "$AUTHORIZATION_SHA256"
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$U_AUDIT" | awk '{print $1}')" = "$U_AUDIT_SHA256"
test "$(sha256sum "$U_SELECTION" | awk '{print $1}')" = "$U_SELECTION_SHA256"
test "$(sha256sum "$U_ATTEMPTS" | awk '{print $1}')" = "$U_ATTEMPTS_SHA256"
test "$(sha256sum "$U_TERMINAL" | awk '{print $1}')" = "$U_TERMINAL_SHA256"
test "$(sha256sum "$FAILURE_AUDIT" | awk '{print $1}')" = "$FAILURE_AUDIT_SHA256"
test ! -e "$OUTPUT"

source /public/home/jiaosz/miniconda3/etc/profile.d/conda.sh
conda activate diff_meets_diff
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

python - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit("WTB-32 v2 requires exactly one visible CUDA device")
name = torch.cuda.get_device_name(0)
if "A800" not in name:
    raise SystemExit(f"WTB-32 v2 requires A800; observed {name}")
print(f"wtb32_v2_device={name}")
print(f"wtb32_v2_torch={torch.__version__}")
print(f"wtb32_v2_cuda={torch.version.cuda}")
PY

usage_csv="$RUN/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
  --format=csv,noheader,nounits -l 5 > "$usage_csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

python "$RUNNER" \
  --contract "$CONTRACT" \
  --source-generation "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --u-terminal-audit "$U_AUDIT" \
  --u-selection-manifest "$U_SELECTION" \
  --u-attempts "$U_ATTEMPTS" \
  --u-terminal-report "$U_TERMINAL" \
  --output-dir "$OUTPUT" \
  --execution-patch-sha256 "$WTB32_V2_PATCH_SHA256" \
  --device cuda

python - "$OUTPUT/terminal_report.json" "$CONTRACT_SHA256" \
  "$WTB32_V2_PATCH_SHA256" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arms = report.get("arm_summaries", {})
checks = report.get("gate_checks", {})
if (
    report.get("schema") != "wq_wyckoff_chart_retraction_preflight_terminal_v2"
    or report.get("ok") is not True
    or report.get("acceptance") != "PASS"
    or report.get("contract_sha256") != sys.argv[2]
    or report.get("execution_patch_sha256") != sys.argv[3]
    or report.get("development_panel_reused") is not True
    or report.get("confirmatory_evidence") is not False
    or report.get("supersedes_job_id") != 28185
    or report.get("lattice_projection_method") != "global_chart_retraction_v1"
    or set(arms) != {"F", "T"}
    or any(arms[arm].get("terminal_cells") != 32 for arm in ("F", "T"))
    or any(arms[arm].get("succeeded_cells") != 32 for arm in ("F", "T"))
    or not checks
    or not all(value is True for value in checks.values())
    or report.get("training_performed") is not False
    or report.get("new_generation_performed") is not False
    or report.get("u_rerun") is not False
    or report.get("mlip_used") is not False
    or report.get("external_api_used") is not False
    or report.get("retry_or_replacement_used") is not False
):
    raise SystemExit("WTB-32 v2 terminal acceptance audit failed")
print("wq_wyckoff_chart_retraction_preflight_acceptance=PASS")
print("wq_wyckoff_chart_retraction_preflight_evidence=development_only")
print("wq_wyckoff_chart_retraction_preflight_F_cells=32")
print("wq_wyckoff_chart_retraction_preflight_T_cells=32")
PY
