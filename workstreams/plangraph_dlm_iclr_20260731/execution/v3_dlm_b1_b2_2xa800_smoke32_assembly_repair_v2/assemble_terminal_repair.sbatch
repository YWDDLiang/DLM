#!/bin/bash
#SBATCH --job-name=v3-dlm-2g-assembly-repair
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2/logs/%x-%j.out
#SBATCH --error=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2/logs/%x-%j.err

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion
EVIDENCE_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1"
REPAIR_ROOT="${PROJECT_ROOT}/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2"
SOURCE_ROOT="${REPAIR_ROOT}/source"
EXECUTION_DIR="${SOURCE_ROOT}/workstreams/plangraph_dlm_iclr_20260731/execution/v3_dlm_b1_b2_2xa800_smoke32_assembly_repair_v2"
PYTHON=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
SCHEDULER_RECORD="${REPAIR_ROOT}/scheduler_29337_array_aware.txt"

mkdir -p "${REPAIR_ROOT}/logs"
test -d "${SOURCE_ROOT}"
test ! -e "${REPAIR_ROOT}/terminal_report.json"
test "$(sha256sum "${EVIDENCE_ROOT}/terminal_report.json" | awk '{print $1}')" = \
  894d4a704214d9ff56f51c09b1c35ad20fb3a108805b28920c3714c584ce2e6f

sacct -n -X -j 29337 \
  --format=JobID,JobIDRaw,State,ExitCode,Elapsed -P > "${SCHEDULER_RECORD}"

"${PYTHON}" "${EXECUTION_DIR}/assemble_terminal_v2.py" \
  --evidence-run-root "${EVIDENCE_ROOT}" \
  --repair-run-root "${REPAIR_ROOT}" \
  --array-job-id 29337 \
  --scheduler-record "${SCHEDULER_RECORD}" \
  --failed-terminal "${EVIDENCE_ROOT}/terminal_report.json" \
  --expected-failed-terminal-sha256 \
    894d4a704214d9ff56f51c09b1c35ad20fb3a108805b28920c3714c584ce2e6f \
  --expected-b1-report-sha256 \
    bedf0800f01344d6b67804d6859c003dca8bcd98648cd3544045243cc88bd35b \
  --expected-b2-report-sha256 \
    ccf8f8c283494266d11403134927cc71d81430dc8a65d398f291379a5557ba78 \
  --output "${REPAIR_ROOT}/terminal_report.json"

echo "h1a2_dlm_2xa800_smoke_assembly_repair_complete=true"
echo "h1a2_dlm_2xa800_smoke_automatic_downstream=false"
