#!/usr/bin/env bash
#SBATCH --job-name=plan1200-tau89
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA800-SXM4-80GB:4
#SBATCH --mem=480G
#SBATCH --time=05:00:00
#SBATCH --output=plan1200-tau89-%j.out
#SBATCH --error=plan1200-tau89-%j.err
set -Eeuo pipefail
readonly ROOT="${H1A2_CANDIDATE_ROOT:?}"
readonly CODE="${H1A2_CODE_ROOT:?}"
readonly BASE="${ROOT}/runs/g2_known1200_baseline_fast_39183"
readonly RUN="${ROOT}/runs/plan1200_tau800_tau900_${SLURM_JOB_ID}"
readonly PY=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/python
readonly TORCHRUN=/public/home/jiaosz/miniconda3/envs/diff_meets_diff/bin/torchrun
readonly REF=/public/home/jiaosz/hengzhang/Code/crysllmgen-main/out/mp_20/22042026/203930/model_494.pt
readonly CRYS=/public/home/jiaosz/hengzhang/Code/crysllmgen-main
mkdir "$RUN" "$RUN/logs"; trap 'set +e; touch "$RUN/_FAILED"; exit 1' ERR
IFS=',' read -r -a gpu <<<"${CUDA_VISIBLE_DEVICES:?}"; [[ ${#gpu[@]} -eq 4 ]]
PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/combine_known1200_proposals.py" --baseline-run "$BASE" --output-dir "$RUN/combined"
refine(){ local steps="$1" devices="$2" port="$3"; env CUDA_VISIBLE_DEVICES="$devices" "$TORCHRUN" --standalone --master-port "$port" --nproc_per_node=2 "$CODE/src/scripts/refine_dlm_with_crysllmgen.py" --proposal-graphs "$RUN/combined/proposal_graphs.pt" --checkpoint "$REF" --output-dir "$RUN/tau$steps/refine" --crysllmgen-dir "$CRYS" --batch-size 1 --seed-from-graph-field refiner_seed --diff-steps "$steps" --max-proposals 1139 >"$RUN/logs/tau$steps.out" 2>"$RUN/logs/tau$steps.err"; }
refine 800 "${gpu[0]},${gpu[1]}" 29643 & p0=$!; refine 900 "${gpu[2]},${gpu[3]}" 29644 & p1=$!; wait "$p0"; wait "$p1"
for steps in 800 900; do PYTHONPATH="$CODE/scripts:$CODE/src" "$PY" "$CODE/scripts/assemble_known1200_refined.py" --accounting-jsonl "$RUN/combined/all_requested_accounting.jsonl" --refine-dir "$RUN/tau$steps/refine" --output-dir "$RUN/tau$steps/generation" --steps "$steps"; done
find "$RUN" -type f ! -name OUTPUTS.sha256 -print0|sort -z|xargs -0 sha256sum >"$RUN/OUTPUTS.sha256"; trap - ERR; touch "$RUN/_SUCCESS"
