#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

e1_job="$(sbatch --parsable "${REPO_ROOT}/slurm/90_story_e1_body_panel.sbatch")"
e2_job="$(sbatch --parsable --dependency="afterok:${e1_job}" "${REPO_ROOT}/slurm/91_story_e2_refine_panel.sbatch")"

printf 'E1 body panel job: %s\n' "${e1_job}"
printf 'E2 refine panel job: %s (afterok:%s)\n' "${e2_job}" "${e1_job}"
