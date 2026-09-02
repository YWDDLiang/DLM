# Paper-method index

Start here. The paper-facing surface is intentionally small. Every module is
documented by its scientific role, exact interface and supporting experiment;
archived experiment scripts are not required to understand the method. Retired
negative methods are summarized once in [`FAILED_METHODS.md`](../FAILED_METHODS.md).

## Method in one line

**C3FD-supported Llama planning → exact Plan-conditioned `7+4N` crystal DLM →
periodic-relational residual denoising → frozen terminal diffusion.**

## Read in this order

1. [`METHOD_AT_A_GLANCE.md`](METHOD_AT_A_GLANCE.md) — one-page intuition and
   why a residual is needed despite global attention.
2. [`SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS.md`](SCIENTIFIC_QUESTION_AND_CONTRIBUTIONS.md)
   — the central question and exactly three contributions.
3. [`METHOD_MAINLINE.md`](METHOD_MAINLINE.md) — equations, interfaces and
   training/inference details.
4. [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md) — main results and matched
   ablations.
5. [`PAPER_STORY.md`](PAPER_STORY.md) — title, abstract, introduction and
   figure/table plan.
6. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — data, seeds, evidence profiles and run
   contracts.

## Evidence at a glance

| Scale | Evidence | What it establishes |
|---|---|---|
| Main reported result | H1-A2 Strict/Meta `105/488` per 1000 | Paper headline |
| Planner | C3FD `1724/2000→2000/2000`; fused `1200/1200` composition-valid | Scientific support survives learned Llama reweighting |
| Plan/DLM interface | `248/248` strict round trips; `1139/1159` valid CIFs at scale | Exact chemistry and variable-length body contract work |
| G2 residual | refined S.U.N. `19/111→24/117`; paired hull `−16.43 meV/atom` | Periodic feedback improves the learned realization path |
| Terminal diffusion | L6 raw→tau800 Strict/Meta `10/66→48/230` | model494 supplies a distinct fine-scale basin transition |
| Scale profile | tau800 main1000 Strict/Meta `81/486` | The complete pipeline remains effective at scale |

The complete scale report is
[`PLAN1200_TAU800_FINAL_20260902.md`](../PLAN1200_TAU800_FINAL_20260902.md).

## Code in one screen

| Concept | Stable paper-facing API | Audited implementation |
|---|---|---|
| Science-Constrained LLM Planner | `crystal_dlm.paper_pipeline.planner` | `c3fd_llama_*`, `c3fd_native_plan.py` |
| Exact crystal language | `crystal_dlm.paper_pipeline.representation` | `dynamic_crystal.py` |
| Periodic-relational denoising | `crystal_dlm.paper_pipeline.periodic_residual` | `periodic_relation_*`, `periodic_geometry_*` |
| Terminal diffusion | `crystal_dlm.paper_pipeline.refinement` | `refine_dlm_with_crysllmgen.py` |
| Metrics | `crystal_dlm.paper_pipeline.evaluation` | `eval_runtime/*`, finalizers |

The facades re-export audited functions; they do not duplicate model
mathematics. `PAPER_PIPELINE.md` at repository root is the execution runbook.
