# Paper-method index

Active experiment checklist:
[`BTRD_STABILITY_CHECKLIST_V2.md`](../BTRD_STABILITY_CHECKLIST_V2.md).

Start here. The paper-facing surface is intentionally small; historical code
and experiments remain available but are not required to understand the final
method.

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
6. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — data, seeds, hashes and run
   contracts.

## Code in one screen

| Concept | Stable paper-facing API | Audited implementation |
|---|---|---|
| Science-Constrained LLM Planner | `crystal_dlm.paper_pipeline.planner` | `c3fd_llama_*`, `c3fd_native_plan.py` |
| Exact crystal language | `crystal_dlm.paper_pipeline.representation` | `dynamic_crystal.py` |
| Periodic-relational denoising | `crystal_dlm.paper_pipeline.periodic_residual` | `periodic_relation_*`, `periodic_geometry_*` |
| Basin transport | `crystal_dlm.paper_pipeline.basin_transport` | `basin_transport.py`, `llada_sft.py` |
| Terminal diffusion | `crystal_dlm.paper_pipeline.refinement` | `refine_dlm_with_crysllmgen.py` |
| Metrics | `crystal_dlm.paper_pipeline.evaluation` | `eval_runtime/*`, finalizers |

The facades re-export audited functions; they do not duplicate model
mathematics. `PAPER_PIPELINE.md` at repository root is the execution runbook.
