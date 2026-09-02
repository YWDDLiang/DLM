# Science-Constrained Crystal Language Pipeline

This branch exposes the research method as one coupled, reproducible generator
rather than a collection of disconnected experimental scripts:

\[
z\sim p_\phi(z\mid \mathcal C_{\mathrm{C3FD}}),\qquad
x_{\mathrm{raw}}\sim p_\theta(x\mid z;\mathcal R_{\mathrm{G2}}),\qquad
x_{\mathrm{final}}\sim K_{494}(\cdot\mid x_{\mathrm{raw}},z).
\]

- `C3FD` supplies typed scientific support inside Llama decoding.
- Llama converts that support into one coherent Compact-V2 Plan `z`.
- The exact dynamic `7+4N` interface conditions every masked-DLM prediction.
- G2 lifts soft DLM states into periodic species relations and returns them as
  a zero-initialized residual during denoising.
- model494 is the fixed terminal diffusion transition of the same coarse-to-
  fine realization process.

## Validate the method

From a source checkout:

```bash
PYTHONPATH=src python -m crystal_dlm.paper_pipeline validate
PYTHONPATH=src python -m crystal_dlm.paper_pipeline show
PYTHONPATH=src python -m crystal_dlm.paper_pipeline stage sample-plan
```

The CLI is intentionally read-only. It validates every versioned component,
repository path, stage order and scientific invariant, then returns the audited
Slurm/Python contract for a requested stage. It never submits work itself.

## Canonical stages

| Order | Stage | Audited contract |
|---:|---|---|
| 1 | Build C3FD scientific support | `slurm/53_c3fd_v25_requested1000.sbatch` |
| 2 | Build fused Planner data | `slurm/109_build_c3fd_llama_fused_data.sbatch` |
| 3 | Train Llama typed Planner | `slurm/110_train_c3fd_llama_typed_planner.sbatch` |
| 4 | Sample one global Plan | `slurm/125_sample_final_fused_planner.sbatch` |
| 5 | Build Compact-V2 DLM data | `slurm/86_build_c3fd_native_sft_data.sbatch` |
| 6 | Train Plan-conditioned DLM | `slurm/87_train_c3fd_native_teacher_sft.sbatch` |
| 7 | Train periodic-relation DLM | `slurm/129_train_g2_full_epoch_ab.sbatch`, method A |
| 8 | Generate and refine headline profile | `slurm/126_g2_final_generation.sbatch` |
| 9 | Evaluate raw/refined structures | `slurm/127_g2_final_offline_eval.sbatch` |
| 10 | Join fixed official hull and finalize | `scripts/finalize_c3fd_g2_final_sun.py` |

Machine paths and environment activation stay in execution contracts or a
user-owned local environment file. They are not embedded in portable configs.

## Evidence profiles

The method architecture is shared, while the evidence profiles answer
different questions:

- `main_reported_result`: paper headline Strict/Meta S.U.N. `105/488` per 1,000.
- `prospective_headline`: registered step348 G2, fresh seed22 cohort; establishes
  the end-to-end system result (`24/117` refined Strict/Meta S.U.N.).
- `full_epoch_mechanism`: selected G2-PBC-R step1696; establishes stronger raw
  periodic realization (`128/256` Direct versus BASE `118/256`) and rejects the
  uncertainty-gated B variant.
- `plan1200_scale_validation`: combines the selected tau800 endpoint into a
  1,000-row scale profile and reaches Strict/Meta `81/486`.

The profiles are never merged into a synthetic checkpoint or selected-row
result. Their exact mapping lives in `configs/paper/mainline_v1.json`.

## Configuration and evidence

- Paper-method index: `docs/paper/README.md`.
- One-page method intuition: `docs/paper/METHOD_AT_A_GLANCE.md`.
- Portable configs: `configs/paper/*.json`.
- Pipeline inventory: `docs/paper/PIPELINE_INVENTORY.md`.
- Legacy compatibility: `docs/paper/LEGACY_TO_MAINLINE_MAP.md`.
- Reproduction ledger: `docs/paper/REPRODUCIBILITY.md`.
- Architecture source: `docs/paper/architecture_mainline.mmd`.
- Prospective result: `docs/36H_FINAL_REPORT_C3FD_G2_20260901.*`.
- Full-epoch mechanism result: `docs/G2_FULL_EPOCH_AB_FINAL_20260901.*`.
- Plan1200 tau800 scale result: `docs/PLAN1200_TAU800_FINAL_20260902.md`.

No retry, replacement, survivor filtering, reranking, best-of-N, checkpoint
selection, or test-outcome training is part of the paper pipeline.
