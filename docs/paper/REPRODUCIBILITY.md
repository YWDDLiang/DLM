# Reproducibility contract

## Portable source contract

The paper-facing source of truth is `configs/paper/mainline_v1.json`. Run:

```bash
PYTHONPATH=src python -m crystal_dlm.paper_pipeline validate
```

Validation fails closed if a stage disappears, an implementation path changes,
G2-B becomes active, strict 125-image PBC is removed, the Plan schema changes,
or inference introduces result selection.

Runtime wrappers hard-stop only on scientific integrity: immutable data and
checkpoint identity, complete sample-index accounting, finite outputs, and the
absence of retry/result selection. Code-commit differences, optional logs and
nonessential cache coverage are recorded in audit manifests instead of
aborting expensive jobs.

## Data and training

| Learned transition | Data | Optimization | Paper identity |
|---|---|---|---|
| C3FD scientific support | MP20-derived typed composition actions | C3FD-v2.5 seed17 | v2.5 final |
| Llama typed Planner | typed-witness train/val `24558/8158` | seed85017, one epoch, LR2e-5, effective batch16, final only | typed Planner final |
| Compact-V2 DLM | full MP20 train/val `27136/9047`, teacher Plan only | seeds82017/82018, 3392 updates, paper seed82017 | seed82017 step3392 |
| G2-PBC-R | same full MP20 teacher data | seed81017, 1696 updates, LR5e-6, final only | method A step1696 |

The current Planner data is a typed-witness subset; only the DLM uses the full
MP20 standard split. This distinction is encoded in separate versioned configs
and cannot be silently collapsed by the CLI.

## Train/serve interface

`C3FD_NATIVE_PLAN_V2` is the sole paper Plan serializer. Its exact fields and
order are shared between Planner output and DLM conditioning. The DLM body is
dynamic `7+4N`: seven global lattice/header tokens and one element/X/Y/Z tuple
per atom. Exact N and element multiplicities are part of the interface, not a
post-generation repair.

## Periodic relational denoising

The promoted implementation uses:

- bounded 125-image triclinic minimum-image distances;
- normalized species-aware margin
  `ReLU((m-d)/m)^2`, with
  `m=clamp(0.55(r_i+r_j),0.60 Å,1.40 Å)`;
- metric/RDF/overlap/coordination weights `0.1/0.1/0.2/0.05`;
- rank64, two-layer, zero-output-initialized relation residual;
- exact step-zero logit equality;
- no q0 uncertainty gate.

The immutable geometry audit checked 2,321,081 pairs and found zero 125-image
disagreements against pymatgen.

## Inference profiles

| Profile | Plans | DLM/refiner | Denominator and role |
|---|---:|---|---|
| main reported | 1,000 | reported H1-A2 protocol | paper headline `105/488` |
| prospective headline | 256 | stream17, DLM seed91117, model494 tau800 | matched BASE/G2 effect |
| full-epoch mechanism | 256 | matched A/B, model494 tau800 | geometry implementation evidence |
| Plan1200 scale | 1,200 sampled; 1,159 official-known; 1,139 valid CIFs | G2-PBC-R and model494 tau800 | main1000 = first861 prior-main valid rows + all139 remainder rows |

Every new profile uses one Plan and one DLM trajectory per requested ordinal.
The prospective profile uses Planner seed22, temperature0.9/top-p0.95, DLM
temperature0.7 exact-axis and refiner seed101117 by sample index. Plan1200 keeps
the same one-trajectory contract and fixes tau800 after the matched depth test.

## Evaluation

Direct composition/structure/joint validity, N/U/NU, CHGNet and official hull
are joined by `sample_idx` under one immutable attempt ledger. Official unknowns
remain unknown and never count stable. Direct and model494 are frozen inherited
components with pinned hashes; the paper reports both raw learned-DLM and
complete-system endpoints.

## Result records

- Main reported result: `105/488` per 1,000.
- Fresh prospective: `docs/36H_FINAL_REPORT_C3FD_G2_20260901.*`.
- Full-epoch mechanism: `docs/G2_FULL_EPOCH_AB_FINAL_20260901.*`.
- Plan1200 scale: `docs/PLAN1200_TAU800_FINAL_20260902.md`.

Credentials, machine-local paths, checkpoints and generated datasets are not
stored in Git. Detailed content identities remain in machine-owned execution
manifests rather than the reader-facing method documentation.
