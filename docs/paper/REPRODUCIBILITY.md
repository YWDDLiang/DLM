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

| Learned transition | Data | Optimization | Frozen identity |
|---|---|---|---|
| C3FD scientific support | MP20-derived typed composition actions | C3FD-v2.5 seed17 | checkpoint `87c1673f...d17d53` |
| Llama typed Planner | typed-witness train/val `24558/8158` | seed85017, one epoch, LR2e-5, effective batch16, final only | adapter `7638b05d...84c75` |
| Compact-V2 DLM | full MP20 train/val `27136/9047`, teacher Plan only | seeds82017/82018, 3392 updates, paper seed82017 | adapter `06cd5465...62b20` |
| G2-PBC-R | same full MP20 teacher data | seed81017, 1696 updates, LR5e-6, final only | adapter/relation `712c9419...bfce` / `675d68b9...b9f8` |

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

## Inference

- Planner source/sampling seed22, temperature0.9, top-p0.95.
- DLM stream17, seed91117, temperature0.7, exact-axis.
- One Plan and one DLM trajectory per requested ordinal.
- model494 tau800, refiner seed101117 by sample index.
- Fixed requested denominator256; every parse/refinement failure retained.

## Evaluation

Direct composition/structure/joint validity, N/U/NU, CHGNet and official hull
are joined by `sample_idx` under one immutable attempt ledger. Official unknowns
remain unknown and never count stable. Direct and model494 are frozen inherited
components with pinned hashes; the paper reports both raw learned-DLM and
complete-system endpoints.

## Result identities

| Artifact | SHA-256 |
|---|---|
| Final prospective Plan | `5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb` |
| Prospective official source | `138e547f9f2d19c52e55586b96d7c9394d38a6330da0d49a9cf017dca641b6a6` |
| Prospective final JSON | `1b99aa33d3d6072006e17309874866af862067f4becd37caeec5f154f99b3070` |
| Full-epoch evaluation outputs | `6ca71897b28d425780e4f4bbe9a5693502c5be4bb7c74a970624bf9a4efcaa00` |
| Full-epoch final JSON | `b50dd8d291daf46d29ff916e0b34395e7f252a11cbaaa8e99e6378f4a8819881` |

Credentials, machine-local paths, checkpoints and generated datasets are not
stored in Git. A reproduction host supplies them through its local environment
and verifies every hash before execution.
