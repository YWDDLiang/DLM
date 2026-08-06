# H1 Immutable Fallback Manifest

State: `frozen_verified`

Verification date: 2026-07-31

This manifest protects the H1 series as the final submission fallback. It does
not transfer ownership of H1 artifacts to the new PlanGraph-DLM workstream and
does not authorize a rerun.

## Read-only roots

The following locations must not be edited, deleted, moved, or used as output
directories by the new workstream:

- `workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation`
- `workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728`
- `runs/20260603_034533-h1a2-epoch2-3-fullmetrics`
- `runs/20260729_h1a2c_jointchem_v1`
- `runs/20260731_h1a2c_p0_p1_sun256_exploratory_v1`
- `runs/20260731_h1a2c_p0_p1_sun256_mpcomplete_v4`

References to those paths are read-only. Any new checkpoint, cache, evaluation,
or report must be written under a fresh PlanGraph-DLM run root.

At verification time, the H1 source workstream is present locally; the frozen
baseline is retained as an archive, and the listed `runs/` identities live on
the execution cluster rather than in this local checkout. Absent local paths
remain reserved names and may not be recreated as new PlanGraph-DLM outputs.

## Frozen bundles

| Artifact | SHA-256 | Verified |
|---|---|---|
| `workstreams/r5c_reactivation_20260728/r5c_frozen_baseline_20260728.tar.gz` | `ad1b7f5b9ee0df0c06396ef1d3865f7a5e7b2e4d3f4b46216445288e04be8325` | yes |
| `workstreams/r5c_reactivation_20260728/r5c_reactivation_bundle_20260728.tar.gz` | `63f699f670ab8c450e7e196ec824f7009ea5a4e9f6e7aee7f743d81f51d25d1b` | yes |

The archive hashes match their adjacent `.sha256` files.

## Source-integrity checks

All entries in these manifests passed `sha256sum -c` on 2026-07-31:

- `workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/SOURCE_SHA256.txt`
- `workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/H1A2_EPOCH2_CODE_SHA256.txt`
- `workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_exploratory_p0_p1_v1/SOURCE_SHA256.txt`
- `workstreams/r5c_reactivation_20260728/h1a2_epoch2_innovation/sun_mp_completion_p0_p1_v1/SOURCE_SHA256.txt`

The workspace currently has no usable Git metadata: the project-level `.git`
path is empty and `git status` does not recognize a repository. Preservation
therefore relies on path isolation plus repeated SHA verification, not on Git
rollback.

## Frozen model and report identities

| Role | Frozen identity |
|---|---|
| H1-A2 epoch-2 Planner adapter | `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a` |
| Frozen R5-C body adapter | `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d` |
| Frozen continuous refiner checkpoint | `573e9b10af64b266b7c6cde4d0f8bdd8a7388fa98d36e2e82db341af3e511e7e` |
| JointChem execution/source manifest | `aedd7e5b4b91720ea2e8490f563eb747886c848c64a9d98196a6ab1cf7b9f874` |
| Corrected JointChem terminal report | `6bfb9cd2927fcfb6017b303f1eed993545c99d37877e968268b2417b3d921ea6` |
| P0 plan raw generations | `bfaf2f9aa92ef4212d11bc71484ae6a60be13fd7239f107f08e419190afedb3e` |
| P1 step-50 selected-checkpoint manifest | `9cad344703e435d32c76f4d242bf7c3ef39a8c1d4a3af72cf36353d22c88644a` |
| P1 plan raw generations | `ac7dd485e81b68ee3cf9b7b5facfe0fbc1869fd3344f90652effffc9b4c00c47` |
| Paired-256 S.U.N. execution manifest | `f4a4e94a6df1acb10e8c6ad8d0b712448371d1dcea79027f6c2b07b8a3303d34` |
| Original paired-256 terminal report | `0faa322603bef556d6bbb00ce1551067dee4bfbf41c13dad56af1d1b1c99112d` |

The corrected JointChem decision is `stop_no_plan_candidate`. It is part of
the H1 evidence and must not be silently replaced by a later reinterpretation.
The R5-C adapter SHA above is the exact identity used by the paired-256
runtime stack; it does not replace any separate raw-body package identity
recorded in the frozen historical dossier.

## Baseline evidence retained for the paper

### Historical 1,000-sample results

| System | Composition valid | Structure valid | Joint valid | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|
| Historical CrysLLMGen | 89.2% | 99.9% | 89.1% | 9.0% | 46.1% |
| H1-A2 epoch 2 | 87.8% | 99.9% | 87.7% | 9.4% | 47.4% |

Coverage-adjusted H1-A2 values, 9.71% strict and 48.94% metastable, are
descriptive diagnostics only and are not substitutes for all-attempt results.
The conditional gold-plan R5-C reference, approximately 10.61% strict and
74.38% metastable after adjustment, remains an upper-reference diagnostic.

### H1 Planner failure evidence

Among 1,186 parsed H1 plans:

- 1,044 were composition-valid;
- 142 were invalid;
- 98 failures were charge-neutrality failures;
- 37 were Pauling-rule failures; and
- 7 lacked usable oxidation states.

### Paired-256 P0/P1 exploratory evidence

All counts use the first 256 attempts with no replacement.

| Arm | Composition valid | Planner-caused invalid | DLM/CIF invalid | Strict S.U.N. | Meta S.U.N. |
|---|---:|---:|---:|---:|---:|
| P0 frozen epoch 2 | 211/256 | 37 | 8 | 22/256 = 8.59% | 125/256 = 48.83% |
| P1 ValidReplay step 50 | 208/256 | 35 | 13 | 16/256 = 6.25% | 106/256 = 41.41% |

These paired results are exploratory and do not authorize promotion.

## Preservation rules

1. Never train in place from an H1 checkpoint directory.
2. Never overwrite an H1 manifest, terminal report, cache, or selected
   checkpoint.
3. Never reuse an H1 run root for a new Slurm job.
4. Never alter H1 denominators or replace failed samples in a retrospective
   report.
5. Before any PlanGraph-DLM submission and after its completion, rerun all four
   source-integrity checks listed above.
6. Any H1-derived initialization must be copied into a new immutable input
   ledger and identified by its original SHA-256.
7. No API key, token, or other secret may be serialized in a manifest.

## Fallback activation

H1 becomes the primary submission line if, by 2026-08-31, the new method lacks
all of the following:

- a frozen implementation and reproducible manifest;
- a successful three-seed confirmation plan in progress or completed;
- a clear positive mechanism result attributable to Planner or DLM changes;
- no material regression in generation success, composition validity, or
  fixed-panel likelihood; and
- a credible route to stable strict S.U.N. above 10% and metastable S.U.N.
  above 50%.

Fallback activation means preserving the H1 scientific story and using the new
experiments only where they improve the evidence. It does not mean rewriting
the H1 historical record.
