# PlanGraph-DLM Implementation Tasks V1

Status: `remote_preflight_complete_engineering32_manifest_frozen_no_submission`

The tasks below are ordered by information value. A later phase begins only
after the preceding gate is satisfied.

## Phase 0: protect the fallback

- [x] Verify both frozen H1 archives against their SHA-256 sidecars.
- [x] Verify the H1, epoch-2 recovery, paired-256 S.U.N., and MP-completion
  source manifests.
- [x] Record H1 checkpoint, report, and baseline identities.
- [x] Add a pre-submit read-only check that refuses any output path under an
  H1 root.
- [x] Add a command that reruns all H1 checksum manifests before and after a
  new experiment.

Deliverable: an H1 guard script and passing dry-run tests.

## Phase 1: freeze the PlanGraph data contract

- [x] Define a versioned JSON schema for composition, oxidation assignment,
  symmetry/lattice constraints, site groups, and dependency edges.
- [x] Write a deterministic converter from the frozen supervised examples.
- [x] Produce a field-coverage and failure report before training.
- [x] Run a schema-only feasibility audit on the local test mirror: 9,046 of
  9,046 rows converted. Do not use its distributions for tuning.
- [x] Prove that energy/stability metadata are excluded from serialized
  prompts and targets.
- [x] Implement an atomic full-denominator builder that refuses existing
  outputs, records ordered identities, and rejects cross-split overlap.
- [x] Freeze train/validation row identities and a fixed validation panel.
- [x] Add schema round-trip, count, leakage, and malformed-record tests.

Deliverable: `plangraph_v1` schema, conversion manifest, leakage report, and
unit tests.

Stop if fewer than 98% of eligible frozen training rows convert without
heuristic repair. Diagnose the schema instead of dropping hard rows silently.

## Phase 2: implement planned DLM corruption

- [x] Factor the current iid candidate/mask path into a directly testable
  model-free forward policy.
- [x] Implement pure D1 current-order groups.
- [x] Implement pure D2 PlanGraph dependency groups.
- [x] Add an explicit `iid_fraction`/`planned_fraction` configuration.
- [x] Make CPU policy selection seed-reproducible.
- [x] Represent planned loss as active-group-only, separate from future masks.
- [x] Preserve exact current behavior when `planned_fraction=0`.
- [x] Add token-coverage, prerequisite, future-mask, and loss-mask tests.
- [x] Add training-collator padding and dynamic-length integration tests.
- [x] Make the fixed validation-panel iid mask stateless and independent of
  global Torch RNG.
- [x] Verify exact CPU/CUDA stateless-mask parity on the local CUDA device.
- [x] Run a deterministic 10,000-trial D1 CPU mask simulation and D2 unit
  simulation.
- [x] Add a content-keyed shuffled-order mechanism control without changing
  row identities or group membership.

Deliverable: tested corruption policies plus a CPU-only simulation report of
mask frequencies and group coverage.

Stop if D2 requires a representation change. Use the registered D1 feasibility
fallback for this submission and defer the representation change.

## Phase 3: Planner-only screen

- [ ] Freeze a 512-ordinal seed ledger.
- [ ] Evaluate P0, PG, and PG-shuffle with all attempts retained.
- [ ] Attribute parse, schema, charge, Pauling, and oxidation failures.
- [ ] Compare unique-formula rate and distribution drift.
- [ ] Apply G1 without changing thresholds.

Deliverable: one immutable Planner report and a continue/stop decision.

## Phase 4: DLM preflight and checkpoint selection

- [x] Freeze a separate four-arm 32-row engineering-pilot manifest; this is a
  review point and does not authorize submission.
- [ ] Run the 32-row engineering pilot to measure wall time and memory only;
  exclude its outputs from every scientific result and later initialization.
- [ ] Freeze an A800-hour envelope from that pilot before full submission.
- [ ] Run loss/gradient preflight on D0, D1, and D2.
- [ ] Train the 2:1 iid:planned arms for at most 400 updates.
- [ ] Evaluate every 50 updates on the frozen panel.
- [ ] Select checkpoints using the registered NLL and direct-margin rule.
- [ ] Run D2 1:2 only if D2 2:1 passes.
- [ ] Run the shuffled-order control at the bounded diagnostic budget.
- [ ] Scan logs for OOM, NaN, CUDA, NCCL, traceback, and silent zero-loss
  groups.

Deliverable: training reports, fixed-panel likelihood report, selected
checkpoint manifests, and G2 decision.

## Phase 5: primary paired-256 study

- [ ] Freeze the first 256 all-attempt ordinal seeds.
- [ ] Run M0/M1/M2/M3 with the same frozen refiner and generation settings.
- [ ] Produce one record per attempt with the earliest failure label.
- [ ] Evaluate S.U.N. from the frozen cache/API snapshot.
- [ ] Report paired bootstrap, McNemar, factorial interaction, diversity, and
  distribution drift.
- [ ] Apply G3 without replacing failed attempts.

Deliverable: paired-256 terminal report and a confirmation decision.

## Phase 6: confirmation and paper freeze

- [ ] Freeze three 1,000-attempt seed ledgers before generation.
- [ ] Run only M0 and the preregistered winning candidate.
- [ ] Evaluate raw all-attempt and accepted-N protocols.
- [ ] Apply G4 per seed and pooled.
- [ ] Freeze the method by 2026-08-31.
- [ ] Activate H1 fallback immediately if G4 cannot support the paper.
- [ ] Begin the full manuscript before final-scale evaluation completes.

Deliverable: confirmation report, frozen claims table, and first complete
manuscript by 2026-09-05.

## Resource and execution constraints

- Each training/generation element may use at most one A800 and eight CPUs.
- CPU allocation must remain at or below eight CPUs per A800.
- Concurrency and wall-time require an explicit execution manifest.
- No automatic downstream, automatic promotion, or implicit job submission.
- No job may be cancelled, resubmitted, or expanded by a monitoring task.
- Runtime API keys must be injected through the scheduler environment and
  redacted from logs; they must never be written into this workstream.

## Immediate next execution slice

The frozen-data gate is complete:

1. the owner restored the frozen outer/nested SSH path;
2. read-only discovery pinned the train/validation source, tokenizer, base
   model, and R5-C initialization;
3. `build_plangraph_v1_sft_data.py` published 36,183/36,183 rows with no
   cross-split training-pair overlap;
4. `preflight_planned_corruption_data.py` passed D1, D2, and D2-shuffle on
   36,183/36,183 rows with the real tokenizer at `max_length=768`; and
5. the canonical H1 fallback still reports 68/68 registered entries intact.

The A800 H1 worktree is a non-authoritative incomplete mirror with missing
files only; no present registered file has drifted. Do not patch that mirror.

`ENGINEERING_PILOT_32_MANIFEST_V1.json` is now the next review point. A new
explicit user authorization is required before writing the run root or
submitting its proposed four-arm A800 array. This task list does not authorize
Slurm submission.
