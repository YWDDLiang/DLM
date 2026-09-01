# Geometry and Stability Fallback Review

Status: APPROVED. This document governs only the fallback after
the already-running Basin-Target Residual Distillation (BTRD) experiment.

## Understanding lock

- The paper mainline is fixed: a C3FD-constrained Llama Planner produces a
  Compact-V2 Plan; a masked `7+4N` DLM realizes the structure; the periodic G2
  residual exposes high-salience lattice/site relations; model494 is the
  frozen endpoint refiner.
- The next result must improve raw structural validity and thermodynamic
  stability/S.U.N. without changing Planner compositions or resampling Plans.
- The method must remain reproducible, train-only, and explainable as an
  extension of the existing periodic residual. No evaluation outcome, hull
  query, survivor selection, reranking, or best-of-N signal may enter training.
- Six A800s are the global ceiling. Existing Plans and cached labels are reused
  whenever the scientific comparison permits it.
- BTRD remains the primary experiment. A fallback is activated only if BTRD is
  technically complete but does not show a useful raw direction.

Terminology used below:

- Direct is the conjunction of composition validity and CrysLLMGen structural
  validity;
- official-known means the reference phase diagram was available before DLM
  generation;
- Strict S.U.N. is reconstructed, novel and unique with official energy above
  hull at most 0; and
- Meta S.U.N. uses the same intersection with energy above hull at most
  0.1 eV/atom.

## Revised design under review

### Current experiment — Basin-Target Residual Distillation (BTRD-tau200)

The current implementation is interpreted as target distillation, not as
identification of a physical diffusion vector field. A frozen model494 endpoint
replaces the clean geometry target for the registered teacher rows; masked CE
and periodic geometry losses teach the existing G2 residual to reconstruct this
basin-target geometry. The base DLM and Compact-V2 LoRA remain frozen.

### Sole fallback — Tau800 Tail-Aware BTRD (BTRD-800T)

If BTRD-tau200 is technically complete but provides no useful raw direction,
run one fixed fallback rather than selecting between multiple methods:

1. reuse the same frozen MP20-train rows, Plans, G2 raw proposals, site order,
   and model494 seed;
2. replace the tau200 endpoint with one tau800 endpoint teacher;
3. keep the original MP20 anchor rows and the same row-level anchor
   substitution rule for unavailable teacher endpoints;
4. train only the existing G2 residual for the same 512 updates; and
5. change the existing overlap term from pair mean to the already implemented
   `kappa=0.10, beta=0.50` sitewise tail mixture, while keeping its total weight
   at 0.20.

The frozen tau800 model494 endpoint is a stronger empirical endpoint proxy; it
is not called a direct stability label. The tail-risk term makes a catastrophic
nearest-neighbor collision salient instead of diluting it among all pairs.
BTRD-800T is explicitly a bundled fallback that changes both endpoint depth
and overlap aggregation. It adds no inference module, no energy/hull label, no
online preference loop, and no intermediate-trajectory claim. Its tail
temperature is denoted `kappa=0.10` and its tail mixture `beta=0.50`, avoiding
confusion with model494 tau200/tau800 step counts.

## Fast diagnosis and execution rule

Evaluate BTRD-tau200 first on the fixed first256 Plans using raw body
accounting, Direct, raw CHGNet, and cached official hull/S.U.N. "Useful raw
direction" means that every Fixed256 promotion condition in
`BTRD_METHOD_CONTRACT.md` passes. An engineering or accounting failure never
activates the fallback. A scientific non-pass activates the sole registered
BTRD-800T experiment; there is no geometry-versus-stability branch chosen from
evaluation outcomes. BTRD-800T must use an untouched confirmation remainder and
disclose that its design followed a development diagnosis.

The BTRD-800T design in this document is locked before any BTRD Fixed256 result
is read. The first256 is the disclosed development block that determines
whether the already-registered fallback runs; the remaining 903 rows are never
used to design or activate it and serve only as confirmation.

The denominator ledger is explicit: the Planner produced 1,200 requests; 1,159
have official reference support and form the executable official-known queue;
41 remain unavailable rather than being mapped stable. The frozen first256 is
the prefix of the official-known queue and the remaining 903 rows are its
confirmation remainder. Membership files and SHA-256 values must be written
before BTRD evaluation. The current G2 baseline contains 20/1,159
non-constructible bodies; they remain failures in the fixed 1,159 denominator
and are skipped only for downstream compute.

Metric denominators are fixed as follows:

- Planner composition validity uses all 1,200 Planner requests;
- body construction, composition validity, Direct validity and Strict/Meta
  S.U.N. use the 1,159 official-known requests, with non-constructible bodies
  counted as failures;
- CHGNet and official-hull continuous summaries use their explicit known
  subsets but always report missing counts; and
- valid-CIF rates are secondary CrysLLMGen-compatible conditional metrics,
  never replacements for the fixed 1,159 rate.

BTRD-800T is a registered design, not yet an executable job. Before activation it
must have a tau800-specific data manifest/builder/wrapper, consume the immutable
39184 combined proposal SHA without regenerating bodies, preserve global row
indices and refiner seed, explicitly set tail temperature 0.10 and mix 0.50,
and record expected/actual walltime, A800-hours and storage. The teacher is
described as a frozen model494 endpoint proxy, not a proven physical vector
field or direct stability label.

## Decision log

| Decision | Alternatives considered | Current rationale | Review status |
|---|---|---|---|
| Keep BTRD primary | replace it immediately; online RL | It is already running and distills a frozen model494 endpoint proxy without outcome labels. | locked |
| Replace F1/F2 with sole BTRD-800T fallback | invalid-geometry counterfactuals; multi-horizon trajectory; decoder repair; energy RL | Challenger showed that near-singular counterfactuals exceed the audited PBC domain and stochastic intermediate states do not identify a physical vector field. BTRD-800T instead bundles a deeper endpoint proxy with an existing collision-tail objective. | revised after challenger |
| Interpret BTRD as endpoint-target distillation | claim raw-to-basin transport or vector-field recovery | The current data contains a model494 endpoint proxy but not an explicit paired raw source geometry. | revised after challenger |
| Dual-report denominators | silently drop failures; count only all requests | It permits fair CrysLLMGen comparison while preserving end-to-end accountability. | accepted |

## Review record

### Skeptic / Challenger

Disposition: REJECTED the initial F1/F2 design.

Accepted objections:

- the current dataset does not preserve an explicit G2 raw source geometry, so
  raw-to-basin vector-field language was unsupported;
- fixed 125-image PBC was not audited for deliberately near-singular cells;
- stochastic model494 intermediate states were not a physical vector field;
- choosing F1 versus F2 from first256 outcomes would be method-level adaptive
  selection;
- fixed `p_mask=0.5` language did not match the trainer's random mask schedule.

Resolution: remove invalid-cell counterfactuals, remove multi-horizon claims,
remove the adaptive F1/F2 branch, interpret the current run as endpoint-target
distillation, and register only BTRD-800T as the stronger fixed fallback.

### Constraint Guardian

Disposition: REVISE.

Accepted requirements:

- equate fallback activation with all registered Fixed256 scientific
  conditions and exclude engineering failures;
- freeze first256/remainder membership and hashes before evaluation;
- require a tau800-specific executable contract before BTRD-800T can run;
- audit proposal/refiner global indices, N, and ordered species before any
  BTRD training;
- record the exact 39184 code/data/model provenance and actual compute; and
- call model494 output an endpoint proxy rather than a proven physical basin
  vector field.

Implementation response: `audit_btrd_teacher_order.py` now defines the
fail-closed identity audit, and the BTRD training wrapper requires its success
marker. Remaining provenance, membership hashes and observed cost are populated
after job39184 reaches a terminal state.

Paper/user and arbiter reviews remain pending.

### Paper / User Advocate

Disposition: REVISE.

Accepted requirements:

- use `frozen model494 endpoint proxy` consistently and avoid vector-field or
  direct-stability-label language;
- rename the bundled fallback BTRD-800T and distinguish `kappa=0.10` tail
  temperature from model494 tau;
- define the 1,200, 1,159 and valid-CIF denominators by metric;
- state that BTRD-800T was locked before reading Fixed256 outcomes; and
- define Direct, official-known, Strict S.U.N. and Meta S.U.N. at first use.

All requested changes are incorporated above.

### Integrator / Arbiter

Final disposition: **APPROVED**.

The arbiter accepted every recorded objection after the endpoint-proxy
wording, early terminology definitions and `raw Direct >= baseline +1/256`
condition were fixed. No unresolved design objection remains. Runtime
provenance and measured cost are terminal-result fields, not permission to
change the method.
