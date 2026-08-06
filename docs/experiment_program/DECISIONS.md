# Flagship Paper Decision Log

Updated: 2026-07-17, v3 implementation freeze

## D1 — One Flagship Paper

- Decision: build one oral-quality crystal-generation paper.
- Alternatives: separate DLM, MP-Doob, and co-diffusion papers.
- Reason: the alternatives overlap and would split compute, evidence, and
  reviewer attention. DLM is evaluated as the discrete transition engine of the
  flagship, not protected as an independent claim.

## D2 — Crystal-Only Scope

- Decision: evaluate only crystal generation and crystal-specific recovery
  tasks.
- Alternatives: add molecules, DNA, recommendation, Sudoku, or general
  black-box optimization.
- Reason: generality comes from space groups, orbit topologies, datasets,
  corruption regimes, material families, and evaluators.

## D3 — Stratified Wyckoff Quotient State

- Decision: define the state as a disjoint union of continuous crystal strata
  indexed by an unordered multiset of Wyckoff orbit instances.
- Discrete fields: space group, active orbit multiset, Wyckoff type, and species.
- Deterministic fields: multiplicity and total atom count.
- Continuous fields: symmetry-compatible lattice metric and orbit free
  coordinates.
- Reason: topology changes alter the legal continuous coordinates and may alter
  their dimension. A flat token string does not express this structure.

## D4 — Trans-Dimensional Revision Is The Main Contribution

- Decision: the main mechanism is reversible orbit birth, death, species change,
  and Wyckoff-type change across strata, with continuous denoising inside the
  current stratum.
- Alternatives: one-way discrete proposal followed by a refiner; generic joint
  diffusion with one shared timestep; cross-attention without explicit events.
- Reason: MatterGen, SymmCD, and MCFlow already cover weaker joint or
  bidirectionally conditioned formulations. Generic co-denoising is not enough.

## D5 — Geometry-Adaptive True Remasking

- Decision: committed discrete fields may be returned to the masked state and
  changed when calibrated geometry evidence indicates conflict.
- Required evidence: discrete confidence, collision deficit, coordination
  anomaly, lattice strain, symmetry residual, continuous-score norm, and basin
  uncertainty. No MLIP evidence enters the revision head.
- True-remask criterion: an already committed field must be able to become
  masked and later take a different value; orbit death/birth must remove/create
  continuous state.
- Reason: the legacy sampler and LLaDA-Rec-style adaptive commitment do not
  revise committed tokens.

## D6 — DLM Is Prioritized But Falsifiable

- Decision: a typed masked/set denoiser is the prioritized discrete engine.
- Required controls: matched WQ-AR, WQ-D3PM, monotonic WQ-DLM, confidence-only
  revision, and joint-no-revision.
- Promotion: DLM must beat WQ-AR/D3PM on high-corruption full-protostructure
  recovery and pass end-to-end matched-compute gates.
- Failure policy: if DLM fails, run only one stratified pilot with the strongest
  discrete kernel and remove DLM-superiority language.
- Reason: existing results show that the old exact-length DLM is a constrained
  coarse executor, not evidence of general DLM superiority.

## D7 — Explicit Variable Orbit Topology

- Decision: the semantic state is a dynamic unordered orbit multiset with explicit
  birth/death transitions.
- Batching: padding or a NULL canvas is allowed only as an implementation
  artifact that does not enter the probability definition, loss, or output.
- Required tests: permutation, padding, support, cardinality, and chart-bridge
  invariance.
- Reason: a fixed padded canvas would repeat the legacy PAD shortcut and make
  the contribution look like a direct MiAD mirage-token transplant.

## D8 — Formal Support Before Full Training

- Decision: full training is blocked until event kernels, target-chart bridges,
  compatibility masks, and projectors pass formal numerical tests.
- Claims not made: exact likelihood, detailed balance, or RJ-MCMC validity.
- Reason: a sum of CE and geometry losses plus cross-attention is not by itself a
  coherent stratified generative process.

## D9 — Matched Attribution Is Mandatory

- Decision: all causal WQ comparisons share representation, continuous backend,
  data, optimizer family, update count, final frozen refiner, and
  method-independent pair IDs/noise. Audit attempt IDs remain method-specific.
- Tolerances: parameter count within 5 percent and training FLOPs within
  10 percent.
- Inference: report exact component calls and actual FLOPs, wall time, memory,
  and GPU-hours.
- Reason: B-ATOM-AR versus WQ-DLM changes both representation and engine and
  cannot establish a DLM contribution.

## D10 — Prior-Art Claim Boundary

- Decision: do not claim the first crystal DLM, the first joint
  discrete--continuous diffusion, the first bidirectional atom/structure model,
  or the first variable-cardinality generator.
- Direct neighbors: MatterGen, WyckoffDiff, SymmCD, SGEquiDiff, MCFlow, MiAD,
  CrysLLMGen (*LLM Meets Diffusion*), and LLaDA-Rec.
- Candidate novelty: topology-indexed continuous strata plus orbit-level
  dimension-changing events and constraint-aware adaptive true rollback.

## D11 — MP-Doob Is Optional

- Decision: activate marginal preservation only after a pre-registered
  family/prototype drift greater than 2 pp.
- Alternatives: make MP-Doob a co-equal contribution or enable it by default.
- Reason: it must control documented collapse, not rescue a failed core claim.

## D12 — No New DFT

- Decision: make claims about robust MLIP-predicted S.U.N., not DFT or
  experimental stability.
- Evaluation: one guide MLIP, one primary held-out MLIP, one secondary held-out
  MLIP, evaluator-specific hulls, public-label calibration where available, and
  separate held-out-only and all-MLIP consensus.
- Naming: use MLIP-S.U.N. whenever the stability label comes only from MLIPs.

## D13 — Attempt-Level Evaluation

- Decision: every submitted attempt remains in the denominator.
- Failures: parse, topology, bridge, projection, graph, refinement, relaxation,
  unsupported element, missing energy/hull, cache mismatch, timeout, duplicate
  ID, and seed mismatch are failures.
- Prohibited: generate-1200/select-1000, survivor-only metrics,
  coverage-adjusted headlines, output-order truncation, retries, and
  best-of-seed reporting.
- CrysLLMGen note: its reported sampling removes roughly 2--5 percent invalid
  compositions. Our matched one-way baseline retains those submitted attempts
  as failures and never replaces them.

## D14 — Relaxation Consistency Is Conditional

- Decision: add relaxation consistency only after geometry-adaptive topology
  revision passes its development gate.
- Promotion: guide gains must transfer by at least 1 pp to each held-out MLIP
  and reduce raw-to-relaxed displacement and strain.
- Failure policy: remove relaxation consistency without weakening the core
  stratified revision paper.

## D15 — Reversible Legacy Cleanup

- Decision: keep superseded artifacts under
  archive/20260710_pre_wyckoff/ and preserve the two canonical historical runs.
- Reason: retain negative results, evaluator provenance, and the legacy DLM
  needed for the monotonic historical baseline.

## D16 — CrysLLMGen Is The Direct One-Way Hybrid Baseline

- Decision: implement B-WQ-DISC-ONCE as a CrysLLMGen-style discrete/geometry
  proposal injected into continuous diffusion at a validation-selected
  intermediate time, while freezing topology.
- Required controls: tau/T in {0.25, 0.50, 0.75, 1.00}, proposed geometry,
  topology-conditioned fresh noise, and matched random geometry.
- Attribution: the core must beat the best frozen tau at equal calls and must
  correct initially wrong topology. A better proposal alone is insufficient.
- Reason: *LLM Meets Diffusion* already establishes that a chemically strong
  discrete proposal followed by continuous refinement can improve crystal
  metrics, but explicitly lacks mutual feedback.

## D17 — Executable Metric Contract

- Decision: every metric freezes denominator, stage, evaluator, reference,
  matcher/tolerance, subset hash, and selection rule.
- Naming: use MLIP-SUN@0.0 and MLIP-SUN@0.1; never compare these as if they were
  DFT S.U.N.
- Mandatory additions: raw/common-refiner/relaxed stages; prototype,
  protostructure, structure, and substitution-aware novelty; space-group and
  Wyckoff-dimension JSD; density/element/atom/orbit distribution distances;
  force, convergence, displacement, and end-to-end yield per GPU-hour.
- Statistics: duplicate components are computed on the fixed batch. Ordinary
  with-replacement bootstrap does not recompute uniqueness; U(n) uses repeated
  sampling without replacement.

## D18 — Four A800s For Four Weeks

- Decision: hard cap four A800 GPUs, 28 calendar days, 2050 usable GPU-hours
  versus the physical maximum of 2688.
- Reserve: at least 800 GPU-hours for frozen champion/final runs in Week 4.
- Freezes: DLM decision Day 7, champion Day 14, core Day 17, final method Day 21,
  evidence Day 28. No tuning after Day 21.
- First cuts: full OOD, FWD/FMD, basin calibration, and external retraining.
  Never cut the champion/final three seeds, 10k denominator, second held-out
  MLIP, or Tier-A symmetry/novelty diagnostics.

## Closed Implementation Decisions

The evaluator versions, symmetry tolerance policy, native 6-layer CSP/set
backbone, dynamic ragged-set state, fixed committed space group, orbit-event
support, call grids, matcher panel, threshold rule, training boundaries, and
four-week cell matrix are now frozen in `protocol_v3.yaml` and
`experiment_registry_v1.yaml`. Asset SHA256 values are filled only by the
login-node asset lock and must exist before the first evaluator job.

## D19 — Label-Free Geometry Evidence And Torus Score

- Decision: revision evidence during training is computed from the currently
  corrupted structure and candidate state only. Corruption labels, clean
  Wyckoff types, and score targets cannot enter the revision-head input.
- Continuous-score evidence: detached orbit RMS from the current pre-evidence
  coordinate head, with the frozen log compression in the protocol.
- Orbit target: the gradient of the wrapped-Gaussian log density on the torus,
  evaluated with the frozen integer-image radius. It is not an unwrapped
  Euclidean residual disguised as periodic noise.
- Event budget: at most one explicit topology-event-kernel draw per reverse
  step. Parallel categorical-chain or masked-field commits are field updates,
  not extra topology-event draws.
- Reason: these choices prevent recovery-label leakage and make the claimed
  geometry-to-topology feedback and periodic score objective falsifiable.

## D20 — Narrow CHGNet Metadata Waiver

- Observation: the frozen environment has Torch 2.4.0+cu121, while the pinned
  `chgnet==0.4.2` wheel declares `torch>=2.4.1`.
- Authorization: the user explicitly authorized retaining Torch 2.4.0 and
  installing only CHGNet itself with `--no-deps` on 2026-07-17.
- Boundary: the resolver must still install and validate every active non-Torch
  CHGNet dependency. Within the project dependency closure, `pip check` may
  contain exactly this one CHGNet/Torch mismatch. Unrelated global-environment
  failures that existed before the run are hash recorded and must remain
  byte-for-byte unchanged; any new or removed line is a hard failure.
- Evidence: the wheel metadata, full pip-check line, retained Torch build,
  source-bundle SHA256, and post-install validation are stored in the immutable
  model dependency-waiver record and bound into the MLIP asset lock.
- Promotion: CPU and offline Slurm CUDA checkpoint smoke remain mandatory. A
  runtime failure removes CHGNet rather than widening the waiver or changing
  Torch.

## D21 — Evaluator-Scoped MatterSim Runtime

- Observation: `mace-torch==0.3.13` requires `e3nn==0.4.4`, while
  `mattersim==1.1.2` requires `e3nn>=0.5.0`; these requirements cannot be
  satisfied in one import path.
- Authorization: on 2026-07-17 the user authorized the necessary environment
  accommodations while retaining the frozen Python, Torch, CUDA, NumPy, and
  materials ABI stack.
- Decision: keep PyXtal, CHGNet, and MACE in `diff_meets_diff`; install only
  MatterSim's reviewed force-field inference closure in an immutable
  evaluator-scoped `PYTHONPATH` target. Torch, NumPy, pymatgen, spglib, Triton,
  and CUDA packages continue to come from the protected core environment.
- Boundary: omitted MatterSim workflow, Azure, phonon, API, logging, and
  training-only distributions are forbidden in the isolated target. PyG is an
  evaluator-only dependency and is not used by the WQ model.
- Evidence: the official MatterSim wheel SHA256, exact resolver report,
  dependency waiver, every target-tree file hash, retained Torch build, and
  source-bundle SHA256 are frozen. CHGNet, MatterSim, and MACE are each loaded
  in a separate process and Slurm runtime scope.
- Promotion: CPU and offline CUDA smoke must pass independently for all three
  evaluators. A scope failure removes that evaluator result; it cannot be fixed
  by mixing the incompatible `e3nn` versions in one process.

## D22 — Exact Pure-Python Source Builds

- Observation: the registered CHGNet/MACE dependency closure includes
  `nvidia-ml-py3==7.352.0` and `python-hostlist==2.3.0`, for which PyPI does not
  publish compatible wheels.
- Decision: download only those two exact official PyPI sdists on the login
  node, verify their frozen SHA256 values, and build no-dependency pure-Python
  wheels without build isolation. The exact Python, pip, setuptools, and wheel
  versions are recorded. All other distributions remain wheel-only.
- Boundary: MACE and CHGNet themselves are installed with `--no-deps` only
  after their complete reviewed direct requirements have been separately
  resolved. Installed MACE metadata must exactly match version 0.3.13, every
  direct requirement must be satisfied, and `pip check` may still report only
  the registered CHGNet/Torch metadata line.
- Evidence: source URLs, sdist hashes, built-wheel hashes, wheel metadata, and
  the complete active wheelhouse file set are bound into
  `wheelhouse_lock_v4.json` and the MLIP asset lock.

## D23 — Versioned Active Evaluator Stack After Failed Locks

- Observation: the first source-wheel attempt created `wheelhouse_lock.json`,
  after which a retry rebuilt its two source-derived wheel files. The lock and
  current files therefore no longer agree; changing or deleting that lock
  would erase failure provenance.
- A second attempt created a valid `wheelhouse_lock_v2.json` but failed before
  runtime promotion: MatterSim 1.1.2 imported APIs absent from core
  setuptools 82 and ASE 3.28. That v2 lock, wheelhouse, failed runtime staging,
  and diagnostic copies are also immutable failure evidence.
- Decision: preserve both predecessor locks and the later failed v3 run. The
  only active evaluator stack is `wqcodiff-evaluator-stack-v4`, stored in
  `wheelhouse_v4/`, `source_sdists_v4/`, and `wheelhouse_lock_v4.json`.
- Reproducibility: v4 source wheels are built once with build isolation off,
  pip's wheel cache disabled, `SOURCE_DATE_EPOCH=315532800`, and
  `PYTHONHASHSEED=0`. A present v4 lock causes verification instead of
  rebuilding. Its failed-predecessors section binds both old lock hashes and
  records any wheel mismatches.
- Boundary: runtime and MLIP asset locks may bind only the v4 lock. Neither
  predecessor is repaired, promoted, or used for offline installation.

## D24 — MatterSim 1.1.2 Packaging/API Compatibility Pins

- Observation: MatterSim 1.1.2 imports `pkg_resources` at package import and
  imports `full_3x3_to_voigt_6_stress` from `ase.constraints`. Setuptools 82
  removed `pkg_resources`, while ASE 3.28 stopped exporting the helper from
  that path despite MatterSim's open-ended metadata constraints.
- Evidence: a preserved diagnostic first reproduced each failure, then loaded
  every registered MatterSim inference import with `setuptools==81.0.0` and
  `ase==3.27.0`. Their official PyPI wheel SHA256 values are frozen in the
  protocol, wheelhouse lock, waiver, and runtime lock.
- Decision: install those two compatibility wheels only in MatterSim's v4
  isolated target. Core ASE 3.28 and setuptools 82 remain unchanged for
  CHGNet/MACE and source-wheel construction.
- Boundary: this is an evaluator packaging compatibility pin, not a model or
  scientific-method change. Any further import/API mismatch is a new hard
  failure; no source monkey patching is permitted.

## D25 — Top-Level Wheel Metadata Identity

- Observation: the v3 apply stopped before producing a lock because the
  scanner counted vendored `*.dist-info/METADATA` files inside the official
  setuptools wheel as candidate identities for the wheel itself.
- Decision: preserve the complete failed v3 run and partial wheelhouse. The v4
  scanner accepts exactly one root-level `<distribution>.dist-info/METADATA`
  record and ignores nested vendored records, matching the wheel layout
  contract without altering any distribution.
- Evidence: a synthetic regression wheel contains both a root setuptools
  record and a nested vendored record; the test must select only setuptools.
  The v4 official-wheel SHA checks remain unchanged.

## D26 — CPU Slurm Commands Must Preserve the Activated Environment

- Observation: the first P1/P2 CPU smoke passed its Python-3.10 environment
  doctor, then the launcher invoked the payload through `bash -lc`. Cluster
  login startup files reset that child to base Python 3.9, so both jobs failed
  before running any codec or formal transition.
- Decision: the CPU launcher uses a non-login `bash -c` child after activating
  `diff_meets_diff`. It retains `/usr/bin/time -v` accounting while inheriting
  the already-audited PATH and conda state.
- Evidence: the original jobs 26079/26080 remain immutable infra-failure
  records. A source regression test requires activation to precede execution
  and forbids reintroducing `bash -lc` at the payload boundary.

## D27 — FP32 Periodic Aggregation at the BF16 Geometry Boundary

- Observation: the first two-update CUDA training-pipeline smoke (job 26101)
  reached the model but failed because the FP32 periodic cutoff promoted BF16
  edge messages to FP32 while the in-place `index_add_` destination remained
  BF16. This is a mixed-precision implementation failure, not an OOM or a
  scientific-gate result.
- Decision: allocate the message aggregation buffer from the post-cutoff
  message tensor, accumulate in that promoted dtype, normalize there, and cast
  once to the node-feature dtype before the residual node MLP. This retains
  FP32 accumulation at the geometry boundary without changing architecture,
  losses, data, seeds, or the registered BF16 training contract.
- Evidence: job 26101 remains immutable failure evidence. A CPU BF16-autocast
  regression reproduces the exact feature/geometry dtype boundary; a new run
  ID is required for the CUDA retry.

## D28 — Primitive Expansion Follows Declared Centering, Not Decoration

- Observation: the first four-attempt unconditional sampling smoke completed
  its ledger but two P-lattice attempts (space groups 77 and 100) were marked
  projection failures. Spglib found an accidental half-volume translation in
  each one-species special-position decoration even though the committed
  space-group centering factor was one.
- Decision: a P conventional lattice maps to itself exactly. For centered
  groups, derive and cache the conventional-to-primitive lattice transform
  from a deterministic generic general-position orbit in a well-conditioned
  reference lattice of the same crystal system. Never infer the quotient's
  primitive convention from a partially generated decoration.
- Boundary: intended/redetected symmetry remains separately reported; an
  accidental translation may affect redetection but cannot change the
  registered Wyckoff primitive multiplicities or graph atom count.
- Evidence: job 26108 and all four terminal attempt records remain immutable.
  A regression reproduces the SG 77 special-orbit case, while the existing
  rhombohedral and face-centered exact-Jacobian tests guard centered groups.

## D29 — Every Training Checkpoint Is Bound to Source and Dataset Bytes

- Decision: Day-7 and later training requires the lowercase SHA256 of the
  installed frozen source bundle. Run manifests, optimizer checkpoints, and
  EMA checkpoints store that digest together with the absolute path, byte
  count, and streaming SHA256 of every input JSONL shard.
- Enforcement: resume and shared-boundary forks reject any source or dataset
  identity mismatch before loading state. A path-stable but byte-mutated
  dataset, or a checkpoint produced by a different source overlay, therefore
  cannot enter a matched comparison.

## D30 — Day-7 Cells Run in Four Immutable Modulo Lanes

- Decision: the 297 materialized recovery cells are partitioned by their
  phase-local ordinal modulo four. A lane resolves only the frozen dataset,
  checkpoint, and post-calibration threshold placeholders, invokes each cell
  without a shell, and records start/terminal events plus artifact hashes.
- Enforcement: a lane refuses to start if any selected cell already has an
  output or attempt ledger. A nonzero cell stops the lane; completed cells are
  never silently retried or replaced, and the four-lane plan is the concurrency
  ceiling rather than a best-effort scheduler setting.

## D31 — Workflow Scripts Use Explicit Package Boundaries

- Observation: the first remote v17 suite reached 90 tests but the new lane
  test could not import `scripts.a800`; a preserved historical
  `scripts/__init__.py` made the remote directory a regular package while the
  local tree had namespace-package semantics.
- Decision: both `scripts` and `scripts.a800` now have explicit package markers
  in the active source manifest. This removes host-dependent import behavior
  without changing the Day-7 registry, model, data, or experiment protocol.

## D32 — Day-7 Event Enumeration Cannot Synchronize Per Candidate

- Observation: the first 64-structure/16-call recovery timing run used only
  618 MiB peak GPU memory and about 67 W mean power. A registered `cProfile`
  run attributed 69.5 of 90.4 reverse-loop seconds to event scoring, including
  roughly 1.15 million individual CUDA-to-CPU scalar transfers; all 16 model
  forwards together accounted for only 2.23 seconds.
- Decision: compute every registered log-softmax on GPU exactly as before,
  transfer each compact head once per structure/step, and enumerate the same
  ordered legal support on CPU. Cache immutable PyXtal position, DoF, type,
  and unit-affine metadata inside each catalog instance. These changes do not
  alter event support, event ordering, logits, RNG consumption, backbone-call
  accounting, or the bridge.
- Validation: require per-event logit equality against the former
  per-candidate implementation, paired artifact equivalence on the same
  checkpoint/corruption seeds, and a fresh 64-structure Slurm timing run before
  any full Day-7 lane is submitted.
- Thread boundary: `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` remain hard-gated at exactly 1.
  Slurm CPU allocation is resource capacity and must never be interpreted as
  permission to increase a BLAS thread count above the user-approved 1--2.

## D33 — Every Slurm Job Uses an Isolated Bytecode Namespace

- Observation: the first v21 source test read a stale v20 test bytecode file.
  Deterministic patch archives normalize source mtime to the Unix epoch, and
  the corrected test retained the same byte count, satisfying CPython's
  timestamp/size cache key even though its SHA256 changed. Running the same
  source with a fresh `PYTHONPYCACHEPREFIX` passed all four sampling tests.
- Decision: every CPU and GPU launcher sets `PYTHONDONTWRITEBYTECODE=1` and an
  absolute, job-specific `PYTHONPYCACHEPREFIX` below the immutable run
  directory. The environment doctor hard-fails if either boundary is absent.
  This isolates imports from historical `__pycache__` files without deleting
  any remote evidence or changing source contents, model math, or protocol.

## D34 — Event-Enumeration Optimization Passes the Paired Numeric Gate

- Evidence: the v19 and v22 64-structure cells use the same checkpoint,
  hash-fixed subset, corruption seed, pairing ID, 16 backbone calls, and
  batch size. Core elapsed time fell from 81.6068 to 36.3963 seconds
  (`2.242x`). All 64 pair IDs/seeds, discrete traces, topology hashes,
  mechanism counts, StructureMatcher results, and aggregate recovery metrics
  are exactly equal.
- Numeric boundary: three `tangent_coordinate_error` diagnostics differed
  across the separate CUDA jobs by at most `1.391e-7`; every other semantic
  field was equal. This is below the already registered `1e-6` chart/projector
  numerical tolerance and cannot affect a categorical decision or matcher
  result. Both raw reports are retained; equality is not claimed bitwise for
  continuous CUDA diagnostics.
- Budget consequence: scaling the measured 64-attempt cell gives about
  7.3 GPUh for threshold calibration, 116.5 GPUh for the primary matrix, and
  11.7 GPUh for interventions. The recovery total is therefore about
  135.5 GPUh before training and remains admissible under the 180 GPUh Week-1
  cap; no BLAS thread count or backbone-call budget was increased.

## D35 — Shared-Fork EMA State Must Follow the Live Model Device

- Observation: the first v23 Day-7 method-specific fork wave (Slurm jobs
  26261, 26262, 26264, and 26265) failed on the first EMA update. Checkpoints
  are intentionally loaded with `map_location="cpu"`, but the loaded EMA
  shadow tensors were not moved back to the CUDA device. The four independent
  variants produced the same `cuda:0` versus `cpu` error before completing an
  update; their immutable logs and GPU-hour records are retained.
- Decision: `ExponentialMovingAverage.load_state_dict` now requires the
  checkpoint loader to supply the live model device and relocates every shadow
  tensor before training resumes. Model, optimizer, scheduler, sampler, RNG,
  EMA values, losses, data, and registered hyperparameters are unchanged.
- Validation: a CPU regression locks the explicit relocation contract, a CUDA
  regression exercises the original failure boundary inside Slurm, and the
  v24 shared stage is retrained rather than relabeling or mutating the v23
  checkpoint. Every retry uses a new immutable run ID. OpenBLAS, OMP, MKL, and
  NumExpr remain fixed at one thread.

## D36 — Relational Metrics Use One Globally Pooled Generation Artifact

- Observation: final sampling is intentionally split across three training
  seeds, but uniqueness computed independently inside each seed would miss
  cross-seed duplicates and could inflate SUN.
- Decision: immutable per-seed attempt JSONLs are pooled in deterministic
  `(training_seed, sampling_seed, ordinal, attempt_id)` order before any
  novelty or uniqueness evaluation. Pooling rejects mixed methods, duplicate
  attempt or pair IDs, nonterminal rows, and deviations from the registered
  `3334/3333/3333` training-seed counts.
- Evidence boundary: the pool manifest hashes every source, the exact attempt
  and pair ID sets, all checkpoints/source bundles/revision locks, and the
  pooled artifact. Per-seed survivor-only relational metrics are not eligible
  for a paper table.

## D37 — A Hull Relaxation Queue Is Evaluator-and-Contract Specific

- Decision: a pending or closed hull queue may drive relaxation only when its
  evaluator identity and full calculator/relaxation contract hash match the
  requested runtime. This check occurs before reading pending IDs.
- Consequence: a MatterSim queue cannot schedule MACE or CHGNet relaxations,
  and a queue created under a different checkpoint/package/relaxation contract
  cannot be reused even if its reference IDs happen to match.

## D38 — Final Tables Separate Novelty Components and Mechanism Counts

- Decision: the denominator-preserving metric table reports stability,
  standard uniqueness, full-structure novelty, anonymous-prototype novelty,
  species--Wyckoff protostructure novelty, substitution-aware novelty, their
  joint SUN variants, and failure rate separately.
- Mechanism accounting: generation metadata repeated across evaluator/stage
  rows is first deduplicated by `(method, attempt_id)` and required to agree.
  Birth, death, Wyckoff-type, species, dimension-change, revision-fill,
  revision-event, and churn summaries are then computed once per submitted
  attempt. Synthetic wrong-to-right/right-to-wrong precision and recall remain
  in the registered corruption-recovery report, where ground truth exists.

## D39 — Week-2 Training Is Materialized Only After the Day-7 Engine Freeze

- Decision: the Day-7 winner among AR/D3PM/DLM is an explicit input to a
  seven-job immutable Week-2 plan: two representation-matched shared stages to
  update 60,000 and five method-specific screens to update 85,000.
- Continuation boundary: every screen points to the full optimizer checkpoint
  at update 60,000. The 85k EMA is labeled validation-only; continuation to
  100k must use `checkpoint_0085000.pt`, never `model_ema_final.pt`.
- The plan binds all eight full-data shards, source/protocol/registry hashes,
  representation dependency, training seed 11, full 100k schedule, and the
  four-lane concurrency cap before any Week-2 submission.
- Each Slurm job resolves exactly one plan entry without shell interpolation,
  refuses an existing output/evidence path, hashes every dependency optimizer
  checkpoint, and requires the expected training-complete/checkpoint/EMA
  artifacts before writing its terminal completion record.

## D40 — Week-2 Sampling Is Matched Before Any MLIP Screening

- Decision: the five 85k routes expand to eight immutable sampling
  configurations because `B-WQ-DISC-ONCE` retains its registered
  `tau/T={0.25,0.5,0.75,1.0}` validation grid. Every configuration first runs
  one 256-attempt preflight and then the same three 1,000-attempt sampling
  seeds at 64 backbone calls.
- Pairing boundary: all 24 development cells use the single frozen
  `week2-matched-screen-v1` pairing namespace, so ordinal/seed initial noise
  is identical across routes and handoff values. Preflight uses a separate
  namespace and is never pooled into a development result.
- Provenance: the sampling plan hashes all five validation EMA checkpoints,
  the Day-7 threshold lock, the Week-2 training plan, and the active
  source/protocol/registry identities. Per-attempt and evaluator rows carry
  `experiment_id`, pairing namespace, revision threshold, temperature, and
  the explicit DISC-ONCE handoff value; different handoff values cannot be
  silently merged under the common method label.
- Execution: four modulo-partitioned lanes refuse changed checkpoints,
  changed threshold locks, existing cell artifacts, and incomplete terminal
  evidence. Invalid generation proposals remain terminal attempts and are
  never retried or replaced.

## D41 — Day-14 Champion Selection Is Frozen Before Week-2 Results

- Selection data: exactly 3,000 matched validation attempts per configuration
  under raw MatterSim `MLIP-SUN@0.1`. CHGNet, MACE, common-refiner, relaxed,
  test-split, and survivor-only values cannot select the comparator.
- Two-stage rule: first select one DISC-ONCE handoff from the four registered
  tau values; then select the comparator among the Day-7 discrete engine,
  `B-WQ-JOINT-NOREV`, and that single DISC-ONCE configuration. Rank by
  `MLIP-SUN@0.1`, Novel&Unique, `MLIP-SUN@0.0`, lower failure rate, lower
  actual generation calls, then lexicographically lower configuration ID.
- Eligibility boundary: `B-ATOM-JOINT` remains a representation baseline and
  `M-WQ-STRAT-GEO` remains the proposed route; neither may become the matched
  WQ champion even if its validation number is larger.
- The Day-14 lock verifies an identical pair set and evaluator/hull/novelty/
  matcher contract across all eight configurations, hashes every input, and
  binds both the selected champion and stratified route to their full 85k
  optimizer checkpoints. Validation EMA files can select but can never resume
  training.

## D42 — Day-7 Recovery Does Not Run the Non-Gating Structure Matcher

- Evidence: the preserved v24 threshold-calibration pilot completed the model
  recovery for a batch, then spent more than 30 minutes of single-core CPU
  time inside `StructureMatcher.fit` for one recovered structure. The
  append-only artifact stopped at ordinal 155 while the process remained
  runnable at essentially 100% CPU. Repeating this auxiliary diagnostic would
  convert hundreds of unrelated, already registered attempts into Slurm
  job-level timeouts.
- Decision: Day-7 recovery records `structure_match=null` and
  `structure_match_status=deferred_not_registered_day7_gate`. This field is
  absent from every preregistered DLM promotion condition; exact
  protostructure recovery, edit distance, tangent error, interventions, and
  attempt failures remain unchanged.
- Boundary: the v24 pilot remains immutable infrastructure-failure evidence
  and cannot enter a scientific table. A new source/run identity is required;
  attempt IDs from v24 are never reused. Standard, strict, and lenient
  StructureMatcher panels remain mandatory in the bounded final evaluation
  workflow and are not replaced by this Day-7 decision.

## D43 — Final Structure Matching Is Bounded and Fails Conservatively

- Contract: every `StructureMatcher.fit`, anonymous-fit, or anonymous-mapping
  call in the final relational metric panel has a 5-second POSIX wall-time
  limit. The timeout and its conservative policy are included in the matcher
  contract hash, so an older novelty-reference lock cannot be mixed with this
  evaluator.
- Denominator policy: duplicate-comparison timeout/error is scored as a
  duplicate; full-structure or anonymous-prototype timeout/error is scored as
  non-novel; substitution-mapping timeout/error is scored as
  substitution-derived. Generated attempts therefore remain in the metric
  denominator and no timeout can improve SUN.
- Reporting: per-attempt matcher timeout/error counters are written into the
  evaluator artifact. The standard, strict, and lenient matcher values are
  still computed independently under their frozen tolerances; the guard only
  prevents a pathological comparison from blocking an evaluator lane.

## D44 — GPU Budget Audits Span Every Frozen Runtime Root

- Observation: preserving an in-flight failed source snapshot while validating
  a clean replacement requires side-by-side server roots. Auditing only the
  current root's `runs/` directory would omit GPU hours already consumed by
  the earlier root.
- Decision: the GPU-budget tool accepts one or more run roots, rejects a
  duplicate root or duplicate Slurm job ID across them, and sums all validated
  `wqcodiff_slurm_usage_v1` records before applying weekly, total, and Week-4
  reserve gates. The GPU launcher accepts an explicit colon-separated root
  list and materializes that combined audit before submission.
- Boundary: a new source/runtime directory never resets the 2,050-GPUh paper
  budget. Failed pilots, smoke jobs, and paper runs count exactly once using
  their actual terminal Slurm usage artifacts.

## D45 — Recovery CLI Must Carry Runtime Source Provenance

- Evidence: the first clean-runtime regression reached the registered
  recovery CLI but exited before creating an attempt because the parser
  accepted `--runtime-source-bundle-sha256` while the CLI-to-config adapter
  omitted it. Slurm job 26323 is retained as an infrastructure failure and is
  not a scientific attempt.
- Decision: the CLI adapter materializes all recovery fields in one tested
  mapping, including the runtime source-bundle SHA. A boundary regression test
  fails if that provenance field is dropped again.
- Boundary: job 26323 is never resumed or relabeled. Validation of the fix uses
  a fresh source bundle, run ID, experiment ID, and attempt namespace.

## D46 — MatterSim S.U.N. Reuses the Frozen R5-C Executor

- Decision: MatterSim S.U.N. is executed by the exact R5-C implementation
  `scripts/run_mattergen_sun_eval.py`, hash
  `510bcf297247dfab7a77ff7aa564072806f49b0c212fe670d3221d1788ef305b`.
  The frozen runtime arguments remain CUDA, 500 relaxation steps, 0.05 eV/Å
  force tolerance, 512 atoms per batch, and the disordered matcher.
- Paper update: the primary held-out call supplies the registered MatterSim
  5M checkpoint and evaluator-specific frozen MP20 reference LMDB. Generated
  structures preserve input order and are joined back to immutable attempt
  IDs; unsupported and nonconverged structures remain in the submitted
  denominator.
- Comparability boundary: an optional R5-C compatibility panel may use the old
  1M checkpoint and `reference_MP2020correction.gz`, but it is labeled a
  diagnostic and cannot replace the registered 5M headline.

## D47 — Numeric Thread Limits Are Enforced Before Optional Imports

- Evidence: a login-node protocol check without a shell prefix inherited an
  unsafe OpenBLAS thread count and attempted to create 64 threads before NumPy
  finished importing.
- Decision: importing `crystal_dlm.wqcodiff` now sets OpenBLAS, OMP, MKL, and
  NumExpr thread counts to exactly one before any optional numerical package
  can load. The protocol validator and a package-reload regression test both
  reject an increased count.
- Boundary: Slurm launchers retain their independent hard gates at one thread;
  this import-time guard is defense in depth for read-only login-node commands,
  not authorization to run CUDA work outside Slurm.

## D48 — Coordinate Scores Use Sigma-Squared DSM and v24 Pilots Are Quarantined

- Evidence: an independent audit of the completed v24 10k pilots found that
  geometry contributed 99.64–99.68% of the absolute component loss and every
  logged update exceeded the 1.0 gradient-clip threshold. The wrapped score
  target scales as `1/sigma`, but its squared error was averaged uniformly
  over timesteps. A separate Slurm audit (job 26340) strictly loaded all five
  EMA state dicts and found zero non-finite values, while also confirming that
  their frozen protocol hash `f4e270...` differs from the active protocol.
- Decision: atom-coordinate score matching is weighted by `sigma(t)^2` before
  the valid-atom mean. This is the standard denoising score-matching
  normalization; it preserves the score parameterization used by reverse-time
  sampling while preventing low-noise targets from dominating the shared
  encoder. The weight is protocol-locked and receives an analytic
  scale-invariance regression test.
- Boundary: the v24 checkpoints remain immutable diagnostic evidence. They are
  neither relabeled nor admitted through a protocol allowlist. Day-7 is
  restarted with fresh current-protocol/source checkpoints and fresh attempt
  namespaces. Future logs explicitly record pre-clip norm, clipping status,
  and geometry's fraction of the absolute component sum.

## D49 — Periodic Bridge Likelihood and Component-Gradient Gate

- Evidence: the matched v34 100-update CUDA smoke (Slurm job 26344) reduced
  geometry from 99.34% to 4.07% on the identical first batch and to 6.58% on
  average, but all 100 updates still crossed the global clip threshold. The
  largest event occurred at update 23: bridge NLL 137.80 and pre-clip norm
  13,917.54. Across the smoke, bridge loss and gradient norm had correlation
  0.996. An independent log audit confirmed that the sigma-squared DSM repair
  is isolated and valid, while recommending against a blind 6k continuation.
- Semantic defect: first-orbit, birth, and target-stratum bridge coordinates
  are sampled from a Gaussian modulo one, but their training objectives used
  an ordinary Euclidean Gaussian NLL. That objective assigns different loss
  to equivalent torus points and permits learned variance collapse to turn a
  rare boundary example into an unbounded shared-encoder gradient.
- Decision: all three periodic-coordinate heads use the wrapped-normal
  log-likelihood computed by log-sum-exp over integer images `[-8,8]`.
  Periodic scale heads use a smooth sigmoid map to the frozen interval
  `[0.02,0.5]`; lattice charts retain their nonperiodic Gaussian likelihood.
  Training logs record supervised scale counts/min/mean/max and the actual
  global clip multiplier.
- Gate: before any replacement 6k run, a diagnostic-only Slurm job evaluates
  16 hash-fixed batches and records every atomic loss term's global, shared
  backbone, and task-specific gradient norm. The audit updates no parameters,
  is not paper eligible, and must be finite. The v34 smoke remains immutable
  diagnostic evidence; a fresh source hash, run ID, and checkpoint namespace
  are required after this objective change.

## D50 — Atomic Loss Aggregation Uses One Canonical Float32 Fold

- Evidence: the first remote v35 validation (Slurm job 26345) passed 136 of
  137 executed tests, with three expected skips. The sole failure compared the
  registered total against the sum of its 17 atomic diagnostic terms. Both
  expressions contained exactly the same normalized losses, but different
  float32 parenthesization exceeded a tolerance by a few ULPs for that seeded
  initialization.
- Decision: the training total is the canonical left-to-right sum of the
  `WQLossTerms` tuple, exactly matching the component-gradient diagnostic.
  Grouped values such as event payload, geometry, and prior remain reporting
  views only. This changes neither term definitions nor their equal weights.
- Boundary: job 26345 remains an immutable validation failure. The correction
  receives a new source hash and a fresh validation run before any CUDA
  gradient audit or replacement training is submitted.

## D51 — Append-Only Ledgers Validate New Bytes Incrementally

- Evidence: the Day-7 ledger implementations re-parsed the complete attempt
  and artifact JSONL file before every append. A 4,096-attempt recovery cell
  therefore performed quadratic JSON parsing even though every cooperative
  writer already held the same exclusive file lock. The 32-attempt regression
  could not expose this scaling failure, and its measured inference time left
  insufficient margin in the registered 36-hour primary lanes.
- Decision: each ledger instance keeps the inode, validated byte offset, line
  count, and immutable-key/status index. While holding the existing exclusive
  lock, it parses only complete newline-terminated bytes appended since that
  offset; inode replacement or truncation resets the cache and revalidates the
  whole file. Every individual record remains canonical JSON, append-only,
  flushed, and fsynced, and artifact SHA256 values are byte-identical to the
  previous implementation.
- Gate: cross-writer duplicate tests, partial-tail rejection, canonical-byte
  parity, and a parse-count regression must pass under a fresh source hash.
  Before the 737,280-attempt primary array is launched, each discrete engine
  must also complete an independent 256-attempt, four-full-batch CUDA timing
  cell using the same frozen checkpoint/data/call contract. The primary
  walltime and GPU-hour reservation are frozen from that measured slope, not
  from the 32-attempt regression.
- Boundary: completed v36 calibration jobs and their ledger bytes are never
  modified or relabeled. The optimization changes no model, corruption,
  sampling, pairing, attempt identity, terminal denominator, or scientific
  metric; primary recovery uses a new runtime/source hash and records the v36
  checkpoint source hash separately.

## D52 — v39 Runtime Acceleration Is a Semantics-Gated Candidate

- Evidence: the frozen v38e Day-7 run showed approximately one GPU-percent
  utilization while each lane consumed only one of its twelve allocated CPU
  cores. The longest registered cell was D3PM at 90% deletion: its completed
  seed-303 lane took 7.01 hours versus roughly 0.39 hours at 70% deletion.
  Per-batch timing had a 13.75x max/min ratio, and the four slowest batches
  accounted for 49.6% of the cell. Read-only code/profile inspection localized
  this long tail to serial PyXtal expansion, spglib redetection, Python pair
  loops, over-built legal-event supports, and per-orbit D3PM tensor setup.
- Candidate: v39 removes one unused conventional expansion, constructs only
  the requested corruption-event support, vectorizes the exact 20-atom
  periodic pair calculation, avoids the unrestricted D3PM candidate tensor,
  hoists invariant Wyckoff metadata, and offers ordered CPU preparation using
  up to the twelve cores already allocated to a lane. Model inference,
  per-attempt reverse updates, event sampling, and RNG consumption remain in
  registered order. BLAS/OpenMP/MKL/NumExpr remain fixed at one thread.
- Local gate: registered-event subsequence tests and geometry reference tests
  pass. Two five-round local primitive benchmarks measured 351--360x for the
  deletion-support microkernel and 6.33--6.66x for geometry signals, with
  exact event equality and zero floating-point difference on each benchmark input.
  These microbenchmarks are not reported as end-to-end speedups.
- Server gate: v39 is disabled for paper evidence until the locked server
  environment passes Torch/PyXtal/spglib tests and a hash-fixed 256-attempt
  v38-versus-v39 comparison. Terminal status/reason, topology hashes, ordered
  traces, calls, mechanism counters, exact recovery, and edit distance must be
  equal per attempt; tangent-coordinate error tolerance is 1e-7. A separate
  same-node three-repeat timing gate requires at least 2x on ordinary cells,
  3x on 90% deletion, D3PM-0.9 at or below 2.3 hours, and AR at or below 1.4
  hours before the optimized runtime becomes the default execution copy.
- Boundary: Slurm job 26401 and all v38 attempts remain immutable. They are
  neither cancelled nor supplemented by v39. The v39 bundle is built locally
  and may be deployed only after the complete frozen v38f audit; any
  equivalence failure is a no-go, not a reason to replace or retry attempts.

## D53 — Parent Scheduler Length Is Distinct from the Refinement Horizon

- Evidence: the first CrysLLMGen CSP Gate-A mapping job (26475) was preserved
  as a failure after strict loading found six 1001-entry schedule buffers in
  the released `model_494.pt`, while the local port had instantiated 801-entry
  buffers. The vendored upstream training and sampling entry points both
  default to a 1000-step scheduler; the historical R5-C invocation separately
  starts refinement at `t=800`.
- Decision: the parent checkpoint is always constructed with a 1000-step
  scheduler. Official sampling still executes the registered `800 -> 0`
  horizon (1600 decoder calls), and matched 32-step sampling still uses the
  registered grid from 800 to 1 (64 decoder calls). WQ time conditioning also
  remains on that 800-step refinement horizon. Protocol and parity records now
  encode both quantities explicitly.
- Boundary: this correction occurred before any Gate-A pass, model training,
  or scientific sampling attempt. Job 26475 is not relabeled or overwritten;
  the corrected mapping/parity run uses a fresh source hash and output path.

## D54 — Exact Upstream Run Type and Lossless Fixed-Width Edit Evidence

- Evidence: after the scheduler correction, preserved Gate-A job 26482 reached
  the decoder and exposed that `run_type=sample` forces a float64 lattice head
  while the validation batch was float32. The vendored CrysLLMGen CLI defaults
  to `run_type=train`, and the active atom/WQ loaders already use that mode.
  Separately, the first complete mixed-edit materialization retained all 54,270
  examples but found 47 above 512 tokens; their true maximum was 545.
- Decision: disabled-extension parity now binds the exact upstream
  `run_type=train`. Mixed-edit evidence retains all six 4-bit geometry signals
  per orbit but omits redundant numeric orbit labels: fixed-width six-nibble
  rows are concatenated in the same presentation order as the proposal. The
  fixed prompt is shortened accordingly. A full-tokenizer dry audit measured
  a maximum of 472 and zero examples above 512 without deleting or truncating
  any sample or signal.
- Boundary: jobs 26480 and 26482 remain immutable failures. The two v6 smoke
  jobs were cancelled once their source became ineligible for a common-source
  Gate-A lock; no scientific attempt was sampled. All corrected checks and
  smokes use a new source hash and fresh output namespaces.

## D55 — Transparent One-Step Parity Avoids CUDA Atomic-Scatter Noise

- Evidence: v7 mapped the complete checkpoint and passed the paired four-step
  A800 sampler at maximum absolute error `4.77e-7`, but two consecutive calls
  to the same decoder differed by up to `7.63e-6`. The wrapper contains only a
  direct decoder call; the excess arose from CUDA atomic reductions in
  `torch_scatter`, not from extension logic.
- Decision: the one-step transparent-wrapper comparison runs the exact parent
  checkpoint, decoder, inputs, and batch of eight on deterministic CPU and
  keeps the frozen `1e-6` threshold. Strict checkpoint mapping and the paired
  sampler remain on A800. The report records the one-step device and that CUDA
  atomic scatter was excluded from this identity check.
- Boundary: tolerance is not relaxed and v7 job 26491 remains a failed audit.
  The final Gate-A run and both smokes use a fresh common source hash; no model
  training or scientific sampling is admitted before that audit passes.
