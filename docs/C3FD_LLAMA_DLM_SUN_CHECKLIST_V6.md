# C3FD–Llama–DLM S.U.N. execution checklist v6

Date: 2026-08-31

Execution window: through 2026-09-02 23:30 Asia/Shanghai

Status: **active after explicit user approval**

This checklist supersedes v4 and draft v5. It authorizes the two routes in
`C3FD_LLAMA_DLM_TWO_ROUTE_METHOD_V1.md` under the existing six-A800/two-job
ceiling and immutable-run rules.

## Phase 0 — stop stale work and freeze method

- [x] Cancel deterministic-completion job38914; never resume/evaluate partials.
- [x] Stop alignment/listwise training before any alignment weights exist.
- [x] Retire compact V2 reruns and deterministic scientific-field completion.
- [x] Freeze Route F, Route M, and the DLM-central paper interpretation.

## Phase 1 — calculate already available development S.U.N.

- [x] Collect faithful H0/R0S eval38603 raw/refined official inputs.
- [x] Collect compact-V2 fresh-SFT canary eval38768 raw/refined official inputs.
- [x] Exclude train-only alignment pool38881, malformed canary38420,
  cancelled38914, and already-official D3PO.
- [x] Freeze one development official-input manifest and SOURCE_SHA.
- [x] Execute one credential-isolated MP query, unset the key immediately, and
  finalize separate faithful and compact-V2 development S.U.N. reports.

This development query is explicitly requested before new training. A later
prospective query covers only the new F/M cohort; neither query may be repeated
after success.

## Phase 2 — minimal shared F/M implementation

- [x] Add one composition-prefill Rich-suffix prompt/target helper.
- [x] Add one fixed-size frozen C3FD semantic-state feature packer for M.
- [x] Add one two-layer projector that maps the feature vector to `K` Llama
  soft-prefix embeddings and prepends them through `inputs_embeds`.
- [x] Extend the existing MP20 H1 Planner data builder with `formula_only` and
  `c3fd_soft_prefix` conditioning modes.
- [x] Reuse the existing Llama+LoRA trainer and loss; add no new backbone or
  custom attention path. Only M's prefix projector is new.
- [x] Extend sampling to lock the C3FD formula and generate only the rich
  suffix, then validate the canonical seven-line H1-A2 Plan.
- [x] Add train/serve round-trip, prefix-shape/mask, formula immutability,
  no-fill, and F/M visible-answer-identity tests.

## Phase 3 — train and interface canary

- [x] Build immutable MP20 train/validation F/M data with identical source rows
  and rich targets.
- [x] Train two fixed seeds per route for one adaptation epoch from the same
  historical H1-A2 rich Planner initialization; final checkpoint only.
- [x] Report all four terminal checkpoints without seed selection.
- [x] Run one fixed small syntax/interface canary; it cannot change settings.

## Phase 4 — prospective DLM realization

- [x] Freeze one new C3FD all-attempt fixed256 composition ledger before DLM
  outcomes; retain failed requests and do not top up.
- [x] Generate F/M Plans on identical compositions/order.
- [x] Run F/M × streams17/18 through the same old H1-A2 rich DLM and model494
  tau800, one Plan and one trajectory per request.
- [x] Run one fixed-denominator raw-first/refined offline evaluation.
- [x] Execute one prospective official MP query and report Strict/Meta stable,
  S.U.N., Direct/N/U/NU, CHGNet, hull ECDF/quantiles, and uncertainty.

Terminal fixed-256 result (two streams, no attempt replacement):

- requested-denominator composition validity: F `500/512 = 97.66%`, M
  `503/512 = 98.24%`; conditional on a parsed body, both are `100%`;
- raw Direct structural validity: F `217/512 = 42.38%`, M
  `229/512 = 44.73%`;
- refined Direct structural validity: F `499/512 = 97.46%`, M
  `502/512 = 98.05%`;
- raw Strict/Meta S.U.N.: F `2.734% / 10.547%`, M
  `3.125% / 10.156%`;
- refined Strict/Meta S.U.N.: F `6.641% / 38.281%`, M
  `7.617% / 37.695%`.

The paired M-minus-F raw and refined CHGNet confidence intervals cross zero;
M is therefore retained for its integrated C3FD conditioning, slightly higher
composition/body validity, and higher raw Direct validity, not for a claimed
energy advantage. F remains the disclosed formula-only ablation. The official
query was executed once for 254 chemsystems (`246` resolved, `8` unresolved),
and must not be repeated.

## Phase 5 — paper closure

- [x] Classify C3FD proposal correctness, F/M conditioning, and DLM realization
  as SUPPORTED/CANDIDATE/UNSUPPORTED.
- [ ] Keep raw DLM evidence primary and refined-only evidence secondary.
- [ ] Update BUILD_STATUS, PAPER_STORY, negative archive, tests, commits, and
  final resource ledger.

## Phase 6 — conditional post-F/M DLM stabilization

The active planning contract is
`DLM_POST_FM_STRUCTURAL_LEARNING_AND_REFINER_FEEDBACK_PLAN_V2.md`. The user has
authorized it to begin after the fixed F/M prospective result is complete and
disclosed. The earlier decode-mask-first v1 plan is superseded.

The user subsequently selected the Compact-V2 return described in
`ADR_C3FD_LLAMA_COMPACT_V2_MAINLINE_20260831.md`. The completed F/M experiment
remains immutable evidence; M is the winner within that comparison, but the
active candidate mainline is now C3FD-soft-prefix Llama producing the canonical
Compact-V2 Plan for the already-trained Compact-V2 DLM.

- [ ] Add the minimal three-soft-field Compact-V2 Llama target/sampler by
  reusing Route M's feature packer, projector, and LoRA trainer.
- [ ] Train one fixed Planner seed from MP20-train teacher Compact-V2 targets;
  Llama may not change C3FD `N/elements/counts/anion_framework`.
- [ ] Reuse job38703 Compact-V2 DLM seed82017 and one fixed stream17. Do not
  retrain the same two-epoch DLM merely because Llama was inserted upstream.
- [ ] Run a matched direct-C3FD-V2 versus Llama-V2 body+Direct screen on one
  frozen ledger/noise stream. Promote Llama only if comp-valid is at least 95%,
  body is within -1 pp, and raw Direct is noninferior; otherwise retain direct
  Compact V2. No new MP query.
- [ ] Apply G1 and optional later G2 to the resulting Compact-V2 DLM mainline,
  not to the old-rich DLM.

- [x] Audit whether model494-relaxed geometry was previously distilled into the
  DLM: it was proposed but never executed. SGTC and D3PO are different methods.
- [x] Audit current decoder constraints: schema/exact chemistry, nondegenerate
  lattice, and exact/PBC duplicate rejection already exist.
- [x] Retain each F/M route whose final requested-denominator composition
  validity is at least 95%. S.U.N. is not a Planner-retention gate; all outcomes
  remain disclosed.
- [x] Freeze M as the winner within the completed F/M comparison and F as its
  formula-only ablation. The later Compact-V2 pivot supersedes M as the active
  mainline candidate without deleting the F/M result.
- [ ] Freeze one new outcome-blind Plan ledger once for final post-training
  comparison and share it across BASE/G1/G2/feedback and all streams.
- [ ] Reuse CHGNet results only for exact-identical raw structures with full
  per-attempt remapping; never merge near-equivalent or model494-refined rows.
- [ ] For G1/G2 development, run body+Direct first. Continue to model494/CHGNet
  only if comp-valid is at least95%, body is within-1 pp of control, and both
  methods improve raw struct-valid under the one frozen training seed/stream.
- [ ] If that gate fails, archive every raw/Direct row and stop downstream
  compute. Do not delete the arm; do not apply this shortcut after final
  prospective arms are frozen.
- [ ] Decompose frozen raw failures into parse, composition, lattice,
  PBC-distance `<0.5 A`, CrystalNN/graph, and valid-but-high-energy classes;
  report how much invalidity is actually geometric.
- [ ] Run a CPU-only quantization audit for the current 0.1-A/1-degree/0.01-frac
  `7+4N` tokens. Extend the vocabulary only if tokenization itself changes
  structural validity by more than 1%.
- [ ] Add differentiable periodic metric/RDF/short-distance/coordination losses
  to DLM training. Do not add a new inference-time geometry mask.
- [ ] Train one frozen-seed G1 and one frozen-seed G2 candidate. Reuse one base
  cell and one fixed development stream; run the candidate jobs in parallel
  when G2 is ready.
- [ ] Add a zero-initialized two-layer periodic residual species-pair/RBF
  relation adapter only if token round-trip passes and auxiliary geometry errors
  and its independent implementation gates pass. Require step-0 equality to G1
  and compare against a same-update no-adapter
  continuation. Reuse the CTV 4096-d output-head pre-hook/equality path; do not
  add epochs or search schedules to force the trigger.
- [ ] Before G2, unit-test the acyclic `q0 -> soft geometry -> h' -> q1` forward,
  circular fractional-coordinate means, entropy-gated messages, SPD metric
  projection, triclinic neighboring-image distances, and translation/
  same-species-permutation invariance.
- [ ] Zero only the residual output projection, log adapter gradient norms for
  steps 0--10, and profile gathered typed-logit O(N^2) memory/runtime; forbid
  dense pair-by-vocabulary tensors.
- [ ] Build MP20-train-only one-trajectory model494 basin-SFT targets, requiring
  exact composition, Direct structural validity, and finite refined energy.
- [ ] Build one fixed K=4 same-composition pool and run shared-mask,
  reference-corrected group-relative diffusion preference with raw validity
  lexicographically before refined energy.
- [ ] Query MP references only after a complete immutable batch exists, from a
  login-side process; never query from GPU training jobs or per update.
- [ ] If the complete DLM structural-learning route remains raw-negative,
  implement the matched C3FD-conditioned AR CrysLLMGen fallback. Do not launch
  it in parallel with the DLM route.
- [ ] Treat refined-only gains as system effects; require raw Direct/energy
  evidence for a DLM contribution.

## Credential and execution boundary

Use the user-provided MP credential only in a temporary nonambient child
environment for each explicitly authorized query. Never write or echo it to
Git, docs, checklist, automation, commands, logs, hashes, manifests, or
archives. Unset it immediately and verify no process/runtime copy remains.

No retry/replacement/rerank/best-of-N, outcome-based setting selection,
energy-only reward, vanilla dLLM GRPO, C3FD retraining, or compact-V2 rerun is
authorized. The AR body executor is authorized only as the final fallback after
the DLM route is terminal.
