# C3FD–Llama–DLM S.U.N. execution checklist v6

Date: 2026-08-31

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
- [ ] Run F/M × streams17/18 through the same old H1-A2 rich DLM and model494
  tau800, one Plan and one trajectory per request.
- [ ] Run one fixed-denominator raw-first/refined offline evaluation.
- [ ] Execute one prospective official MP query and report Strict/Meta stable,
  S.U.N., Direct/N/U/NU, CHGNet, hull ECDF/quantiles, and uncertainty.

## Phase 5 — paper closure

- [ ] Classify C3FD proposal correctness, F/M conditioning, and DLM realization
  as SUPPORTED/CANDIDATE/UNSUPPORTED.
- [ ] Keep raw DLM evidence primary and refined-only evidence secondary.
- [ ] Update BUILD_STATUS, PAPER_STORY, negative archive, tests, commits, and
  final resource ledger.

## Phase 6 — conditional post-F/M DLM stabilization

The planning contract is
`DLM_POST_FM_REFINER_DISTILLATION_GEOMETRY_PLAN_V1.md`. This phase is not
authorized to start until the fixed F/M prospective result is complete and
disclosed.

- [x] Audit whether model494-relaxed geometry was previously distilled into the
  DLM: it was proposed but never executed. SGTC and D3PO are different methods.
- [x] Audit current decoder constraints: schema/exact chemistry, nondegenerate
  lattice, and exact/PBC duplicate rejection already exist.
- [ ] Apply the frozen M-mainline noninferiority rule. Keep F as the disclosed
  ablation; never delete either route's result.
- [ ] If M passes, add one species-aware PBC gross-overlap token mask and run a
  train/validation-only decoder A/B with fixed noise.
- [ ] Build one immutable MP20-train-only, one-trajectory-per-row model494
  relaxed-target dataset under M; no energy filtering or winner selection.
- [ ] Train two fresh DLM LoRA seeds from the shared pretrained crystal base,
  mixing original MP20 and single-refiner-target geometry CE.
- [ ] Run the frozen 2x2 weight × decoder screen, raw first. Only after that
  screen may a new prospective cohort and official query be frozen.
- [ ] Treat refined-only gains as system effects; require raw Direct/energy
  evidence for a DLM stability contribution.

## Credential and execution boundary

Use the user-provided MP credential only in a temporary nonambient child
environment for each explicitly authorized query. Never write or echo it to
Git, docs, checklist, automation, commands, logs, hashes, manifests, or
archives. Unset it immediately and verify no process/runtime copy remains.

No retry/replacement/rerank/best-of-N, outcome-based setting selection, AR body
executor, RL/GRPO/SMC, C3FD retraining, compact-V2 rerun, or alignment is
authorized.
