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

- [ ] Collect faithful H0/R0S eval38603 raw/refined official inputs.
- [ ] Collect compact-V2 fresh-SFT canary eval38768 raw/refined official inputs.
- [ ] Exclude train-only alignment pool38881, malformed canary38420,
  cancelled38914, and already-official D3PO.
- [ ] Freeze one development official-input manifest and SOURCE_SHA.
- [ ] Execute one credential-isolated MP query, unset the key immediately, and
  finalize separate faithful and compact-V2 development S.U.N. reports.

This development query is explicitly requested before new training. A later
prospective query covers only the new F/M cohort; neither query may be repeated
after success.

## Phase 2 — minimal shared F/M implementation

- [ ] Add one composition-prefill Rich-suffix prompt/target helper.
- [ ] Add one fixed-size frozen C3FD semantic-state feature packer for M.
- [ ] Add one two-layer projector that maps the feature vector to `K` Llama
  soft-prefix embeddings and prepends them through `inputs_embeds`.
- [ ] Extend the existing MP20 H1 Planner data builder with `formula_only` and
  `c3fd_soft_prefix` conditioning modes.
- [ ] Reuse the existing Llama+LoRA trainer and loss; add no new backbone or
  custom attention path. Only M's prefix projector is new.
- [ ] Extend sampling to lock the C3FD formula and generate only the rich
  suffix, then validate the canonical seven-line H1-A2 Plan.
- [ ] Add train/serve round-trip, prefix-shape/mask, formula immutability,
  no-fill, and F/M visible-answer-identity tests.

## Phase 3 — train and interface canary

- [ ] Build immutable MP20 train/validation F/M data with identical source rows
  and rich targets.
- [ ] Train two fixed seeds per route for one adaptation epoch from the same
  historical H1-A2 rich Planner initialization; final checkpoint only.
- [ ] Report all four terminal checkpoints without seed selection.
- [ ] Run one fixed small syntax/interface canary; it cannot change settings.

## Phase 4 — prospective DLM realization

- [ ] Freeze one new C3FD all-attempt fixed256 composition ledger before DLM
  outcomes; retain failed requests and do not top up.
- [ ] Generate F/M Plans on identical compositions/order.
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

## Credential and execution boundary

Use the user-provided MP credential only in a temporary nonambient child
environment for each explicitly authorized query. Never write or echo it to
Git, docs, checklist, automation, commands, logs, hashes, manifests, or
archives. Unset it immediately and verify no process/runtime copy remains.

No retry/replacement/rerank/best-of-N, outcome-based setting selection, AR,
RL/GRPO/SMC, C3FD retraining, compact-V2 rerun, or alignment is authorized.
