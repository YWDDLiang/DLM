# Execution Checklist: Llama-Programmed Basin Closure

Status: **native closure official result complete; basin-value preflight is next**

Authoritative active design:
[`12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md`](12_LLAMA_PROGRAMMED_BASIN_CLOSURE.md).
Historical SPAD/SPAD-E records remain evidence, not active instructions.

## Scientific pain point and adequacy decision

- [x] Define the remaining failure correctly: exact composition and Direct
  execution are already saturated; the unresolved object is the coupled
  periodic distribution of lattice and coordinates, whose raw structures have
  excessive energy, force, stress and short-contact tails.
- [x] Make the Llama program act on that same object. It first controls the
  predictor order, then controls the reverse species blocks used to revise the
  completed geometry; it is not a decorative Plan field.
- [x] Train the two conditionals that were previously out of distribution:
  sequential `L | X,c,P` and suffix-visible
  `X_block | L,X_other,c,P`, using exactly the masks served at inference.
- [x] Verify the mechanism without sacrificing the validity base. On fixed256,
  composition remained 256/256, fast structural validity rose 255->256, paired
  raw CHGNet energy improved by median -0.3244 eV/atom with clustered 95% CI
  [-1.4508,-0.0956], stress improved and force was non-adverse.
- [x] Test whether the native low-energy mechanism becomes a stability/SUN
  gain under frozen common relaxation and official hull evaluation. Energy
  survived (median hull delta -0.02274 eV/atom), but Strict S.U.N. was 7/256
  and Meta S.U.N. 56/256 versus frozen BS 7/256 and 54/256: the threshold-tail
  gain is not yet material. model494 remains a fallback, not the explanation.

## A. Frozen method boundary

- [x] Keep C3FD reachable chemical support and the trained Planner-Llama.
- [x] Keep the trained `species_program` pointer and explicit canonical fallback.
- [x] Keep exact `7+4N`; do not add tokens, a GNN or a continuous residual head.
- [x] Keep current BS semantics unchanged: predictor plus all-species anchor
  backfill, no cell closure.
- [x] Keep model494 as a separately labelled fallback.
- [x] Reject the historical G2 route and the old instantaneous
  Potential-Closure objective as active methods.
- [x] Reject the old `cell -> anchor_second -> anchor_first` ledger for this
  method; it matches neither current BS nor the new closure.

## B. Workspace and audit

- [x] Preserve prior dirty worktrees unchanged.
- [x] Create clean worktree
  `D:\codex_work\ai4s\DLM_llama_programmed_basin_closure` on branch
  `codex/llama-programmed-basin-closure` from remote commit `a22337a`.
- [x] Exclude the local raw-`E0`-safe posterior implementation.
- [x] Selectively retain the independently useful 3/6-token deployed action
  scorer.
- [x] Complete scientific, skeptic, resource, paper and code audits.
- [x] Pass all local pure tests and remote Torch integration tests.
- [x] Confirm no credential value, run artifact or model checkpoint is tracked.

## C. Outcome-free predictor asset

- [x] Let job39658 finish naturally.
- [x] Verify `27,136` source records, program accounting, body accounting,
  failure rows and terminal marker.
- [x] Retain only `reference_body` as a predictor-state asset.
- [x] Do not reuse its old `%3` transaction ledger.

## D. Closure runtime

- [x] Add reverse species-block slot compilation with exact one-pass coverage.
- [x] Add independent `revise_spad_species_blocks` runtime.
- [x] Add opt-in `--spad-basin-closure`:
  predictor -> cell -> reverse species blocks.
- [x] Reject simultaneous `--spad-backfill` and `--spad-basin-closure`.
- [x] Require closure-capability metadata before formal inference.
- [x] Preserve current BS output and seed behavior bit-for-bit.
- [x] Test cell atomic rollback, site rollback, non-active-token immutability,
  suffix visibility, row-local RNG and final no-mask/composition invariants.

## E. Closure-CE data and training

- [x] Build and independently audit a new teacher-only full-MP20 closure corpus
  with 27,136 train and 9,047 validation rows.
- [x] Implement cell states: all coordinates visible; sequential lattice suffix masked;
  one active lattice component supervised.
- [x] Implement species-block states: other species/full suffix visible; active block
  remainder masked; one active coordinate component supervised.
- [x] Keep `N/elements` visible and exact on every row.
- [ ] Add bounded same-structure cell/coordinate/short-pair corruptions only
  after clean state-replay tests pass.
- [x] Run a six-state-per-split remote canary covering first/last lattice
  components, X/Y/Z and a multi-site block; inspect finite loss/gradient before
  formal training.
- [x] Train one full-MP20 seed from BS for one registered endpoint: job 39700,
  1,696 updates, unique step-1696, validation loss 2.2679.

## F. Closure-CE raw screen

- [x] Implement and independently audit a paired native-screen path that reuses
  frozen BS and generates closure-CE only on fixed256 first.
- [x] Run fast validity, 125-image PBC distance tail, VPA agreement and CHGNet
  E/F/stress on fixed256; common relaxation is the next registered stage.
- [x] Do not run expensive Direct or model494 in the native screen.
- [x] Continue only if execution validity is retained and physical
  nonstationarity improves materially; do not use S.U.N. as a checkpoint or
  row-selection rule.

## G. Basin-value preflight

- [x] Authorize this stage only after closure-CE demonstrated a real native
  energy control surface but insufficient S.U.N. tail conversion.
- [ ] Freeze 128 train-only on-policy closure states spanning cell, XYZ,
  high-N and high-multiplicity cases.
- [ ] Generate at most four outcome-blind legal actions per state.
- [ ] Execute the exact remaining closure with common random numbers.
- [ ] Calibrate one short relaxation horizon against the normal relaxation;
  report rank agreement, ties and per-stage variation.
- [ ] Report best-vs-no-op terminal energy headroom, proposal-path mass,
  scorer gradients, peak memory and elapsed time.
- [ ] If headroom is negligible or scorer/backward is infeasible, stop. Do not
  expand sources, epochs, candidates or KL to force a pass.

## H. Conditional basin training

- [ ] Only after G passes, freeze 2,048--4,096 value sources.
- [ ] Use only current 6-token cell and 3-token XYZ action scoring in pass one.
- [ ] Use terminal short-relax basin value after the remaining closure.
- [ ] Keep hard validity and reference KL; do not restore a hard raw-energy
  non-increase constraint.
- [ ] Alternate clean closure CE and posterior updates; no CE on generated
  states.
- [ ] One seed, one endpoint, no early stop or checkpoint selection.

## I. Final evaluation

- [ ] Compare BS, closure-CE and conditional closure-basin on the same fixed
  requests.
- [x] Evaluate closure-CE raw first: fast validity, E/F/stress, relaxation,
  N/U and Strict/Meta S.U.N., with paired wins/losses.
- [ ] If the basin preflight passes, evaluate closure-basin raw using the same
  Strict/Meta S.U.N., with paired wins/losses.
- [ ] Run one canonical-versus-Llama program mechanism comparison at fixed
  DLM/composition/noise.
- [ ] Run fixed model494 only after native improvement; report it separately.
- [ ] The desired endpoint is native Strict/Meta `10%/50%`, but claims follow
  observed effect rather than the target.

## J. Resources and operating rules

- Maximum `4 A800`, `4 CPU/GPU`, at most two jobs.
- Use `starteam5090 -> tmux ssha800/ssha800_2`; do not reconnect an existing
  nested session unnecessarily.
- No unrelated jobs, GPU probes, row replacement, reranking or best-of-N.
- CHGNet may batch 8--16 structures per GPU; do not create 8--16 Python
  processes per GPU.
- GitHub pushes originate from the local workstation.
- Materials Project credentials remain outside Git/docs/checklists/commands,
  and are not needed before a final authorized official query.
