# Execution Checklist: Llama-Programmed Basin Closure

Status: **24-hour SUN10/50 sprint; K10-4104 four-GPU action/value build**

Current decision: the 128-group K10 run is a completed feasibility-scale
result, not the data-scale endpoint. Its tau800 stream18 result is Strict/Meta
`15/119` of 256 with raw/refined validity `254/256`; native raw is Strict/Meta
`3/48`, with N/U `252/256`. Scale the same scientific object to 4,104 unique
MP20-train deployment states, using the actually schedulable two-, three- or
four-GPU topology at each stage; do not add epochs to the same 128 groups or
add a clean-CE-only route.

Sprint deadline: **2026-09-05 22:00 Asia/Shanghai**. The target is final
prospective Strict/Meta S.U.N. at least `10%/50%` under a fixed, fully disclosed
protocol. The deadline authorizes aggressive use of the registered compute and
methods, not denominator changes, seed/checkpoint selection, test leakage,
reranking or result fabrication.

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
- [x] Freeze 128 outcome-blind MP20-train plans, balanced 64 cell/64 terminal
  XYZ and stratified across N, multiplicity and species count; generate their
  on-policy closure trajectories with job 39731.
- [ ] Generate at most four outcome-blind legal actions per state.
- [ ] Execute the exact remaining closure with common random numbers.
- [ ] Calibrate one short relaxation horizon against the normal relaxation;
  report rank agreement, ties and per-stage variation.
- [ ] Report best-vs-no-op terminal energy headroom, proposal-path mass,
  scorer gradients, peak memory and elapsed time.
- [ ] If headroom is negligible or scorer/backward is infeasible, stop. Do not
  expand sources, epochs, candidates or KL to force a pass.

Current execution:

- [x] Freeze cohort with job 39730.
- [x] Finish reusable on-policy rollout body from job 39731: 128/128
  trajectories and closure logs, 127/128 parse/Plan/graph. The wrapper's obsolete
  128/128 geometry assertion failed after science completion; retain the invalid
  endpoint as a negative example and do not rerun generation.
- [x] Correct type-by-stratum assignment at state materialization; cover early,
  middle, late and terminal XYZ cursors without replacing any source.
- [x] Implement cursor-aware continuation and pure log replay: 128/128 states,
  64 cell/64 XYZ, four cursor buckets of 16, and the invalid endpoint retained.
- [x] Diagnose real-model old-final replay: 100/128 unpadded and 104/128 with
  original left padding. The remaining stochastic flips are batch-shape
  numerical sensitivity, not state/log inconsistency. Do not spend 8x compute
  emulating an old random batch. Register one batch-1 counterfactual
  continuation with common seeds for every candidate; old-final exact match is
  a reported nonblocking diagnostic.
- [x] Finish outcome-blind K<=4 action/terminal job 39748: 128 groups,
  K=`1:3, 2:4, 3:9, 4:112`, 486 retained candidates and 486 legal terminals,
  with all provisional-state EFSM calls known and no group failure.
- [x] Finish the two-GPU K3/5/10/20 CHGNet headroom label recovery 39752.
  Job 39750 failed before any CHGNet call because the validator rejected the
  legitimate `cursor=None` on cell transactions; the type-aware fix leaves all
  candidates and scientific parameters unchanged.
- [x] Authorize K10 training from the preregistered train-only decision: value
  coverage `100%`; all 128 groups paired; K10 median best-vs-no-op headroom
  `167.85 meV/atom`; 100 groups exceed `10 meV/atom`; K10/K20 pooled Kendall
  tau-b `0.590`. K10-selected actions also improve E0 by median
  `22.20 meV/atom` (71 lower, 37 higher), so basin and raw signals are not
  globally opposed.

## H. Conditional basin training

- [x] After G passed, freeze the 128-group train-only primary pilot with 486
  retained legal terminal actions; defer a larger source expansion unless the
  primary result proves it necessary.
- [x] Use only current 6-token cell and 3-token XYZ action scoring in pass one.
- [x] Use terminal K10 short-relax basin value after the remaining closure.
- [x] Keep hard validity and reference KL; do not restore a hard raw-energy
  non-increase constraint.
- [x] Alternate clean closure CE and posterior updates; no CE on generated
  states.
- [x] One seed, one endpoint, no early stop or checkpoint selection.

Completed training: job 39770 used 2 A800/8 CPU for the preregistered K10 primary,
four passes over 128 groups, interleaving 256 clean full-MP20 closure-CE and 256
posterior updates. All 512 updates completed; every group was seen four times,
with 492 informative and 20 retained zero-information posterior exposures.
Jobs 39753--39755 are pre-update engineering negatives. The
39755 diagnosis found that the step-0 no-grad policy/reference equality pass
left every LoRA parameter frozen before the gradient probe; commit `727d02a`
restores the trainable policy explicitly and adds a regression test. No data,
value target, learning rate, schedule or other scientific parameter changed.
Job 39758 then passed the probe and reached update 96 before a terminal invalid
cell exposed an in-place Boolean support-mask autograd bug. Commit `63b5b79`
uses an equivalent out-of-place terminal mask and adds a mixed-validity cell
backward regression; remote scoring and trainer tests pass. Job 39758 remains a
negative engineering run and contributes no selected checkpoint.
Job 39759 completed all 512 finite updates but failed before checkpoint save
because its exact four-pass coverage counter used NCCL-unsupported `int16`.
Commit `f911950` changes that bookkeeping tensor to `int32`; it does not change
the trained objective or schedule. The failed process produced no checkpoint,
so job 39770 restarts from the same registered closure-CE policy.

## I. Final evaluation

- [ ] Primary-result-first: run only K10 on frozen stream18, then evaluate its
  tau800 refined endpoint. Do not repeat closure-CE, four-cell validation or
  Direct before the primary S.U.N. result. The zero-second pending job 39774 was
  cancelled before science when the resource policy changed. Its replacement
  runs the single K10 arm as deterministic distributed shards, then merges by
  sample index before one tau800 refinement result. Use four ranks when all four
  GPUs are schedulable. Slurm reported six of eight node GPUs already allocated
  by other active jobs, so job 39779 uses both remaining GPUs/8 CPU immediately;
  it is one K10 experiment with two data-parallel ranks, not two arms.
- [x] Evaluate closure-CE raw first: fast validity, E/F/stress, relaxation,
  N/U and Strict/Meta S.U.N., with paired wins/losses.
- [ ] Evaluate K10 raw and tau800 refined Strict/Meta S.U.N. as the two primary
  endpoints, without Direct. Run both endpoint relaxations concurrently when
  at least two GPUs are available; defer all method/control comparisons.
- [ ] In parallel with the native value path, evaluate one preregistered
  low-noise `model494 tau200` bridge (job 39732) on the same closure-CE raw256;
  disclose it regardless of outcome and do not tune tau from its result.
- [x] Complete the fixed tau200 bridge and existing-cache official evaluation:
  generation/refinement job 39732, two-shard common relaxation 39733 and
  finalizer 39735. The result is Strict `15/256 = 5.86%`, Meta
  `98/256 = 38.28%`, with `256` reconstructed and `244` novel-unique. Relative
  to native closure raw this adds 8 Strict and 42 Meta S.U.N. outcomes, but it
  does not meet `10%/50%`.
- [x] Treat the completed fixed256 as development after this first outcome.
  Job 39737 generated tau400/tau600 and job 39747 completed the current-system
  tau800 anchor at 256/256. Job 39757 evaluated tau400/600/800 sequentially
  with one GPU, existing official cache and no Direct/query. Strict/Meta SUN
  are respectively `11/110`, `19/121`, and `23/119` out of 256. The
  preregistered balanced target-attainment rule selected tau800 before any
  held-out stream18 outcome was read;
  any chosen tau must be confirmed on a disjoint frozen cohort before a final
  prospective claim.
- [ ] The desired endpoint is native Strict/Meta `10%/50%`, but claims follow
  observed effect rather than the target.

## J. Resources and operating rules

- Maximum `4 A800`, `4 CPU/GPU`, at most two jobs.
- Use `starteam5090 -> tmux ssha800/ssha800_2`; do not reconnect an existing
  nested session unnecessarily.
- No unrelated jobs, GPU probes, row replacement, reranking or best-of-N.
- CHGNet may batch 8--16 structures per GPU; do not create 8--16 Python
  processes per GPU.
- Support any actually schedulable `1..4` GPU count, including odd counts. For
  heterogeneous raw/refined workloads, give all ranks to raw first and then
  reuse all ranks for refined; do not strand GPUs behind equal endpoint splits.
- GitHub pushes originate from the local workstation.
- Materials Project credentials remain outside Git/docs/checklists/commands,
  and are not needed before a final authorized official query.

## K. Scale-up and 1000-valid endpoint

- [x] Freeze 4,104 outcome-blind, distinct MP20-train rollout sources from the
  full 27,136 Plan/program pool: balanced cell/XYZ and cursor phase, no old
  three-stage ledger. The fixed cohort is
  `$ROOT/cohorts/spad_basin_scale4104_train_v1_20260905`. Job 39792 completed
  in 01:20:52 on three A800: 4,104/4,104 Plan-matched, parseable and graph-valid
  deployment rollouts; the materialized states contain 2,052 cell and 2,052 XYZ
  transactions, four 513-row XYZ cursor buckets, 4,104 unique source rows and
  zero reference-log replay mismatches. Outcomes were not read.
- [ ] Build one dynamic-K<=4 complete transaction group per state and label
  K10 only. Reuse the frozen K10 rule; do not repeat K3/K5/K20 calibration.
  The zero-science pending three-GPU job 39793 was cancelled and replaced by
  job 39794, now running with four A800/16 CPU. Dependency job 39796 will label
  E0+K10 only on four A800 immediately after 39794. These are throughput-only
  changes; the cohort, policy, seeds and scientific construction are unchanged.
- [ ] Train one posterior epoch from closure-CE: 4,104 posterior exposures and
  4,104 clean anchors across the actually schedulable 2/3/4 ranks, one seed and
  final checkpoint only.
- [ ] Freeze a new 256 C3FD->Llama program cohort before outcome evaluation:
  Planner seed 24, evaluation stream 19, DLM seed 93117, refiner seed 103117,
  fixed tau800. Report raw and refined Strict/Meta S.U.N. without Direct.
- [ ] If either preregistered primary endpoint reaches Strict>=10% and
  Meta>=50%, immediately launch the paper-scale 1000-valid-CIF run. Continue a
  fixed random stream until 1,000 parseable CIFs are accumulated; discard only
  parser/CIF failures without reading energy, hull, novelty or S.U.N. Report
  total requests and discarded-CIF count alongside the conditional-on-valid
  1,000 denominator.
- [x] Implement and unit-test the parser-only denominator constructor
  `select_first_parseable_spad_body.py`. It freezes the first 1,000 independently
  parseable CIFs in source order, keeps the matching proposal graphs, records
  every request/discard before the cutoff, and cannot consume outcome values.
