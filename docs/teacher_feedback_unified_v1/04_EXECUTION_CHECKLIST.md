# Execution Checklist: Llama-Programmed Basin Closure / Dual-Objective Review

Current status: **2026-09-06 05:07 Asia/Shanghai: round0 evaluation39910 COMPLETED52m16. Native Strict/Meta SUN7/57 of256 (2.734375/22.265625%); tau800 SUN19/126 (7.421875/49.21875%), versus reference6/55 and19/121. No physical A/B joint improvement is established. K8 label shardA39911 RUNNING4GPUs,1728/5462; shardB39916 RUNNING2GPUs,64/2730. Both shards are already submitted and disjoint. K8 sample39896 is complete8192requests/8093successes/99failures. Final K8 student and independent main remain pending.**

Latest new-task entry: [19 Architecture and execution](19_RESUMED_ARCHITECTURE_AND_EXECUTION.md).
Latest user amendment: [20 Data sufficiency and delivery](20_DATA_SUFFICIENCY_AND_DELIVERY_20260906.md).
Latest audit and release: [21 Terminal reproducibility](21_TERMINAL_REPRODUCIBILITY_AUDIT_20260906.md).
Continuous-refiner extension: [22 Code assessment and proposal](22_CONTINUOUS_DIFFUSION_EXTENSION_ASSESSMENT.md).
First-round evaluation: [25 Matched development results](25_ROUND0_MATCHED_EVALUATION.md).
A increase diagnosis: [26 Paired geometry and coverage](26_ROUND0_GAP_INCREASE_DIAGNOSIS.md).
The delegated continuous-diffusion/SDE audit is complete in
[24 Training and SDE audit](24_CONTINUOUS_DIFFUSION_TRAINING_AND_SDE_AUDIT.md).
It is documentation only; the main agent continues the K4/K8 execution pipeline.
This is a proposed follow-on design, not a trained refiner or a change to the
currently scheduled fixed-model494 comparisons. Preserve today's active pipeline.
The user explicitly confirmed this deferral at04:00: prioritize K4/K8 evaluations;
keep model494 frozen in this round and do not add further method extensions.
Keep the completed initial1024*K4 pool; expand the ONE train-only refresh to
the same1024 conditions*K8 (8192 requests), then two passes. No additional small
ablations. K4/K8 is now an explicit recorded data budget, not an inference selector.
Warmup39853/conditions39857 artifacts were verified and reused. New-process
replay39869 checked all153 decisions of two paths with zero error. Optimized
four-GPU training check39877 completed4path+1CE updates, gradient finite,
initial16decision replay error <=1e-6; this is an ineligible engineering model.
First128*K4 collection39872 finished512requests/509success; labels39873 finished
111verified/205not-converged/193invalid-terminal/3generation-failure. On the65
verified conditions, the diagnostic teacher has ~54meV/atom improvement in each
mean A/B, not a student result. Remaining896*K4 collection39878 completed58m47:
3584requests/3561success. Combined4096requests/4070success/26failure; six-GPU
labeling39885 is running on the remaining3584. The reference39884 completed28m35:
native Strict/Meta SUN6/256 and55/256; verified-subset1/256 and26/256. These are
reference results, not the new policy; hull is known for247, unresolved for8,
and1 input failed reconstruction.
At01:02,39885 had2112/3584labels:519verified/775not-converged/804invalid-terminal/
14generation-failure, with no recorded software errors. The K4/K8 amendment is
deployed atf6c372e; local focused21tests and remote path-training9tests passed,
and remote226 shell syntax passed. These overlapping suites are not added together.
Heartbeat01:07:39885 remainsRUNNING54m04;2368/3584labels complete. One occurrence
has relaxation_energy_increased and is retained with zero teacher weight under
the existing verification rule. No software error or new job submission.
User status check01:14:39885 RUNNING1h00m39,2734/3584labels complete;660verified,
1007not-converged,1049invalid-terminal,17generation-failure,1energy-increased.
850labels remain across the initial complete pool. No final student SUN exists.
Latest combined local suite:91PASS. Current implemented code is on the requested
branch; preserve the two pre-existing untracked PMTR files. Full-path trainer,
reference, tau800 and parser-only independent-main entrypoints are implemented;
future main sampling is guarded by a final-method lock. No new-method SUN exists.
The earlier task remains interrupted. At00:35 the existing ten-minute heartbeat
was migrated to this task and activated; this task is the sole execution owner.

## Execution ownership after resume

The user explicitly resumed this same mainline. Continue toward measured Strict
SUN>10% and Meta SUN>50%, without guarantees, outcome-selected replacement or
historical relabeling. Preserve the original deadline. Heartbeat
`llm-dlm-sun-24h` now targets `01a071d2-ebe3-79b1-a311-14a5107cb6c6`; do not
restore execution in the earlier task. Main agent
implements/tests/deploys/monitors directly; no coding delegation.

Immediate work: monitor K8 labels shardA39911 and shardB39916; do not resubmit either.
The complete8192-request K8 sample is certified by sample/_SUCCESS and
SAMPLE_FINAL.json; do not regenerate it.39910completed native/tau800 atstep1020,
explicitlyround0_diagnostic.39911uses four GPUs, shard-count3/shard-ranks0,1
(5462requests).39916uses two GPUs, shard-count3/shard-ranks2 (2730requests).
Merge both complete label outputs;
never relabel the full8192or duplicate shards. Build round1teacher with--candidates8, review its
coverage and concentration, then continue the step1020optimizer for two passes.
The refresh uses a count-based6..24decision budget per path/pass, capped by the
registered98304real scalar-state budget; see20. Evaluate the final policy against
the matched reference39893, then freeze before main1000.
Do not repeat completed warmup, preparation, prefix collection or checks.

**Latest release overrides the earlier data-quality hold:**39891 audited all973
verified terminals;972passed and one exceeded the prespecified1meV representation
threshold by only7.08micro-eV. The-499prediction was reproduced, not proven a
recording error. See21for the full limitation. New labels/teacher are in
data/spad_state_{labels,teacher}_round0_consistent_20260906_v2.39892uses this
teacher from the original39853warmup. Never use the invalidatedv1teacher or39889.
Reference verification39893 completed42seconds and preserved the primary counts.
Use runs/spad_state_reference_verification_39893/native-evaluation and
tau800-evaluation for the final comparison. Reference Strict/Meta SUN is
2.34375/21.484375% native and7.421875/47.265625% tau800; these remain reference
results. First new-student development results are provisionally budgeted for
08:00–10:00; this estimate depends on refresh labeling speed. The newly requested
round0process result is provisionally expected04:00–05:00 after sampling releases
GPUs; it is not the final result or a small component ablation. Deadline19:19 remains.

## Historical handoff pause

Read [current handoff](../HANDOFF_STATE_PROGRAMMED_SPAD_20260905.md) first.
The user asked to stop local progress because the session is lagging, then clarified
that A800 and running tasks should be left alone. No job was cancelled or restarted.
Automation `llm-dlm-sun-24h` is now PAUSED; do not resume it without explicit approval.

Last scoped remote snapshot, approximately 2026-09-05 21:21 Asia/Shanghai:
- 39850 preflight FAILED 1:0 / 00:01:50 (missing required constructor argument).
- 39852 recovery preflight COMPLETED 0:0 / 00:01:57.
- **39853 full-MP20 state warmup COMPLETED 0:0 / 00:23:18.**
- **39857 predicted train conditions COMPLETED 0:0 / 00:00:52.**
- No active/pending `spad-state-*` job in that snapshot; no claim about other jobs.

39852 recorded 51 sampled decisions and replay-checked only the first three (error0);
same-instance conditioner reload was checked, not fresh-process full-model reload.
No new SUN was computed. The sampling CLI/full-path trainer are still missing, and
the uncommitted labeler test currently fails at import/collection. See the handoff
for exact files and next steps. Earlier execution instructions below are suspended.

## Current authorization: one 24-hour implementation sprint

The user approved the audited dual-objective plan and requested full effort toward
Strict SUN>10% and Meta SUN>50% in 24 hours. On the same turn the quota increased
to six A800, four CPU per GPU. The initial deadline is 2026-09-06 19:19
Asia/Shanghai. This authorizes the new plan, not restarting PMTR/K10 or fabricating
results. Pass the combined interface checks before formal GPU training.

Current proposal: [17 State-conditioned dual-objective SPAD](17_STATE_CONDITIONED_TERMINAL_BASIN_PLAN.md).
Review decisions: [18 Review and resolutions](18_DUAL_OBJECTIVE_REVIEW_AND_DECISIONS.md).

- [x] Recheck retained C3FD/Planner-Llama species program, canonical 7+4N and
  deployed predictor/reverse-species closure from current code.
- [x] Specify joint optimization of original-to-relaxed energy gap A and
  terminal basin energy B, with fixed composition/program weights.
- [x] Complete read-only mathematical and engineering reviews; distinguish
  a feasible empirical teacher from guaranteed student/SUN improvement.
- [x] Review the single-method paper story against the teacher's feedback;
  retain LLM-programmed execution without claiming online control or end-to-end training.
- [x] Correct the scalar-state/backprop budget and record untested interfaces.
- [x] Obtain user confirmation; six A800 / 24 CPU maximum, one main method.
- [x] Run the combined real-model preflight 39852 with the limited coverage recorded
  above; do not infer exhaustive coverage from its success marker.
- [x] Complete fresh-process checkpoint loading and production-path coverage before
  the next formal generation:
  injection/gradient/reload/throughput; joint commit/rollback/support;
  complete attempted-trace sampling/scoring/zero-weight/HT consistency.
  Coverage is the153fresh-process decisions and16initial training decisions
  above, plus CPU contracts; it is not exhaustive replay of the full4096pool.
- [x] Full MP20 warmup: 27136 sources, one source state each; freeze embedding/head,
  train retained LoRA and the small state conditioner; job39853 completed.
- [x] Prepare the frozen predicted training conditions with unchanged chemistry;
  job39857 completed. Read its manifest after user-authorized resumption.
- [x] Collect 1024 train conditions x K4 full paths; use first128 for label/protocol
  diagnostics, not composition selection; preserve all attempts and failures.
- [ ] Build the verified empirical A/B teacher with feasible common improvement;
  fit full-path likelihood for two passes, then one train-only K8 refresh/two passes.
- [ ] Replace preliminary data-budget estimates with both rounds' actual verified
  paths, covered conditions, multiple-candidate conditions, ESS and update counts.
- [ ] Report per-condition teacher improvements/worsening, total variation from
  its verified uniform reference and gain concentration; do not interpret the
  global mean constraint as a per-condition or student guarantee.
- [ ] Check remaining compute at13:00, assemble findings/missing items at18:00,
  and deliver actual results by the retained2026-09-06 19:19 deadline.
- [ ] One fixed256 comparison with one reference; report native e/F/stress,
  paired A/B, Stable/N/U/SUN and the separately labelled tau800 fallback.
- [ ] Freeze the method before independent main sampling; if using 1000 valid CIFs,
  retain source order and disclose all requested/failed attempts.

Resources: at most 6 A800, 4 CPU per GPU, at most two jobs; no inference MLIP/reranking
or test-outcome training. Prefer four-GPU training with unchanged effective batch16,
plus two-GPU data/evaluation; use all six for independent generation/label shards.
The requested 24h target is an effort deadline, not a guaranteed scientific result.

Latest work-division instruction: implementation, tests, deployment, monitoring,
and archiving are performed directly by the main agent. Subagents are reserved
for large-scale audit or research only. The remaining coding sidecar was stopped
and its unfinished work taken back; no further coding delegation is authorized.

Implementation progress at 2026-09-05 19:26 Asia/Shanghai:
- Persistent local SSH to starteam5090 established; attached to existing
  ssha800:1.0 with TERM=xterm-256color; ssha800_2 untouched.
- Scoped project queue check found no active spad/pmtr/c3fd/basin/closure job.
- Main agent owns runtime/trace integration and deployment. Two bounded workers
  own basin_path_objective plus tests and periodic_state_conditioning plus tests.
- New ten-minute heartbeat `llm-dlm-sun-24h` is ACTIVE until the requested deadline;
  the prior h1-a2-stability-ccfd automation no longer exists and was not duplicated.
- Local isolated .venv is being prepared for CPU tests. No training job submitted yet.

Update 2026-09-05 20:34 Asia/Shanghai:
- CPU suite: 105 local tests PASS; remote state/runtime/objective 18+6+6 PASS.
- Current deployed source is the isolated
  `/public/home/jiaosz/ywliang/ai4s/.sscd_state_programmed_20260905_v1` checkout.
  Code/docs pushed from the local machine (6dc53b5, a8c6c60).
- Preflight job39850 used 2A800/8CPU and failed after 00:01:50. Both ranks completed
  five finite-gradient steps (~0.30s/step); failure was a missing required
  `min_lattice_rad` argument when creating sampling constraints after reload.
  It is an engineering preflight, not an eligible policy or SUN result.
- Fix pins the retained `min_lattice_rad=1e-4`; an AST contract regression test
  covers both new entrypoints. Preserve job39850 and retry in a new run only.
- Full-MP20 warmup entrypoint and four-GPU wrapper are implemented; not submitted
  until the real preflight success marker is present.
- Parameter-identical preflight recovery job39852 was submitted from bb9eebf;
  check its actual outcome before any further preflight or warmup submission.

The historical checklist below is retained for provenance only.

## Historical closed K10 sprint — not active instructions

Historical status: **closed prospective K10 sprint; target not met**

Final decision: the data-scale 4,104-state K10 posterior route completed one
full pass plus its single preregistered warm-start pass. Final stream21 raw
Strict/Meta S.U.N. is `5/49` of 256; tau800 is `14/123`, with 256/256
reconstructed and 254/230 novel-unique. Exact `26/128` and near `23/125` gates
both fail. Do not launch paper1000 or any additional method iteration.

Sprint deadline: **2026-09-05 22:00 Asia/Shanghai**. The target is final
prospective Strict/Meta S.U.N. at least `10%/50%` under a fixed, fully disclosed
protocol. The deadline authorizes aggressive use of the registered compute and
methods, not denominator changes, seed/checkpoint selection, test leakage,
reranking or result fabrication.

Historical closed design:
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
- [x] Evaluate K10 raw and tau800 refined Strict/Meta S.U.N. as the two primary
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
- [x] Build one dynamic-K<=4 complete transaction group per state and label
  K10 only. Reuse the frozen K10 rule; do not repeat K3/K5/K20 calibration.
  The zero-science pending three-GPU job 39793 was cancelled and replaced by
  job 39794, which completed in 01:40:17 on four A800/16 CPU. It produced all
  4,104 groups with retained-candidate histogram K1/K2/K3/K4 =
  25/292/409/3,378, 15,348 retained candidates and zero contract errors;
  outcomes, selection and replacement remained false. Job 39796 completed
  E0+K10-only labels in 00:31:36 on four A800: 4,104 groups, 15,348 candidates,
  complete E0/K10 coverage for all 15,347 legal candidates, median K10
  best-vs-no-op headroom 143.93 meV/atom, and 2,879 groups above 10 meV/atom.
  These are throughput-only changes; the cohort, policy, seeds and scientific
  construction are unchanged.
- [x] Train one posterior epoch from closure-CE: 4,104 posterior exposures and
  4,104 clean anchors across the actually schedulable 2/3/4 ranks, one seed and
  final checkpoint only. Job 39797 reached 64 finite updates but failed before
  any checkpoint save because NCCL initialized before rank-local device binding:
  ranks 1--3 each stranded about 6.37 GiB on GPU0 and rank0 missed a 16 MiB
  allocation. Commit 5f24a02 binds `cuda:LOCAL_RANK` before NCCL and replaces
  all-visible-device seeding with identical rank-local seeding; 17 local and 17
  remote tests pass. Four-rank recovery 39798 reproduced the same update-64
  memory boundary, showing that rank binding fixed initialization order but did
  not remove the approximately 6.37 GiB auxiliary allocation per additional
  rank on physical GPU0. Both 39797 and 39798 are negative engineering runs;
  neither saved an eligible checkpoint. Job 39799 now runs the same one-epoch
  science on the preregistered three-rank topology (12 CPU), which removes one
  auxiliary allocation while preserving all 4,104 posterior and clean-anchor
  exposures. In parallel, the fourth A800 ran the disjoint seed-24 Planner
  source job 39800. Data, LR, objective and seeds remained unchanged. Job 39799
  completed in 01:00:58 with 2,736 finite optimizer updates, exactly 1,368
  clean-CE and 1,368 posterior updates, all 4,104 groups seen once, 3,889
  informative exposures and 215 retained zero-information exposures. Only its
  final policy was saved.
- [x] Freeze a new 256 C3FD->Llama program cohort before outcome evaluation:
  Planner seed 24, evaluation stream 19, DLM seed 93117, refiner seed 103117,
  fixed tau800. Plan job 39800 completed 256/256 parsed and composition-valid;
  freeze job 39801 retained all requested ordinals without outcome reads or
  replacement at `$ROOT/cohorts/spad_prospective_seed24_256_v1_20260905`
  (255 unique exact compositions and one disclosed duplicate). Report raw and
  refined Strict/Meta S.U.N. without Direct.
  Final generation/refinement job 39802 completed in 00:15:42 on four A800:
  body decoded/parsed/Plan-matched/graph-valid 256/256 and raw/refined assembly
  both succeeded 256/256. Raw-first S.U.N. job 39803 completed its four-A800
  CHGNet/newness science with no Direct, then its old-cache finalizer stopped on
  an uncovered chemsys. A fresh official cache resolved 245/253 chemsys with
  eight explicit unknowns. The recovered final raw
  Strict/Meta S.U.N. is 12/61; tau800 is 14/107. The exact and near-line gates
  are both false.
- [x] Apply the registered paper-scale gate after the final endpoint. Neither
  endpoint reached Strict>=10% and Meta>=50%, and the final endpoint also missed
  the fixed 23/125 near gate, so paper1000 was correctly not launched. The
  preregistered rule was: if either primary endpoint reaches Strict>=10% and
  Meta>=50%, immediately launch the paper-scale 1000-valid-CIF run. Per the
  user-authorized, outcome-independent near-miss rule fixed before stream19,
  launch is also allowed when the same endpoint is no more than three counts
  short on either target: Strict>=23/256 and Meta>=125/256. Continue a
  fixed random stream until 1,000 parseable CIFs are accumulated; discard only
  parser/CIF failures without reading energy, hull, novelty or S.U.N. Report
  total requests and discarded-CIF count alongside the conditional-on-valid
  1,000 denominator.
- [x] Implement and unit-test the parser-only denominator constructor
  `select_first_parseable_spad_body.py`. It freezes the first 1,000 independently
  parseable CIFs in source order, keeps the matching proposal graphs, records
  every request/discard before the cutoff, and cannot consume outcome values.
- [x] Prepare the gated paper-scale generation/refinement wrapper. It remains
  unsubmitted until exact 10%/50% or the fixed 23/125 near-miss rule is met,
  and preregisters Planner seed 25, stream 20,
  DLM seed 94117, refiner seed 104117, a 1,200-request fixed source stream and
  the first 1,000 parser-valid CIFs as the conditional evaluation denominator.
- [x] Prepare the paper-scale raw-first evaluation wrapper and conservative
  existing-cache finalizer. All available ranks evaluate raw first and refined
  second; Direct and new official queries are omitted, and any chemsys absent
  from the retained MP cache remains an unknown/non-stable row in denominator
  1,000 rather than being dropped.

## L. Single final iteration after a miss

- [x] If the new stream19 endpoint misses even the fixed near-line gate
  (Strict 23/256 and Meta 125/256 in the same endpoint), first publish a paired
  diagnosis covering training movement, K10 teacher headroom/reachability,
  raw geometry/energy and raw-to-tau800 retention. Classify the failure as
  exactly one of `UNDERTRAINED`, `LOCAL_CONTROL_INSUFFICIENT`,
  `REFINER_WASHOUT`, or `EVALUATION_COVERAGE`.
- [x] Classify stream19 as `UNDERTRAINED`: coverage is 4,104/4,104, the K10
  teacher has strong cell/XYZ headroom, gradients are balanced and finite, but
  one state exposure did not reduce sampled posterior loss from the first to
  last training quarter. Full evidence is in
  `13_STREAM19_DIAGNOSIS_AND_FINAL_ITERATION.md`.
- [x] Permit exactly one reproducible final iteration selected by that physical
  diagnosis: one additional unchanged-LR K10 posterior pass for undertraining;
  the existing bounded Plan-VPA projection for insufficient cell control; or
  the preregistered tau600 topology-preserving bridge when tau800 erases a real
  raw gain. Cache-coverage failure changes reporting only, not the model. The
  selected branch was one warm-start K10 pass from job39799: three-A800 training
  job39805 completed successfully with the same 4,104 labels, LR5e-6, clean
  interleave and one-pass schedule (2,736 finite updates; 3,889 informative
  posterior exposures; step-0 equality passed; only the final policy saved).
  Planner seed26 job39806 and the outcome-blind 256-row cohort freeze job39807
  also completed. Final fixed stream21 generation/refinement job39810 completed
  in 00:15:49 on four A800s: requested/body/parsed/Plan-match/graphs/refined are
  all 256/256, with no retry or replacement. Raw-first/tau800 evaluation job
  39814 completed in 00:16:07. A fresh query resolved 247/251 chemsys, yielding
  hull coverage 252/256 after row mapping; the four unknown rows remain
  non-stable. Raw Strict/Meta S.U.N. is 5/49 and tau800 is 14/123. Exact and
  near gates fail; no paper1000 run is authorized.
- [x] Consume the single allowed final iteration with job39805; do not launch a
  second final iteration or edit data, labels,
  denominators, seeds or checkpoints to manufacture 10%/50%. Aggressive compute
  is authorized; outcome curation and fabricated results are not.

## M. Closeout

- [x] Stop method changes after the single registered final iteration.
- [x] Keep all 256 attempts in both endpoint denominators.
- [x] Record the fresh official coverage and explicit unresolved rows.
- [x] Record that complete execution did not transfer into strict stability:
  the warm-start pass changed refined Meta S.U.N. from 107 to 123 but left
  refined Strict S.U.N. at 14 and reduced raw Strict/Meta from 12/61 to 5/49
  across independent prospective streams.
- [x] Do not launch paper1000.
- [x] Update the repository README and final diagnosis with the observed result.

## N. User-authorized PMTR continuation

The earlier K10 closeout remains final for that failed method. The user later
authorized one new, scientifically distinct PMTR route on the clean branch.

- [x] Freeze the scientific object before coding: learn a local retraction from
  a coherent quantized `SPD(3) x T^(3N)` corruption to the original MP20-train
  relaxed structure; do not optimize a reward or generated endpoint.
- [x] Preserve current C3FD composition support, Llama species program, SPAD
  `7+4N` transactions, triclinic PBC support, suffix visibility, and atomic
  rollback.
- [x] Implement shared SPD/PBC geometry primitives and exact MIC vectors.
- [x] Implement coherent corruption and inference-matched paired repair rows.
- [x] Implement an optional transaction context/logit-transform interface;
  `None` and a zero PMTR head retain existing output/RNG behavior.
- [x] Implement the zero-initialized lattice/site repair head, normalized
  continuous objectives, and old-to-target native-token probability transport.
- [x] Implement the inference transform with no MLIP dependency.
- [ ] Complete the offline-only batched CHGNet certificate builder.
- [ ] Complete the head-only trainer and checkpoint format. Freeze the retained
  SPAD DLM/LoRA during the first integrated run to protect the measured 99.8%
  execution-validity base; alternate clean identity and corrupted repair
  updates rather than mixing two losses in one step.
- [ ] Complete the opt-in production sampling path and explicit no-MLIP runtime
  dependency test.
- [ ] Run all focused and adjacent remote tests, then one tiny real-checkpoint
  smoke test.
- [ ] Run one integrated MP20-train 512-row preflight: certify corruptions,
  train the small head, and evaluate one fixed actual-SPAD raw cohort. This is a
  mechanism-transfer test, not a hyperparameter or result sweep.
- [ ] If the preflight transfers to actual raw structures without reducing
  composition/parse/PBC validity, run the fixed full `27,136 x 2 epochs`
  head-only training once on up to four A800 with four CPU per GPU.
- [ ] Evaluate the registered raw endpoint first. Keep model494 tau800 as a
  separately labelled fallback; do not use CHGNet, force, stress, energy,
  reranking, replacement, or candidate selection at inference.
- [ ] If the fixed endpoint reaches Strict/Meta S.U.N. `10%/50%`, start the
  already defined first-1,000 parser-valid paper endpoint. Do not manufacture
  counts or alter outcomes.
