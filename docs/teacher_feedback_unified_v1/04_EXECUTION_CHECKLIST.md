# Execution Checklist: Scientific-State Commit Decoding

Status: **EXECUTION APPROVED — implementation-level module audit is first**

## Phase 0 — design and clean workspace

- [x] Preserve the existing dirty worktree and all user changes.
- [x] Create clean branch `codex/unified-scientific-decoding` from the retained
  `codex/force-score-g2` endpoint.
- [x] Create separate worktree
  `D:\codex_work\ai4s\DLM_unified_scientific_decoding`.
- [x] Draft unified, Track-A, Track-B and cross-representation documents.
- [x] Complete sequential skeptic, constraint, reader and arbiter reviews.
- [x] Incorporate every accepted objection into the decision log.
- [x] Create active ten-minute heartbeat
  `sscd-a-b-approval-and-execution` with this approval HOLD.
- [x] Obtain explicit user approval to execute.

## Phase 0.5 — implementation-level module audits

- [ ] Audit C3FD–Llama action support, learned residual participation and
  proposal composition validity separately from body composition retention.
- [ ] Audit the deployed DLM tokenizer from the actual base and retained
  checkpoints: every special token is one ID; N=1–20, supported elements,
  500 positive length values, 179 angle values and periodic coordinate values
  cover MP20 train/validation and the intended deployment domain.
- [ ] Count all teacher values that clip, alias, split into multiple tokens or
  fall outside support. Verify mask/pad/eos IDs and embedding/output rows in
  every checkpoint.
- [ ] Verify exact `7+4N` token count and prompt+body context length for every
  MP20 train/validation row.
- [ ] Audit AR text→semantic action→DLM special token→continuous array
  round-trips, including length-domain intersection and coordinate 000/100
  aliases.
- [ ] Audit whether current DLM APIs support non-contiguous future-first block
  commits, anchor-first generation, suffix-visible backfilling and true
  re-masking without converting the sampler into left-to-right decoding.
- [ ] Audit the Llama→DLM training/serve signal: ProgramHead-A ownership,
  SLA-A field targets, rollout-matched agreement-gate states, mask-time
  calibration and state refresh after each joint commit.
- [ ] Map each teacher requirement to one implementation component and one
  adjacent experiment; remove any component that lacks a measurable role.
- [ ] Update the unified method/decision log from the audit before the first
  scientific training submission.

## Phase 1 — shared interface implementation

Begins only after approval.

- [ ] Add canonical crystal-state dataclasses with ephemeral but
  transaction-stable serialization-slot IDs.
- [ ] Wrap the existing CrysLLMGen text parser/renderer as Codec A.
- [ ] Wrap dynamic `7+4N` special-token conversion as Codec B.
- [ ] Add the model494 tensor adapter as Codec C.
- [ ] Build semantic value tables for element, length, angle and coordinate
  families; never join vocabularies by token ID.
- [ ] Freeze the common positive length domain `0.1–50.0 Å` and aggregate
  periodic coordinate aliases `000/100` by log-sum-exp before fusion.
- [ ] Add AR-boundary Semantic Logit Adapter heads.
- [ ] Add a diagnostic native-text candidate-trie likelihood scorer.
- [ ] Bound trie diagnostics to 512 MP20-train rows and at most one available
  position per field family per row.
- [ ] Implement one atomic commit transaction across canonical state, AR
  transcript, DLM canvas and PBC graph cache.
- [ ] Replace `delta-round` and uncertified bounded-image collision checks
  with validated triclinic MIC for every complete site-triplet candidate.
- [ ] Implement one-forward block logits: progressive width-32 lattice beam
  and default top-four-per-axis XYZ beam (64 triplets), with one atomic commit
  after exact PBC filtering.
- [ ] On 256 MP20-train structures, measure legal-triplet coverage,
  candidate-MIC pairs/s and peak memory; freeze top eight only if coverage
  requires it and measured throughput remains feasible.
- [ ] Run local codec, quantization, site-identity and partial-state tests.

Deliverable: one small CPU/interface test report. No scientific metric is read.

## Phase 2 — shared Llama controller data

- [ ] Reuse the frozen successful C3FD–Llama chemical Planner.
- [ ] Build MP20-train species-order soft labels from one teacher-forced
  forward per row of the frozen starting AR-body checkpoint; aggregate
  per-species native-text confidence and same-composition polymorph
  disagreements. Freeze labels before BodyAdapter-A training; do not use DLM,
  dynamic relabeling or energy counterfactuals.
- [ ] Build full MP20-train AR body text in program order.
- [ ] Attach semantic field labels from the same canonical teacher state.
- [ ] For every source row, keep teacher-Plan and frozen-predicted-Plan
  same-schema conditioning views with combined source weight one.
- [ ] Keep MP20 validation for diagnostics and leave test/prospective outcomes
  unseen.
- [ ] Audit that AR decimal precision and DLM bins map to the same values.

Deliverable: one train/validation manifest with program, text and semantic
labels; no generated or model494-refined structure becomes a teacher.

## Phase 3A — Track A LLM-only executor, 2 A800

- [ ] Train one fresh `BodyAdapter-A` for one full source-weighted
  MP20-train epoch: LoRA r8/alpha32/dropout0.05, effective source batch16,
  exactly 1,696 updates on two A800.
- [ ] Jointly train native text CE, `SLA-A` field CE and
  `ProgramHead-A` species-order loss.
- [ ] Keep the retained chemical Planner frozen.
- [ ] Freeze BodyAdapter-A, SLA-A and ProgramHead-A together as the exact
  controller reused by every Track-B cell.
- [ ] Save one endpoint; do not choose an intermediate checkpoint from quality
  outcomes.
- [ ] Implement A0 Program+SLA with syntax/exact composition only and A1 as the
  same weights plus lattice/PBC commit control.
- [ ] Verify exact Plan inventory, canonical text parse and per-field
  SLA/native-head calibration.

## Phase 3B — Track B LLM-guided DLM, up to 4 A800

Track B is the paper-priority route. Controller-dependent B training/generation
may use 4 A800 after Track A freezes; the remaining 2 A800 serve Track-A
baselines, data preparation or evaluation.

- [ ] While Track A trains, implement the common block schedule and run only
  Track-B code/tests plus controller-independent BC baseline preparation.
- [ ] After the Track-A endpoint freezes, implement frozen-weight adjacent
  cells BC/BO/BG/BP: canonical order, Llama order, SLA/gate, then PBC.
- [ ] Train the small agreement gate on MP20-train rollout states using fused
  teacher-value CE plus an alpha-to-zero regularizer and a field/stage KL cap.
- [ ] Keep the agreement gate language-only; periodic-risk features enter only
  after fusion in BP.
- [ ] Calibrate DLM logits by field family, remaining-mask ratio and block
  stage; all adjacent cells use the same mask-time schedule.
- [ ] Starting once from the retained Compact-V2 endpoint, train B2 for one
  full MP20-train epoch with the actual program/block mask schedule: LoRA
  r8/alpha32/dropout0.05, effective source batch16, exactly 1,696 updates on
  two A800.
- [ ] Keep exact `N` and element slots visible; train only lattice/coordinate
  body decisions.
- [ ] Keep geometric hard support out of CE normalization when a quantized
  teacher value conflicts; report those rows and exclude them only from the
  geometry-controller auxiliary term.
- [ ] Do not include CHGNet, hull, model494 endpoints or generated evaluation
  outcomes in B2 training.
- [ ] Train only DLM weights in B2; never update the frozen Track-A controller.
- [ ] Keep G2 residual as a shared-subset mechanism ablation until its
  interaction with the new schedule is known.

## Phase 4 — first matched result

- [ ] Freeze one outcome-blind 256-request Plan/program ledger.
- [ ] Before scale generation, load one BF16 Llama and one BF16 DLM on a single
  assigned A800 with `device_map=None`, no CPU offload and batch1; record peak
  allocated/reserved memory and keep at least 8 GiB headroom.
- [ ] Run A0/A1 and BC/BO/BG/BP/B2 with one trajectory per request and two
  common sampling streams for one frozen model seed.
- [ ] Use no retry, replacement, reranking or best-of-N.
- [ ] First compute body/parse/exact composition, raw Direct, graphability,
  minimum-distance distribution, volume and condition number.
- [ ] Run Direct concurrently across CPU workers rather than serially.
- [ ] Benchmark CHGNet persistent worker counts `4/8/10` on one assigned
  A800 with exactly 8 allocated CPU cores and one thread per worker. Ten workers
  is permitted only as process oversubscription inside those same 8 cores;
  never allocate more than 8 CPU cores per GPU.
- [ ] Run Direct outside the CHGNet worker pool or on explicitly unclaimed CPU
  cores; total active CPU allocation never exceeds the job request.
- [ ] Deduplicate the structure union before CHGNet and run A/B evaluations in
  parallel.
- [ ] Report Planner proposal composition validity and body composition
  retention as separate fixed-requested-denominator metrics; each route's
  retention target is at least 95%.
- [ ] Report Llama/DLM argmax agreement, fusion-change frequency,
  permutation-aware teacher accuracy, Llama calls, DLM NFE and wall time.

If a cell cannot produce comparable exact-composition bodies, diagnose its
interface before spending on terminal model494. A scientifically negative raw
result is retained and reported; it is not repaired by selecting another
sample.

## Phase 5 — Candidate E1 continuous-response corrector

Starts after the A/B raw result is understood. E1 is not counted as a paper
contribution until its promotion criterion is met.

- [ ] Expose the actual deterministic deployed model494
  `tau800→799` transition; do not concatenate raw `pred_x/pred_l` as a force
  or assume an in-distribution score.
- [ ] Generate exactly 1,024 complete graphable B2 predictor states on
  MP20-train compositions and 256 disjoint MP20-validation development states
  with the same predicted-Plan/runtime contract used at inference.
- [ ] Train Candidate-E1-owned `Confidence-E1` with coordinate-site and
  lattice heads from CHGNet force/stress and finite-difference energy labels;
  freeze it separately from the unchanged Track-A controller and B2.
- [ ] Define usable response as calibrated confidence at least 0.60. Start B3
  only when validation directional AUC exceeds 0.55 and more than half of usable
  responses lower calibration energy without increasing feasibility risk.
- [ ] Keep CHGNet absent from production decoding.
- [ ] Implement one complete-state B3 corrector block with a no-op candidate.
- [ ] Project coordinate drift through PBC torus displacement and lattice drift
  through log-metric tangent.
- [ ] Apply hard support, soft feasibility potential and a cited primal-dual
  risk/KL solver in semantic action space.
- [ ] Compare B2/B2C0/B3 raw first; run terminal model494 only after those raw
  outcomes are frozen.
- [ ] Add B2C0: use the same selected block, first-transition call, fresh
  Llama/SLA call, DLM corrector pass and random stream as B3, but set the
  response residual to zero.
- [ ] Select the B2C0/B3 block only from frozen analytic risk and predictor
  uncertainty; model494 response never affects block identity.
- [ ] Attribute B2→B2C0 to generic block reopening and B2C0→B3 to the
  continuous response; both raw and refined comparisons are compute-matched.

## Phase 6 — final system evidence

- [ ] Apply the same terminal model494 setting to A1, B2, B2C0 and B3.
- [ ] Report raw and refined endpoints separately.
- [ ] Treat CHGNet and the existing MP-reference/CHGNet-candidate hull as
  surrogate endpoints.
- [ ] Use a held-out second MLIP for independent surrogate robustness.
- [ ] Promote response steering only if B3 improves two-stream pooled
  second-MLIP median energy over B2C0 and requested-denominator body/Direct are
  each no worse than B2C0 by more than one percentage point; also report the
  total B3−B2 effect.
- [ ] Use a registered DFT candidate-energy subset only if the paper makes an
  ab initio/thermodynamic claim.
- [ ] Report Strict/Meta S.U.N., N/U/NU, composition validity, CIF validity,
  minimum-distance ECDF and compute.
- [ ] Evaluate the target `Strict S.U.N. >10%` and
  `Meta S.U.N. >50%` on the frozen requested denominator and two common
  streams. The target guides iteration but never licenses row deletion.
- [ ] Write the final method diagram, adjacent comparisons and contribution
  table.

## Evidence-driven iteration

- A small engineering failure is fixed in place without changing the scientific
  comparison and is recorded once with its cause.
- A scientific negative is retained, diagnosed by module and followed only by
  a causally motivated adjacent revision—not a parameter sweep.
- A small positive is reproduced on the second common stream and explained
  through token coverage, commit behavior, geometry and energy diagnostics.
- A large positive is frozen immediately, archived as the candidate main
  endpoint and celebrated only after the paired evidence is verified.

## Resource schedule

Hard resource ceiling after approval:

- at most 6 A800 total;
- at most 2 concurrent jobs;
- normally 2–3 A800 and 8 CPU per GPU for each job;
- Track A training uses 2 A800. The second job may use 2 A800 for BC
  generation/throughput work while Track A trains; BO/BG/BP and gate work wait
  for the frozen Track-A controller;
- after Track A freezes, Track B receives up to 4 A800 and the remaining 2 are
  reserved for the A baseline/evaluator, while the global two-job limit remains;
- dual-model inference uses BF16, explicit one-GPU placement, no CPU offload
  and batch1 unless the measured memory canary safely raises it;
- evaluation uses persistent CHGNet workers and concurrent Direct CPU workers.

Provisional critical path after approval:

| Work | Expected wall time |
|---|---:|
| shared codecs/tests/data | 3–5 h |
| program labels + AR data | 2–4 h |
| Track A training + controller-independent B work, overlapping | 4–8 h |
| controller-dependent BO/BG/BP + B2 training | 3–6 h |
| seven-cell, two-stream fixed256 generation/raw evaluation | 5–8 h |
| 1,024/256-state diffusion corrector calibration/implementation | 4–7 h |
| final terminal refine/evaluation/report | 3–5 h |

These estimates are replaced by measured canary throughput; they are not
scientific gates.

## Minimal invariants

- different tokenizers communicate only through semantic values;
- C3FD is not described as a fine-coordinate model;
- incomplete structures do not receive force/energy labels;
- hard collision decisions use only complete lattice and atomic XYZ-triplet
  candidates;
- Llama program and SLA priors are recomputed from the latest committed state;
- model494 drift never revives an illegal semantic action;
- no experiment starts before user approval.
