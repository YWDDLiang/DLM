# Execution Checklist: B-First Scientific Programmed Denoising

Status: **EXECUTION APPROVED — tokenizer/checkpoint audit precedes GPU training**

Authoritative method update:
[`06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md`](06_MODULE_AUDIT_AND_B_FIRST_PIVOT.md).

## Phase 0 — workspace and review

- [x] Preserve the original dirty checkout.
- [x] Create clean local branch/worktree
  `codex/unified-scientific-decoding` at
  `D:\codex_work\ai4s\DLM_unified_scientific_decoding`.
- [x] Push the branch from the local machine.
- [x] Attach `starteam5090 → tmux ssha800`; keep the session alive.
- [x] Create clean A800 HTTPS clone
  `/public/home/jiaosz/ywliang/ai4s/.sscd_unified_d0e616b_https`.
- [x] Complete design, skeptic, resource, reader and arbiter review.
- [x] Receive explicit user approval.
- [x] Update ten-minute heartbeat `sscd-a-b-approval-and-execution`.

## Phase 1 — complete special-token and checkpoint audit

- [x] Source inventory: 2,481 crystal special-token strings; 2,457 used by
  dynamic `7+4N`.
- [x] Real step-3392 boundary probes: N/lattice/angle/Pu/XYZ 000/100 each
  encode as one token; tokenizer size 128,830; pad/eos 126081 differs from mask
  126336.
- [ ] Add one reusable audit command and tests.
- [ ] On the real tokenizer, verify all 2,481 tokens exist, encode atomically,
  decode consistently and use unique IDs.
- [ ] Inspect adapter safetensors without loading the 6 GiB tensor; verify input
  embedding/output-head saved shapes cover all 128,830 IDs.
- [ ] Parse all MP20 train 27,136 and validation 9,047 rows and report:
  N range, element coverage, length/angle/coordinate ranges, clipping, zero
  lengths, coordinate-100 frequency, quantized duplicates and invalid lattices.
- [ ] Verify every body has semantic and tokenizer length exactly `7+4N`.
- [ ] Audit teacher and frozen-predicted Plan prompt + body against max length
  382.
- [ ] Verify production uses the exact-length sampler; mark the universal
  87-position EOS-tail sampler historical-only.
- [ ] Save one concise JSON/CSV/Markdown audit and update this checklist.

## Phase 2 — core SPAD interfaces

### 2.1 Planner program

- [ ] Implement `program_from_planner_trace(plan, semantic_trace)`.
- [ ] Fold oxidation-state variants of one element while preserving first
  selected element order.
- [ ] Require `species_program` to be an exact permutation of unique Plan
  elements and preserve counts/N.
- [ ] For MP20 teacher compositions, derive a program by constrained replay of
  the frozen Planner over only the remaining target species/count actions.
- [ ] Store the program as metadata; do not let canonical element sorting erase
  it and do not add fake DLM tokens for it.

### 2.2 Exact DLM canvas

- [ ] Make strict dynamic parsing reject non-whitespace text outside schema
  tokens.
- [ ] Validate N in every dynamic length helper.
- [ ] Require generation schedules to cover each `7+4N` position exactly once.
- [ ] Resolve the actual tokenizer mask ID and verify it differs from
  pad/eos/bos/unk/crystal IDs and fits input/output vocab rows.
- [ ] Exclude zero-length tokens from production support.
- [ ] Aggregate coordinate 000/100 logits as one torus action and emit canonical
  000 for new commits.

### 2.3 Stateful bidirectional sampler

- [ ] Refactor the current sampler into
  `initialize_canvas / constrained_forward / commit / remask / resume`.
- [ ] Compile Planner species order into non-contiguous native DLM positions.
- [ ] Implement anchor-first prediction: lattice, one anchor site per species,
  remaining sites.
- [ ] Implement one suffix-visible remask/backfill sweep over early anchors in
  reverse program order.
- [ ] Preserve all non-active suffix tokens exactly during backfill.
- [ ] Keep the old anchor triplet as an explicit no-op candidate.
- [ ] Refresh the full DLM forward after every committed transaction.

### 2.4 Geometry

- [ ] Commit the complete six-value lattice only after a valid Gram
  determinant.
- [ ] Treat XYZ as one site transaction; do not commit X/Y before a legal Z
  completion exists.
- [ ] Replace duplicate-bin-only masking with a validated triclinic PBC
  minimum-distance check at the 0.5 Å boundary.
- [ ] Preserve the old committed coordinate as provisional geometry during a
  remask; never use an unrestricted soft mean that can create a ghost site.

## Phase 3 — tests proving DLM necessity

- [ ] All 2,481 token atomicity and checkpoint reload tests.
- [ ] Future-first step map: a later storage position commits before an earlier
  one under the Planner program.
- [ ] Suffix dependency: modifying a visible later site changes logits of an
  earlier masked anchor on a real context-sensitive model.
- [ ] Remask invariant: only the selected XYZ block changes; later suffix,
  exact N and elements remain unchanged.
- [ ] Program coverage: every dynamic position is resolved once in predictor
  and only registered anchors are revisited.
- [ ] Train/serve mask parity for program predictor and correction states.
- [ ] Triclinic/000–100/0.5 Å/near-singular boundary tests.

## Phase 4 — B-route data and training

Track B is the priority route.

- [ ] Reuse frozen C3FD–Llama Planner and current Compact-V2 DLM.
- [ ] Build full MP20 teacher and frozen-predicted same-schema Plan views;
  answers are identical and combined source weight is one.
- [ ] Add `species_program` and two mask classes:
  program-matched predictor state and complete-state anchor-remask state.
- [ ] Retain ordinary random-mask examples so general denoising is not erased.
- [ ] Start with decoder-only cells before training.
- [ ] Train one BS LoRA from the retained Compact-V2 endpoint:
  r8/alpha32/dropout0.05, LR 5e-6, effective source batch16, exactly 1,696
  updates, one model seed, only the endpoint eligible.
- [ ] Use up to 4 A800 and 32 CPU for B; no CHGNet/model494/test outcome enters
  training.

## Phase 5 — adjacent B experiments

Freeze one 256-request predicted-Plan/program ledger and two common sampling
streams.

| Cell | Intervention |
|---|---|
| BC | retained DLM, canonical monotone schedule |
| BP | same weights, Planner-program anchor-first schedule |
| BR | BP + one suffix-visible anchor-remask sweep |
| BS | BR after one schedule-matched MP20 LoRA epoch |

- [ ] One Plan and one trajectory per request; no replacement, reranking or
  best-of-N.
- [ ] Run all body generation cells with exact N/elements prefilled.
- [ ] First report body, proposal composition validity, body composition
  retention, raw Direct, graphability, minimum-distance ECDF, collision type,
  lattice volume/condition and backfill changes.
- [ ] Run BR-no-suffix only on a shared mechanism subset to prove the value of
  visible future context.
- [ ] Deduplicate the structure union before energy evaluation.
- [ ] Benchmark CHGNet 4/8/10 one-thread workers within 8 allocated CPU cores
  per GPU; run Direct concurrently only on explicitly allocated CPUs.
- [ ] Report two streams separately and pooled; do not choose a stream.

## Phase 6 — Track A system comparison

Track A no longer blocks B.

- [ ] Build one thin Plan/program-conditioned CrysLLMGen AR dataset/trainer.
- [ ] Train or adapt one LLM-only body endpoint on up to 2 A800 while B
  evaluation/training occupies up to 4 A800.
- [ ] Use the same predicted Plan/program ledger and report raw A versus BC/BP/
  BR/BS.
- [ ] Keep AR native text separate from DLM special tokens.

## Phase 7 — Candidate E1 stability feedback

Only after the SPAD result is frozen:

- [ ] Use 1,024 BS-generated MP20-train states and 256 validation states.
- [ ] Test whether the deployed continuous refiner's one-step response or
  CHGNet force can supervise a DLM geometry residual/confidence module.
- [ ] Keep force/energy undefined on incomplete or ungraphable structures.
- [ ] Compare an equal-compute zero-response remask with active feedback.
- [ ] Require raw geometry and held-out stability improvement before terminal
  full refinement.
- [ ] If this path is weak, retain it as a negative and keep SPAD as the core.

## Phase 8 — final evidence

- [ ] Apply the same terminal continuous-refiner setting to frozen A and B
  cells.
- [ ] Report raw/refined endpoints separately.
- [ ] Report Strict/Meta S.U.N., N/U/NU, Direct, composition retention and
  compute on the fixed requested denominator.
- [ ] Evaluate the target `Strict S.U.N. >10%` and
  `Meta S.U.N. >50%`; targets never authorize row/seed/checkpoint selection.
- [ ] Treat CHGNet and MP-reference/CHGNet-candidate hull as surrogate metrics;
  use an independent MLIP or registered DFT subset for stronger claims.
- [ ] Update README, paper story, method diagram and contribution table.

## Resource and iteration policy

- maximum 6 A800, 2 concurrent jobs and 8 CPU per GPU;
- B receives up to 4 A800; A/evaluation receives up to 2;
- simple implementation/testing/monitoring stays in the main task; complex
  bounded audits use subagents;
- small engineering failures receive one minimal repair and one root-cause
  record;
- scientific negatives remain visible and may motivate one adjacent method
  revision, not a sweep;
- small positives are checked on the second common stream;
- large positives are frozen, verified and then celebrated.
