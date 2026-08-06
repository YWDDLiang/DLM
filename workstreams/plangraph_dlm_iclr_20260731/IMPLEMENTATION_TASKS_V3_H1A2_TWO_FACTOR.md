# H1-A2 Two-Factor V3 Implementation Tasks

Status: `scientific_training_authorized_packaging`

## Corrected locally

- [x] Separate structure-derived R5-C teacher Plans from model-sampled H1-A2
  Plans.
- [x] Mark the existing sidecar dataset as body-DLM supervision only.
- [x] Restore Planner-model and body-DLM as two independent learned factors.
- [x] Use frozen H1-A2 P0+B0 as the fully-de-novo baseline.
- [x] Initialize DLM candidates from frozen R5-C B0 rather than the raw base.
- [x] Remove shuffle from the registered experiment and executable V2/V3 DLM
  choices.
- [x] Retain D1/B1 as the same-budget continuation control for D2/B2.

## Planner factor — required

- [x] Freeze a genuinely new P* method record that preserves the H1-A2
  seven-line output and changes learned Planner behavior.
- [x] Explicitly exclude the failed JSON PG, ValidReplay, and JointChem
  checkpoints from P* initialization or selection.
  - Result: the frozen method and prepared scientific manifest permit only
    the H1-A2 epoch-2 P0 initialization and enumerate all three candidates as
    ineligible.
- [x] Freeze P* data, loss, update budget, validation panel, and Plan-only
  checkpoint rule before any training.
  - Result: 3,200 frozen microbatches with gradient accumulation 8 give
    exactly 400 optimizer updates, so the Planner budget is already one
    complete epoch.
- [x] Implement and test fail-closed seven-line field spans, reference
  field-balanced loss accounting, look-ahead vocabularies, Plan/body-prompt
  hashes, and rank-independent Planner ordinal seeds.
- [x] Implement and test deterministic common-stream construction with exact
  3,200/256 quotas, content-hash selection, source-SHA fail-close, no
  replacement, and identical P-control/P* row order.
- [x] Materialize and hash those streams from the frozen real H1-A2 source on
  the execution cluster.
- [x] Implement the Torch field-balanced loss and training-only look-ahead
  heads, exact single-pass update/cadence enforcement, and auditable auxiliary
  head checkpoint inventories.
- [x] Numerically validate the Torch objective and full trainer in the
  registered one-A800 32/32 smoke.
- [x] Add tests proving the model still proposes every formula/rich-field
  value and no deterministic renderer, retry, repair, or filter replaces it.
  - Result: strict seven-line raw-output persistence rejects missing/empty
    fields and any extra renderer text; inference provenance must be
    `model_sampled_h1a2_planner`.
- [ ] Run a 512 all-attempt P0/P* Plan-only screen and stop if P* fails.

## DLM factor — required

- [x] Publish the complete 45,229-row teacher-plan sidecar on the execution
  cluster with the provenance fields intact.
  - CPU job 29331 completed `0:0`; all 45,229 rows were published from frozen
    source archive
    `df455438a3b00c6df3eaa54b99df43e95798dafa0d25355b00f985a367dcbde8`.
- [x] Verify prompt/answer byte parity and the real tokenizer at max length
  382.
  - D1 and D2 each passed all 36,183 train+validation rows. Observed maximum
    model lengths were 333 train and 325 validation; terminal report SHA-256
    is `75e9297a28b375cd594d9c55b77c313f13430cff1800b3c8c8287420cdd29ad8`.
- [x] Run the revised 32-row B1/B2 smoke from frozen R5-C adapter B0.
  - Array 29337 completed both arms `0:0`; CPU assembly 29338 exposed a
    JobIDRaw array-parsing false negative. New CPU-only repair job 29345
    completed `0:0`, preserves the failed report, and reruns no GPU work.
- [x] Freeze the DLM learning rate, one-epoch budget, validation cadence, and
  resource envelope after the smoke.
  - Resource envelope is frozen at 2xA800/8 CPU/64 GiB/2h per arm,
    concurrency one, global batch 16.
  - The smoke LR `5e-5` raised fixed-panel validation loss by 7.90% (B1)
    and 6.52% (B2). That risk evidence remains visible; the user explicitly
    restored the historical scientific LR `5e-5`. There is no sweep or
    separate two-update acceptance gate.
  - One complete DLM epoch is 27,136 rows / global batch 16 = 1,696 optimizer
    updates. Validation is frozen at steps
    `0,212,424,636,848,1060,1272,1484,1696` on the same 100-row global panel.
  - The real-ledger panel audit passed: global ordinals 0–99 exactly once,
    report SHA `47c3a591…`, and panel ledger SHA `f4fad132…`.
  - Only the terminal step-1696 checkpoint is eligible. Historical R5-C
    completed the same two-A800 1,696-update epoch in 37m58s.
- [ ] Train B1 and B2 for exactly one complete epoch (1,696 updates), with no
  metric-based early stopping.
- [ ] Select B2 only if it beats both B0 likelihood noninferiority and the B1
  registered dependency margin at the terminal step-1696 checkpoint.

## Factorial runner — required

- [ ] Build one immutable prompt/Planner-seed ledger.
- [x] Freeze the exact H1-A2 prompt, tokenizer/chat-template, seven-line output,
  parser, and model-sampled Plan SHA contract.
- [x] Prove that the body `plan_state`/PlanGraph is compiled from the persisted
  sampled P0/P* Plan and can never fall back to a structure-derived teacher
  Plan.
  - Result: the fresh body runner consumes only the persisted model-sampled
    record; B0 selects current-order `d1`, B* compiles `d2` from that record,
    and teacher provenance fails closed.
- [x] Enforce prompt/answer byte parity, additive tokenization, exact
  `7 + 4*N` body length, and identical tokenizer/vocabulary identities across
  training, validation, checkpoint loading, and inference.
  - Result: sidecar preflight established training/validation byte parity;
    the inference runner requires an expected vocabulary SHA and rechecks
    additive tokenization, decoded-token identity, and exact semantic length
    for every completed attempt.
- [x] Ensure M00/M01 share each realized P0 Plan and M10/M11 share each
  realized P* Plan.
- [x] Freeze per-ordinal body noise, refiner noise, and evaluation order.
  - Result: the fresh Planner/body/refiner runners consume stateless
    per-ordinal seeds and strict ordered rank merges; local and A800 runtime
    suites passed 50/50.
- [-] Preserve all Planner, PlanGraph compilation, body, refiner, and
  evaluator failures in each 256-attempt denominator.
  - Runtime result: Planner, PlanGraph compilation, body, and refiner failures
    are preserved by the strict assembler. Evaluator failure propagation
    remains part of the future frozen G3 evaluator integration.
- [ ] Report M10-M00, M01-M00, M11-M00, and factorial interaction.

## Multi-GPU — component-specific execution contract

- [x] Historical R5-C body-DLM training used NCCL DDP on 2xA800 via
  `torchrun --nproc_per_node=2`.
- [x] Freeze the historical body-training batch contract: per-device batch
  `1`, gradient accumulation `8`, world size `2`, global effective batch `16`.
- [x] Current `scripts/llada_sft.py` retains the NCCL DDP code path.
- [x] Planner/body sampling have rank-sharded code paths.
- [x] Run one targeted 2xA800 integration smoke for the new B1/B2
  planned-corruption path; this is not a generic one-vs-two-GPU parity test.
  - Frozen run root:
    `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260801_h1a2_dlm_b1_b2_2xa800_smoke32_v1`.
  - A800-side targeted tests passed 9/9 before submission; B1 and B2 both
    completed `0:0` and passed their per-arm engineering gates. Corrected
    terminal job 29345 also completed `0:0`.
- [x] Confirm the fixed validation panel has a unique denominator without DDP
  padding duplicates and that only rank 0 publishes checkpoints/reports.
  - Implemented locally with a no-padding rank-strided sampler and exact
    coverage report: 32 -> 16/16; 9,047 -> 4,524/4,523; 5/5 focused tests
    pass. Runtime confirmed exact 16/16 coverage, zero duplicate/missing rows,
    and rank-0-only publication for both B1 and B2.
- [x] Freeze B1/B2 learning rate and update count after that smoke while
  retaining global effective batch `16`.
  - Result: user-directed historical LR `5e-5`, exactly one epoch / 1,696
    updates, validation every 212 updates plus step 0, and
    terminal-checkpoint-only selection. The adverse two-update smoke remains
    part of the evidence.
- [x] Replace rank/batch-dependent inference RNG with stateless per-ordinal
  Planner/body/refiner seeds.
  - Result: historical runners remain untouched; fresh Planner, body, and
    refiner runners consume the frozen ordinal contract and passed A800
    import plus 50/50 targeted tests.
- [ ] Add DDP support for P* Planner training if multi-GPU Planner training is
  actually needed.
- [ ] Freeze world size/rank mapping before any paired multi-GPU inference.

P* Planner training remains single-GPU. B1/B2 body-DLM training uses the
historical 2xA800 contract after its targeted new-path smoke. Paired generation
is gated and frozen separately.

## Submission gate

No job may be submitted until:

1. the P* method record is frozen and implemented;
2. the B1/B2 revised smoke passes from the frozen R5-C adapter;
3. the I/O alignment and factorial pairing tests pass;
4. fresh source, execution, authorization, run-root, and submission records
   are created;
5. H1 fallback assets remain read-only.

There is no automatic refiner experiment, G4, retry, repair, replacement,
filtering, reranking, self-training, or energy/S.U.N-guided selection.

Prepared scientific manifests are indexed at
`execution/v3_scientific_training_drafts_v1/DRAFT_MANIFEST_INDEX.json`.
Explicit scientific-training authorization is recorded in
`AUTHORIZATION_V3_SCIENTIFIC_TRAINING_20260801.json`; the DLM LR override is
recorded in `PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`.
Packaging, focused validation, fresh immutable bundles, and submission
records remain before launch. The exact 100-row panel audit is complete, and
the superseded two-update LR recheck will not be run.
