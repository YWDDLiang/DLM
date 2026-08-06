# H1-A2-Aligned V2 Implementation Tasks

Status: `local_implementation_in_progress_no_submission`

## Completed locally

- [x] Retire generated PlanGraph JSON from future execution.
- [x] Keep the H1-A2 seven-line Planner and official sampler unchanged.
- [x] Add an atomic R5-C prompt/answer-preserving sidecar builder.
- [x] Strip source metadata, IDs, energy, stability, and S.U.N. fields.
- [x] Remove D2-shuffle from the registered experiment and executable choices.
- [x] Use D1 as the same-budget planned-order control for D2.
- [x] Add matched H1/D1 and D2 inference schedule builders.
- [x] Keep `sample_llada_r5_exact_length.py` default behavior at H1 exact-plan.
- [x] Add unit tests for byte parity, leakage, denominator retention,
  cross-split overlap, and rejection of the removed shuffle policy.

## Required before any Slurm submission

- [ ] Build the complete sidecar dataset on the execution cluster from the
  frozen R5-C train/val/test JSONL.
- [ ] Verify 45,229/45,229 rows and every model-visible ordered SHA.
- [ ] Run the real LLaDA tokenizer preflight at max length 382.
- [ ] Re-run the 32-row A800 engineering smoke because V2 uses the H1-A2
  prompt and matched D1/D2 schedules rather than the v1 JSON representation.
- [ ] Freeze a paired H1 body runner that consumes the registered per-ordinal
  noise bank with each arm's registered inference schedule; do not use the
  ordinary unpaired sampler for paired-256.
- [ ] Freeze a new source bundle, execution manifest, authorization record,
  run root, and submission record.
- [ ] Confirm two-A800 DDP produces the expected 1,696 updates.

## Scientific sequence

1. Train D0 for one full R5-C-equivalent epoch.
2. Require D0 conditional-body parity before interpreting candidate arms.
3. Train D1 and D2 with identical rows and training budget.
4. Use H1 exact-plan inference for D0/D1, matched PlanGraph inference for D2,
   and compare D2 directly with the same-budget D1 control.
5. Select no checkpoint using S.U.N., energy, or generation outcomes.
6. Run paired-256 on the frozen H1 P0 plan/noise ledgers only after G1 passes.
7. Stop after paired-256; no automatic refiner experiment or G4.

## Explicitly prohibited

- generated PlanGraph Planner;
- reuse of v1 PG/PG-shuffle checkpoints;
- PlanGraph JSON inside a model-visible prompt;
- any schedule key derived from the target body answer;
- max length 768 inherited from the failed JSON representation;
- H1 path mutation;
- adaptive threshold, seed, denominator, data, or checkpoint changes.
