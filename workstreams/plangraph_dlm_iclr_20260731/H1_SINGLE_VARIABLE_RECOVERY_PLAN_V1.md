# H1 Single-Variable Recovery Plan V1

State: `active_real_ledger_preflight`

Date: 2026-08-02

This plan supersedes combined Planner-plus-DLM execution after the V8
diagnostic. Historical H1-A2, canonical R5-C, and the frozen CrysLLMGen
refiner remain read-only.

## Governing rule

Every scientific comparison starts from an exact H1 control. A stage may
change exactly one principal factor:

1. Planner checkpoint or Planner method;
2. body-DLM checkpoint;
3. body decoding schedule; or
4. evaluation coverage snapshot.

No stage may change two of these together. Seeds, ordinal ledger, batching
semantics, tokenizer, body representation, refiner checkpoint, 800 reverse
steps, denominator, and evaluation cache remain fixed unless the stage exists
solely to validate one of those engineering identities.

## Ordered stages

### R0 — H1 control identity

- Reuse the immutable H1 P0 Planner outputs and attempt ledger.
- Reuse the immutable R5-C body adapter, exact-plan body schedule, body batch
  size 8, original paired body noise, original refiner noise, parent batch
  size 64, frozen `model_494`, and exactly 800 reverse steps.
- Re-run P0 only into a fresh output root.
- Require byte-identical body attempts and semantically identical proposal
  tensors relative to the frozen H1 paired-256 result.
- Do not require byte-identical continuous refined coordinates. The retained
  failed byte-parity replay proved that the frozen CUDA refiner is not
  bitwise reproducible even on the same A800 node. Apply
  `H1_REFINER_REPRODUCIBILITY_AMENDMENT_V1.md` before any refined metric.
- Do not run direct metrics or S.U.N.

### R1 — Plan identity engineering repair

- Preserve raw model text and its SHA independently from canonical parser text
  and its SHA.
- Exact seven-line formatting is advisory after the frozen parser has accepted
  an ordinal.
- Require every parser-accepted real-ledger ordinal to reach body compilation.
- This is an engineering gate and cannot change a scientific metric or model.

### R2 — body schedule isolation

- Hold P0, B0 weights, H1 ordinals, noise, tokenizer, refiner, and cache fixed.
- Compare only H1 exact-plan scheduling against one corrected PlanGraph
  scheduling implementation on 32 or 64 attempts.
- Require zero excess duplicate-coordinate failures before any 256-attempt
  run.

### R3 — body checkpoint isolation

- Run only after R2 proves the corrected schedule safe.
- Cross the frozen B0 and candidate body checkpoint under one common safe
  schedule; do not change Planner.

### R4 — Planner isolation

- Hold B0, H1 body schedule, noise, refiner, and cache fixed.
- Compare P0 with one Planner candidate only after its independent Plan-only
  gate passes.

### R5 — composition

- Combine Planner and body improvements only after both have independently
  passed their paired controls.
- A combined run is confirmatory; it is never used to rescue a failed
  single-factor result.

## Stop rules

- Any unexplained discrete baseline replay mismatch stops downstream work.
- Continuous refined-output differences require the registered repeatability
  protocol; they may not be accepted through a post-hoc numeric tolerance.
- Any new failure class or more than two percentage points of completion loss
  stops expansion beyond the small screen.
- S.U.N. comparisons require one common completed frozen cache/API snapshot.
- Raw all-attempt results remain primary; coverage-adjusted values are
  descriptive only.
