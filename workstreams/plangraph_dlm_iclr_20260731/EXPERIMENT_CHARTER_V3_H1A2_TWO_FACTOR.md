# H1-A2 Planner + DLM Two-Factor Charter V3

Status: `pstar_method_frozen_local_implementation_in_progress`

Correction date: 2026-08-01

This charter supersedes V2 for every future execution. No V3 job is
authorized or submitted.

## 1. Scientific correction

R5-C does not provide a learned, fully-de-novo Planner baseline. Its
conditional training/evaluation Plan is derived from the target structure.
It is valid body-DLM supervision and an oracle-like conditional upper bound,
but it is not evidence that a model proposed a useful Plan.

The end-to-end baseline is instead:

```text
goal/prompt
  -> frozen H1-A2 epoch-2 Planner samples one seven-line Plan
  -> frozen R5-C exact-length body DLM realizes that sampled Plan
  -> frozen CrysLLMGen model_494 refines the body
  -> raw all-attempt evaluation
```

The new work has two distinct learned factors:

1. improve the model-generated H1-A2 Planner while preserving its seven-line
   language and one-shot sampling contract;
2. improve the R5-C body DLM with dependency-aligned corruption and denoising.

The two effects must be measured separately before making a joint claim.

## 2. Hypotheses

### H-P — Planner-model improvement

An H1-A2-compatible Planner candidate can improve composition and joint-Plan
validity without changing the seven-line output schema, replacing sampled
values deterministically, or gaming the chemistry distribution.

### H-D — DLM improvement

Starting from the frozen R5-C body adapter, a short mixed iid/planned
continuation with a matched inference schedule can improve body realization
for model-sampled Plans while preserving likelihood and completion.

### H-J — joint effect

The selected Planner and DLM improvements should compose. Their individual
effects and factorial interaction are reported; a strong negative interaction
is a scientific failure, not something to hide with checkpoint reselection.

## 3. Frozen H1-A2 baseline

### Planner P0

- Meta-Llama-3-8B base;
- H1-A2 epoch-2 adapter SHA-256
  `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`;
- exact seven-line output:

```text
formula: ...
anion: ...
charge: ...
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

- official sampler only;
- temperature `0.9`, top-p `0.95`, top-k `50`, max-new-tokens `96`;
- no sample ID, retry, replacement, repair, filtering, or reranking.

### Body B0

- frozen LLaDA-8B-Instruct base identity;
- frozen R5-C exact-length adapter SHA-256
  `5c39976b6ab237cbab32cbfeb1c23a557571e1c7d2b60c1e60cbb450166ae76d`;
- `dynamic_v1`, exact answer length `7 + 4*N`, maximum `87`;
- frozen H1 exact-Plan generation schedule and legal-token masks.

### Refiner and evaluator

- frozen CrysLLMGen `model_494`;
- exact 800 reverse steps;
- original direct metrics and frozen S.U.N. implementation/snapshot;
- no energy or S.U.N. signal before output and checkpoint freeze.

H1-A2 P0+B0 is the immutable fully-de-novo fallback and the factorial
reference. It is not replaced by a newly trained approximation.

## 4. Plan provenance firewall

Two Plan sources have different scientific meanings:

| Plan source | Permitted use | Prohibited interpretation |
|---|---|---|
| structure-derived R5-C teacher `plan_state` | body-DLM training, tokenizer/NLL/mask preflight | learned Planner result or end-to-end headline |
| sampled P0/P* seven-line Plan | fully-de-novo body generation and factorial evaluation | teacher/oracle Plan |

The current H1-A2 sidecar builder explicitly labels its Plan provenance as
`structure_derived_teacher_plan_state` and marks it ineligible as end-to-end
Planner evidence.

At evaluation, PlanGraph must be compiled only from the realized sampled
seven-line Plan. If that Plan cannot be parsed or compiled, the attempt fails
and stays in the denominator.

### End-to-end input/output alignment gate

Alignment is a scientific identity requirement, not a formatting cleanup:

- P0 and P* use the same frozen H1-A2 system/user prompt template, tokenizer,
  chat template, seven-line output schema, and parser;
- the raw model-sampled seven-line Plan is persisted once with a SHA-256
  identity before body generation;
- the body `plan_state` and non-visible PlanGraph are compiled only from that
  persisted sampled Plan; a structure-derived teacher Plan may never replace
  it at inference;
- within M00/M01 and M10/M11, the exact body prompt bytes and realized Plan
  SHA-256 are identical;
- B0/B1/B2 training preserves the original R5-C prompt and answer bytes, and
  tokenization must satisfy
  `tokenize(prompt + answer) == tokenize(prompt) + tokenize(answer)`;
- every dynamic-v1 answer has exactly `7 + 4*N` semantic/token positions and
  the tokenizer/vocabulary identity is the same for training, validation,
  checkpoint loading, and inference;
- body output-to-refiner conversion is identical for all four factorial arms.

Any byte, tokenizer, schema, Plan identity, answer-length, or conversion
mismatch is a fatal preflight failure. It may not be repaired by substituting
another row or Plan.

## 5. Planner factor

### P0 — frozen H1-A2 epoch 2

No training. P0 supplies model-proposed baseline Plans.

### P* — H1-A2 Look-Ahead Consistent Planner v1

P* must:

- initialize from the frozen epoch-2 adapter;
- retain the identical seven-line model-visible schema and official sampler;
- change learned Planner behavior, not merely post-process or replace formula
  fields;
- use no generated-crystal, S.U.N., energy, hull, MLIP, or MP-API outcome;
- expose every failed parse/chemistry attempt without retry or replacement;
- freeze its data, loss, update budget, and checkpoint rule before training.

The frozen method record is `PLANNER_PSTAR_METHOD_V1.md`. It uses a
field-balanced target-only loss plus training-only look-ahead heads that
predict later seven-line categories from causal formula/lattice boundaries.
The heads are discarded for inference, so the model still samples the exact
same visible seven-line schema without repair or constraints. P-control uses
the same stream, budget, and field-balanced loss without look-ahead.

The failed generated-JSON PG checkpoint and the completed
ValidReplay/JointChem `stop_no_plan_candidate` result are immutable negative
evidence and are not eligible as P*. The new method is frozen but not yet fully
implemented or A800-smoke validated, so the Planner factor is not ready for
scientific execution.

Planner-only selection uses a common 512-ordinal prompt/seed ledger and no
crystal generation. P* is eligible only if:

- parse/completion is no more than `0.5` percentage points below P0;
- composition-valid improves by at least `2` percentage points;
- validation Plan NLL is within `+1%` of P0;
- unique-formula rate is at least `95%` of P0;
- mean atom count differs by at most `0.5`;
- no registered marginal TVD worsens by more than `0.02`;
- all-metal and single-element shortcut rates do not inflate.

## 6. DLM factor

The DLM screen starts from B0, not the raw LLaDA base.

| Body arm | Initialization | Training corruption | Inference schedule | Role |
|---|---|---|---|---|
| B0 | frozen R5-C adapter | none | frozen H1 exact-Plan | baseline |
| B1 | B0 | iid:current-order = 2:1 | frozen H1 exact-Plan | same-budget continuation control |
| B2 | B0 | iid:compiled-PlanGraph = 2:1 | compiled PlanGraph | proposed DLM method |

There is no shuffle arm.

B1 and B2 use identical ordered teacher rows, prompt/answer bytes, optimizer
budget, and validation panel. The structure-derived teacher PlanGraph is
non-visible sidecar metadata. It does not turn the training row into a
model-proposed Plan.

The body-DLM screen runs one complete epoch. With 27,136 frozen training rows,
two ranks, per-device batch `1`, and gradient accumulation `8`, the global
effective batch is `16` and one epoch is exactly `1,696` optimizer updates.
There is no metric-based early stopping. Both arms use the single
user-directed historical learning rate `5e-5`; there is no sweep and the
earlier proposed two-update learning-rate recheck is superseded. The adverse
two-update smoke at `5e-5` remains part of the evidence and is not relabeled.

Validation runs before training and at eight equal one-eighth-epoch intervals:
steps `0, 212, 424, 636, 848, 1060, 1272, 1484, 1696`. Each rank evaluates
the first 50 rows of its frozen no-padding rank-strided validation shard,
giving one common 100-row global panel. Only the terminal step-1696 checkpoint
is eligible for the registered B1/B2 comparison. This amendment is frozen in
`PROTOCOL_AMENDMENT_V3_DLM_ONE_EPOCH_20260801.json`; the learning-rate field is
superseded by
`PROTOCOL_OVERRIDE_V3_DLM_LR5E5_AUTHORIZED_20260801.json`.

B2 becomes B* only if:

- all losses and gradients are finite;
- fixed-panel NLL is within `+1%` of B0 initialization;
- conditional body completion is no worse than B0 by more than `1` point;
- the matched PlanGraph dependency margin is positive;
- B2 strictly beats B1 on the same registered margin;
- checkpoint selection uses no generation, S.U.N., energy, or hull outcome.

If B2 fails, there is no claimed DLM improvement. B1 is retained only as a
control and is not silently promoted to the PlanGraph method.

## 7. Confirmatory 2x2

Only after P* and B* independently pass:

| Arm | Model-generated Planner | Body DLM | Frozen refiner |
|---|---|---|---|
| M00 | P0 | B0 | model_494 |
| M10 | P* | B0 | model_494 |
| M01 | P0 | B* | model_494 |
| M11 | P* | B* | model_494 |

Interpretation:

- Planner effect: `M10 - M00`;
- DLM effect: `M01 - M00`;
- joint effect: `M11 - M00`;
- interaction: `(M11 - M01) - (M10 - M00)`.

Pairing rules:

- M00/M01 share the exact realized P0 Plan for each ordinal;
- M10/M11 share the exact realized P* Plan for each ordinal;
- P0 and P* share prompt identity and Planner seed, but their sampled Plans
  may differ because that difference is the Planner treatment;
- within each Planner pair, B0/B* share body noise, refiner noise, and
  evaluation order;
- every arm uses exactly 256 registered attempts and raw all-attempt
  denominators.

R5-C conditional gold-Plan results are reported only as an oracle/body upper
bound and are not a fifth arm.

## 8. Paired-256 gates

- all arms retain 256/256 attempt identities;
- generation completion is at least `97%`;
- no arm loses more than `2` structure-validity points versus M00;
- M10 establishes the registered Planner composition-valid gain;
- M01 establishes a positive DLM direct effect without changing the Planner;
- M11 beats M00 on joint validity;
- M11 strict S.U.N. improves by at least `1` point or exceeds `10%`;
- M11 meta S.U.N. improves by at least `3` points or exceeds `50%`;
- M11 beats at least one single-factor arm and interaction is not strongly
  negative;
- likelihood and distribution gates remain satisfied.

The report must decompose earliest failures into Planner parse/chemistry,
PlanGraph compilation, body generation, structure/refiner, and evaluator
failures.

## 9. Distributed execution status

Multi-GPU status is component-specific. Body-DLM training already has a
historical two-GPU contract; Planner training and paired inference remain
separate questions:

| Component | Established code/history | V3 registered status |
|---|---|---|
| P* Planner training | existing H1-A2 continuation scripts are single-GPU | single-GPU |
| H1-A2 Planner sampling | rank-sharded distributed inference exists | paired use pending stateless per-ordinal RNG |
| B1/B2 body-DLM training | canonical R5-C launcher uses NCCL DDP with `torchrun --nproc_per_node=2` on 2xA800 | reuse the historical two-GPU contract after a targeted planned-corruption smoke |
| B0/B* body sampling | rank-sharded distributed inference exists | paired-256 use pending frozen per-ordinal noise/mapping |

The historical R5-C body-DLM launcher freezes per-device batch `1`, gradient
accumulation `8`, world size `2`, and therefore global effective batch `16`.
Its canonical learning rate is `5e-5`; the completed two-update B1/B2 smoke
showed immediate fixed-panel degradation at that value. The user explicitly
restored that historical `5e-5` value for V3 scientific training despite the
smoke risk; it is frozen once without a sweep, and the scientific decision is
deferred until both arms complete all `1,696` updates.
V3 does not require a generic one-GPU-versus-two-GPU parity proof and does not
downgrade body-DLM training to one GPU.

Before an official B1/B2 run, the new planned-corruption path needs one
targeted two-GPU integration smoke confirming:

- both ranks consume the registered corruption policy and finite losses;
- global effective batch remains `16`, with the frozen accumulation and
  update-count interpretation;
- the fixed validation panel has a unique denominator without distributed
  padding duplicates;
- checkpoint/report publication is rank-0-only after successful barriers.

Paired multi-GPU inference is a different gate. It still requires stateless
per-ordinal Planner/body/refiner seeds derived from
`(frozen_seed, sample_idx, stage)`, plus identical world size,
rank-to-ordinal mapping, and merge order across compared arms. Until that
inference gate passes, paired generation uses a separately frozen execution
layout even though B1/B2 training uses 2xA800.

## 10. Stop rules

- no JSON Planner or model-visible PlanGraph text;
- no shuffle arm;
- no use of R5-C teacher Plans as learned Planner evidence;
- no raw-base DLM retraining presented as the H1-A2 baseline;
- no reuse of failed PG, ValidReplay, or JointChem checkpoints as P*;
- no adaptive threshold, seed, denominator, data, or checkpoint changes;
- no retry, repair, replacement, filtering, reranking, or self-training;
- no refiner modification or automatic G4;
- no V3 submission until the Planner method and DLM continuation manifest are
  both frozen.

## 11. Paper claim if successful

> A model-generated H1-A2-compatible Plan and a dependency-aligned diffusion
> language model provide separable and composable improvements in fully
> de-novo crystal generation, without search, repair, or energy supervision.
