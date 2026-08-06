# H1 N-conditioned Planner-free DLM Parallel Plan V1

Status: `decision_only_parallel_track_no_execution_authorization`

Date: 2026-08-04

Working name: `H1-NPDLM`

## 0. Decision and amendment scope

This document records a new, independent research plan requested by the user:
train and evaluate an exact-length Body-DLM that does not consume an external
Planner output, while continuing the existing Planner improvement line in
parallel.

The two research tracks are therefore:

| Track | Frozen starting point | First principal treatment | Role |
|---|---|---|---|
| Planner track | `P0 -> B0/R5-C -> model_494` | `P0` versus `P0+CR-Plan` | improve external chemical planning |
| Planner-free track | frozen B0/R5-C parent | same-update rich-Plan continuation versus `N`-conditioned Planner-free DLM | test whether the DLM can learn its own composition plan |

This is a portfolio amendment, not a rewrite of
`ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md`. The old roadmap remains
immutable evidence of the decision available before this user instruction.
The present document supersedes only its single-active-research-track
allocation. It does not alter the frozen H1/R03 results, CR-Plan scientific
contract, common evaluator, or any historical stop.

This plan authorizes no training, generation, refinement, Materials Project
query, checkpoint selection, promotion, or downstream action. Execution
requires a separate, release-ready annex with exact identities, budgets,
source SHAs, run roots, tests, and explicit authorization.

## 1. Precise claim

The first model is not fully unconditional. It is:

> an external-Planner-free, atom-count-conditioned exact-length masked
> diffusion language model.

The atom count `N` remains visible because it fixes the exact semantic length
`7+4N`. The model must generate the element identities, lattice, and
coordinates. It must not receive formula, element list, counts, space group,
lattice hints, symmetry hints, Wyckoff hints, dimensionality, chemical-system
metadata, or any hidden equivalent of the rich Plan.

The useful distinction is:

- **Planner-free**: no separate Planner model or sampled Plan is supplied;
- **not planning-free**: the DLM may learn an internal composition-first
  planning process through its element-token phase;
- **not unconditional**: `N` is an explicit condition and must be disclosed
  in every paper table and claim.

The primary scientific question is whether the special-token DLM can generate
a non-collapsed chemical composition and compatible geometry when only `N`
is fixed.

## 2. Why this is worth running

The current H1 system pre-fills both the count and every element slot from the
rich Plan. The Body-DLM therefore commits only `6+3N` generated semantic
positions: six lattice/angle tokens and `X/Y/Z` for each atom. Composition
validity is largely owned by the Planner, and a composition reward is
constant within a fixed-Plan Body rollout group.

The proposed model keeps the same exact `7+4N` representation but pre-fills
only the count token. Its committed generated-position count becomes
`6+4N`:

```text
N is visible
  -> generate E_1 ... E_N
  -> generate a, b, c, alpha, beta, gamma
  -> generate X_1 ... X_N
  -> generate Y_1 ... Y_N
  -> generate Z_1 ... Z_N
```

This change has three potential benefits:

1. the 94 element special tokens become genuine policy actions instead of
   fixed context;
2. the DLM can learn composition and geometry jointly, which is a stronger
   use of its bidirectional masked-diffusion structure;
3. later element-stage RL can receive non-constant chemistry credit, whereas
   current fixed-Plan Body RL cannot directly improve composition.

These counts describe final committed token positions, not the size of a
mask-aware RL trajectory. A future policy trace must also account for every
candidate-token decision and every reveal-position decision. Under the
previously proposed `K=1` all-candidate construction, candidate draws scale as
`21+2N(N+1)` before reveal terms (`861` at `N=20`), not merely as the `86`
committed positions.

It also creates a useful system-level treatment test. The estimand is the
total effect, under fixed continuation exposure, of removing the rich prompt
and element prefill while adding `N` element-generation positions. A win
supports the feasibility of eliminating external Plan content at fixed `N`;
it does not prove that every external Planner function is unnecessary. A
failure localizes difficulty to the combined conditioning/action change; by
itself it does not prove that an external Planner is theoretically essential.

## 3. What this plan does not claim

The first experiment does not:

- generate `N`;
- remove exact-length conditioning;
- add a chemistry grammar, charge mask, valence mask, repair, rejection
  sampling, retry, filter, or reranker;
- change the continuous refiner;
- use safe-axis, CR-Plan, PILS-L, RL, or a new tokenizer at the same time;
- claim that all 2,481 special tokens are now fully used;
- treat a lower strict S.U.N. unknown rate or a conditional denominator as a
  scientific gain;
- select a checkpoint using generation, Direct, S.U.N., MP, hull, novelty,
  or downstream results.

Generating `N` is a later and separate factor. It would reintroduce the known
count-prior and max-canvas collapse risks and would require a new amendment.

## 4. Registered arms

### 4.1 Required model comparison

| Arm | Model-facing condition | Prefilled answer tokens | Generated answer tokens | Purpose |
|---|---|---|---|---|
| `R_cont` | frozen rich Plan | `N`, all `E_i` | lattice and `X/Y/Z` | same-update trained rich-Plan continuation control |
| `N_only` | constant generic prompt plus visible `N` | `N` only | all `E_i`, lattice and `X/Y/Z` | Planner-free candidate |

Both arms start from the same frozen Body checkpoint and receive the same
number of examples, optimizer updates, effective batch size, corruption
family mixture, checkpoint schedule, and likelihood-only selection rule. They
are not described as equal-budget arms: the prompts have different lengths
and `N_only` predicts `N` additional element positions. Token count, predicted
positions, FLOPs, wall time, and GPU hours must be reported separately.

The untouched historical B0/R5-C is retained as a reported anchor, but it is
not the primary causal control: comparing a newly trained `N_only` model only
against untouched B0 would confound Planner removal with additional training.

### 4.2 Two required evaluation views

One N source cannot support both a perfectly paired ablation and a fully
operational Planner-free claim. The same `N_only` checkpoint is therefore
evaluated in two frozen views:

| View | `N` source | Valid interpretation | Denominator |
|---|---|---|---|
| `N_from_P0` | `N` extracted from one frozen, model-sampled P0 raw Plan ledger | paired Plan-content ablation: remove every Plan field except `N` | report parsed-Plan body-conditional and original all-attempt wrapper separately |
| `N_train_prior` | stateless draw from the frozen train-only MP-20 count prior | operational external-Planner-free N-conditioned generator | its own raw all-attempt ledger; not falsely called paired with P0 |

`N_from_P0` must not be called operational Planner-free because P0 supplied
the count. P0 parse failures remain visible in the end-to-end wrapper and
cannot be dropped to improve either arm. The body-conditional subset is a
secondary mechanistic estimand, never the only denominator.

`N_train_prior` executes no Planner. It is compared with the full H1 system as
a system-level population comparison, not a paired causal contrast. Results
must be reported per `N` and standardized to a frozen count prior where
appropriate; different naturally realized count distributions must not be
hidden by a paired test.

No test structure, target formula, target space group, or target lattice may
construct an inference Plan. Training-only rich Plans may be
structure-derived teacher supervision under the existing provenance firewall;
every generation-time rich Plan must be raw model-sampled P0 evidence.

### 4.3 Optional explanatory arm

An optional, explicitly exploratory `C_only` arm may later receive only
composition information:
`formula/elements/counts/N`, with `N` and elements prefilled. Then:

- `R_cont - C_only` estimates the value of non-composition rich-Plan fields;
- `C_only - N_only` estimates the value of the external composition plan.

`C_only` is not part of the first GPU cycle. Because it is not unconditionally
scheduled before the first outcomes, it is permanently post-hoc exploratory
for this plan and cannot confirm the original `R_cont` versus `N_only` claim.
Any confirmatory three-arm study needs a new preregistration and independent
ledger.

## 5. Frozen representation and decoding contract

### 5.1 Sequence

Every attempt has exactly one semantic token in each position:

```text
[N,
 a, b, c, alpha, beta, gamma,
 E_1, X_1, Y_1, Z_1,
 ...
 E_N, X_N, Y_N, Z_N]
```

The total semantic length is exactly `7+4N`. There is no fixed 87-position
canvas in the primary experiment, no empty-slot generation, and no
post-generation truncation. The serialized answer remains atom-block
interleaved; the reveal schedule below can nevertheless complete all element
positions before revealing lattice and coordinate positions.

Using zero-based answer-relative positions for atom index `i in [0,N)`, the
non-contiguous groups are:

```text
N       = 0
lattice = 1..6
E_i     = 7 + 4i
X_i     = 8 + 4i
Y_i     = 9 + 4i
Z_i     = 10 + 4i
```

Gate −1 must verify from token IDs, not decoded strings, that there is no
answer-internal BOS/EOS, the first token encodes the registered `N`, and
exactly `N` complete `E/X/Y/Z` atom blocks follow.

### 5.2 `N_only` schedule

The count token is visible and frozen at step zero. The registered reveal
groups are:

```text
E -> lattice -> X -> Y -> Z
```

All element positions are completed and frozen before lattice or coordinate
generation. All `X` precede all `Y`, and all `Y` precede all `Z`. No mixed-axis
schedule is allowed. Within each group, the same frozen confidence/reveal rule
and temperature are used in both arms wherever their action spaces overlap.
Random streams for geometry are derived from
`(ordinal, group, step, position)` and are independent of the additional
element-phase draws; merely sharing one global seed is not sufficient
pairing.

The schedule is a composition-first extension of the frozen D1 exact-length
design. It is not the R03 safe-axis candidate, whose refined meta-S.U.N. stop
remains in force.

### 5.3 Legal support

At inference, probability is normalized only over that position's registered
legal special-token family:

- `E_i`: legal element tokens;
- lattice/angle positions: the frozen position-specific numeric families;
- `X/Y/Z`: the frozen coordinate family.

Both arms retain the same frozen generation-time dynamic hard masks:
schema-family support, the zero-length/lattice-volume constraint, and the
duplicate-coordinate constraint. These are online legal-support constraints,
not post-hoc validators and not repairs. Parser, exact-length, and graph
checks remain validators after decoding.

The first SFT experiment retains the current full-vocabulary CE objective for
both arms. It does not silently introduce support-renormalized training CE.
Inference family masks therefore must not be described as the SFT loss
support. Any later support-renormalized SFT would be a new principal factor.

The element phase must not see the target formula through a mask, candidate
set, prompt, sidecar, filename, row ID, batch order, or validator. The legal
element support is the unchanged full registered element family; pure SFT
does not contract it using charge, valence, or target-composition knowledge.

### 5.4 Formula construction

For `N_only`, full and reduced formulas are reconstructed deterministically
from generated element token IDs by canonical counting after the element
phase. They are never copied from training metadata or a Plan and are never
fed back into sampling.

Composition is evaluated for every raw attempt as soon as the count and
element tokens are parseable, even if lattice, coordinate, graph, or refiner
stages later fail. An unparseable composition is false on the original
all-attempt denominator; composition must not be evaluated only on Body or
refiner survivors.

## 6. Leakage firewall

The `N_only` conditioning view may contain only:

```text
constant generic instruction
visible count token N
masked/observed answer state
```

The supervised target answer necessarily exists as the training suffix label.
That is not leakage. Target-derived information is forbidden in the prompt,
immutable conditioning, masks, group construction, sample weights, or any
other field visible before the corresponding target position is legitimately
revealed by corruption. A revealed target suffix token is an allowed
denoising state, not an external condition.

`N` is disclosed through all of its unavoidable channels: the count token,
canvas length, attention length, schedule/group length, and positional
layout. No claim may imply that the count token is its only observable
encoding.

Audit metadata must live in a separate sidecar that the dataloader and model
cannot read. A preflight must audit the final tensors and generation kwargs,
not merely JSON field names, and prove all of the following:

1. prompt bytes are constant across all `N_only` rows;
2. among rows with the same `N` and the same corruption state, conditioning
   `input_ids`, attention, group IDs, immutable-position masks, and generation
   kwargs are byte-identical before target-dependent reveals;
3. no formula, element, composition, space-group, lattice, symmetry, target
   crystal, sample ID, source index, or material identifier occurs in
   conditioning;
4. shuffling or sorting cannot make composition recoverable from batch
   position;
5. target element tokens are masked whenever the element group is trained;
6. `N_only` model records contain no `plan_state`, formula, elements, or
   counts field accessible to the model loader;
7. a dedicated `N_only` validator checks the registered `N`, exact length,
   schema, and graph invariants but never calls
   `validate_answer_matches_plan` or compares with a hidden composition;
8. output and audit identities are joined only after generation by an opaque
   ordinal ledger.

Any leakage mismatch is an engineering stop before model or GPU work.

## 7. Matched SFT contract

### 7.1 Frozen common factors

The execution annex must freeze identical values for:

- parent checkpoint and tokenizer, including all special-token IDs and SHAs;
- MP-20 train/validation/test splits and row order;
- exact answer text and atom ordering;
- LoRA target modules, rank, alpha, dropout, initialization, and seed;
- optimizer, LR, scheduler, precision, global batch, accumulation, and update
  count;
- current token-level CE reduction/normalization, with no chemistry or
  per-family reward weighting;
- corruption-family mixture and semantic streams on overlapping groups;
- validation panel and likelihood-only checkpoint selection;
- sampling temperature, per-group step/NFE policy, batch partition, attempt
  seeds, and denominator;
- downstream `model_494`, exact 800 reverse steps, batch 1, Direct evaluator,
  novelty reference, and strict/meta S.U.N. evaluator.

The exact parent identity must be imported from
`H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md` into the executable
manifest. This planning document intentionally does not invent or shorten a
checkpoint identity.

### 7.2 Corruption coverage

The two immutable-visible sets are different by design:

```text
R_cont immutable-visible = {N, every E_i}
N_only immutable-visible = {N}
```

Both IID and planned input masks and loss masks must exclude each arm's
immutable-visible positions. The current D1 implementation includes an
atom-count group and IID may mask any answer token, so it cannot be reused
unchanged. Existing rich-Plan SFT that predicts or masks element tokens also
does not match `R_cont` inference.

An arm-independent opaque ordinal key, never `hash(prompt+answer)`, drives
corruption. The arms share stateless semantic streams only for overlapping
lattice and `X/Y/Z` groups. Active-group indices are not assumed identical
because `N_only` owns an additional element group. D2/PlanGraph corruption is
forbidden in the first comparison because its target-derived grouping would
add a second factor.

Training must cover the group-level inference regime: `N` visible, all
elements initially masked for `N_only`, and all future groups masked when an
active group is trained. The frozen corruption mixture must include an
explicit fully-masked initial element state rather than relying only on random
partial masking.

Planned corruption provides group-level alignment, not exact exposure to the
confidence-selected within-group states seen at inference. Gate −1 must
compare training and inference state coverage and disclose gaps. The mixture
cannot be tuned after generation results. Per-family NLL is reported for
elements, lattice, and each coordinate axis, but the first experiment does
not introduce adaptive family reweighting. Raw total NLL is not compared
across arms as though the supervised action sets were identical. The report
must include predicted-token counts, geometry-family exposure, loss
contribution, and actual NFE by group.

### 7.3 Count ledger

For the `N_from_P0` view, `R_cont` and `N_only` use the same parsed,
model-sampled P0 count per ordinal. For the operational `N_train_prior` view,
the count is drawn statelessly from a train-only prior and no Planner runs.
No test-set target count, post-selection, retry, or replacement may create
either ledger. Results must include:

- all-attempt aggregate;
- per-`N` completion, composition, structure, joint, strict, and meta;
- the frozen-prior-marginalized aggregate;
- count-frequency and failure-frequency tables.

This prevents an apparent gain caused by sampling easier or unary counts.

## 8. Historical warnings

The repository already contains a true dynamic-v1 constant-prompt path and a
sampler capable of generating `N`, elements, lattice, and coordinates. Its
historical count-generating experiments showed count-prior concentration and
composition/charge-neutrality weakness. Those runs are not the same treatment
as this plan because they generated `N`, used a max canvas, came from a
different training lineage, and did not use the present
controlled-continuation contract.

Conversely, merely invoking the existing
`--no-freeze-plan-composition` switch is not a valid test. That path can still
receive a rich Plan prompt and validate output against Plan composition; it
does not provide same-update no-Plan SFT or a leakage-free evaluator.

The historical evidence therefore gives a warning, not a verdict:

- do not regenerate `N` in the first experiment;
- do not reuse a rich Plan prompt;
- do not interpret unary or all-metal shortcuts as composition success;
- do not advance without an explicit charge-failure and diversity taxonomy.

## 9. Stage gates

No stage starts automatically when the preceding stage passes.

### Gate −1: CPU/read-only contract

Required outputs:

- executable data contract for `R_cont` and `N_only`;
- exact model-facing leakage audit;
- equality of supervised answers, ordinals, `N`, and atom order;
- equality of semantic corruption streams on overlapping lattice and `X/Y/Z`
  groups, plus a separately registered `N_only` element stream;
- proof that every answer is exactly `7+4N`;
- legal-support and schedule unit tests;
- per-token-family train/validation/test coverage;
- dry-run manifest with frozen source paths and no hidden Plan dependency;
- focused reproduction of the historical no-freeze flag showing why it is
  not the proposed method.

Pass condition: every identity and leakage test passes. Any mismatch is an
engineering stop.

Before R0 starts, the execution annex must replace every qualitative gate
word with an exact count, rate, margin, or interval rule; freeze the primary
endpoint and multiplicity policy; freeze one likelihood-only checkpoint rule;
and freeze the R3 design and ledgers. These decisions cannot be deferred until
R1 output exists.

### R0: bounded 32-attempt engineering screen

Purpose: implementation validity, not a scientific claim.

Required:

- one tiny controlled training smoke for each arm with finite losses;
- exactly 32 raw all-attempt ordinals;
- `N` identity and exact `7+4N` length on 32/32;
- element phase applied on 32/32 `N_only` attempts;
- parser/schema success on 32/32, or every failure retained with a new
  engineering stop;
- zero illegal element tokens, target-composition leakage, retry, repair,
  replacement, filter, or rerank;
- shared batch partition and paired stateless seeds;
- no refiner, Direct, S.U.N., MP query, or downstream selection.

### R1: 64-attempt mechanism screen

`N_from_P0` supplies the paired Plan-information contrast.
`N_train_prior` supplies a separately labeled operational view. At minimum:

- 64/64 `N` and exact-length identities;
- `N_only` completion/graph-valid failures no more than two above `R_cont` in
  the paired view;
- no new failure class;
- exact preregistered upper margins for elective unary (`N>1`), all-metal,
  maximum element fraction, duplicate full/reduced formula, and top-1
  full/reduced formula rates;
- unique-formula, element-marginal, arity, and chemical-system coverage
  reported against `R_cont` and frozen train/test reference distributions;
- a primary genuine chemistry endpoint that excludes elective unary and
  all-metal shortcuts and requires the registered charge, Pauling, and
  oxidation-state checks;
- aggregate Direct composition validity as a secondary endpoint;
- complete composition taxonomy: genuine valid, charge failure, Pauling
  failure, oxidation-state missing, all-metal, forced unary at `N=1`,
  elective unary at `N>1`, parse/schema failure, exact/reduced train overlap;
- no checkpoint or threshold selected from these metrics.

The exact collapse margins and coverage floors must already be in the
execution annex before R0. A high parser rate alone is not a pass.

### R2: paired-256 scientific screen

Only after a separately authorized R1 pass:

- use the one likelihood-selected checkpoint per arm frozen before R1; no
  reselection is permitted after R1;
- use exactly 256 raw all-attempt ordinals; `N_from_P0` keeps its paired
  Plan/count ledger, while `N_train_prior` keeps its separate operational
  count ledger;
- preserve every upstream failure;
- pass every Body success through the same frozen CrysLLMGen `model_494`,
  exact 800 reverse steps, batch 1;
- evaluate with the common Direct evaluator and common novelty reference;
- compute strict `E_hull <= 0.0` and meta `E_hull <= 0.1` S.U.N. under the
  frozen common coverage semantics;
- report completion, composition, structure, joint, unique, novel,
  novel-unique, strict, meta, McNemar transitions, confidence intervals, and
  per-`N` effects;
- report pre-refiner to post-refiner transitions so a gain is not
  misattributed to the DLM.

R2 is a preliminary scientific screen, not headline confirmation. The
execution-annex numeric rules for genuine chemistry validity, aggregate
Direct validity, diversity, structure noninferiority, and meta
noninferiority are evaluated before any strict improvement is considered.
There is no qualitative override.

### R3: independent confirmation

The only registered headline confirmation form is four truly independent
256-attempt panels, fixed before R1 and matching the original roadmap's claim
ladder. It runs only if the frozen R2 rule requests confirmation and the user
supplies new authorization. A 1,024-attempt alternative cannot be selected
after seeing R2. Process repeats with the same scientific ledger do not count
as independent scientific samples. Without R3, the result remains
preliminary/appendix evidence.

## 10. Parallel-operation firewall

### 10.1 Portfolio clauses changed and retained

This user-directed amendment changes exactly one existing portfolio clause:

- **superseded for planning only**: the roadmap's rule that only one research
  method may be actively planned at a time.

The following roadmap clauses remain in force:

- CR-Plan remains the only currently allocated ICLR headline method;
- its target/hard compute caps remain ring-fenced and are not available to
  NPDLM;
- PILS-L remains only the registered CR-Plan cold backup;
- RL retains `0 A800 GPUh`;
- the independent-confirmation claim ladder and 2026-09-05 science freeze
  remain unchanged;
- failure of CR-Plan does not automatically promote NPDLM, and failure of
  NPDLM does not promote CR-Plan.

Under this document alone, NPDLM is a parallel
`exploratory/appendix/future-method` plan with `0 authorized GPUh`. Before R0,
a separate global portfolio annex must freeze an additional NPDLM cap, a new
global cap, resource priority, and paper-allocation rule. If that annex does
not exist before R1 outputs, the default result policy is:

| Outcome | Frozen paper allocation |
|---|---|
| CR-Plan passes, NPDLM passes | CR-Plan retains headline slot; NPDLM is appendix/future work |
| CR-Plan passes, NPDLM fails | CR-Plan retains headline slot |
| CR-Plan fails, NPDLM passes | frozen H1 remains headline fallback; NPDLM does not auto-replace it |
| both fail | frozen H1 fallback |

The allocation cannot be changed by selecting whichever R1/R2 endpoint looks
best. A co-main or replacement rule must be authorized and frozen before R1
outputs, not after them.

### 10.2 Execution identities

The Planner and Planner-free tracks may proceed concurrently only under
separate identities:

| Item | Planner track | Planner-free track |
|---|---|---|
| source root | CR-Plan-specific | NPDLM-specific |
| run root | CR-Plan-specific | NPDLM-specific |
| data/ordinal ledger | frozen Planner ledger | frozen NPDLM controlled ledger |
| checkpoint namespace | Planner-only | Body-DLM-only |
| selection | Plan likelihood/mechanism only | Body likelihood only |
| first treatment | decoding support | external Plan removal |

They may share frozen baseline assets and the common downstream evaluator, but
may not:

- tune one track using the other's outcomes;
- combine CR-Plan and `N_only` in the first cycle;
- share a mutable run root, checkpoint directory, or selection report;
- transfer a checkpoint based on Direct/S.U.N. results;
- let failure of one track automatically promote the other;
- consume the other track's predeclared compute budget without a new
  allocation.

The Planner-free Gate −1 can be prepared in parallel with Planner engineering
once execution is explicitly authorized. GPU stages require an independent
budget annex. The proposed time-box is:

| Internal target | Planner-free deliverable | Stop if missed |
|---|---|---|
| 2026-08-06 | Gate −1 contract and tests | no GPU work |
| 2026-08-08 | R0 engineering evidence | no R1 |
| 2026-08-12 | 64-attempt mechanism decision | remove from ICLR critical path |
| 2026-08-20 | paired-256 screen, if authorized | no headline claim |
| 2026-08-31 | independent confirmation, if authorized | preliminary/ablation only |
| 2026-09-05 | common science/table freeze | no new factor |

These dates are planning cutlines, not execution authorization.

## 11. Chemistry and RL follow-ons

The pure same-update controlled SFT comparison comes first. Chemistry
constraints and RL are separate later factors.

### 11.1 Conditional chemistry follow-on

If `N_only` retains diversity and its dominant residual failure is charge
reachability, a second paired experiment may add a prefix charge-reachability
mask only to the element phase. It must use the same checkpoint and all other
frozen factors.

This is a conditional-go follow-on, not part of `N_only`:

- no post-hoc repair or rejection;
- no simultaneous Pauling/space-group/lattice constraint bundle;
- compare mask off versus mask on;
- measure affected rate, viable-support rate, composition validity, diversity,
  joint, strict, and meta;
- require a separate novelty and execution decision relative to CR-Plan.

### 11.2 Conditional RL follow-on

If controlled SFT proves that element generation is viable, the preferred RL
sequence is:

```text
controlled N-only SFT
  -> element-stage RL
  -> joint element + geometry RL
  -> one multi-fidelity LoRA with randomized pre/post-refiner labels
```

The policy factorization is:

```text
pi(C, X | N) = pi_E(C | N) * pi_G(X | N, C)
```

Element actions receive chemistry and expected downstream credit; geometry
actions receive completion, geometry, refiner compatibility, stability, and
diversity credit. Credit must include token probabilities normalized after
schema, zero-length/volume, duplicate, and any registered state-dependent
hard masks, plus reveal-position probability. Current deterministic top-k
remasking does not provide that joint behavior likelihood. A future RL annex
must define an exact Plackett–Luce or equivalent joint trace, exact
replay/resume, and separate element/geometry baselines. Two permanent
pre/post-refiner models are not the default.

RL remains outside the ICLR execution queue until a separate plan reopens it.

## 12. Decision matrix

| Observed result | Interpretation | Action |
|---|---|---|
| `N_only` passes every frozen R2 rule and then the fixed R3 confirmation | evidence that fixed-`N` external-Plan content can be removed under this treatment | apply the paper-allocation rule frozen before R1 |
| geometry/structure rules pass but the primary genuine-chemistry rule fails | useful geometry diagnostic, not a system promotion | scientific stop; analyze element-stage chemistry descriptively |
| any frozen elective-unary/all-metal/top-mode/coverage threshold fails | invalid shortcut or collapse | scientific stop |
| comp improves only after a charge mask | chemistry constraint is the method, not pure no-Plan SFT | register a new factor and claim it honestly |
| `R_cont` passes the frozen paired superiority rule | empirical value for rich Plan plus hard element prefill under the registered total treatment | retain Planner system; do not claim theoretical necessity |
| a frozen decision interval crosses its boundary | inconclusive under the registered budget | retain as preliminary/appendix; no adaptive repeat or new endpoint |

Both a win and a clean failure are publishable evidence. Selective small-panel
reporting, denominator changes, or adding chemistry/RL after seeing failures
would destroy that value.

## 13. Proposed ICLR positioning

Only after fixed R3 confirmation and a pre-R1 paper-allocation amendment, the
strongest defensible claim is not “we removed all planning.” It is:

> Exact-length masked crystal diffusion can internalize composition-first
> behavior in the same special-token sequence and operate without external
> Plan content when atom count is supplied, while preserving a frozen
> continuous refinement and evaluation pipeline.

The most informative paper table would include:

1. untouched H1/B0 historical anchor;
2. same-update `R_cont`, with tokens/FLOPs/GPUh disclosed;
3. `N_only`;
4. optional `C_only` as exploratory unless independently preregistered;
5. current CrysLLMGen and external reported baselines, clearly separated from
   common-evaluator causal comparisons.

If unsuccessful, the experiment still supports the PlanGraph decomposition by
quantifying exactly what the external Planner contributes.

## 14. Required execution annex

Before any action beyond documentation, create one immutable annex containing:

- exact source and run roots that do not already exist;
- source manifest and archive SHAs;
- parent model, tokenizer, data, answer, and split SHAs;
- model-facing and audit-sidecar schemas;
- corruption and count ledgers;
- LoRA/training/sampling configurations and budget caps;
- 32/64/256 panels and seed derivation;
- preregistered gates and noninferiority margins;
- common refiner/evaluator identities;
- Slurm partition validation with `sinfo`;
- submission record and `automatic_downstream=false`.

No stage may be inferred as authorized from this plan alone.

## 15. Governing evidence

- `H1_BODY_DLM_COMPLETE_PROTOCOL_AND_RL_DESIGN_V1.md`
- `H1_BODY_SPECIAL_TOKEN_COVERAGE_AUDIT_V1.md`
- `H1_PLANNER_CHEMISTRY_DLM_RL_FEASIBILITY_REPORT_V1.md`
- `H1_R03_SAFE_AXIS_REPRODUCIBILITY_REPORT_V1.md`
- `ICLR2027_TIMEBOXED_FULL_IMPROVEMENT_ROADMAP_V1.md`
- `crystal_dlm/dynamic_crystal.py`
- `crystal_dlm/r5_dynamic_length.py`
- `crystal_dlm/planned_corruption.py`
- `scripts/build_dynamic_crystal_sft_data.py`
- `scripts/build_h1_formula_only_body_sft_data.py`
- `scripts/sample_llada_dynamic_crystals.py`
- `scripts/sample_llada_r5_exact_length.py`
- `scripts/llada_sft.py`
