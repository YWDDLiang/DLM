# Second-Contribution Review: Formula, DLM Credit, and Composition Correctness

Date: 2026-08-28

Disposition: **APPROVED for CCFD Phase 0 -> Phase 1 only.** Phase 2 tokenizer
training is conditional. RL is not authorized.

## Understanding lock

The objective is one relatively small, defensible second contribution with a
clear solved problem. Rich Plan is optional. Stability/S.U.N. remains the main
system metric, but a composition-correctness contribution must not be presented
as a stability method. Formula, body, refiner and evaluator effects remain
causally separated. Public `105/488` is unchanged.

## 1. Should rich Plan be removed and only formula generated?

### The experiment already exists

The historical H1-B route trained a formula-only Planner and a formula-only
exact-length body DLM. It is therefore a direct test of this proposal, not just
an analogy.

| Result | H1-A rich Plan | H1-B formula only | CrysLLMGen reference |
|---|---:|---:|---:|
| body graph acceptance | 99.21% | 100.00% | 97.25% |
| refined composition valid | 82.87% | 85.88% | 90.70% |
| refined structural valid | 100.00% | 100.00% | 99.80% |
| coverage recall | 78.28% | 78.18% | 96.69% |
| Strict S.U.N. adjusted | 5.29% | 5.54% | 9.31% |
| Meta S.U.N. adjusted | 43.17% | 43.13% | 47.67% |

Formula-only improved execution and composition modestly but did not improve
Meta stability. Its geometry collapsed toward dataset-average templates:

- all angles 90 degrees: `81.57%` versus `33.47%` for rich Plan;
- any two lattice lengths equal: `53.33%` versus `21.51%`;
- graph validity stayed high, so this was not a parser failure.

The cause is structural underdetermination:

```text
formula -> many possible lattices, space groups, prototypes and coordinate basins
```

Plain token CE responds by producing frequent orthogonal/equal-length
geometries. model494 can repair local validity but cannot reconstruct the
missing structural mode reliably.

### Decision

- Do not rerun formula-only as a new contribution or expected stability fix.
- Rich natural-language Plan can be removed, but the executor still needs a
  minimal structural mode if formula-only remains the interface.
- Such a mode must be frozen before results and evaluated separately from the
  composition method; it is not part of the composition-correctness claim.
- The active hard-anchor experiment is still informative because it uses the
  stronger public checkpoint and exact paired seeds, but it cannot erase the
  historical H1-B result.

## 2. Is joint XYZ the same as safe-axis/H1-A2?

No. It is the opposite.

### Frozen schedules

H1-A2 D1:

```text
composition -> lattice -> all X -> all Y -> all Z
```

R03 safe-axis:

```text
composition -> lattice
            -> PlanGraph-grouped X blocks
            -> PlanGraph-grouped Y blocks
            -> PlanGraph-grouped Z blocks
```

Both enforce the invariant:

```text
every X/Y group strictly precedes every Z group
```

The duplicate-coordinate mask constrains candidate Z only after X and Y are
known. The mixed-XYZ schedule allows Z first; a later X/Y action can then create
a duplicate without passing through the guard.

### Historical and new evidence

Historical paired-32:

- D1 body success `31/32`;
- mixed-axis D2 `14/32`;
- 18 new duplicate-coordinate failures.

The 2026-08-28 repeat reproduced the same mechanism:

| Seed | full-axis | full-joint | hard-axis | hard-joint |
|---|---:|---:|---:|---:|
| 17 | 254/256 | 158/256 | 256/256 | 213/256 |
| 18 | 251/256 | 161/256 | 256/256 | 204/256 |

`full_joint` contained `97/95` duplicate failures and `hard_joint` `43/52`.
The proposed atom-major follow-up was cancelled before execution because an
early atom's Z would still precede later atoms' X/Y. Job `35556` was cancelled
while dependency-pending and performed no scientific computation.

### Decision

- Keep D1/safe-axis legality fixed.
- Mixed XYZ is closed negative evidence, not a contribution.
- Any future learned ordering may choose only within the safe-axis legal action
  set and must be compared with existing confidence ordering.

## 3. DLM methods worth borrowing

| Mechanism | Primary precedent | Relevance | Contribution status here |
|---|---|---|---|
| planned/reversible unmasking | [P2](https://arxiv.org/abs/2502.03540) | learns where to update and can remask committed tokens | useful baseline; not novel by itself |
| supervised where-to-unmask | [Where-to-Unmask](https://arxiv.org/abs/2602.09501) | learns a planner from ground-truth token margins | possible non-RL engineering baseline |
| discrete guidance | [Simple Guidance](https://arxiv.org/abs/2412.10193) | derives guidance for discrete transitions | required if guidance is revisited |
| noisy-state energy prediction | [DAO-G/DAO-P](https://arxiv.org/abs/2503.10471) | predicts energy at diffusion timesteps | direct prior art; generic reuse is not a new claim |
| diffusion RL | [d1/diffu-GRPO](https://arxiv.org/abs/2504.12216) | first critic-free policy gradient for masked dLLMs | mandatory RL baseline |
| step-wise unbiased PG | [AGRPO](https://arxiv.org/abs/2510.04019) | optimizes Markov denoising steps | preferred over naive sequence GRPO |
| intermediate-state credit | [DiSPO](https://arxiv.org/abs/2602.06462) | branches masked states and updates newly filled tokens | closest to proposed token/segment reward |

The main non-RL lesson is that the DLM makes two decisions: what token to place
and where/when to commit it. The current low-confidence sampler is myopic and
irreversible. P2-style remasking or a supervised path planner can be tested as
an engineering baseline, but learned unmasking has enough prior art that it is
not presently the paper's second contribution.

### Existing local RL must not be reused as formal evidence

The repository already contains TraceRL scaffolding and historical runs:

| Route | composition valid | all-metal/shortcut behavior | decision |
|---|---:|---:|---|
| RL v1 | 81.93% | shortcut 54.62% | no material gain |
| RL v2 aware | 85.38% | shortcut 57.71% | reward hacking |
| R5-D12b | 69.53% | all-metal 43.36% | no promotion |
| R5-D13 | 75.39% | all-metal 44.92% | no promotion |

The formal audit found that old TraceRL reconstructed rather than recorded
states, recomputed old log-probability after rollout, omitted legal-support
renormalization and temperature, ignored position/remask probability, and did
not implement exact resume. It is a heuristic scaffold only.

### If RL is eventually unavoidable

The scientifically valid last-resort design is not naive GRPO:

1. freeze formula, N, elements, counts and safe-axis legality;
2. sample a same-formula group of body trajectories;
3. use identical model494 noise across branches;
4. reward with within-formula normalized continuous energy after the frozen
   refiner, with explicit parse/duplicate/Direct failure penalties;
5. record the true joint token-and-position behavior probability;
6. use an existing dLLM RL baseline such as AGRPO/DiSPO plus reference KL,
   ESS, ratio and clip fail-closed gates;
7. validate with MatterSim and official hull, not CHGNet alone.

Token-level causal claims require controlled branching: changing a token or
segment must reproducibly predict the final energy delta across refiner seeds.
Absent that evidence, a terminal reward copied to every token is not credit
assignment.

RL remains unauthorized because it is not a small contribution, has direct
prior art, and the cheaper composition/refiner questions are unresolved.

## 4. What MP-20 `comp_valid` actually measures

The frozen Direct evaluator uses the legacy CrysLLMGen/SMACT rule:

- unary compositions pass automatically;
- all-metal compositions pass automatically;
- otherwise it searches catalog oxidation-state combinations for charge
  neutrality and applies a Pauling electronegativity test.

It is a heuristic chemistry screen, not ground-truth synthesizability.

### Real MP-20 ceiling under this metric

| Split | rows | legacy `comp_valid` |
|---|---:|---:|
| train | 27,136 | 90.50% |
| validation | 9,047 | 90.24% |
| test | 9,046 | 90.95% |

Train failure decomposition:

| Reason | Count | Share of all train | Share of invalid |
|---|---:|---:|---:|
| charge-neutrality fail | 1,377 | 5.07% | 53.41% |
| Pauling/ratio rejected | 918 | 3.38% | 35.61% |
| oxidation state missing | 283 | 1.04% | 10.98% |

The valid 90.50% includes `9,302` all-metal and `226` unary shortcuts. In the
true MP-20 `E_hull=0` subset, legacy `comp_valid` is only 88.22%; 11.78% of
known on-hull entries fail the heuristic. Therefore 100% legacy validity is not
a scientifically valid target by itself.

### H1-A2 composition

For the historical refined1000 cohort:

- valid `878/1000 = 87.8%`;
- valid reasons: 544 ionic/Pauling, 328 all-metal, 6 unary;
- invalid reasons: 79 charge-neutrality, 36 Pauling/ratio, 7 missing oxidation;
- model494 changed atom order but had zero atom-type multiset mismatches.

Thus composition validity is fixed by the Planner/formula and cannot be repaired
by body coordinates or continuous refinement.

## 5. Element/valence tokenizer analysis

The user's idea is technically sound but overlaps strongly with
[CrysVCD](https://arxiv.org/abs/2507.19799): CrysVCD already uses 217
element-oxidation tokens, 21 count embeddings, electronic-configuration
initialization, an AR Transformer and a charge-balance post-filter. Its
element-valence decomposition covers about 97.6% of its selected MP-20 scope.

Our earlier plaintext count-valence arm also demonstrates that representation
alone is insufficient:

- train/validation assignment coverage `96.66/96.32%`;
- round-trip `100%`;
- generated parsed Plans `491/512`;
- emitted-neutral only `247/491 = 50.31%`;
- all-metal rose to `45.82%`.

The missing ingredient was not another label; it was executable conservation.

## 6. Approved small contribution candidate: CCFD

Name:

> **Conservation-Constrained Formula Decoding (CCFD)**

Claim boundary:

> A drop-in online compiler that ensures atom-count and charge conservation
> during AR formula generation while preserving the all-request denominator and
> reporting chemistry coverage and distribution drift.

It does not claim stability, synthesizability, or a first element-valence
tokenizer.

### Phase 0 — CPU contract audit

Freeze before training/sampling:

- ionic/alloy/unary/mixed-valence semantics;
- permitted oxidation-state catalog and unknown policy;
- train-only vocabulary and any tokenizer merges;
- legacy SMACT false-rejection baseline on real MP-20;
- exact SMACT4/ICSD24 secondary audit;
- formula, family, arity, N and all-metal strata.

Required: >=95% representability and 100% round-trip on representable rows.

### Phase 1 — smallest causal experiment

Use one existing AR formula checkpoint and tokenizer:

| Arm | Decode | Purpose |
|---|---|---|
| F0 | current/free | control |
| F1 | online finite-state conservation | CCFD treatment |

FSM state contains:

```text
remaining atom budget N
remaining total charge Q
canonical species ordering
mixed-valence legality
alloy/unary branch state
```

EOS is legal only for a completed valid branch. Each request receives one
trajectory; dead ends remain failures and no survivor replacement is allowed.

Two seeds x 1,000 requests per arm. Required:

- requested-denominator conservation validity >=99%;
- external composition correctness strictly above F0;
- no all-metal/family drift over 3 pp;
- N/arity TVD <=0.05;
- novelty and uniqueness no worse than -1 pp;
- both seeds agree and pooled 95% CI is positive.

If F1 passes, the defensible contribution is the online conservation compiler.
No tokenizer claim is needed.

### Phase 2 — conditional tokenizer factorial

Run only if Phase 1 shows that tokenizer fragmentation/sequence length, rather
than decode legality, remains a measurable bottleneck.

| Token representation | Free decode | CCFD decode |
|---|---|---|
| current text/atomic | F0 | F1 |
| true species-valence + count special tokens | T0 | T1 |

Use matched backbone, data, initialization, updates and sampling budget. A
same-backbone CrysVCD-style factorized-token reimplementation is the causal
baseline; official CrysVCD is external context only. Optional BPE-like macro
merges are train-only ablations and every macro must carry additive atom-count,
charge and element metadata.

If only F1 improves, the claim stays CCFD. If T1-F1 is independently positive,
then and only then may the tokenizer be included in the contribution.

## 7. Stability remains a separate track

- Formula-only is not the stability candidate.
- Global axis/safe-axis remains frozen.
- Complete the two-seed full/hard-axis x raw/model494 diagnostic.
- Calibrate `tau={0,200,500,800}` only if raw-versus-refiner results trigger the
  predeclared gate.
- CHGNet-labelled energy supervision requires independent MatterSim/official
  validation before training.
- Final replacement still requires Strict S.U.N. >=10%, Meta S.U.N. >=50%,
  and body/Direct/novelty/uniqueness/retention noninferiority.

Composition and stability results must not be merged into one causal claim.

## 8. Multi-agent decision log

### Skeptic

Accepted objections:

- formula-only already failed;
- tokenization does not imply stability;
- CrysVCD overlaps element-valence-count generation;
- learned unmasking, energy guidance and dLLM RL have direct prior art;
- legacy `comp_valid` is biased;
- old pair route failed a data gate, not the preference algorithm itself.

Disposition: REVISE.

### Constraint Guardian

Accepted requirements:

- no resampling, repair or survivor denominator;
- all-metal/mixed-valence/unknown branches explicit;
- train-only vocabulary and leakage audit;
- tokenizer and FSM effects isolated;
- official CrysVCD labelled external;
- distribution and two-seed gates;
- RL unauthorized.

Disposition: REVISE pending frozen contracts.

### User Advocate

Accepted objections:

- macro/BPE terminology obscured the small problem;
- CrysVCD fairness was unclear;
- improved enforced validity must not be called learned chemistry or stability.

Disposition: REVISE to CCFD-first sequential experiment.

### Integrator / Arbiter

Final disposition:

> **APPROVED for Phase 0 -> Phase 1 under the stated gates and claim limits.**
> Phase 2 remains conditional. RL remains unapproved.

No reviewer objection was rejected on its merits.

## 9. Exact next action

Do not start another GPU training job yet. First execute the Phase 0 CPU audit
and freeze its protocol. Then run F0 versus F1 using the same formula checkpoint
and requested denominators. This is the smallest experiment that can determine
whether the contribution is real.
