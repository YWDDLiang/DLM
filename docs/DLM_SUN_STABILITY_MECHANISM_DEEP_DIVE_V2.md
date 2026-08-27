# DLM–S.U.N. Stability Mechanism Deep Dive V2

Date: 2026-08-28

Status: mechanism diagnosis complete; the first causal 2×2 L6 ablation is
running. This document supersedes the earlier recommendation to keep extending
plain CE or to add a `target_stability` token.

## Decision

Do not treat the current negative preference-pair pilot as the end of the DLM
route. It only rejected one sparse, one-pair-per-Plan DPO-style dataset under
its frozen yield gate. The next work targets the actual conversion bottleneck:

1. remove unreliable soft Plan fields from the DLM branch while preserving
   exact formula, atom count, elements and stoichiometric counts;
2. stop committing X, then Y, then Z coordinates irreversibly and denoise all
   coordinate dimensions as one coupled block;
3. if those inference mechanisms are insufficient, train a noisy-state energy
   critic on all labelled generated structures and use it inside the discrete
   denoising trajectory;
4. distil relaxed low-energy bodies into the executor so that the initial body
   lands in a better basin before model494 and CHGNet relaxation.

The design remains non-RL. It does not expose `E0`, `stable`, `E_hull` or an
energy target in the Plan or DLM prompt, and it does not select a final sample
by best-of-K reranking.

## What S.U.N. is measuring

For an all-request denominator, S.U.N. is an intersection rather than a single
model score:

```text
Strict S.U.N. = reconstructed ∩ novel ∩ unique ∩ official E_hull <= 0
Meta   S.U.N. = reconstructed ∩ novel ∩ unique ∩ official E_hull <= 0.1 eV/atom
```

Unknown official hull is missing data and is never converted to unstable.

On the frozen raw1000 total-2-epoch diagnostic:

| Stage | Strict | Meta |
|---|---:|---:|
| Stable, before novelty/uniqueness | 102 | 587 |
| S.U.N. | 81 | 489 |
| Stable-to-S.U.N. retention | 79.41% | 83.30% |
| Absolute target | 100 | 500 |

The executor therefore needs approximately `+19` Strict and `+11` Meta S.U.N.
At the observed retention this means roughly `+24` Strict-stable and `+14`
Meta-stable structures, not a wholesale change in novelty.

The novel-unique hull distribution also shows exploitable near-boundary mass:

| Threshold | Count |
|---|---:|
| `E_hull <= 0` | 81 |
| `E_hull <= 0.01` | 107 |
| `E_hull <= 0.05` | 265 |
| `E_hull <= 0.10` | 489 |

Thus a modest downward energy shift near the hull can cross both requested
gates. Increasing syntactic success is not the main remaining objective.

## Why ordinary CE stopped helping

The current masked DLM training objective samples a timestep, masks answer
tokens and minimizes weighted token cross-entropy. It has no term for pair
distance, coordination, force, compatible formation energy or hull distance.
The exact-Plan sampler pre-fills atom count and element identities, so the free
variables are chiefly lattice and discretized coordinates.

Its current generation path then commits:

```text
N -> element identities -> lattice -> all X -> all Y -> all Z
```

Committed tokens are not re-masked. A locally confident X token is therefore
frozen before its Y/Z context exists, although stability depends on full 3D
distances under the periodic lattice. This creates a train/inference mismatch:
random-mask CE can use bidirectional context during training, while the fixed
axis schedule imposes an irreversible causal order at sampling time.

The total-epoch experiment confirms the distinction:

| Checkpoint | Body | Direct joint | Strict stable/S.U.N. | Meta stable/S.U.N. |
|---|---:|---:|---:|---:|
| total 2 epochs | 985 | 871 | 102 / 81 | 587 / 489 |
| total 3 epochs | 992 | 878 | 100 / 79 | 578 / 477 |

More CE learned format validity but did not learn lower-energy basin entry.
That is why another epoch sweep is not justified.

## What the local stability audit says

These are associations on the raw1000 total-2 checkpoint, not causal effects.
They are used to choose interventions, not to alter the reported cohort.

### Soft rich fields are not reliable stability guides

| Audit group | n | Strict stable | Meta stable | median `E_hull` |
|---|---:|---:|---:|---:|
| lattice–SG match | 431 | 8.1% | 53.8% | 0.0881 |
| lattice–SG mismatch | 569 | 11.8% | 62.4% | 0.0683 |
| both volume and lattice–SG match | 228 | 9.2% | 55.7% | 0.0879 |
| not both | 772 | 10.5% | 59.6% | 0.0739 |

The generated soft fields are therefore not calibrated enough to amplify with
ordinary CFG. In particular, the existing unconditional CFG branch masks the
entire prompt, so it also removes formula and atom-count anchors. That logit
difference confounds chemistry with soft-condition guidance.

### Chemistry and length expose two bottlenecks

| Group | Strict stable | Meta stable | median `E_hull` |
|---|---:|---:|---:|
| halide | 23.4% | 85.1% | 0.031 |
| all-metal | 16.2% | 76.4% | 0.041 |
| oxide | 3.1% | 42.3% | 0.121 |
| sulfide | 4.4% | 42.6% | 0.119 |
| charge-fail | 5.4% | 47.3% | 0.104 |
| `N=1–4` | 19.1% | 72.1% | 0.040 |
| `N=13–20` | 3.5% | 50.0% | 0.098 |

This separates:

- a proposal-chemistry problem: charge-implausible and difficult long/ionic
  compositions are overrepresented in the hard tail;
- an executor-geometry problem: even under fixed composition, the text DLM has
  weak periodic 3D and energy inductive bias.

Planner distribution changes cannot be used to hide the second problem. Any
future Planner work must report family, charge and N drift explicitly and be
evaluated separately from executor conversion.

## Relevant external mechanisms

- [MatterGen](https://arxiv.org/abs/2312.03687) jointly denoises atom types,
  coordinates and periodic lattice and uses property adapters. Its design
  supports joint geometric state modelling rather than an axis-wise text
  commitment schedule.
- [CDVAE](https://arxiv.org/abs/2110.06197),
  [DiffCSP](https://arxiv.org/abs/2309.04475) and
  [FlowMM](https://arxiv.org/abs/2406.04713) all encode periodic/E(3) geometry
  directly. They explain why token CE can improve validity without learning the
  local/global stability manifold as efficiently.
- [Siamese Foundation Models for Crystal Structure
  Generation](https://arxiv.org/abs/2503.10471) couples a generator with an
  energy predictor trained on stable and unstable structures and predicts
  energy at intermediate diffusion timesteps. This is the closest precedent
  for the proposed noisy-state critic.
- [Simple Guidance Mechanisms for Discrete Diffusion
  Models](https://arxiv.org/abs/2412.10193) shows that guidance for discrete
  diffusion should be derived in the discrete transition process; copying
  continuous CFG mechanically is not generally correct.
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) motivates reference-corrected
  pairwise training, but our one-primary-pair-per-Plan pilot yielded only
  `95/27` train/validation pairs and failed its frozen `96/24` gate. That rejects
  this sparse data construction, not thermodynamic supervision in general.
- [CrysVCD](https://arxiv.org/abs/2507.19799) separates valence-balanced
  composition generation from geometric diffusion and applies stability
  guidance to geometry. Our count-valence text Planner arm failed because a
  physics-labelled teacher did not make a BPE model execute coupled charge
  arithmetic; the useful lesson is architectural separation, not copying its
  element tokenizer into the current LLaMA prompt.
- [UniMat](https://arxiv.org/abs/2311.09235) explicitly notes that ordinary
  reconstruction metrics can be misaligned with stable-material discovery,
  matching our body-versus-S.U.N. divergence.

## Experiment 1 — causal inference-factorial (running)

Use the public H1-A2 DLM weights, frozen first256 raw Plans, model494 and paired
sample-index RNG. Run two seeds over a 2×2 factorial:

| Arm | Condition presented to DLM | Coordinate commitment |
|---|---|---|
| `full_axis` | full JSON Plan state | X, then Y, then Z |
| `full_joint` | full JSON Plan state | all XYZ together |
| `hard_axis` | formula/N/elements/counts only | X, then Y, then Z |
| `hard_joint` | formula/N/elements/counts only | all XYZ together |

No weight training, energy label, CFG scale, RL or final reranking is involved.
`hard_joint` is the predeclared primary candidate. Promotion requires pooled
Strict and Meta improvement, no more than 1 pp loss in body, Direct, novelty,
uniqueness or retention, and no more than 1 pp Strict/Meta loss in either seed.

This factorial identifies whether the immediate bottleneck is condition noise,
coordinate schedule, or their interaction. It must be reported in full even if
one cell looks best.

## Experiment 2 — noisy-state thermodynamic critic

Run this only after Experiment 1 is finalized. It is a new mechanism and does
not reopen the failed sparse-pair gate.

### Data

1. Freeze a new train-only Plan cohort disjoint by exact formula from L6/L7.
2. Generate multiple bodies per Plan under the selected condition/schedule.
3. Keep every Direct-valid body with known compatible CHGNet energy; do not
   require a hard `0.06 eV/atom` pair.
4. Normalize energy within exact composition/Plan, using rank or energy above
   the within-Plan minimum. This removes composition as a shortcut.
5. Corrupt each body at sampled DLM timesteps and train the critic to predict
   within-Plan rank/energy from the noisy state and timestep.
6. Split by formula before generation. Novelty is never a training label.

This uses all eligible structures rather than one extreme pair per Plan. A
held-out rank-correlation and low-versus-high AUC gate is required before any
generator update or guided sampling.

### Guidance

The critic is a sidecar; no stability field is added to the prompt. At each
masked-denoising block:

1. obtain generator logits;
2. form a small set of high-probability lattice/coordinate proposals;
3. score their partially denoised states with the timestep-aware critic;
4. adjust only the current transition logits by the calibrated critic score;
5. allow low-confidence or geometrically conflicting committed coordinates to
   be re-masked on the next iteration.

This is trajectory guidance, not final best-of-K selection. The unguided DLM
probability remains in the score to prevent surrogate-only collapse. One
guidance coefficient is calibrated on train-only validation; no test-time grid
or per-composition tuning is allowed.

### Gates

- critic held-out Spearman correlation and low/high AUC must beat a
  Plan-family/N baseline;
- L6 two-seed Strict and Meta directions must both be positive;
- body, Direct, novelty, uniqueness and stable-to-S.U.N. retention must be
  noninferior by 1 pp;
- report energy quantiles and `0/0.01/0.05/0.1` threshold counts;
- only then run requested-1000 L7, where the absolute gate remains
  Strict S.U.N. `>=10%` and Meta S.U.N. `>=50%`.

## Experiment 3 — relaxed-winner distillation

If the critic predicts energy but guidance is too expensive or unstable,
distil the low-energy basin directly:

1. for each train-only Plan, keep the lowest-energy Direct-valid generated body
   and its CHGNet-relaxed geometry;
2. serialize the relaxed geometry under the same hard-anchor prompt;
3. fine-tune with ordinary masked CE only on relaxed winners, mixed with stable
   MP-20 examples and a frozen-reference KL term;
4. use high-energy bodies only in the critic/ranking loss, never as CE targets;
5. evaluate at one conservative training budget with the same two-seed L6
   contract.

This is on-policy basin distillation: it teaches the text executor geometries
that its own downstream relaxation actually maps toward lower energy. It is
more targeted than another epoch over the generic MP-20 distribution.

## Architecture boundary

If hard/joint scheduling, a validated critic and relaxed-winner distillation
all fail, the evidence points to the representation itself. The next honest
architecture is then:

```text
rich Plan / composition model
        -> periodic E(3)-equivariant continuous lattice-coordinate decoder
        -> frozen common evaluation and official hull
```

That would borrow the geometric inductive bias of DiffCSP/FlowMM/MatterGen
while preserving the paper's Plan-to-executor interface. It should be framed as
an executor redesign, not as another hidden Planner optimization.

## Contribution-point framing

The defensible contribution is not “more DLM epochs” or “a stability token.”
The strongest coherent claim to test is:

> A reliability-aware rich-Plan executor preserves hard chemical anchors,
> couples periodic 3D coordinate commitment, and uses noisy-state
> thermodynamic feedback to improve fixed-Plan stable conversion without RL or
> final-sample reranking.

The current 2×2 experiment tests the first two clauses. A successful
timestep-aware critic plus L7 confirmation would establish the third and make
this a genuine second contribution. Until those gates pass, it remains a
mechanism-driven candidate rather than a paper claim.
