# CTV-DLM Stability Contract V1

Date: 2026-08-28

Status: **APPROVED for prerequisite implementation and one 256-completion
resource canary only.** Formal Branch-Q generation, Q training, L6 and L7 are
not yet authorized.

CTV-DLM means **Counterfactual Terminal-cost Value Diffusion Language Model**.
It is a stability-directed masked crystal generation system, not a claim that
the frozen base DLM parameters already encode thermodynamic stability.

## Scientific question

After C³FD-v2.5 freezes a chemically certified de novo composition, can real
counterfactual terminal-cost supervision alter legal geometry-token
probabilities during the masked reverse trajectory and improve Strict and Meta
S.U.N. under compute-matched controls?

The system decomposition is:

```text
C³FD-v2.5 certified composition
        -> minimal composition specification
        -> frozen base masked DLM
        -> CTV token-cost head and normalized discrete guidance
        -> frozen model494 tau800 / common relaxation
        -> Strict / Meta S.U.N.
```

## Closed negative evidence

- another plain CE epoch improved body/Direct but reduced Strict and Meta;
- rich soft fields did not predict lower hull energy;
- global XYZ-joint commitment caused duplicate-coordinate and Direct failure;
- tau 0/200/500/800 moved along a novelty--stability trade-off and retained
  tau800;
- one-extreme-pair-per-Plan DPO retained only 95/27 train/validation pairs and
  failed its frozen 96/24 gate;
- historical TraceRL has invalid behavior-probability accounting and reward
  shortcuts and remains closed.

## Frozen minimal-spec base

The visible prompt is one compact sorted-key JSON object followed by the body
marker:

```text
{"N":5,"charge":"certified_neutral","counts":[3,2],"elements":["O","Fe"],"family":"oxide","formula":"O3Fe2"}
dynamic_crystal_body:
```

Rules:

- `elements` are sorted by atomic number and `counts` are aligned;
- `N == sum(counts)` and `formula` is rendered from the same arrays;
- `charge` is one of `certified_neutral`, `all_metal`, or `single_element`;
- the exact C³FD valence witness remains a machine sidecar, not a prompt token;
- lattice, space group, volume, prototype and stability/energy fields are not
  visible;
- compact UTF-8 JSON uses sorted keys and separators `(',', ':')`;
- prompt length must be <=128 tokenizer tokens.

The base is trained exactly once from the retained control `step-1000` resume
for a fixed total-two-epoch continuation. There is no checkpoint search.
Tokenizer, base checkpoint, safe-axis schedule, model494, tau800 and common
refiner hashes are frozen before Branch data.

## Counterfactual terminal-cost estimand

Every frozen-base trajectory has exactly two candidate intervention
milestones: the first safe-axis reverse step at which visible free-geometry
fraction reaches `0.60` and `0.80`.

At each milestone:

1. the frozen base-confidence rule chooses exactly one lattice/coordinate
   position `j`;
2. force one legal token `v` at `j`;
3. complete with the frozen base DLM;
4. run model494 at tau800 with a common refiner seed;
5. run the frozen common CHGNet relaxation/evaluation.

The primary known terminal cost is compatible CHGNet formation energy per atom
after this frozen pipeline, in eV/atom; lower is better. A pair is tied when
`|delta cost| < 1e-4 eV/atom`.

For state `s`, position `j` and action `v`:

```text
Q(s,j,v) = E[known terminal cost | force v, frozen continuation]
A(s,j,v) = Q(s,j,v) - mean_known_actions Q(s,j,.)
```

Train has one continuation seed. Validation has two common
continuation/refiner seeds and averages known converged returns. An action with
zero known returns has no Q target. Parse, Direct, refiner convergence and
energy-known outcomes are recorded separately as feasibility targets; unknown
energy is missing, never high cost.

## Branch actions and identities

Each state must yield exactly eight distinct legal actions:

- frozen-base argmax;
- tokens at cumulative legal-probability quantiles
  `{0.05,0.15,0.30,0.50,0.70,0.85,0.95}`;
- ties are resolved by token id.

If quantiles collide with one another or with argmax, the state is a protocol
failure. No substitute token, repeated action or adaptive quantile is allowed.

The independent identity is the atomic-number-sorted integer composition
reduced by the gcd of counts. Formula strings are not sufficient.

- the old 256-Plan critic source cohort overlaps the complete C³FD
  seed17/seed18 sources by `1/0` reduced identities; the outcome-blind L6
  selection below does not contain that identity, so the frozen Branch
  canary/train/validation sets require no row deletion;
- C³FD seeds share 24 reduced identities;
- L6 is the outcome-blind first 256 seed17 requests whose identity is absent
  from the entire seed18 set;
- L7 retains every seed18 requested-1000 attempt and is called a one-time DLM
  downstream holdout, not a globally fresh test set.

## Resource canary

The only currently authorized GPU rollout is:

```text
8 Plans x 2 milestones x 8 actions x 2 continuations = 256 completions
```

- 4 A800, 300G host RAM;
- resource/engineering metrics only; no energy/rank/success result may alter
  the method;
- every branch id is keyed by reduced identity, Plan/sample index, milestone,
  position, action token and continuation seed;
- output must contain 256 unique ledger rows;
- report GPU-hours, samples/s, per-branch time, MaxRSS, peak device memory,
  disk bytes and all failure reasons;
- formal 360G work can be reconsidered only if observed/extrapolated peak RAM
  is below 270G.

The proposed formal budget remains unauthorized:

```text
train:      128 Plans x 2 x 8 x 1 = 2048 completions
validation:  32 Plans x 2 x 8 x 2 = 1024 completions
total:                                  3072 completions
```

## Q model and support

Generator weights and logits remain frozen. Two Q heads are trained on
disjoint Plan groups (with bootstrap only inside each group) and consume frozen
hidden states. Loading either head must leave base logits bit-identical.

The head predicts cost over the complete currently legal geometry-token
support. An action is supported only if:

- it has >=8 known branch returns across >=4 train Plans **inside each head's
  own disjoint data group**;
- both heads return finite values;
- both centered advantages share sign outside the neutral band
  `|A|<=0.005 eV/atom`, or both are inside the band;
- inter-head absolute disagreement is <=0.02 eV/atom.

No-success actions, one-head missing values, insufficient coverage or
disagreement are unsupported and receive exactly zero advantage.

A state is guided only when supported actions carry >=70% of frozen-base legal
probability mass. Validation additionally requires guided-state coverage >=60%
and projected sample fallback <=40%. All coverage and fallback quantities are
reported.

Feasibility is diagnostic and never deletes a token or enters the primary
guidance logits.

## Discrete guidance and compute-matched control

For supported actions only:

```text
p_gamma(v | s,j) proportional to
    p_DLM(v | s,j) * exp(-gamma * A_hat(s,j,v))
```

The distribution is normalized over the complete legal support. Unsupported
actions use `A_hat=0`. There is no top-K, uncertainty division, Doob claim,
position change, remasking or terminal reranking.

`gamma` has units `eV^-1` and is selected once from `{0,5,10}` on Branch
validation by minimum Plan-mean empirical expected absolute cost over the eight
observed actions, subject to all support/feasibility gates. Ties choose lower
gamma. Strict and Meta are never used. If gamma0 wins, the route stops before
L6.

The compute-matched gamma0 arm runs identical Q heads, full legal masks,
forward counts, position schedule, step count and counter-based RNG. Its masked
per-token probabilities must equal frozen base probabilities. RNG keys are
`composition_id/sample_idx/step/position/continuation_seed` and never depend on
sequential random-number consumption.

The primary estimand is the joint two-milestone CTV policy. The two
interventions are not separately causal because the 0.60 action changes the
state observed at 0.80.

## Symmetry and structure checks

Frozen consistency transforms are:

- deterministic same-species permutations from a prewritten per-row list;
- fractional-coordinate wrap modulo one;
- global translations `(0,0,0)` and `(0.25,0.5,0.75)` modulo one;
- all six axis permutations with lattice and coordinates transformed together.

No claim of full equivalent-cell invariance is made. Structural uniqueness
uses `StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10 degrees)`.

## MatterSim gates

Gate A is the already-running 1752-structure audit. It measures cross-MLIP rank
transfer, not independent physical truth.

Before formal Branch work, the late (`0.80`) milestone's CHGNet-lowest and
highest observed actions for 32 frozen validation Plans produce 64 fixed
structures. MatterSim-v1.0.0-5M performs full cell/coordinate relaxation with
`FrechetCellFilter + FIRE`, `fmax=0.05 eV/Angstrom`, `max_steps=200` and frozen
device/precision. Gate B fails unless all 64 converge with finite energy; their
extreme action AUC must exceed 0.60.

## Frozen scientific gates

Branch-Q validation:

- Plan-bootstrap Spearman 95% lower bound >0;
- action pairwise AUC >0.60;
- cross-continuation/refiner action-sign agreement >0.60;
- feasibility AUROC >0.70;
- symmetry rank agreement >0.90;
- oxide, sulfide and N13--20 direction positive when each has >=5 Plans and
  >=20 comparisons;
- MatterSim-relaxed extreme action AUC >0.60;
- support/coverage/fallback gates above.

L6, only after a new authorization:

- two DLM seeds over the frozen C³FD L6 cohort;
- compute-matched gamma0 versus guided;
- pooled Strict and Meta both positive and neither seed below -1pp;
- body, Direct, StructureMatcher uniqueness and stable-to-S.U.N. retention
  point deltas >=-1pp;
- paired discordance, confidence intervals, energy quantiles and cost reported;
- L6 is go/no-go only.

L7, only after another authorization:

- run seed18 requested1000 once;
- Strict S.U.N. >=10% and Meta S.U.N. >=50%;
- both improve over matched base;
- report paired CI/McNemar and call thresholds benchmark attainment, not
  confidence-bound success;
- public `105/488` remains unchanged until the complete replacement passes.

## Authorization boundary

Prerequisite code, audits and tests are authorized. A single 256-completion
resource canary is authorized only after Gate A, base freeze, reduced-identity
manifests, base-logit equivalence and canary hard assertions pass. Formal
Branch-Q generation, Q training, L6, L7, distillation and RL remain NO-GO.
