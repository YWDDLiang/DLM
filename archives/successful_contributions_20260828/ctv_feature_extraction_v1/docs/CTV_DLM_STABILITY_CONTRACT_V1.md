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
        -> instantiate exact 7+4N typed crystal state
        -> anchor N and element multiset
        -> frozen base masked DLM over 6+3N free geometry tokens
           with selected support and one predeclared legal schedule
        -> CTV token-cost head and normalized discrete guidance
        -> frozen model494 tau800 / common relaxation
        -> Strict / Meta S.U.N.
```

CTV-DLM is an additive extension of the existing exact-cardinality masked
executor. It does not replace or absorb the `7+4N` representation, dynamic
length, typed schema, composition anchors, selected support, frozen commitment
policy, or pre/post-refiner attribution evidence. CTV is schedule-compatible;
the primary experiment retains the current X -> Y -> Z exact-axis policy only
to isolate the value-guidance effect. The policy is not itself a new CTV claim.

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

The executor retains the crystal-specific special-token language rather than
emitting generic text: `<N_###>`, `<LA/LB/LC_###>`, `<AA/AB/AG_###>`,
`<E_symbol>` and `<X/Y/Z_###>`.  For a certified candidate composition it
instantiates exactly `7+4N` typed body positions, anchors the count and element
tokens, and turns the remaining `6+3N` geometry tokens into a structure.  The
primary DLM claim is therefore candidate-to-structure execution; stability
awareness is an additional improvement to that executor.

The base is trained exactly once from the retained control `step-1000` resume
for a fixed total-two-epoch continuation. There is no checkpoint search.
Tokenizer, base checkpoint, primary exact-axis schedule, model494, tau800 and
common refiner hashes are frozen before Branch data.

The incomplete tau900/1000 extension is not evidence for or against either
setting: two memory-failed attempts left asymmetric partial cells and no
complete two-seed comparison. Tau800 is retained because it is the only
high-tau setting with a complete matched evaluation. A future sequential
900/1000 sensitivity run may occur only after CTV L6 and cannot select or
redefine the primary CTV policy.

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

The seven quantile targets are projected one-to-one onto unused legal tokens
by minimum absolute distance to each token's CDF midpoint; ties prefer higher
base probability and then lower token id. The argmax token is reserved first.
This deterministic distinct-CDF projection was frozen after the engineering
canary showed that literal CDF crossings collide under a concentrated base
distribution. It uses no energy, success or downstream outcome. Fewer than
eight legal tokens remains a protocol failure; repeated actions and adaptive
quantile values remain forbidden.

The independent identity is the atomic-number-sorted integer composition
reduced by the gcd of counts. Formula strings are not sufficient.

- the old 256-Plan critic source is filtered by the frozen C³FD benchmark
  certificate before positional selection: `34` rows are retained in a reject
  ledger, then the first `8`, next `128`, and first validation `32` certified
  rows are frozen; no energy, stability, novelty or success outcome is used;
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

## Deferred independent MatterSim audit

The 1752-structure MatterSim audit measures cross-MLIP rank transfer, not
independent physical truth.  It is intentionally deferred until an internal
CTV/H1-A2 candidate has demonstrated a positive L6 signal.  It is not a
prerequisite for the engineering canary, Branch-Q construction, or the first
internal L6 screen.

Before a final external claim, the late (`0.80`) milestone's CHGNet-lowest and
highest observed actions for 32 frozen validation Plans produce 64 fixed
structures. MatterSim-v1.0.0-5M may perform full cell/coordinate relaxation with
`FrechetCellFilter + FIRE`, `fmax=0.05 eV/Angstrom`, `max_steps=200` and frozen
device/precision. This is confirmation evidence: all 64 must converge with
finite energy and their extreme-action AUC must exceed 0.60 before a
cross-potential stability claim is made.

## Frozen scientific gates

Branch-Q validation:

- Plan-bootstrap Spearman 95% lower bound >0;
- action pairwise AUC >0.60;
- cross-continuation/refiner action-sign agreement >0.60;
- feasibility AUROC >0.70;
- symmetry rank agreement >0.90;
- oxide, sulfide and N13--20 direction positive when each has >=5 Plans and
  >=20 comparisons;
- support/coverage/fallback gates above.

L6, only after a new authorization:

- two DLM seeds over the frozen C³FD L6 cohort;
- compute-matched gamma0 versus guided;
- pooled Strict or Meta S.U.N. improves and the other is no worse than `-1 pp`;
- pooled body and Direct deltas are each >=`-3 pp`;
- StructureMatcher uniqueness and novelty deltas are each >=`-5 pp`;
- stable-to-S.U.N. retention is reported and vetoes only a degradation worse
  than `-10 pp`;
- both seeds are reported, but no per-seed `-1 pp` hard veto is used;
- paired discordance, confidence intervals, energy quantiles and cost reported;
- L6 is go/no-go only.

L7, only after another authorization:

- run seed18 requested1000 once;
- Strict S.U.N. >=10% and Meta S.U.N. >=50%;
- pooled body and Direct remain within `-3 pp` of matched base;
- structural uniqueness and novelty remain within `-5 pp`;
- retention is diagnostic unless it falls by more than `10 pp`;
- report paired CI/McNemar and call thresholds benchmark attainment, not
  confidence-bound success;
- public `105/488` remains unchanged until the complete replacement passes.

## Frozen fallback ladder

Failure does not return immediately to an unrelated Planner or MatterSim
experiment.  The H1-A2 executor is improved in the following fixed order:

1. **Engineering fallback:** if 16-way action batches or model494 cause memory
   failure, execute the same 256 branch ledger in smaller sequential
   microbatches.  Actions, random keys and denominators do not change.
2. **Coverage fallback:** if the value head lacks supported legal-token mass,
   expand only over the already frozen certified Branch source and/or add
   continuation replicates.  No threshold lowering or outcome-selected Plans.
3. **Positive-direction but insufficient S.U.N.:** distil the converged
   low-energy winner for each fixed composition into the same special-token
   DLM. High-energy bodies are contrastive/ranking examples and are never CE
   targets.
4. **No value direction:** train a geometry-token curriculum on real stable
   MP-20 structures, with lattice/XYZ masked losses emphasized. Matched
   unstable structures supply only an auxiliary energy-ordering loss; no
   `stable`, energy, or target-stability prompt token is introduced.
5. **Last resort:** a DLM-native RL method may be audited only after the above
   non-RL routes fail and only with valid diffusion likelihood accounting.

Every fallback preserves `7+4N`, exact composition, special tokens, typed
legal support, requested denominators, and raw-body versus model494 reporting.
It receives a new method label and immutable result directory.

## Authorization boundary

Prerequisite code, audits and tests are authorized. A single 256-completion
resource canary is authorized after base freeze, certified reduced-identity
manifests, base-logit equivalence and canary hard assertions pass.  A clean
engineering canary authorizes the frozen formal Branch-Q build; its internal
validation gates authorize L6.  L7, distillation, stable-token curriculum and
RL remain separately stage-gated. MatterSim is deferred until a positive L6
candidate requires independent cross-potential confirmation.
