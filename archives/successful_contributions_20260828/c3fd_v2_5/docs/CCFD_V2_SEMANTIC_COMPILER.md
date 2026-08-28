# C³FD-v2 Semantic Composition Compiler

Date: 2026-08-28

Status: candidate design and CPU audit in progress. Contribution point 1 and
the public `105/488` result are frozen.

Working name:

> **C³FD-v2 — Contextual Chemistry-Constrained Formula Decoding**

The third C is the exact conservation compiler: target atom count and total
charge are tracked as hard state rather than learned approximately. The name is
provisional until the matched composition-validity experiment passes.

## Problem isolated by CCFD-v1

The first CCFD arm raised internal oxidation-state assignment from `94.90%` to
`99.15%`, but independent legacy comp-valid changed only `1724/2000` to
`1725/2000`. The treatment changed the sampled formula in `1883/2000` paired
requests and shifted the atom-count distribution. Among F1 formulas accepted
by the internal assignment but rejected independently, the main reasons were
charge-neutrality failure, Pauling rejection and unavailable oxidation states.

Therefore the missing mechanism was not more labels. It was an executable,
metric-aligned semantic decoder.

## What is borrowed from CrysVCD

[CrysVCD](https://arxiv.org/abs/2507.19799) establishes three useful ideas:

1. generate composition before expensive geometry generation;
2. represent species by element and oxidation state rather than formula text;
3. initialize species representations with electronic-configuration features
   and represent integer counts separately.

CrysVCD uses 217 element-valence tokens, 21 count embeddings, a static
electronic-configuration projection, separate alloy and ionic GPT-2 models,
random tuple permutations during training, and a charge-balance filter after
generation.

## What must be stronger and distinct here

| Mechanism | CrysVCD-style baseline | C³FD-v2 candidate |
|---|---|---|
| representation | static element-valence token + count | typed `N`, element, valence and count actions |
| chemical context | Transformer sequence context | train-only element-valence composition graph plus shared backbone context |
| atom count | implied by generated counts | `N` emitted first and locked |
| charge constraint | terminal filter | exact remaining-atom/charge reachability at every action |
| ordering | random tuple permutations | canonical semantic order; set/graph context is permutation-aware |
| alloy/ionic model | two Transformers | one shared Planner backbone; branch is decoder state |
| validity target | internal valence balance | independent benchmark certificate plus extended-only unknown certificate |
| output | composition for CSP | unchanged seven-line `h1_rich_plan_v1` after deterministic rendering |
| failure | filtered composition | one request, one trajectory; dead ends remain failures |

Electronic configuration is a node feature, not the contribution by itself.
The candidate contribution is online semantic conservation with contextual
chemistry and independent certification.

## One-model architecture

The model remains one Planner. A shared Transformer produces hidden states for
four small structured heads:

```text
shared request/backbone state
  -> N head
  -> element-valence head
  -> count head
  -> rich-property head (anion/lattice/spacegroup/volume)
```

Species embedding:

```text
learned element embedding
+ learned oxidation-state embedding
+ projected electronic-configuration / periodic features
+ learned count embedding
+ contextual composition-graph message
```

The graph nodes are `(element, oxidation state)` and are weighted by counts.
Train-only pair statistics or a small message-passing layer model the fact that
some species co-occur more often. This is a soft logit prior only. It cannot
make an atom/charge-illegal action legal.

No BPE merge represents a formula or an element pair. BPE frequency is a text
compression signal, whereas the required invariants are chemical and integer
valued.

## Semantic decoder contract

Action sequence:

```text
SelectN(N)
AddSpecies(element, oxidation_state, count) ...
EndComposition
```

Frozen invariants:

- `1 <= N <= 20`, emitted exactly once;
- `remaining_atoms = N - sum(count)` always lies in `[0,N]`;
- `net_charge = sum(oxidation_state * count)`;
- semantic species keys are canonical and non-repeating;
- benchmark-strict generation uses one oxidation state per element;
- broader adjacent same-sign mixed valence is diagnostic only;
- zero-valence and ionic branches do not mix;
- every retained action has at least one reachable certified completion;
- EOS requires exact atom and charge conservation;
- rendering and reparsing recover the same unreduced N and composition.

Dual certificate:

- `benchmark_compatible`: independent frozen CrysLLMGen/SMACT endpoint passes;
- `extended_only`: exact broader valence ledger but independent endpoint fails;
  it is recorded as unknown and is never a positive target;
- validator errors are indeterminate, never silently unstable or valid.

## Physics and co-occurrence are separate

Electronic configuration gives a useful prior for individual ions. It does not
say whether two species belong together at a given stoichiometry. C³FD-v2 adds
a train-only composition graph:

```text
edge score(i,j) = smoothed PMI((element_i,valence_i),
                               (element_j,valence_j))
```

The CPU audit evaluates the graph only on formula-disjoint validation rows. It
must beat a marginal/random-pair baseline before it is allowed into logits.
Rare but legal pairs remain available because the graph never acts as a mask.

## Training objective

```text
L = L_N
  + L_element
  + L_valence
  + L_count
  + lambda_pair * L_context_pair
  + lambda_soft * L_rich_properties
```

`lambda_pair` is selected once from train/validation calibration so that the
soft prior cannot dominate the model. There is no stability token, E0 token,
RL reward or final-sample reranking in this composition experiment.

## Matched effect experiment

No empirical comparison to CrysVCD is required. It is design inspiration and
related work only. After CPU audits pass, run one matched causal comparison:

1. `P0`: the current frozen rich-Plan Planner;
2. `C3`: one full C³FD-v2 candidate with physics species features, locked N,
   online reachability, contextual pair graph and dual certificate.

Use matched initialization, teacher rows, updates, sampling budget and two
seeds. Internal feature flags remain available for engineering diagnosis, but
they do not create extra paper arms unless the full candidate fails and a
specific failure must be isolated. The primary question is simply whether the
complete candidate produces a real independent composition-validity gain.

## Promotion gates

CPU gate before any Planner GPU job:

- train/validation exact N, charge, composition and rich-render round trip are
  all 100% on compiled rows;
- benchmark certificate agrees exactly with the independent evaluator;
- every independent-valid row remains representable;
- raw1000 semantic coverage is within 3 percentage points of train;
- formula-disjoint validation pair-prior AUC exceeds 0.60 with at least 1000
  positive pairs and at least 95% known-node coverage.

Matched sampling gate:

- independent all-request comp-valid strictly improves in both seeds and pooled
  95% CI is positive;
- formula/Plan parse and requested denominators are noninferior;
- all-metal, unary, family and arity drift stay within 3 percentage points;
- N TVD is at most 0.05;
- novelty and uniqueness are no worse than -1 percentage point;
- no repair, replacement, survivor filtering or post-generation reranking.

Only after these gates may the same frozen DLM/refiner check downstream
non-harm. Composition improvement is not described as a stability effect.
