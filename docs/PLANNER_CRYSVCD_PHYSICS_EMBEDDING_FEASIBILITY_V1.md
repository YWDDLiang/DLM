# Planner Physics-Inspired Element Embedding Feasibility V1

Date: 2026-08-27

Status: isolated design/audit only; not part of the active DLM training arm.

## Source mechanism

CrysVCD represents a composition as element–oxidation-state/count pairs. Its
formula Transformer combines three vectors for every pair:

```text
learned element-valence token embedding
+ projected full/valence-shell electronic configuration
+ learned count embedding (0..20)
```

The model then generates a charge-balanced composition before the geometric
diffusion stage.

## Two-problem factorization

The Planner must solve two distinct problems, with different semantics and
failure handling.

### Problem 1 — correct compound composition

This is a hard-contract task. The model must generate a chemically meaningful
element/count assignment before any soft crystal properties are proposed.

Required constraints:

- valid element identities and positive integer counts;
- deterministic `N = sum(counts)`;
- one canonical formula renderer, with no independent formula/count arithmetic;
- charge neutrality for ionic compounds under at least one supported oxidation
  assignment;
- explicit zero-oxidation representation for alloys rather than treating every
  all-metal formula as automatically desirable;
- no invalid mixed-sign oxidation assignment for one element;
- parser and validator failure retained in the raw denominator.

Recommended factor:

```text
p(composition) = p(element-valence/count sequence)
```

This factor uses the CrysVCD-inspired electronic-configuration/count
representation. Formula, elements, counts and N are rendered deterministically
from the generated sequence.

### Problem 2 — properties appropriate for that composition

This is a conditional distribution task, not a set of independent categorical
labels. A valid marginal lattice or space-group bucket can still be incompatible
with the generated formula or with the other soft fields.

Recommended factorization:

```text
p(rich Plan)
  = p(composition)
  * p(anion, charge | composition)
  * p(lattice | composition, anion, charge)
  * p(spacegroup | composition, lattice)
  * p(volume | composition, lattice, spacegroup)
```

Interpretation:

- `anion` and `charge` are chemistry checks and should be deterministic or
  near-deterministic whenever the formula supports them;
- `lattice`, `spacegroup` and `volume` are soft, potentially multimodal
  properties and should remain sampled distributions;
- impossible lattice–space-group combinations receive a hard compatibility
  mask;
- plausible but rare combinations remain available and must not be removed by
  a mode-seeking argmax rule.

Train the property factor with both marginal CE and a joint compatibility loss.
Construct hard negatives by keeping composition fixed while replacing one or
more soft fields with a tuple drawn from another composition/structure. The
model must rank the matched tuple above the corrupted tuple.

```text
L_planner
  = L_composition
  + lambda_chem * L_anion_charge
  + lambda_soft * (L_lattice + L_spacegroup + L_volume)
  + lambda_joint * L_tuple_compatibility
```

Downstream DLM body/stability conversion may later provide an auxiliary
realizability label, but it must not replace the teacher property targets or be
used to alter the frozen raw1000 confirmation cohort.

## Why it is not a direct drop-in

The current H1 Planner:

- is a pretrained text LLaMA with LoRA;
- emits the six-field/seven-line `h1_rich_plan_v1` text;
- uses ordinary BPE text tokens for element symbols and digits;
- freezes base embeddings under the current LoRA target modules;
- computes formula-derived N/elements/counts only after parsing.

Adding a numerical electronic-configuration vector to ordinary BPE rows would
modify text tokens shared with unrelated contexts and would not reliably bind a
symbol to one oxidation state. A faithful adaptation needs explicit
element-valence/count tokens or an auxiliary head.

## Selected one-model pilot

Keep the public Planner output unchanged. The selected pilot uses one
autoregressive Planner to emit a train/internal species--valence--count prefix
followed by the conditionally generated property tuple:

```text
valence_plan:
<EL_Fe_OX_2P> <COUNT_1>
<EL_Fe_OX_3P> <COUNT_2>
<EL_O_OX_2N>  <COUNT_4>
end: valence_plan

rich_plan:
formula: Fe3O4
anion: oxide
charge: neutral_plausible
lattice: ...
spacegroup: ...
volume: ...
end: plan
```

The implementation reuses the existing fixed-width `P01..P07` count-valence
representation. Repeated element identities are legal when their oxidation
states differ, so mixed-valence formulas such as Fe3O4 are represented as
Fe(II)x1 + Fe(III)x2 + O(-II)x4 and deterministically collapse back to one
formula/count list. The same model then predicts charge/lattice/space-group/
volume fields. At the API boundary, the internal line is rendered to the
canonical six scientific fields plus `end: plan`.

This uses the central CrysVCD idea now--explicit element-valence species,
integer count embeddings/tokens, exact charge-balance search, and mixed-valence
support--without claiming to reproduce CrysVCD's separate formula Transformer.
An electronic-configuration projection is a second, isolated ablation after
the representation itself passes composition and property gates.

Implementation options retained for later ablation:

1. **Auxiliary valence head:** share the LLaMA hidden state, predict explicit
   valence/count tokens with a small trainable head, and retain the existing
   rich-Plan text head. Lowest risk to the public tokenizer.
2. **Special-token prefix:** add explicit element-valence/count tokens to the
   tokenizer, initialize their trainable deltas from electronic configurations,
   supervise the prefix, then deterministically strip it before exposing the
   seven-line rich Plan.
3. **Dedicated composition Transformer:** reproduce the CrysVCD architecture and
   feed its formula into the existing rich-field Planner. Highest fidelity but a
   different model and contribution; out of current scope.

The first pilot is the single text model with the explicit compact prefix. It
is the smallest experiment that tests whether the physical representation
improves both hard composition validity and conditional property plausibility.
If it passes, an auxiliary electronic-configuration projection can initialize
or regularize the same prefix token states without creating a second Planner.

## Audit correction (2026-08-27)

The first serializer audit reported 100% round trip but only about 25--26%
known/neutral oxidation assignments and 0% on raw1000 because it inherited a
single-valence SMACT annotation and did not include oxidation coverage in its
authorization gate. That authorization was invalidated before any Planner GPU
training.

The corrected audit:

- searches exact charge balance with uniform states first and adjacent,
  same-sign mixed valences second;
- uses the original CrysVCD species table first, then a version-recorded
  pymatgen common-state extension and only then its full integer oxidation
  catalog; every selected extension element and catalog tier is reported;
- emits explicit zero-valence species for the supported alloy branch;
- never treats an unknown assignment as unstable or silently drops it;
- requires >=95% assignment and neutral coverage on train and validation;
- requires raw1000 coverage to be no more than 3 percentage points below
  train;
- checks exact composition and soft-field round trips before authorizing GPU
  training.

## Required data audit

Before implementation, measure on the exact Planner train/val and frozen raw1000:

- oxidation-state assignment coverage;
- unique element–oxidation-state pairs;
- mixed-valence frequency;
- count range and count-token coverage;
- charge-neutral assignment rate;
- formulas with multiple valid oxidation assignments;
- all-metal cases where oxidation state zero is the appropriate representation;
- lanthanide/actinide/exotic-state exclusions relative to the current Plan
  support.

No missing assignment may silently become `unstable` or be dropped from final
evaluation.

## Feasibility gates

- at least 95% train/validation oxidation-assignment coverage, with every
  uncovered category disclosed;
- raw1000 coverage no worse than train by more than 3 pp;
- rich-Plan parse rate, exact formula/count/N correctness and charge-neutral
  rate noninferior to P0;
- teacher-matched joint property tuple NLL improves without worsening any soft
  field marginal NLL;
- lattice–space-group hard compatibility violations are zero;
- volume plausibility and density-support diagnostics are noninferior;
- no increase in all-metal or unary shortcuts;
- auxiliary valence accuracy positive on exact formula/count matching;
- downstream DLM evaluation remains a separate factorial arm.

## Experimental isolation

The active DLM sufficiency and stability-preference program uses the frozen P0
Plans. A physics-embedding Planner candidate may be evaluated only after:

1. the CPU-only coverage audit passes;
2. the DLM sufficient base is frozen;
3. the candidate is compared against P0 with the same frozen DLM/refiner;
4. proposal-mix changes and within-Plan conversion are reported separately.

It must not be used to rescue a failed DLM candidate or to replace the frozen
raw1000 two-round test cohort.

## Post-factorial decision (2026-08-28)

The compact text-prefix pilot is terminally negative. Although deterministic
teacher assignment covered about 96% of train/validation and 94.1% of raw1000,
the generated count-valence arm produced only `247/491=50.31%` emitted-neutral
parsed Plans and shifted strongly toward all-metal formulas. This establishes
that tokenizing correct oxidation annotations is not sufficient for a BPE
autoregressive decoder to execute coupled integer charge arithmetic.

If Planner work resumes, only the following one-backbone design remains
scientifically justified:

1. the shared Planner transformer predicts a latent state from the request;
2. a species/count head predicts element–oxidation-state candidates and counts;
3. a finite-state/dynamic-programming decoder permits only sequences that can
   still complete to exact N and net charge zero;
4. a separate rich-property head predicts anion, lattice, space group and
   volume with calibrated uncertainty;
5. the public output is deterministically rendered to the same seven-line rich
   Plan;
6. sampling is calibrated within frozen family/arity/N strata so easier
   all-metal/halide mass cannot replace difficult oxide/sulfide mass.

This is still one learned Planner backbone; the constrained decoder is an
inference algorithm rather than a second language model. CrysVCD's
electronic-configuration projection may initialize the species embeddings, but
only after the constrained species/count head itself passes exact composition,
neutrality and drift gates. It must not again be tested as free-form text.

The soft-property head and formula head must be evaluated separately. The DLM
should consume a soft field only when its held-out calibration is adequate;
otherwise that field becomes `unknown` while formula/N/elements/counts remain
hard anchors. This reliability-aware interface is the Planner counterpart of
the active DLM conditioning ablation.
