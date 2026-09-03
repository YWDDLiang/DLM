# Energy-Shaped SPAD: Basin-Aligned Programmed Backfill

Status: **APPROVED; user authorized immediate one-day implementation**

## 1. One-line method

> C3FD restricts Llama to chemically reachable Plans, Llama programs which
> species the crystal DLM constructs and revisits, and a frozen
> diffusion–potential teacher reshapes that suffix-visible backfill posterior
> toward valid lower-energy endpoints without leaving the reference trust
> region.

The final method is still called **SPAD**.  “Basin-aligned backfill” is a
training mechanism inside SPAD, not a second headline acronym or a detachable
post-processing module.

## 2. Scientific question

> How can a scientific LLM program a non-causal crystal DLM so that every
> discrete revision preserves exact chemistry and periodic feasibility while
> learning which revision reaches a lower-energy continuous basin?

This directly joins the teacher's three requirements:

1. scientific knowledge changes the Llama action distribution;
2. Llama decides the crystal-DLM execution order;
3. continuous diffusion and potential feedback improve that same programmed
   DLM operation rather than appearing as a separate pipeline.

## 3. One connected computation

```text
C3FD chemical support + Llama residual logits
                    |
                    v
       exact Plan + Llama species program
                    |
                    v
       7+4N DLM predictor in programmed order
                    |
                    v
  Llama-selected early site is re-masked; full suffix stays visible
                    |
                    v
  legal XYZ-triplet posterior is shaped by diffusion–potential utility
                    |
                    v
       committed raw crystal -> fixed model494 tau800 -> final crystal
```

This is **hierarchical scientific action shaping**, not raw-logit fusion across
incompatible tokenizers:

- C3FD defines reachable chemical actions;
- Llama supplies residual chemical preference and the species program;
- the program defines a non-contiguous DLM state/action sequence;
- PBC geometry defines legal backfill actions;
- the continuous teacher redistributes probability only among those actions.

## 4. Module roles are unambiguous

| Object | Role | Train-time use | Inference-time use |
|---|---|---|---|
| C3FD | symbolic scientific support | fixes legal Planner actions | constrains Llama Plan decoding |
| Planner Llama | learned policy/program | predicts exact species permutation | emits Plan and program |
| crystal DLM | discrete structure policy | learns programmed backfill posterior | predicts `7+4N` lattice/XYZ tokens |
| model494 `R_494^800` | continuous transition | maps a candidate revision to its terminal basin state | common final refiner |
| CHGNet `E_phi,F_phi` | learned potential/critic | endpoint energy target; raw force is a mechanism diagnostic | absent from final DLM sampling |

The terminal basin utility is

\[
V_{800}(x)=E_\phi(R_{494}^{800}(x)).
\]

model494 is the state transition; CHGNet is the energy/force evaluator.  Neither
is called an exact physical world model or a DFT oracle.

## 5. The DLM-native action

For a frozen Plan `c`, Llama program `o`, and completed SPAD body, select the
next registered backfill site `i` in program order.  Re-mask only its XYZ
triplet and retain lattice, species and every other site—including the entire
future suffix—as state `s_i`.

An action is a complete coordinate triplet

\[
a_i=(X_i,Y_i,Z_i)\in\mathcal A_i(s_i).
\]

`A_i` is represented by the DLM's existing coordinate special tokens:

- 0.01 fractional increments on each axis;
- `000` and `100` are one torus value;
- candidate displacement is measured with strict triclinic minimum images;
- exact `N`, species, lattice and non-active tokens are immutable;
- a triplet producing a sub-0.5 A PBC contact is outside the action set.

Validity and stability are therefore **lexicographic**, not two weighted
rewards.  An action enters `A_i` only if it preserves the exact Plan/body
schema, positive finite lattice, atom count/type order, non-duplicate periodic
coordinates, the 0.5 A triclinic PBC boundary and fast graph construction.
Only then may terminal energy change its probability.  The source no-op is
always retained when the source body is valid.  If no legal alternative exists,
the target is the no-op delta distribution and the example cannot trade
validity for lower energy.

Unlike an arbitrary completion-level preference, all compared actions share
one bitwise-identical input state.  Unlike an AR edit, the right-hand suffix is
not regenerated.

## 6. Finite-action posterior

The DLM gives sequential X/Y/Z logits for one transaction.  For each sampled
legal triplet, define its reference score from the exact three conditional
decisions:

\[
S_{\rm ref}(a_i\mid s_i)
=\log p(X_i\mid s_i)
+\log p(Y_i\mid s_i,X_i)
+\log p(Z_i\mid s_i,X_i,Y_i).
\]

This is an explicit finite-action distribution for one transaction; no claim
is made that the DLM exposes a normalized likelihood over every full crystal
or one globally consistent Gibbs trajectory.

For `K=4` legal triplets, action zero is always the original no-op XYZ and the
other three are legal samples from the frozen reference DLM.  The teacher
target is

\[
q^*(a_i\mid s_i)\propto
\exp\left[
S_{\rm ref}(a_i\mid s_i)
-\beta\widetilde V_{800}(x[a_i])
\right].
\]

`beta` is maximized subject to

\[
D_{\rm KL}(q^*\Vert p_{\rm ref})\le 0.05\;\text{nat}.
\]

Thus stability is optimized inside explicit validity and trust regions rather
than through a manually weighted “validity + energy” logit sum.  Novelty and
uniqueness remain fixed-denominator evaluation outcomes in version 1; explicit
reference KL is the only anti-collapse mechanism claimed.

Training minimizes

\[
L_{\rm terminal}=D_{\rm KL}(q^*\Vert p_\theta)
\]

on this suffix-visible backfill transaction.

Equivalently, version 1 solves

\[
\min_q\;\mathbb E_q[V_{800}]
+\tau D_{\rm KL}(q\Vert p_{\rm ref})
\quad\text{s.t.}\quad
\operatorname{supp}(q)\subseteq\mathcal A_i^{\rm valid}(s_i).
\]

The zero-invalidity support constraint is enforced before energy labels are
read.  Unknown-energy legal actions stay in accounting but receive no energy
preference; they are never interpreted as high or low energy.

## 7. Potential target and force diagnostic

Terminal `V_800` is the only energy target in version 1.  Raw CHGNet force is
recorded at the same Llama-selected backfill site as a scientific mechanism
diagnostic.  For an adjacent legal coordinate-token displacement `Delta r_a`,

\[
-\Delta E_\phi(a)
=F_{\phi,i}^{\top}\Delta r_a+O(\|\Delta r_a\|^2).
\]

The analysis reports whether force work predicts the ordering induced by
`V_800` after exact token quantization.  It does not alter the target, loss or
candidate weights.  This preserves the potential/force interpretation without
combining raw and terminal objectives at different trajectory points.  A
future dense-credit extension is justified only if this same-transaction
correlation is positive.

Force is never copied into a token residual, and stress is absent from version
1.  Lattice and XYZ actions receive terminal `V_800` supervision through their
effect on the final crystal.

## 8. Training objective

Every source group contributes two equal-weight views:

1. ordinary/SPAD CE retention;
2. terminal backfill posterior.

The full loss is

\[
L=L_{\rm SPAD\text{-}CE}
+\lambda_TL_{\rm terminal}
+\lambda_{\rm KL}L_{\rm ref\text{-}KL}.
\]

A single 64-group train-only gradient-norm calibration fixes the two
contribution scales.  It is not a hyperparameter search and no evaluation
outcome changes the coefficients.  Exact composition is prefilled, and
inference retains the existing schema/PBC support.

### 8.1 Teacher availability and forward inference

`V_800` is available **only while constructing MP20-train supervision**.  It
is not queried while a new crystal is being decoded.  SPAD-E distils the
offline action posterior into DLM parameters.

Deployment is one forward generation process:

1. C3FD–Llama produces the Plan and species program;
2. the DLM predictor completes lattice and every site in programmed order;
3. with a complete future canvas now available, SPAD re-masks one earlier XYZ
   transaction while leaving the suffix unchanged;
4. the trained SPAD-E DLM chooses the revised XYZ from Plan, program, lattice
   and visible future context alone;
5. after all discrete backfill is complete, model494 tau800 runs once as the
   common terminal transition.

There is no inference-time CHGNet call, counterfactual model494 rollout,
candidate energy evaluation, reranking or look-ahead oracle.  The key DLM
advantage is precisely that step 3 can retain future tokens and rewrite the
past; an AR executor would have to discard and regenerate that suffix.

## 9. MP20-train-only teacher data

MP20 supplies all training compositions and reference structures.

### Retention view

- all 27,136 MP20-train bodies;
- frozen predicted Llama-pointer program, not oracle contact-tree order;
- ordinary, predictor and suffix-visible backfill mask states;
- origin, `000/100`, within-species order and equivalent-basis transforms are
  applied jointly to all views.

### Energy view

- 2,048 outcome-blind MP20-train Plans;
- one frozen BS predictor trajectory per Plan;
- one Llama-programmed suffix-visible backfill state per trajectory;
- `K=4` legal DLM triplets from that exact state: mandatory no-op plus three
  frozen-reference samples;
- 8,192 complete candidates total;
- one model494 tau800 endpoint and CHGNet endpoint energy per candidate;
- raw CHGNet force at the selected site for correlation analysis only.

Invalid/ungraphable actions remain in fixed accounting but do not receive
finite energy preferences.  A group with fewer than two energy-known legal
actions contributes CE only.

Before token scoring, every candidate is mapped to the same body-site order,
lattice basis and periodic origin.  No prospective/test outcome, MP online
query, replacement, reranking or final-evaluation selection enters this data.

## 10. Bounded compute

Measured extrapolation from completed jobs:

- 8,192 BS bodies on four A800: approximately 1--1.5 hours;
- 8,192 model494 tau800 transitions on four A800: approximately 4--5 hours;
- batched raw CHGNet force diagnostics: minutes rather than hours;
- two equal-compute LoRA cells: approximately one hour;
- total data plus training: approximately 5--7 hours before evaluation.

The active data stage uses one four-GPU job alongside the two-GPU response
evaluation. Training uses one two-GPU job containing two isolated one-GPU
cells. This respects the two-job/six-GPU limit;
training does not start while unrelated jobs occupy the required resources.

## 11. Minimal identifiable experiment

Paper names remain simple; historical B0/BC/BP/BR/BS labels move to the
ablation appendix.

| Paper cell | Internal meaning | Equal compute | GPUs |
|---|---|---:|---:|
| SPAD | current schedule-matched model | reference | 0 |
| SPAD-CE | one additional equal-data CE/KL epoch | yes | 2 |
| SPAD-E | replace matched neutral targets with terminal `q*` | yes | 2 |

Both trained cells use one seed, 348 updates with six microgroups per update
(2,088 group exposures, at least one shuffled pass through trainable groups),
LR 5e-6 and exactly the same source groups/mask counts.
SPAD-E is preregistered as the method.  A positive SPAD-E receives a second
seed.

Mechanism evaluation requires no fourth training cell:

- score the same frozen energy-labelled backfill states with their suffix
  visible and with that suffix hidden;
- measure probability mass on terminal-preferred and locally downhill actions;
- test whether the energy alignment gain specifically uses future context.

This tests whether terminal preference depends on future context at the exact
state where the method operates.

Evaluation order:

1. fixed-denominator body accounting and fast structural validity;
2. raw CHGNet energy/force diagnostics and fixed-denominator N/U;
3. model494 tau800 and surrogate S.U.N. only for the preregistered full method;
4. expensive Direct is `DEFERRED_COST` unless required for a final table.

The current official 256 cohort is development-only because its result informed
this design.  A confirmatory paper claim requires a newly frozen cohort before
SPAD-E outcomes and a separately authorized official/independent endpoint.

## 12. What is new relative to failed attempts

| Retired attempt | Why it failed | Energy-shaped SPAD change |
|---|---|---|
| BTRD | copied one model494 endpoint into token targets | compares legal actions at one common backfill state |
| Force-Score | copied continuous force direction despite token quantization | force proposes; decoded finite energy verifies |
| old 64-group listwise | only 33 groups had a valid anchor and used a generic candidate objective | SPAD's high-validity programmed backfill creates 2,048 DLM-native groups |
| G2 expected geometry | averaged multimodal token probabilities into fictitious coordinates | every triplet remains a discrete mode |
| arbitrary BPA revision 0 | compared pseudo-likelihoods under different masks | one explicit suffix-visible XYZ transaction defines the entire posterior |

## 13. Paper contribution structure

The paper has one core method and three linked technical contributions.

### Core method

**Scientific LLM-programmed crystal diffusion with constrained, energy-shaped
non-causal backfill.**

### Contributions

1. **Scientific LLM action shaping.** C3FD injects typed chemical reachability
   into Llama decoding; Llama retains learned residual choice and emits an exact
   species construction program.  This yields near-perfect composition
   validity without reducing the LLM to a fixed enumerator.
2. **Programmed non-causal DLM execution.** The Llama program becomes
   non-contiguous `7+4N` denoising order, and suffix-visible backfill lets the
   DLM revise an earlier species anchor from complete future crystal context.
3. **Diffusion–potential-shaped backfill.** model494 terminal transitions and
   CHGNet endpoint energy define a bounded posterior over the same
   Llama-selected DLM revision, connecting discrete generation to continuous
   stability without inference-time reranking.  Force supplies a first-order
   mechanism analysis of that local action, not a second training target.

The third contribution is not “run a refiner after the DLM.”  The downstream
transition and potential teach the DLM's own non-causal backfill distribution.

## 14. Theory claims and limits

Claims intended for the main text:

1. under one fixed energy functional, same-composition energy ordering equals
   that functional's surrogate-hull ordering because the reference term is
   constant;
2. on the finite legal backfill action set, exponential energy reweighting is
   the solution of a KL-constrained expected-utility problem;
3. force work is a first-order explanation of local coordinate-token energy
   change; version 1 tests its correlation with terminal action ordering rather
   than using it as supervision;
4. suffix-visible state is a DLM-specific channel through which future crystal
   context changes an earlier atom's energy-shaped posterior.

Explicit limits:

- CHGNet/model494 supervision is a surrogate, not DFT proof;
- the finite backfill posterior is not a normalized full-crystal likelihood;
- force correlation is diagnostic and not independent potential validation;
- version 1 does not claim a lattice-stress token theorem.

## 15. Review decision log

### Skeptic review — revision 0 rejected

Accepted corrections:

- restricted hull equivalence to one energy functional;
- replaced unproven full-sequence joint by an explicit finite-state posterior;
- required one bitwise-identical masked state;
- retained energy-gap magnitude;
- made the current prospective cohort development-only;
- added equal-compute CE and explicit reference-KL controls;
- excluded invalid rows from finite energy preference;
- removed lattice stress from version 1;
- replaced oracle contact order with frozen predicted programs.

### Constraint review — revision 1 rejected as unbounded

Accepted corrections:

- capped terminal labels at 2,048 groups x four actions;
- removed the unbounded all-atom local-work teacher from version 1;
- froze one 5--7 hour data/training plan;
- required a new deterministic backfill-state collator and dynamic PBC action
  support rather than reusing the old random listwise wrapper;
- reduced training to two equal-compute one-GPU cells inside one two-GPU job;
- retained unknown-energy rows in denominator accounting.

### Paper/user review — revision 1 required major revision

Accepted corrections:

- kept one public method name, SPAD;
- moved energy alignment specifically into Llama-programmed suffix-visible
  backfill rather than arbitrary mask states;
- defined model494 as transition and CHGNet as potential/force critic;
- collapsed the method into one finite backfill posterior;
- added suffix-visible versus suffix-hidden mechanism evaluation;
- separated final method, equal-compute control and ablation names;
- requires existing mainline documents to be updated together if the arbiter
  accepts this revision.

### First arbiter review — revision 2 returned REVISE

Accepted decisions:

- no-op is mandatory action zero in every `K=4` set;
- version 1 removes the undefined discovery dual and reports fixed-denominator
  N/U under an explicit reference KL;
- version 1 removes raw-work supervision because it is evaluated at a
  different trajectory point from terminal `V_800`;
- SPAD-E is the only preregistered full method; SPAD-CE is its equal-compute
  control;
- 2,048 groups and one seed are a development decision only; confirmation
  requires seed two and a newly frozen cohort.

## 16. Final arbitration request

Revision 3 now contains only one new learning object:

\[
q^*(a_i\mid s_i)\propto
p_{\rm ref}(a_i\mid s_i)
\exp[-\beta\widetilde V_{800}(x[a_i])],
\]

where `s_i` is the Llama-programmed suffix-visible backfill state, action zero
is no-op, all four actions are PBC-legal XYZ triplets, and the KL budget is
fixed.  Force is analysis only.  The remaining implementation prerequisites
are the common-state collator, dynamic legal-action builder and one-GPU trainer
preflight; they add no scientific alternatives.

### Final arbiter — APPROVED

The arbiter accepted revision 3 without a remaining conceptual blocker.  The
frozen method is SPAD-E: one Llama-programmed suffix-visible state, mandatory
no-op plus three legal reference-DLM actions, terminal `V_800` reweighting,
0.05-nat trust region and explicit reference KL.  Force is diagnostic only;
there is no discovery dual, force/stress loss or inference-time critic.

Implementation must complete the common-state collator, dynamic PBC action
builder, energy normalization/KL solver, one-GPU step-0 preflight and
suffix-visible versus suffix-hidden mechanism evaluation before training.
