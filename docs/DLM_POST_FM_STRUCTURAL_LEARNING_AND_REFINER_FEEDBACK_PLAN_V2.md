# Post-F/M structural learning and refiner-feedback roadmap v2

Date: 2026-08-31

Status: authorized to begin only after the fixed F/M prospective run is
terminal and fully disclosed

Supersedes `DLM_POST_FM_REFINER_DISTILLATION_GEOMETRY_PLAN_V1.md`.

## Method decision

The current paper remains DLM-centered. C3FD is the composition-correctness
contribution, F/M are interfaces that connect C3FD to the historical Rich Plan,
and the masked crystal DLM remains responsible for lattice and site realization.

F and M are retained as usable Planner routes when their final generated-body
composition validity is at least `95%` on the fixed requested denominator.
This is a retainability rule only: it does not require a S.U.N. gain and it does
not turn Planner validity into a DLM-stability claim. Both F and M results stay
in the report. M remains the preferred integrated route when it passes because
it carries frozen C3FD state into the Llama Rich Expander; F remains the
formula-only ablation and engineering fallback.

## Why decoding constraints are no longer the primary route

The current `7+4N` representation already has enough numerical granularity for
MP20-scale structural learning:

- lattice lengths: `0.1 Angstrom` bins;
- lattice angles: `1 degree` bins;
- fractional coordinates: `0.01` bins;
- exact atom count and element multiset special tokens;
- maximum `N=20` under the current MP20 contract.

It can therefore represent gross collisions, cell volume, periodic pair
distances, and coordination-scale differences. The missing information is not
primarily a new token. The existing loss treats lattice/coordinate fields as
independent token-classification targets and does not explicitly compare the
periodic geometry induced by those tokens.

Do not add a new inference-time collision mask as the next contribution. First
teach the model structural validity during training. Existing schema, exact
composition, nondegenerate-lattice, and exact-duplicate guards remain unchanged
as parser/interface invariants rather than a new scientific method.

## Geometry-correlation diagnosis

The hypothesis that weak lattice/site coupling causes low raw structural
validity is **high confidence**; the claim that it explains thermodynamic
stability by itself is only **medium confidence**.

The current implementation provides implicit context but no explicit relational
objective:

1. `forward_process()` masks geometry positions independently. The
   bidirectional Transformer can attend to visible lattice/site tokens, but
   `compute_loss_components()` applies cross entropy independently to each
   selected token.
2. The typed logits for `LA/LB/LC`, angles, and every `X/Y/Z` position are not
   converted into a joint periodic structure during the loss. A low token NLL
   therefore does not imply legal volume, safe PBC distances, or a coherent
   coordination environment.
3. The production exact-axis sampler commits lattice, all X, all Y, and all Z
   as separate confidence-ordered groups. Later axes can condition on earlier
   axes, but no joint score compares two complete coordinate assignments.
4. The prebuilt exact-length JSONL stores one site order and one coordinate
   origin. Unlike the CSV training path, the JSONL loader does not perform
   origin-shift augmentation. Equivalent periodic structures can therefore
   create avoidable token-level disagreement.

The historical evidence has the same direction:

- joint-XYZ scheduling caused duplicate coordinates and large body/Direct
  losses, so merely committing more coordinates together is not sufficient;
- single-token CTV prediction was near chance, indicating that stability is a
  sequence/global property;
- positive-only SGTC geometry CE improved teacher-forced NLL but not S.U.N.;
- model494 changes raw Direct `188 -> 457` on the matched L6 evidence, showing
  that continuous relational geometry repairs a large fraction of failures.

R03 supplies a narrower but directly relevant schedule result. D1 committed
the global X axis, then Y, then Z. D2_SAFE_AXIS preserved that invariant while
grouping sites inside each axis according to PlanGraph. On the frozen
four-process panel it changed body completion `246 -> 248/256`, pooled joint
validity by `+5/1024`, Strict S.U.N. `99 -> 117`, and Meta S.U.N. `523 -> 496`.
The four repeats shared one Plan cohort and are not independent Planner seeds.

R03 therefore supports only two inferences for the new design:

- relational/commitment structure is a real DLM design variable;
- even a small intervention can polarize the hull distribution, so a new
  geometry module should begin as an identity-preserving correction rather
  than replacing the pretrained hidden state.

R03 did not train or test a residual network and cannot be cited as evidence
that a residual adapter improves stability.

Consequently, Phase G must first measure whether collisions/metric failures
actually explain raw invalidity, then add an invariant joint-geometry loss. It
must not assume that improved structural validity automatically implies lower
hull energy.

## Phase G — geometry-aware masked-DLM training

### G0. Failure taxonomy and quantization sufficiency audit

First decompose existing raw failures on frozen attempts into mutually
exclusive primary causes:

- body parse/schema failure;
- exact-composition mismatch;
- nonpositive/near-singular lattice metric;
- PBC minimum distance below `0.5 Angstrom`;
- CrystalNN/graph construction failure after the above pass;
- structurally valid but high raw/refined energy.

Report the fraction of raw invalidity explained by lattice/periodic-distance
failures, overall and by N/arity/family. This is the direct test of the geometry-
correlation hypothesis.

Before GPU training, round-trip MP20 train/validation CIFs through the current
special tokens and compare original versus tokenized structures:

- parse and exact-composition preservation;
- minimum PBC distance and collision-label changes;
- lattice metric and volume-per-atom error;
- species-pair radial-distribution and coordination error;
- Direct structural-validity changes.

If tokenization itself changes structural validity by more than `1%`, stop and
redesign the representation. Otherwise freeze `7+4N`; no vocabulary extension
is needed for this cycle.

### G1. Periodic geometry objective

Add losses computed from the legal-token softmax at randomly masked geometry
positions. No target CIF or structure is exposed at inference.

```text
L_G = L_masked_CE
    + lambda_metric  * L_metric_tensor_and_log_VPA
    + lambda_pair    * L_species_pair_periodic_RDF
    + lambda_overlap * L_short_distance_barrier
    + lambda_coord   * L_coordination_histogram
```

- Decode expected lattice/coordinate bin values differentiably from the typed
  special-token distributions.
- Use the lattice metric tensor and minimum-image periodic distances.
- Compare sorted/species-pair distance distributions rather than site indices,
  making the target invariant to same-species permutation.
- Apply random global fractional translation and same-species site permutation.
- Keep N and element tokens visible and unsupervised during geometry-only masks.
- Calibrate the four loss scales once by train-only gradient norms; run no grid.

MP20 positive structures teach the geometric manifold. They do not by
themselves teach which generated polymorph is better, so Phase G is followed by
refiner feedback rather than more plain CE epochs.

### G2. Conditional periodic residual relation adapter

Do not start with a graph-network rewrite. First run G1 with the unchanged DLM
backbone. Promote a small relation adapter only when all of the following are
true on held-out chemical systems:

- token round-trip validity passes, so representation resolution is not the
  bottleneck;
- G1 improves metric/RDF/overlap auxiliary errors;
- raw structural validity does not improve consistently across the two fixed
  training seeds.

The adapter keeps the same vocabulary and sampler. For each structure it pools
the element/XYZ hidden states into at most 20 site states, forms all periodic
site pairs, encodes species pair plus minimum-image distance with radial basis
features, applies two low-rank message-passing layers, and scatters a correction
back to the site and lattice token states:

```text
delta_h = W_out RelationMP(LN(h), G, fractional_coordinates, species)
h_prime = h + delta_h
```

`W_out` is initialized exactly to zero. The candidate must therefore reproduce
the G1 logits to numerical tolerance before its first optimizer step. The
residual touches lattice/XYZ states only; prompt, N, and element anchors retain
the original path. Scalar distance/metric features make the adapter periodic
and rotation-invariant. Its output is trained with the G1 losses and the
original CE/reference anchor.

If G2 fires, compare two matched continuations from the same G1 checkpoint:

- control: the same additional updates with no relation adapter;
- candidate: the zero-initialized periodic residual adapter with the same data,
  masks, optimizer steps, and seeds.

This prevents ordinary extra training from being attributed to the residual
path. The scientific contribution is the periodic relational correction; the
skip connection is the conservative integration mechanism.

This G2 trigger distinguishes two failures: if relation losses themselves do
not learn, the targets/objective are wrong; if they learn but generation does
not change, the factorized token backbone needs an explicit relational path.
No extra CE epoch or schedule search is allowed between G1 and this decision.

## Phase R — model494-in-the-loop self-improvement

Model494 is an automatic scientific annotator, but `refined` is not synonymous
with `good`. It can return an invalid or high-energy structure and can rescue an
off-manifold raw body, which previously caused post-refiner preference training
to damage raw validity. Labels must therefore be hierarchical.

### R1. Low-variance basin SFT

Use MP20-train compositions only with chemical-system-held-out validation.
For each frozen M Plan, generate one raw DLM body and run model494 tau800 once.
Keep every attempt in accounting. A refined body can become a positive CE target
only when it:

1. preserves the exact atom multiset;
2. parses and passes Direct structural validity;
3. has finite post-refiner energy.

Serialize that single refined geometry into the same `7+4N` tokens. Mix it with
the original MP20 teacher body and a frozen-reference anchor. Do not select the
lowest-energy member of a group in this SFT stage. Train fresh LoRA adapters
from the shared pretrained crystal LLaDA base with two fixed seeds.

This stage provides a stable warm start but is not sufficient by itself: it
imitates acceptable basins and does not learn relative thermodynamic quality.

### R2. Group-relative refiner preference

For each train-only composition and fixed M Plan, sample a fixed `K=4` group of
raw DLM bodies, refine all four with common registered model494 seeds, and keep
all failures. Construct one lexicographic reward inside the composition:

1. parse and exact-composition validity;
2. raw Direct structural validity and minimum-distance safety;
3. refined Direct structural validity;
4. lower same-composition refined CHGNet energy;
5. lower raw CHGNet energy and smaller raw-to-refined correction as secondary
   tie breakers.

Center and scale rewards only inside each composition. Every composition has
total weight one. Use the existing shared-mask, legal-support,
reference-corrected K-way DLM objective with a best-valid denoising anchor.
This is an offline **group-relative diffusion preference** method, not vanilla
AR-GRPO and not a claim of exact sequence likelihood.

Vanilla GRPO is not the first choice for a masked DLM: dLLM sequence
probabilities are intractable and ratio estimates are high variance. Recent
dLLM work addresses this with ELBO/group estimators or stabilization rather
than copying AR-GRPO directly:

- GDPO: https://arxiv.org/abs/2510.08554
- StableDRL: https://arxiv.org/abs/2603.06743
- diffusion D3PO: https://arxiv.org/abs/2311.13231

Run at most one on-policy refresh only after the frozen offline method improves
held-out raw Direct and raw/refined energy for both training seeds. Do not begin
with multi-round online RL.

### R3. Official hull handling

GPU compute jobs never access the network. Generation/refinement first writes a
complete immutable union. A login-side collector may then perform one batched
MP reference query and freeze the returned phase ledger.

Official hull is validation and final-evaluation evidence, not an online reward
queried after every policy update. If train-only official hull is ever included
as a secondary preference label, its source compositions and query manifest
must be frozen before training and a disjoint prospective cohort must remain
untouched.

## Phase L — C3FD-conditioned CrysLLMGen fallback

This route is worth implementing only if the geometry-aware and
refiner-feedback DLM still fails to improve raw realization. It changes the
paper center and is therefore the final fallback, not a parallel main run.

```text
frozen C3FD composition/state
        -> M soft-prefix projector
        -> autoregressive Llama crystal generator
        -> model494 tau800
        -> common evaluation
```

Treat the frozen C3FD encoder, M projector, and Llama LoRA as one callable
component. C3FD fixes formula/N and supplies the soft prefix; Llama generates
the structure. Preserve composition by prefilling the same atom-count and
element-slot special tokens used by the DLM rather than returning to fragile
free-form CIF element text. Llama predicts lattice and coordinates
autoregressively in the same `7+4N` representation, which keeps DLM/AR
comparison matched.

Apply the same R1 basin SFT and R2 K=4 refiner-feedback pool. AR likelihoods are
tractable, so standard DPO/GRPO-style optimization is easier here, but the same
raw-validity-first reward and reference/coverage regularization remain
mandatory. Crystal-specific group RL also warns that an energy-only reward can
collapse candidate coverage; a relevant current example is CrystalGRPO:
https://arxiv.org/abs/2608.06582.

The fallback paper story would be `C3FD-constrained CrysLLMGen`, with the DLM
reported as a controlled negative/ablation. It must not be silently substituted
for the DLM method after observing the same prospective outcomes.

## Ordered execution checklist

- [ ] Finish and disclose the current F/M generation, raw/refined evaluation,
  and official S.U.N. result.
- [ ] Retain every F/M route with requested-denominator final composition
  validity `>=95%`; classify DLM quality separately.
- [ ] Run the CPU-only `7+4N` quantization-sufficiency audit.
- [ ] Decompose existing raw failures into parse, composition, lattice,
  collision, graph, and energy-only classes before training.
- [ ] Implement and unit-test differentiable metric/RDF/overlap/coordination
  losses without changing inference decoding.
- [ ] Train two geometry-aware DLM seeds on MP20 train only and run a fixed
  train/chemsys-validation raw-first screen.
- [ ] Add the zero-initialized two-layer periodic residual relation adapter only
  if the frozen G2 trigger fires; verify step-0 G1-logit equality and compare
  against a same-update no-adapter continuation.
- [ ] Build the one-trajectory model494 basin-SFT dataset and train two fresh
  anchored adapters.
- [ ] Build one immutable K=4 group pool and run the offline shared-mask
  group-relative preference update.
- [ ] Freeze the complete DLM method on train/validation before creating a new
  prospective cohort or querying official hull.
- [ ] If the DLM route remains raw-negative, implement the matched
  C3FD-conditioned AR CrysLLMGen fallback; otherwise keep it documentation-only.
- [ ] Report C3FD, F/M, geometry-aware DLM, refiner feedback, and any AR fallback
  as separate SUPPORTED/CANDIDATE/UNSUPPORTED contributions.

## Explicitly out

- new decode-time geometry masks as the primary method;
- treating every model494 output as positive;
- energy-only rewards that can prefer raw-invalid bodies;
- vanilla dLLM GRPO without a valid likelihood/ratio estimator;
- MP network calls inside GPU jobs;
- best-of-N, survivor filtering, replacement, or result-selected seeds;
- launching the AR fallback before the DLM structural-learning route is
  terminal.
