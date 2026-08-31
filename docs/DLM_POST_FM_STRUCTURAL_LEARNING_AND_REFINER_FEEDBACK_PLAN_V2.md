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

## Phase G — geometry-aware masked-DLM training

### G0. Quantization sufficiency audit

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
- [ ] Implement and unit-test differentiable metric/RDF/overlap/coordination
  losses without changing inference decoding.
- [ ] Train two geometry-aware DLM seeds on MP20 train only and run a fixed
  train/chemsys-validation raw-first screen.
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
