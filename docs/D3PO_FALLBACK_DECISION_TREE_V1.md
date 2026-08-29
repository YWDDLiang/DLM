# DLM stability two-day fallback decision tree v1

Status: preregistered before the main D3PO fixed256 result is available.

## Non-negotiable attribution

```text
C³FD -> outcome-blind composition + N
DLM  -> stable structural realization at that exact composition
model494 -> fixed geometric basin projection
```

No fallback may add C³FD-generated CIFs, external rich structural answers,
best-of-N, survivor filtering, replacement, checkpoint/seed selection, or a
test-derived composition tilt. The public `105/488` result stays immutable.

## Frozen cohorts

- Main D3PO test: 256 reduced-composition-unique Plans; certified prompt SHA
  `21a20c8eca10c30953f486ee00301a872e3c32b853bb0acbe187be2f9d94d3f5`.
- Sealed fallback holdout: another 256 seed17 Plans, disjoint from L6, L7,
  noisy-pair training data, and the main test. Raw SHA
  `ac321dea0592222f1f0b35342163b5fb91df6d95d3d683b9f077ad55cf077094`;
  certified prompt SHA
  `ba0611224d23e0e5e74836789e881ac4a3f6ccd60656f99036fcb3fed4ff4e08`.
  It remains unopened for scientific outcomes unless a fallback below is
  triggered. Because both main and fallback cohorts come from the same C³FD
  seed17 ledger, this is a **sealed composition-disjoint confirmatory split**,
  not an independent Planner-seed test. It may be used exactly once; after any
  fallback evaluation it is permanently burned for this paper cycle.

## Statistical unit and endpoint hierarchy

The independent analysis unit is the set of `256` compositions. Two common RNG
streams are paired process replicates, not independent observations, and may
not inflate the denominator to 512 or 1024.

- For each training seed, first average candidate-minus-base continuous-energy
  deltas over the two streams within each composition.
- Run paired cluster bootstrap over the resulting 256 composition deltas.
- Report the two training seeds separately and every seed×stream cell; a pooled
  cohort CI does not estimate population variance across training seeds.
- S.U.N. uniqueness is cohort-level and nonlinear. McNemar is descriptive,
  never the primary significance claim.

Frozen endpoint order:

1. refined paired CHGNet energy delta;
2. paired official e-hull delta;
3. raw/pre-model494 paired CHGNet energy delta;
4. Meta stable and S.U.N.;
5. Strict, Direct and NU safety/secondary outcomes.

## Main-result classification

Every main cell completes. These are evidence classes, not adaptive hard gates.

| Class | Frozen observation | Interpretation | Next action |
|---|---|---|---|
| P: robust endpoint positive | All four seed×stream refined mean deltas are <0; both training-seed averaged-stream deltas are <0; composition-bootstrap refined CI upper bound <0; official hull agrees; Meta improves for both training seeds with Direct/NU retained | DLM preference learning works | Archive success; no fallback |
| M: continuous positive | All four refined mean deltas are <0 and both seed aggregates are <0, but composition-bootstrap CI crosses 0 or threshold changes are weak | Mechanistic DLM success; sparse threshold crossing | Archive; optionally Fallback B on sealed holdout only |
| I: interface erasure | Both seeds shift raw energy left, but tau800 removes/reverses it | DLM works; DLM→refiner bridge is mismatched | Reuse identical raw bodies for bridge diagnosis; do not retrain/select DLM |
| U: seed-unstable | Only one training seed or one stream improves | Repeats H1-A2/R03 fragility | No seed selection; classify D3PO as non-robust and use Fallback A only if time remains |
| N: objective negative | Validation preference may improve, but neither seed yields a raw paired-energy left shift | D3PO did not alter free generation | Do not amplify with guidance; use Fallback A |
| E: engineering | Failure before scientific updates/samples | No scientific inference | Preserve run/root-cause; one identical recovery only when explicitly authorized |

## Fallback B — late-only policy/reference guidance

Authorized only for class **M**, where both D3PO policies already provide a
replicated energy-improving direction. It is forbidden for N or U.

- Keep the two frozen D3PO step348 policies and base reference.
- At sampling states with remaining geometry-mask fraction `<=0.25`, use one
  predeclared legal-support logit combination:

  ```text
  log p_guided = log p_policy + 0.5 * (log p_policy - log p_reference)
  ```

- Earlier states use the policy unchanged, following published masked-CFG
  evidence that strong early guidance can degrade generation.
- Exactly one strength and one late threshold; no sweep.
- Evaluate both policy seeds under the two common streams on the sealed fallback
  holdout. No completed-sample scoring or reranking.
- This is reported as a sampling sensitivity, not a new trained model.

## Fallback A1 — listwise continuous-energy alignment

This is the unique trained-objective fallback for class **N** or replicated U
when D3PO validation learned a preference direction but main-test raw energy did
not move. The existing source is naturally listwise: about `3706` physically
distinct structures grouped into 3--8 same-composition candidates. Pairwise
D3PO expanded these groups into `5857` rows.

For a composition group, use robust centered post-model494 energy rewards and
one shared typed geometry mask for every candidate. With
`S_theta(y)` equal to the current reference-corrected, `1/(p|G|)` normalized
masked-sequence score, freeze a LAIR-style objective:

```text
r_i = -robust_z_within_composition(E_i)
p_i = softmax(r_i / tau)
w_i = p_i - 1/K
L_list = -sum_i w_i S_theta(y_i)
         + lambda/K * sum_i S_theta(y_i)^2
         + 0.20 * L_best_denoising
```

- continuous energy gaps and every candidate are retained;
- centered weights sum to zero within composition, preventing chemistry/N
  shortcuts;
- the quadratic term bounds implicit-reward magnitude;
- every composition has total training weight one;
- two training seeds and one final checkpoint each remain mandatory;
- inference is still one Plan -> one trajectory, never listwise sampling or
  best-of-N;
- numeric `tau/lambda` must be frozen by train-only score/gradient calibration
  before GPU work, with no main/fallback outcome access or sweep.

This is described as a shared-noise masked-listwise variant inspired by
[Diffusion LAIR](https://arxiv.org/abs/2605.26491), not as an exact transfer of
its continuous-image objective.

## Fallback A1b — masked NCA-style absolute alignment

Use only if pairwise D3PO raises relative preference metrics while winner
likelihood, body validity or novelty falls. Unlike pairwise margins, an
NCA-style term constrains the absolute sign of the implicit reward. It remains
secondary because the frozen candidate pool mixes source policies and the DLM
score is a denoising-ratio surrogate, so the original NCA assumptions are not
exact. No separate reference-only resampling is authorized in this sprint, and
NCA receives no paper-facing confirmatory run if listwise uses the sealed
holdout.

## Fallback A2 — DLM-internal self-intent trailer

This is a reserve, not a second confirmatory fallback. It may replace listwise
only if the listwise data/implementation fails a preregistered zero-GPU
feasibility audit **before** any fallback-holdout outcome is generated. Once
listwise burns the sealed holdout, self-intent cannot receive a paper-facing
claim in this cycle. It must not consume main/fallback outcome labels.

The multi-agent review selected exactly two field-specific trailer slots after
the unchanged dynamic `7+4N` body:

```text
original dynamic body, <VPA_Q0..Q7>, <CN_ENV_0..7>
```

- `VPA_Q`: eight train-only quantiles of `log(volume/N)`, a robust global scale
  that directly affects over-dense/over-sparse initial basins;
- `CN_ENV`: eight train-only medoids of a coarse CrystalNN effective-
  coordination histogram, representing the local topological basin that
  model494 may refine within but not cross.

The choice is evidence-bound rather than decorative. In the archived mechanism
diagnostic, volume-bin-matched structures have official e-hull median `0.0733`
versus `0.0809 eV/atom` when mismatched and Meta-stable rates `330/534=61.8%`
versus `257/451=57.0%`; Strict barely changes, so VPA is expected to affect
continuous/Meta outcomes first. Conversely, a lattice/space-group match is
associated with a worse median (`0.0881` versus `0.0683 eV/atom`). These are
non-causal diagnostics, but they reject the unsupported shortcut “add an SG
token and stability will rise.”

Appending rather than prepending preserves every historical body position.
Use small field-specific embedding/classification heads rather than resizing
the full LLaDA vocabulary. Both slots start masked at inference, are predicted
by the DLM, then frozen for the unchanged lattice -> all-X -> all-Y -> all-Z
schedule; the parser/model494 consume only the original body.

Do not add Bravais/space-group/Wyckoff/prototype/oxidation tokens in this
sprint. A space-group label without a Wyckoff-aware parameterization is not a
symmetry constraint; prototype labels risk retrieval attribution; density and
packing duplicate VPA; oxidation is already a C³FD chemistry sidecar and does
not distinguish same-composition polymorphs.

Required properties:

- composition and N remain visible and exact;
- latent targets come only from the training crystal itself;
- the DLM predicts latent intent and geometry jointly or coarse-to-fine;
- no energy/hull value is provided as an inference condition;
- dynamic length, legal supports, source hashes and token initialization are
  frozen before GPU work;
- intent dropout and body-to-intent reconstruction are required to prevent
  teacher-latent dependence or posterior collapse;
- two training seeds and a matched latent-base control are mandatory;
- the sealed fallback holdout is the only paper-facing test;
- total concurrent usage stays within six A800, four-to-eight CPUs per GPU.

Before full training, report train/chemsys-held-out VPA accuracy, CN-medoid
accuracy, ordinal error, latent entropy and an oracle-latent body diagnostic.
If oracle intent does not improve raw energy, this route stops without GPU
training. If oracle works but self-prediction fails, the bottleneck is
composition-to-intent and no external rich Plan may be substituted as a main
result.

## Bridge-only diagnosis for class I

Use the already generated raw bodies. Compare the current clean-body-as-`x_tau`
path with one legal forward-corruption construction at tau800 under common
refiner RNG. This cannot change DLM generation, denominator, or select bodies.
It is an interface attribution experiment, not a replacement headline.

## Forty-eight-hour execution ledger

1. **Training terminal:** preserve both seed logs, canaries, manifests,
   step348 hashes, validation and failure ledgers.
2. **Main generation/refinement:** one six-A800/48-CPU job for exactly six
   cells; preserve body/refiner manifests per cell.
3. **Offline evaluation:** one six-A800/48-CPU job for six refined plus six raw
   cells; preserve Direct/full-CHGNet outputs and all stderr.
4. **Official hull:** one union manifest and at most one fresh query; unknown is
   missing, never unstable.
5. **Terminal report:** MD/JSON/CSV with per-seed, per-stream and pooled paired
   energy/hull, ECDF/quantiles, SUN, CI and McNemar.
6. **Decision:** assign exactly one class P/M/I/U/N/E before opening the sealed
   fallback holdout.
7. **Archive:** copy positive or negative terminal package, update
   BUILD_STATUS/PAPER_STORY, run tests, commit and push. Failures remain beside
   their root-cause note and are never overwritten.

For every GPU stage report three resource quantities separately: expected
GPU-hours, scheduler kill ceiling from requested GPUs×walltime, and observed
GPU-hours. The expected `18 A800-hours` for the main path is not presented as a
hard scheduler ceiling.

## Multi-agent method ranking

1. Current shared-noise D3PO.
2. For replicated energy-positive but threshold-weak M: late-only guidance.
3. For objective-negative N/U: listwise continuous-energy alignment.
4. Masked NCA-style alignment is an exploratory method reserve, not another
   confirmatory holdout arm after listwise.
5. VPA+CN self-intent may replace listwise only after a pre-outcome feasibility
   NO-GO; it cannot follow a listwise confirmatory result on the same holdout.
6. Online SEPO/diffu-GRPO is deferred beyond the two-day sprint because it
   requires repeated on-policy generation/refinement and correct transition
   accounting under irreversible full-axis commitment.
7. Particle Gibbs/SMC/tree search is permanently excluded from the DLM claim:
   it is trajectory best-of-N/search and breaks one-Plan-one-trajectory
   attribution.
