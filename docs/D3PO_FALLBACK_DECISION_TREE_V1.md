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
  triggered.

## Main-result classification

Every main cell completes. These are evidence classes, not adaptive hard gates.

| Class | Frozen observation | Interpretation | Next action |
|---|---|---|---|
| P: robust endpoint positive | Both training seeds shift raw and refined paired energy left in both common streams; pooled official hull agrees; Meta S.U.N. positive with Direct/NU retained | DLM preference learning works | Archive success; no fallback |
| M: continuous positive | Both seeds/streams shift energy and hull left, but Strict/Meta threshold counts are small or unchanged | Mechanistic DLM success; sparse threshold crossing | Archive; optionally Fallback B on sealed holdout only |
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

## Fallback A — DLM-internal structural intent

Authorized for class **N** or replicated U, subject to a zero-GPU label audit
and one frozen schema. It must not consume main/fallback outcome labels.

The intended minimal design is a small self-predicted latent channel derived
from train-only MP20 structures, with all latent states masked at inference.
Candidate contents are limited to crystal-system/symmetry class and a
train-quantiled volume/packing descriptor. The multi-agent review must choose
at most one compact schema before implementation.

Required properties:

- composition and N remain visible and exact;
- latent targets come only from the training crystal itself;
- the DLM predicts latent intent and geometry jointly or coarse-to-fine;
- no energy/hull value is provided as an inference condition;
- dynamic length, legal supports, source hashes and token initialization are
  frozen before GPU work;
- two training seeds and a matched latent-base control are mandatory;
- the sealed fallback holdout is the only paper-facing test;
- total concurrent usage stays within six A800, four-to-eight CPUs per GPU.

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
