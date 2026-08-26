# Planner easy/hard mechanism audit

This audit uses the 1,219 hull-known historical attempts that seeded Planner
self-improvement. Rates below are descriptive and use historical attempt-level
Strict/Meta S.U.N. labels.

## Easy and hard chemistry

| Feature | n | Meta | Strict | Reading |
|---|---:|---:|---:|---|
| halide | 121 | 66.1% | 15.7% | easiest well-supported family |
| all-metal | 358 | 65.4% | 12.6% | easy, but a major shortcut risk |
| other framework | 506 | 60.3% | 11.3% | largely easy metallic/mixed chemistry |
| oxide | 413 | 31.0% | 1.2% | dominant hard family |
| sulfide | 84 | 35.7% | 7.1% | Meta-hard |
| chalcogenide | 56 | 41.1% | 7.1% | intermediate |
| arity 2 | 205 | 56.6% | 8.8% | easier than arity 4 |
| arity 4 | 256 | 32.8% | 3.5% | consistently hard |
| N=1–4 | 166 | 51.2% | 13.9% | easiest cardinality bin |
| N=13–20 | 439 | 41.2% | 4.1% | hardest cardinality bin |

The system can approach 10/50 by increasing halide, all-metal and small-N
frequency, but that is proposal-mix optimization rather than evidence of better
conditional structure realization.

## Structural Plan fields

| Plan field | n | Meta | Strict | Reading |
|---|---:|---:|---:|---|
| volume/atom 5–9 | 38 | 26.3% | 0.0% | strongly unfavorable |
| volume/atom 10–14 | 427 | 38.4% | 2.6% | common but unfavorable |
| volume/atom 15–19 | 328 | 46.6% | 8.8% | transition region |
| volume/atom 20–24 | 208 | 55.3% | 13.9% | favorable and supported |
| volume/atom 25–29 | 129 | 59.7% | 10.1% | favorable and supported |
| volume/atom 30–34 | 50 | 64.0% | 8.0% | Meta-favorable, smaller support |
| volume/atom 35–39 | 24 | 62.5% | 29.2% | promising but sparse |
| triclinic | 359 | 41.2% | 5.6% | common and relatively hard |
| trigonal | 249 | 49.8% | 10.8% | balanced favorable region |
| hexagonal | 164 | 54.9% | 7.3% | Meta-favorable |
| cubic | 33 | 57.6% | 9.1% | favorable but small support |

These fields are suitable mechanism targets because they can move while
formula, composition, arity and N remain fixed. They only matter downstream if
the executor actually consumes rich Plan hints.

## What strong20 V3 changed

- pooled body `504→506`, Direct joint `437→445`, novel `437→443`, N∩U
  `437→442`, Strict `34→37`;
- Meta all-attempt `216→213`;
- halide count `27→38`, contributing Meta `15→24` and Strict `6→7`;
- volume/atom 20–24 count `73→85`, contributing Meta `38→51` and Strict
  `5→9`;
- volume/atom 10–14 count stayed 209, but Strict moved `5→9` while Meta moved
  `72→64`;
- all-metal count changed only `107→112`, yet Strict fell `16→12` within that
  broad category.

V3 therefore mixed three effects: an easy-family shift, a useful structural
hint shift, and within-bin Strict/Meta trade-offs. A scalar reward cannot
separate them reliably.

## Non-RL mechanism roadmap

### V4 now running: Meta guard

- original H1-A2 P0 is the control;
- candidate uses 400 updates, not the drifting 800-update control;
- corrected 20% weighted SFT;
- reward `2×Meta + Strict`;
- proposal-shift coefficient zero;
- only within measured chemistry-stratum residuals receive preference.

### Conditional V5: stratum-preserving Plan-field preference

Run only after V4 is read out.

1. Cross-fit separate Meta and Strict propensities.
2. Preserve total replay weight for every family×arity×N-bin×all-metal stratum.
3. Within each stratum, reweight exact formula and the lattice/SG/volume tuple.
4. Require nonnegative Meta advantage before applying a Strict bonus.
5. Use supervised weighted likelihood only—no policy gradient, reward-model RL,
   best-of-K structure selection, or downstream reranking.
6. If rich-field preference changes Plans but not outcomes, pair it with the
   already trained counterfactual-grounded DLM rather than increasing Planner
   reward strength again.

This design distinguishes an easy-composition shortcut from a genuine
specification-to-realization mechanism.
