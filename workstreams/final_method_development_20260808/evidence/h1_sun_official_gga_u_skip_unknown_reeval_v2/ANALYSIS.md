# Official-MP S.U.N. repair: terminal analysis

## Outcome

The official Materials Project repair completed successfully. Job `31737`
re-evaluated all 16 frozen cells in parallel and job `31738` assembled the
terminal report; both completed `0:0`. No Planner, DLM/body, model-494,
CHGNet, novelty, uniqueness, or reconstruction work was rerun.

The result does **not** support MP-cache contamination as the main cause of
the low Plan1200 V4 strict S.U.N. values. Replacing only the stability path
changes most cell totals by zero to a few samples. Across cells, the mean
clean-minus-old `E_hull` shift is between `-0.000938` and `+0.000075`
eV/atom, far too small to close the historical-versus-V4 gap.

The complete repeat-level strict and meta-S.U.N. tables, with every numerator
and denominator, are in [RESULTS_COMPLETE.md](RESULTS_COMPLETE.md). Exact
machine-readable counts, per-repeat McNemar tests, and all 50,000-draw
bootstrap results are in [terminal_report.json](terminal_report.json).

## Stability contract

- Official `MPRester.get_entries_in_chemsys()` output only.
- `compatible_only=True`, thermo type `GGA_GGA+U`.
- Fresh official query spool; no historical/August cache row was reused.
- 2,550 of 2,630 chemical systems have complete official references.
- The remaining 80 systems all lack the official Yb unary reference. They
  are explicit `hull_unknown`, never silently counted as stable, and are
  excluded only from columns labelled `skip MP unknown`.
- The terminal continuation issued zero new MP requests.

## Effect of replacing the old S path

These are pooled counts shown only to quantify evaluator perturbation; the
three V4 repeats remain the inferential units.

| Frozen panel | Endpoint | Old | Official clean | Change |
|---|---|---:|---:|---:|
| historical R03 refined256, 4 repeats | strict S.U.N. | 117/1024 | 120/1024 | +3 |
| historical R03 refined256, 4 repeats | meta-S.U.N. | 496/1024 | 500/1024 | +4 |
| V4 R03 pre-model494, 3 repeats | strict S.U.N. | 70/3000 | 71/3000 | +1 |
| V4 R03 pre-model494, 3 repeats | meta-S.U.N. | 390/3000 | 395/3000 | +5 |
| V4 R03 post-model494, 3 repeats | strict S.U.N. | 195/3000 | 200/3000 | +5 |
| V4 R03 post-model494, 3 repeats | meta-S.U.N. | 1325/3000 | 1334/3000 | +9 |
| V4 B3 pre-model494, 3 repeats | strict S.U.N. | 52/3000 | 56/3000 | +4 |
| V4 B3 pre-model494, 3 repeats | meta-S.U.N. | 395/3000 | 395/3000 | 0 |
| V4 B3 post-model494, 3 repeats | strict S.U.N. | 188/3000 | 192/3000 | +4 |
| V4 B3 post-model494, 3 repeats | meta-S.U.N. | 1308/3000 | 1313/3000 | +5 |

## Headline strict S.U.N.

| Panel | Repeat-level official-clean fixed-all rates | Repeat-level rates after skipping explicit MP unknowns |
|---|---|---|
| historical R03 refined256 post-model494 | 10.94%, 12.50%, 11.72%, 11.72% | 11.34%, 12.96%, 12.15%, 12.15% |
| V4 R03 pre-model494 | 2.70%, 2.20%, 2.20% | 2.78%, 2.26%, 2.27% |
| V4 R03 post-model494 | 7.30%, 6.70%, 6.00% | 7.48%, 6.85%, 6.15% |
| V4 B3 pre-model494 | 2.10%, 1.10%, 2.40% | 2.16%, 1.13%, 2.47% |
| V4 B3 post-model494 | 7.10%, 6.00%, 6.10% | 7.29%, 6.15%, 6.25% |

Skipping unresolved MP systems therefore raises a typical rate by only about
`0.1`--`0.5` percentage points. Historical R03 remains well above V4 after
using the same official stability contract.

## Paired V4 inference

Rate differences are right minus left. Confidence intervals are hierarchical
paired bootstraps over repeat blocks and paired ordinals (50,000 draws).

| Comparison | Endpoint | Difference (pp) | 95% CI (pp) | P(difference > 0) |
|---|---|---:|---:|---:|
| B3 - R03, pre-model494 | strict | -0.514 | [-1.338, +0.378] | 0.131 |
| B3 - R03, post-model494 | strict | -0.274 | [-1.094, +0.479] | 0.248 |
| post - pre, R03 | strict | +4.425 | [+3.402, +5.449] | 1.000 |
| post - pre, B3 | strict | +4.665 | [+3.536, +5.763] | 1.000 |
| B3 - R03, pre-model494 | meta | +0.000 | [-1.442, +1.511] | 0.497 |
| B3 - R03, post-model494 | meta | -0.717 | [-3.246, +1.641] | 0.300 |
| post - pre, R03 | meta | +32.212 | [+30.013, +34.420] | 1.000 |
| post - pre, B3 | meta | +31.491 | [+27.487, +34.877] | 1.000 |

The clean analysis confirms a large, reproducible model-494 refinement gain,
but finds no reliable R03-versus-B3 difference on these three Plan1200
cohorts. It also confirms that the residual historical-R03 versus V4 gap is
upstream of the official stability evaluator. The leading remaining locus is
the changed Plan sampling/cohort distribution (legacy global-RNG first256
versus three stateless-ordinal first1000 parse-success cohorts), not the MP
cache or the absence of refinement.

## Evidence identity

- Run source manifest SHA256:
  `a1883c9e820b7ca1ebd795180fd9f7ecd71bf26d971c20d41c239c5819fff5e5`
- Submission-continuation source manifest SHA256:
  `ccc0f2e88634efa499971f50ff19837bf0b4ebe67181dee2397e4d713511e344`
- Results Markdown SHA256:
  `b6daf9918f3596700bf7a196da9a8e58afe68f82f9942b89dc8ea65be7a53544`
- Terminal JSON SHA256:
  `651588b75561a8be9de4ebecbddbb204f17f7d4cdb505eaba20ce670999f254b`
