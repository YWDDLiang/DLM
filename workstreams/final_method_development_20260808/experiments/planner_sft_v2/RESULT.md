# Result: Planner SFT-v2

Status: `RAW256_AND_FOUR_CELL_TERMINAL_NO_PROMOTION_PROTECTED_P0`

Fixed endpoint training completed at 4,505 optimizer updates with one complete
36,038-record ledger pass and final accumulation size 6. The common raw64
ledger SHA is
`f9d6f1bd99f80b37a46ace88d226ce16d92a5692f925d17c757a53434b74c6d1`.

| metric | P0 | SFT-v2 | delta |
|---|---:|---:|---:|
| parse | 64/64 | 63/64 | -1 |
| completion | 64/64 | 64/64 | 0 |
| legacy SMACT3.1 comp_valid | 34/64 (53.125%) | 52/64 (81.25%) | +18/64 (+28.125 pp) |
| legacy nonshortcut primary | 17/64 | 51/64 | +34 |
| all-metal shortcut | 17/64 | 1/64 | -16 |
| exact SMACT4 valid (secondary) | 33/64 | 36/64 | +3 |
| exact SMACT4 uniform primary | 14/64 | 32/64 | +18 |
| unique formula | 64/64 | 63/64 | -1.5625 pp |
| fixed-alphabet element coverage | 64/94 | 60/94 | -4.2553 pp |
| parsed mean N | 10.90625 | 10.34921 | -0.55704 |

Paired legacy flips are 23 candidate-only versus 5 P0-only. Exact two-sided
McNemar p=`0.0009122341871261597`; the fixed 10,000-draw paired bootstrap
95% interval for the absolute gain is `[14.0625 pp, 42.1875 pp]`.

The registered raw64 scientific gate fails on element coverage, absolute
mean-N drift, and a new `ValueError` generation failure class. V12 also labels
the stage engineering-failed because the old P0 schema lacks the embedded
validator field expected by the new no-charge identity check; the formal
SMACT3.1 recomputation itself is complete and unaffected. Terminal report SHA:
`5342a4f8e0f5695bfff8680406569d55916cdb66ce8a1d7aabd9f6c2d06a9f0c`.

The user-authorized, non-promotional V14 raw256 diagnostic produced exactly
256 raw all-attempt rows for both P0 and SFT-v2, with zero exit codes and
per-arm success markers. The common ledger SHA is
`d5a3ac87458969816a0b27313fd9deecae47d2ddb10289ec08b9d93c5db48669`;
the P0 and SFT-v2 raw SHAs are respectively
`201ca978486260fd19ddd5908f847b8b4aa00f6d3593d4e7a3862bc373583151`
and
`eebb958a75343b11de91e66808232a4c9aba3052dfa540298d9f3149f4ddcaf1`.
Parse counts are 254/256 and 255/256.

The complete local-only exact SMACT4 secondary audit passed with 100% official
witness parity and manifest SHA
`ed17201d01a5f4f3a601892309ad671b45fe55d41cd15b1252aac8053bf4c6c4`:

| raw256 exact SMACT4 metric | P0 | SFT-v2 | delta |
|---|---:|---:|---:|
| valid | 135/256 | 122/256 | -13 |
| uniform primary | 56/256 | 101/256 | +45 |
| all-metal shortcut | 72/256 | 10/256 | -62 |

Thus the chemistry-first SFT signal transfers strongly to uniform-primary
generation and suppresses shortcuts, but does not improve the broader exact
SMACT4-valid count on this ledger.

V18 completed the formal legacy SMACT3.1 computation on the same raw256
ledger before an old-P0-schema identity gate returned job exit `2:0`:

| raw256 legacy SMACT3.1 metric | P0 | SFT-v2 | delta |
|---|---:|---:|---:|
| composition-valid | 128/256 (50.0%) | 195/256 (76.171875%) | +67 (+26.171875 pp) |
| nonshortcut primary | 56/256 | 186/256 | +130 |
| all-metal shortcut | 72/256 | 9/256 | -63 |
| parse | 254/256 | 255/256 | +1 |
| fixed-alphabet coverage | 80/94 | 78/94 | -2 |

The complete V18 report SHA is
`1503bb66d670174edd30bef401c6ebbf4f4f8c05f53a8f7326a7b760dfe45b61`.
V20 then failed only because it passed the repaired assembly-source SHA to a
manifest that intentionally binds the earlier raw-generation source SHA.
V21 and V22 stopped before SBatch on generator guards and are immutable. V23
also stopped before SBatch because its new root retained `_v20_` and hit the
exact stale-marker guard. V24 used a clean root; normal-CPU assembly job
`31329` completed `0:0`, with SFT terminal-report SHA
`cf51e4067981dfecee6e07e508edc84a138b0b7a93003e91a487db66f1938e5b`
and stage-summary SHA
`445d58c2b567acb688e554b511917340eb1ff172de22df8d49d11c0045945f46`.
Raw, model, ledger, evaluator, and reported science counts remain unchanged.
The formal result is a scientific stop rather than a promotion.

## Current-run four-cell Direct/S.U.N. terminal

The user-authorized comparison completed under one current-run pipeline and
one 256-attempt denominator per cell. Direct composition validity applies GCD
before the frozen R03 validity code; all four cells share the same completed
455-system MP cache and frozen R03E S.U.N. evaluator.

| cell | comp/joint | structure | COV-P | COV-R | novel-unique | strict S.U.N. | meta S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0+B0 (M00) | 216/256 | 247/256 | 96.4844 | 85.2200 | 219/256 | 21/256 | 116/256 |
| SFT-v2+B0 (M10) | 214/256 | 246/256 | 96.4844 | 67.5768 | 207/256 | 13/256 | 91/256 |
| P0+B3 (M01) | 223/256 | 252/256 | 98.4375 | 83.2854 | 223/256 | 19/256 | 119/256 |
| SFT-v2+B3 (M11) | 213/256 | 247/256 | 96.4844 | 71.6781 | 210/256 | 16/256 | 102/256 |

At fixed B0, SFT-v2 changes composition/joint by -2/256 (-0.78125 pp;
95% CI [-6.640625, 5.078125] pp; exact McNemar p=0.893853), strict S.U.N.
by -8/256, and meta S.U.N. by -25/256 (-9.765625 pp; 95% CI
[-17.96875, -1.5625] pp; p=0.0260756). At fixed B3, it changes
composition/joint by -10/256 and meta S.U.N. by -17/256, with intervals that
include zero. The raw256 legacy composition gain therefore does not transfer
to the complete downstream pipeline. SFT-v2 is not promoted and P0 remains
the protected Planner.

Array `31374` and assembler `31375` both completed `0:0`. Terminal report SHA:
`cdd23113f86e97c5f747e7c97cf24a531231d68b32420cdf03909d8de2806fb6`.

## Separate P0 Plan1200 downstream attempt

A later P0-only planner stage successfully produced three distinct raw1200
batches with 1,189/1,193/1,194 parse successes and froze the first 1,000
parse successes per repeat. The downstream R03/B3 comparison nevertheless
failed before body generation because those frozen rows do not carry the
top-level `parsed` key required by the body runtime; the preflight omitted
that exact schema assertion.

Consequently this route yields no new Planner, CrysLLMGen, S.U.N., or
post-refine result and does not change the protected-P0 decision or the
SFT-v2 scientific stop. The requested native full-1,000 diffusion-refine
supplement also remains unfulfilled. See
`H1_P0_PLAN1200_R03_B3_PREPOST_REPEATS3_EXECMODE_FAILURE_V3.md`.
