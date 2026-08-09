# Result: Planner SFT-v2

Status: `RAW64_COMPOSITION_GAIN_SAFETY_GATE_STOP_USER_AUTHORIZED_DIAGNOSTIC_DOWNSTREAM`

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
SMACT4-valid count on this ledger. Formal legacy SMACT3.1 assembly256, Direct,
and S.U.N. metrics remain pending; no promotional claim is made.
