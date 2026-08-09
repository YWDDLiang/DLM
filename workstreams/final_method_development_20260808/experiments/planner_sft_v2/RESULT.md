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
The formal result is a scientific stop rather than a promotion; the
user-authorized Direct and S.U.N. comparison remains pending.
