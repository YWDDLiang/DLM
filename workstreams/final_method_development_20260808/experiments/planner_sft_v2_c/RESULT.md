# Result: Planner SFT-v2-C

Status: `RAW64_COMPOSITION_GAIN_SCIENTIFIC_STOP_NOT_SHORTLISTED`

Fixed endpoint training completed at 4,505 optimizer updates on the identical
record multiset. On the same raw64 ledger, formal legacy SMACT3.1 comp_valid is
52/64 (81.25%), versus P0 34/64 (53.125%): +18/64 (+28.125 pp). Parse is
62/64, completion 64/64, nonshortcut primary 51/64, all-metal shortcut 1/64,
unique formula 61/64, element coverage 54/94, and parsed mean N 9.79032.
Exact SMACT4 secondary counts are 31/64 valid and 26/64 uniform primary.

Paired legacy flips are 24 candidate-only versus 6 P0-only; exact two-sided
McNemar p=`0.001430906355381012`, with fixed 10,000-draw paired-bootstrap gain
interval `[12.5 pp, 43.75 pp]`. The candidate fails parse, uniqueness, element
coverage, mean-N, and no-new-`ValueError` gates. Terminal report SHA:
`f88060e026c666c6a7d8d0a79dae29bb60786b95aa1ed5fe9029b768cd9dd0f4`.
