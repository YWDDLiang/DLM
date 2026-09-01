# C3FD+Llama / Compact-V2 G2 final report

Date: 2026-09-01  
Claim scope: one fresh prospective cohort, one pre-registered stream; not a
multi-seed robustness claim.

## Outcome

The final pipeline is complete:

`C3FD legal support + Llama typed scoring → Compact-V2 masked DLM → periodic-relation G2 → model494 tau800 → official MP hull`

The fused Planner was composition-valid on all `256/256` prospective requests.
G2 refined S.U.N. reached Strict `24/256 = 9.375%` and Meta
`117/256 = 45.703%`. This recovers the historical H1-A2 operating range, but
misses the registered `10%/50%` stretch target by two Strict and eleven Meta
samples. No result was deleted, replaced, reranked, or used to choose a seed or
checkpoint.

## Frozen prospective contract

- Cohort: 256 unique exact compositions and 256 chemsystems; exact overlap with
  MP20 and every registered development cohort is zero.
- Cohort ledger SHA-256:
  `90109e1b5398bba679ad11b92bfadc85f050af44f758f6cbc024bfe88234e842`.
- Fused Plan: requested/parsed/composition-valid `256/256`; Plan SHA-256
  `5f1ae510fb35d7bbe0b5da4b32b0302f49d78dae653c5c31493db8a2219a54cb`.
- Matched DLM noise and Plan ledger for BASE/G2; stream17, exact-axis,
  temperature 0.7, model494 tau800; one trajectory per request.
- Fixed requested denominator: 256 for every metric.

## Final official results

| Stage | Arm | Direct | N | U | NU | Hull known | Strict S.U.N. | Meta S.U.N. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | BASE | 118/256 | 250 | 251 | 250 | 241 | 5/256 (1.953%) | 41/256 (16.016%) |
| raw | G2 | 121/256 | 251 | 252 | 251 | 245 | 9/256 (3.516%) | 47/256 (18.359%) |
| refined | BASE | 251/256 | 219 | 251 | 219 | 246 | 19/256 (7.422%) | 111/256 (43.359%) |
| refined | G2 | 251/256 | 216 | 252 | 216 | 247 | 24/256 (9.375%) | 117/256 (45.703%) |

Five Yb-containing chemsystems are explicit official-hull unknowns because the
fresh MP GGA/GGA+U response lacks a Yb unary reference. Unknowns remain in the
fixed denominator and are never counted stable.

## Matched evidence

- Raw Direct: `118→121`; this is a small gain.
- Raw CHGNet G2−BASE: `+35.78 meV/atom`, 95% bootstrap CI
  `[-265.15,+337.59]`; no reliable raw energy improvement.
- Raw official hull G2−BASE: `+82.52 meV/atom`, CI
  `[-214.42,+377.41]`; no reliable raw hull improvement.
- Refined CHGNet G2−BASE: `-16.05 meV/atom`, CI
  `[-24.76,-8.23]` on 249 common rows.
- Refined official hull G2−BASE: `-16.43 meV/atom`, CI
  `[-25.26,-8.62]` on 244 common known rows.
- Refined Strict S.U.N. discordance: G2-only 6, BASE-only 1, exact McNemar
  `p=0.125`.
- Refined Meta S.U.N. discordance: G2-only 22, BASE-only 16, exact McNemar
  `p=0.418`.

The continuous refined endpoint supports a system-level G2 benefit. The raw
endpoint and binary S.U.N. endpoints do not establish a seed-robust DLM
stability improvement.

## Research-question answers

1. **Can C3FD be integrated into a learned Llama Planner without looking like
   post-hoc enumeration? — SUPPORTED.** C3FD provides internal legal action
   support and Llama changes the decision distribution; final prospective Plan
   composition validity is `256/256` without repair or rejection.
2. **Does ordinary Compact-V2 teacher SFT solve raw structural stability? —
   UNSUPPORTED.** Earlier raw Direct was low, and the final BASE is only
   `118/256`; model494 supplies most of the structural conversion.
3. **Does the periodic relation adapter teach raw DLM stability? — CANDIDATE.**
   G2 adds three raw Direct outcomes, but raw continuous CIs cross zero.
4. **Does G2 improve the complete DLM+refiner system? — SUPPORTED within one
   stream.** Refined hull shifts left by `16.43 meV/atom` with a CI below zero
   and S.U.N. changes `19/111→24/117`.
5. **Were the prospective `10%/50%` targets reached? — NO.** The final endpoint
   is `9.375%/45.703%`; outcomes are retained unchanged.

## Paper contribution boundary

- **SUPPORTED:** C3FD-constrained learned composition planning with
  `256/256` prospective composition validity.
- **SUPPORTED:** end-to-end Compact-V2 DLM + fixed model494 recovery to
  H1-A2-like S.U.N. on a fresh cohort.
- **CANDIDATE:** G2 periodic relation learning as a DLM contribution; report the
  refined hull improvement, the small raw gain, and the one-stream limitation.
- **UNSUPPORTED:** claims that G2 alone solves raw stability, that the target
  `10%/50%` was met, or that model494 improvement proves the DLM learned a
  stable raw manifold.

## Compute and execution

- Final prospective Planner sampling: job39128, 1 A800, 292 s,
  `0.081 A800-h`.
- Final generation/refinement: job39137, 2 A800, 3594 s,
  `1.997 A800-h`.
- Final offline evaluation: job39139, 4 A800, 4247 s,
  `4.719 A800-h`.
- The fused-Planner/G1/G2/final route consumed approximately `13.179 A800-h`
  including preserved engineering negatives. The preceding Compact-V2
  teacher-SFT and canary chain consumed `16.929 A800-h`; combined evidence-chain
  usage is approximately `30.108 A800-h`, excluding abandoned alignment and
  earlier rich-interface diagnostics.
- Official MP querying and finalization used login-side CPU only.

## Reproducibility and artifacts

- Final official run: `runs/c3fd_g2_final_official_20260901_v1`.
- Final report: `runs/c3fd_g2_final_sun_20260901_v1/_SUCCESS`.
- Positive archive:
  `archive/c3fd_g2/final_sun_positive_20260901/_ARCHIVE_SUCCESS`.
- Official source manifest SHA-256:
  `138e547f9f2d19c52e55586b96d7c9394d38a6330da0d49a9cf017dca641b6a6`.
- Final JSON SHA-256:
  `1b99aa33d3d6072006e17309874866af862067f4becd37caeec5f154f99b3070`.
- Output hash verification passed for all four cell reports and attempt files.
- Finalizer implementation/tests commit: `1552044`; local and remote unittest
  suites passed `9/9` in total.

The MP credential was supplied only to the one official-query child process,
was removed immediately afterward, and was never copied into Git, docs,
commands, logs, hashes, manifests, archives, or ambient shell state. Post-query
audit found zero query processes and zero ambient credential variables.

