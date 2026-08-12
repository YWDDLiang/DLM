# Archived R03 versus original H1-A2 first256 official E_hull completion

Status: `TERMINAL_COMPLETE_R03_STRICT_BEST_RECOVERED`

## Outcome

The stability-only completion finished successfully without a Slurm job. It
reused the terminal archived first256 generation, model-494 refine800,
CHGNet-energy, novelty, and uniqueness ledgers byte-for-byte. No planner,
body, refinement, relaxation, U/N, training, RL, retry, replacement, repair,
filter, or rerank was run.

The official contract was
`mp_api.client.MPRester.get_entries_in_chemsys()`, `compatible_only=True`,
and `GGA_GGA+U`, using database version `2026.04.13`. Of 221 distinct
novel-unique chemical systems, 211 were already resolved by the clean cache.
Exactly ten systems were queried once. The genuinely missing `C-S` system
resolved; nine Yb-containing systems still lacked a complete official unary
reference set and remain explicit `hull_unknown` rather than being counted as
unstable. Final coverage is therefore 212 resolved and 9 unresolved systems.

| arm | generated | joint valid | novel+unique | hull evaluated | hull unknown | strict S.U.N. | meta-S.U.N. |
|---|---:|---:|---:|---:|---:|---:|---:|
| original H1-A2 D1 | 246/256 | 212/256 | 223/256 | 214 | 9 | 24/256 (9.38%) | 118/256 (46.09%) |
| R03 D2 safe-axis | 248/256 | 213/256 | 224/256 | 215 | 9 | **28/256 (10.94%)** | **128/256 (50.00%)** |

R03's official-clean strict count exactly recovers the registered historical
best count of `28/256`. Relative to the incomplete archived-cache evaluation,
R03 changes from 14 to 28 strict successes (`15` false-to-true and `1`
true-to-false) and from 73 to 128 meta successes (`55` false-to-true and no
true-to-false flips). H1-A2 changes from 12 to 24 strict and from 70 to 118
meta successes. The earlier low archived values were therefore dominated by
missing hull coverage, not by a failure to reproduce the archived body or
refinement computation.

The official-clean snapshot does not reproduce every historical frozen-cache
label: H1-A2 is `24/118` versus historical `27/133`, while R03 meta is 128
versus historical 122. Those differences are retained as protocol/database
snapshot differences; only R03 strict matches exactly. Nine attempts per arm
remain hull-unknown because all unresolved systems contain Yb.

## Paired result

| endpoint | H1-A2 only | R03 only | discordant | exact two-sided p |
|---|---:|---:|---:|---:|
| strict S.U.N. | 2 | 6 | 8 | 0.2890625 |
| meta-S.U.N. | 14 | 24 | 38 | 0.14330665 |

This single paired repeat recovers the R03 strict absolute best but does not
establish a statistically significant arm difference.

## Credential and engineering gates

- The credential was sent through a mode-0600 one-time carrier.
- The 5090-side carrier was deleted immediately after the forward transfer.
- The A800-side carrier was destroyed before the first HTTP request.
- No credential value was serialized into source, results, logs, evidence,
  Slurm state, or Git.
- Exit code is `0`; all pipeline, official-query, stability-reevaluation, and
  results-complete markers are present.

## Evidence

- Run root: `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260813_h1_r03_h1a2_archived_first256_official_ehull_completion_v1`
- Source manifest SHA-256: `5f82b415fca42718229baa68f06b649c0b1678b14c0f74c678d1008613b9d813`
- Terminal report SHA-256: `63128c86b9ba0f1688c3f8db543512e6c19d5a68bbabd6dc538f10c1b79f85b0`
- Results Markdown SHA-256: `f89bd7109a437d9fb9ae0e62f170db6c9a18fb59822551114ff7f0558aad64a3`
- Returned evidence archive SHA-256: `6e2d09954328eccb09c75793745490af932ea0fd9244ef68d44677bc280f4af4`
- Local evidence: `evidence/h1_r03_h1a2_archived_first256_official_ehull_completion_v1/`
