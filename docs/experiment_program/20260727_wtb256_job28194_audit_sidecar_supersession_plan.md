# WTB-256 job28194 audit-sidecar supersession

## Purpose

Job `28194` ended before any scientific attempt because the atomic installer
and runtime Gate-A code disagreed about one authorization label. All installed
bytes, the frozen scientific contract, the A800 allocation, and the exact
`diff_meets_diff` environment were correct. This supersession repairs only
that audit-sidecar inconsistency.

The supersession execution identity is
`wq_wyckoff_chart_retraction_confirmatory256_sup28194_v1`. The scientific
identity remains
`wq_wyckoff_chart_retraction_confirmatory256_v1`, including contract SHA
`293c026d2f371b592a81e8e4d3982b4cb65ae3b0d90b82bf72a639caae24b77a`.
Keeping the scientific identity unchanged preserves ordinals `512–767`,
training seed `11`, sampling seed `101`, all paired source/noise identifiers,
the three `R/U/T` arms, direct metrics, exact S.U.N. evaluation, and promotion
rules.

## Exact correction

1. Register the old WTB-256 local-preparation authorization in runtime Gate-A.
2. Register one new supersession authorization in both installer and Gate-A.
3. Require the installer and runtime authorization sets to be exactly equal.
4. Invoke `GateALock.load(...)` with the installed new patch before creating
   the Slurm claim.
5. Use new claim, record, output, log, patch, archive, and job-name identities.
6. Preserve job `28194` and every associated artifact unchanged.

## Fail-closed execution

Before claim creation the wrapper checks:

- the exact new installed patch record and submission authorization;
- the unchanged scientific contract SHA;
- the immutable job28194 terminal audit, record, claim, stdout, stderr, and
  empty output directory;
- one A800, eight CPU, 96 GiB, 18 hours, no array;
- `GateALock.load(...)` against the exact new patch;
- absence of every new claim, record, output, and active job identity.

Only after all checks pass may the wrapper create the unique claim and invoke
`sbatch` once. Submission failure, runtime failure, or scientific failure is
terminal and is never retried.

## Scope boundary

This is evaluation only. It does not authorize training, fine-tuning,
replacement sampling, best-of selection, reranking, altered denominators, a
different scientific contract, more than eight CPU per A800, or changes to
unrelated jobs. Scientific results will be audited before any later
training-or-paper decision.
