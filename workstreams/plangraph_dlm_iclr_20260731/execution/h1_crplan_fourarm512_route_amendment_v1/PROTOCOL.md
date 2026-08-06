# H1 CR-Plan four-arm 512 route amendment V1

## Scope and status

This is one new preregistered, Plan-only mechanism experiment authorized after
the clean E1 physical-feasibility pass. It is not a repair of the frozen V4
engineering terminal and does not overwrite or reinterpret paired-32, V4, or
E1.

The run ends at the four-arm 512 terminal. It cannot automatically start Body
generation, refinement, Direct structure evaluation, S.U.N., paired-64,
paired-256, independent panels, training, checkpoint selection, promotion, or
formal G3/G4.

## Frozen causal contrast

The four arms are:

1. `off`: original P0 sampling;
2. `grammar_only`: formula syntax and atom-budget legality;
3. `terminal_only`: grammar plus the shared fail-closed terminal charge gate;
4. `full_prefix`: the same terminal gate plus exact preterminal charge
   reachability.

All arms use the same model, adapter, tokenizer, rich seven-line prompt,
sampling parameters, batch size one, ordinal role, and independent 512-entry
science ledger. The only treatment is the legal token support during formula
generation. Each legal-support distribution is renormalized once and sampled
once. Retry, replacement, repair, filtering, reranking, and fallback are
forbidden.

## Endpoint policy

Terminal-only and full-prefix share exactly one endpoint:

- unary and true all-metal compositions retain the frozen Direct shortcuts;
- otherwise every element must exist in the frozen SMACT oxidation-state
  table;
- table-missing non-shortcuts fail closed as
  `charge_applicable_oxidation_state_missing`;
- uniform neutral witnesses are primary;
- mixed-valence neutral witnesses are legal but nonprimary;
- Pauling is measured by the frozen composition evaluator and is not a
  decoding constraint.

Shortcut-valid attempts are disclosed separately and excluded from the
primary gain. Missing-state elements may not act as numerically neutral future
suffixes.

## Independent ledger

The base seed is deterministically derived before generation from:

`h1_crplan_fourarm512_route_amendment_v1|science_ledger_v1`

Its SHA-256 is
`46cc5f7595aa311b5fea7d8fbd49619c37b39b86772f99b21292ab2fc6a76412`,
and the frozen 31-bit base seed is `1187798901`. Ordinals are exactly
`0..511`; every arm uses the same `planner_sampling/shared` derived seed for
an ordinal. This ledger is disjoint from paired-32 and E1.

## Primary endpoint and gates

The primary contrast is `full_prefix - terminal_only`, on the raw
all-attempt denominator of 512 per arm. A composition is measured with the
frozen CrysLLMGen-aligned SMACT evaluator. The primary count includes
charge-applicable valid compositions and excludes unary/all-metal shortcuts.

All gates were fixed before opening the ledger:

- primary non-shortcut composition-valid gain is at least `+11/512`;
- at least 5% of charge-applicable full-prefix attempts have a real
  preterminal full-versus-terminal support difference;
- full-prefix has zero charge-applicable terminal-certificate failures;
- full-prefix parse and completion counts are not below terminal-only;
- unique-formula rate and fixed-alphabet element-coverage rate each lose at
  most 2 percentage points;
- full-prefix shortcut-valid count does not exceed terminal-only;
- ordinal, seed, prompt, tokenizer, policy-contract, identity, diagnostics,
  and forbidden-operation checks all pass.

Raw composition-valid counts including shortcuts, uniform-primary counts,
mixed-valence counts, reason/stratum taxonomies, paired discordance/McNemar,
support removal, blocked newline counts, removed probability mass, state work,
and all raw latency values are secondary reports.

## V4 state and E1 latency interpretation

V4 remains an immutable engineering stop at its registered 100,000-state
budget. This amendment neither changes the state definition nor reuses that
gate. Exact semantic state work is retained and reported only.

E1 is the frozen prerequisite showing physical viability. Because the four
arms are separate Slurm array tasks and may occupy different nodes/cold
paths, this panel introduces no new cross-job latency-ratio gate. Latencies
must be complete and finite, all raw values remain in the attempt records,
and any OOM, timeout, exception, or fallback is an engineering failure.

## Execution

One GPU array `0-3%2` runs one immutable arm per task. A `normal`-partition
assembly job runs with `afterany` dependency so engineering failures still
produce a fail-closed terminal when possible. Both declared partitions must be
verified by `sinfo` before the first `sbatch`.

The terminal may be:

- `mechanism_gate_pass`: eligible only for a separately authorized paired-64;
- `scientific_stop`: retain frozen H1 and stop CR-Plan;
- `engineering_failure`: preserve evidence and stop.

No downstream job is submitted by this package.
