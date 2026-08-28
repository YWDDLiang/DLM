# C³FD-v2.2 CPU audit v1 — engineering failure

Date: 2026-08-28

This run produced no scientific result and authorized no GPU work.

- Input: frozen `c3fd_semantic_v21_step1b_20260828` vocabulary and supported
  `(family,N,arity)` strata.
- Frozen runtime limit: 300 seconds.
- Observed: still running at 312 seconds; terminated with SIGTERM.
- Cause: the exact suffix-existence method recursively materialized every
  viable action at every deeper state instead of stopping after the first
  witness.
- Recovery: preserve the exact action set, but short-circuit only the internal
  Boolean existence query at its first valid suffix. The public top-level
  legality method still returns every viable immediate action.

No threshold, family rule, benchmark certificate, sampling parameter, model,
data row, or outcome label changed.
