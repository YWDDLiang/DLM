# Exact-support performance review record

Three independent propose/red-team reviews converged on the same decision:

- `NO-GO` for freezing or submitting four-arm 512 now;
- conditional `GO` only for an exact implementation plus one bounded,
  predeclared same-node performance probe;
- reject blanket seek bypass, raw fixed-top-50 legality, lazy/deferred support
  telemetry, or moving mandatory work outside the attempt timer.

The accepted implementation subset is the intersection of all reviews:

1. combined grammar/terminal/full trie traversal with character-order rejection
   semantics preserved;
2. exact bitset acceleration for Boolean prefix charge reachability;
3. lazy materialization and chemistry-free lexical caching;
4. full online masks and telemetry; no distributional shortcut.

Logical reachable-state counts remain semantic popcounts/work counts. A bitset
word is not counted as one state, and tokenizer/setup costs are reported
separately rather than hidden.

After this preflight, the only permitted GPU action is a new immutable,
balanced/interleaved same-node P0/terminal/full probe. Its frozen hard gates are:

- full-prefix median latency no more than `1.5x` P0;
- full-prefix p95 latency no more than `2x` P0;
- every full-prefix attempt at most 100,000 logical DP states;
- exact masks, token IDs, step diagnostics, certificates, and RNG-consumption
  parity against the eager reference backend;
- no OOM, timeout, fallback, retry, replacement, repair, filter, or rerank.

Failure is the CR-Plan engineering terminal. Thresholds, denominators, seeds,
oxidation states, endpoint policy, or the implementation family may not be
changed after observing the probe.

