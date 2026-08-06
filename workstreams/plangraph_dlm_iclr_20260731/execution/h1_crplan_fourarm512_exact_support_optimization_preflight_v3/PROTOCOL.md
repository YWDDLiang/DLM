# H1 CR-Plan exact-support optimization preflight V3

Status: `cpu_exact_audit_only_not_a_512_release`

This immutable, CPU-only preflight audits the one bounded performance
implementation allowed after the V2 tokenizer-policy audit. It loads the exact
frozen Planner tokenizer and frozen SMACT table, but no Planner weights and no
GPU.

The scientific contract is unchanged:

- formula grammar, tokenizer fragments, prompt, seed roles, `1 <= N <= 20`,
  oxidation-state table, missing-state policy, terminal semantics, and sampling
  support are frozen;
- unary and true frozen-Direct all-metal formulas remain explicit shortcuts;
- every other formula containing a table-missing element fails closed;
- full-prefix differs from terminal-only only by preterminal neutral
  reachability;
- no retry, replacement, repair, filtering, reranking, Body/refiner/Direct/S.U.N.
  run, training, selection, promotion, or downstream action is permitted.

The implementation may only change how the exact answer is computed:

- one combined trie traversal returns the nested grammar/terminal/full supports;
- speculative terminal support uses an exact decision-only certificate, while
  the sampled formula still constructs the original deterministic full witness;
- prefix Boolean reachability uses exact Python-integer charge bitsets;
- grammar materialization is lazy and its pure syntactic reachability check is
  constant-time;
- processor probability telemetry remains online and inside generation latency.

Release checks before any GPU performance probe:

1. Exact 128,256-token optimized-versus-scalar equality of legal token IDs,
   terminal token IDs, rejection taxonomy/counts, and sampled-terminal
   certificate hashes over all formula-label seek suffixes and registered
   adversarial cursors.
2. The same equality for every unique formula-value cursor recorded by the
   immutable paired-32 real-model candidate trace.
3. Exact mixed-charge bitset/set, prefix bitset/set, and decision/full-certificate
   logical parity over deterministic exhaustive/randomized compositions.
4. Full online mask and mask→top-k→top-p probability equality on random,
   illegal-global-top-50 backfill, kth-tie, top-p-boundary, fewer-than-50-legal,
   and empty-support fixtures.
5. Frozen Direct alignment for all eight missing-state elements, unary/all-metal
   shortcut precedence, and terminal/full endpoint consistency.

Any mismatch is an engineering terminal and forbids the performance probe and
four-arm 512. A clean CPU report still does not authorize 512; it only permits
the separately frozen, same-node P0/terminal/full performance probe.

