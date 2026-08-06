# H1 CR-Plan missing-state policy and support optimization review V1

Status: `exact_support_optimization_cpu_audit_staged_512_closed`

Date: 2026-08-04

Scope: four-arm Plan-only 512 preparation after the clean paired-32 V5
terminal. This document does not authorize Body generation, refinement,
Direct/S.U.N. evaluation, training, promotion, or automatic downstream work.

## 1. Triggering evidence

Paired-32 job `30358` completed `0:0` in `02:23:48`; terminal SHA256 is
`a1a34a663d691480c6aefb9beb4e9174c9483a95e81669cb6b4888f96d598a80`.
All paired-32 engineering gates passed. Control/candidate composition validity
was `17/32 -> 18/32`, with one candidate-only discordance.

That result cannot be attributed to prefix reachability. The frozen SMACT
table has no oxidation states for `He`, `Ne`, `Ar`, `Pm`, `At`, `Rn`, `Fr`,
or `Ra`. Paired-32 allowed a formula containing a missing-state element as a
non-applicable terminal stratum. An unfinished formula could therefore always
append one such element while atom budget remained, making the conservative
prefix predicate structurally degenerate toward terminal-only.

The current implementation is also outside the registered 512 engineering
bounds: candidate median/p95 latency was `112.813/215.895 s`, versus
`2.835/3.022 s` for control, and maximum per-attempt DP states was
`1,714,193`.

## 2. Frozen policy decision for the 512 mechanism panel

The four arms remain:

1. original P0;
2. grammar-only;
3. terminal-only;
4. full-prefix reachability.

Terminal-only and full-prefix share the exact same endpoint policy:

- unary compositions retain the frozen Direct single-element shortcut;
- all-metal compositions retain the frozen Direct alloy shortcut;
- otherwise every element must have at least one state in the frozen SMACT
  oxidation-state table;
- a table-missing non-shortcut composition is fail-closed with stratum
  `charge_applicable_oxidation_state_missing`;
- uniform neutral witnesses are primary;
- mixed-valence neutral witnesses remain legal but non-primary;
- Pauling remains measured downstream, not a hard decoding constraint.

This matches the ordering of the frozen Direct classifier: unary and all-metal
shortcuts are evaluated before `oxidation_state_missing`, while every other
missing-state composition is invalid. It does not add states, infer a neutral
charge for a missing element, or modify the SMACT table.

For full-prefix reachability, missing-state elements are not numeric neutral
suffixes. A prefix is retained only when it can terminate under the shared
endpoint policy or has a frozen-table charge-reachable suffix within the
remaining atom budget. Terminal-only performs no preterminal reachability
test. Thus the sole full-versus-terminal intervention remains preterminal
legal support.

Shortcut-valid attempts are reported separately and excluded from the primary
gain. The policy was chosen from evaluator semantics before opening the
independent 512 ledger; it may not be changed after results.

## 3. Identity and causal safeguards

- The oxidation-table SHA remains policy-independent and unchanged.
- A separate constraint-contract SHA binds table SHA plus missing-state
  policy.
- Tokenizer, model, checkpoint, prompt, temperature, top-p/top-k, atom budget,
  stateless ordinal seed role, and all denominators remain identical.
- Every tokenizer token is expanded character by character through the same
  formula FSM.
- Terminal/full certificates are independently recomputed and must match the
  parser-visible formula exactly.
- There is one sample from the renormalized legal support and no retry,
  replacement, repair, filter, rerank, or fallback.

## 4. Semantics-preserving support optimization

The original support implementation evaluated every candidate tokenizer
fragment independently. Shared lexical prefixes repeatedly executed the same
immutable FSM and charge DP, producing the paired-32 latency/state explosion.

The first development implementation built a frozen character trie over decoded
token fragments. Its V2 exact-tokenizer audit passed, but inspection showed that
the runtime still made three complete support traversals per constrained step
(active, grammar, and terminal/grammar reference) and eagerly constructed
terminal witnesses for speculative newline tokens. Trie sharing alone therefore
could not meet the registered same-node latency bound.

Three independent propose/red-team reviews all rejected direct 512 release,
blanket seek bypass, raw fixed-top-50 legality, and deferred/off-clock telemetry.
They conditionally approved one exact implementation and bounded probe:

- one combined traversal produces nested full/terminal/grammar supports;
- speculative terminal branches compute the exact decision/stratum only, while
  the sampled formula still constructs the original deterministic witness;
- Boolean prefix charge reachability uses exact integer bitsets;
- partial formula materializations are lazy and pure grammar reachability is
  constant-time;
- the complete mask and all probability/support telemetry remain online.

Logical DP state accounting remains based on semantic reachable-state
popcounts/work. A bitset word is never redefined as one state, and setup or
telemetry work cannot be moved outside the registered timing boundary.

The original scalar implementation remains available only as
`support_scalar_reference` for release parity tests. The trie is not allowed
to change:

- legal token IDs;
- terminal token IDs;
- rejection taxonomy/counts;
- terminal certificates;
- probability renormalization;
- empty-support behavior.

Local focused tests currently pass `39/39`, including:

- table SHA stability and policy-contract separation;
- missing-state fail-closed behavior;
- unary/all-metal shortcut preservation;
- terminal/full endpoint consistency;
- no missing-state suffix escape;
- DP/brute-force checks;
- tokenizer-fragment trie versus scalar support equality at multiple cursor
  phases and modes;
- combined full/terminal/grammar support nesting and scalar equality;
- bitset mixed-charge equality with the retained set oracle;
- decision-only terminal strata equality with full certificates;
- parser/FSM identity and empty-support fail-closed behavior.

## 5. Exact frozen-tokenizer and Direct-policy release audit

The first immutable exact-tokenizer audit (`preflight_v1`) correctly established
full support parity but failed one audit fixture: it had incorrectly registered
`Fe-Pm` as an all-metal formula. Frozen Direct excludes `Pm` from its metal set
and therefore classifies `Fe-Pm` as `oxidation_state_missing`; CR-Plan already
made the same fail-closed decision. V1 was preserved unchanged and no scientific
policy, tokenizer, model, seed, atom budget, or sampling rule was altered.

The audit-only V2 repair replaced that fixture with actual frozen-Direct
all-metal cases (`Fe-Cu`, `Na-Fr`, and `Ba-Ra`) and added a separate explicit
`Fe-Pm` precedence check. Local tests passed `36/36`; isolated and A800 source
tests passed `32/32`.

The A800 CPU-only, offline V2 audit passed:

- source-manifest SHA256
  `72c66c4c99a92f3b717ff608c1cd71d1a18538a25c140844a9a73c9f95d0fe5c`;
- transfer-archive SHA256
  `596d7518adeb3c4a434a6b66671eee9757b2eada3ff5440a152a7acd84914642`;
- report SHA256
  `59173a6edb3902660702d6d30ddffbe2c4b02ad37c628160ed19db7472c05611`;
- exact decoded vocabulary `128,256`, fragment SHA256
  `ec480511a72d7da7633236a25c3764cd3a72fbd7f534197be9f3492884872153`;
- legal token IDs, terminal IDs, rejection taxonomy/counts, cursor
  signatures, and certificate hashes all matched the scalar oracle;
- all eight missing-state elements, unary shortcuts, true all-metal
  shortcuts, and missing-versus-all-metal precedence matched frozen Direct;
- maximum audit-cursor trie DP states were `76,267`, below the registered
  `100,000` bound;
- aggregate trie/scalar support time was
  `10.3488/121.9064 s` (about `11.78x` faster);
- network, GPU, model loading, generation, retry/repair/filter/rerank, and
  downstream activity were all false.

Remote evidence root:
`/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260804_h1_crplan_fourarm512_exact_tokenizer_preflight_repair_v2`.
Its read-only `_SUCCESS` marker was created only after the report gates passed.

## 6. Exact-support V3/V4 audit terminal

No 512 source or job was frozen. The first immutable optimized-support audit
source, V3, preserved the accepted exact combined-trie/bitset implementation but
its remote launcher selected base Python 3.9. Import therefore failed on
`dataclass(slots=True)` before the audit began. That execution is sealed as a
runtime-only engineering failure:

- source-manifest SHA256
  `9409b9ee8a45ff15448e495f249544f9547e54c36aeffc8c946e651d799e100d`;
- transfer-archive SHA256
  `a425fa47892b08f6a5bb84c775c66d1bc85567a63557c17388d9cc2873425f75`;
- failure-report SHA256
  `2629c49278fd83c6dc39f2df74446e5939554acc3542ab69fbca8ea90f1cf148`.

A bounded runtime probe established that the existing
`diff_meets_diff` environment provides Python 3.10.18,
`dataclass(slots=True)`, and Transformers 4.54.0. V4 changed only the selected
interpreter and added a runtime-repair record/manifest; the scientific code and
audit script remained byte-identical to V3. Local tests passed `39/39`, the
isolated archive passed `35/35`, and the same A800 test suite passed `35/35`.

The A800 login-node CPU-only, offline V4 audit then completed with process
status `pass` and no internal parity failures. It established:

- source-manifest SHA256
  `1f6dcd99906bd0b7607529affdf1a18937bd9c1b1be37a9811718bf6844baf02`;
- transfer-archive SHA256
  `d4838032bb8721e09f6ed6754d53f3c9371653ab49264355e35ffddf2c7e60d1`;
- audit-report SHA256
  `3a10b0da700fcb4df2c27e124238f534b78c9adeebdfa49b86a7e5c19dac7b1c`;
- terminal-report SHA256
  `55df7801e24f3bfd013e2e41f0cb96babe2a20ff3c5fb8a5b8e0b23073a5e627`;
- exact decoded vocabulary `128,256`, with the unchanged fragment SHA256
  `ec480511a72d7da7633236a25c3764cd3a72fbd7f534197be9f3492884872153`;
- full optimized/scalar equality for fixed fixtures and all `169` unique
  formula-value cursors from the paired-32 candidate trace (`507` mode rows);
- exact mixed-charge bitset/set, prefix bitset/set, and
  decision/full-certificate parity;
- exact full-mask and mask→top-k→top-p equality, including illegal-global-top-50
  backfill, ties, top-p boundaries, small support, and empty support;
- exact missing-state, unary, true-all-metal, precedence, and terminal/full
  endpoint alignment with frozen Direct;
- no model loading, GPU, network, generation, retry/replacement/repair,
  filtering, reranking, or downstream action.

However, the preregistered logical-state hard gate failed. Fixed fixtures peaked
at `72,390` semantic DP states, but the real paired-32 cursor set peaked at
`415,689`, versus the immutable limit of `100,000`. The audit script's internal
status is therefore not a release decision: the outer terminal gate correctly
records `engineering_terminal_state_budget_exceeded`, creates `_FAILED`, and
does not create `_SUCCESS`.

This is the CR-Plan engineering terminal for the frozen ICLR route. No same-node
performance probe, four-arm 512, paired-64, paired-256, or independent panel was
submitted. The result may not be repaired by changing the state definition,
moving work off-clock, selecting easier cursors/seeds, shrinking denominators,
altering thresholds, or introducing another implementation family after seeing
the audit.

## 7. E1 exploratory physical-performance probe

After the V4 terminal, the user separately authorized one bounded exploratory
question: whether the exact implementation is physically fast enough on the
real Planner despite exceeding the V4 logical-state budget. This E1 sidecar did
not repair, overwrite, reinterpret, or reuse the V4 `100,000`-state gate. It
also did not authorize four-arm 512 or any downstream scientific panel.

The immutable E1 identity is:

- run root
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_e1_physical_performance_probe_v1`;
- job `30435`, `gpu` partition, node99, `COMPLETED 0:0` in `01:04:07`;
- source-manifest SHA256
  `aab2c113e97adb615b9b4d0dc64e3948580e11db0d7cce505474f03a90647e24`;
- transfer-archive SHA256
  `c0766a6bef7b4166c5ea7ad77d9d7bc40c9db30af280ff9f6aef0e748fd7cebd`;
- submission-record SHA256
  `15ac458579e4be2fc00c17077c2fb44c92f8a2f191d3da375577b3b7be77573c`;
- terminal-report SHA256
  `b7c8e2bfd1bdec7c62b29d1cc2996d69086d4cd5aef9de847d16959bc449996b`.

The balanced probe used `18` engineering ordinals in each of `off`,
`terminal_only`, and `full_prefix`, plus scalar full-prefix references for
ordinals `2` and `11`. All three primary files contain exactly `18` rows, for
`54` primary attempts; runner and evaluator exit codes are both zero,
`_SUCCESS` exists, and `_FAILED` does not.

Primary synchronized latency was:

| Mode | Count | Median (s) | p95 (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|---:|
| `off` | 18 | 2.593208 | 88.804610 | 2.410626 | 88.804610 |
| `terminal_only` | 18 | 3.806238 | 10.930144 | 3.040696 | 10.930144 |
| `full_prefix` | 18 | 3.807525 | 10.910715 | 3.057534 | 10.910715 |

The preregistered full/off ratios are `1.468268` for the median and `0.122862`
for p95, so both physical gates passed. The p95 ratio must be interpreted
literally but cautiously: it is dominated by one `off` cold-path observation
of `88.804610 s`. The median is the more informative result and passes only
narrowly below the `1.5x` cap. In contrast, terminal-only and full-prefix
medians are essentially identical, showing that exact preterminal
reachability adds negligible physical cost once the shared constrained-support
path is active.

Mechanism and identity checks also passed:

- `14/14` charge-applicable full-prefix attempts had a real preterminal
  full-versus-terminal support difference (`100%`, versus the registered
  `>=5%` gate);
- actual-trace optimized/scalar support parity was exact over `105` unique
  formula cursors, `36` diagnostic step traces, and `264` sampled legal checks,
  with zero errors;
- scalar reruns for ordinals `2` and `11` matched optimized token IDs, text,
  parsing, certificates, prompt/input hashes, seed, and diagnostic hashes
  exactly; optimized/scalar latency was `3.921056/34.906765 s` and
  `3.296967/28.707581 s`, respectively;
- prompt/input/seed identity was `100%`, failures were empty, and no forbidden
  operation occurred.

E1 reports maximum cumulative semantic-state work of `1,567,682` for both
terminal-only and full-prefix (`0` for off). This is intentionally disclosed
without relabeling states or applying the V4 release gate: E1 answers a
different physical-cost question and leaves the V4 state-budget failure
unchanged.

The one-time model load was `2168.482 s`; mode-support setup was
`2.504768 s` for terminal-only and `2.507866 s` for full-prefix. E1 performed
no Body, refiner, Direct, S.U.N., network, training, reselection, promotion, or
automatic downstream action. The original V4 terminal remains byte-identical
at SHA256
`55df7801e24f3bfd013e2e41f0cb96babe2a20ff3c5fb8a5b8e0b23073a5e627`.

The defensible decision is therefore a bounded **GO for a new, separately
preregistered route amendment only**. The physical implementation is viable
and the reachability mechanism is active, but E1 is not a scientific result,
does not reopen the frozen route by itself, and does not authorize automatic
512 submission. Any continuation must freeze a new causal contract, ledger,
raw-latency reporting (including the cold outliers), and scientific gates
before execution.

## 8. Four-arm Plan-only 512 route-amendment terminal

The separately authorized route amendment was frozen before execution as
`h1_crplan_fourarm512_route_amendment_v1`. Its immutable evidence is:

- run root
  `/public/home/jiaosz/ywliang/ai4s/diffsion_language_model_meets_diffusion/runs/20260805_h1_crplan_fourarm512_route_amendment_v1`;
- source-manifest SHA256
  `f0241f913f839d6570697265f86a6dee698ca8b234f3fce5e48eadcdeec1ec0e`;
- transfer-archive SHA256
  `9aa3d32cc4e7e38f6094ed4257cfc1ced23006857bad07f18bfae2c8a20e9a3c`;
- independent science-ledger SHA256
  `0bd468b963da4462b3e5bb9c1482c11183fa09ba067b9b6f26ffd885618ae9d8`;
- submission-record SHA256
  `22a4c3e332fe4f0fec0f38603a6a7e36aad37e6817ba451981689d7de0032e53`;
- preflight-report SHA256
  `e55ac59010c74aad7c5d2a0f181e8f08849d3d1099b8d9ede4d82a4649aaa58a`;
- terminal-report SHA256
  `8cef9c2a2b5f6c814f2837af5a719e821ddd0102ca51c8c35e6543c8a9716d47`.

All four GPU array elements completed `0:0` with exactly the frozen raw
ordinals `0..511`: `off` job 30442 in `01:20:49`, `grammar_only` job 30443 in
`01:31:00`, `terminal_only` job 30448 in `01:36:41`, and `full_prefix` job
30441 in `01:39:08`. The after-any assembly job 30444 wrote the complete
terminal report and then exited `2:0` by design because the terminal decision
was failure. Thus the nonzero assembly status is the fail-closed terminalization
of complete evidence, not a missing denominator.

Raw all-attempt Plan metrics were:

| Arm | Parse | Completion | Composition valid | Nonshortcut / primary | Shortcut | Unique formula | Element coverage | Median / p95 latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 509 | 512 | 276 | 126 / 126 | 150 | 501/512 | 80/94 | 2.526 / 2.693 |
| `grammar_only` | 512 | 512 | 276 | 126 / 126 | 150 | 504/512 | 80/94 | 3.430 / 3.997 |
| `terminal_only` | 505 | 505 | 278 | 128 / 128 | 150 | 497/512 | 79/94 | 3.441 / 4.036 |
| `full_prefix` | 512 | 512 | 284 | 129 / 129 | 155 | 503/512 | 80/94 | 3.574 / 4.176 |

The full-versus-terminal raw composition comparison has six candidate-only
flips and zero baseline-only flips: `+6/512`, exact McNemar `p=0.03125`.
That apparent gain does not survive the preregistered causal stratification.
Five of the six extra valid outputs are shortcuts; excluding shortcuts leaves
only one candidate-only flip, `+1/512`, with exact McNemar `p=1.0`. The frozen
primary mechanism gate required at least `+11/512`, while shortcut-valid
outputs were required not to increase. Both gates therefore fail:

- nonshortcut composition gain: `+1`, required `>=+11`;
- shortcut-valid change: `+5`, required `<=0`;
- full-prefix charge-applicable terminal failures: `0`, pass;
- parse/completion noninferiority: `+7/+7`, pass;
- unique-formula and fixed-alphabet element-coverage losses: no loss, pass.

The evaluator recorded `512` preterminal-affected attempts, `2,885` affected
steps, and a true registered `>=5%` mechanism-activity gate. Its stored affected
rate divides this all-attempt numerator by `357` charge-applicable full
attempts and is consequently greater than one. It is retained verbatim as a
telemetry/accounting anomaly, not interpreted as a literal probability and
not used to rescue the failed outcome gates.

Full-prefix telemetry remains fully disclosed: median cumulative semantic
states `346,725.5`, maximum `2,592,229`, sum `225,914,529`, maximum cache size
`3,597,819`, `18,096` blocked newlines, prefix-only removed-mass sum
`11.053278`, reachability removed-mass sum `25.809460`, and total removed-mass
sum `37.838073`. Rejection totals were `453,118,633` grammar,
`851,500` prefix-reachability, and `18,096` terminal-charge. These state counts
are report-only under the amendment; they do not alter or retroactively pass
the frozen V4 `100,000`-state gate.

The engineering gate also fails independently: `terminal_only` recorded seven
generation errors, so its parse/completion counts are `505/512`, and the
registered no-generation-failure condition is false. There was no OOM,
timeout, retry, replacement, repair, filtering, reranking, fallback, Body,
refiner, Direct-structure, S.U.N., network, training, reselection, promotion,
or downstream run. The V4 and E1 terminal reports remain byte-identical.

The terminal decision is therefore **STOP CR-Plan and retain frozen H1**.
There is no defensible prefix-reachability composition claim: the significant
raw paired result is shortcut-driven, the nonshortcut effect is only
`+1/512`, the shortcut safeguard fails, and an independent engineering failure
is present. Paired-64, paired-256, and independent panels are not eligible and
were not submitted. The result must not be repaired by dropping the seven
failed attempts, counting shortcuts as primary gain, selecting seeds, changing
the missing-state policy or thresholds, or adding another factor.
