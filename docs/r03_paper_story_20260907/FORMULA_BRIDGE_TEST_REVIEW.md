# Native H1A2 / C3FD formula bridge: acceptance matrix

This is a bounded CPU test/review artifact for the approved integration. It
does not repeat the historical C3FD composition-validity experiment and does
not validate neural-model quality or final SUN.

## Frozen inference boundary

The preserved H1A2 prompt is `H1_PLANNER_PROMPT_STYLE_RICH_PLAN`, whose value is
`h1_rich_plan_v1`, with `sample_idx=None` and `add_special_tokens=False`. It has
no formula prefill. The native reference is
`workstreams/plangraph_dlm_iclr_20260731/execution/v3_factorial_runtime_staging_v5/source/crystal_dlm/h1a2_factorial_contract.py`
in the original repository. `build_planner_input_contract` records exact prompt
text and input IDs. The old `phase0_assignment` runner requires a different
formula-prefill prompt and cannot be enabled unchanged for this experiment.

## Independent reference

`tests/r03_formula_bridge_reference.py` exhaustively enumerates completed
compositions over a small injected synthetic vocabulary. It tests exact atom
and charge conservation, every available single-valence-per-element witness,
strict Pauling ordering, and the unary/all-metal zero-charge branches. It then
renders every display-element permutation and both explicit/implicit count-1
spellings. Prefix membership uses plain string prefixes of this finite terminal
set. No production parser, canonical-state traversal, cached witness or DP
result is reused by this reference.

These synthetic properties define a known test language; they are not a
replacement for the production scientific vocabulary or independent endpoint.

## Required behavior

| Case | Required behavior | Failure this detects |
|---|---|---|
| `F` / `Fe`, `F2F` / `F2Fe` | Preserve every compatible one/two-letter symbol branch | Greedy commitment, false duplicate-element rejection |
| `Li1`, `Li10`, `Li11`, `Li20` | Retain open count branches until a delimiter or EOS | Premature count commitment |
| `Fe2O3` and `O3Fe2` | Both use the same chemical support | Accidental canonical display-order constraint |
| Prefix `Fe`, completion `FeO` | Lower-Z element O remains available | Applying canonical max-Z suffix search to unordered display text |
| `FeO`, `Fe2O3` | Keep all compatible latent oxidation assignments | Choosing one oxidation state too early |
| Mixed-valence-only formula | Do not invent a single-state benchmark witness | Accidentally promoting extended-only support |
| Prefix compatible with several N/family/arity targets | Use their union; no random external target is selected | Changed composition policy via target sampling |
| N=2, arity=2, oxide; prefix `Li` | Reject when no joint atom/charge/family completion exists | Intersecting separately reachable conditions |
| `2O3\nanion: ...` token | Check its entire formula portion, allow arbitrary suffix after newline | Checking only first character or constraining rich fields |
| `for` + `mula: Xx\n` token | Candidate-level detection rejects crossed invalid formula value | Waiting until label already exists before enabling checks |
| Leading newline before `formula:` | Do not confuse it with formula completion | Global `if newline in decoded` bypass |
| Whole valid `formula: Fe2O3\nanion:` token | Accept without prefill or text rewriting | Boundary-fragment loss |
| Formula newline before count/chemistry complete | Reject newline | EOS-only syntax validation |
| Complete formula followed by another rich line | Return unmodified logits; same-prefix equality | Persistent mask or accidental rich-state reset |
| Empty token support | Explicit failed request / documented EOS sentinel, no replacement | Silent backoff, invalid unmasked continuation or retry |
| Mixed batch: formula row + rich row | Mask only the first row and preserve legal values exactly | Batch-wide state or score mutation |

All **39 tests passed in 2.012 seconds** on 2026-09-07: 11 independent-reference
sanity tests, 13 production semantic equivalence tests, and 15 native token / CPU
tensor tests. There were no skipped tests. The command was:

```powershell
& 'C:/Users/admin/miniconda3/envs/exp1/python.exe' -m unittest discover -s tests -p test_r03_formula_bridge_contract.py -v
```

The checked bridge file had SHA256
`d92907b6a5343ff1b7141b63b9521aeacc2ccb3c39cb1857f1fb9823fa1daabe`.
The local `exp1` environment is Python 3.10.19 with torch. No model, GPU,
network, physics evaluation or new training data was used.

Two initial test expectations were corrected after reading the native rich
parser: it permits padding around field values, and a `None` token mask can
also denote an unchanged seek phase with no invalid crossing candidates.
Internal formula whitespace still cannot discard a chemical suffix, and a
dedicated test checks this boundary. Neither correction changed the reference
chemical language or relaxed the substantive viability checks.

The small exhaustive domain has 44 accepted compositions, 240 terminal text
spellings and 391 viable text prefixes. The production tests also check each
prefix with single-character mutations, giving both recall and precision
checks instead of testing only hand-picked successful paths. Optional synthetic
fixed-stratum tests use `(N, arity, family)` to test joint viability; the
production experiment retains the latent target union.

The unordered charge-bitset review found the following consistent with the
declared witness language: every completed element is excluded from the future
element set, charge support has an adequate exact integer range, atom/arity
budgets close jointly, and family-required hits propagate in the same suffix
state. The pinned 88-element / 393-stratum domain was subsequently timed below;
the actual P0 tokenizer and model still require an integration measurement.

The wrapper now performs an exact coarse lexical filter before expensive
semantic checks, then prunes first-character groups using prefix viability.
It preserves one leading lower-case character for partial symbols, leading
digits for count continuation, native field padding, and arbitrary text after
the first formula newline. Seek-phase label crossings still use the complete
token vocabulary. New tests compare its complete allowed set against the
independent reference, including hostile word fragments and `Na1` to `Na10`.
The exposed `stats()` reports vocabulary size, retained lexical fragments,
first-character groups, cache hits and semantic-candidate work counts.

An exact charge-envelope prune now rejects a requested suffix charge outside
`remaining_atoms * [minimum_available_oxidation, maximum_available_oxidation]`.
The bounds use compiled oxidation-value bitmasks. They are a necessary
condition only; all surviving paths still undergo the original exact
atom/arity/family/charge checks. Zero-slot complete states bypass the envelope
and retain their existing terminal rule. A tight Na/Cl finite-language test
checks both positive and negative remaining-charge directions.

## Bounded timing on the actual domain

The domain SHA256 was
`a2b4a49eff72790cabaa75977414dba9eba45812ad02944d46e023a9b1d804a4`.
The benchmark uses 410 distinct character prefixes from the first 64 distinct
historical H1A2 formulas. Wrapper timing uses 28 prefixes from its first four
formulas and the pinned B0 ByteLevel vocabulary as a workload proxy: 128,830
IDs, 126,080 nonempty fragments, and 5,941 fragments retained by the lexical
filter. This is **not the P0 tokenizer or an actual P0 BPE trajectory**. No
neural model, GPU or physical evaluator was called. Each run had a 240-second
watchdog and completed in under 20 seconds.

| Measurement | Before envelope | Final compiled envelope |
|---|---:|---:|
| 410 cold semantic queries, total | 0.275 s | 0.121 s |
| 28 cold wrapper queries, total | 17.142 s | 13.045 s |
| Largest cold wrapper query | 6.007 s | 1.859 s |
| Cold `Mn12` wrapper query | 6.007 s | 1.737 s |
| 28 repeated wrapper queries, total | 0.213 ms | 0.202 ms |
| Suffix-cache misses | 1,608,740 | 868,187 |

The suffix cache reached its declared 250,000-entry cap in both runs. The new
envelope cache contained 108,784 entries. These are cache-entry measurements,
not a process-memory profile. All recorded semantic decisions and wrapper
allowed-token counts matched before/after; complete allowed-set equivalence is
also covered by the independent small-domain tests. The roughly 24% cold-time
reduction does not establish server/P0 throughput.

Before, intermediate, final timings and a machine-readable comparison are
retained under `execution/FORMULA_BRIDGE_CPU_*.json`. The exact repeatable
driver is `tests/benchmark_r03_formula_bridge_cpu.py`.

An additional pointer review found that export must honor explicit
`attempt_status` failures even when a row carries a parseable `plan_state`.
The pointer agent added that check and a regression fixture. Its deterministic
canonical formula replay is explicitly control-only; original sampled Plan
fields stay unchanged, and both CLIs require the native rich prompt with no
sample ID.

The integrated-body ABI was also checked using a full Fe2O3 rich Plan. The
current and frozen parsers produced the same composition state and the same
native body prompt (SHA256
`df2e4ad7cbca175be63d0f12fbd20b6594f75af4a4c51b62243f641e16b0a9c3`).
The runner retains upstream failures and verifies any supplied body prompt.
Two-seed ledgers must keep unique request IDs, or run separately per seed.

The physical-transfer and repair-execution source review found no blocking
scientific or parameter-isolation defect. The trainer freezes all parameters
before enabling only the existing B0 LoRA matrices; its optimizer receives
only those matrices, with frozen parameter versions and full trained
embedding/head equality checked afterward. The saved q values are carried
into a declared offline weighted conditional-likelihood objective, not reused
as rewards for newly sampled actions. Energy labels must be train-only and
verified; there is no new-label path or fabricated rich-field input. Finite
energy magnitude itself does not delete or reweight a uniform reference group.

Repair execution uses the same prompt, support and temperature as training. It
keeps construction artifacts, stages each complete XYZ transaction, preserves
every nonactive original token, and rolls back on unavailable support. The P
reload checks the frozen tokenizer and trained tables. Conservative global
geometry support can reject a one-site change when another unchanged pair is
invalid; this limits what local repair can accomplish and is not a guarantee of
repairing every invalid initial structure. Real model memory, step throughput,
and G/P generation behavior remain integration measurements.
