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

All **36 tests passed in 2.021 seconds** on 2026-09-07: 11 independent-reference
sanity tests, 12 production semantic equivalence tests, and 13 native token / CPU
tensor tests. There were no skipped tests. The command was:

```powershell
& 'C:/Users/admin/miniconda3/envs/exp1/python.exe' -m unittest discover -s tests -p test_r03_formula_bridge_contract.py -v
```

The checked bridge file had SHA256
`b728fb192485a15db8f5c32d205e1595bd24764b2cd196bf9b3f1d2c59086cc1`.
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
state. Production-vocabulary timing and memory are a separate outstanding
integration check; tiny-domain correctness does not establish its throughput.

An additional pointer review found that export must honor explicit
`attempt_status` failures even when a row carries a parseable `plan_state`.
The pointer agent added that check and a regression fixture. Its deterministic
canonical formula replay is explicitly control-only; original sampled Plan
fields stay unchanged, and both CLIs require the native rich prompt with no
sample ID.
