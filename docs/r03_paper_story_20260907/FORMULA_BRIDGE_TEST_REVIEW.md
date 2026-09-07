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

The first 11 reference-sanity tests passed locally on 2026-09-07. This initial
check validates the reference itself. Production equivalence results will be
recorded after the new bridge interface is connected. The local Python runtime
has no torch, so tensor equality either requires an available CPU torch runtime
or remains an explicitly skipped local check until the integration test.
