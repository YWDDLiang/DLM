# H1 CR-Plan EVAL-0 / CR-0 / paired-32 runtime-repair V2 protocol

## Purpose

This execution tests one principal factor only: the legal token support while
the frozen H1 P0 Planner emits the value of its first `formula:` line.

The prompt, Planner weights, rich seven-line schema, sampler hyperparameters,
ordinal RNG, Body, refiner, Direct evaluator, and S.U.N. evaluator are not
changed. Body/refiner/Direct/S.U.N. are not run in this engineering screen.

## V1 timeout diagnosis and bounded runtime repair

The immutable V1 CR-0 job `30325` reached Slurm `TIMEOUT` after `00:40:12`.
Its 26 focused tests all passed in `0.397s`, but no CR-0 terminal report or
success marker was written and paired-32 was never submitted. Accounting
showed only `1.44s` aggregate CPU time while maximum RSS rose to `359292 KiB`
and filesystem reads reached `273.06M`. This is incompatible with a
compute-bound DP or audit loop and is consistent with cold loading the remote
Python stack from the shared filesystem.

Code inspection found one avoidable heavyweight import: the probability
identity sub-audit imported all of PyTorch only to compare softmax before and
after applying an all-vocabulary support mask. V2 replaces that sub-audit with
the same full-vocabulary comparison using a stable Python binary64 softmax.
It also emits timestamped shell and Python phase events and gives CR-0 a
two-hour scheduler envelope so shared-filesystem cold starts cannot become a
scientific gate.

No Planner, tokenizer, formula FSM, DP, oxidation-state policy, logits
processor, sampler setting, seed, input, denominator, paired-32 evaluator, or
scientific acceptance threshold is changed. V1 evidence remains immutable.

## Frozen successful anchor

The pre-treatment H1 source and parameters are archived separately as:

`source_bundles/20260804_h1_success_anchor_pre_crplan_v1.tar.gz`

Archive SHA256:

`16e18b2ea9a8a781a9f8f2e8919cbb5b179748035c1c884362bfe9fb9348fb80`

The remote model assets are not duplicated into that archive. Their absolute
paths, byte sizes, and SHA256 identities are recorded in its `BACKUP_README.md`.

## Frozen Planner

- Base: `Meta-Llama-3-8B`
- P0 adapter model SHA256:
  `65766c7485bd5ad8e180f3f5d99b83bef0488c251acd9278cb8bc2ad2518aa3a`
- Prompt: `h1_rich_plan_v1`, no sample ID
- `temperature=0.9`, `top_p=0.95`, `top_k=50`
- `max_new_tokens=96`, `max_atoms=20`
- base seed `17029`
- RNG: `stateless_ordinal_v1`, role `shared`
- raw all-attempt denominator: 32 per arm

## Applicability semantics frozen before endpoints

The installed SMACT default `Element.oxidation_states` table is normalized,
hashed, and written to the CR-0 report.

Terminal formulae are partitioned before observing paired endpoints:

1. `charge_applicable_uniform_neutral`
   - At least two elements, not all metal, no table-missing element.
   - A single integer oxidation state per element makes the exact counts
     neutral.
   - This is the only primary charge-witness stratum and is aligned to the
     frozen Direct composition evaluator's charge abstraction.
2. `charge_applicable_mixed_valence_only`
   - No uniform witness, but an integer allocation of oxidation states across
     atoms is neutral.
   - The formula remains reachable to avoid false exclusion (for example,
     mixed-valence classes), but it is not counted as a primary gain.
3. `charge_not_applicable_unary`,
   `charge_not_applicable_all_metal`, and
   `charge_not_applicable_table_missing`
   - Allowed and reported separately.
   - Never counted as primary charge gain.
4. `charge_applicable_no_neutral_witness`
   - Formula newline is illegal in terminal/full modes.

Pauling electronegativity is intentionally not a hard decoding constraint in
V1. A table-relative witness is not claimed to be a physically true oxidation
assignment.

## Modes

- `off`: exact frozen P0 generation path.
- `grammar_only`: flat element/count grammar and `1 <= N <= 20`.
- `terminal_only`: grammar plus fail-closed terminal charge gate.
- `full_prefix`: terminal-only plus memoized mixed-valence charge-set
  reachability at every sampled tokenizer prefix.

The first paired screen compares `off` and `full_prefix`. The other modes are
implemented and audited now, but their independent four-arm 512 experiment is
not automatically submitted.

`full_prefix` reports the support removed relative to `terminal_only`, not
relative only to grammar. Because a terminal formula containing an element
missing from the frozen oxidation table is explicitly allowed as a
non-applicable stratum, any prefix that can still append such an element also
remains reachable. This conservative rule prevents the prefix DP from silently
excluding a completion that the terminal contract permits.

If CR-0 finds any table-missing element, it marks the resulting structural
degeneracy explicitly. In that case paired-32 is interpreted only as
`off` versus a terminal charge gate with a conservative prefix mask; no
prefix-control gain is claimed, and the missing-state policy must be reviewed
before authorizing the independent four-arm experiment.

Formula-prefill prompts are forbidden for constrained arms: the generated
continuation must contain the exact contiguous `formula:` label so the masked
FSM and the final parser observe the same value. A parsed candidate is accepted
only when the generated text has exactly one formula field and the exact frozen
seven-field order, the final FSM certificate equals the independently parsed
composition, and the independently recomputed terminal certificate is
identical. Spaced labels and duplicate formula fields fail closed.

Repeated lexical elements such as `FeOFeO2` remain legal. Their raw occurrence
is retained in the audit, then counts are canonicalized (`Fe2O3`) once; this
policy avoids silently deleting a token path while making evaluator identity
reproducible.

## No hidden sample operations

There is one sampling action on the renormalized legal support. There is no
beam search, candidate set, retry, replacement, repair, filter, rerank,
fallback, or endpoint-based mask tuning. Empty support raises a dedicated
error and records that raw ordinal as a failed-closed attempt.

## EVAL-0 / CR-0 gate

CR-0 runs in the frozen remote environment before any GPU sampling and must
pass all of:

- source manifest identity;
- installed SMACT table identity;
- tokenizer whole-decode versus per-token-fragment concatenation on the
  frozen fixture panel;
- tokenizer incremental decode identity on every legal continuation for a
  frozen set of partial formula prefixes;
- formula cursor recovery when labels/newlines cross token fragments;
- DP versus brute-force charge-set parity;
- uniform/mixed witness recomputation;
- mixed-valence, unary, all-metal, and table-missing applicability fixtures;
- alignment to `classify_smact_validity`;
- zero false exclusions on evaluator-valid fixtures;
- probability identity when legal support is the full vocabulary;
- ordinal/resume RNG identity;
- explicit empty-support detection;
- DP state count below 100,000 on the audit.

A failing CR-0 exits nonzero. The paired GPU job is not pre-submitted: a
separate one-shot release script requires observed `COMPLETED 0:0`, `_SUCCESS`,
`terminal_report.status=pass`, unchanged source identity, and a fresh GPU
partition preflight.

## paired-32 engineering gate

Both arms must contain ordered raw ordinals `0..31`. The candidate must have:

- zero tokenizer/FSM dead end;
- zero silent fallback/retry/repair;
- zero charge-applicable terminal failure;
- parse loss no worse than `-1/32`;
- completion loss no worse than `-1/32`;
- preserved seven-line/canonical identity;
- exact paired sampling seed identity.
- exact candidate decoded-token-fragment SHA identity with the tokenizer audited
  by CR-0, including the audited vocabulary size, EOS/PAD token IDs, and
  padding side;
- exact paired Planner input-prompt SHA and token-ID SHA identity;
- exact scientific-output parity of the untouched control with historical P0
  ordinals 0..31 from the frozen 512-attempt source (whose complete raw file
  SHA256 is
  `7c630cecd7654d103f67ad2edf69da869c22408cec95454c26f2ddd6870d3e70`);
- per-attempt DP/cache telemetry and control/candidate generation median/p95
  latency.

Composition validity is reported using the frozen Direct evaluator. Primary
composition-valid yield additionally excludes mixed-valence-only and all
charge-not-applicable shortcuts.

Passing paired-32 authorizes only preparation of a new, independent four-arm
Plan-only 512 protocol. It does not automatically submit that experiment.
