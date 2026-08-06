# WQ formula-plan one-epoch v2

## Why this run exists

The 200-update v1 pilot did not fail because formula planning was chemically
ineffective. It generated an exact plan/body match for every completed body and
51/52 completed bodies were composition-valid. Its all-attempt score fell
because 12/64 bodies entered a deterministic decoder dead end:

`tokenizer has no legal planned-proposal continuation`

The old mask accepted a Wyckoff multiplicity whenever it fit at least one
current species count. It did not prove that the residual count vector remained
representable. More training could reduce how often the model selected such a
path, but could never eliminate the unsupported state.

## Primary change: exact remaining-count reachability

For a selected space group, let its distinct primitive Wyckoff
multiplicities be an unbounded coin set. A species count is reachable iff it is
an exact non-negative sum of those multiplicities.

The v2 decoder applies that condition before:

1. accepting a space group for the complete formula;
2. accepting a Wyckoff type for any remaining species;
3. assigning the selected Wyckoff type to a species.

Every accepted action therefore leaves every residual species count reachable.
`STOP` remains legal only after exact exhaustion. This is a support mask, not a
retry, repair, reranker, or learned guidance term.

The formula-plan cursor also no longer accepts an invalid formula merely because
the 20-atom or species-code support is exhausted. Count macros that would
immediately force that invalid terminal state are masked.

## Training comparison

- Starting adapter: unchanged selected epoch-3 WQ LoRA.
- Data: exact immutable v1 chemistry-plan dataset, 61,393 examples.
- Data SHA256:
  `b96ca8c08ddf1fe65daa9568f1e9507d1afa4d5886794a4e7115c8a08df4e48e`.
- Mixture: 24,557 formula plans, 24,557 formula-conditioned bodies, and
  12,279 direct-edit replay rows.
- Schedule: exactly 1.0 epoch (expected 959-960 optimizer updates).
- Effective batch: 64; BF16; SDPA; one A800; eight CPUs.
- Learning rate: `1e-5`, constant with 5% warmup.
- Initialization is from the original epoch-3 adapter, not the v1 200-update
  continuation, so the result measures one total epoch rather than 1.208 epochs.

## Development evaluation

The run reuses the unchanged 64-attempt frozen baseline and the same paired
development seeds. Formula generation and body generation are logged
separately. The primary checks are:

- formula-plan success at least 63/64;
- body success at least 63/64;
- composition-valid at least 63/64;
- exact plan/body agreement on every body success;
- at most one baseline-valid to formula-invalid pair;
- no retry, replacement, repair, best-of-N, or reranking.

This is a development gate, not the final held-out paper panel. CrysLLMGen,
CHGNet, MP API, and S.U.N remain downstream until the direct generation and
chemistry gate passes.

