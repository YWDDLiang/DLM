# H1 CR-Plan four-arm-512 preflight repair V2

This preflight is CPU-only and non-networked. It loads the exact frozen
Planner tokenizer and SMACT table, but no model weights and no GPU.

It must pass before an immutable four-arm Plan-only 512 source is frozen:

- exact decoded-token-fragment trie/scalar support parity;
- missing-state fail-closed alignment with frozen Direct;
- unary/all-metal shortcut ordering alignment;
- terminal/full endpoint consistency;
- explicit table and constraint-contract identities.

V2 preserves V1 as immutable failure evidence. V1 incorrectly registered
`Fe-Pm` as an all-metal fixture even though the frozen Direct evaluator's metal
set excludes `Pm`. V2 removes that false fixture assumption and audits the
actual Direct precedence explicitly: `Fe-Pm` must fail closed as
`oxidation_state_missing`. No CR-Plan policy, model, tokenizer, sampling
distribution, oxidation-state table, or gate changed.

The preflight performs no generation, retry, replacement, repair, filtering,
reranking, Body/refiner/Direct/S.U.N. pipeline execution, training, selection,
promotion, or downstream action.
