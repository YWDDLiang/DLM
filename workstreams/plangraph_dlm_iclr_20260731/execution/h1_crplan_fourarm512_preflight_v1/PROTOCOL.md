# H1 CR-Plan four-arm-512 preflight V1

This preflight is CPU-only and non-networked. It loads the exact frozen
Planner tokenizer and SMACT table, but no model weights and no GPU.

It must pass before an immutable four-arm Plan-only 512 source is frozen:

- exact decoded-token-fragment trie/scalar support parity;
- missing-state fail-closed alignment with frozen Direct;
- unary/all-metal shortcut ordering alignment;
- terminal/full endpoint consistency;
- explicit table and constraint-contract identities.

The preflight performs no generation, retry, replacement, repair, filtering,
reranking, Body/refiner/Direct/S.U.N. pipeline execution, training, selection,
promotion, or downstream action.
