# Plan assets placeholder

The release requires:

```text
h1a2_parsed_1186.jsonl
r03_raw_256.jsonl
r03_parsed_256.jsonl
r03_seed_ledger_256.jsonl
h1a2_learned_rich.jsonl
r5c_gold_rich.jsonl
mp20_train_rich.jsonl
```

The parsed Plan file is the default quick-reproduction input. If a Planner
checkpoint is absent, the launcher falls back to this file even when Plan
resampling was requested.

## Story-panel assets

`h1a2_learned_rich.jsonl` contains learned H1-A2 Planner outputs.
`r5c_gold_rich.jsonl` contains held-out MP-20-derived gold Plans and is a
conditional executor reference, not a fully de novo Plan source.
`mp20_train_rich.jsonl` contains the deterministic training Plan labels used
only as the collision/nearest-neighbor reference in the Plan audit.

Each non-empty line must be a JSON object containing one of `plan_state`,
`r5_plan_state`, or `parsed_plan`. The nested Plan object must include:

```text
formula, elements, counts, N, anion_framework,
lattice_system, spacegroup_bucket, volume_per_atom_bin
```

The E1 builder matches learned and gold Plans using only `N`, element arity,
and `anion_framework`. It then creates the fixed 768-task ledger before any
body result exists.
