# Attribution input schema

The `h1a2-attribution` command reads one or more normalized JSONL cohorts.
Each row represents one requested scientific attempt; failed stages remain as
rows rather than disappearing from the denominator.

## Required fields

```json
{
  "attempt_id": "cohort/000001",
  "ordinal": 1,
  "formula": "Li2O",
  "stages": {
    "requested": true,
    "decoded": true,
    "plan_eligible": true,
    "body_attempted": true,
    "body_success": true,
    "refined": true,
    "reconstructed": true,
    "hull_known": true
  },
  "outcomes": {
    "novel": true,
    "unique": true,
    "strict_sun": false,
    "meta_sun": true
  }
}
```

## Optional Plan and provenance fields

```json
{
  "generated_anion": "oxide",
  "lattice": "orthorhombic",
  "spacegroup": "sg_016_074",
  "volume": "volpa_010_014",
  "plan_source": "learned",
  "plan_id": "plan-17",
  "planner_seed": 17,
  "body_arm": "full",
  "body_seed": 12345,
  "refiner_seed": 67890,
  "hull_status": "known"
}
```

Formula parsing derives element set, arity, raw atom count, composition family,
atom-count bin, and all-metal/unary shortcut status. Explicit normalized
fields may be supplied to override derivation.

## Stage semantics

Stage values are monotone. If `body_success=true`, all preceding stages must be
true. `hull_known=false` is not equivalent to unstable. `strict_sun` and
`meta_sun` are all-attempt outcomes and should be false for attempts that never
reached a known hull only when computing requested-attempt lower bounds.

## Command

```bash
h1a2-attribution \
  --cohort h1=normalized_h1.jsonl \
  --cohort r5c=normalized_r5c.jsonl \
  --pair h1:r5c \
  --reference h1 \
  --paired-key ordinal --paired-known-stage hull_known \
  --output-json results/attribution.json \
  --output-md results/attribution.md
```

The report contains funnels, stage distributions, adjacent-stage TVD/JSD,
stratum outcomes, symmetric decomposition, standardized outcomes, support
coverage, ESS, and weight diagnostics.
